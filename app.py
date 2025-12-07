import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timezone, timedelta
import numpy as np

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Hybrid Sniper Bot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
    }
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 5px;
    }
    .risk-tag {
        color: #00FF00;
        font-weight: bold;
        border: 1px solid #00FF00;
        padding: 5px;
        border-radius: 5px;
    }
    .ai-tag {
        color: #00CCFF;
        font-weight: bold;
        border: 1px solid #00CCFF;
        padding: 5px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Hybrid Sniper | Neural Net + XGBoost")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

asset_map = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Gold (GC=F)": "GC=F",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/JPY": "GBPJPY=X",
    "US30 (Dow Jones)": "^DJI",
    "US500 (S&P 500)": "^GSPC",
    "German Index (DAX)": "^GDAXI",
}

selected_asset = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
SYMBOL = asset_map[selected_asset]

EXEC_INTERVAL = st.sidebar.selectbox("Execution Timeframe", ["5m", "15m", "1h"], index=0)

if EXEC_INTERVAL in ["5m", "15m"]:
    PERIOD = "59d"
    st.sidebar.warning(f"⚠️ {EXEC_INTERVAL} data is limited to last 60 days.")
else:
    PERIOD = st.sidebar.select_slider("Data Lookback", options=["3mo", "6mo", "1y", "2y"], value="6mo")

CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
STARTING_CAPITAL = 10000

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='risk-tag'>
🛡️ <b>Risk Logic:</b><br>
• 50% Exit @ <b>1.5 RR</b><br>
• 50% Exit @ <b>3.0 RR</b><br>
</div>
<div class='ai-tag'>
🧠 <b>Dual-Brain Logic:</b><br>
• Model 1: <b>XGBoost</b> (Trees)<br>
• Model 2: <b>Neural Network</b> (MLP)<br>
• Consensus: <b>Voting Classifier</b>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA ENGINE
# ==========================================
@st.cache_data(ttl=60)
def fetch_and_merge_data(ticker, period, exec_interval):
    try:
        # 1. Exec Data
        df_exec = yf.download(ticker, period=period, interval=exec_interval, progress=False)
        if df_exec.empty: return pd.DataFrame()
        if isinstance(df_exec.columns, pd.MultiIndex): df_exec.columns = df_exec.columns.get_level_values(0)
        
        # 2. Confirmation Data
        df_1h = yf.download(ticker, period="1y", interval="1h", progress=False)
        if isinstance(df_1h.columns, pd.MultiIndex): df_1h.columns = df_1h.columns.get_level_values(0)
        
        df_4h = yf.download(ticker, period="1y", interval="1h", progress=False)
        if isinstance(df_4h.columns, pd.MultiIndex): df_4h.columns = df_4h.columns.get_level_values(0)
        df_4h = df_4h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

        # 3. Indicators
        def calc_ind(df, suffix=""):
            df[f'RSI{suffix}'] = ta.rsi(df['Close'], length=14)
            df[f'EMA50{suffix}'] = ta.ema(df['Close'], length=50)
            df[f'EMA200{suffix}'] = ta.ema(df['Close'], length=200)
            df[f'Trend{suffix}'] = np.where(df[f'EMA50{suffix}'] > df[f'EMA200{suffix}'], 1, -1)
            return df

        df_exec = calc_ind(df_exec, "")
        df_1h = calc_ind(df_1h, "_1h")
        df_4h = calc_ind(df_4h, "_4h")
        df_exec['ATR'] = ta.atr(df_exec['High'], df_exec['Low'], df_exec['Close'], length=14)

        # 4. Merge
        df_exec = df_exec.sort_index()
        df_1h = df_1h.sort_index()
        df_4h = df_4h.sort_index()

        df_merged = pd.merge_asof(df_exec, df_1h[['RSI_1h', 'Trend_1h']], left_index=True, right_index=True, direction='backward')
        df_merged = pd.merge_asof(df_merged, df_4h[['RSI_4h', 'Trend_4h']], left_index=True, right_index=True, direction='backward')

        # 5. Session
        df_merged['Hour'] = df_merged.index.hour
        df_merged['Session'] = "ASIAN"
        df_merged.loc[(df_merged['Hour'] >= 7) & (df_merged['Hour'] <= 16), 'Session'] = "LON"
        df_merged.loc[(df_merged['Hour'] >= 13) & (df_merged['Hour'] <= 21), 'Session'] = "NY"

        df_merged.dropna(inplace=True)
        return df_merged

    except Exception as e:
        return pd.DataFrame()

with st.spinner(f"Fetching Data ({EXEC_INTERVAL})..."):
    df = fetch_and_merge_data(SYMBOL, PERIOD, EXEC_INTERVAL)

if df.empty:
    st.error("❌ Data Fetch Failed. Yahoo API throttling or invalid ticker.")
    st.stop()

# ==========================================
# 4. HYBRID AI ENGINE (XGB + NN)
# ==========================================
def run_simulation(df, threshold):
    # Features
    features = ['RSI', 'RSI_1h', 'RSI_4h', 'Trend_1h', 'Trend_4h', 'ATR']
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Train/Test Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Scaling (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df['Target']
    X_test = scaler.transform(test_df[features])
    
    # --- DEFINE MODELS ---
    # 1. Neural Network (Pattern Recognition)
    clf1 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    
    # 2. XGBoost (Decision Tree Logic)
    clf2 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss')
    
    # 3. Voting Classifier (Soft = Average of Probabilities)
    model = VotingClassifier(estimators=[('nn', clf1), ('xgb', clf2)], voting='soft')
    model.fit(X_train, y_train)
    
    # Live Signal
    last_candle = df[features].iloc[[-1]]
    last_candle_scaled = scaler.transform(last_candle)
    live_probs = model.predict_proba(last_candle_scaled)[0]
    live_buy_conf = live_probs[1]
    live_sell_conf = live_probs[0]
    
    # --- BACKTEST ---
    test_probs = model.predict_proba(X_test)
    trades = []
    final_balance = STARTING_CAPITAL
    
    # Stateful Trade Loop
    i = 0
    while i < len(test_df) - 1:
        if test_df['Session'].iloc[i] == "ASIAN": 
            i += 1
            continue
            
        prob_buy = test_probs[i][1]
        prob_sell = test_probs[i][0]
        
        action = "WAIT"
        if prob_buy > threshold: action = "BUY"
        elif prob_sell > threshold: action = "SELL"
        
        if action == "WAIT":
            i += 1
            continue
            
        # Trade Setup
        entry_idx = i + 1
        if entry_idx >= len(test_df): break
        
        entry_price = test_df['Open'].iloc[entry_idx]
        atr = test_df['ATR'].iloc[i]
        
        sl_dist = atr * 1.0
        tp1_dist = atr * 1.5
        tp2_dist = atr * 3.0
        
        sl_price = entry_price - sl_dist if action == "BUY" else entry_price + sl_dist
        tp1_price = entry_price + tp1_dist if action == "BUY" else entry_price - tp1_dist
        tp2_price = entry_price + tp2_dist if action == "BUY" else entry_price - tp2_dist
        if action == "SELL": tp2_price = entry_price - tp2_dist

        # Execution State
        qty_remaining = 1.0
        tp1_hit = False
        trade_pnl_accum = 0.0
        exit_reason = "End of Data"
        
        j = entry_idx
        while j < len(test_df):
            curr_high = test_df['High'].iloc[j]
            curr_low = test_df['Low'].iloc[j]
            curr_close = test_df['Close'].iloc[j]
            
            # SL Check
            sl_triggered = False
            if action == "BUY" and curr_low <= sl_price: sl_triggered = True
            elif action == "SELL" and curr_high >= sl_price: sl_triggered = True
            
            if sl_triggered:
                trade_pnl_accum -= (qty_remaining * 1.0) 
                exit_reason = "SL Hit" if not tp1_hit else "TP1 then SL"
                break 
            
            # TP1 Check
            if not tp1_hit:
                tp1_triggered = False
                if action == "BUY" and curr_high >= tp1_price: tp1_triggered = True
                elif action == "SELL" and curr_low <= tp1_price: tp1_triggered = True
                
                if tp1_triggered:
                    trade_pnl_accum += (0.5 * 1.5)
                    qty_remaining = 0.5
                    tp1_hit = True
            
            # TP2 Check
            tp2_triggered = False
            if action == "BUY" and curr_high >= tp2_price: tp2_triggered = True
            elif action == "SELL" and curr_low <= tp2_price: tp2_triggered = True
            
            if tp2_triggered:
                trade_pnl_accum += (0.5 * 3.0)
                exit_reason = "Full TP (3R)"
                break 
                
            # Time Exit (4 hours)
            if (j - entry_idx) > 48: 
                final_pnl_dist = (curr_close - entry_price) if action == "BUY" else (entry_price - curr_close)
                final_r = final_pnl_dist / sl_dist
                trade_pnl_accum += (qty_remaining * final_r)
                exit_reason = "Time Exit"
                break
            j += 1
            
        risk_dollars = final_balance * 0.02
        pnl_dollars = risk_dollars * trade_pnl_accum
        final_balance += pnl_dollars
        
        trades.append({
            "Date": test_df.index[entry_idx].strftime('%m-%d %H:%M'),
            "Type": action,
            "Price": entry_price,
            "TP1": tp1_price,
            "TP2": tp2_price,
            "Result": exit_reason,
            "Net R": trade_pnl_accum,
            "PnL": pnl_dollars
        })
        i = j + 1
            
    win_rate = 0
    if len(trades) > 0:
        wins = sum(1 for t in trades if t['PnL'] > 0)
        win_rate = (wins / len(trades)) * 100
        
    return live_buy_conf, live_sell_conf, pd.DataFrame(trades), final_balance, win_rate

live_buy, live_sell, trade_hist, end_bal, win_rate = run_simulation(df, CONFIDENCE)

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================
c_hour = datetime.now(timezone.utc).hour
c_session = "ASIAN (Sleep)"
if 7 <= c_hour <= 16: c_session = "LONDON (Active)"
elif 13 <= c_hour <= 21: c_session = "NEW YORK (Active)"

signal = "WAIT"
if live_buy > CONFIDENCE: signal = "BUY"
elif live_sell > CONFIDENCE: signal = "SELL"

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
curr_price = df['Close'].iloc[-1]
trend_4h = "BULLISH" if df['Trend_4h'].iloc[-1] == 1 else "BEARISH"

m1.metric("Live Price", f"${curr_price:,.2f}")
m2.metric("4H Trend", trend_4h, delta="Filter", delta_color="normal")
m3.metric("Hybrid Signal", signal, f"{max(live_buy, live_sell)*100:.1f}% Conf")
m4.metric("Session", c_session)

# Chart
st.markdown("### 🧬 Hybrid Analysis (NN + XGB)")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index[-100:], open=df['Open'][-100:], high=df['High'][-100:], low=df['Low'][-100:], close=df['Close'][-100:], name="Price"))
st.plotly_chart(fig, use_container_width=True)

# Execution
col_exec, col_data = st.columns([2, 1])
with col_exec:
    st.info(f"**AI Logic:** Consensus between **Neural Network** (Patterns) and **XGBoost** (Logic).")
    st.progress(float(live_buy))
with col_data:
    if st.button("🔄 SCAN NOW"): st.rerun()

# Backtest
st.markdown("---")
st.subheader(f"🧪 Split-Target Backtest (Last 20% of {PERIOD})")
b1, b2, b3, b4 = st.columns(4)
b1.metric("End Balance", f"${end_bal:,.2f}", f"{(end_bal-STARTING_CAPITAL)/STARTING_CAPITAL*100:.1f}%")
b2.metric("Win Rate", f"{win_rate:.1f}%")
b3.metric("Trades", len(trade_hist))
b4.metric("Strategy", "1.5R / 3.0R Split")

if not trade_hist.empty:
    st.dataframe(
        trade_hist.iloc[::-1],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "TP1": st.column_config.NumberColumn(label="Target 1", format="%.2f"),
            "TP2": st.column_config.NumberColumn(label="Target 2", format="%.2f"),
            "Net R": st.column_config.NumberColumn(label="Total R", format="%.2f R"),
            "PnL": st.column_config.NumberColumn(format="$%.2f"),
        }
    )
