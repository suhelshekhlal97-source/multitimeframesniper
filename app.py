import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone, timedelta
import numpy as np

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Multi-TF Sniper Bot",
    page_icon="⚔️",
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
    div[data-testid="stExpander"] {
        background-color: #1E1E1E;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Multi-Timeframe Sniper | 5m/15m Execution")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

# Asset Map
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

# --- CRITICAL: EXECUTION TIMEFRAME ---
# User chooses where to ENTER trades (5m or 15m)
EXEC_INTERVAL = st.sidebar.selectbox("Execution Timeframe", ["5m", "15m", "1h"], index=0)

# Limit history based on interval (Yahoo API Limit)
# 5m/15m data is ONLY available for the last 60 days.
if EXEC_INTERVAL in ["5m", "15m"]:
    PERIOD = "59d" # Max safe limit for intraday
    st.sidebar.warning(f"⚠️ {EXEC_INTERVAL} data is limited to last 60 days by Exchange.")
else:
    PERIOD = st.sidebar.select_slider("Data Lookback", options=["3mo", "6mo", "1y", "2y"], value="6mo")

CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
RR_RATIO_STR = st.sidebar.select_slider("Risk:Reward Target", options=["1:1.5", "1:2", "1:3", "1:4"], value="1:2")
TARGET_RR = float(RR_RATIO_STR.split(":")[1])
STARTING_CAPITAL = 10000

st.sidebar.markdown("---")
st.sidebar.info(f"**Strategy:**\n- Execute: **{EXEC_INTERVAL}**\n- Confirm: **1H & 4H Trend**\n- Filter: **Session**")

# ==========================================
# 3. MULTI-TIMEFRAME DATA ENGINE
# ==========================================
@st.cache_data(ttl=60)
def fetch_and_merge_data(ticker, period, exec_interval):
    try:
        # 1. Fetch EXECUTION Data (e.g. 5m)
        df_exec = yf.download(ticker, period=period, interval=exec_interval, progress=False)
        if df_exec.empty: return pd.DataFrame()
        if isinstance(df_exec.columns, pd.MultiIndex): df_exec.columns = df_exec.columns.get_level_values(0)
        
        # 2. Fetch CONFIRMATION Data (1h and 4h)
        # We fetch "1y" for higher TFs to ensure we have enough EMA history
        df_1h = yf.download(ticker, period="1y", interval="1h", progress=False)
        if isinstance(df_1h.columns, pd.MultiIndex): df_1h.columns = df_1h.columns.get_level_values(0)
        
        df_4h = yf.download(ticker, period="1y", interval="1h", progress=False) # Get 1h first...
        if isinstance(df_4h.columns, pd.MultiIndex): df_4h.columns = df_4h.columns.get_level_values(0)
        # ...then resample to 4h manually to be robust
        df_4h = df_4h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

        # 3. Calculate Indicators on EACH Timeframe SEPARATELY
        def calc_ind(df, suffix=""):
            # Basic Momentum
            df[f'RSI{suffix}'] = ta.rsi(df['Close'], length=14)
            df[f'EMA50{suffix}'] = ta.ema(df['Close'], length=50)
            df[f'EMA200{suffix}'] = ta.ema(df['Close'], length=200)
            # Trend Bias (1 = Bullish, -1 = Bearish)
            df[f'Trend{suffix}'] = np.where(df[f'EMA50{suffix}'] > df[f'EMA200{suffix}'], 1, -1)
            return df

        df_exec = calc_ind(df_exec, "")       # e.g. RSI
        df_1h = calc_ind(df_1h, "_1h")        # e.g. RSI_1h
        df_4h = calc_ind(df_4h, "_4h")        # e.g. RSI_4h
        
        # Calculate ATR on Execution TF for stops
        df_exec['ATR'] = ta.atr(df_exec['High'], df_exec['Low'], df_exec['Close'], length=14)

        # 4. MERGE (The Magic Step)
        # We merge 1h/4h data onto the 5m timestamps using "backward" search 
        # (This means for a 10:05 candle, we take the 1h data available at 10:00)
        df_exec = df_exec.sort_index()
        df_1h = df_1h.sort_index()
        df_4h = df_4h.sort_index()

        # Merge 1H
        df_merged = pd.merge_asof(
            df_exec, 
            df_1h[['RSI_1h', 'Trend_1h']], 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        
        # Merge 4H
        df_merged = pd.merge_asof(
            df_merged, 
            df_4h[['RSI_4h', 'Trend_4h']], 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )

        # 5. Session Filter
        df_merged['Hour'] = df_merged.index.hour
        df_merged['Session'] = "ASIAN"
        df_merged.loc[(df_merged['Hour'] >= 7) & (df_merged['Hour'] <= 16), 'Session'] = "LON"
        df_merged.loc[(df_merged['Hour'] >= 13) & (df_merged['Hour'] <= 21), 'Session'] = "NY"

        df_merged.dropna(inplace=True)
        return df_merged

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

with st.spinner(f"Fetching Multi-Timeframe Data ({EXEC_INTERVAL} + 1H + 4H)..."):
    df = fetch_and_merge_data(SYMBOL, PERIOD, EXEC_INTERVAL)

if df.empty:
    st.error("❌ Data Fetch Failed. Yahoo API might be throttling 5m data. Try waiting 60s.")
    st.stop()

# ==========================================
# 4. AI MODEL & SIMULATION
# ==========================================
def run_simulation(df, threshold, target_rr):
    # Features include MULTI-TIMEFRAME data now
    features = ['RSI', 'RSI_1h', 'RSI_4h', 'Trend_1h', 'Trend_4h', 'ATR']
    
    # Target: Did price go UP in the next candle?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Train/Test Split (Train on first 80%, Test on last 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['Target']
    
    # Train Model
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Live Prediction
    last_candle = df[features].iloc[[-1]]
    live_probs = model.predict_proba(last_candle)[0]
    live_buy_conf = live_probs[1]
    live_sell_conf = live_probs[0]
    
    # --- BACKTEST ON TEST DATA (Last 20% of history) ---
    # We predict on the WHOLE test set at once for speed
    test_probs = model.predict_proba(test_df[features])
    
    trades = []
    final_balance = STARTING_CAPITAL
    
    for i in range(len(test_df) - 1):
        idx = test_df.index[i]
        
        # Skip Asian Session (Optional: Toggle this)
        if test_df['Session'].iloc[i] == "ASIAN": continue

        prob_buy = test_probs[i][1]
        prob_sell = test_probs[i][0]
        
        action = "WAIT"
        if prob_buy > threshold: action = "BUY"
        elif prob_sell > threshold: action = "SELL"
        
        if action != "WAIT":
            entry = test_df['Open'].iloc[i+1]
            atr = test_df['ATR'].iloc[i]
            
            # Risk Calc
            risk_p = atr * 1.0
            reward_p = atr * target_rr
            
            sl = entry - risk_p if action == "BUY" else entry + risk_p
            tp = entry + reward_p if action == "BUY" else entry - reward_p
            
            # Outcome Check
            next_high = test_df['High'].iloc[i+1]
            next_low = test_df['Low'].iloc[i+1]
            next_close = test_df['Close'].iloc[i+1]
            
            outcome = "EXIT"
            pnl_amt = 0
            rr_realized = 0
            
            if action == "BUY":
                if next_low <= sl: 
                    outcome = "SL HIT"
                    rr_realized = -1.0
                elif next_high >= tp: 
                    outcome = "TP HIT"
                    rr_realized = target_rr
                else: 
                    rr_realized = (next_close - entry) / risk_p
            else: # SELL
                if next_high >= sl: 
                    outcome = "SL HIT"
                    rr_realized = -1.0
                elif next_low <= tp: 
                    outcome = "TP HIT"
                    rr_realized = target_rr
                else: 
                    rr_realized = (entry - next_close) / risk_p
            
            # Balance Update (2% Risk)
            risk_dollars = final_balance * 0.02
            pnl_dollars = risk_dollars * rr_realized
            final_balance += pnl_dollars
            
            trades.append({
                "Date": idx.strftime('%m-%d %H:%M'),
                "Session": test_df['Session'].iloc[i],
                "Type": action,
                "Price": entry,
                "RR": rr_realized,
                "PnL": pnl_dollars
            })
            
    win_rate = 0
    if len(trades) > 0:
        wins = sum(1 for t in trades if t['PnL'] > 0)
        win_rate = (wins / len(trades)) * 100
        
    return live_buy_conf, live_sell_conf, pd.DataFrame(trades), final_balance, win_rate

live_buy, live_sell, trade_hist, end_bal, win_rate = run_simulation(df, CONFIDENCE, TARGET_RR)

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================
# --- HEADER ---
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
trend_1h = "BULLISH" if df['Trend_1h'].iloc[-1] == 1 else "BEARISH"

m1.metric("Live Price", f"${curr_price:,.2f}")
m2.metric("4H Trend", trend_4h, delta="Strong" if trend_4h=="BULLISH" else "-Weak", delta_color="normal")
m3.metric("AI Signal", signal, f"{max(live_buy, live_sell)*100:.1f}% Conf")
m4.metric("Session", c_session)

# --- CHART ---
st.markdown("### 📊 Multi-TF Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index[-100:], open=df['Open'][-100:], high=df['High'][-100:], low=df['Low'][-100:], close=df['Close'][-100:], name="Price"))
# Add 4H Trend Overlay (Colored Background)
st.plotly_chart(fig, use_container_width=True)

# --- EXECUTION ---
col_exec, col_data = st.columns([2, 1])
with col_exec:
    st.info(f"**AI Logic:** Combining {EXEC_INTERVAL} RSI with **4H Trend ({trend_4h})** and **1H Trend ({trend_1h})**.")
    st.progress(float(live_buy))
    st.caption(f"Bullish: {live_buy*100:.1f}% | Bearish: {live_sell*100:.1f}%")

with col_data:
    if st.button("🔄 SCAN NOW"): st.rerun()

# --- BACKTEST RESULTS ---
st.markdown("---")
st.subheader(f"🧪 Walk-Forward Backtest (Last 20% of {PERIOD})")
b1, b2, b3, b4 = st.columns(4)
b1.metric("End Balance", f"${end_bal:,.2f}", f"{(end_bal-STARTING_CAPITAL)/STARTING_CAPITAL*100:.1f}%")
b2.metric("Win Rate", f"{win_rate:.1f}%")
b3.metric("Trades", len(trade_hist))
b4.metric("Target RR", f"1:{TARGET_RR}")

if not trade_hist.empty:
    st.dataframe(
        trade_hist.iloc[::-1], # Newest first
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "PnL": st.column_config.NumberColumn(format="$%.2f"),
            "RR": st.column_config.NumberColumn(format="%.2f x"),
        }
    )
