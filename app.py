import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import numpy as np

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sniper Bot Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
    }
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Sniper Bot | Live Execution Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

# --- ASSET LIST ---
asset_map = {
    # Crypto
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    
    # Forex
    "Gold (XAU/USD)": "GC=F",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/JPY": "GBPJPY=X",
    
    # Indices
    "US30 (Dow Jones)": "^DJI",
    "US500 (S&P 500)": "^GSPC",
    "German Index (DAX)": "^GDAXI",
    "Japan Index (Nikkei 225)": "^N225"
}

selected_asset = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
SYMBOL = asset_map[selected_asset]

# --- NEW: TIMEFRAME SELECTION ---
INTERVAL = st.sidebar.selectbox("Execution Timeframe", ["1h", "30m", "15m"], index=0)

# --- NEW: PERIOD LOGIC (Max 59 days for Intraday) ---
# Yahoo Finance limits 15m/30m data to the last 60 days. 
# We set it to "59d" to be safe and compliant with your "less than 2 months" request.
if INTERVAL in ["15m", "30m"]:
    st.sidebar.info(f"⚠️ {INTERVAL} data is limited to last 59 days by Exchange.")
    PERIOD = "59d"
else:
    # For 1h, we give the option, but keep it low as requested
    PERIOD = st.sidebar.select_slider("Data Lookback", options=["1mo", "59d"], value="59d")

CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
RR_RATIO_STR = st.sidebar.select_slider("Risk:Reward Target", options=["1:1", "1:1.5", "1:2", "1:3"], value="1:2")
STARTING_CAPITAL = 10000

# Parse RR Ratio
TARGET_RR = float(RR_RATIO_STR.split(":")[1])

st.sidebar.markdown("---")
st.sidebar.caption(f"Strategy: **XGBoost Trend + Session Filter**")

# ==========================================
# 3. BACKEND ENGINE
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Calculate Indicators
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['MACD'] = ta.macd(data['Close'])['MACD_12_26_9']
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        data['SMA_200'] = ta.sma(data['Close'], length=200)
        
        # Session Filter (UTC Hours)
        data['Hour'] = data.index.hour
        data['Session'] = "ASIAN"
        data.loc[(data['Hour'] >= 7) & (data['Hour'] <= 16), 'Session'] = "LON"
        data.loc[(data['Hour'] >= 13) & (data['Hour'] <= 21), 'Session'] = "NY"
        
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()

with st.spinner(f"Fetching {PERIOD} of {INTERVAL} Data for {SYMBOL}..."):
    df = fetch_market_data(SYMBOL, PERIOD, INTERVAL)

if df.empty:
    st.error(f"❌ Connection Failed. Market data for {SYMBOL} is unavailable (Market might be closed or API limit reached).")
    st.stop()

# ==========================================
# 4. AI & ADVANCED BACKTEST ENGINE
# ==========================================
def run_simulation(df, threshold, target_rr):
    # 1. Prepare Data
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'MACD', 'ATR']
    
    # 2. Train Model
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]
    
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(X, y)
    
    # 3. Get Probabilities
    all_probs = model.predict_proba(df[features]) 
    
    # 4. Live Signal
    live_prob_buy = all_probs[-1][1]
    live_prob_sell = all_probs[-1][0]
    
    # 5. Run Realistic Backtest (High/Low Check)
    trades = []
    
    # Skip first 50 candles for indicators to settle
    start_idx = 50
    if len(df) < 50: start_idx = 0

    for i in range(start_idx, len(df)-1):
        # Data for Decision
        session = df['Session'].iloc[i]
        if session == "ASIAN": continue # Skip Asian
            
        prob_buy = all_probs[i][1]
        prob_sell = all_probs[i][0]
        atr = df['ATR'].iloc[i]
        
        action = "WAIT"
        if prob_buy > threshold: action = "BUY"
        elif prob_sell > threshold: action = "SELL"
            
        if action != "WAIT":
            # Trade Parameters
            entry_price = df['Open'].iloc[i+1]
            risk_dist = atr * 1.0 # Stop Loss Distance
            reward_dist = atr * target_rr # Take Profit Distance
            
            sl_price = 0
            tp_price = 0
            
            if action == "BUY":
                sl_price = entry_price - risk_dist
                tp_price = entry_price + reward_dist
            else: # SELL
                sl_price = entry_price + risk_dist
                tp_price = entry_price - reward_dist
            
            # Check Outcome on NEXT Candle
            next_high = df['High'].iloc[i+1]
            next_low = df['Low'].iloc[i+1]
            next_close = df['Close'].iloc[i+1]
            
            result = "EXIT"
            pnl = 0
            achieved_rr = 0.0
            
            # Logic: Did we hit TP or SL?
            if action == "BUY":
                if next_low <= sl_price:
                    result = "SL HIT"
                    pnl = -risk_dist
                    achieved_rr = -1.0
                elif next_high >= tp_price:
                    result = "TP HIT"
                    pnl = reward_dist
                    achieved_rr = target_rr
                else:
                    pnl = next_close - entry_price
                    achieved_rr = pnl / risk_dist
            
            elif action == "SELL":
                if next_high >= sl_price:
                    result = "SL HIT"
                    pnl = -risk_dist
                    achieved_rr = -1.0
                elif next_low <= tp_price:
                    result = "TP HIT"
                    pnl = reward_dist
                    achieved_rr = target_rr
                else:
                    pnl = entry_price - next_close
                    achieved_rr = pnl / risk_dist

            trades.append({
                "Date": df.index[i].strftime('%Y-%m-%d %H:%M'),
                "Type": action,
                "Entry": entry_price,
                "SL": sl_price,
                "TP": tp_price,
                "Result": result,
                "RR Achieved": achieved_rr,
                "PnL": pnl
            })
            
    # Calculate Summary Stats
    total_trades = len(trades)
    win_rate = 0
    final_balance = STARTING_CAPITAL
    
    if total_trades > 0:
        wins = sum(1 for t in trades if t['PnL'] > 0)
        win_rate = (wins / total_trades) * 100
        
        for t in trades:
             # Dynamic Position Sizing: Risk 2% of current balance
             risk_amt = final_balance * 0.02
             
             # Calculate Change based on RR
             if t['Result'] == "SL HIT":
                 final_balance -= risk_amt
             elif t['Result'] == "TP HIT":
                 final_balance += (risk_amt * target_rr)
             else:
                 # Partial Result (Time Exit)
                 final_balance += (risk_amt * t['RR Achieved'])
                 
    return live_prob_buy, live_prob_sell, pd.DataFrame(trades), final_balance, win_rate, total_trades

prob_buy, prob_sell, trade_history, final_bal, win_rate, total_trades = run_simulation(df, CONFIDENCE, TARGET_RR)

# ==========================================
# 5. DASHBOARD UI
# ==========================================
current_hour = datetime.now(timezone.utc).hour
current_session = "ASIAN (Sleep)"
session_color = "off"
if 7 <= current_hour <= 16:
    current_session = "LONDON (Active)"
    session_color = "normal"
elif 13 <= current_hour <= 21:
    current_session = "NEW YORK (Active)"
    session_color = "normal"

signal = "WAIT"
if prob_buy > CONFIDENCE:
    signal = "BUY"
elif prob_sell > CONFIDENCE:
    signal = "SELL"

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
current_price = df['Close'].iloc[-1]
change = current_price - df['Open'].iloc[-1]
atr = df['ATR'].iloc[-1]

col1.metric("Live Price", f"${current_price:,.2f}", f"{change:.2f}")
col2.metric("Current Session", current_session, delta_color=session_color)
col3.metric("AI Signal", signal, f"{max(prob_buy, prob_sell)*100:.1f}% Conf")
col4.metric("Risk (1 ATR)", f"${atr:.2f}")

# --- CHART ---
st.markdown("### 📊 Live Market Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name="SMA 200"))
fig.update_layout(height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig, use_container_width=True)

# --- EXECUTION PANEL ---
c1, c2 = st.columns([2, 1])
with c1:
    bull_conf = float(prob_buy)
    bear_conf = float(prob_sell)
    st.info(f"**AI Prediction ({INTERVAL}):** {bull_conf*100:.1f}% Bullish vs {bear_conf*100:.1f}% Bearish")
    st.progress(bull_conf)
with c2:
    if st.button("🔄 REFRESH DATA"):
        st.rerun()

# --- PERFORMANCE SECTION ---
st.markdown("---")
st.subheader(f"📈 Backtest Performance (Last {PERIOD})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Final Balance", f"${final_bal:,.2f}", f"{(final_bal-STARTING_CAPITAL)/STARTING_CAPITAL*100:.1f}% Return")
m2.metric("Win Rate", f"{win_rate:.1f}%")
m3.metric("Total Trades", total_trades)
m4.metric("Target RR", f"1:{TARGET_RR}")

if not trade_history.empty:
    st.dataframe(
        trade_history.sort_index(ascending=False),
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "SL": st.column_config.NumberColumn("Stop Loss", format="%.2f"),
            "TP": st.column_config.NumberColumn("Take Profit", format="%.2f"),
            "RR Achieved": st.column_config.NumberColumn("Achieved RR", format="%.2f x"),
            "Result": st.column_config.TextColumn("Outcome"),
        }
    )
else:
    st.warning("No trades found in this period matching your Confidence Threshold.")
