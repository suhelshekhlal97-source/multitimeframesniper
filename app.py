import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import numpy as np
import ccxt

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sniper Bot | AI Tuner Edition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 5px; }
    .stDataFrame { border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Sniper Bot | AI Optimization Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Data Source")

# --- REGION SWITCHER ---
EXCHANGE_OPT = st.sidebar.selectbox(
    "Connection Region",
    ["Binance US (USA)", "Kraken (Global/USA)", "Binance Global (Rest of World)"],
    index=0 
)

exchange_id_map = {
    "Binance US (USA)": "binanceus",
    "Kraken (Global/USA)": "kraken",
    "Binance Global (Rest of World)": "binance"
}
ACTIVE_EXCHANGE_ID = exchange_id_map[EXCHANGE_OPT]

# --- ASSET SELECTOR ---
asset_map = {
    "Bitcoin (BTC)": {"symbol": "BTC/USD", "source": "ccxt"},
    "Ethereum (ETH)": {"symbol": "ETH/USD", "source": "ccxt"},
    "Solana (SOL)": {"symbol": "SOL/USD", "source": "ccxt"},
    "Gold (XAU)": {"symbol": "GC=F", "source": "yf"},
    "US30 (Dow)": {"symbol": "^DJI", "source": "yf"},
}
selected_label = st.sidebar.selectbox("Asset", list(asset_map.keys()))
ASSET_INFO = asset_map[selected_label]
SYMBOL = ASSET_INFO["symbol"]
SOURCE = ASSET_INFO["source"]

# Auto-fix Symbol for Binance
if "Binance" in EXCHANGE_OPT and "/USD" in SYMBOL:
    SYMBOL = SYMBOL.replace("USD", "USDT")

INTERVAL = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h"], index=0)

# --- AI HYPERPARAMETERS (THE FIX) ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Brain Tuning")
st.sidebar.caption("Adjust these to fix accuracy drops.")

MODEL_DEPTH = st.sidebar.slider("Max Depth (Complexity)", 3, 12, 6, help="Higher = Smarter but risks overfitting. Lower = Simpler.")
N_ESTIMATORS = st.sidebar.slider("Training Rounds", 50, 500, 150, help="More rounds = deeper learning.")
LEARNING_RATE = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 60) / 100

# ==========================================
# 3. DATA ENGINE
# ==========================================
@st.cache_data(ttl=30)
def fetch_market_data(source, exchange_id, ticker, interval):
    df = pd.DataFrame()
    try:
        limit = 1500 # Fixed high limit for better AI training
        
        if source == "ccxt":
            exchange_class = getattr(ccxt, exchange_id)()
            ohlcv = exchange_class.fetch_ohlcv(ticker, interval, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC')

        elif source == "yf":
            period = "59d" if interval in ["15m","30m"] else "1y"
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if not df.empty:
            # --- ADVANCED FEATURE ENGINEERING ---
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14'] # Trend Strength
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df.dropna(inplace=True)
            
            # Filter bad ADX (Optional, can be used by AI)
            
        return df

    except Exception as e:
        return pd.DataFrame()

with st.spinner(f"Acquiring Live Data for {SYMBOL}..."):
    df = fetch_market_data(SOURCE, ACTIVE_EXCHANGE_ID, SYMBOL, INTERVAL)

if df.empty:
    st.error("❌ Data Fetch Failed. Check Region/Asset.")
    st.stop()

# ==========================================
# 4. XGBOOST AI ENGINE
# ==========================================
def run_ai_analysis(df, depth, estimators, lr, conf_threshold):
    # 1. Target Definition (Did price go up?)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 2. Features (Inputs for the Brain)
    features = ['RSI', 'MACD', 'ATR', 'ADX']
    
    # 3. Train/Test Split (Time Series strict)
    split = int(len(df) * 0.8) # Train on first 80%, Test on last 20%
    
    train_df = df.iloc[:split]
    test_df = df.iloc[split:-1] # Exclude very last candle (no target yet)
    
    # 4. Initialize & Train Model
    model = XGBClassifier(
        n_estimators=estimators,
        max_depth=depth,
        learning_rate=lr,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(train_df[features], train_df['Target'])
    
    # 5. Predictions on TEST Data (Simulating Real Trading)
    # We predict on the WHOLE dataset to get the plot, but metrics come from Test
    all_probs = model.predict_proba(df[features])
    
    # 6. Live Signal (The very last candle)
    last_prob_buy = all_probs[-1][1]
    last_prob_sell = all_probs[-1][0]
    
    # 7. Backtest Loop (On Test Data Only for Realistic Accuracy)
    trades = []
    balance = 10000
    wins = 0
    total = 0
    
    # Loop through the TEST portion of data
    for i in range(split, len(df)-1):
        prob_buy = all_probs[i][1]
        prob_sell = all_probs[i][0]
        
        # Signal Generation
        action = "WAIT"
        if prob_buy > conf_threshold: action = "BUY"
        elif prob_sell > conf_threshold: action = "SELL"
        
        if action != "WAIT":
            price_in = df['Open'].iloc[i+1]
            price_out = df['Close'].iloc[i+1]
            
            # Simple PnL Check (Close to Close)
            pnl = 0
            if action == "BUY":
                pnl = price_out - price_in
            else:
                pnl = price_in - price_out
                
            if pnl > 0: wins += 1
            total += 1
            
            trades.append({
                "Time": df.index[i], 
                "Type": action, 
                "Price": price_in, 
                "PnL": pnl
            })
            
            balance += (pnl * (10000/price_in)) # Mock sizing

    accuracy = (wins/total*100) if total > 0 else 0
    return last_prob_buy, last_prob_sell, pd.DataFrame(trades), accuracy, balance

# Run AI
live_buy, live_sell, history, acc, final_bal = run_ai_analysis(
    df, MODEL_DEPTH, N_ESTIMATORS, LEARNING_RATE, CONFIDENCE
)

# ==========================================
# 5. DASHBOARD
# ==========================================
# Header
c1, c2, c3 = st.columns(3)
c1.metric("Live Price", f"${df['Close'].iloc[-1]:,.2f}")
c2.metric("AI Model Accuracy (Test Data)", f"{acc:.1f}%", delta_color="normal" if acc > 50 else "inverse")
c3.metric("Data Source", ACTIVE_EXCHANGE_ID.upper())

# Signal Display
st.markdown("### 📡 AI Signal Tower")
s1, s2 = st.columns([1,3])
with s1:
    signal = "NEUTRAL"
    if live_buy > CONFIDENCE: signal = "🐂 BUY"
    elif live_sell > CONFIDENCE: signal = "🐻 SELL"
    
    st.info(f"**SIGNAL:** {signal}")
    st.write(f"Bull Prob: {live_buy:.2f}")
    st.write(f"Bear Prob: {live_sell:.2f}")

with s2:
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index[-100:], open=df['Open'][-100:], high=df['High'][-100:], low=df['Low'][-100:], close=df['Close'][-100:], name="Price"))
    # Add Buy/Sell Markers from history
    if not history.empty:
        recent_trades = history[history['Time'] > df.index[-100]]
        buys = recent_trades[recent_trades['Type'] == "BUY"]
        sells = recent_trades[recent_trades['Type'] == "SELL"]
        
        fig.add_trace(go.Scatter(x=buys['Time'], y=buys['Price'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name="AI Buy"))
        fig.add_trace(go.Scatter(x=sells['Time'], y=sells['Price'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name="AI Sell"))
        
    fig.update_layout(height=400, template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# Trade List
st.subheader("📜 Recent AI Executions")
if not history.empty:
    st.dataframe(history.sort_values(by="Time", ascending=False).head(10), use_container_width=True)
