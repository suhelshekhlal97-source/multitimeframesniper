import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import numpy as np  # Essential for the math to work
import ccxt  # For Live Crypto Data

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sniper Bot | Master Terminal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Pro Look
st.markdown("""
<style>
    .stMetric { background-color: #151515; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .stDataFrame { border: 1px solid #333; }
    div[data-testid="stSidebar"] { background-color: #111; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Sniper Bot | Multi-Strategy Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🔌 Connection")

# --- REGION SWITCHER (Fixes Error 451) ---
EXCHANGE_OPT = st.sidebar.selectbox(
    "Data Region",
    ["Binance US (USA Users)", "Kraken (Global/USA)", "Binance Global (Rest of World)"],
    index=0 
)

exchange_id_map = {
    "Binance US (USA Users)": "binanceus",
    "Kraken (Global/USA)": "kraken",
    "Binance Global (Rest of World)": "binance"
}
ACTIVE_EXCHANGE_ID = exchange_id_map[EXCHANGE_OPT]

# --- ASSET SELECTION ---
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

# Auto-Fix Symbol for Binance (They use USDT usually)
if "Binance" in EXCHANGE_OPT and "/USD" in SYMBOL:
    SYMBOL = SYMBOL.replace("USD", "USDT")

INTERVAL = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Strategy Logic")

# --- STRATEGY SWITCHER (THE FIX) ---
STRATEGY_TYPE = st.sidebar.radio(
    "Select Trading Mode:",
    ["♻️ RSI Reversal (High Win Rate)", "🤖 AI Trend Sniper (Big Moves)"]
)

# Settings based on strategy
if "RSI" in STRATEGY_TYPE:
    st.sidebar.caption("Buys Low, Sells High. Great for chop.")
    RSI_OVERBOUGHT = st.sidebar.slider("Overbought (Sell)", 70, 90, 75)
    RSI_OVERSOLD = st.sidebar.slider("Oversold (Buy)", 10, 30, 25)
else:
    st.sidebar.caption("Uses XGBoost to find breakouts.")
    CONFIDENCE = st.sidebar.slider("Min Confidence", 50, 90, 60) / 100

# ==========================================
# 3. ROBUST DATA ENGINE
# ==========================================
@st.cache_data(ttl=15)
def fetch_market_data(source, exchange_id, ticker, interval):
    df = pd.DataFrame()
    try:
        limit = 1000 # Enough for backtest
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
            # Indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df.dropna(inplace=True)
            
        return df

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

with st.spinner(f"Connecting to {ACTIVE_EXCHANGE_ID.upper()}..."):
    df = fetch_market_data(SOURCE, ACTIVE_EXCHANGE_ID, SYMBOL, INTERVAL)

if df.empty:
    st.error("❌ Data not found. Try switching Region to 'Kraken' or 'Binance US'.")
    st.stop()

# ==========================================
# 4. DUAL-ENGINE STRATEGY CORE
# ==========================================
def run_strategy_engine(df, mode):
    trades = []
    balance = 10000
    
    # 1. GENERATE SIGNALS
    if "RSI" in mode:
        # --- LOGIC A: RSI MEAN REVERSION (High Win Rate) ---
        # Buy when RSI < 25, Sell when RSI > 75
        df['Signal'] = 0
        df['Signal'] = np.where(df['RSI'] < RSI_OVERSOLD, 1, df['Signal'])
        df['Signal'] = np.where(df['RSI'] > RSI_OVERBOUGHT, -1, df['Signal'])
        
        # Probabilities are just mock 100% for this manual strategy
        live_prob_buy = 0.9 if df['RSI'].iloc[-1] < RSI_OVERSOLD else 0.0
        live_prob_sell = 0.9 if df['RSI'].iloc[-1] > RSI_OVERBOUGHT else 0.0
        
    else:
        # --- LOGIC B: AI XGBOOST (Trend) ---
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        features = ['RSI', 'MACD', 'ATR']
        
        train_size = int(len(df)*0.8)
        X_train = df[features].iloc[:train_size]
        y_train = df['Target'].iloc[:train_size]
        
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05)
        model.fit(X_train, y_train)
        
        all_probs = model.predict_proba(df[features])
        df['Prob_Buy'] = all_probs[:, 1]
        df['Prob_Sell'] = all_probs[:, 0]
        
        df['Signal'] = 0
        df['Signal'] = np.where(df['Prob_Buy'] > CONFIDENCE, 1, df['Signal'])
        df['Signal'] = np.where(df['Prob_Sell'] > CONFIDENCE, -1, df['Signal'])
        
        live_prob_buy = df['Prob_Buy'].iloc[-1]
        live_prob_sell = df['Prob_Sell'].iloc[-1]

    # 2. RUN BACKTEST SIMULATION
    in_position = False
    entry_price = 0
    position_type = "NONE"
    
    # Simple loop to calculate PnL
    for i in range(1, len(df)-1):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]
        
        # ENTRY LOGIC
        if not in_position and signal != 0:
            entry_price = price
            position_type = "BUY" if signal == 1 else "SELL"
            in_position = True
            trades.append({
                "Time": df.index[i], "Type": "ENTRY " + position_type, 
                "Price": price, "PnL": 0, "Balance": balance
            })
            
        # EXIT LOGIC (Simple Reversal or Fixed TP/SL Logic)
        elif in_position:
            # Exit if signal reverses OR we just use a 1-bar hold for simplicity in this demo
            # Let's use Signal Reversal to exit
            exit_now = False
            if position_type == "BUY" and signal == -1: exit_now = True
            if position_type == "SELL" and signal == 1: exit_now = True
            
            # Or exit if RSI goes neutral (for RSI strategy)
            if "RSI" in mode and 40 < df['RSI'].iloc[i] < 60: exit_now = True
            
            if exit_now:
                pnl = 0
                if position_type == "BUY": pnl = price - entry_price
                else: pnl = entry_price - price
                
                balance += (pnl * (10000/price)) # Mock sizing
                trades.append({
                    "Time": df.index[i], "Type": "EXIT " + position_type, 
                    "Price": price, "PnL": pnl, "Balance": balance
                })
                in_position = False

    # Stats Calculation
    df_trades = pd.DataFrame(trades)
    win_rate = 0
    if not df_trades.empty:
        exits = df_trades[df_trades['Type'].str.contains("EXIT")]
        if not exits.empty:
            wins = len(exits[exits['PnL'] > 0])
            win_rate = (wins / len(exits)) * 100

    return live_prob_buy, live_prob_sell, df_trades, win_rate, balance

# Execute
prob_buy, prob_sell, trade_log, win_rate, final_bal = run_strategy_engine(df, STRATEGY_TYPE)

# ==========================================
# 5. DASHBOARD UI
# ==========================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Live Price", f"${df['Close'].iloc[-1]:,.2f}")
c2.metric("Strategy", "Reversal ♻️" if "RSI" in STRATEGY_TYPE else "Trend 🤖")
c3.metric("Win Rate", f"{win_rate:.1f}%", delta="High Accuracy" if win_rate > 70 else None)
c4.metric("Est. Balance", f"${final_bal:,.2f}")

# SIGNAL
signal_text = "WAIT"
if prob_buy > 0.5: signal_text = "BULLISH 🟢"
if prob_sell > 0.5: signal_text = "BEARISH 🔴"
st.progress(float(prob_buy) if prob_buy > 0 else 0.0)
st.caption(f"Signal Strength: {signal_text} (Buy Prob: {prob_buy:.2f})")

# CHART
st.markdown("### 📊 Market Overview")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
# Add Trades
if not trade_log.empty:
    entries = trade_log[trade_log['Type'].str.contains("ENTRY")]
    exits = trade_log[trade_log['Type'].str.contains("EXIT")]
    fig.add_trace(go.Scatter(x=entries['Time'], y=entries['Price'], mode='markers', marker=dict(color='blue', size=8), name="Entry"))
    fig.add_trace(go.Scatter(x=exits['Time'], y=exits['Price'], mode='markers', marker=dict(color='orange', size=8), name="Exit"))

fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig, use_container_width=True)

# TRADE LOG
if not trade_log.empty:
    st.subheader("📜 Execution Log")
    st.dataframe(trade_log.sort_values(by="Time", ascending=False), use_container_width=True)
else:
    st.info("No trades generated yet. Market conditions do not match strategy rules.")
