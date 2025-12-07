import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import ccxt  # The library for unlimited crypto data

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sniper Bot | Hybrid Terminal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the terminal look
st.markdown("""
<style>
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 5px; }
    .stDataFrame { border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Sniper Bot | Live Data Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Data Connection")

# --- NEW: EXCHANGE SELECTOR (THE FIX) ---
# This dropdown allows you to bypass the Geo-Block
EXCHANGE_OPT = st.sidebar.selectbox(
    "Select Data Source (Region)",
    ["Binance US (USA Users)", "Kraken (USA/Global)", "Binance Global (Rest of World)"],
    index=0 
)

# Map the selection to the correct CCXT ID
exchange_id_map = {
    "Binance US (USA Users)": "binanceus",
    "Kraken (USA/Global)": "kraken",
    "Binance Global (Rest of World)": "binance"
}
ACTIVE_EXCHANGE_ID = exchange_id_map[EXCHANGE_OPT]

# --- ASSET SELECTION ---
asset_map = {
    # Crypto (Uses CCXT)
    "Bitcoin (BTC/USD)": {"symbol": "BTC/USD", "source": "ccxt"},
    "Ethereum (ETH/USD)": {"symbol": "ETH/USD", "source": "ccxt"},
    "Solana (SOL/USD)": {"symbol": "SOL/USD", "source": "ccxt"},
    
    # Forex/Indices (Uses Yahoo Finance)
    "Gold (XAU/USD)": {"symbol": "GC=F", "source": "yf"},
    "EUR/USD": {"symbol": "EURUSD=X", "source": "yf"},
    "US30 (Dow Jones)": {"symbol": "^DJI", "source": "yf"},
    "US500 (S&P 500)": {"symbol": "^GSPC", "source": "yf"},
}

selected_label = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
ASSET_INFO = asset_map[selected_label]
SYMBOL = ASSET_INFO["symbol"]
SOURCE = ASSET_INFO["source"]

# If using Binance US/Global, they prefer USDT pairs usually, but Kraken likes USD.
# We auto-adjust for Binance to ensure data flows.
if "Binance" in EXCHANGE_OPT and "/USD" in SYMBOL and "XAU" not in selected_label:
    SYMBOL = SYMBOL.replace("USD", "USDT") 

# --- TIMEFRAME & LIMITS ---
INTERVAL = st.sidebar.selectbox("Timeframe", ["1h", "30m", "15m", "5m"], index=1)

if SOURCE == "ccxt":
    st.sidebar.success(f"✅ Connected to {EXCHANGE_OPT}")
    CANDLE_LIMIT = st.sidebar.slider("Candle History (Unlimited)", 500, 5000, 1000)
else:
    st.sidebar.warning(f"⚠️ Yahoo Finance (Limited Data)")
    PERIOD = "59d" if INTERVAL in ["15m", "30m", "1h"] else "1y"

CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
TARGET_RR = float(st.sidebar.select_slider("Risk:Reward", ["1:1", "1:2", "1:3"], value="1:2").split(":")[1])

# ==========================================
# 3. HYBRID DATA ENGINE (PATCHED)
# ==========================================
@st.cache_data(ttl=15)
def fetch_market_data(source, exchange_id, ticker, interval, limit_or_period):
    df = pd.DataFrame()
    try:
        # --- ENGINE 1: CCXT (CRYPTO) ---
        if source == "ccxt":
            # Dynamically load the exchange based on user selection
            exchange_class = getattr(ccxt, exchange_id)()
            
            # Map intervals for Kraken/Binance
            # Kraken uses integers for minutes (e.g., 60) in some versions, but string '1h' usually works in modern ccxt
            
            ohlcv = exchange_class.fetch_ohlcv(ticker, interval, limit=limit_or_period)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC')

        # --- ENGINE 2: YFINANCE (STOCKS) ---
        elif source == "yf":
            df = yf.download(ticker, period=limit_or_period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        # --- INDICATORS ---
        if not df.empty:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df.dropna(inplace=True)
            
            # Session Logic
            df['Hour'] = df.index.hour
            df['Session'] = "ASIAN"
            df.loc[(df['Hour'] >= 7) & (df['Hour'] <= 16), 'Session'] = "LON"
            df.loc[(df['Hour'] >= 13) & (df['Hour'] <= 21), 'Session'] = "NY"
            
        return df

    except Exception as e:
        st.error(f"⚠️ API Error: {e}")
        return pd.DataFrame()

# Fetch Data
arg_limit = CANDLE_LIMIT if SOURCE == "ccxt" else PERIOD
with st.spinner(f"Fetching data from {ACTIVE_EXCHANGE_ID.upper()}..."):
    df = fetch_market_data(SOURCE, ACTIVE_EXCHANGE_ID, SYMBOL, INTERVAL, arg_limit)

if df.empty:
    st.error("❌ No data received. Try switching the 'Data Source' in the sidebar.")
    st.stop()

# ==========================================
# 4. TRADING LOGIC
# ==========================================
# (Simplified for brevity - Strategy Engine)
def run_strategy(data):
    # Mock signal generation for visualization
    data['Signal'] = 0
    data['Signal'] = np.where(data['RSI'] < 30, 1, 0) # Buy
    data['Signal'] = np.where(data['RSI'] > 70, -1, data['Signal']) # Sell
    return data

df = run_strategy(df)
latest = df.iloc[-1]

# ==========================================
# 5. DASHBOARD
# ==========================================
c1, c2, c3 = st.columns(3)
c1.metric("Price", f"${latest['Close']:,.2f}")
c2.metric("RSI", f"{latest['RSI']:.1f}")
c3.metric("Source", ACTIVE_EXCHANGE_ID.upper())

st.markdown(f"### 📈 {SYMBOL} Live Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange'), name="SMA 50"))
fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.success(f"Loaded {len(df)} candles from {ACTIVE_EXCHANGE_ID} successfully.")
