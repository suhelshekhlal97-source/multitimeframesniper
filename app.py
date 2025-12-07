import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import numpy as np  # <--- FIXED: Re-added this missing import
import ccxt  # The library for unlimited crypto data

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sniper Bot | Live Terminal",
    page_icon="🎯",
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

st.title("🎯 Sniper Bot | AI Hybrid Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Data Connection")

# --- REGION / EXCHANGE SWITCHER ---
EXCHANGE_OPT = st.sidebar.selectbox(
    "Select Data Source (Region)",
    ["Binance US (USA Users)", "Kraken (USA/Global)", "Binance Global (Rest of World)"],
    index=0 
)

# Map selection to CCXT ID
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

# Auto-correct symbol for Binance (they prefer USDT often)
if "Binance" in EXCHANGE_OPT and "/USD" in SYMBOL and "XAU" not in selected_label:
    SYMBOL = SYMBOL.replace("USD", "USDT") 

# --- TIMEFRAME & LIMITS ---
INTERVAL = st.sidebar.selectbox("Timeframe", ["1h", "30m", "15m", "5m"], index=1)

if SOURCE == "ccxt":
    st.sidebar.success(f"✅ Connected to {EXCHANGE_OPT} (Live)")
    CANDLE_LIMIT = st.sidebar.slider("Candle History (Unlimited)", 500, 5000, 1000)
else:
    st.sidebar.warning(f"⚠️ Yahoo Finance (Limited History)")
    # Yahoo limits intraday data to 60 days
    PERIOD = "59d" if INTERVAL in ["15m", "30m", "1h"] else "1y"

CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
TARGET_RR = float(st.sidebar.select_slider("Risk:Reward", ["1:1", "1:2", "1:3"], value="1:2").split(":")[1])

# ==========================================
# 3. HYBRID DATA ENGINE
# ==========================================
@st.cache_data(ttl=15)
def fetch_market_data(source, exchange_id, ticker, interval, limit_or_period):
    df = pd.DataFrame()
    try:
        # --- ENGINE 1: CCXT (CRYPTO) ---
        if source == "ccxt":
            exchange_class = getattr(ccxt, exchange_id)()
            # Fetch OHLCV
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
            df['SMA_200'] = ta.sma(df['Close'], length=200)
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
    st.error("❌ No data received. Try switching the 'Data Source' to Kraken if Binance US fails.")
    st.stop()

# ==========================================
# 4. AI & BACKTEST ENGINE (RESTORED)
# ==========================================
def run_simulation(df, threshold, target_rr):
    # 1. Prepare Data
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'MACD', 'ATR']
    
    if len(df) < 50: return 0.5, 0.5, pd.DataFrame(), 10000, 0, 0

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
    
    # 5. Backtest Loop
    trades = []
    start_idx = 50 
    final_balance = 10000
    
    for i in range(start_idx, len(df)-1):
        if SOURCE == "yf" and df['Session'].iloc[i] == "ASIAN": continue 
            
        prob_buy = all_probs[i][1]
        prob_sell = all_probs[i][0]
        atr = df['ATR'].iloc[i]
        
        action = "WAIT"
        if prob_buy > threshold: action = "BUY"
        elif prob_sell > threshold: action = "SELL"
            
        if action != "WAIT":
            entry_price = df['Open'].iloc[i+1]
            risk_dist = atr * 1.0 
            reward_dist = atr * target_rr
            
            sl_price = entry_price - risk_dist if action == "BUY" else entry_price + risk_dist
            tp_price = entry_price + reward_dist if action == "BUY" else entry_price - reward_dist
            
            # Check Outcome
            next_high = df['High'].iloc[i+1]
            next_low = df['Low'].iloc[i+1]
            next_close = df['Close'].iloc[i+1]
            
            result = "EXIT"
            pnl = 0
            
            if action == "BUY":
                if next_low <= sl_price:
                    result = "SL HIT"
                    pnl = -risk_dist
                elif next_high >= tp_price:
                    result = "TP HIT"
                    pnl = reward_dist
                else:
                    pnl = next_close - entry_price
            
            elif action == "SELL":
                if next_high >= sl_price:
                    result = "SL HIT"
                    pnl = -risk_dist
                elif next_low <= tp_price:
                    result = "TP HIT"
                    pnl = reward_dist
                else:
                    pnl = entry_price - next_close

            trades.append({
                "Date": df.index[i].strftime('%Y-%m-%d %H:%M'),
                "Type": action,
                "Entry": entry_price,
                "SL": sl_price,
                "TP": tp_price,
                "Result": result,
                "PnL": pnl
            })
            
            # Simple Balance Calc
            if result == "SL HIT": final_balance -= (final_balance * 0.02)
            elif result == "TP HIT": final_balance += (final_balance * 0.02 * target_rr)
                
    total_trades = len(trades)
    win_rate = 0
    if total_trades > 0:
        wins = sum(1 for t in trades if t['PnL'] > 0)
        win_rate = (wins / total_trades) * 100
                  
    return live_prob_buy, live_prob_sell, pd.DataFrame(trades), final_balance, win_rate, total_trades

# Run the Full AI Simulation
prob_buy, prob_sell, trade_history, final_bal, win_rate, total_trades = run_simulation(df, CONFIDENCE, TARGET_RR)

# ==========================================
# 5. DASHBOARD UI
# ==========================================
current_hour = datetime.now(timezone.utc).hour
current_session = "ASIAN"
if 7 <= current_hour <= 16: current_session = "LONDON"
elif 13 <= current_hour <= 21: current_session = "NEW YORK"

signal = "WAIT"
if prob_buy > CONFIDENCE: signal = "BUY"
elif prob_sell > CONFIDENCE: signal = "SELL"

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
current_price = df['Close'].iloc[-1]
change = current_price - df['Open'].iloc[-1]

col1.metric("Asset Price", f"${current_price:,.2f}", f"{change:.2f}")
col2.metric("Session (UTC)", current_session)
col3.metric("AI Signal", signal, f"{max(prob_buy, prob_sell)*100:.1f}% Conf")
col4.metric("Source", ACTIVE_EXCHANGE_ID.upper())

# --- CHART ---
st.markdown(f"### 📊 {SYMBOL} Live Market Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name="SMA 200"))
fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig, use_container_width=True)

# --- RESULTS ---
c1, c2 = st.columns([2, 1])
with c1:
    st.info(f"**AI Prediction:** Bullish {prob_buy*100:.1f}% | Bearish {prob_sell*100:.1f}%")
    st.progress(float(prob_buy))

st.markdown("---")
st.subheader("📈 Backtest Performance")
m1, m2, m3 = st.columns(3)
m1.metric("Est. Final Balance", f"${final_bal:,.2f}", f"{(final_bal-10000)/10000*100:.1f}%")
m2.metric("Win Rate", f"{win_rate:.1f}%")
m3.metric("Total Trades", total_trades)

if not trade_history.empty:
    st.dataframe(trade_history.sort_index(ascending=False), use_container_width=True, hide_index=True)
else:
    st.warning("No trades found in this period matching your parameters.")
