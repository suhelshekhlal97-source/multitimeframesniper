import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import numpy as np
import ccxt  # NEW: For Unlimited Crypto Data

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

st.title("🎯 Sniper Bot | Hybrid Data Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

# --- ASSET MAP (Split by Source) ---
# Format: "Label": {"symbol": "Ticker", "source": "ccxt" or "yf"}
asset_map = {
    # --- CRYPTO (UNLIMITED DATA VIA CCXT) ---
    "Bitcoin (BTC/USDT)": {"symbol": "BTC/USDT", "source": "ccxt"},
    "Ethereum (ETH/USDT)": {"symbol": "ETH/USDT", "source": "ccxt"},
    "Solana (SOL/USDT)": {"symbol": "SOL/USDT", "source": "ccxt"},
    
    # --- FOREX (LIMITED VIA YFINANCE) ---
    "Gold (XAU/USD)": {"symbol": "GC=F", "source": "yf"},
    "EUR/USD": {"symbol": "EURUSD=X", "source": "yf"},
    "USD/JPY": {"symbol": "JPY=X", "source": "yf"},
    
    # --- INDICES (LIMITED VIA YFINANCE) ---
    "US30 (Dow Jones)": {"symbol": "^DJI", "source": "yf"},
    "US500 (S&P 500)": {"symbol": "^GSPC", "source": "yf"},
}

selected_label = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
ASSET_INFO = asset_map[selected_label]
SYMBOL = ASSET_INFO["symbol"]
SOURCE = ASSET_INFO["source"]

# --- TIMEFRAME SELECTION ---
INTERVAL = st.sidebar.selectbox("Execution Timeframe", ["1h", "30m", "15m", "5m"], index=0)

# --- DYNAMIC PERIOD LOGIC ---
PERIOD = "59d" # Default for YFinance limitation

if SOURCE == "ccxt":
    st.sidebar.success(f"✅ Unlimited Data Mode Active (Binance API)")
    # For CCXT we fetch by limit (number of candles), not "period"
    CANDLE_LIMIT = st.sidebar.slider("Candle Lookback Amount", 500, 5000, 1000)
else:
    # YFinance Restrictions
    if INTERVAL in ["1h", "30m", "15m", "5m"]:
        st.sidebar.warning(f"⚠️ YFinance limits {INTERVAL} data to last 60 days.")
        PERIOD = "59d" # Max safe buffer
    else:
        PERIOD = st.sidebar.select_slider("Data Lookback", options=["1mo", "3mo", "1y"], value="1y")

CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
RR_RATIO_STR = st.sidebar.select_slider("Risk:Reward Target", options=["1:1", "1:1.5", "1:2", "1:3"], value="1:2")
STARTING_CAPITAL = 10000
TARGET_RR = float(RR_RATIO_STR.split(":")[1])

st.sidebar.markdown("---")
st.sidebar.caption(f"Source: **{SOURCE.upper()}** | Strategy: **XGBoost Trend**")

# ==========================================
# 3. HYBRID DATA ENGINE
# ==========================================
@st.cache_data(ttl=15) # Refresh every 15 seconds for crypto
def fetch_market_data(asset_info, interval, period_or_limit):
    source = asset_info['source']
    ticker = asset_info['symbol']
    
    df = pd.DataFrame()
    
    try:
        # --- ENGINE 1: CCXT (CRYPTO) ---
        if source == "ccxt":
            exchange = ccxt.binance()
            # Map intervals to CCXT format
            timeframe = interval 
            
            # Fetch OHLCV
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=period_or_limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Force timezone to UTC to match YF logic
            df.index = df.index.tz_localize('UTC')

        # --- ENGINE 2: YFINANCE (TRADFI) ---
        elif source == "yf":
            df = yf.download(ticker, period=period_or_limit, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty: return df

        # --- COMMON INDICATORS ---
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        # Session Filter (UTC Hours)
        df['Hour'] = df.index.hour
        df['Session'] = "ASIAN"
        df.loc[(df['Hour'] >= 7) & (df['Hour'] <= 16), 'Session'] = "LON"
        df.loc[(df['Hour'] >= 13) & (df['Hour'] <= 21), 'Session'] = "NY"
        
        df.dropna(inplace=True)
        return df

    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

# Fetch Data based on source type
arg_limit = CANDLE_LIMIT if SOURCE == "ccxt" else PERIOD
with st.spinner(f"Fetching Live Data for {SYMBOL} via {SOURCE.upper()}..."):
    df = fetch_market_data(ASSET_INFO, INTERVAL, arg_limit)

if df.empty:
    st.error(f"❌ Connection Failed for {SYMBOL}.")
    st.stop()

# ==========================================
# 4. AI & BACKTEST ENGINE
# ==========================================
def run_simulation(df, threshold, target_rr):
    # 1. Prepare Data
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'MACD', 'ATR']
    
    # Check if we have enough data
    if len(df) < 50:
        return 0.5, 0.5, pd.DataFrame(), STARTING_CAPITAL, 0, 0

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
    
    # 5. Run Backtest
    trades = []
    start_idx = 50 
    
    # Simulation Loop
    final_balance = STARTING_CAPITAL
    
    for i in range(start_idx, len(df)-1):
        # Filter Asian Session for Forex (Optional: keep for crypto?)
        if SOURCE == "yf" and df['Session'].iloc[i] == "ASIAN": 
            continue 
            
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
            
            # Next Candle Check
            next_high = df['High'].iloc[i+1]
            next_low = df['Low'].iloc[i+1]
            next_close = df['Close'].iloc[i+1]
            
            result = "EXIT"
            pnl = 0
            achieved_rr = 0.0
            
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
            
            # Calc Balance
            risk_amt = final_balance * 0.02
            if result == "SL HIT": final_balance -= risk_amt
            elif result == "TP HIT": final_balance += (risk_amt * target_rr)
            else: final_balance += (risk_amt * achieved_rr)
                
    total_trades = len(trades)
    win_rate = 0
    if total_trades > 0:
        wins = sum(1 for t in trades if t['PnL'] > 0)
        win_rate = (wins / total_trades) * 100
                  
    return live_prob_buy, live_prob_sell, pd.DataFrame(trades), final_balance, win_rate, total_trades

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

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
current_price = df['Close'].iloc[-1]
change = current_price - df['Open'].iloc[-1]

col1.metric("Asset Price", f"{current_price:,.2f}", f"{change:.2f}")
col2.metric("Session (UTC)", current_session)
col3.metric("AI Signal", signal, f"{max(prob_buy, prob_sell)*100:.1f}% Conf")
col4.metric("Source", "Binance (Live)" if SOURCE == "ccxt" else "Yahoo (Delayed)")

# --- CHART ---
st.markdown(f"### 📊 Market Analysis: {SYMBOL}")
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
m1.metric("Final Balance", f"${final_bal:,.2f}", f"{(final_bal-STARTING_CAPITAL)/STARTING_CAPITAL*100:.1f}%")
m2.metric("Win Rate", f"{win_rate:.1f}%")
m3.metric("Total Trades", total_trades)

if not trade_history.empty:
    st.dataframe(trade_history.sort_index(ascending=False), use_container_width=True, hide_index=True)
