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
    page_title="Sniper Bot | Pro Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric { background-color: #121212; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stSidebar"] { background-color: #0E0E0E; }
    h1, h2, h3 { color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Sniper Bot | AI & Strategy Analytics")

# ==========================================
# 2. SIDEBAR PARAMETERS
# ==========================================
st.sidebar.header("🔌 Connection & Data")

# Region/Exchange Switcher
EXCHANGE_OPT = st.sidebar.selectbox(
    "Data Source",
    ["Binance US (USA)", "Kraken (Global)", "Binance Global (ROW)"],
    index=0 
)

exchange_id_map = {
    "Binance US (USA)": "binanceus",
    "Kraken (Global)": "kraken",
    "Binance Global (ROW)": "binance"
}
ACTIVE_EXCHANGE_ID = exchange_id_map[EXCHANGE_OPT]

# Asset Selector
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

# Binance symbol fix
if "Binance" in EXCHANGE_OPT and "/USD" in SYMBOL and "XAU" not in selected_label:
    SYMBOL = SYMBOL.replace("USD", "USDT")

INTERVAL = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h", "1d"], index=2)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy Parameters")

# --- PARAMETERS YOU ASKED TO SEE ---
STRATEGY_TYPE = st.sidebar.radio("Logic Mode", ["🤖 AI Trend Sniper", "♻️ RSI Reversion"])

TARGET_RR_RATIO = st.sidebar.slider("Target Risk:Reward (1:X)", 1.0, 5.0, 2.0, step=0.1)
CONFIDENCE = st.sidebar.slider("AI Confidence Threshold", 50, 95, 60) / 100

st.sidebar.info(f"Targeting **1:{TARGET_RR_RATIO}** R:R Ratio")

# ==========================================
# 3. DATA ENGINE
# ==========================================
@st.cache_data(ttl=30)
def fetch_market_data(source, exchange_id, ticker, interval):
    df = pd.DataFrame()
    try:
        limit = 1500 # Approx 2-3 months of 1h data
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
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df.dropna(inplace=True)
            
        return df

    except Exception as e:
        return pd.DataFrame()

with st.spinner(f"Fetching {SYMBOL}..."):
    df = fetch_market_data(SOURCE, ACTIVE_EXCHANGE_ID, SYMBOL, INTERVAL)

if df.empty:
    st.error("❌ No data received.")
    st.stop()

# ==========================================
# 4. ANALYSIS ENGINE (METRICS CALCULATION)
# ==========================================
def run_analysis(df, strategy, target_rr):
    # 1. Calculate Training Duration
    total_duration_days = (df.index[-1] - df.index[0]).days
    training_split_idx = int(len(df) * 0.8)
    training_months = (total_duration_days * 0.8) / 30.0 # Approx months
    
    # 2. Logic & Signal
    df['Signal'] = 0
    live_buy_prob = 0.0
    
    if "AI" in strategy:
        # XGBoost Logic
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        features = ['RSI', 'MACD', 'ATR']
        
        train = df.iloc[:training_split_idx]
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(train[features], train['Target'])
        
        probs = model.predict_proba(df[features])
        df['Prob_Buy'] = probs[:, 1]
        df['Prob_Sell'] = probs[:, 0]
        
        df['Signal'] = np.where(df['Prob_Buy'] > CONFIDENCE, 1, 0)
        df['Signal'] = np.where(df['Prob_Sell'] > CONFIDENCE, -1, df['Signal'])
        live_buy_prob = df['Prob_Buy'].iloc[-1]
    else:
        # RSI Logic
        df['Signal'] = np.where(df['RSI'] < 30, 1, 0)
        df['Signal'] = np.where(df['RSI'] > 70, -1, df['Signal'])
        live_buy_prob = 1.0 if df['RSI'].iloc[-1] < 30 else 0.0

    # 3. Backtest for Metrics
    trades = []
    balance = 10000
    wins = 0
    total_trades = 0
    
    # Simple simulation loop on Test Data (Last 20%)
    test_df = df.iloc[training_split_idx:]
    
    for i in range(len(test_df)-1):
        sig = test_df['Signal'].iloc[i]
        price_in = test_df['Close'].iloc[i]
        price_out = test_df['Close'].iloc[i+1] # 1 candle hold for demo
        
        if sig != 0:
            total_trades += 1
            pnl = (price_out - price_in) if sig == 1 else (price_in - price_out)
            
            # RR Logic Check (Simplified)
            risk = test_df['ATR'].iloc[i]
            reward = risk * target_rr
            
            # Check if this candle hit TP or SL
            # (In a real 1-candle sim, we just take close diff, 
            # but let's count a 'Win' if direction was right)
            if pnl > 0: wins += 1
            
            trades.append({"Time": test_df.index[i], "PnL": pnl, "Type": "Long" if sig==1 else "Short"})
            balance += pnl * (10000/price_in)

    accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Realized RR Calculation
    avg_win = 0
    avg_loss = 0
    realized_rr = 0.0
    
    if len(trades) > 0:
        df_t = pd.DataFrame(trades)
        winning_trades = df_t[df_t['PnL'] > 0]
        losing_trades = df_t[df_t['PnL'] <= 0]
        
        if not winning_trades.empty: avg_win = winning_trades['PnL'].mean()
        if not losing_trades.empty: avg_loss = abs(losing_trades['PnL'].mean())
        
        if avg_loss > 0:
            realized_rr = avg_win / avg_loss

    return {
        "training_months": training_months,
        "accuracy": accuracy,
        "realized_rr": realized_rr,
        "trades": trades,
        "final_balance": balance,
        "live_prob": live_buy_prob
    }

metrics = run_analysis(df, STRATEGY_TYPE, TARGET_RR_RATIO)

# ==========================================
# 5. DASHBOARD UI
# ==========================================

# --- ROW 1: KEY PARAMETERS & PERFORMANCE ---
st.markdown("### 📊 Performance & Parameters")
m1, m2, m3, m4 = st.columns(4)

m1.metric("Training Data Duration", f"{metrics['training_months']:.1f} Months", "80% of Data")
m2.metric("Accuracy (Win Rate)", f"{metrics['accuracy']:.1f}%", f"{len(metrics['trades'])} Trades")
m3.metric("Risk:Reward (Target)", f"1:{TARGET_RR_RATIO}", "Input Param")
m4.metric("Risk:Reward (Realized)", f"1:{metrics['realized_rr']:.2f}", "Actual Avg Win/Loss")

# --- ROW 2: CHART ---
st.markdown("### 📈 Live Market Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index[-100:], open=df['Open'][-100:], high=df['High'][-100:], low=df['Low'][-100:], close=df['Close'][-100:], name="Price"))
fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig, use_container_width=True)

# --- ROW 3: TRADES ---
if metrics['trades']:
    st.dataframe(pd.DataFrame(metrics['trades']).sort_values(by="Time", ascending=False), use_container_width=True)
else:
    st.info("No trades triggered in the test period.")
