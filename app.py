import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
# LIST OF ASSETS TO TRADE (Make sure these match your Market Watch exactly!)
SYMBOLS = ["USDJPY","BTCUSD", "XAUUSD", "EURUSD", "DJ30", "ETHUSD"]

TIMEFRAME = mt5.TIMEFRAME_H1  # 1 Hour Timeframe
VOLUME = 0.01                 # Lot size (Adjust for your risk!)
CONFIDENCE_THRESHOLD = 0.65   # AI Confidence
RR_RATIO = 2.0                # Risk:Reward 1:2
DEVIATION = 20                

print(f"--- 🤖 INITIALIZING MULTI-ASSET AI BOT ---")
print(f"Tracking: {SYMBOLS}")

# ==========================================
# 2. CONNECT TO MT5
# ==========================================
if not mt5.initialize():
    print("❌ MT5 Initialization Failed")
    quit()

print(f"✅ Connected to Account: {mt5.account_info().login}")

# ==========================================
# 3. DATA & AI ENGINE
# ==========================================
def get_data(symbol, n=1000):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Indicators
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['EMA200'] = ta.ema(df['close'], length=200)
    
    df.dropna(inplace=True)
    return df

def train_model(symbol):
    print(f"🧠 Training Brain for {symbol}...", end="\r")
    df = get_data(symbol, 2000)
    if df.empty:
        print(f"❌ Could not fetch data for {symbol}")
        return None
        
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['RSI', 'MACD', 'ATR', 'EMA50', 'EMA200']
    
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(df[features].iloc[:-1], df['Target'].iloc[:-1])
    return model

# Train Models for ALL Symbols on Startup
models = {}
for sym in SYMBOLS:
    m = train_model(sym)
    if m: models[sym] = m

print("\n✅ All Models Trained. Starting Watch Loop...")

# ==========================================
# 4. TRADING FUNCTION
# ==========================================
def execute_trade(symbol, action, atr):
    # Check open positions to prevent double-entry
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        return # Already in a trade
        
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return
    
    type_trade = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if action == "BUY" else tick.bid
    sl = price - atr if action == "BUY" else price + atr
    tp = price + (atr * RR_RATIO) if action == "BUY" else price - (atr * RR_RATIO)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": VOLUME,
        "type": type_trade,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": 123456,
        "comment": "Multi-Asset AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"🚀 {symbol} {action} EXECUTED! Ticket: {res.order}")
    else:
        print(f"❌ {symbol} Exec Failed: {res.comment}")

# ==========================================
# 5. LIVE LOOP
# ==========================================
features = ['RSI', 'MACD', 'ATR', 'EMA50', 'EMA200']

while True:
    print("\n--- 🔍 SCANNING MARKETS ---")
    
    for sym in SYMBOLS:
        if sym not in models: continue
        
        # 1. Get Live Data
        df = get_data(sym, 100)
        if df.empty: continue
        
        # 2. Predict
        last_candle = df[features].iloc[[-1]]
        probs = models[sym].predict_proba(last_candle)[0]
        buy_conf, sell_conf = probs[1], probs[0]
        
        atr = df['ATR'].iloc[-1]
        
        # 3. Decision
        status = "NEUTRAL"
        if buy_conf > CONFIDENCE_THRESHOLD:
            status = "BUY SIGNAL 🟢"
            execute_trade(sym, "BUY", atr)
        elif sell_conf > CONFIDENCE_THRESHOLD:
            status = "SELL SIGNAL 🔴"
            execute_trade(sym, "SELL", atr)
            
        print(f"{sym:<8} | Bull: {buy_conf*100:.1f}% | Bear: {sell_conf*100:.1f}% | {status}")
        
    time.sleep(10) # Wait 10s before re-scanning
