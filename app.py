import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
st.set_page_config(layout="wide", page_title="Warp Speed Terminal", page_icon="🚀")

st.markdown("""
<style>
    .stApp { background-color: #0b0c10; color: #c5c6c7; }
    h1, h2, h3 { color: #66fcf1 !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetricValue"] { color: #00FFCC; font-family: 'Courier New'; }
    .stButton>button { width: 100%; border: 1px solid #66fcf1; color: #66fcf1; background: #1f2833; }
    .stButton>button:hover { background: #66fcf1; color: black; }
    .reason-box { background-color: #111; padding: 10px; border-left: 3px solid #00FFCC; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. HELPER FUNCTIONS ---
# ==========================================

def get_data_simple(ticker):
    """
    Η απλή μέθοδος που πέτυχε στο TEST.
    """
    try:
        stock = yf.Ticker(ticker)
        # Παίρνουμε 1 χρόνο δεδομένα
        df = stock.history(period="1y")
        
        # Καθαρισμός Timezone για να μην σκάει το Streamlit
        df.index = df.index.tz_localize(None)
        
        return stock, df
    except Exception as e:
        return None, None

def calculate_technical_indicators(df):
    if df is None or df.empty: return df
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['SMA20'] + (df['STD20'] * 2)
    df['LowerBB'] = df['SMA20'] - (df['STD20'] * 2)
    
    return df

def get_oracle_projection(series):
    # Απλοποιημένος Oracle αλγόριθμος
    if len(series) < 60: return None
    try:
        current_pattern = series.iloc[-30:].values
        best_score = -1
        best_idx = -1
        
        # Σαρώνουμε το ιστορικό
        for i in range(0, len(series)-60, 5):
            candidate = series.iloc[i:i+30].values
            score = np.corrcoef(current_pattern, candidate)[0,1]
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Αν βρεθεί καλό pattern (>50%)
        if best_score > 0.5:
            ghost = series.iloc[best_idx:best_idx+45].values
            # Προσαρμογή στην τωρινή τιμή
            scale = series.iloc[-1] / ghost[29]
            return ghost * scale
            
    except: return None
    return None

def analyze_news_sentiment(stock_obj):
    try:
        news = stock_obj.news
        if not news: return "NEUTRAL"
        score = 0
        count = 0
        for n in news:
            title = n.get('title', '')
            blob = TextBlob(title)
            score += blob.sentiment.polarity
            count += 1
        
        if count == 0: return "NEUTRAL"
        avg = score / count
        if avg > 0.05: return "BULLISH"
        elif avg < -0.05: return "BEARISH"
        else: return "NEUTRAL"
    except: return "NEUTRAL"

# ==========================================
# --- 3. SESSION ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

# ==========================================
# --- 4. APP LOGIC ---
# ==========================================

# ---> LOGIN SCREEN
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center;'>WARP SPEED TERMINAL</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        email = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("ENTER"):
            if email == "admin" and password == "PROTOS123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("WRONG CREDENTIALS")
    
    try: st.image("dashboard.jpg", use_container_width=True)
    except: pass

# ---> MAIN DASHBOARD
else:
    # Sidebar
    with st.sidebar:
        st.title("COMMAND")
        if st.button("LOGOUT"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.info("System Online ✅")

    # Macro Bar
    try:
        c1, c2, c3 = st.columns(3)
        btc = yf.Ticker("BTC-USD").history(period="2d")
        if not btc.empty:
            p_now = btc['Close'].iloc[-1]
            p_prev = btc['Close'].iloc[-2]
            chg = (p_now - p_prev) / p_prev * 100
            c1.metric("BITCOIN", f"${p_now:,.0f}", f"{chg:.2f}%")
        
        vix = yf.Ticker("^VIX").history(period="1d")
        if not vix.empty:
            c2.metric("VIX (Fear)", f"{vix['Close'].iloc[-1]:.2f}")
    except: pass

    st.divider()

    # --- SCANNER ---
    with st.form("scanner_form"):
        c1, c2 = st.columns([4,1])
        txt_input = c1.text_input("ENTER TICKERS (Space separated)", "AAPL TSLA NVDA")
        scan_btn = c2.form_submit_button("INITIATE SCAN 🔎")

    if scan_btn:
        tickers = [t.upper() for t in txt_input.replace(",", " ").split() if t]
        
        if not tickers:
            st.warning("Please enter a symbol.")
        else:
            results_data = []
            
            # Progress Bar
            bar = st.progress(0, text="Scanning Market...")
            
            for i, t in enumerate(tickers):
                # 1. GET DATA (The Simple Way)
                obj, df = get_data_simple(t)
                
                if df is None or df.empty:
                    st.toast(f"❌ Could not load {t}")
                    continue
                
                # 2. CALCS
                df = calculate_technical_indicators(df)
                
                price = df['Close'].iloc[-1]
                
                # Verdict Logic
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                
                verdict = "HOLD"
                reasons = []
                
                if price > ma50:
                    reasons.append("Trend is Bullish (>50MA)")
                    if rsi < 70: verdict = "BUY"
                else:
                    reasons.append("Trend is Bearish (<50MA)")
                    if rsi > 70: verdict = "SELL"
                
                if rsi < 30: 
                    verdict = "STRONG BUY"
                    reasons.append("RSI Oversold (Bounce likely)")
                
                # Info
                try: info = obj.info
                except: info = {}
                
                # Append Results
                results_data.append({
                    "TICKER": t,
                    "PRICE": price,
                    "VERDICT": verdict,
                    "RSI": rsi,
                    "P/E": info.get('trailingPE', 0),
                    "SENTIMENT": analyze_news_sentiment(obj),
                    "REASONS": reasons,
                    "DF": df,   # Keep dataframe for charts
                    "OBJ": obj,  # Keep object for deep info
                    "INFO": info
                })
                
                bar.progress((i+1)/len(tickers))
            
            bar.empty()
            st.session_state['results'] = results_data

    # --- DISPLAY RESULTS ---
    if 'results' in st.session_state and st.session_state['results']:
        data = st.session_state['results']
        
        # 1. TABLE
        simple_data = []
        for d in data:
            simple_data.append({
                "TICKER": d['TICKER'],
                "PRICE": f"${d['PRICE']:.2f}",
                "VERDICT": d['VERDICT'],
                "RSI": f"{d['RSI']:.1f}",
                "SENTIMENT": d['SENTIMENT']
            })
        
        def color_verdict(val):
            color = '#00FFCC' if 'BUY' in val else '#ff4444' if 'SELL' in val else 'white'
            return f'color: {color}; font-weight: bold'

        st.dataframe(pd.DataFrame(simple_data).style.map(color_verdict, subset=['VERDICT']), use_container_width=True)
        
        # 2. DEEP DIVE
        st.markdown("### 🔬 DEEP DIVE")
        opts = [d['TICKER'] for d in data]
        sel = st.selectbox("Select Asset", opts)
        
        # Get selected data
        target = next((item for item in data if item["TICKER"] == sel), None)
        
        if target:
            t1, t2 = st.tabs(["CHART", "DATA"])
            
            with t1:
                df = target['DF']
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                
                # Candles
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                
                # Oracle
                ghost = get_oracle_projection(df['Close'])
                if ghost is not None:
                    dates = [df.index[-30] + timedelta(days=i) for i in range(45)]
                    fig.add_trace(go.Scatter(x=dates, y=ghost, line=dict(color='magenta', dash='dash'), name='Oracle'), row=1, col=1)
                
                # MACD
                if 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00FFCC'), name='MACD'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='red'), name='Signal'), row=2, col=1)
                
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Logic Box
                st.markdown("#### 🧠 LOGIC:")
                for r in target['REASONS']:
                    st.markdown(f"<div class='reason-box'>{r}</div>", unsafe_allow_html=True)

            with t2:
                info = target['INFO']
                c1, c2 = st.columns(2)
                c1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                c2.metric("Target Price", f"${info.get('targetMeanPrice', 0)}")
                st.json(info.get('financials', {})) # Raw data dump if specific keys missing
