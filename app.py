import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import time

# ==========================================
# --- 1. SETUP & STYLE ---
# ==========================================
st.set_page_config(layout="wide", page_title="Warp Speed Terminal", page_icon="🚀")

st.markdown("""
<style>
    .stApp { background-color: #0b0c10; color: #c5c6c7; }
    h1, h2, h3 { color: #66fcf1 !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetricValue"] { color: #00FFCC; font-family: 'Courier New'; }
    .stButton>button { width: 100%; border: 1px solid #66fcf1; color: #66fcf1; background: #1f2833; }
    .stButton>button:hover { background: #66fcf1; color: black; }
    .error-box { background-color: #330000; color: #ffcccc; padding: 10px; border-left: 5px solid red; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. LOGIC FUNCTIONS ---
# ==========================================

def get_indicators(df):
    try:
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
    except: return df

def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news: return "NEUTRAL"
        score = 0
        count = 0
        for n in news:
            if 'title' in n:
                blob = TextBlob(n['title'])
                score += blob.sentiment.polarity
                count += 1
        if count == 0: return "NEUTRAL"
        avg = score / count
        return "BULLISH" if avg > 0.05 else "BEARISH" if avg < -0.05 else "NEUTRAL"
    except: return "NEUTRAL"

def get_oracle(series):
    # Simplified Oracle for stability
    if len(series) < 60: return None
    try:
        current = series.iloc[-30:].values
        best_score = -1
        best_idx = -1
        # Search simplified
        for i in range(0, len(series)-45, 5):
            candidate = series.iloc[i:i+30].values
            # Simple Correlation
            if len(current) == len(candidate):
                score = np.corrcoef(current, candidate)[0,1]
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        if best_score > 0.5:
            ghost = series.iloc[best_idx:best_idx+45].values
            # Rescale to current price
            return ghost * (series.iloc[-1] / ghost[29])
    except: return None
    return None

# ==========================================
# --- 3. SESSION STATE ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = ""

# ==========================================
# --- 4. APP FLOW ---
# ==========================================

# --- A. LOGIN SCREEN ---
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center;'>WARP SPEED TERMINAL</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.info("Login required to access the Matrix.")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("ENTER SYSTEM"):
            if email == "admin" and password == "PROTOS123":
                st.session_state['logged_in'] = True
                st.session_state['user_email'] = email
                st.rerun()
            else:
                st.error("ACCESS DENIED")
    
    # Try to load image safely
    try: st.image("dashboard.jpg", caption="System Preview", use_container_width=True)
    except: st.warning("dashboard.jpg not found (Upload it to GitHub)")

# --- B. MAIN TERMINAL ---
else:
    # Sidebar
    with st.sidebar:
        st.title("COMMAND")
        st.write(f"User: {st.session_state['user_email']}")
        if st.button("LOGOUT"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.markdown("---")
        st.markdown("Support: warpspeedterminal@gmail.com")

    # Macro Bar (Safe Mode)
    try:
        col1, col2, col3 = st.columns(3)
        # Fetching strictly separately to avoid MultiIndex issues
        btc = yf.Ticker("BTC-USD").history(period="5d")['Close']
        vix = yf.Ticker("^VIX").history(period="5d")['Close']
        
        if not btc.empty: 
            chg = (btc.iloc[-1]-btc.iloc[-2])/btc.iloc[-2]*100
            col1.metric("BITCOIN", f"${btc.iloc[-1]:.0f}", f"{chg:.2f}%")
        
        if not vix.empty: 
            col2.metric("VIX (FEAR)", f"{vix.iloc[-1]:.2f}")
            
    except Exception as e: 
        st.error(f"Macro Data Error: {e}")

    st.divider()

    # --- SCANNER INPUT ---
    with st.form("scan_form"):
        c1, c2 = st.columns([4, 1])
        user_input = c1.text_input("ENTER ASSETS (e.g. AAPL TSLA)", "AAPL")
        submitted = c2.form_submit_button("INITIATE SCAN 🔎")

    # --- SCANNER LOGIC ---
    if submitted:
        tickers = [t.strip().upper() for t in user_input.replace(',', ' ').split() if t.strip()]
        
        if not tickers:
            st.warning("Please type a ticker symbol.")
        else:
            data_rows = []
            progress = st.progress(0)
            
            for i, t in enumerate(tickers):
                try:
                    # 1. Fetch History (Safest Method - NO DOWNLOAD, JUST HISTORY)
                    stock = yf.Ticker(t)
                    df = stock.history(period="1y")
                    
                    # 2. Check Data
                    if df.empty:
                        st.markdown(f"<div class='error-box'>❌ {t}: No data found. Check spelling or delisted.</div>", unsafe_allow_html=True)
                        continue
                    
                    if len(df) < 50:
                        st.markdown(f"<div class='error-box'>⚠️ {t}: Not enough history (<50 days).</div>", unsafe_allow_html=True)
                        continue

                    # 3. Fix Timezone (Crucial for Streamlit)
                    df.index = df.index.tz_localize(None)
                    
                    # 4. Calculate
                    df = get_indicators(df)
                    curr_price = df['Close'].iloc[-1]
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    
                    # 5. Logic
                    ma50 = df['Close'].rolling(50).mean().iloc[-1]
                    verdict = "BUY" if curr_price > ma50 else "SELL"
                    
                    # 6. Info (Handle missing keys safely)
                    info = stock.info
                    pe = info.get('trailingPE', 0)
                    peg = info.get('pegRatio', 0)
                    
                    # 7. Add to List
                    data_rows.append({
                        "TICKER": t,
                        "PRICE": f"{curr_price:.2f}",
                        "VERDICT": verdict,
                        "RSI": f"{rsi:.1f}",
                        "P/E": pe,
                        "PEG": peg,
                        "SENTIMENT": get_sentiment(t),
                        "OBJ": stock, # Store object for deep dive
                        "HIST": df    # Store history
                    })
                    
                except Exception as e:
                    st.markdown(f"<div class='error-box'>⚠️ CRITICAL ERROR scanning {t}: {str(e)}</div>", unsafe_allow_html=True)
                
                progress.progress((i + 1) / len(tickers))
            
            st.session_state['scan_results'] = data_rows
            progress.empty()

    # --- RESULTS DISPLAY ---
    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        results = st.session_state['scan_results']
        
        # Create Display DF (exclude objects)
        display_data = [{k:v for k,v in r.items() if k not in ['OBJ', 'HIST']} for r in results]
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        
        # --- DEEP DIVE ---
        st.markdown("### 🔬 DEEP DIVE")
        # Extract ticker list safely
        ticker_list = [r['TICKER'] for r in results]
        
        if ticker_list:
            selected_ticker = st.selectbox("Select Asset to Analyze", ticker_list)
            
            # Find selected data
            target = next((item for item in results if item["TICKER"] == selected_ticker), None)
            
            if target:
                t1, t2 = st.tabs(["CHART & ORACLE", "FUNDAMENTALS"])
                
                with t1:
                    df = target['HIST']
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    # Price
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                    
                    # Oracle Ghost
                    ghost = get_oracle(df['Close'])
                    if ghost is not None:
                        # Plot ghost starting from 30 days ago
                        ghost_dates = [df.index[-30] + timedelta(days=i) for i in range(45)]
                        fig.add_trace(go.Scatter(x=ghost_dates, y=ghost, line=dict(color='magenta', dash='dash'), name='Oracle'), row=1, col=1)
                    
                    # MACD
                    if 'MACD' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00FFCC'), name='MACD'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='red'), name='Signal'), row=2, col=1)
                    
                    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with t2:
                    stock = target['OBJ']
                    info = stock.info
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                    c1.metric("Beta", info.get('beta', 'N/A'))
                    c2.metric("52W High", info.get('fiftyTwoWeekHigh', 0))
                    c2.metric("52W Low", info.get('fiftyTwoWeekLow', 0))
                    c3.metric("Target Price", info.get('targetMeanPrice', 'N/A'))
                    c3.metric("Recommendation", str(info.get('recommendationKey', 'N/A')).upper())
                    
                    st.write("Institutional Holders:")
                    try: st.dataframe(stock.institutional_holders.head())
                    except: st.write("Data Locked")
