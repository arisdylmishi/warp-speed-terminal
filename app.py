import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import time

# ==========================================
# --- 1. CONFIGURATION & STYLE ---
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
    .error-box { background-color: #330000; color: #ffcccc; padding: 10px; border-left: 5px solid red; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. DATA FUNCTIONS ---
# ==========================================

@st.cache_data(ttl=600)  # Cache data for 10 mins
def fetch_stock_data(ticker):
    """
    Fetches data for a single ticker using .history() method.
    Returns (stock_object, dataframe, info_dict)
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 year of history
        df = stock.history(period="1y")
        
        if df.empty:
            return None, None, None
        
        # Clean up timezone info to avoid Streamlit errors
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return stock, df, stock.info
    except Exception as e:
        return None, None, str(e)

def calculate_technical_indicators(df):
    if df is None or df.empty: return df
    try:
        # Create copies to avoid SettingWithCopy warnings
        df = df.copy()
        
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

def get_oracle_projection(series):
    # Simplified Oracle for stability
    if len(series) < 60: return None
    try:
        current = series.iloc[-30:].values
        best_score = -1
        best_idx = -1
        
        # Search simplified
        for i in range(0, len(series)-60, 5):
            candidate = series.iloc[i:i+30].values
            # Simple Correlation check
            if len(current) == len(candidate):
                score = np.corrcoef(current, candidate)[0,1]
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        if best_score > 0.5:
            ghost = series.iloc[best_idx:best_idx+45].values
            # Rescale to current price
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
            if title:
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
# --- 3. SESSION STATE ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'scan_results' not in st.session_state: st.session_state['scan_results'] = []

# ==========================================
# --- 4. APP FLOW ---
# ==========================================

# ---> LOGIN SCREEN
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center;'>WARP SPEED TERMINAL</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.info("Login required to access the Matrix.")
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
        st.markdown("---")
        st.markdown("📧 **Support:**\nwarpspeedterminal@gmail.com")

    # Macro Bar (Safe Mode with fallback)
    try:
        col1, col2, col3 = st.columns(3)
        # Fetching strictly separately to avoid MultiIndex issues
        btc_obj, btc_df, _ = fetch_stock_data("BTC-USD")
        vix_obj, vix_df, _ = fetch_stock_data("^VIX")
        
        if btc_df is not None and not btc_df.empty: 
            p_now = btc_df['Close'].iloc[-1]
            p_prev = btc_df['Close'].iloc[-2]
            chg = (p_now - p_prev) / p_prev * 100
            col1.metric("BITCOIN", f"${p_now:,.0f}", f"{chg:.2f}%")
        else:
            col1.metric("BITCOIN", "N/A", "0.00%")
        
        if vix_df is not None and not vix_df.empty: 
            col2.metric("VIX (FEAR)", f"{vix_df['Close'].iloc[-1]:.2f}")
        else:
            col2.metric("VIX (FEAR)", "N/A")
            
    except Exception as e: 
        st.error(f"Macro Data Error: {e}")

    st.divider()

    # --- SCANNER INPUT ---
    with st.form("scanner_form"):
        c1, c2 = st.columns([4,1])
        txt_input = c1.text_input("ENTER TICKERS (Space separated)", "AAPL TSLA NVDA")
        scan_btn = c2.form_submit_button("INITIATE SCAN 🔎")

    # --- SCANNER LOGIC ---
    if scan_btn:
        tickers = [t.strip().upper() for t in txt_input.replace(",", " ").split() if t.strip()]
        
        if not tickers:
            st.warning("Please enter a symbol.")
        else:
            results_data = []
            
            # Progress Bar
            bar = st.progress(0, text="Scanning Market...")
            
            for i, t in enumerate(tickers):
                # 1. GET DATA (The Simple Way)
                obj, df, info = fetch_stock_data(t)
                
                # ERROR HANDLING: Show error but continue
                if df is None:
                    st.markdown(f"<div class='error-box'>❌ Could not load <b>{t}</b>. Check spelling or try again later. (Error: {info})</div>", unsafe_allow_html=True)
                    time.sleep(0.5) # Small delay to be gentle on API
                    bar.progress((i+1)/len(tickers))
                    continue
                
                if df.empty:
                    st.markdown(f"<div class='error-box'>⚠️ Data empty for <b>{t}</b>. Market might be closed or ticker delisted.</div>", unsafe_allow_html=True)
                    bar.progress((i+1)/len(tickers))
                    continue

                # 2. CALCS
                try:
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
                    
                    # Info Handling
                    if info is None: info = {} # Fallback if info fetch failed
                    
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
                except Exception as e:
                    st.error(f"Error calculating indicators for {t}: {str(e)}")
                
                bar.progress((i+1)/len(tickers))
            
            bar.empty()
            st.session_state['scan_results'] = results_data

    # --- DISPLAY RESULTS ---
    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        data = st.session_state['scan_results']
        
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

        if simple_data:
            st.dataframe(pd.DataFrame(simple_data).style.map(color_verdict, subset=['VERDICT']), use_container_width=True)
        else:
            st.warning("No valid data to display.")
        
        # 2. DEEP DIVE
        st.markdown("### 🔬 DEEP DIVE")
        opts = [d['TICKER'] for d in data]
        
        if opts:
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
                        # Plot ghost starting from 30 days ago
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
                    
                    st.write("**Financial Summary:**")
                    # Safe way to display dictionary without error
                    clean_info = {k: v for k, v in info.items() if v is not None and not isinstance(v, dict)}
                    st.json(clean_info)
