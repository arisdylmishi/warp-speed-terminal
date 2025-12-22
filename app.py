import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob

# ==========================================
# --- 1. CONFIGURATION & PAGE SETUP ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Load Database
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ==========================================
# --- 2. AUTHENTICATION CONTROLLER ---
# ==========================================

# Check session state for auth status
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# If user is NOT logged in, render the LANDING PAGE
if not st.session_state["authentication_status"]:

    # --- SECTION A: BRANDING ---
    st.markdown("""
        <h1 style='text-align: center; color: #00FFCC; font-size: 60px; font-family: "Courier New", monospace;'>
        WARP SPEED TERMINAL
        </h1>
        <h3 style='text-align: center; color: #888; letter-spacing: 2px;'>
        INSTITUTIONAL GRADE MARKET ANALYTICS
        </h3>
        """, unsafe_allow_html=True)
    
    st.divider()

    # --- SECTION B: VIDEO ---
    col_vid1, col_vid2, col_vid3 = st.columns([1, 2, 1])
    with col_vid2:
        # REPLACE WITH YOUR YOUTUBE LINK
        st.video("https://youtu.be/ql1suvTu_ak") 

    # --- SECTION C: DESCRIPTION ---
    st.markdown("---")
    st.markdown("### 📡 SYSTEM OVERVIEW")
    try:
        with open("description.txt", "r", encoding="utf-8") as f:
            desc_text = f.read()
        st.info(desc_text)
    except:
        st.warning("System description file not found.")

    st.markdown("---")

    # --- SECTION D: LOGIN AREA ---
    col_log1, col_log2, col_log3 = st.columns([1, 1, 1])
    with col_log2:
        st.markdown("### 🔑 CLIENT ACCESS")
        # The login widget renders here
        authenticator.login(location='main')
        
        if st.session_state["authentication_status"] is False:
            st.error('Access Denied: Invalid Credentials')
        elif st.session_state["authentication_status"] is None:
            st.caption("Please enter your secure credentials to proceed.")

    # --- SECTION E: PRICING (THE "POP-UP" STYLE) ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # This acts like a sleek "Pop-down" or modal
    with st.expander("💎 NEW USER? VIEW MEMBERSHIP PLANS", expanded=True):
        st.markdown("<h2 style='text-align: center;'>SELECT YOUR TIER</h2>", unsafe_allow_html=True)
        st.write("")
        
        # --- STRIPE LINKS (REPLACE THESE!) ---
        STRIPE_LINKS = {
            "1M": "https://buy.stripe.com/XXXXX_LINK_1_MONTH",
            "3M": "https://buy.stripe.com/XXXXX_LINK_3_MONTHS",
            "6M": "https://buy.stripe.com/XXXXX_LINK_6_MONTHS",
            "1Y": "https://buy.stripe.com/XXXXX_LINK_1_YEAR",
        }
        
        p1, p2, p3, p4 = st.columns(4)
        
        with p1:
            st.markdown("#### MONTHLY")
            st.markdown("## €25 / mo")
            st.caption("Flexible Access")
            st.link_button("SUBSCRIBE NOW", STRIPE_LINKS["1M"], use_container_width=True)
            
        with p2:
            st.markdown("#### QUARTERLY")
            st.markdown("## €23 / mo")
            st.caption("Billed €69 every 3 months")
            st.link_button("SUBSCRIBE (SAVE 8%)", STRIPE_LINKS["3M"], use_container_width=True, type="primary")
            
        with p3:
            st.markdown("#### SEMI-ANNUAL")
            st.markdown("## €20 / mo")
            st.caption("Billed €120 every 6 months")
            st.link_button("SUBSCRIBE (SAVE 20%)", STRIPE_LINKS["6M"], use_container_width=True, type="primary")

        with p4:
            st.markdown("#### ANNUAL")
            st.markdown("## €15 / mo")
            st.caption("Billed €180 yearly")
            st.link_button("SUBSCRIBE (BEST VALUE)", STRIPE_LINKS["1Y"], use_container_width=True, type="primary")

        st.info("ℹ️ NOTE: After payment, you will receive your secure login credentials via email.")


# ==========================================
# --- 3. MAIN APP (LOGGED IN STATE) ---
# ==========================================
elif st.session_state["authentication_status"]:
    
    # Get User Info
    username = st.session_state["username"]
    name = st.session_state["name"]

    # Sidebar Logout
    with st.sidebar:
        st.markdown(f"User: **{name}**")
        st.markdown(f"Status: <span style='color:#00FFCC'>ACTIVE</span>", unsafe_allow_html=True)
        authenticator.logout(location='sidebar')
        st.divider()

    # --- STOCK LOGIC ENGINE ---
    @st.cache_data(ttl=300)
    def get_stock_data_web(tickers_list):
        data = []
        unique_tickers = list(set([t.strip().upper() for t in tickers_list if t.strip()]))
        if not unique_tickers: return []
        
        try:
            hist_data = yf.download(unique_tickers, period="2y", interval="1d", progress=False, auto_adjust=True)
            if hist_data.empty: return []
        except: return []

        for ticker in unique_tickers:
            try:
                # Handle MultiIndex
                if len(unique_tickers) > 1:
                    if ticker not in hist_data['Close'].columns: continue
                    h_close = hist_data['Close'][ticker].dropna()
                    h_vol = hist_data['Volume'][ticker].dropna()
                else:
                    h_close = hist_data['Close'].dropna()
                    h_vol = hist_data['Volume'].dropna()

                if len(h_close) < 50: continue
                
                # Math Logic
                current_price = float(h_close.iloc[-1])
                prev_close = float(h_close.iloc[-2])
                change_pct = ((current_price - prev_close) / prev_close) * 100
                ma50 = h_close.rolling(50).mean().iloc[-1]
                
                # RSI Calculation
                delta = h_close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
                avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi_final = rsi_val.iloc[-1]

                # Fundamentals
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                
                # Verdict Logic
                verdict = "HOLD"
                score = 0
                if current_price > ma50: score += 40
                if rsi_final < 30: score += 30
                if score >= 60: verdict = "BUY"

                data.append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "change_pct": change_pct,
                    "verdict": verdict,
                    "rsi": rsi_final,
                    "hist_close": h_close,
                    "info": info
                })
            except: continue
        return data

    # --- APP DASHBOARD ---
    st.title("🚀 WARP SPEED TERMINAL")
    st.markdown("_Institutional Analytics Suite v18_")
    
    with st.form("scan_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            tickers_input = st.text_input("ASSETS > (e.g. AAPL TSLA NVDA)", help="Separate with space")
        with col2:
            submitted = st.form_submit_button("INITIATE SCAN 🔎", type="primary")
    
    if 'stock_data' not in st.session_state: st.session_state['stock_data'] = []

    if submitted and tickers_input:
        with st.spinner('Accessing Global Markets...'):
            tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
            st.session_state['stock_data'] = get_stock_data_web(tickers_list)

    if st.session_state['stock_data']:
        st.divider()
        # Results Table
        df_display = pd.DataFrame([
            {
                "Ticker": s['ticker'],
                "Price": f"${s['current_price']:.2f}",
                "Change": f"{s['change_pct']:+.2f}%",
                "VERDICT": s['verdict'],
                "RSI": f"{s['rsi']:.0f}"
            } for s in st.session_state['stock_data']
        ])
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Deep Dive
        st.markdown("### 🔬 DEEP DIVE ANALYSIS")
        selected_ticker = st.selectbox("Select Asset:", [s['ticker'] for s in st.session_state['stock_data']])
        
        if selected_ticker:
            stock = next(s for s in st.session_state['stock_data'] if s['ticker'] == selected_ticker)
            
            # Simple Chart
            st.line_chart(stock['hist_close'])
            
            # Fundamentals Grid
            info = stock['info']
            st.markdown("#### FUNDAMENTALS & HEALTH")
            colf1, colf2, colf3, colf4 = st.columns(4)
            colf1.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            colf2.metric("ROE", info.get('returnOnEquity', 'N/A'))
            colf3.metric("Debt/Equity", info.get('debtToEquity', 'N/A'))
            colf4.metric("Free Cash Flow", info.get('freeCashflow', 'N/A'))
