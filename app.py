import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import yfinance as yf
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Αφαιρέθηκε για ταχύτητα, χρησιμοποιούμε st.line_chart
from textblob import TextBlob

# ==========================================
# --- CONFIGURATION & SETUP ---
# ==========================================
st.set_page_config(page_title="Warp Speed Terminal", layout="wide", page_icon="🚀")

# Φόρτωση ρυθμίσεων (Users DB)
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Δημιουργία Authenticator (Διορθωμένο για νέα έκδοση)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ==========================================
# --- LOGIC: LANDING PAGE vs APP ---
# ==========================================

# Η νέα έκδοση της βιβλιοθήκης δεν επιστρέφει μεταβλητές, αλλά αποθηκεύει στο session state
authenticator.login()

if st.session_state["authentication_status"] is False:
    st.error('Λάθος όνομα χρήστη ή κωδικός.')
elif st.session_state["authentication_status"] is None:
    # --- Ο ΧΡΗΣΤΗΣ ΔΕΝ ΕΙΝΑΙ ΣΥΝΔΕΔΕΜΕΝΟΣ: ΔΕΙΞΕ ΤΗ LANDING PAGE ---
    
    st.title("🚀 Warp Speed Terminal")
    st.subheader("Το Απόλυτο Σύστημα Χρηματιστηριακής Πληροφόρησης")
    
    # 2. VIDEO (Βάλε το δικό σου Link εδώ)
    st.video("https://www.youtube.com/watch?v=_YdQeuOZSXY") 

    st.divider()

    # 3. ΠΕΡΙΓΡΑΦΗ
    try:
        with open("description.txt", "r", encoding="utf-8") as f:
            desc_text = f.read()
        st.markdown(desc_text)
    except:
        st.write("Warp Speed Terminal: Advanced Analytics for Serious Traders.")

    st.divider()

    # 4. PRICING TIERS
    st.header("🔓 Ξεκλειδώστε το Terminal")
    
    # --- ΠΡΟΣΟΧΗ: ΑΝΤΙΚΑΤΑΣΤΗΣΕ ΜΕ ΤΑ ΔΙΚΑ ΣΟΥ STRIPE LINKS ---
    STRIPE_LINKS = {
        "1M": "https://buy.stripe.com/XXXXX_LINK_1_MHNA",
        "3M": "https://buy.stripe.com/XXXXX_LINK_3_MHNES",
        "6M": "https://buy.stripe.com/XXXXX_LINK_6_MHNES",
        "1Y": "https://buy.stripe.com/XXXXX_LINK_1_ETOS",
    }
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1 Μήνας")
        st.markdown("# 25€")
        st.link_button("Εγγραφή", STRIPE_LINKS["1M"], use_container_width=True)

    with col2:
        st.markdown("### 3 Μήνες")
        st.markdown("# 23€ / μήνα")
        st.link_button("Εγγραφή", STRIPE_LINKS["3M"], use_container_width=True, type="primary")

    with col3:
        st.markdown("### 6 Μήνες")
        st.markdown("# 20€ / μήνα")
        st.link_button("Εγγραφή", STRIPE_LINKS["6M"], use_container_width=True, type="primary")
        
    with col4:
        st.markdown("### 1 Έτος")
        st.markdown("# 15€ / μήνα")
        st.link_button("Εγγραφή", STRIPE_LINKS["1Y"], use_container_width=True, type="primary")


elif st.session_state["authentication_status"]:
    # ==========================================
    # --- Ο ΧΡΗΣΤΗΣ ΕΙΝΑΙ ΣΥΝΔΕΔΕΜΕΝΟΣ ---
    # ==========================================
    
    # Ανάκτηση ονόματος χρήστη
    username = st.session_state["username"]
    name = st.session_state["name"]

    # Logout Button (Sidebar)
    with st.sidebar:
        st.write(f"Welcome, **{name}**")
        authenticator.logout()
        st.divider()

    # --- STOCK LOGIC ---
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
                # Χειρισμός MultiIndex του yfinance
                if len(unique_tickers) > 1:
                    if ticker not in hist_data['Close'].columns: continue
                    h_close = hist_data['Close'][ticker].dropna()
                    h_vol = hist_data['Volume'][ticker].dropna()
                else:
                    h_close = hist_data['Close'].dropna()
                    h_vol = hist_data['Volume'].dropna()

                if len(h_close) < 50: continue
                
                # Calculations
                current_price = float(h_close.iloc[-1])
                prev_close = float(h_close.iloc[-2])
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                ma50 = h_close.rolling(50).mean().iloc[-1]
                
                # RSI
                delta = h_close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
                avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi_final = rsi_val.iloc[-1]

                # Info
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                
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

    st.title("🚀 Warp Speed Terminal")
    
    with st.form("scan_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            tickers_input = st.text_input("Assets > (e.g. AAPL TSLA)", help="Χωρίστε με κενό")
        with col2:
            submitted = st.form_submit_button("SCAN 🔎", type="primary")
    
    if 'stock_data' not in st.session_state: st.session_state['stock_data'] = []

    if submitted and tickers_input:
        with st.spinner('Scanning...'):
            tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
            st.session_state['stock_data'] = get_stock_data_web(tickers_list)

    if st.session_state['stock_data']:
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

        selected_ticker = st.selectbox("Ανάλυση:", [s['ticker'] for s in st.session_state['stock_data']])
        if selected_ticker:
            stock = next(s for s in st.session_state['stock_data'] if s['ticker'] == selected_ticker)
            st.line_chart(stock['hist_close'])
            
            # Fundamentals Tab
            info = stock['info']
            st.write("### Fundamentals")
            colf1, colf2, colf3 = st.columns(3)
            colf1.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            colf2.metric("ROE", info.get('returnOnEquity', 'N/A'))
            colf3.metric("Debt/Equity", info.get('debtToEquity', 'N/A'))
