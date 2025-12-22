import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
from textblob import TextBlob
# Σημείωση: Το Seaborn (heatmap) μπορεί να βαρύνει το web app, το αφαιρώ για ταχύτητα αρχικά.

# ==========================================
# --- CONFIGURATION & SETUP ---
# ==========================================
st.set_page_config(page_title="Warp Speed Terminal", layout="wide", page_icon="🚀")

# Φόρτωση ρυθμίσεων (Users DB)
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Δημιουργία Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ==========================================
# --- LOGIC: LANDING PAGE vs APP ---
# ==========================================

# Εμφάνιση του Login Widget στο Sidebar ή στο κέντρο
name, authentication_status, username = authenticator.login(location='main')

if authentication_status is False:
    st.error('Λάθος όνομα χρήστη ή κωδικός.')
elif authentication_status is None:
    # --- Ο ΧΡΗΣΤΗΣ ΔΕΝ ΕΙΝΑΙ ΣΥΝΔΕΔΕΜΕΝΟΣ: ΔΕΙΞΕ ΤΗ LANDING PAGE ---
    
    # 1. ΤΙΤΛΟΣ
    st.title("🚀 Warp Speed Terminal")
    st.subheader("Το Απόλυτο Σύστημα Χρηματιστηριακής Πληροφόρησης")
    
    # 2. VIDEO
    try:
        # Βεβαιώσου ότι το αρχείο 'promo_video.mp4' είναι στον ίδιο φάκελο
        st.video("https://youtu.be/_YdQeuOZSXY") 
    except:
        st.warning("Το βίντεο δεν βρέθηκε. Ανεβάστε το 'promo_video.mp4'.")

    st.divider()

    # 3. ΠΕΡΙΓΡΑΦΗ (Φορτώνει από το αρχείο description.txt για ευκολία)
    try:
        with open("description.txt", "r", encoding="utf-8") as f:
            desc_text = f.read()
        st.markdown(desc_text)
    except:
        st.write("Η περιγραφή θα εμφανιστεί εδώ.")

    st.divider()

    # 4. PRICING TIERS (ΣΥΝΔΡΟΜΕΣ)
    st.header("🔓 Ξεκλειδώστε το Terminal")
    st.write("Επιλέξτε το πακέτο που σας ταιριάζει και ξεκινήστε αμέσως.")

    # --- ΠΡΟΣΟΧΗ: ΑΝΤΙΚΑΤΑΣΤΗΣΕ ΜΕ ΤΑ ΔΙΚΑ ΣΟΥ STRIPE LINKS ---
    STRIPE_LINKS = {
        "1M": "https://buy.stripe.com/XXXXX_LINK_1_MHNA",
        "3M": "https://buy.stripe.com/XXXXX_LINK_3_MHNES",
        "6M": "https://buy.stripe.com/XXXXX_LINK_6_MHNES",
        "1Y": "https://buy.stripe.com/XXXXX_LINK_1_ETOS",
    }
    # -----------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1 Μήνας")
        st.markdown("# 25€ / μήνα")
        st.write("Μηνιαία χρέωση.")
        st.link_button("Εγγραφή Τώρα", STRIPE_LINKS["1M"], use_container_width=True)

    with col2:
        st.markdown("### 3 Μήνες")
        st.markdown("# 23€ / μήνα")
        st.write("Χρέωση 69€ ανά τρίμηνο.")
        st.link_button("Εγγραφή Τώρα (Save 8%)", STRIPE_LINKS["3M"], use_container_width=True, type="primary")

    with col3:
        st.markdown("### 6 Μήνες")
        st.markdown("# 20€ / μήνα")
        st.write("Χρέωση 120€ ανά εξάμηνο.")
        st.link_button("Εγγραφή Τώρα (Save 20%)", STRIPE_LINKS["6M"], use_container_width=True, type="primary")
        
    with col4:
        st.markdown("### 1 Έτος (Best Value)")
        st.markdown("# 15€ / μήνα")
        st.write("Χρέωση 180€ ετησίως.")
        st.link_button("Εγγραφή Τώρα (Save 40%)", STRIPE_LINKS["1Y"], use_container_width=True, type="primary")

    st.info("Σημείωση: Μετά την πληρωμή, θα λάβετε τα στοιχεία εισόδου σας μέσω email.")


elif authentication_status is True:
    # ==========================================
    # --- Ο ΧΡΗΣΤΗΣ ΕΙΝΑΙ ΣΥΝΔΕΔΕΜΕΝΟΣ: ΔΕΙΞΕ ΤΟ TERMINAL ---
    # ==========================================
    
    # Έλεγχος Status Συνδρομής (Optional για αργότερα)
    user_status = config['credentials']['usernames'][username].get('status', 'active')
    if user_status != 'active':
        st.error("Η συνδρομή σας έχει λήξει. Παρακαλώ ανανεώστε.")
        st.stop()

    # Sidebar Menu
    with st.sidebar:
        st.write(f"Welcome, **{name}**")
        authenticator.logout('Logout', 'sidebar')
        st.divider()

    # --- ΕΔΩ ΞΕΚΙΝΑΕΙ Ο ΚΩΔΙΚΑΣ ΤΟΥ STOCK TERMINAL (v18 Logic) ---
    
    # --- CONSTANTS ---
    PE_THRESHOLD = 35.0
    MA_MULTIPLIER = 1.4
    GHOST_LOOKBACK = 30
    GHOST_PROJECTION = 15
    
    # --- HELPER FUNCTIONS (Same logic as before, adapted for Streamlit) ---
    @st.cache_data(ttl=300) # Cache data for 5 minutes for speed
    def get_stock_data_web(tickers_list):
        data = []
        unique_tickers = list(set([t.strip().upper() for t in tickers_list if t.strip()]))
        if not unique_tickers: return []
        
        try:
            # Fetch Data handles MultiIndex automatically
            hist_data = yf.download(unique_tickers, period="2y", interval="1d", progress=False, auto_adjust=True)
            if hist_data.empty: return []
        except: return []

        for ticker in unique_tickers:
            try:
                # Extract single stock data from yfinance multi-index result
                if len(unique_tickers) > 1:
                    h_close = hist_data['Close'][ticker].dropna()
                    h_high = hist_data['High'][ticker].dropna()
                    h_low = hist_data['Low'][ticker].dropna()
                    h_vol = hist_data['Volume'][ticker].dropna()
                else:
                    h_close = hist_data['Close'].dropna()
                    h_high = hist_data['High'].dropna()
                    h_low = hist_data['Low'].dropna()
                    h_vol = hist_data['Volume'].dropna()

                if len(h_close) < 200: continue
                
                # --- CALCULATIONS (Simplified for Web) ---
                current_price = h_close.iloc[-1]
                prev_close = h_close.iloc[-2]
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                ma50 = h_close.rolling(50).mean().iloc[-1]
                ma200 = h_close.rolling(200).mean().iloc[-1]
                
                # RSI
                delta = h_close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
                avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]

                # RVOL
                avg_vol = h_vol.rolling(50).mean().iloc[-1]
                curr_vol = h_vol.iloc[-1]
                rvol = curr_vol / avg_vol if avg_vol > 0 else 1.0

                # Info & Fundamentals
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                pe = info.get("trailingPE", None)
                peg = info.get("pegRatio", None)
                
                bubble = False
                if pe and pe > PE_THRESHOLD and current_price > ma200 * MA_MULTIPLIER: bubble = True

                # Verdict Logic (Simplified)
                score = 0
                if current_price > ma50: score += 30
                if pe and 0 < pe < 25: score += 20
                if rsi_val < 30: score += 20
                try:
                    if info.get("targetMeanPrice", 0) > current_price: score += 30
                except: pass

                verdict = "HOLD"
                if score >= 80: verdict = "STRONG BUY"
                elif score >= 60: verdict = "BUY"
                elif score <= 30: verdict = "SELL"

                # Sniper Score (Simplified)
                sniper_score = int((score + (rvol*10)) / 1.5)
                if sniper_score > 100: sniper_score = 100

                # Sentiment (Basic)
                sentiment = "NEUTRAL"
                try:
                    news = t_obj.news[:3]
                    sent_score = 0
                    for n in news:
                        blob = TextBlob(n['title'])
                        sent_score += blob.sentiment.polarity
                    if sent_score > 0.1: sentiment = "BULLISH"
                    elif sent_score < -0.1: sentiment = "BEARISH"
                except: news = []

                stock_obj = {
                    "ticker": ticker,
                    "current_price": current_price,
                    "change_pct": change_pct,
                    "verdict": verdict,
                    "sniper_score": sniper_score,
                    "rvol": rvol,
                    "bubble": bubble,
                    "peg": peg,
                    "rsi": rsi_val,
                    "sentiment": sentiment,
                    "hist_close": h_close, # Keep for charts/matrix
                    "info": info,
                    "news": news
                }
                data.append(stock_obj)
            except Exception as e:
                 print(f"Error {ticker}: {e}")
                 continue
        return data

    # --- MAIN APP UI ---
    st.title("🚀 Warp Speed Terminal")
    
    # 1. MACRO BAR (Simplified)
    try:
        macro_tickers = ["^VIX", "^TNX", "BTC-USD", "CL=F"]
        m_data = yf.download(macro_tickers, period="2d", progress=False)['Close']
        m_txt = []
        for t in macro_tickers:
            if len(m_data) >= 2:
                cur = m_data[t].iloc[-1]
                prev = m_data[t].iloc[-2]
                chg = ((cur-prev)/prev)*100
                m_txt.append(f"{t.replace('^','').split('-')[0]}: {cur:.2f} ({chg:+.1f}%)")
        st.caption(" | ".join(m_txt))
    except: st.caption("Macro Data Offline")

    # 2. INPUT & DASHBOARD
    with st.form("scan_form"):
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1:
            tickers_input = st.text_input("Assets > (e.g. AAPL TSLA MSFT)", help="Χωρίστε με κενό ή κόμμα")
        with col_in2:
            submitted = st.form_submit_button("INITIATE SCAN 🔎", type="primary")
    
    if 'stock_data' not in st.session_state: st.session_state['stock_data'] = []

    if submitted and tickers_input:
        with st.spinner('Scanning Markets...'):
            tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
            st.session_state['stock_data'] = get_stock_data_web(tickers_list)

    # 3. RESULTS TABLE
    if st.session_state['stock_data']:
        df_display = pd.DataFrame([
            {
                "Ticker": s['ticker'],
                "Price": f"${s['current_price']:.2f}",
                "Change": f"{s['change_pct']:+.2f}%",
                "VERDICT": s['verdict'],
                "SNIPER": s['sniper_score'],
                "RVOL": f"{s['rvol']:.1f}",
                "Bubble": "🚨 YES" if s['bubble'] else "NO",
                "PEG": f"{s['peg']:.2f}" if s['peg'] else "-",
                "RSI": f"{s['rsi']:.0f}",
                "Sentiment": s['sentiment']
            } for s in st.session_state['stock_data']
        ])
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 4. MATRIX & TOOLS
        st.divider()
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("Correlation Matrix 📊"):
                if len(st.session_state['stock_data']) < 2:
                    st.warning("Χρειάζονται τουλάχιστον 2 μετοχές για το Matrix.")
                else:
                    closes = {s['ticker']: s['hist_close'] for s in st.session_state['stock_data']}
                    df_corr = pd.DataFrame(closes).corr()
                    st.write("### Correlation Matrix")
                    # Χρήση απλού dataframe με gradient χρώματα αντί για seaborn για ταχύτητα
                    st.dataframe(df_corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))

        # 5. DEEP DIVE SELECTION
        st.divider()
        st.header("Deep Dive Analysis")
        selected_ticker = st.selectbox("Επιλέξτε μετοχή για ανάλυση:", [s['ticker'] for s in st.session_state['stock_data']])
        
        if selected_ticker:
            stock = next(s for s in st.session_state['stock_data'] if s['ticker'] == selected_ticker)
            info = stock['info']

            # --- DEEP DIVE TABS ---
            tab_analysis, tab_fund, tab_risk, tab_chart = st.tabs(["Analysis & AI", "Fundamentals", "Risk & Insiders", "Advanced Charts"])

            with tab_analysis:
                col_a1, col_a2 = st.columns([1,2])
                with col_a1:
                    st.metric("Price", f"${stock['current_price']:.2f}", f"{stock['change_pct']:+.2f}%")
                    st.write(f"### VERDICT: {stock['verdict']}")
                    st.write(f"Sniper Score: **{stock['sniper_score']}/100**")
                    if stock['bubble']: st.error("⚠️ BUBBLE SIGNAL DETECTED")
                with col_a2:
                    st.write(f"Sentiment: **{stock['sentiment']}**")
                    st.write("Latest News & AI Analysis:")
                    for n in stock['news']:
                        st.markdown(f"- [{n['title']}]({n['link']})")

            with tab_fund:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.write(":: Valuation ::")
                    def safe_get(d, k): return d.get(k, 'N/A')
                    st.write(f"Market Cap: {safe_get(info, 'marketCap')}")
                    st.write(f"P/E Ratio: {safe_get(info, 'trailingPE')}")
                    st.write(f"PEG Ratio: {safe_get(info, 'pegRatio')}")
                with col_f2:
                    st.write(":: Efficiency & Health ::")
                    st.write(f"ROE: {safe_get(info, 'returnOnEquity')}")
                    st.write(f"Profit Margin: {safe_get(info, 'profitMargins')}")
                    st.write(f"Debt/Equity: {safe_get(info, 'debtToEquity')}")
                    st.write(f"Free Cash Flow: {safe_get(info, 'freeCashflow')}")

            with tab_risk:
                st.write(f"Beta: {safe_get(info, 'beta')}")
                st.write(f"Short Ratio: {safe_get(info, 'shortRatio')}")
                st.write(f"Insiders Held: {safe_get(info, 'heldPercentInsiders')}")

            with tab_chart:
                st.line_chart(stock['hist_close'])
                st.caption("Basic Price Chart (Advanced Charts with Oracle are heavier for web, showing basic version here)")

    else:
        st.info("Εισάγετε tickers παραπάνω και πατήστε SCAN για να ξεκινήσετε.")
