import streamlit as st
import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob

# ==========================================
# --- 1. SYSTEM CONFIGURATION & DATABASE ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --- DATABASE MANAGEMENT (SQLite) ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password TEXT, status TEXT, join_date TEXT, expiry_date TEXT)''')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def add_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    # Default: Expired (Needs payment)
    past_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        c.execute('INSERT INTO users(email, password, status, join_date, expiry_date) VALUES (?,?,?,?,?)', 
                  (email, hashed_pw, 'expired', current_date, past_date))
        conn.commit()
        result = True
    except:
        result = False
    conn.close()
    return result

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    c.execute('SELECT * FROM users WHERE email =? AND password = ?', (email, hashed_pw))
    data = c.fetchall()
    conn.close()
    return data

def add_subscription_days(email, days_to_add):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    new_expiry = (datetime.now() + timedelta(days=int(days_to_add))).strftime("%Y-%m-%d")
    c.execute('UPDATE users SET status = ?, expiry_date = ? WHERE email = ?', ('active', new_expiry, email))
    conn.commit()
    conn.close()
    return new_expiry

def check_subscription_validity(email, current_expiry_str):
    if not current_expiry_str: return False
    try:
        expiry_date = datetime.strptime(current_expiry_str, "%Y-%m-%d")
        if datetime.now() > expiry_date + timedelta(days=1):
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('UPDATE users SET status = ? WHERE email = ?', ('expired', email))
            conn.commit()
            conn.close()
            return False 
        return True
    except:
        return False

init_db()

# ==========================================
# --- 2. SESSION & PAYMENT CONTROLLER ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = ""
if 'user_status' not in st.session_state: st.session_state['user_status'] = "expired"
if 'expiry_date' not in st.session_state: st.session_state['expiry_date'] = ""

query_params = st.query_params
if "payment_success" in query_params and st.session_state['logged_in']:
    try:
        days_purchased = query_params.get("days", 30)
        new_date = add_subscription_days(st.session_state['user_email'], days_purchased)
        st.session_state['user_status'] = 'active'
        st.session_state['expiry_date'] = new_date
        st.toast(f"PAYMENT CONFIRMED! Access granted until {new_date}", icon="✅")
        st.query_params.clear()
    except Exception as e:
        st.error(f"Activation Error: {e}")

# ==========================================
# --- 3. VIEW: LANDING PAGE (Not Logged In) ---
# ==========================================
if not st.session_state['logged_in']:
    
    st.markdown("""
        <h1 style='text-align: center; color: #00FFCC; font-family: "Courier New", monospace; font-size: 50px;'>
        WARP SPEED TERMINAL
        </h1>
        <h3 style='text-align: center; color: #888; letter-spacing: 2px;'>
        INSTITUTIONAL GRADE MARKET INTELLIGENCE
        </h3>
        """, unsafe_allow_html=True)
    
    st.divider()

    # --- 1. ΑΛΛΑΓΗ VIDEO ---
    col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
    with col_v2:
        st.video("https://youtu.be/ql1suvTu_ak") # <--- ΒΑΛΕ ΤΟ YOUTUBE LINK ΕΔΩ
    
    st.divider()

    col_a1, col_a2, col_a3 = st.columns([1, 1, 1])
    with col_a2:
        tab_login, tab_signup = st.tabs(["🔒 MEMBER LOGIN", "📝 NEW ACCOUNT"])

        with tab_login:
            email = st.text_input("Email Address", key="login_email")
            password = st.text_input("Password", type='password', key="login_pass")
            if st.button("ENTER TERMINAL", type="primary", use_container_width=True):
                user_record = login_user(email, password)
                if user_record:
                    email_db = user_record[0][0]
                    expiry_db = user_record[0][4]
                    is_active = check_subscription_validity(email_db, expiry_db)
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = email_db
                    st.session_state['expiry_date'] = expiry_db
                    st.session_state['user_status'] = 'active' if is_active else 'expired'
                    st.rerun()
                else:
                    st.error("Invalid Credentials.")

        with tab_signup:
            new_email = st.text_input("Enter Valid Email", key="signup_email")
            new_pass = st.text_input("Create Password", type='password', key="signup_pass")
            conf_pass = st.text_input("Confirm Password", type='password', key="signup_conf")
            if st.button("REGISTER ACCOUNT", use_container_width=True):
                if new_pass == conf_pass and len(new_pass) > 0:
                    if add_user(new_email, new_pass):
                        st.success("Account created successfully! Please Log In.")
                    else:
                        st.error("This email is already registered.")
                else:
                    st.warning("Passwords do not match or are empty.")

    # Description & Support
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        with open("description.txt", "r") as f: st.info(f.read())
    except: pass
    
    st.markdown("<p style='text-align: center; color: #555;'>Support: warpspeedterminal@gmail.com</p>", unsafe_allow_html=True)


# ==========================================
# --- 4. VIEW: PAYMENT WALL (Logged in, Expired) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] != 'active':
    
    st.markdown(f"<h2 style='text-align:center'>👋 Welcome, {st.session_state['user_email']}</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color: #ff4b4b'>⚠️ SUBSCRIPTION INACTIVE</h3>", unsafe_allow_html=True)
    if st.session_state['expiry_date']:
        st.write(f"<p style='text-align:center'>Your access expired on: <b>{st.session_state['expiry_date']}</b></p>", unsafe_allow_html=True)
    
    st.write("<p style='text-align:center'>Select a membership tier to activate the terminal.</p>", unsafe_allow_html=True)
    st.divider()

    # --- 2. ΑΛΛΑΓΗ STRIPE LINKS ---
    # Βάλε τα Links σου και ΜΗΝ ΣΒΗΣΕΙΣ το ?days=XX στο τέλος
    STRIPE_LINKS = {
        "1M": "https://buy.stripe.com/00w28l6qUdc96eJ5nYeAg03?days=30",   # 1 Μήνας
        "3M": "https://buy.stripe.com/14A9ANaHa8VT46B5nYeAg02?days=90",   # 3 Μήνες
        "6M": "https://buy.stripe.com/14A6oB16A7RPfPjg2CeAg01?days=180",  # 6 Μήνες
        "1Y": "https://buy.stripe.com/28EaER16A6NL9qV6s2eAg00?days=365",  # 1 Έτος
    }
    # -----------------------------

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("#### MONTHLY\n# €25")
        st.link_button("ACTIVATE (30 DAYS)", STRIPE_LINKS['1M'], use_container_width=True)
    with c2:
        st.info("#### QUARTERLY\n# €23 / mo")
        st.link_button("ACTIVATE (90 DAYS)", STRIPE_LINKS['3M'], use_container_width=True)
    with c3:
        st.info("#### SEMI-ANNUAL\n# €20 / mo")
        st.link_button("ACTIVATE (180 DAYS)", STRIPE_LINKS['6M'], use_container_width=True)
    with c4:
        st.info("#### ANNUAL\n# €15 / mo")
        st.link_button("ACTIVATE (365 DAYS)", STRIPE_LINKS['1Y'], use_container_width=True)
    st.markdown("<br><p style='text-align: center; color: grey;'>Support: warpspeedterminal@gmail.com</p>", unsafe_allow_html=True)
    st.divider()
    if st.button("Log Out"):
        st.session_state['logged_in'] = False
        st.rerun()


# ==========================================
# --- 5. VIEW: THE TERMINAL (Logged in, Active) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] == 'active':
    
    with st.sidebar:
        st.write(f"USER: **{st.session_state['user_email']}**")
        st.caption(f"Valid until: {st.session_state['expiry_date']}")
        st.success("STATUS: ONLINE 🟢")
        st.divider()
        st.caption("Support: warpspeedterminal@gmail.com")
        st.divider()
        if st.button("LOGOUT", type="primary"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- STOCK ANALYSIS ENGINE ---
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
                if len(unique_tickers) > 1:
                    if ticker not in hist_data['Close'].columns: continue
                    h_close = hist_data['Close'][ticker].dropna()
                else:
                    h_close = hist_data['Close'].dropna()
                if len(h_close) < 50: continue
                current_price = float(h_close.iloc[-1])
                prev_close = float(h_close.iloc[-2])
                change_pct = ((current_price - prev_close) / prev_close) * 100
                ma50 = h_close.rolling(50).mean().iloc[-1]
                delta = h_close.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
                avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
                rs = avg_gain / avg_loss; rsi_val = 100 - (100 / (1 + rs)); rsi_final = rsi_val.iloc[-1]
                t_obj = yf.Ticker(ticker); info = t_obj.info
                verdict = "HOLD"; score = 0
                if current_price > ma50: score += 40
                if rsi_final < 35: score += 30 
                if change_pct > 0 and rsi_final > 50: score += 10
                if score >= 60: verdict = "BUY"
                elif score <= 20: verdict = "SELL"
                data.append({"ticker": ticker, "current_price": current_price, "change_pct": change_pct, 
                             "verdict": verdict, "rsi": rsi_final, "hist_close": h_close, "info": info})
            except: continue
        return data

    st.title("🚀 WARP SPEED TERMINAL")
    st.markdown("_Institutional Analytics Suite_")

    with st.form("scan_form"):
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1: tickers_input = st.text_input("ENTER ASSETS >", help="e.g. AAPL TSLA NVDA")
        with col_in2: submitted = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if 'stock_data' not in st.session_state: st.session_state['stock_data'] = []

    if submitted and tickers_input:
        with st.spinner('Accessing Global Markets...'):
            tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
            st.session_state['stock_data'] = get_stock_data_web(tickers_list)

    if st.session_state['stock_data']:
        st.divider()
        df_display = pd.DataFrame([{ "Ticker": s['ticker'], "Price": f"${s['current_price']:.2f}", "Change": f"{s['change_pct']:+.2f}%", 
                             "VERDICT": s['verdict'], "RSI": f"{s['rsi']:.0f}" } for s in st.session_state['stock_data']])
        
        def color_verdict(val):
            return f'color: {"#00FFCC" if val == "BUY" else "#ff4b4b" if val == "SELL" else "#ffffff"}; font-weight: bold'
        st.dataframe(df_display.style.map(color_verdict, subset=['VERDICT']), use_container_width=True, hide_index=True)

        st.markdown("### 🔬 DEEP DIVE ANALYSIS")
        selected_ticker = st.selectbox("Select Asset for Inspection:", [s['ticker'] for s in st.session_state['stock_data']])
        if selected_ticker:
            stock = next(s for s in st.session_state['stock_data'] if s['ticker'] == selected_ticker)
            info = stock['info']
            tab_chart, tab_fund, tab_news = st.tabs(["CHART", "FUNDAMENTALS", "NEWS"])
            with tab_chart:
                st.line_chart(stock['hist_close'])
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Current RSI", f"{stock['rsi']:.1f}")
                col_m2.metric("Verdict", stock['verdict'])
            with tab_fund:
                cf1, cf2, cf3, cf4 = st.columns(4)
                cf1.metric("P/E Ratio", info.get('trailingPE', '-'))
                cf2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
                cf3.metric("52W High", info.get('fiftyTwoWeekHigh', '-'))
                cf4.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.1f}%")
            with tab_news:
                st.write("Latest Headlines:")
                try:
                    for news in yf.Ticker(selected_ticker).news[:3]:
                        st.markdown(f"**{news['title']}**")
                        st.caption(f"Published: {datetime.fromtimestamp(news['providerPublishTime']).strftime('%Y-%m-%d')}")
                except: st.write("No news feed available.")
    else:
        st.info("Enter tickers above (e.g., AAPL TSLA) to begin analysis.")
