import streamlit as st
import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ==========================================
# --- 1. CONFIGURATION & CUSTOM STYLING ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Custom CSS για να μοιάζει με Web App (SaaS)
st.markdown("""
    <style>
        /* Κρύψε το menu του Streamlit για καθαρό look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Γραμματοσειρές και Επικεφαλίδες */
        h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 800; letter-spacing: -1px; }
        h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
        
        /* Στυλ Κουμπιών */
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: bold;
            height: 3em;
        }
        
        /* Μεγέθυνση αριθμών στα Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. DATABASE MANAGEMENT ---
# ==========================================
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

def login_user_db(email, password):
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
    # BACKDOOR ΓΙΑ ADMIN - ΠΑΝΤΑ ΕΝΕΡΓΟΣ
    if email == "admin": return True
    
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
# --- 3. SESSION & STATE CONTROLLER ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = ""
if 'user_status' not in st.session_state: st.session_state['user_status'] = "expired"
if 'expiry_date' not in st.session_state: st.session_state['expiry_date'] = ""

# Έλεγχος επιστροφής από Stripe
query_params = st.query_params
if "payment_success" in query_params and st.session_state['logged_in']:
    try:
        days_purchased = query_params.get("days", 30)
        new_date = add_subscription_days(st.session_state['user_email'], days_purchased)
        st.session_state['user_status'] = 'active'
        st.session_state['expiry_date'] = new_date
        st.toast(f"VIP ACCESS GRANTED until {new_date}", icon="🚀")
        st.query_params.clear()
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Activation Error: {e}")

# ==========================================
# --- 4. VIEW: LANDING PAGE (SALES MODE) ---
# ==========================================
if not st.session_state['logged_in']:
    
    # HERO SECTION
    st.markdown("""
        <div style='text-align: center; padding: 50px 20px; background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,255,204,0.05) 100%); border-bottom: 1px solid #333;'>
            <h1 style='color: #00FFCC; font-size: 60px; margin-bottom: 10px;'>WARP SPEED TERMINAL</h1>
            <p style='font-size: 24px; color: #aaa;'>Institutional Grade Market Intelligence for Everyone.</p>
            <p style='font-size: 16px; color: #666;'>Real-time Scanning • Technical Analysis • Risk Management</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_main_1, col_main_2 = st.columns([1, 1], gap="large")
    
    with col_main_1:
        st.markdown("### ⚡ UNLEASH THE DATA")
        st.markdown("""
        Stop guessing. Start executing. The Warp Speed Terminal gives you the same tools used by hedge funds:
        
        * **Live Matrix Scanner:** Find movers instantly with institutional verdicts.
        * **Deep Dive Analytics:** Automated Technicals, Fibonacci levels, and Sentiment.
        * **Risk Profiles:** See major holders, insider activity, and short float.
        """)
        
        st.markdown("---")
        st.subheader("🔑 ACCESS TERMINAL")
        
        tab_login, tab_signup = st.tabs(["LOG IN", "REGISTER"])
        
        with tab_login:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type='password', key="login_pass")
            
            # LOGIN BUTTON
            if st.button("LAUNCH TERMINAL", type="primary", width="stretch"):
                # --- BACKDOOR CHECK ---
                if email == "admin" and password == "PROTOS123":
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = "admin"
                    st.session_state['expiry_date'] = "LIFETIME VIP"
                    st.session_state['user_status'] = 'active'
                    st.success("ADMIN OVERRIDE ACTIVATED 🔓")
                    time.sleep(0.5)
                    st.rerun()
                # ----------------------
                else:
                    user_record = login_user_db(email, password)
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
                        st.error("Incorrect credentials.")

        with tab_signup:
            new_email = st.text_input("New Email", key="signup_email")
            new_pass = st.text_input("New Password", type='password', key="signup_pass")
            conf_pass = st.text_input("Confirm Password", type='password', key="signup_conf")
            
            # REGISTER BUTTON
            if st.button("CREATE ACCOUNT", width="stretch"):
                if new_pass == conf_pass and len(new_pass) > 0:
                    if add_user(new_email, new_pass):
                        st.success("Account created! Please log in.")
                    else:
                        st.error("Email already exists.")
                else:
                    st.warning("Passwords do not match.")

    with col_main_2:
        # VIDEO SHOWCASE
        st.video("https://youtu.be/ql1suvTu_ak") 
        st.caption("See the Warp Speed Terminal in action.")

    # SCREENSHOT GALLERY
    st.markdown("<br><br><h2 style='text-align: center; color: #fff;'>PLATFORM PREVIEW</h2><br>", unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("**THE MATRIX SCANNER**")
        try: st.image("dashboard.png", caption="Real-time Multi-Asset Scan & Verdicts", width="stretch")
        except: st.info("[Image: dashboard.png not found]")
        
    with feat_col2:
        st.markdown("**DEEP DIVE ANALYSIS**")
        try: st.image("analysis.png", caption="Automated Technicals, Levels & Sentiment", width="stretch")
        except: st.info("[Image: analysis.png not found]")

    with feat_col3:
        st.markdown("**RISK & INSIDERS**")
        try: st.image("risk_insiders.png", caption="Major Holders & Risk Profile", width="stretch")
        except: st.info("[Image: risk_insiders.png not found]")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #555;'>© 2025 Warp Speed Inc. | Support: warpspeedterminal@gmail.com</p>", unsafe_allow_html=True)


# ==========================================
# --- 5. VIEW: PRICING WALL (Logged in, Expired) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] != 'active':
    
    st.markdown(f"<h2 style='text-align:center'>👋 Welcome back, {st.session_state['user_email']}</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; background-color: #2b1c1c; padding: 10px; border-radius: 10px; border: 1px solid #ff4b4b; margin-bottom: 20px;'>⚠️ YOUR SUBSCRIPTION HAS EXPIRED</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align:center;'>Choose your Plan</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #aaa;'>Unlock full access to the Matrix, Deep Dives, and Sniper alerts.</p><br>", unsafe_allow_html=True)

    STRIPE_LINKS = {
        "1M": "https://buy.stripe.com/00w28l6qUdc96eJ5nYeAg03?days=30",
        "3M": "https://buy.stripe.com/14A9ANaHa8VT46B5nYeAg02?days=90",
        "6M": "https://buy.stripe.com/14A6oB16A7RPfPjg2CeAg01?days=180",
        "1Y": "https://buy.stripe.com/28EaER16A6NL9qV6s2eAg00?days=365",
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
            <h3>STARTER</h3>
            <h1 style='color: #fff;'>€25<span style='font-size: 15px'>/mo</span></h1>
            <p style='color: #666;'>Flexible</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("GET 1 MONTH", STRIPE_LINKS['1M'], width="stretch")

    with col2:
        st.markdown("""
        <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
            <h3>QUARTERLY</h3>
            <h1 style='color: #fff;'>€23<span style='font-size: 15px'>/mo</span></h1>
            <p style='color: #00FFCC; font-weight: bold;'>SAVE €6</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("GET 3 MONTHS", STRIPE_LINKS['3M'], width="stretch")

    with col3:
        st.markdown("""
        <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
            <h3>SEMI-ANNUAL</h3>
            <h1 style='color: #fff;'>€20<span style='font-size: 15px'>/mo</span></h1>
            <p style='color: #00FFCC; font-weight: bold;'>SAVE €30</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("GET 6 MONTHS", STRIPE_LINKS['6M'], width="stretch")

    with col4:
        st.markdown("""
        <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #00FFCC; position: relative;'>
            <div style='position: absolute; top: -10px; left: 50%; transform: translateX(-50%); background: #00FFCC; color: #000; padding: 2px 10px; border-radius: 10px; font-weight: bold; font-size: 12px;'>BEST VALUE</div>
            <h3>ANNUAL</h3>
            <h1 style='color: #fff;'>€15<span style='font-size: 15px'>/mo</span></h1>
            <p style='color: #00FFCC; font-weight: bold;'>SAVE €120</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("GET 1 YEAR", STRIPE_LINKS['1Y'], type="primary", width="stretch")

    st.divider()
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()


# ==========================================
# --- 6. VIEW: THE TERMINAL (LOGGED IN & ACTIVE) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] == 'active':
    
    with st.sidebar:
        st.write(f"PILOT: **{st.session_state['user_email']}**")
        st.caption(f"LICENSE EXPIRY: {st.session_state['expiry_date']}")
        st.success("SYSTEM ONLINE 🟢")
        st.divider()
        st.caption("Support: warpspeedterminal@gmail.com")
        st.divider()
        if st.button("EJECT (LOGOUT)", type="primary"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- STOCK ANALYSIS ENGINE ---
    @st.cache_data(ttl=300)
    def get_stock_data_web(tickers_list):
        data = []
        # Καθαρισμός λίστας
        unique_tickers = list(set([t.strip().upper() for t in tickers_list if t.strip()]))
        if not unique_tickers: return []
        
        try:
            # Download data - Χωρίς threading για αποφυγή rate limits του Yahoo
            hist_data = yf.download(unique_tickers, period="2y", interval="1d", progress=False, auto_adjust=True, threads=False)
            if hist_data.empty: return []
        except Exception as e:
            return []

        for ticker in unique_tickers:
            try:
                # Διαχείριση δεδομένων ανάλογα αν είναι 1 ή πολλά tickers
                if len(unique_tickers) > 1:
                    try:
                        h_close = hist_data['Close'][ticker].dropna()
                    except KeyError:
                        continue
                else:
                    h_close = hist_data['Close'].dropna()
                
                if len(h_close) < 50: continue
                
                # --- Ασφαλής μετατροπή δεδομένων (Fix Float Conversion) ---
                curr_val = h_close.iloc[-1]
                prev_val = h_close.iloc[-2]
                
                current_price = float(curr_val.item()) if hasattr(curr_val, 'item') else float(curr_val)
                prev_close = float(prev_val.item()) if hasattr(prev_val, 'item') else float(prev_val)
                
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                # Υπολογισμοί δεικτών
                ma50 = h_close.rolling(50).mean().iloc[-1]
                if hasattr(ma50, 'item'): ma50 = float(ma50.item())
                
                delta = h_close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
                avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
                
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi_final = rsi_val.iloc[-1]
                if hasattr(rsi_final, 'item'): rsi_final = float(rsi_final.item())
                
                # --- Verdict Logic ---
                t_obj = yf.Ticker(ticker)
                try: info = t_obj.info
                except: info = {}
                
                verdict = "HOLD"
                score = 0
                if current_price > ma50: score += 40
                if rsi_final < 35: score += 30 
                if change_pct > 0 and rsi_final > 50: score += 10
                
                if score >= 60: verdict = "BUY"
                elif score <= 20: verdict = "SELL"
                
                data.append({
                    "ticker": ticker, 
                    "current_price": current_price, 
                    "change_pct": change_pct, 
                    "verdict": verdict, 
                    "rsi": rsi_final, 
                    "hist_close": h_close, 
                    "info": info
                })
            except Exception as e: 
                continue
        return data

    st.title("🚀 WARP SPEED TERMINAL")
    st.markdown("_Institutional Analytics Suite_")

    # Φόρμα εισαγωγής Tickers
    with st.form("scan_form"):
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1: 
            tickers_input = st.text_input("ENTER ASSETS >", help="Γράψε σύμβολα με κενό, π.χ.: AAPL TSLA BTC-USD")
        with col_in2: 
            submitted = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if 'stock_data' not in st.session_state: st.session_state['stock_data'] = []

    # Λογική εκτέλεσης SCAN
    if submitted:
        if not tickers_input:
            st.warning("⚠️ Please enter at least one ticker symbol (e.g. AAPL).")
        else:
            with st.spinner('Connecting to Global Markets...'):
                tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
                scan_results = get_stock_data_web(tickers_list)
                st.session_state['stock_data'] = scan_results

                if not scan_results:
                    st.error(f"❌ No data found for: {tickers_input}")
                    st.info("💡 Try using valid tickers (e.g., 'AAPL', 'NVDA', 'BTC-USD'). Check spelling.")

    # Εμφάνιση αποτελεσμάτων
    if st.session_state['stock_data']:
        st.divider()
        df_display = pd.DataFrame([{ "Ticker": s['ticker'], "Price": f"${s['current_price']:.2f}", "Change": f"{s['change_pct']:+.2f}%", 
                             "VERDICT": s['verdict'], "RSI": f"{s['rsi']:.0f}" } for s in st.session_state['stock_data']])
        
        def color_verdict(val):
            return f'color: {"#00FFCC" if val == "BUY" else "#ff4b4b" if val == "SELL" else "#ffffff"}; font-weight: bold'
        
        st.dataframe(df_display.style.map(color_verdict, subset=['VERDICT']), width="stretch", hide_index=True)

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
                pe = info.get('trailingPE', '-')
                mcap = info.get('marketCap', 0)
                high52 = info.get('fiftyTwoWeekHigh', '-')
                margins = info.get('profitMargins', 0)

                cf1.metric("P/E Ratio", pe)
                cf2.metric("Market Cap", f"${mcap/1e9:.1f}B" if isinstance(mcap, (int, float)) else "-")
                cf3.metric("52W High", high52)
                cf4.metric("Profit Margin", f"{margins*100:.1f}%" if isinstance(margins, (int, float)) else "-")
            
            with tab_news:
                st.write("Latest Headlines:")
                try:
                    t_news = yf.Ticker(selected_ticker).news
                    if t_news:
                        for news in t_news[:3]:
                            st.markdown(f"**{news['title']}**")
                            try:
                                pub_time = datetime.fromtimestamp(news['providerPublishTime']).strftime('%Y-%m-%d')
                                st.caption(f"Published: {pub_time}")
                            except: pass
                    else:
                        st.write("No news found.")
                except: st.write("News feed unavailable.")
    elif not submitted:
        st.info("Enter tickers above (e.g., AAPL TSLA) and press INITIATE SCAN.")
