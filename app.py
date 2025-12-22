import streamlit as st
import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# ==========================================
# --- 1. CONFIGURATION & CUSTOM STYLING ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 800; letter-spacing: -1px; }
        h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
        
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: bold;
            height: 3em;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. LOGIC: ANALYTICS & MATH ---
# ==========================================

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + (std * 2), sma - (std * 2)

def calculate_fibonacci(hist):
    lookback = 126
    if len(hist) < lookback: lookback = len(hist)
    recent = hist.iloc[-lookback:]
    high = recent['High'].max()
    low = recent['Low'].min()
    diff = high - low
    levels = {
        0.236: high - (diff * 0.236),
        0.382: high - (diff * 0.382),
        0.5: high - (diff * 0.5),
        0.618: high - (diff * 0.618)
    }
    return levels

def analyze_sentiment(news_items):
    score = 0
    count = 0
    for item in news_items:
        blob = TextBlob(item['title'])
        score += blob.sentiment.polarity
        count += 1
    if count == 0: return "NEUTRAL"
    avg = score / count
    if avg > 0.05: return "BULLISH"
    if avg < -0.05: return "BEARISH"
    return "NEUTRAL"

def find_similar_pattern(hist_series, lookback_window=30, projection=15):
    # Oracle Ghost Logic
    if len(hist_series) < (lookback_window * 4): return None
    current_pattern = hist_series.iloc[-lookback_window:].values
    
    # Normalize
    c_min, c_max = current_pattern.min(), current_pattern.max()
    if c_max == c_min: return None
    current_norm = (current_pattern - c_min) / (c_max - c_min)
    
    best_score = -1
    best_idx = -1
    search_range = len(hist_series) - lookback_window - projection - 1
    
    for i in range(0, search_range, 5): # Step 5 for speed
        candidate = hist_series.iloc[i : i+lookback_window].values
        if candidate.max() == candidate.min(): continue
        cand_norm = (candidate - candidate.min()) / (candidate.max() - candidate.min())
        try:
            score = np.corrcoef(current_norm, cand_norm)[0, 1]
            if score > best_score:
                best_score = score
                best_idx = i
        except: continue

    if best_score > 0.75: # Threshold
        ghost_data = hist_series.iloc[best_idx : best_idx + lookback_window + projection].copy()
        # Scale ghost to current price
        scale = hist_series.iloc[-1] / ghost_data.iloc[lookback_window-1]
        return ghost_data.iloc[lookback_window:] * scale
    return None

# ==========================================
# --- 3. DATABASE MANAGEMENT ---
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
    if make_hashes(password) == hashed_text: return hashed_text
    return False

def add_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    past_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    current_date = datetime.now().strftime("%Y-%m-%d")
    try:
        c.execute('INSERT INTO users(email, password, status, join_date, expiry_date) VALUES (?,?,?,?,?)', 
                  (email, hashed_pw, 'expired', current_date, past_date))
        conn.commit()
        result = True
    except: result = False
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
    except: return False

init_db()

# ==========================================
# --- 4. SESSION & STATE ---
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
        st.toast(f"VIP ACCESS GRANTED until {new_date}", icon="🚀")
        st.query_params.clear()
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Activation Error: {e}")

# ==========================================
# --- 5. VIEW: LANDING PAGE ---
# ==========================================
if not st.session_state['logged_in']:
    st.markdown("""
        <div style='text-align: center; padding: 50px 20px; background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,255,204,0.05) 100%); border-bottom: 1px solid #333;'>
            <h1 style='color: #00FFCC; font-size: 60px; margin-bottom: 10px;'>WARP SPEED TERMINAL</h1>
            <p style='font-size: 24px; color: #aaa;'>Institutional Grade Market Intelligence.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_main_1, col_main_2 = st.columns([1, 1], gap="large")
    with col_main_1:
        st.markdown("### ⚡ UNLEASH THE DATA")
        st.markdown("""
        **Warp Speed Terminal** is a professional analysis platform that synthesizes Technical Analysis, Fundamental Data, and Artificial Intelligence.
        """)
        st.markdown("---")
        st.subheader("🔑 ACCESS TERMINAL")
        tab_login, tab_signup = st.tabs(["LOG IN", "REGISTER"])
        with tab_login:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type='password', key="login_pass")
            if st.button("LAUNCH TERMINAL", type="primary", width="stretch"):
                if email == "admin" and password == "PROTOS123":
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = "admin"
                    st.session_state['expiry_date'] = "LIFETIME VIP"
                    st.session_state['user_status'] = 'active'
                    st.success("ADMIN OVERRIDE ACTIVATED 🔓")
                    time.sleep(0.5); st.rerun()
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
                    else: st.error("Incorrect credentials.")
        with tab_signup:
            new_email = st.text_input("New Email", key="signup_email")
            new_pass = st.text_input("New Password", type='password', key="signup_pass")
            conf_pass = st.text_input("Confirm Password", type='password', key="signup_conf")
            if st.button("CREATE ACCOUNT", width="stretch"):
                if new_pass == conf_pass and len(new_pass) > 0:
                    if add_user(new_email, new_pass): st.success("Account created! Please log in.")
                    else: st.error("Email already exists.")
                else: st.warning("Passwords do not match.")

    with col_main_2:
        st.video("https://youtu.be/ql1suvTu_ak") 
        
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 READ FULL SYSTEM DESCRIPTION", expanded=True):
        st.markdown("""
        ### 🛡️ Ultimate Stock Market Intelligence System
        
        **1. Central Control Panel (Smart Dashboard)**
        The Investor's Headquarters.
        * **Macro Climate Bar:** Live monitoring of the global market (VIX/Fear Index, 10-Year Bonds, Bitcoin, Oil).
        * **Smart Watchlist & Memory:** The user inputs tickers (e.g., AAPL, NVDA), and the system automatically saves them.
        * **The Evaluation Algorithm:**
            * *Verdict:* A clear command signal (STRONG BUY, BUY, HOLD, SELL).
            * *Sniper Score (/100):* A quantitative scoring of the opportunity.
            * *Bubble Alert:* Detection of overvalued stocks.
            * *RVOL & RSI:* Detection of unusual volume and oversold levels.

        **2. Deep Analysis (Deep Dive View)**
        Double-clicking opens a full "X-ray" tab for the stock:
        * **Analysis & AI Tab:** NLP engine "reads" the news, analyzes sentiment (Bullish/Bearish).
        * **Fundamentals Tab (Enriched):** Valuation metrics (P/E, PEG), ROE, Debt-to-Equity, FCF, Profit Margins (Economic Moat).
        * **Wall Street:** Comparison with analyst forecasts.
        * **Risk Tab:** Volatility analysis (Beta), Short Float, and Institutional Holders.

        **3. Advanced Charting & "The Oracle"**
        * **Oracle Projection:** The algorithm scans historical data and projects a forecast line (Ghost) for the future.
        * **SPY Overlay:** Compares the stock's performance directly against the S&P 500.
        * **Technical Tools:** Bollinger Bands, Fibonacci Levels, MACD, Volume.

        **4. Management & Export Tools**
        * **Correlation Matrix:** Heatmap to check correlations between portfolio stocks.
        * **Data Export:** Instant export of all data to Excel/CSV.
        """)

    st.markdown("<br><h2 style='text-align: center; color: #fff;'>PLATFORM PREVIEW</h2><br>", unsafe_allow_html=True)
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown("**THE MATRIX SCANNER**")
        try: st.image("dashboard.png", caption="Real-time Multi-Asset Scan", width="stretch")
        except: st.info("[Image: dashboard.png not found]")
    with feat_col2:
        st.markdown("**DEEP DIVE ANALYSIS**")
        try: st.image("analysis.png", caption="Automated Technicals", width="stretch")
        except: st.info("[Image: analysis.png not found]")
    with feat_col3:
        st.markdown("**RISK & INSIDERS**")
        try: st.image("risk_insiders.png", caption="Risk Profile", width="stretch")
        except: st.info("[Image: risk_insiders.png not found]")

# ==========================================
# --- 6. VIEW: PRICING WALL (Logged in, Expired) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] != 'active':
    st.markdown(f"<h2 style='text-align:center'>👋 Welcome back, {st.session_state['user_email']}</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; background-color: #2b1c1c; padding: 10px; border-radius: 10px; border: 1px solid #ff4b4b; margin-bottom: 20px;'>⚠️ YOUR SUBSCRIPTION HAS EXPIRED</div>", unsafe_allow_html=True)
    STRIPE_LINKS = {
        "1M": "https://buy.stripe.com/00w28l6qUdc96eJ5nYeAg03?days=30",
        "3M": "https://buy.stripe.com/14A9ANaHa8VT46B5nYeAg02?days=90",
        "6M": "https://buy.stripe.com/14A6oB16A7RPfPjg2CeAg01?days=180",
        "1Y": "https://buy.stripe.com/28EaER16A6NL9qV6s2eAg00?days=365",
    }
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.link_button("GET 1 MONTH (€25)", STRIPE_LINKS['1M'], width="stretch")
    with col2: st.link_button("GET 3 MONTHS (€23/mo)", STRIPE_LINKS['3M'], width="stretch")
    with col3: st.link_button("GET 6 MONTHS (€20/mo)", STRIPE_LINKS['6M'], width="stretch")
    with col4: st.link_button("GET 1 YEAR (€15/mo)", STRIPE_LINKS['1Y'], type="primary", width="stretch")
    st.divider()
    if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

# ==========================================
# --- 7. VIEW: THE TERMINAL (LOGGED IN & ACTIVE) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] == 'active':
    
    with st.sidebar:
        st.write(f"PILOT: **{st.session_state['user_email']}**")
        st.caption(f"EXPIRY: {st.session_state['expiry_date']}")
        st.success("SYSTEM ONLINE 🟢")
        if st.button("EJECT (LOGOUT)", type="primary"): st.session_state['logged_in'] = False; st.rerun()

    # --- MACRO BAR ---
    with st.container():
        try:
            macro_ticks = ["^VIX", "^TNX", "BTC-USD", "CL=F"]
            m_data = yf.download(macro_ticks, period="5d", progress=False)['Close'].iloc[-1]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("VIX (Fear)", f"{m_data['^VIX']:.2f}")
            mc2.metric("10Y Bond", f"{m_data['^TNX']:.2f}")
            mc3.metric("Bitcoin", f"${m_data['BTC-USD']:.0f}")
            mc4.metric("Crude Oil", f"${m_data['CL=F']:.2f}")
        except: st.caption("Macro data unavailable")
    st.divider()

    # --- MAIN ENGINE ---
    @st.cache_data(ttl=300)
    def get_stock_data_web(tickers_list):
        data = []
        unique_tickers = list(set([t.strip().upper() for t in tickers_list if t.strip()]))
        if not unique_tickers: return []
        try:
            hist_data = yf.download(unique_tickers, period="2y", interval="1d", progress=False, auto_adjust=True, threads=False)
            if hist_data.empty: return []
        except: return []

        for ticker in unique_tickers:
            try:
                if len(unique_tickers) > 1:
                    try: h_close = hist_data['Close'][ticker].dropna()
                    except KeyError: continue
                    try: h_high = hist_data['High'][ticker].dropna()
                    except: h_high = h_close
                    try: h_low = hist_data['Low'][ticker].dropna()
                    except: h_low = h_close
                    try: h_vol = hist_data['Volume'][ticker].dropna()
                    except: h_vol = h_close * 0
                else:
                    h_close = hist_data['Close'].dropna()
                    h_high = hist_data['High'].dropna()
                    h_low = hist_data['Low'].dropna()
                    h_vol = hist_data['Volume'].dropna()
                
                if len(h_close) < 50: continue
                
                curr_val = h_close.iloc[-1]
                prev_val = h_close.iloc[-2]
                current_price = float(curr_val.item()) if hasattr(curr_val, 'item') else float(curr_val)
                prev_close = float(prev_val.item()) if hasattr(prev_val, 'item') else float(prev_val)
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                ma50 = h_close.rolling(50).mean().iloc[-1]
                ma200 = h_close.rolling(200).mean().iloc[-1]
                if hasattr(ma50, 'item'): ma50 = float(ma50.item())
                if hasattr(ma200, 'item'): ma200 = float(ma200.item())

                rsi_final = calculate_rsi(h_close).iloc[-1]
                if hasattr(rsi_final, 'item'): rsi_final = float(rsi_final.item())
                
                avg_vol = h_vol.rolling(50).mean().iloc[-1]
                curr_vol = h_vol.iloc[-1]
                rvol = curr_vol / avg_vol if avg_vol > 0 else 1.0

                t_obj = yf.Ticker(ticker)
                try: info = t_obj.info
                except: info = {}
                
                pe = info.get("trailingPE", None)
                bubble = False
                if pe and pe > 35 and current_price > ma200 * 1.4: bubble = True

                verdict = "HOLD"
                score = 0
                if current_price > ma50: score += 40
                if rsi_final < 35: score += 30 
                if change_pct > 0 and rsi_final > 50: score += 10
                if score >= 60: verdict = "BUY"
                elif score <= 20: verdict = "SELL"
                
                # SNIPER SCORE
                sniper_score = 0
                if bubble: sniper_score -= 20
                if rvol > 1.5: sniper_score += 20
                if verdict == "BUY": sniper_score += 40
                if rsi_final > 45 and rsi_final < 65: sniper_score += 20

                # GET NEWS
                news_items = t_obj.news
                
                data.append({
                    "ticker": ticker, 
                    "current_price": current_price, 
                    "change_pct": change_pct, 
                    "verdict": verdict, 
                    "rsi": rsi_final, 
                    "hist_close": h_close,
                    "hist_high": h_high,
                    "hist_low": h_low,
                    "info": info,
                    "rvol": rvol,
                    "bubble": bubble,
                    "sniper": sniper_score,
                    "news": news_items,
                    "yfinance_obj": t_obj
                })
            except Exception as e: continue
        return data

    st.title("🚀 WARP SPEED TERMINAL")
    with st.form("scan_form"):
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1: tickers_input = st.text_input("ENTER ASSETS >", help="e.g. AAPL TSLA BTC-USD")
        with col_in2: submitted = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if 'stock_data' not in st.session_state: st.session_state['stock_data'] = []

    if submitted and tickers_input:
        with st.spinner('Accessing Global Markets...'):
            tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
            st.session_state['stock_data'] = get_stock_data_web(tickers_list)

    if st.session_state['stock_data']:
        st.divider()
        # Main Table
        df_display = pd.DataFrame([{ 
            "Ticker": s['ticker'], 
            "Price": f"${s['current_price']:.2f}", 
            "Change": f"{s['change_pct']:+.2f}%", 
            "VERDICT": s['verdict'], 
            "SNIPER": s['sniper'],
            "RVOL": f"{s['rvol']:.1f}",
            "Bubble": "🚨 YES" if s['bubble'] else "NO",
            "RSI": f"{s['rsi']:.0f}" 
        } for s in st.session_state['stock_data']])
        
        def color_verdict(val):
            return f'color: {"#00FFCC" if val == "BUY" else "#ff4b4b" if val == "SELL" else "#ffffff"}; font-weight: bold'
        st.dataframe(df_display.style.map(color_verdict, subset=['VERDICT']), width="stretch", hide_index=True)

        # BUTTONS FOR EXTRA TOOLS
        c_exp, c_mat = st.columns(2)
        with c_exp:
            if st.button("📥 EXPORT DATA TO CSV", width="stretch"):
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button("DOWNLOAD CSV", csv, "scan_results.csv", "text/csv")
        with c_mat:
            show_matrix = st.checkbox("SHOW CORRELATION MATRIX")

        if show_matrix and len(st.session_state['stock_data']) > 1:
            closes = {s['ticker']: s['hist_close'] for s in st.session_state['stock_data']}
            corr_df = pd.DataFrame(closes).corr()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.divider()
        st.markdown("### 🔬 DEEP DIVE ANALYSIS")
        selected_ticker = st.selectbox("Select Asset:", [s['ticker'] for s in st.session_state['stock_data']])
        
        if selected_ticker:
            stock = next(s for s in st.session_state['stock_data'] if s['ticker'] == selected_ticker)
            info = stock['info']
            
            tab_chart, tab_fund, tab_news, tab_risk = st.tabs(["CHART & ORACLE", "FUNDAMENTALS", "NEWS & AI", "RISK"])
            
            with tab_chart:
                # Custom Matplotlib Chart to match Tkinter logic
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax1.plot(stock['hist_close'].index, stock['hist_close'], label='Price', color='white')
                # BB
                sma = stock['hist_close'].rolling(20).mean()
                std = stock['hist_close'].rolling(20).std()
                ax1.plot(sma.index, sma + 2*std, color='cyan', alpha=0.3)
                ax1.plot(sma.index, sma - 2*std, color='cyan', alpha=0.3)
                # Oracle Ghost
                ghost = find_similar_pattern(stock['hist_close'])
                if ghost is not None:
                    last_date = stock['hist_close'].index[-1]
                    fut_dates = [last_date + timedelta(days=i) for i in range(len(ghost))]
                    ax1.plot(fut_dates, ghost, color='magenta', linestyle='--', label='Oracle Projection')
                
                ax1.set_facecolor('#0e1117')
                ax2.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax1.tick_params(colors='white')
                ax2.tick_params(colors='white')
                
                # MACD
                macd, sig = calculate_macd(stock['hist_close'])
                ax2.plot(macd.index, macd, color='cyan')
                ax2.plot(sig.index, sig, color='red')
                ax1.legend()
                st.pyplot(fig)

            with tab_fund:
                f1, f2, f3 = st.columns(3)
                f1.metric("P/E Ratio", info.get('trailingPE', '-'))
                f1.metric("PEG Ratio", info.get('pegRatio', '-'))
                f2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
                f2.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.1f}%")
                f3.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")
                f3.metric("Debt/Equity", info.get('debtToEquity', '-'))
                st.info(f"Free Cash Flow: ${info.get('freeCashflow', 'N/A')}")
                
            with tab_news:
                st.write("Latest Headlines & AI Sentiment:")
                if stock['news']:
                    for n in stock['news'][:5]:
                        sent = analyze_sentiment([n])
                        color = "green" if sent=="BULLISH" else "red" if sent=="BEARISH" else "grey"
                        st.markdown(f"**[{sent}]** [{n['title']}]({n['link']})")
                else: st.write("No news found.")

            with tab_risk:
                r1, r2 = st.columns(2)
                r1.metric("Beta", info.get('beta', '-'))
                r2.metric("Short Float", f"{info.get('shortRatio', 0)}")
                st.markdown("#### Major Holders")
                try: st.dataframe(stock['yfinance_obj'].institutional_holders.head(), hide_index=True)
                except: st.write("Data not available")

    elif not submitted:
        st.info("Enter tickers above and press INITIATE SCAN.")
