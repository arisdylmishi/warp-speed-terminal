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
# --- 1. CONFIGURATION & STYLE ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Dark Mode Professional UI
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
        
        /* Metric Box Styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
            color: #00FFCC;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. ADVANCED LOGIC (AI, ORACLE, MATH) ---
# ==========================================

def calculate_indicators(hist):
    # RSI
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
    hist['STD20'] = hist['Close'].rolling(window=20).std()
    hist['UpperBB'] = hist['SMA20'] + (hist['STD20'] * 2)
    hist['LowerBB'] = hist['SMA20'] - (hist['STD20'] * 2)
    
    return hist

def analyze_sentiment(news_items):
    score = 0
    count = 0
    for item in news_items:
        # Safe get for title
        title = item.get('title', '')
        if not title: continue
        
        try:
            blob = TextBlob(title)
            score += blob.sentiment.polarity
            count += 1
        except: continue
        
    if count == 0: return "NEUTRAL", 0
    avg = score / count
    if avg > 0.05: return "BULLISH", avg
    if avg < -0.05: return "BEARISH", avg
    return "NEUTRAL", avg

def find_oracle_pattern(hist_series, lookback=30, projection=15):
    """The Oracle Ghost Algorithm"""
    if len(hist_series) < (lookback * 4): return None
    
    current_pattern = hist_series.iloc[-lookback:].values
    c_min, c_max = current_pattern.min(), current_pattern.max()
    if c_max == c_min: return None
    current_norm = (current_pattern - c_min) / (c_max - c_min)
    
    best_score = -1
    best_idx = -1
    # Search history for similar patterns
    search_range = len(hist_series) - lookback - projection - 1
    
    for i in range(0, search_range, 5): # Step 5 for performance
        candidate = hist_series.iloc[i : i+lookback].values
        if candidate.max() == candidate.min(): continue
        cand_norm = (candidate - candidate.min()) / (candidate.max() - candidate.min())
        try:
            score = np.corrcoef(current_norm, cand_norm)[0, 1]
            if score > best_score:
                best_score = score
                best_idx = i
        except: continue

    if best_score > 0.70: # 70% Similarity Threshold
        ghost = hist_series.iloc[best_idx : best_idx + lookback + projection].copy()
        # Scale ghost to match current price
        scale_factor = hist_series.iloc[-1] / ghost.iloc[lookback-1]
        ghost_future = ghost.iloc[lookback:] * scale_factor
        return ghost_future
    return None

# ==========================================
# --- 3. DATABASE (LOGIN SYSTEM) ---
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
# --- 4. SESSION & AUTH FLOW ---
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
        st.toast(f"VIP ACTIVATED until {new_date}", icon="🚀")
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
    
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("### ⚡ UNLEASH THE DATA")
        st.info("Professional analysis synthesizing Technicals, Fundamentals, and AI.")
        
        tab_login, tab_signup = st.tabs(["LOG IN", "REGISTER"])
        with tab_login:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Password", type='password', key="l_pass")
            if st.button("LAUNCH TERMINAL", type="primary"):
                if email == "admin" and password == "PROTOS123":
                    st.session_state.update({'logged_in': True, 'user_email': "admin", 'expiry_date': "LIFETIME", 'user_status': 'active'})
                    st.rerun()
                else:
                    user = login_user_db(email, password)
                    if user:
                        is_active = check_subscription_validity(user[0][0], user[0][4])
                        st.session_state.update({'logged_in': True, 'user_email': user[0][0], 'expiry_date': user[0][4], 'user_status': 'active' if is_active else 'expired'})
                        st.rerun()
                    else: st.error("Invalid Credentials")
                    
        with tab_signup:
            new_email = st.text_input("New Email", key="s_email")
            new_pass = st.text_input("New Password", type='password', key="s_pass")
            if st.button("CREATE ACCOUNT"):
                if add_user(new_email, new_pass): st.success("Created! Please Log In.")
                else: st.error("Email exists.")

    with c2:
        st.video("https://youtu.be/ql1suvTu_ak")
    
    st.divider()
    with st.expander("📖 READ FULL SYSTEM DESCRIPTION"):
        st.markdown("""
        ### 🛡️ Warp Speed Terminal Features
        **1. Central Control Panel:** Macro Climate Bar (VIX, 10Y, BTC), Smart Watchlist, Verdicts (Buy/Sell).
        **2. Deep Analysis:** AI Sentiment, Advanced Fundamentals (ROE, FCF, Moat), Wall St Targets.
        **3. The Oracle:** Ghost Projection algorithm, SPY Benchmarking, Fibonacci/Bollinger.
        **4. Management:** Correlation Matrix Heatmaps, CSV Export.
        """)
        
    cols = st.columns(3)
    imgs = ["dashboard.png", "analysis.png", "risk_insiders.png"]
    caps = ["Matrix Scanner", "Deep Dive", "Risk Profile"]
    for c, img, cap in zip(cols, imgs, caps):
        with c:
            try: st.image(img, caption=cap, width=None) # Let Streamlit handle width
            except: st.caption(f"[{cap} Preview]")

# ==========================================
# --- 6. VIEW: PAYWALL ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] != 'active':
    st.warning(f"⚠️ SUBSCRIPTION EXPIRED for {st.session_state['user_email']}")
    links = {
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
        st.title("WARP SPEED")
        st.caption(f"User: {st.session_state['user_email']}")
        if st.button("LOGOUT"): st.session_state['logged_in'] = False; st.rerun()

    # --- MACRO BAR ---
    try:
        m_data = yf.download(["^VIX", "^TNX", "BTC-USD", "CL=F"], period="2d", progress=False)['Close']
        if not m_data.empty:
            last = m_data.iloc[-1]
            prev = m_data.iloc[-2]
            cols = st.columns(4)
            names = {"^VIX": "VIX (Fear)", "^TNX": "10Y Bond", "BTC-USD": "Bitcoin", "CL=F": "Oil"}
            for idx, (sym, name) in enumerate(names.items()):
                try:
                    val = last[sym]
                    chg = val - prev[sym]
                    cols[idx].metric(name, f"{val:.2f}", f"{chg:.2f}")
                except: pass
    except: pass
    st.divider()

    # --- SCANNER ENGINE ---
    @st.cache_data(ttl=300)
    def scan_market(tickers):
        results = []
        try:
            data = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=False)
        except: return []
        
        for t in tickers:
            try:
                df = data[t] if len(tickers) > 1 else data
                if df.empty or len(df) < 50: continue
                
                # Basic Calcs
                df = calculate_indicators(df) # Add RSI, MACD, BB
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev)/prev)*100
                rsi = df['RSI'].iloc[-1]
                
                # Verdict Logic
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                verdict = "HOLD"
                if curr > ma50 and rsi < 70: verdict = "BUY"
                if curr < ma50 or rsi > 70: verdict = "SELL"
                if rsi < 30: verdict = "STRONG BUY"
                
                # Sniper Score
                score = 50
                if verdict == "BUY": score += 20
                if rsi < 30: score += 20
                if df['Volume'].iloc[-1] > df['Volume'].mean(): score += 10 # Volume Spike
                
                info = yf.Ticker(t).info
                results.append({
                    "Ticker": t, "Price": curr, "Change": chg, "RSI": rsi, 
                    "Verdict": verdict, "Score": score, "History": df, "Info": info
                })
            except: continue
        return results

    # --- MAIN INTERFACE ---
    with st.form("scanner"):
        c1, c2 = st.columns([3, 1])
        with c1: query = st.text_input("ENTER ASSETS", "AAPL TSLA NVDA BTC-USD")
        with c2: run_scan = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if run_scan:
        ticks = [t.strip().upper() for t in query.replace(",", " ").split()]
        st.session_state['data'] = scan_market(ticks)

    if 'data' in st.session_state and st.session_state['data']:
        # 1. TABLE
        df_view = pd.DataFrame(st.session_state['data'])[['Ticker', 'Price', 'Change', 'Verdict', 'Score', 'RSI']]
        st.dataframe(df_view.style.map(lambda x: "color: green" if x == "BUY" else "color: red", subset=['Verdict']), use_container_width=True)
        
        # 2. ACTIONS
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("Show Correlation Matrix"):
                prices = {d['Ticker']: d['History']['Close'] for d in st.session_state['data']}
                if len(prices) > 1:
                    fig, ax = plt.subplots()
                    sns.heatmap(pd.DataFrame(prices).corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else: st.warning("Need >1 asset for matrix.")
        with c_act2:
            csv = df_view.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV", csv, "warp_scan.csv", "text/csv")

        # 3. DEEP DIVE
        st.divider()
        st.subheader("🔬 DEEP DIVE")
        sel_t = st.selectbox("Select Asset", [d['Ticker'] for d in st.session_state['data']])
        target = next(d for d in st.session_state['data'] if d['Ticker'] == sel_t)
        
        t1, t2, t3, t4 = st.tabs(["CHART & ORACLE", "FUNDAMENTALS", "NEWS AI", "RISK"])
        
        with t1: # Charting
            hist = target['History']
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist.index, hist['Close'], label='Price', color='white')
            ax.plot(hist.index, hist['UpperBB'], label='Upper BB', color='cyan', alpha=0.3)
            ax.plot(hist.index, hist['LowerBB'], label='Lower BB', color='cyan', alpha=0.3)
            
            # ORACLE GHOST
            ghost = find_oracle_pattern(hist['Close'])
            if ghost is not None:
                # Create future dates
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(len(ghost))]
                ax.plot(future_dates, ghost, label='Oracle Ghost', color='magenta', linestyle='--')
            
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.legend()
            st.pyplot(fig)
            
        with t2: # Fundamentals
            i = target['Info']
            c1, c2, c3 = st.columns(3)
            c1.metric("P/E Ratio", i.get('trailingPE', '-'))
            c1.metric("ROE", f"{i.get('returnOnEquity', 0)*100:.2f}%")
            c2.metric("PEG Ratio", i.get('pegRatio', '-'))
            c2.metric("Profit Margin", f"{i.get('profitMargins', 0)*100:.2f}%")
            c3.metric("Debt/Equity", i.get('debtToEquity', '-'))
            c3.metric("Free Cash Flow", f"${i.get('freeCashflow', 0)/1e9:.2f}B")
            
        with t3: # News AI
            st.write("Recent News Sentiment:")
            news = target.get('news', [])
            if news:
                for n in news[:5]:
                    sent, score = analyze_sentiment([n])
                    color = "green" if sent == "BULLISH" else "red" if sent == "BEARISH" else "gray"
                    # Safe get title/link
                    title = n.get('title', 'No Title')
                    link = n.get('link', '#')
                    st.markdown(f"**:{color}[{sent}]** [{title}]({link})")
            else: st.write("No news found.")
            
        with t4: # Risk
            i = target['Info']
            c1, c2 = st.columns(2)
            c1.metric("Beta (Volatility)", i.get('beta', '-'))
            c2.metric("Short Ratio", i.get('shortRatio', '-'))
            st.caption("Institutional Holders:")
            try: st.dataframe(target['yfinance_obj'].institutional_holders.head())
            except: st.write("Data hidden")

    elif not run_scan:
        st.info("Enter tickers above and press INITIATE SCAN.")
