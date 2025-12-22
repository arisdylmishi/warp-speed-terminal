import streamlit as st
import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Dark Mode & Terminal Styling
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; font-weight: 800; letter-spacing: -0.5px; }
        
        .stButton>button {
            width: 100%;
            border-radius: 4px;
            font-weight: bold;
            height: 3em;
            text-transform: uppercase;
            border: 1px solid #333;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem;
            color: #00FFCC; /* Neon Cyan */
            font-family: 'Courier New', monospace;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #aaa;
        }
        
        /* Custom Box for Explanations */
        .reason-box {
            background-color: #111;
            padding: 10px;
            border-left: 3px solid #00FFCC;
            margin-bottom: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. ADVANCED LOGIC ---
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
    if not news_items: return "NEUTRAL", 0
    
    for item in news_items:
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
    """The Oracle Ghost Algorithm (Optimized for visibility)"""
    if len(hist_series) < (lookback * 4): return None
    
    current_pattern = hist_series.iloc[-lookback:].values
    c_min, c_max = current_pattern.min(), current_pattern.max()
    if c_max == c_min: return None
    current_norm = (current_pattern - c_min) / (c_max - c_min)
    
    best_score = -1
    best_idx = -1
    search_range = len(hist_series) - lookback - projection - 1
    
    # Check every 2 steps to find patterns better
    for i in range(0, search_range, 2): 
        candidate = hist_series.iloc[i : i+lookback].values
        if candidate.max() == candidate.min(): continue
        cand_norm = (candidate - candidate.min()) / (candidate.max() - candidate.min())
        try:
            score = np.corrcoef(current_norm, cand_norm)[0, 1]
            if score > best_score:
                best_score = score
                best_idx = i
        except: continue

    # Lowered threshold to 0.60 so it appears more often
    if best_score > 0.60:
        ghost = hist_series.iloc[best_idx : best_idx + lookback + projection].copy()
        scale_factor = hist_series.iloc[-1] / ghost.iloc[lookback-1]
        ghost_future = ghost.iloc[lookback:] * scale_factor
        return ghost_future
    return None

def format_large_number(num):
    if not num or isinstance(num, str): return "N/A"
    if num >= 1e12: return f"${num/1e12:.2f}T"
    if num >= 1e9: return f"${num/1e9:.2f}B"
    if num >= 1e6: return f"${num/1e6:.2f}M"
    return f"${num:.2f}"

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
    # --- DESCRIPTION SECTION ---
    with st.expander("📖 READ FULL SYSTEM DESCRIPTION", expanded=True):
        st.markdown("""
        ### Warp Speed Terminal: The Ultimate Stock Market Intelligence System
        Warp Speed Terminal is a professional analysis platform that synthesizes Technical Analysis, Fundamental Data, and Artificial Intelligence.
        
        #### Detailed Features:
        **1. Central Control Panel (Smart Dashboard)**
        * **Macro Climate Bar:** Live monitoring of the global market (VIX, 10Y, BTC, Oil).
        * **Verdict:** Clear BUY/SELL signals based on 50/200 MA and RSI.
        * **Sniper Score (/100):** Quantitative scoring of the opportunity.
        
        **2. Deep Analysis (Deep Dive View)**
        * **Wall Street:** Analyst Price Targets and Consensus.
        * **Fundamentals:** Market Cap, Dividend Yield, ROE, FCF.
        * **Risk:** Beta, Short Float, Institutional Holders.
        
        **3. Advanced Charting & "The Oracle"**
        * **Oracle Projection:** Algorithm identifying past patterns (Ghost) to forecast future.
        * **SPY Overlay:** Benchmarking against S&P 500.
        
        **4. Management**
        * **Correlation Matrix** & **Data Export**.
        """)
        
    st.markdown("<br><h2 style='text-align: center; color: #fff;'>PLATFORM PREVIEW</h2><br>", unsafe_allow_html=True)
    cols = st.columns(3)
    # Using explicit file names as requested (PNG)
    imgs = ["dashboard.png", "analysis.png", "risk_insiders.png"]
    caps = ["Matrix Scanner", "Deep Dive", "Risk Profile"]
    for c, img, cap in zip(cols, imgs, caps):
        with c:
            try: st.image(img, caption=cap, use_container_width=True) 
            except: st.info(f"[{cap} Preview - Check {img} in Repo]")
            
    st.markdown("<p style='text-align: center; color: #555; margin-top: 50px;'>Support: warpspeedterminal@gmail.com</p>", unsafe_allow_html=True)

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
    with col1: st.link_button("GET 1 MONTH (€25)", STRIPE_LINKS['1M'], use_container_width=True)
    with col2: st.link_button("GET 3 MONTHS (€23/mo)", STRIPE_LINKS['3M'], use_container_width=True)
    with col3: st.link_button("GET 6 MONTHS (€20/mo)", STRIPE_LINKS['6M'], use_container_width=True)
    with col4: st.link_button("GET 1 YEAR (€15/mo)", STRIPE_LINKS['1Y'], type="primary", use_container_width=True)
    
    st.markdown("<br><p style='text-align: center; color: #555;'>Support: warpspeedterminal@gmail.com</p>", unsafe_allow_html=True)
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
        st.markdown("---")
        st.markdown("📧 **Support:**\nwarpspeedterminal@gmail.com")

    # --- MACRO BAR (Robust Error Handling) ---
    with st.container():
        try:
            macro_ticks = ["^VIX", "^TNX", "BTC-USD", "CL=F"]
            m_data = yf.download(macro_ticks, period="5d", progress=False)['Close']
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            names = {"^VIX": "VIX (Fear)", "^TNX": "10Y Bond", "BTC-USD": "Bitcoin", "CL=F": "Oil"}
            
            if not m_data.empty and len(m_data) >= 2:
                last_row = m_data.iloc[-1]
                prev_row = m_data.iloc[-2]
                
                for idx, (sym, name) in enumerate(names.items()):
                    val = last_row.get(sym, np.nan)
                    prev_val = prev_row.get(sym, np.nan)
                    
                    if pd.isna(val) or pd.isna(prev_val) or prev_val == 0:
                        cols = [mc1, mc2, mc3, mc4]
                        cols[idx].metric(name, "N/A", "N/A")
                    else:
                        chg = ((val - prev_val) / prev_val) * 100
                        cols = [mc1, mc2, mc3, mc4]
                        cols[idx].metric(name, f"{val:.2f}", f"{chg:+.2f}%")
            else:
                st.caption("Macro data unavailable (Market Closed/API Limit)")
        except Exception as e: 
            st.caption(f"Macro Data Error: {str(e)}")
            
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
                # Handle single vs multi ticker logic
                if len(tickers) > 1:
                    if t not in data.columns.levels[0]: continue
                    df = data[t].copy()
                else:
                    df = data.copy() # If single ticker, data is already the dataframe
                
                if df.empty or len(df) < 50: continue
                
                # Indicators
                df = calculate_indicators(df)
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev)/prev)*100
                rsi = df['RSI'].iloc[-1]
                
                # Verdict & Explanations Logic
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                ma200 = df['Close'].rolling(200).mean().iloc[-1]
                
                verdict = "HOLD"
                reasons = [] # Store reasons for Deep Dive
                
                if curr > ma50:
                    reasons.append(f"✓ Price (${curr:.2f}) > 50MA (${ma50:.2f}) -> Bullish Trend")
                    if rsi < 70: verdict = "BUY"
                else:
                    reasons.append(f"✗ Price (${curr:.2f}) < 50MA (${ma50:.2f}) -> Bearish Trend")
                    if rsi > 70: verdict = "SELL"
                
                if rsi < 30: 
                    verdict = "STRONG BUY"
                    reasons.append(f"✓ RSI ({rsi:.0f}) is Oversold -> Potential Bounce")
                elif rsi > 70:
                    verdict = "SELL"
                    reasons.append(f"✗ RSI ({rsi:.0f}) is Overbought -> Pullback Risk")
                
                # Sniper Score
                score = 50
                if verdict == "BUY": score += 20
                if verdict == "STRONG BUY": score += 35
                if rsi < 30: score += 20
                
                vol_mean = df['Volume'].rolling(50).mean().iloc[-1]
                curr_vol = df['Volume'].iloc[-1]
                rvol = curr_vol / vol_mean if vol_mean > 0 else 1.0
                if rvol > 1.5: 
                    score += 10
                    reasons.append(f"⚡ High Volume (RVOL {rvol:.1f}) -> Institutional Activity")
                
                # Info
                info = yf.Ticker(t).info
                pe = info.get('trailingPE', None)
                bubble = "NO"
                if pe and pe > 35 and curr > ma200 * 1.4: 
                    bubble = "🚨 YES"
                    score -= 20
                    reasons.append("⚠️ Bubble Alert: High P/E & Extended Price")
                
                peg = info.get('pegRatio', 'N/A')
                
                # Wall Street Data
                target_price = info.get('targetMeanPrice', 'N/A')
                consensus = info.get('recommendationKey', 'N/A').upper().replace('_', ' ')
                
                # Sentiment
                news = yf.Ticker(t).news
                sent, sent_score = analyze_sentiment(news)
                if sent == "BULLISH": reasons.append("✓ News Sentiment is Positive")
                elif sent == "BEARISH": reasons.append("✗ News Sentiment is Negative")
                
                results.append({
                    "Ticker": t, "Price": curr, "Change": chg, "Verdict": verdict, "Sniper": score, 
                    "RVOL": rvol, "Bubble": bubble, "PEG": peg, "RSI": rsi, "Sentiment": sent,
                    "History": df, "Info": info, "News": news, "Reasons": reasons,
                    "TargetPrice": target_price, "Consensus": consensus
                })
            except Exception as e: continue
        return results

    # --- MAIN INTERFACE ---
    with st.form("scanner"):
        c1, c2 = st.columns([3, 1])
        with c1: query = st.text_input("ENTER ASSETS", "AAPL TSLA NVDA BTC-USD JPM COIN")
        with c2: run_scan = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if run_scan:
        ticks = [t.strip().upper() for t in query.replace(",", " ").split()]
        if ticks:
            st.session_state['data'] = scan_market(ticks)
        else:
            st.warning("Please enter at least one symbol.")

    if 'data' in st.session_state and st.session_state['data']:
        # 1. TABLE (Dashboard Style)
        df_view = pd.DataFrame([{
            "TICKER": d['Ticker'],
            "PRICE": f"{d['Price']:.2f}",
            "CHANGE %": f"{d['Change']:+.2f}%",
            "VERDICT": d['Verdict'],
            "SNIPER": d['Sniper'],
            "RVOL": f"{d['RVOL']:.1f}",
            "BUBBLE?": d['Bubble'],
            "PEG": d['PEG'],
            "RSI": f"{d['RSI']:.1f}",
            "SENTIMENT": d['Sentiment']
        } for d in st.session_state['data']])
        
        def highlight_verdict(val):
            color = '#00FFCC' if 'BUY' in val else '#ff4b4b' if 'SELL' in val else 'white'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(df_view.style.map(highlight_verdict, subset=['VERDICT']), use_container_width=True, hide_index=True)
        
        # 2. ACTIONS
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("Show Correlation Matrix"):
                prices = {d['Ticker']: d['History']['Close'] for d in st.session_state['data']}
                if len(prices) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(pd.DataFrame(prices).corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else: st.warning("Need >1 asset for matrix.")
        with c_act2:
            csv = df_view.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV", csv, "warp_scan.csv", "text/csv")

        # 3. DEEP DIVE
        st.divider()
        st.subheader("🔬 DEEP DIVE ANALYSIS")
        sel_t = st.selectbox("Select Asset", [d['Ticker'] for d in st.session_state['data']])
        target = next(d for d in st.session_state['data'] if d['Ticker'] == sel_t)
        
        t1, t2, t3, t4 = st.tabs(["CHART & ORACLE", "FUNDAMENTALS & WALL ST", "NEWS AI", "RISK"])
        
        with t1: # PLOTLY CHART
            hist = target['History']
            ghost = find_oracle_pattern(hist['Close'])
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0
