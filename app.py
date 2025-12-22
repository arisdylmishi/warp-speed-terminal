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

# Professional Dark Mode Styling (Bloomberg Style)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* General App Styling */
        .stApp {
            background-color: #0b0c10;
            color: #c5c6c7;
        }
        
        /* Fonts */
        h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; font-weight: 800; letter-spacing: -0.5px; color: #66fcf1 !important; }
        p, div, label, span { font-family: 'Consolas', monospace; }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 2px;
            font-weight: bold;
            height: 3em;
            text-transform: uppercase;
            background-color: #1f2833;
            color: #66fcf1;
            border: 1px solid #45a29e;
        }
        .stButton>button:hover {
            background-color: #45a29e;
            color: #0b0c10;
        }
        
        /* Metrics Styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.1rem;
            color: #00e676; /* Neon Green */
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem;
            color: #aaa;
        }
        
        /* Table Styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #333;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. ADVANCED LOGIC (MATH, AI, ORACLE) ---
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
    if not news_items: return "NEUTRAL"
    score = 0
    count = 0
    for item in news_items:
        title = item.get('title', '')
        if not title: continue
        try:
            blob = TextBlob(title)
            score += blob.sentiment.polarity
            count += 1
        except: continue
        
    if count == 0: return "NEUTRAL"
    avg = score / count
    if avg > 0.05: return "BULLISH"
    if avg < -0.05: return "BEARISH"
    return "NEUTRAL"

def find_oracle_pattern(hist_series, lookback=30, projection=15):
    """The Oracle Ghost Algorithm"""
    if len(hist_series) < (lookback * 4): return None
    
    current_pattern = hist_series.iloc[-lookback:].values
    c_min, c_max = current_pattern.min(), current_pattern.max()
    if c_max == c_min: return None
    current_norm = (current_pattern - c_min) / (c_max - c_min)
    
    best_score = -1
    best_idx = -1
    # Search history (optimized step)
    search_range = len(hist_series) - lookback - projection - 1
    
    for i in range(0, search_range, 3): 
        candidate = hist_series.iloc[i : i+lookback].values
        if candidate.max() == candidate.min(): continue
        cand_norm = (candidate - candidate.min()) / (candidate.max() - candidate.min())
        try:
            score = np.corrcoef(current_norm, cand_norm)[0, 1]
            if score > best_score:
                best_score = score
                best_idx = i
        except: continue

    if best_score > 0.75:
        ghost = hist_series.iloc[best_idx : best_idx + lookback + projection].copy()
        scale_factor = hist_series.iloc[-1] / ghost.iloc[lookback-1]
        ghost_future = ghost.iloc[lookback:] * scale_factor
        return ghost_future
    return None

def calculate_fibonacci(hist):
    high = hist['High'].max()
    low = hist['Low'].min()
    diff = high - low
    return {
        0.236: high - (diff * 0.236),
        0.382: high - (diff * 0.382),
        0.5: high - (diff * 0.5),
        0.618: high - (diff * 0.618)
    }

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
    with st.expander("📖 READ FULL SYSTEM DESCRIPTION", expanded=True):
        st.markdown("""
        ### Warp Speed Terminal: The Ultimate Stock Market Intelligence System
        Warp Speed Terminal is a professional analysis platform that synthesizes Technical Analysis, Fundamental Data, and Artificial Intelligence.
        
        #### Detailed Features:
        **1. Central Control Panel (Smart Dashboard)**
        * **Macro Climate Bar:** Live monitoring (VIX, Bonds, BTC, Oil).
        * **Evaluation:** Verdicts, Sniper Score, Bubble Alerts, RVOL.
        
        **2. Deep Analysis**
        * **AI Tab:** NLP news sentiment (Bullish/Bearish).
        * **Fundamentals:** P/E, PEG, ROE, FCF, Moat.
        * **Risk:** Beta, Short Float, Institutional Holders.
        
        **3. Advanced Charting & "The Oracle"**
        * **Oracle Projection:** Algorithm scanning patterns to forecast future price (Ghost).
        * **SPY Overlay:** Benchmark against S&P 500.
        
        **4. Management**
        * **Correlation Matrix** & **Data Export**.
        """)
        
    st.markdown("<br><h2 style='text-align: center; color: #fff;'>PLATFORM PREVIEW</h2><br>", unsafe_allow_html=True)
    cols = st.columns(3)
    # Using explicit file names as requested
    imgs = ["dashboard.png", "analysis.jpg", "risk_insiders.png"]
    caps = ["Matrix Scanner", "Deep Dive", "Risk Profile"]
    for c, img, cap in zip(cols, imgs, caps):
        with c:
            try: st.image(img, caption=cap, use_container_width=True) 
            except: st.info(f"[{cap} Preview]")

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
        st.markdown("**Support:**\nwarpspeedterminal@gmail.com")

    # --- MACRO BAR (FIXED NAN) ---
    try:
        macro_ticks = ["^VIX", "^TNX", "BTC-USD", "CL=F"]
        # Fetch minimal data to be fast
        m_data = yf.download(macro_ticks, period="5d", progress=False)['Close']
        
        cols = st.columns(4)
        names = {"^VIX": "VIX (Fear)", "^TNX": "10Y Bond", "BTC-USD": "Bitcoin", "CL=F": "Oil"}
        
        if not m_data.empty and len(m_data) >= 2:
            last = m_data.iloc[-1]
            prev = m_data.iloc[-2]
            for idx, (sym, name) in enumerate(names.items()):
                try:
                    val = last.get(sym, np.nan)
                    pval = prev.get(sym, np.nan)
                    if pd.notna(val) and pd.notna(pval):
                        chg = ((val - pval) / pval) * 100
                        cols[idx].metric(name, f"{val:.2f}", f"{chg:+.2f}%")
                    else:
                        cols[idx].metric(name, "N/A", "0.00%")
                except: cols[idx].metric(name, "N/A", "0.00%")
        else:
            for idx, name in enumerate(names.values()): cols[idx].metric(name, "Loading...", "")
    except: pass
    st.divider()

    # --- SCANNER ENGINE (ROBUST) ---
    @st.cache_data(ttl=300)
    def scan_market(tickers):
        results = []
        try:
            data = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=False)
        except: return []
        
        for t in tickers:
            try:
                # Data Handling
                if len(tickers) > 1:
                    if t not in data.columns.levels[0]: continue
                    df = data[t].copy()
                else:
                    df = data.copy()
                
                if df.empty or len(df) < 50: continue
                
                # Indicators
                df = calculate_indicators(df)
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev)/prev)*100
                rsi = df['RSI'].iloc[-1]
                
                # Verdict Logic
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                ma200 = df['Close'].rolling(200).mean().iloc[-1]
                
                verdict = "HOLD"
                if curr > ma50 and rsi < 70: verdict = "BUY"
                if curr < ma50 or rsi > 70: verdict = "SELL"
                if rsi < 30: verdict = "STRONG BUY"
                
                # Sniper Score
                score = 50
                if verdict == "BUY": score += 20
                if rsi < 30: score += 20
                vol_mean = df['Volume'].rolling(50).mean().iloc[-1]
                if df['Volume'].iloc[-1] > vol_mean * 1.5: score += 10
                
                # Info
                info = yf.Ticker(t).info
                pe = info.get('trailingPE', None)
                bubble = "NO"
                if pe and pe > 35 and curr > ma200 * 1.4: 
                    bubble = "🚨 YES"
                    score -= 20
                
                peg = info.get('pegRatio', 'N/A')
                
                # Sentiment
                news = yf.Ticker(t).news
                sent = analyze_sentiment(news)
                
                # RVOL
                rvol = df['Volume'].iloc[-1] / vol_mean if vol_mean > 0 else 1.0

                results.append({
                    "Ticker": t, 
                    "Price": curr, 
                    "Change": chg, 
                    "Verdict": verdict, 
                    "Sniper": score, 
                    "RVOL": rvol,
                    "Bubble": bubble,
                    "PEG": peg,
                    "RSI": rsi,
                    "Sentiment": sent,
                    "History": df, 
                    "Info": info,
                    "News": news
                })
            except: continue
        return results

    # --- MAIN INTERFACE ---
    with st.form("scanner"):
        c1, c2 = st.columns([3, 1])
        with c1: query = st.text_input("ENTER ASSETS", "AAPL TSLA NVDA BTC-USD JPM COIN")
        with c2: run_scan = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if run_scan:
        ticks = [t.strip().upper() for t in query.replace(",", " ").split()]
        st.session_state['data'] = scan_market(ticks)

    if 'data' in st.session_state and st.session_state['data']:
        # 1. TABLE
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
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.heatmap(pd.DataFrame(prices).corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else: st.warning("Need >1 asset for matrix.")
        with c_act2:
            csv = df_view.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV", csv, "warp_scan.csv", "text/csv")

        # 3. DEEP DIVE (RICH PLOTLY CHARTS)
        st.divider()
        st.subheader("🔬 DEEP DIVE ANALYSIS")
        sel_t = st.selectbox("Select Asset", [d['Ticker'] for d in st.session_state['data']])
        target = next(d for d in st.session_state['data'] if d['Ticker'] == sel_t)
        
        t1, t2, t3, t4 = st.tabs(["CHART & ORACLE", "FUNDAMENTALS", "NEWS AI", "RISK"])
        
        with t1: 
            # PLOTLY INTERACTIVE CHART
            hist = target['History']
            ghost = find_oracle_pattern(hist['Close'])
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Candlesticks
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            # BB
            fig.add_trace(go.Scatter(x=hist.index, y=hist['UpperBB'], line=dict(color='cyan', width=1), name='Upper BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['LowerBB'], line=dict(color='cyan', width=1), name='Lower BB'), row=1, col=1)
            
            # Oracle Ghost
            if ghost is not None:
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(len(ghost))]
                fig.add_trace(go.Scatter(x=future_dates, y=ghost, line=dict(color='magenta', dash='dash', width=2), name='Oracle Ghost'), row=1, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='#00FFCC'), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], line=dict(color='#ff4b4b'), name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD']-hist['Signal'], marker_color='gray', name='Hist'), row=2, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"{target['Ticker']} Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            i = target['Info']
            c1, c2, c3 = st.columns(3)
            c1.metric("P/E Ratio", i.get('trailingPE', '-'))
            c1.metric("ROE", f"{i.get('returnOnEquity', 0)*100:.2f}%" if i.get('returnOnEquity') else '-')
            c2.metric("PEG Ratio", i.get('pegRatio', '-'))
            c2.metric("Profit Margin", f"{i.get('profitMargins', 0)*100:.2f}%" if i.get('profitMargins') else '-')
            c3.metric("Debt/Equity", i.get('debtToEquity', '-'))
            c3.metric("Free Cash Flow", f"${i.get('freeCashflow', 0)/1e9:.2f}B" if i.get('freeCashflow') else '-')
            
        with t3:
            st.write("Recent News Sentiment:")
            if target['News']:
                for n in target['News'][:5]:
                    # Check title existance
                    t_title = n.get('title', 'No Title')
                    t_link = n.get('link', '#')
                    # Sentiment
                    try: 
                        blob = TextBlob(t_title)
                        pol = blob.sentiment.polarity
                        sent = "BULLISH" if pol > 0.05 else "BEARISH" if pol < -0.05 else "NEUTRAL"
                    except: sent = "NEUTRAL"
                    
                    color = "green" if sent == "BULLISH" else "red" if sent == "BEARISH" else "gray"
                    st.markdown(f"**:{color}[{sent}]** [{t_title}]({t_link})")
            else: st.write("No news found.")
            
        with t4:
            i = target['Info']
            c1, c2 = st.columns(2)
            c1.metric("Beta (Volatility)", i.get('beta', '-'))
            c2.metric("Short Ratio", i.get('shortRatio', '-'))
            st.caption("Institutional Holders:")
            # Safe Fetch
            try: st.dataframe(yf.Ticker(sel_t).institutional_holders.head())
            except: st.write("Data hidden")

    elif not run_scan:
        st.info("Enter tickers above and press INITIATE SCAN.")
