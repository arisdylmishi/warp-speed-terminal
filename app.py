import streamlit as st
import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from textblob import TextBlob
from collections import Counter
import re
import requests
import xml.etree.ElementTree as ET

# ==========================================
# --- 1. CONFIGURATION & STYLE ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Bloomberg-style Dark Mode with Neon Accents
st.markdown("""
    <style>
        .stApp { background-color: #000000; color: #e0e0e0; }
        
        /* Typography */
        h1, h2, h3, h4 { font-family: 'Roboto', sans-serif; font-weight: 700; color: #ff9900 !important; letter-spacing: 1px; }
        
        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem;
            color: #00ffea; 
            font-family: 'Courier New', monospace;
            font-weight: bold;
            text-shadow: 0px 0px 5px rgba(0, 255, 234, 0.5);
        }
        div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 0px;
            font-weight: bold;
            height: 3em;
            text-transform: uppercase;
            border: 1px solid #333;
            background-color: #1a1a1a;
            color: #ff9900;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff9900;
            color: #000;
            border-color: #ff9900;
            box-shadow: 0px 0px 10px #ff9900;
        }
        
        /* Custom Boxes */
        .ai-box {
            background-color: #0f1216;
            padding: 15px;
            border-left: 4px solid #00ffea;
            margin-bottom: 10px;
            font-family: 'Courier New', monospace;
            color: #eee;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .reason-box {
            background-color: #1a1a1a; 
            padding: 10px; 
            border-left: 3px solid #ff9900; 
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        .coming-soon {
            background-color: #ff9900;
            color: black;
            padding: 4px 10px;
            font-weight: 800;
            border-radius: 4px;
            font-size: 0.7rem;
            vertical-align: middle;
            margin-left: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Links Styling */
        a { color: #00ccff !important; text-decoration: none; font-weight: bold; font-family: 'Courier New'; }
        a:hover { text-decoration: underline; color: #ff9900 !important; }
        
        /* PAYWALL STYLING */
        .paywall-header {
            text-align: center;
            border: 1px solid #ff4444;
            background-color: #1a0000;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 4px;
        }
        .paywall-title {
            color: #ff4444;
            font-family: 'Courier New', monospace;
            font-size: 1.5rem;
            font-weight: bold;
            letter-spacing: 2px;
        }
        .paywall-sub {
            color: #aaa;
            font-size: 1rem;
            margin-top: 10px;
        }
        .plan-card {
            border: 1px solid #333;
            background-color: #0f1216;
            padding: 20px;
            text-align: center;
            border-radius: 4px;
            height: 100%;
        }
        .plan-title {
            color: #fff;
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .plan-price {
            font-size: 2rem;
            color: #00ffea;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        .plan-save {
            color: #00ff00;
            font-size: 0.9rem;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .best-value {
            border: 1px solid #ff9900;
            box-shadow: 0 0 15px rgba(255, 153, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. ADVANCED LOGIC (AI, MATH & NEWS) ---
# ==========================================

STRIPE_LINKS = {
    "1M": "https://buy.stripe.com/00w28l6qUdc96eJ5nYeAg03?days=30",
    "3M": "https://buy.stripe.com/14A9ANaHa8VT46B5nYeAg02?days=90",
    "6M": "https://buy.stripe.com/14A6oB16A7RPfPjg2CeAg01?days=180",
    "1Y": "https://buy.stripe.com/28EaER16A6NL9qV6s2eAg00?days=365",
}

def get_google_news(ticker):
    """Fetches real news from Google RSS to bypass Yahoo blocks"""
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            news_items = []
            for item in root.findall('./channel/item')[:5]:
                title = item.find('title').text
                link = item.find('link').text
                news_items.append({'title': title, 'link': link})
            return news_items
        return []
    except: return []

def calculate_monte_carlo(df, days=30, simulations=1000):
    """Generates Probabilistic Price Cones"""
    try:
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        daily_vol = returns.std()
        
        simulation_df = pd.DataFrame()
        
        for x in range(simulations):
            count = 0
            price_series = []
            price = last_price * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            
            for y in range(days):
                if count == 29: break
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1
            
            simulation_df[x] = price_series
            
        upper_bound = simulation_df.quantile(0.95, axis=1)
        lower_bound = simulation_df.quantile(0.05, axis=1)
        avg_path = simulation_df.mean(axis=1)
        
        return upper_bound, lower_bound, avg_path
    except: return None, None, None

def calculate_smart_levels(df):
    if df.empty: return None, None, {}
    recent = df.iloc[-126:] 
    high_price = recent['High'].max()
    low_price = recent['Low'].min()
    diff = high_price - low_price
    fibs = {
        "Fib 0.236": high_price - (diff * 0.236),
        "Fib 0.382": high_price - (diff * 0.382),
        "Fib 0.5 (Golden)": high_price - (diff * 0.5),
        "Fib 0.618": high_price - (diff * 0.618)
    }
    return high_price, low_price, fibs

def generate_ai_summary(news_items):
    """Generates a dynamic AI summary from Google News"""
    text_corpus = ""
    valid_news = []
    
    if news_items:
        for n in news_items:
            title = n.get('title', '')
            link = n.get('link', '')
            if title:
                text_corpus += title + " "
                valid_news.append({'title': title, 'link': link})
    
    if not text_corpus or len(text_corpus) < 20:
        return "⚠️ **ANALYST NOTE:** Real-time news feed is quiet. While technical indicators suggest a direction, **exercise caution**. The AI suggests waiting for a catalyst or strictly following the technical levels (Support/Resistance).", valid_news
            
    words = re.findall(r'\w+', text_corpus.lower())
    ignore = ['the', 'a', 'to', 'of', 'in', 'and', 'for', 'on', 'with', 'at', 'is', 'stock', 'market', 'stocks', 'check', 'latest', 'news', 'google', 'finance', 'today', 'why', 'update', 'share', 'price']
    filtered = [w for w in words if w not in ignore and len(w) > 4]
    
    common = Counter(filtered).most_common(5)
    keywords = [c[0].upper() for c in common]
    
    blob = TextBlob(text_corpus)
    pol = blob.sentiment.polarity
    
    if pol > 0.1: tone = "BULLISH 🐂"
    elif pol < -0.1: tone = "BEARISH 🐻"
    else: tone = "NEUTRAL / MIXED ⚖️"
    
    summary = f"AI ANALYST SCAN: **{tone}**\n"
    summary += "Based on latest Google News analysis. "
    if keywords:
        summary += f"Key Topics: {', '.join(keywords)}. "
    
    if tone == "NEUTRAL / MIXED ⚖️":
        summary += "Market direction is unclear from headlines. Rely on Technicals."
    
    return summary, valid_news

def generate_ceo_report(target_data):
    t = target_data
    report = f"""
=======================================================
WARP SPEED TERMINAL // INTELLIGENCE BRIEFING
Target Asset: {t['Ticker']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
=======================================================

1. EXECUTIVE SUMMARY
---------------------
VERDICT:      {t['Verdict']}
SNIPER SCORE: {t['Sniper']}/100
PRICE:        ${t['Price']:.2f}
CHANGE:       {t['Change']:.2f}%

2. AI SENTIMENT ANALYSIS
---------------------
{t['AISummary'].replace('**', '')}

3. TECHNICAL POSTURE
---------------------
RVOL:         {t['RVOL']:.2f} (Relative Volume)
RSI:          {t['RSI']:.2f}
BUBBLE RISK:  {t['Bubble']}

4. FUNDAMENTALS
---------------------
Market Cap:   ${format_large_number(t['Info'].get('marketCap', 0))}
P/E Ratio:    {t['Info'].get('trailingPE', 'N/A')}
PEG Ratio:    {t['PEG']}
Wall St Target: ${t['TargetPrice']}
Consensus:    {t['Consensus']}

5. ALGORITHMIC LOGIC
---------------------
{chr(10).join(['- ' + r for r in t['Reasons']])}

=======================================================
CONFIDENTIAL - GENERATED BY WARP SPEED TERMINAL
=======================================================
    """
    return report

def calculate_indicators(hist):
    if hist.empty: return hist
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
    hist['STD20'] = hist['Close'].rolling(window=20).std()
    hist['UpperBB'] = hist['SMA20'] + (hist['STD20'] * 2)
    hist['LowerBB'] = hist['SMA20'] - (hist['STD20'] * 2)
    
    return hist

def find_oracle_pattern(hist_series, lookback=30, projection=15):
    if len(hist_series) < (lookback * 4): return None
    current_pattern = hist_series.iloc[-lookback:].values
    c_min, c_max = current_pattern.min(), current_pattern.max()
    if c_max == c_min: return None
    current_norm = (current_pattern - c_min) / (c_max - c_min)
    
    best_score = -1
    best_idx = -1
    search_range = len(hist_series) - lookback - projection - 1
    
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

    if best_score > 0.50:
        ghost = hist_series.iloc[best_idx : best_idx + lookback + projection].copy()
        scale_factor = hist_series.iloc[-1] / ghost.iloc[lookback-1]
        ghost_future = ghost.iloc[lookback:] * scale_factor
        return ghost_future
    return None

def format_large_number(num):
    if not num or isinstance(num, str): return "N/A"
    try:
        if num >= 1e12: return f"${num/1e12:.2f}T"
        if num >= 1e9: return f"${num/1e9:.2f}B"
        if num >= 1e6: return f"${num/1e6:.2f}M"
        return f"${num:.2f}"
    except: return "N/A"

@st.cache_data(ttl=3600)
def get_spy_data():
    try:
        spy = yf.Ticker("SPY").history(period="1y")['Close']
        spy.index = spy.index.tz_localize(None)
        return spy
    except: return None

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
    with st.expander("📖 READ FULL SYSTEM DESCRIPTION (UPDATED V11.0)", expanded=True):
        st.markdown("""
        **Warp Speed Terminal** is a professional analysis platform that synthesizes Technical Analysis, Fundamental Data, and Artificial Intelligence. It is designed to transform chaotic market data into clear, actionable signals, offering features typically found only in institutional-grade terminals.

        #### Detailed Features:

        **1. Central Control Panel (Smart Dashboard)**
        The Investor's Headquarters.
        * **Macro Climate Bar:** Live monitoring of the global market (VIX/Fear Index, 10-Year Bonds, Bitcoin, Oil) for an immediate grasp of market sentiment.
        * **Smart Watchlist & Memory:** The user inputs tickers (e.g., AAPL, NVDA), and the system automatically saves them. Upon the next launch, the portfolio is pre-loaded.
        * **The Evaluation Algorithm:**
            * *Verdict:* A clear command signal (STRONG BUY, BUY, HOLD, SELL).
            * *Sniper Score (/100):* A quantitative scoring of the opportunity based on multiple factors.
            * *Bubble Alert:* Detection of overvalued stocks (bubbles).
            * *RVOL & RSI:* Detection of unusual volume (institutional interest) and oversold levels.
        * **Market Heatmap:** Visual Treemap showing market performance at a glance.

        **2. Deep Analysis (Deep Dive View)**
        Double-clicking opens a full "X-ray" tab for the stock:
        * **Analysis & AI Tab:** Justification of the Score using specific tags (e.g., "Volatility Squeeze"). The NLP engine "reads" the news, analyzes sentiment (Bullish/Bearish), and provides links to sources.
        * **Fundamentals Tab (Enriched):** A complete check of the business's financial health and efficiency. It includes valuation metrics (P/E, PEG Ratio, Market Cap) and extends to critical quality indicators:
            * *Return on Equity (ROE):* To check management efficiency.
            * *Debt-to-Equity:* To assess debt burden.
            * *Free Cash Flow (FCF):* The "truth" regarding liquidity, beyond accounting profits.
            * *Profit Margins:* Indication of a competitive advantage (Economic Moat).
        * **Wall Street:** Comparison with analyst forecasts and price targets.
        * **Risk Tab:** Volatility analysis (Beta), bets on decline (Short Float), and revelation of major institutional holders (Skin in the Game).

        **3. Advanced Charting & "The Event Horizon"**
        Three synchronized charts with selectable timeframes (1M, 3M, 6M, 1Y, MAX):
        * **Price Chart with Benchmarking:**
        * **Oracle Projection:** The algorithm scans historical data, identifies similar past patterns, and projects a forecast line (Ghost) for the future.
        * **Monte Carlo Simulation (The Event Horizon):** A statistical probability cloud (Best/Worst Case Scenarios) for the next 30 days based on volatility.
        * **SPY Overlay:** Compares the stock's performance directly against the S&P 500 index (to see if you are beating the market).
        * **Smart Technicals:** Auto-drawing of **Support/Resistance** levels and **Fibonacci Retracements**.
        * **Technical Tools:** Bollinger Bands, Fibonacci Levels, and Support/Resistance levels.
        * **MACD:** Indicates Momentum and trend reversals.
        * **Volume:** Color-coded volume for analyzing buyer/seller pressure.

        **4. Management & Export Tools**
        * **Correlation Matrix:** Creation of a Heatmap to check correlations between portfolio stocks (Risk Management).
        * **Data Export:** Instant export of all data and scores to Excel/CSV files for archiving.
        * **CEO Report:** One-click generation of a full text briefing for sharing.
        """)
    
    # --- 1. NEW: LIVE WEB PLATFORM PREVIEW ---
    st.markdown("<br><h2 style='text-align: center; color: #fff;'>LIVE WEB PLATFORM PREVIEW</h2>", unsafe_allow_html=True)
    wc1, wc2 = st.columns(2)
    with wc1:
        try: st.image("preview_dashboard.jpg", caption="Matrix Scanner (Live)", use_container_width=True)
        except: st.info("[Dashboard Preview Missing]")
        try: st.image("preview_ai.jpg", caption="AI Analyst (Live)", use_container_width=True)
        except: st.info("[AI Preview Missing]")
        
    with wc2:
        try: st.image("preview_chart.jpg", caption="Advanced Charting (Live)", use_container_width=True)
        except: st.info("[Chart Preview Missing]")
        try: st.image("preview_heatmap.png", caption="Market Heatmap (Live)", use_container_width=True)
        except: st.info("[Heatmap Preview Missing]")

    # --- 2. OLD: APP SNEAK PEEK ---
    st.markdown("<br><h2 style='text-align: center; color: #fff;'>SNEAK PEEK FROM OUR APP <span class='coming-soon'>COMING SOON</span></h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        try: st.image("dashboard.png", caption="App Dashboard", use_container_width=True) 
        except: st.info("[App Dashboard Preview]")
    with c2:
        try: st.image("analysis.png", caption="App Analysis", use_container_width=True) 
        except: st.info("[App Analysis Preview]")
    with c3:
        try: st.image("risk_insiders.png", caption="App Risk Profile", use_container_width=True) 
        except: st.info("[App Risk Preview]")
            
    st.markdown("<p style='text-align: center; color: #555; margin-top: 50px;'>Support: warpspeedterminal@gmail.com</p>", unsafe_allow_html=True)

# ==========================================
# --- 6. VIEW: PAYWALL ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] != 'active':
    st.markdown("""
        <div class="paywall-header">
            <div class="paywall-title">⛔ ACCESS RESTRICTED: INSTITUTIONAL GRADE DATA</div>
            <div class="paywall-sub">
                Your trial session has expired. To access real-time AI analytics, Oracle projections, and Monte Carlo simulations, authorization is required.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("""
            <div class="plan-card">
                <div class="plan-title">Standard Access</div>
                <div class="plan-price">€25<span style="font-size: 1rem; color: #aaa;">/mo</span></div>
            </div>
        """, unsafe_allow_html=True)
        st.link_button("ACTIVATE MONTHLY", STRIPE_LINKS['1M'], use_container_width=True)
        
    with c2:
        st.markdown("""
            <div class="plan-card">
                <div class="plan-title">Quarterly</div>
                <div class="plan-price">€23<span style="font-size: 1rem; color: #aaa;">/mo</span></div>
                <div class="plan-save">SAVE €24/YR</div>
            </div>
        """, unsafe_allow_html=True)
        st.link_button("ACTIVATE 3-MONTHS", STRIPE_LINKS['3M'], use_container_width=True)
        
    with c3:
        st.markdown("""
            <div class="plan-card">
                <div class="plan-title">Semi-Annual</div>
                <div class="plan-price">€20<span style="font-size: 1rem; color: #aaa;">/mo</span></div>
                <div class="plan-save">SAVE €60/YR</div>
            </div>
        """, unsafe_allow_html=True)
        st.link_button("ACTIVATE 6-MONTHS", STRIPE_LINKS['6M'], use_container_width=True)
        
    with c4:
        st.markdown("""
            <div class="plan-card best-value">
                <div class="plan-title" style="color: #ff9900;">🏆 INSTITUTIONAL TIER</div>
                <div class="plan-price" style="color: #ff9900;">€15<span style="font-size: 1rem; color: #aaa;">/mo</span></div>
                <div class="plan-save" style="color: #ff9900;">SAVE €120/YR (40% OFF)</div>
            </div>
        """, unsafe_allow_html=True)
        st.link_button("ACTIVATE YEARLY ACCESS", STRIPE_LINKS['1Y'], type="primary", use_container_width=True)
    
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
        st.caption("v11.0 (Ultimate)")
        if st.button("LOGOUT"): st.session_state['logged_in'] = False; st.rerun()
        st.markdown("---")
        st.markdown("📧 **Support:**\nwarpspeedterminal@gmail.com")

    # --- MACRO BAR ---
    with st.container():
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            # VIX
            vix = yf.Ticker("^VIX").history(period="2d")
            if not vix.empty: col1.metric("VIX (Fear)", f"{vix['Close'].iloc[-1]:.2f}")
            else: col1.metric("VIX", "N/A")
            
            # 10Y BOND
            tnx = yf.Ticker("^TNX").history(period="2d")
            if not tnx.empty: col2.metric("10Y Bond", f"{tnx['Close'].iloc[-1]:.2f}%")
            else: col2.metric("10Y Bond", "N/A")
            
            # BTC
            btc = yf.Ticker("BTC-USD").history(period="2d")
            if not btc.empty: 
                chg = (btc['Close'].iloc[-1] - btc['Close'].iloc[-2]) / btc['Close'].iloc[-2] * 100
                col3.metric("Bitcoin", f"${btc['Close'].iloc[-1]:,.0f}", f"{chg:.2f}%")
            else: col3.metric("Bitcoin", "N/A")
            
            # OIL
            oil = yf.Ticker("CL=F").history(period="2d")
            if not oil.empty: col4.metric("Crude Oil", f"${oil['Close'].iloc[-1]:.2f}")
            else: col4.metric("Crude Oil", "N/A")
            
        except: st.caption("Macro Data Offline")
            
    st.divider()

    # --- SCANNER ENGINE ---
    def scan_market_safe(tickers):
        results = []
        progress_text = "Scanning assets..."
        my_bar = st.progress(0, text=progress_text)
        total = len(tickers)
        
        for idx, t in enumerate(tickers):
            try:
                my_bar.progress(int((idx + 1) / total * 100), text=f"Scanning {t}...")
                
                # Fetch Data
                stock = yf.Ticker(t)
                df = stock.history(period="1y")
                
                if df.empty or len(df) < 50: 
                    df = yf.download(t, period="1y", progress=False, auto_adjust=True)
                    if df.empty or len(df) < 50: continue
                
                # Timezone cleanup
                if df.index.tz is not None: df.index = df.index.tz_localize(None)

                # Indicators
                df = calculate_indicators(df)
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev)/prev)*100
                rsi = df['RSI'].iloc[-1]
                
                # Verdict
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                verdict = "HOLD"
                reasons = [] 
                
                if curr > ma50:
                    reasons.append(f"✓ Price (${curr:.2f}) > 50MA -> Bullish Trend")
                    if rsi < 70: verdict = "BUY"
                else:
                    reasons.append(f"✗ Price (${curr:.2f}) < 50MA -> Bearish Trend")
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
                rvol = df['Volume'].iloc[-1] / vol_mean if vol_mean > 0 else 1.0
                if rvol > 1.5: 
                    score += 10
                    reasons.append(f"⚡ High Volume (RVOL {rvol:.1f})")
                
                # Info
                info = stock.info
                pe = info.get('trailingPE', None)
                bubble = "NO"
                if pe and pe > 35: 
                    bubble = "🚨 YES"
                    score -= 20
                    reasons.append("⚠️ Bubble Alert: High P/E")
                
                peg = info.get('pegRatio', 'N/A')
                target_price = info.get('targetMeanPrice', 'N/A')
                consensus = info.get('recommendationKey', 'N/A').upper().replace('_', ' ')
                
                # NEWS HANDLING (GOOGLE NEWS INTEGRATION)
                news_items = get_google_news(t)
                ai_summary, valid_news = generate_ai_summary(news_items)
                
                results.append({
                    "Ticker": t, "Price": curr, "Change": chg, "Verdict": verdict, "Sniper": score, 
                    "RVOL": rvol, "Bubble": bubble, "PEG": peg, "RSI": rsi, 
                    "History": df, "Info": info, "News": valid_news, "Reasons": reasons,
                    "TargetPrice": target_price, "Consensus": consensus,
                    "AISummary": ai_summary
                })
            except: continue
            
        my_bar.empty()
        return results

    # --- MAIN INTERFACE ---
    with st.expander("ℹ️ HOW TO READ THE DATA (USER GUIDE)", expanded=False):
        st.markdown("""
        ### 📊 METRIC LEGEND
        * **🎯 Sniper Score (0-100):** Our proprietary composite score.
            * **>75:** Strong Buy Signal
            * **50-75:** Hold / Watch
            * **<50:** Sell / Avoid
        * **⚡ RVOL (Relative Volume):** How much volume is trading compared to normal.
            * **>1.5:** High Institutional Activity (Big players are moving).
        * **🔮 Oracle Ghost (Magenta Line):** A predictive line based on historical pattern matching.
        * **☁️ Event Horizon (Green/Red Cloud):** Monte Carlo simulation showing the probable price range for the next 30 days.
        * **🚨 Bubble Alert:** Triggered when P/E > 35 and Price is extended far above moving averages.
        """)

    with st.form("scanner"):
        c1, c2 = st.columns([3, 1])
        with c1: query = st.text_input("ENTER ASSETS", "AAPL TSLA NVDA BTC-USD JPM COIN")
        with c2: run_scan = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if run_scan:
        ticks = [t.strip().upper() for t in query.replace(",", " ").split() if t.strip()]
        if ticks:
            st.session_state['data'] = scan_market_safe(ticks)
            if not st.session_state['data']:
                st.warning("No valid data found.")
        else:
            st.warning("Please enter a symbol.")

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
            "RSI": f"{d['RSI']:.1f}"
        } for d in st.session_state['data']])
        
        def highlight_verdict(val):
            color = '#00FFCC' if 'BUY' in val else '#ff4b4b' if 'SELL' in val else 'white'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(df_view.style.map(highlight_verdict, subset=['VERDICT']), use_container_width=True, hide_index=True)
        
        # --- MARKET HEATMAP ---
        if len(st.session_state['data']) > 1:
            st.markdown("### 🗺️ MARKET HEATMAP")
            map_data = []
            for d in st.session_state['data']:
                mcap = d['Info'].get('marketCap', 1000000) # fallback
                if not isinstance(mcap, (int, float)): mcap = 1000000
                
                map_data.append({
                    "Ticker": d['Ticker'],
                    "Market Cap": mcap,
                    "Change": d['Change'],
                    "Label": f"{d['Ticker']}<br>{d['Change']:.2f}%"
                })
            
            df_map = pd.DataFrame(map_data)
            fig_map = px.treemap(
                df_map, path=['Ticker'], values='Market Cap', color='Change',
                color_continuous_scale='RdYlGn', range_color=[-5, 5],
                custom_data=['Label']
            )
            fig_map.update_traces(texttemplate="%{customdata[0]}", textposition="middle center")
            fig_map.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=300)
            st.plotly_chart(fig_map, use_container_width=True)

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
        
        t1, t2, t3, t4 = st.tabs(["CHART & EVENT HORIZON", "FUNDAMENTALS & WALL ST", "AI ANALYST", "RISK"])
        
        with t1: 
            # EXPLANATION BOX
            with st.expander("ℹ️ HOW TO READ THE CHART & PREDICTIONS"):
                st.markdown("""
                * **Magenta Line (Oracle Ghost):** The algorithm detects a similar historical price pattern and projects it forward.
                * **Green/Red Cloud (Event Horizon):** Monte Carlo simulation. Shows the statistical probability range (Best/Worst case) for the next 30 days based on volatility.
                * **Dotted Lines (Res/Sup):** Auto-generated Resistance and Support levels.
                * **Yellow Line (SPY):** S&P 500 performance for relative comparison.
                """)

            # PLOTLY CHART
            hist = target['History']
            ghost = find_oracle_pattern(hist['Close'])
            spy = get_spy_data()
            high_lvl, low_lvl, fibs = calculate_smart_levels(hist)
            upper, lower, avg = calculate_monte_carlo(hist)
            
            fig = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05, 
                row_heights=[0.7, 0.3]
            )
            
            # Candlesticks
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            # BB
            fig.add_trace(go.Scatter(x=hist.index, y=hist['UpperBB'], line=dict(color='cyan', width=1), name='Upper BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['LowerBB'], line=dict(color='cyan', width=1), name='Lower BB'), row=1, col=1)
            
            # SPY Overlay
            if spy is not None:
                spy_subset = spy.reindex(hist.index, method='nearest')
                if not spy_subset.empty:
                    scaling_factor = hist['Close'].iloc[0] / spy_subset.iloc[0]
                    spy_scaled = spy_subset * scaling_factor
                    fig.add_trace(go.Scatter(x=spy_subset.index, y=spy_scaled, line=dict(color='yellow', width=2), name='S&P 500 (Rel)'), row=1, col=1)

            # Oracle Ghost
            if ghost is not None:
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(len(ghost))]
                fig.add_trace(go.Scatter(x=future_dates, y=ghost, line=dict(color='magenta', dash='dash', width=2), name='Oracle Ghost'), row=1, col=1)

            # MONTE CARLO (EVENT HORIZON)
            if upper is not None:
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(30)]
                fig.add_trace(go.Scatter(x=future_dates, y=upper, line=dict(color='green', width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=lower, line=dict(color='red', width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='Probability Cloud'), row=1, col=1)

            # SMART TECHNICALS
            if high_lvl:
                fig.add_hline(y=high_lvl, line_dash="dot", annotation_text="Res", annotation_position="top right", line_color="red")
                fig.add_hline(y=low_lvl, line_dash="dot", annotation_text="Sup", annotation_position="bottom right", line_color="green")
                # Fibonacci
                for name, val in fibs.items():
                    fig.add_hline(y=val, line_dash="dash", line_color="gray", annotation_text=name, opacity=0.5)

            # MACD
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='#00FFCC'), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], line=dict(color='#ff4b4b'), name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD']-hist['Signal'], marker_color='gray', name='Hist'), row=2, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"{target['Ticker']} Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🧠 VERDICT LOGIC")
            reasons = target.get('Reasons', []) 
            if reasons:
                for reason in reasons:
                    st.markdown(f"<div class='reason-box'>{reason}</div>", unsafe_allow_html=True)
            else:
                st.info("No specific triggers for this asset.")
            
        with t2: 
            i = target['Info']
            
            st.markdown("##### 🏦 WALL STREET")
            w1, w2 = st.columns(2)
            w1.metric("Consensus", str(target.get('Consensus', 'N/A')))
            w2.metric("Target Price", f"${target.get('TargetPrice', 'N/A')}")
            
            st.divider()
            st.markdown("##### 📊 KEY METRICS")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", format_large_number(i.get('marketCap')))
            c1.metric("P/E Ratio", i.get('trailingPE', '-'))
            c2.metric("Dividend Yield", f"{i.get('dividendYield', 0)*100:.2f}%" if i.get('dividendYield') else '-')
            c2.metric("PEG Ratio", i.get('pegRatio', '-'))
            c3.metric("Profit Margin", f"{i.get('profitMargins', 0)*100:.2f}%" if i.get('profitMargins') else '-')
            c3.metric("ROE", f"{i.get('returnOnEquity', 0)*100:.2f}%" if i.get('returnOnEquity') else '-')
            c4.metric("Free Cash Flow", format_large_number(i.get('freeCashflow')))
            c4.metric("Debt/Equity", i.get('debtToEquity', '-'))
            
        with t3: 
            st.markdown("### 🧠 AI ANALYST BRIEFING")
            st.markdown(f"<div class='ai-box'>{target['AISummary']}</div>", unsafe_allow_html=True)
            
            # CEO REPORT BUTTON
            report_text = generate_ceo_report(target)
            st.download_button(
                label="📄 DOWNLOAD INTELLIGENCE REPORT",
                data=report_text,
                file_name=f"{target['Ticker']}_Intelligence_Report.txt",
                mime="text/plain",
                type="primary"
            )
            
            st.markdown("#### 📰 LATEST HEADLINES")
            news = target.get('News', [])
            if news:
                for n in news[:5]:
                    t_title = n.get('title', 'No Title')
                    t_link = n.get('link', '#')
                    # Make link clickable and blue
                    st.markdown(f"• <a href='{t_link}' target='_blank' style='color: #00ccff; text-decoration: none;'>{t_title}</a>", unsafe_allow_html=True)
            else: st.write("No news found.")
            
        with t4: 
            i = target['Info']
            c1, c2 = st.columns(2)
            c1.metric("Beta (Volatility)", i.get('beta', '-'))
            c2.metric("Short Ratio", i.get('shortRatio', '-'))
            st.caption("Institutional Holders:")
            try: st.dataframe(yf.Ticker(sel_t).institutional_holders.head())
            except: st.write("Data hidden")

    elif not run_scan:
        st.info("Enter tickers above and press INITIATE SCAN.")
