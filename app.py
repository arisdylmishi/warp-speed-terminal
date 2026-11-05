import streamlit as st
from supabase import create_client, Client
import hashlib
import bcrypt
import os
import yfinance as yf
from yahooquery import Ticker as YQTicker
import finnhub
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
import secrets
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

# ==========================================
# --- 1. CONFIGURATION & CYBERPUNK THEME ---
# ==========================================
st.set_page_config(
    page_title="Warp Speed Terminal", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- THEME INJECTION ---
st.markdown("""
    <style>

        /* GLOBAL VARS */
        :root {
            --bg: #050505;
            --primary: #00ff41; /* Neon Green */
            --secondary: #00d4ff; /* Cyan */
            --text: #e0e0e0;
            --border: rgba(0, 255, 65, 0.4);
            --surface: #0a0a0a;
        }

        /* MAIN APP CONTAINER */
        .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace;
        }

        /* FONT FIX */
        .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, .stButton, .stTextInput, label, .stTable {
            font-family: ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace !important;
        }

        /* SCANLINE OVERLAY */
        .stApp::before {
            content: " ";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), 
                        linear-gradient(90deg, rgba(255,0,0,0.03), rgba(0,255,0,0.01), rgba(0,0,255,0.03));
            z-index: 9999;
            background-size: 100% 3px, 3px 100%;
            pointer-events: none;
        }

        /* TYPOGRAPHY */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary) !important;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
            letter-spacing: 1px;
        }

        /* INPUT FIELDS */
        .stTextInput input {
            background-color: #000 !important;
            color: var(--primary) !important;
            border: 1px solid var(--primary) !important;
            border-radius: 0px !important;
        }
        .stTextInput input:focus {
            box-shadow: 0 0 10px var(--primary) !important;
        }

        /* BUTTONS */
        .stButton>button {
            width: 100%;
            border-radius: 0px;
            font-weight: 700;
            height: 3em;
            text-transform: uppercase;
            border: 1px solid var(--primary);
            background-color: rgba(0, 255, 65, 0.1);
            color: var(--primary);
            transition: all 0.3s;
            letter-spacing: 2px;
        }
        .stButton>button:hover {
            background-color: var(--primary);
            color: #000;
            box-shadow: 0 0 20px var(--primary);
        }

        /* METRICS */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: var(--secondary) !important; 
            font-weight: bold;
            text-shadow: 0 0 10px var(--secondary);
        }
        div[data-testid="stMetricLabel"] {
            color: #888 !important;
            font-size: 0.8rem !important;
        }

        /* EXPANDERS & BOXES */
        .streamlit-expanderHeader {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--primary) !important;
            border-radius: 0px !important;
        }
        
        .ai-box {
            background-color: rgba(0, 212, 255, 0.05);
            padding: 15px;
            border-left: 4px solid var(--secondary);
            margin-bottom: 10px;
            color: #eee;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .reason-box {
            background-color: rgba(0, 255, 65, 0.05); 
            padding: 10px; 
            border-left: 3px solid var(--primary); 
            margin-bottom: 5px;
            font-size: 0.9em;
            border: 1px solid rgba(0, 255, 65, 0.2);
        }
        
        /* TABLE FIXES */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            background-color: #050505 !important;
        }
        [data-testid="stDataFrame"] * {
            color: var(--text) !important;
            background-color: #050505 !important;
        }
        table {
            color: var(--text) !important;
            background-color: #050505 !important;
            border-collapse: collapse !important;
            width: 100%;
            font-family: ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace !important;
        }
        th {
            background-color: #111 !important;
            color: var(--primary) !important;
            border-bottom: 1px solid var(--primary) !important;
            padding: 10px !important;
        }
        td {
            border-bottom: 1px solid #333 !important;
            padding: 8px !important;
        }

        /* PAYWALL CARDS */
        .plan-card {
            border: 1px solid var(--primary);
            background-color: rgba(20, 20, 20, 0.8);
            padding: 30px;
            text-align: center;
            height: 100%;
            transition: 0.3s;
            margin-bottom: 20px;
        }
        .plan-card:hover {
            box-shadow: 0 0 25px rgba(0, 255, 65, 0.2);
            transform: translateY(-5px);
            border-color: var(--secondary);
        }
        .plan-title {
            color: var(--primary);
            font-size: 1.4rem;
            font-weight: bold;
            margin-bottom: 15px;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border);
            padding-bottom: 15px;
            letter-spacing: 2px;
        }
        .plan-price {
            font-size: 2.5rem;
            color: var(--secondary);
            font-family: ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace;
            font-weight: bold;
            text-shadow: 0 0 10px var(--secondary);
            margin: 20px 0;
        }
        .best-value {
            border: 2px solid var(--secondary);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
            background: linear-gradient(180deg, rgba(0,212,255,0.05) 0%, rgba(0,0,0,0) 100%);
        }
        .feature-list {
            text-align: left;
            margin: 20px 0;
            color: #aaa;
            font-size: 0.9rem;
        }
        .feature-item {
            margin-bottom: 8px;
        }
        .save-badge {
            background-color: var(--primary);
            color: black;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            margin-left: 10px;
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

def stripe_link(plan_key):
    base = STRIPE_LINKS[plan_key]
    email = st.session_state.get('user_email', '')
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}client_reference_id={quote(email)}"

PRIVACY_POLICY_TEXT = """
**Last updated: October 2026**

Warp Speed Terminal ("we", "us", "the Service") respects your privacy. This policy explains what data we collect, why, and what rights you have over it.

**1. What We Collect**
- Account data: your email address and a securely hashed password (we never store your plain-text password).
- Subscription data: your subscription status, trial usage, and expiry date.
- Payment data: we do not collect or store your card details. Payments are processed entirely by Stripe, and we only receive confirmation that a payment succeeded, plus the email you used at checkout.

**2. Why We Collect It**
- To create and secure your account.
- To grant and track trial and subscription access.
- To send essential account emails (e.g. email verification).
- To provide customer support if you contact us.

**3. Who We Share It With**
We use the following third-party services to operate the Service: Supabase (database hosting), Stripe (payment processing), Resend (transactional email), Finnhub / Yahoo Finance (market data), and Railway (app hosting). We do not sell your data, and we do not use it for advertising.

**4. Cookies**
The Service uses only strictly necessary session cookies required to keep you logged in. We do not use tracking, analytics, or advertising cookies.

**5. How Long We Keep Your Data**
We retain your account data for as long as your account is active. If you ask us to delete your account, we will delete your personal data within a reasonable time, except where legally required to retain certain records.

**6. Your Rights**
You may request a copy of your data, request correction, or request deletion at any time by emailing support@warpspeedterminal.com. EU/EEA users may also lodge a complaint with their local data protection authority.

**7. Children**
The Service is not directed at, and is not intended for use by, anyone under 18.

**8. Contact**
support@warpspeedterminal.com
"""

TERMS_OF_SERVICE_TEXT = """
**Last updated: October 2026**

By creating an account or using Warp Speed Terminal, you agree to these Terms.

**1. Eligibility**
You must be at least 18 years old to use the Service or hold a paid subscription.

**2. Not Financial Advice**
All data, scores, signals, and verdicts shown are generated algorithmically for informational purposes only and do not constitute financial, investment, or trading advice. Trading involves risk, including loss of principal. Always do your own research and consult a licensed financial advisor.

**3. Accuracy of Data**
The Service aggregates data from third-party sources. We do not guarantee any data, score, or signal is complete, accurate, or up to date. The Service is provided "as is."

**4. Free Trial**
New accounts may be granted a limited free trial, restricted in duration and number of scans. We may change or discontinue it at any time.

**5. Subscriptions & Billing**
Paid subscriptions are billed via Stripe on a recurring basis. By subscribing, you authorize recurring charges until you cancel.

**6. Refunds**
General policy: refunds are issued at our discretion — contact support@warpspeedterminal.com. EU/EEA customers have a statutory 14-day cancellation right; by confirming immediate access at signup, you acknowledge this right may end once the Service has been fully performed (e.g. your trial or subscription access has been used).

**7. Acceptable Use**
You agree not to circumvent trial limits, scrape or resell data from the Service, or use it unlawfully. We may suspend accounts that violate these terms.

**8. Limitation of Liability**
To the maximum extent permitted by law, we are not liable for trading or financial losses arising from your use of the Service.

**9. Governing Law**
These Terms are governed by the laws of Greece, except where mandatory consumer protection laws of your own country apply.

**10. Contact**
support@warpspeedterminal.com
"""

def safe_float(val):
    """Titanium Shield: Forces strings/mixed data to float to prevent app crash"""
    if val is None: return 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        val = val.strip().replace(',', '').replace('%', '').replace('$', '')
        if val == '-' or val == '' or val == 'N/A' or val == 'nan': return 0.0
        multi = 1.0
        if val.endswith('B'): multi = 1e9; val = val[:-1]
        elif val.endswith('M'): multi = 1e6; val = val[:-1]
        elif val.endswith('K'): multi = 1e3; val = val[:-1]
        elif val.endswith('T'): multi = 1e12; val = val[:-1]
        try: return float(val) * multi
        except: return 0.0
    return 0.0

def format_large_number(num):
    if not num or isinstance(num, str): return "N/A"
    try:
        num = safe_float(num)
        if num >= 1e12: return f"${num/1e12:.2f}T"
        if num >= 1e9: return f"${num/1e9:.2f}B"
        if num >= 1e6: return f"${num/1e6:.2f}M"
        return f"${num:.2f}"
    except: return "N/A"

def get_google_news(ticker):
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
        return "⚠️ **ANALYST NOTE:** Real-time news feed is quiet. Exercise caution.", valid_news
            
    words = re.findall(r'\w+', text_corpus.lower())
    ignore = ['the', 'a', 'to', 'of', 'in', 'and', 'for', 'on', 'with', 'at', 'is', 'stock', 'market']
    filtered = [w for w in words if w not in ignore and len(w) > 4]
    common = Counter(filtered).most_common(5)
    keywords = [c[0].upper() for c in common]
    
    blob = TextBlob(text_corpus)
    pol = blob.sentiment.polarity
    if pol > 0.1: tone = "BULLISH 🐂"
    elif pol < -0.1: tone = "BEARISH 🐻"
    else: tone = "NEUTRAL / MIXED ⚖️"
    
    summary = f"AI ANALYST SCAN: **{tone}**\nBased on recent headlines. "
    if keywords: summary += f"Key Topics: {', '.join(keywords)}. "
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
Market Cap:   {format_large_number(t['Info'].get('marketCap', 0))}
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
    if isinstance(hist.columns, pd.MultiIndex): 
        try: hist.columns = hist.columns.get_level_values(0)
        except: pass
    if 'Close' not in hist.columns: return hist

    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 0.001)
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

@st.cache_data(ttl=3600)
def get_spy_data():
    try:
        spy = yf.Ticker("SPY").history(period="1y")
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        spy = spy['Close']
        spy.index = spy.index.tz_localize(None)
        return spy
    except: return None

# ==============================================================
# --- OMNI-CHANNEL DATA AGGREGATOR (6-LAYER FETCHING) ---
# ==============================================================

def fetch_finviz_data(ticker):
    """Scrapes Finviz table when APIs fail."""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = requests.get(url, headers=headers, timeout=5)
        
        if req.status_code != 200: return {}
        
        dfs = pd.read_html(req.text, attrs={'class': 'snapshot-table2'})
        if not dfs: return {}
        
        df = dfs[0]
        data = {}
        for i in range(0, len(df.columns), 2):
            for index, row in df.iterrows():
                key = str(row[i])
                val = str(row[i+1])
                data[key] = val
        
        # Parse Finviz Recom (1=Strong Buy, 5=Sell)
        recom_num = safe_float(data.get('Recom', '3.0'))
        recom_text = "Hold"
        if recom_num <= 1.5: recom_text = "Strong Buy"
        elif recom_num <= 2.5: recom_text = "Buy"
        elif recom_num >= 4.5: recom_text = "Sell"
        elif recom_num >= 3.5: recom_text = "Underperform"
        
        # Extract Financial Health metrics specifically
        return {
            'marketCap': safe_float(data.get('Market Cap')),
            'trailingPE': safe_float(data.get('P/E')),
            'pegRatio': safe_float(data.get('PEG')),
            'beta': safe_float(data.get('Beta')),
            'dividendYield': safe_float(data.get('Dividend %')) / 100, 
            'profitMargins': safe_float(data.get('Profit Margin')) / 100,
            'shortRatio': safe_float(data.get('Short Float')) / 100,
            'targetMeanPrice': safe_float(data.get('Target Price')),
            'recommendationKey': recom_text,
            'totalRevenue': safe_float(data.get('Sales')), 
            'totalCash': safe_float(data.get('Cash/sh')) * safe_float(data.get('Shs Out')), 
            'ebitda': safe_float(data.get('Income')) 
        }
    except: return {}

def fetch_finnhub_consensus(ticker):
    """Gets Consensus from Finnhub Free Tier"""
    try:
        finnhub_key = st.secrets.get("finnhub", {}).get("api_key", "")
        if not finnhub_key: return "N/A"
        client = finnhub.Client(api_key=finnhub_key)
        
        if "BTC" in ticker: return "N/A (Crypto)"
        
        res = client.recommendation_trends(ticker)
        if not res: return "N/A"
        
        latest = res[0]
        counts = {
            'Strong Buy': latest.get('strongBuy', 0),
            'Buy': latest.get('buy', 0),
            'Hold': latest.get('hold', 0),
            'Sell': latest.get('sell', 0)
        }
        winner = max(counts, key=counts.get)
        return winner
    except: return "N/A"

def fetch_stockanalysis_holdings(ticker):
    """PRIMARY HOLDINGS SOURCE: StockAnalysis.com Scraper"""
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/ownership/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = requests.get(url, headers=headers, timeout=5)
        
        if req.status_code != 200: return None
        
        dfs = pd.read_html(req.text)
        if dfs:
            df = dfs[0]
            if 'Owner' in df.columns:
                df = df.rename(columns={'Owner': 'Holder', 'Shares': 'Shares', 'Date': 'Date Reported', 'Value': 'Value'})
                return df.head(10)
        return None
    except: return None

def fetch_marketwatch_holdings(ticker):
    """BACKUP HOLDINGS SOURCE: MarketWatch Scraper"""
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}/holdings"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        dfs = pd.read_html(r.text)
        for df in dfs:
            if 'Name' in df.columns:
                return df[['Name', 'Recent Activity']].head(10)
        return None
    except: return None

def fetch_insiders_stockanalysis(ticker):
    """PRIMARY INSIDER SOURCE: StockAnalysis"""
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/insider-trading/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        dfs = pd.read_html(r.text)
        if dfs: return dfs[0].head(10)
        return None
    except: return None

def fetch_finviz_insiders(ticker):
    """BACKUP INSIDER SOURCE: Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        dfs = pd.read_html(r.text)
        if len(dfs) > 5:
            df = dfs[-1]
            if 'Insider Trading' in str(df.columns) or 'SEC Form 4' in str(df.columns) or 'Date' in str(df.columns): return df.head(10)
        return None
    except: return None

def fetch_options_chain(ticker):
    """DERIVATIVES: Puts/Calls, Volatility, Max Pain"""
    try:
        tk = yf.Ticker(ticker)
        # Get nearest expiry
        if not tk.options: return None
        expiry = tk.options[0]
        opt = tk.option_chain(expiry)
        
        calls = opt.calls
        puts = opt.puts
        
        # Metrics
        put_vol = puts['volume'].sum()
        call_vol = calls['volume'].sum()
        pcr = put_vol / call_vol if call_vol > 0 else 0
        
        return {
            'pcr': pcr,
            'expiry': expiry,
            'calls': calls.sort_values('openInterest', ascending=False).head(5)[['strike', 'lastPrice', 'volume', 'openInterest']],
            'puts': puts.sort_values('openInterest', ascending=False).head(5)[['strike', 'lastPrice', 'volume', 'openInterest']]
        }
    except: return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_data(ticker, period="1y"):
    df = None
    info = {}
    
    # --- 1. PRICE HISTORY (YFinance -> Finnhub) ---
    try:
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        p = period.lower() if period.lower() in valid_periods else "1y"
        
        df = yf.download(ticker, period=p, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except:
        pass 

    if df is None or df.empty:
        try:
            finnhub_key = st.secrets.get("finnhub", {}).get("api_key", "")
            if finnhub_key:
                client = finnhub.Client(api_key=finnhub_key)
                lookup = "BINANCE:BTCUSDT" if "BTC" in ticker else ticker
                end = int(time.time())
                start = end - (365 * 24 * 60 * 60)
                res = client.stock_candles(lookup, 'D', start, end)
                if res.get('s') == 'ok':
                    df = pd.DataFrame({
                        'Open': res['o'], 'High': res['h'], 'Low': res['l'], 
                        'Close': res['c'], 'Volume': res['v']
                    })
                    df.index = pd.to_datetime(res['t'], unit='s')
        except: pass

    if df is None or df.empty: return None, None

    # --- 2. FUNDAMENTALS (Cascade: YahooQuery -> Finviz -> Finnhub) ---
    
    # 2.1 YahooQuery (Primary)
    try:
        yq = YQTicker(ticker)
        d = yq.get_modules('summaryDetail defaultKeyStatistics financialData price assetProfile calendarEvents').get(ticker, {})
        if isinstance(d, dict) and 'summaryDetail' in d:
             info = {
                'marketCap': d.get('price', {}).get('marketCap'),
                'trailingPE': d.get('summaryDetail', {}).get('trailingPE'),
                'pegRatio': d.get('defaultKeyStatistics', {}).get('pegRatio'),
                'beta': d.get('summaryDetail', {}).get('beta'),
                'shortRatio': d.get('defaultKeyStatistics', {}).get('shortRatio'),
                'targetMeanPrice': d.get('financialData', {}).get('targetMeanPrice'),
                'recommendationKey': d.get('financialData', {}).get('recommendationKey', 'Hold'),
                'longBusinessSummary': d.get('assetProfile', {}).get('longBusinessSummary', 'Description Unavailable'),
                'sector': d.get('assetProfile', {}).get('sector', 'N/A'),
                'industry': d.get('assetProfile', {}).get('industry', 'N/A'),
                # NEW DATA POINTS
                'fiftyTwoWeekHigh': d.get('summaryDetail', {}).get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': d.get('summaryDetail', {}).get('fiftyTwoWeekLow'),
                'totalCash': d.get('financialData', {}).get('totalCash'),
                'totalDebt': d.get('financialData', {}).get('totalDebt'),
                'totalRevenue': d.get('financialData', {}).get('totalRevenue'),
                'ebitda': d.get('financialData', {}).get('ebitda'),
                'earningsDate': d.get('calendarEvents', {}).get('earnings', {}).get('earningsDate', ['N/A'])
            }
    except: pass

    # 2.2 Finviz (Secondary - Fill Gaps)
    if not info.get('trailingPE') or not info.get('totalRevenue'):
        finviz_data = fetch_finviz_data(ticker)
        if finviz_data:
            for k, v in finviz_data.items():
                if k not in info or not info[k]:
                    info[k] = v

    # 2.3 Finnhub (Tertiary - Backup & Crypto)
    if not info.get('recommendationKey') or info.get('recommendationKey') == 'N/A':
        info['recommendationKey'] = fetch_finnhub_consensus(ticker)

    try:
        finnhub_key = st.secrets.get("finnhub", {}).get("api_key", "")
        if finnhub_key:
            client = finnhub.Client(api_key=finnhub_key)
            prof = client.company_profile2(symbol=ticker)
            quote = client.quote(ticker)
            
            if not info.get('marketCap'):
                info['marketCap'] = prof.get('marketCapitalization', 0) * 1_000_000
            if not info.get('sector'):
                info['sector'] = prof.get('finnhubIndustry', '-')
            if not info.get('longBusinessSummary') or info.get('longBusinessSummary') == 'Description Unavailable':
                info['longBusinessSummary'] = f"A company in the {prof.get('finnhubIndustry', 'Market')} sector."
            
            info['currentPrice'] = quote.get('c', 0)
    except: pass

    return df, info

def scan_market_safe(tickers, period="1y"):
    results = []
    my_bar = st.progress(0)
    total = len(tickers)
    
    for idx, t in enumerate(tickers):
        try:
            my_bar.progress(int((idx + 1) / total * 100))
            df, info = fetch_ticker_data(t, period)
            
            if df is None or df.empty: continue
            
            df = calculate_indicators(df)
            if 'Close' not in df.columns: continue

            curr = df['Close'].iloc[-1]
            if not info.get('currentPrice'): info['currentPrice'] = curr

            prev = df['Close'].iloc[-2]
            chg = ((curr - prev)/prev)*100
            
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            if pd.isna(ma50): ma50 = curr

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
            
            score = 50
            if verdict == "BUY": score += 20
            if verdict == "STRONG BUY": score += 35
            if rsi < 30: score += 20
            
            vol_mean = df['Volume'].rolling(50).mean().iloc[-1]
            current_vol = df['Volume'].iloc[-1]
            rvol = current_vol / vol_mean if (vol_mean > 0 and not pd.isna(vol_mean)) else 1.0
            
            if rvol > 1.5: 
                score += 10
                reasons.append(f"⚡ High Volume (RVOL {rvol:.1f})")
            
            pe = safe_float(info.get('trailingPE'))
            bubble = "NO"
            if pe > 35: 
                bubble = "🚨 YES"
                score -= 20
                reasons.append("⚠️ Bubble Alert: High P/E")
            
            peg = safe_float(info.get('pegRatio'))
            target_price = safe_float(info.get('targetMeanPrice'))
            rec_key = str(info.get('recommendationKey', 'N/A')).title()
            if rec_key == 'None' or rec_key == '0': rec_key = "N/A"
            
            news_items = get_google_news(t)
            ai_summary, valid_news = generate_ai_summary(news_items)
            
            # --- FETCH DERIVATIVES ---
            deriv = fetch_options_chain(t)
            
            results.append({
                "Ticker": t, "Price": curr, "Change": chg, "Verdict": verdict, "Sniper": score, 
                "RVOL": rvol, "Bubble": bubble, "PEG": peg, "RSI": rsi, 
                "History": df, "Info": info, "News": valid_news, "Reasons": reasons,
                "TargetPrice": target_price, "Consensus": rec_key,
                "AISummary": ai_summary, "Derivatives": deriv
            })
        except Exception:
            continue
        
    my_bar.empty()
    return results

# ==========================================
# --- 3. CLOUD DATABASE (SUPABASE DIRECT) ---
# ==========================================
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_connection()

def _pw_bytes(password):
    return password.encode("utf-8")[:72]

def make_hashes(password):
    return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode("utf-8")

def legacy_sha256(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if hashed_text.startswith("$2"):
        try:
            return bcrypt.checkpw(_pw_bytes(password), hashed_text.encode("utf-8"))
        except Exception:
            return False
    return legacy_sha256(password) == hashed_text
def generate_verify_token():
    return secrets.token_urlsafe(24)

def send_verification_email(email, token):
    api_key = os.environ.get("RESEND_API_KEY", "")
    base_url = os.environ.get("APP_BASE_URL", "https://warpspeedterminal.com")
    if not api_key:
        return False
    verify_link = f"{base_url}/?verify=1&token={token}&email={quote(email)}"
    html = f"""
    <div style="background:#050505;color:#e0e0e0;font-family:monospace;padding:40px;text-align:center;">
      <h1 style="color:#00ff41;letter-spacing:2px;">⚡ WARP SPEED TERMINAL</h1>
      <p style="color:#00d4ff;letter-spacing:1px;">SYSTEM ACCESS PENDING</p>
      <p style="margin:30px 0;">One step left before your trial run begins. Confirm this terminal is yours:</p>
      <a href="{verify_link}" style="display:inline-block;border:1px solid #00ff41;color:#00ff41;padding:12px 28px;text-decoration:none;letter-spacing:2px;">&gt; VERIFY &amp; LAUNCH</a>
      <p style="margin-top:30px;color:#555;font-size:12px;">If you didn't request this, ignore this transmission.</p>
    </div>
    """
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": "Warp Speed Terminal <access@warpspeedterminal.com>",
                "to": [email],
                "subject": "⚡ Verify your terminal access",
                "html": html,
            },
            timeout=10,
        )
        return True
    except Exception:
        return False
def add_user(email, password):
    if not supabase: return False
    hashed_pw = make_hashes(password)
    past_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    current_date = datetime.now().strftime("%Y-%m-%d")
    token = generate_verify_token()
    try:
        supabase.table("users").insert({
            "email": email, "password": hashed_pw, "status": 'unverified',
            "join_date": current_date, "expiry_date": past_date,
            "verified": False, "verify_token": token
        }).execute()
        send_verification_email(email, token)
        return True
    except: return False

def login_user_db(email, password):
    if not supabase: return []
    try:
        res = supabase.table("users").select("*").eq("email", email).execute()
        if not res.data: return []
        row = res.data[0]
        stored = str(row.get("password", ""))
        if not check_hashes(password, stored): return []
        if not stored.startswith("$2"):
            try:
                supabase.table("users").update({"password": make_hashes(password)}).eq("email", email).execute()
            except Exception:
                pass
        return [list(row.values())]
    except Exception:
        return []

def add_subscription_days(email, days_to_add):
    if not supabase: return None
    new_expiry = (datetime.now() + timedelta(days=int(days_to_add))).strftime("%Y-%m-%d")
    supabase.table("users").update({"status": "active", "expiry_date": new_expiry}).eq("email", email).execute()
    return new_expiry

def check_subscription_validity(email, current_expiry_str):
    if email == "admin": return True
    if not current_expiry_str: return False
    if not supabase: return False
    try:
        try:
            expiry_dt = datetime.strptime(current_expiry_str, "%Y-%m-%d %H:%M:%S")
            grace = timedelta(seconds=0)
        except ValueError:
            expiry_dt = datetime.strptime(current_expiry_str, "%Y-%m-%d")
            grace = timedelta(days=1)
        if datetime.now() > expiry_dt + grace:
            supabase.table("users").update({"status": "expired"}).eq("email", email).execute()
            return False
        return True
    except: return False
def check_and_consume_trial_scan(email):
    if email == "admin": return True
    if not supabase: return True
    try:
        res = supabase.table("users").select("is_trial, trial_scans_used").eq("email", email).execute()
        if not res.data: return True
        row = res.data[0]
        if not row.get("is_trial"): return True
        used = row.get("trial_scans_used") or 0
        if used >= 1:
            return False
        supabase.table("users").update({"trial_scans_used": used + 1}).eq("email", email).execute()
        return True
    except Exception:
        return True
# ==========================================
# --- 4. SESSION & AUTH FLOW ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = ""
if 'user_status' not in st.session_state: st.session_state['user_status'] = "expired"
if 'expiry_date' not in st.session_state: st.session_state['expiry_date'] = ""

if "verify" in st.query_params:
    v_token = st.query_params.get("token", "")
    v_email = st.query_params.get("email", "")
    try:
        res = supabase.table("users").select("*").eq("email", v_email).execute()
        if res.data:
            row = res.data[0]
            if row.get("verified"):
                st.info("✅ This terminal is already verified. Please log in.")
            elif row.get("verify_token") and row.get("verify_token") == v_token:
                trial_minutes = int(os.environ.get("TRIAL_MINUTES", "1"))
                new_expiry_dt = datetime.now() + timedelta(minutes=trial_minutes)
                supabase.table("users").update({
                    "verified": True,
                    "status": "active",
                    "expiry_date": new_expiry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "verify_token": None,
                    "is_trial": True,
                    "trial_scans_used": 0
                }).eq("email", v_email).execute()
                st.success(f"⚡ ACCESS GRANTED // Trial active for {trial_minutes} minute(s). Please log in below.")
            else:
                st.error("⛔ Invalid or expired verification link.")
        else:
            st.error("⛔ Invalid or expired verification link.")
    except Exception:
        st.error("⛔ Verification failed. Please try again.")
    st.query_params.clear()
# ==========================================
# --- SECURE PAYMENT VERIFICATION ---
# ==========================================

# ==========================================
# --- 5. VIEW: LANDING PAGE ---
# ==========================================
if not st.session_state['logged_in']:
    st.markdown("""
        <div style='text-align: center; padding: 50px 20px;'>
            <h1 style='color: #00FF41; font-size: 60px; margin-bottom: 10px; text-shadow: 0 0 10px rgba(0,255,65,0.8);'>WARP SPEED TERMINAL</h1>
            <p style='color: #444; font-size: 0.8rem; letter-spacing: 3px; margin-top: -5px;'>EST. 2025</p>
            <p style='font-size: 24px; color: #00d4ff; letter-spacing: 2px;'>UNLEASH THE DATA</p>
        </div>
    """, unsafe_allow_html=True)
    
    if "payment_success" in st.query_params:
        st.success("✅ PAYMENT RECEIVED! Please Log In to complete activation.", icon="🔓")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("### ⚡ SYSTEM ACCESS")
        st.info("Professional analysis synthesizing Technicals, Fundamentals, and AI.")
        
        tab_login, tab_signup = st.tabs(["LOG IN", "REGISTER"])
        with tab_login:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Password", type='password', key="l_pass")
            if st.button("LAUNCH TERMINAL", type="primary"):
                admin_user = st.secrets.get("admin", {}).get("username", "")
                admin_pass = st.secrets.get("admin", {}).get("password", "")
                if admin_user and email == admin_user and password == admin_pass:
                    st.session_state.update({'logged_in': True, 'user_email': "admin", 'expiry_date': "LIFETIME", 'user_status': 'active'})
                    st.rerun()
                else:
                    user = login_user_db(email, password)
                    if user:
                        if user[0][2] == 'unverified':
                            st.warning("📡 ACCESS PENDING // Check your inbox and click the verification link to launch your trial.")
                        else:
                            is_active = check_subscription_validity(user[0][0], user[0][4])
                            st.session_state.update({'logged_in': True, 'user_email': user[0][0], 'expiry_date': user[0][4], 'user_status': 'active' if is_active else 'expired'})
                            st.rerun()
                    else: st.error("Invalid Credentials")
                    
        with tab_signup:
            new_email = st.text_input("New Email", key="s_email")
            new_pass = st.text_input("New Password", type='password', key="s_pass")
            agree_terms = st.checkbox("I am 18+ and agree to the Terms of Service and Privacy Policy. I understand that by requesting immediate access, I may lose my EU 14-day cancellation right once the Service has been used.", key="s_agree")
            if st.button("CREATE ACCOUNT"):
                if not agree_terms:
                    st.warning("⚠️ You must agree to the Terms before creating an account.")
                elif add_user(new_email, new_pass):
                    st.success("📡 TRANSMISSION SENT // Check your inbox to verify access and launch your trial.")
                else:
                    st.error("Email exists.")

    with c2:
        st.video("https://youtu.be/ql1suvTu_ak")
    
    st.divider()
    with st.expander("📖 READ FULL SYSTEM DESCRIPTION (UPDATED V12.0)", expanded=True):
        st.markdown("""
        **Warp Speed Terminal** is a professional analysis platform that synthesizes Technical Analysis, Fundamental Data, and Artificial Intelligence. It is designed to transform chaotic market data into clear, actionable signals, offering features typically found only in institutional-grade terminals.

        #### 🚀 NEW: Titanium Data Engine (V12.0)
        We have integrated a **6-Layer Omni-Channel Aggregator** that pulls data from multiple institutional-grade sources simultaneously. If one source fails, the next one takes over instantly to ensure **zero dead data**.

        #### Detailed Features:

        **1. 🔍 The "X-Ray" Vision (Fundamentals)**
        * **Valuation:** P/E, PEG, Price-to-Sales.
        * **Health:** Cash on Hand, Total Debt, EBITDA, Revenue.
        * **Dividends:** Yield and Payouts.
        * **Company Identity:** Full description, Sector, Industry, and Employee count.

        **2. 🧠 The "Brain" (Analysis)**
        * **The Oracle:** Pattern-matching algorithm that predicts future moves based on history.
        * **The Event Horizon:** Monte Carlo simulation running 1,000 future scenarios.
        * **AI Briefing:** Sentiment analysis reading the news so you don't have to.

        **3. 🕵️ The "Insider" Edge (Institutional Data)**
        * **Hedge Funds:** See who owns the stock (Vanguard, Blackrock).
        * **Insider Trading:** See if the CEO is buying or selling their own shares.
        * **Short Interest:** See if Wall Street is betting against the stock.

        **4. 🔮 The "Wall Street" View**
        * **Consensus:** "Buy/Sell" ratings from top analysts.
        * **Price Targets:** Exact price predictions from Wall St.
        * **Earnings:** Next earnings date countdown.

        **5. 📈 Derivatives & Options (NEW)**
        * **Put/Call Ratio:** Real-time sentiment (Bullish vs Bearish).
        * **Option Chain:** Top active contracts for the next expiry.
        """)
    
    st.markdown("<br><h2 style='text-align: center;'>LIVE WEB PLATFORM PREVIEW</h2>", unsafe_allow_html=True)
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

    # --- SNEAK PEEK FROM OUR APP (RE-ADDED) ---
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
    # ----------------------------------------

    st.markdown("<p style='text-align: center; color: #444; font-size: 0.75rem; max-width: 700px; margin: 15px auto 0;'>⚠️ Not financial advice. All data, scores, and verdicts shown are for informational purposes only and do not constitute investment advice or a recommendation to buy or sell any security. Trading involves risk, including loss of principal. Do your own research and consult a licensed financial advisor before making investment decisions.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        with st.expander("🔒 Privacy Policy"):
            st.markdown(PRIVACY_POLICY_TEXT)
    with pc2:
        with st.expander("📜 Terms of Service"):
            st.markdown(TERMS_OF_SERVICE_TEXT)
# ==========================================
# --- 6. VIEW: PAYWALL (REDESIGNED) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] != 'active':
    st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="color: #00FF41; text-shadow: 0 0 15px rgba(0,255,65,0.6);">⛔ ACCESS RESTRICTED</h1>
            <p style="color: #888; font-size: 1.1rem;">Institutional-grade data requires an active terminal license.</p>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
            <div class="plan-card">
                <div class="plan-title">Monthly</div>
                <div class="plan-price">€25<span style="font-size: 1rem; color: #888;">/mo</span></div>
                <div class="feature-list">
                    <div class="feature-item">✓ Full Terminal Access</div>
                    <div class="feature-item">✓ Real-time AI Analysis</div>
                    <div class="feature-item">✓ Insider Trading Data</div>
                </div>
                <a href="{stripe_link('1M')}" target="_blank">
                    <button style="background: transparent; border: 1px solid #00FF41; color: #00FF41; padding: 10px 20px; width: 100%; font-weight: bold; cursor: pointer;">ACTIVATE</button>
                </a>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
            <div class="plan-card">
                <div class="plan-title">Quarterly</div>
                <div class="plan-price">€23<span style="font-size: 1rem; color: #888;">/mo</span></div>
                <div class="feature-list">
                    <div class="feature-item">✓ Save €24 per year</div>
                    <div class="feature-item">✓ Quarterly Billing</div>
                    <div class="feature-item">✓ Full Access</div>
                </div>
                <a href="{stripe_link('3M')}" target="_blank">
                    <button style="background: transparent; border: 1px solid #00FF41; color: #00FF41; padding: 10px 20px; width: 100%; font-weight: bold; cursor: pointer;">ACTIVATE</button>
                </a>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
            <div class="plan-card">
                <div class="plan-title">Semi-Annual</div>
                <div class="plan-price">€20<span style="font-size: 1rem; color: #888;">/mo</span></div>
                <div class="feature-list">
                    <div class="feature-item">✓ Save €60 per year</div>
                    <div class="feature-item">✓ 6-Month Billing</div>
                    <div class="feature-item">✓ Full Access</div>
                </div>
                <a href="{stripe_link('6M')}" target="_blank">
                    <button style="background: transparent; border: 1px solid #00FF41; color: #00FF41; padding: 10px 20px; width: 100%; font-weight: bold; cursor: pointer;">ACTIVATE</button>
                </a>
            </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
            <div class="plan-card best-value">
                <div class="plan-title" style="color: #00d4ff;">🏆 Yearly</div>
                <div class="plan-price" style="color: #00d4ff;">€15<span style="font-size: 1rem; color: #888;">/mo</span></div>
                <div class="feature-list">
                    <div class="feature-item">✓ <b>40% DISCOUNT</b> applied</div>
                    <div class="feature-item">✓ Priority Support</div>
                    <div class="feature-item">✓ Early Access to V2.0</div>
                </div>
                <a href="{stripe_link('1Y')}" target="_blank">
                    <button style="background: #00d4ff; border: none; color: black; padding: 12px 20px; width: 100%; font-weight: bold; cursor: pointer; box-shadow: 0 0 15px rgba(0,212,255,0.4);">ACTIVATE YEARLY</button>
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><p style='text-align: center; color: #555;'>Secure Payment via Stripe. Cancel Anytime.</p>", unsafe_allow_html=True)
    st.divider()
    if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

# ==========================================
# --- 7. VIEW: THE TERMINAL (LOGGED IN & ACTIVE) ---
# ==========================================
elif st.session_state['logged_in'] and st.session_state['user_status'] == 'active':
    
    with st.sidebar:
        st.title("WARP SPEED")
        st.caption(f"User: {st.session_state['user_email']}")
        st.caption("v12.0 (Titanium)")
        if st.button("LOGOUT"): st.session_state['logged_in'] = False; st.rerun()
        st.markdown("---")
        st.markdown("📧 **Support:**\nsupport@warpspeedterminal.com")
        st.caption("⚠️ Not financial advice. For informational purposes only.")

    # --- MACRO BAR ---
    with st.container():
        try:
            col1, col2, col3, col4 = st.columns(4)
            vix = yf.Ticker("^VIX").history(period="2d")
            if not vix.empty: col1.metric("VIX (Fear)", f"{vix['Close'].iloc[-1]:.2f}")
            else: col1.metric("VIX", "N/A")
            
            tnx = yf.Ticker("^TNX").history(period="2d")
            if not tnx.empty: col2.metric("10Y Bond", f"{tnx['Close'].iloc[-1]:.2f}%")
            else: col2.metric("10Y Bond", "N/A")
            
            btc = yf.Ticker("BTC-USD").history(period="2d")
            if not btc.empty: 
                chg = (btc['Close'].iloc[-1] - btc['Close'].iloc[-2]) / btc['Close'].iloc[-2] * 100
                col3.metric("Bitcoin", f"${btc['Close'].iloc[-1]:,.0f}", f"{chg:.2f}%")
            else: col3.metric("Bitcoin", "N/A")
            
            oil = yf.Ticker("CL=F").history(period="2d")
            if not oil.empty: col4.metric("Crude Oil", f"${oil['Close'].iloc[-1]:.2f}")
            else: col4.metric("Crude Oil", "N/A")
        except: st.caption("Macro Data Offline")
            
    st.divider()

    with st.form("scanner"):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1: query = st.text_input("ENTER ASSETS", "AAPL TSLA NVDA BTC-USD JPM COIN")
        with c2: period = st.selectbox("TIMEFRAME", ["1y", "1mo", "3mo", "6mo", "2y", "5y"])
        with c3: run_scan = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if run_scan:
        if not check_and_consume_trial_scan(st.session_state['user_email']):
            st.warning("⚡ TRIAL SCAN USED // Your free trial includes 1 scan. Upgrade below to run more.")
        else:
            ticks = [t.strip().upper() for t in query.replace(",", " ").split() if t.strip()]
            if ticks:
                st.session_state['data'] = scan_market_safe(ticks, period)
                if not st.session_state['data']:
                    st.warning("No valid data found. If this persists, the data feed is temporarily blocked.")
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
            color = '#00FF41' if 'BUY' in val else '#ff4b4b' if 'SELL' in val else 'white'
            return f'color: {color}; font-weight: bold'
        
        st.table(df_view.style.map(highlight_verdict, subset=['VERDICT']))
        
        # --- MARKET HEATMAP ---
        if len(st.session_state['data']) > 1:
            st.markdown("### 🗺️ MARKET HEATMAP")
            map_data = []
            for d in st.session_state['data']:
                mcap = d['Info'].get('marketCap', 1000000)
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
        
        t1, t2, t3, t4, t5, t6 = st.tabs(["CHART", "FUNDAMENTALS", "AI BRIEF", "RISK", "INSIDERS", "OPTIONS"])
        
        with t1: 
            with st.expander("ℹ️ HOW TO READ THE CHART & PREDICTIONS"):
                st.markdown("""
                * **Magenta Line (Oracle Ghost):** The algorithm detects a similar historical price pattern and projects it forward.
                * **Green/Red Cloud (Event Horizon):** Monte Carlo simulation. Shows the statistical probability range.
                * **Dotted Lines (Res/Sup):** Auto-generated Resistance and Support levels.
                """)

            hist = target['History']
            ghost = find_oracle_pattern(hist['Close'])
            spy = get_spy_data()
            high_lvl, low_lvl, fibs = calculate_smart_levels(hist)
            upper, lower, avg = calculate_monte_carlo(hist)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['UpperBB'], line=dict(color='cyan', width=1), name='Upper BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['LowerBB'], line=dict(color='cyan', width=1), name='Lower BB'), row=1, col=1)
            
            if spy is not None:
                spy_subset = spy.reindex(hist.index, method='nearest')
                if not spy_subset.empty:
                    scaling_factor = hist['Close'].iloc[0] / spy_subset.iloc[0]
                    spy_scaled = spy_subset * scaling_factor
                    fig.add_trace(go.Scatter(x=spy_subset.index, y=spy_scaled, line=dict(color='yellow', width=2), name='S&P 500 (Rel)'), row=1, col=1)

            if ghost is not None:
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(len(ghost))]
                fig.add_trace(go.Scatter(x=future_dates, y=ghost, line=dict(color='magenta', dash='dash', width=2), name='Oracle Ghost'), row=1, col=1)

            if upper is not None:
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(30)]
                fig.add_trace(go.Scatter(x=future_dates, y=upper, line=dict(color='green', width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=lower, line=dict(color='red', width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='Probability Cloud'), row=1, col=1)

            if high_lvl:
                fig.add_hline(y=high_lvl, line_dash="dot", annotation_text="Res", annotation_position="top right", line_color="red")
                fig.add_hline(y=low_lvl, line_dash="dot", annotation_text="Sup", annotation_position="bottom right", line_color="green")

            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='#00FF41'), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], line=dict(color='#ff4b4b'), name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD']-hist['Signal'], marker_color='gray', name='Hist'), row=2, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"{target['Ticker']} Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 52 WEEK RANGE BAR ---
            current = safe_float(target['Info'].get('currentPrice', 0))
            low52 = safe_float(target['Info'].get('fiftyTwoWeekLow', 0))
            high52 = safe_float(target['Info'].get('fiftyTwoWeekHigh', 0))
            
            if low52 > 0 and high52 > 0:
                pct = (current - low52) / (high52 - low52)
                st.write(f"**52-Week Range:** ${low52:.2f} — **${current:.2f}** — ${high52:.2f}")
                st.progress(max(0.0, min(1.0, pct)))

            st.markdown("#### 🧠 VERDICT LOGIC")
            reasons = target.get('Reasons', []) 
            if reasons:
                for reason in reasons:
                    st.markdown(f"<div class='reason-box'>{reason}</div>", unsafe_allow_html=True)
            else:
                st.info("No specific triggers for this asset.")
            
        with t2: 
            i = target['Info']
            
            # --- COMPANY PROFILE SECTION ---
            st.markdown("##### 🏢 COMPANY PROFILE")
            desc = i.get('longBusinessSummary', 'No description available.')
            if len(desc) > 400: desc = desc[:400] + "..."
            st.info(desc)
            
            c1, c2 = st.columns(2)
            c1.metric("Sector", i.get('sector', 'N/A'))
            c2.metric("Industry", i.get('industry', 'N/A'))
            
            st.divider()
            
            # --- FINANCIAL HEALTH SECTION ---
            st.markdown("##### 💰 FINANCIAL HEALTH")
            f1, f2, f3 = st.columns(3)
            f1.metric("Total Revenue", format_large_number(i.get('totalRevenue')))
            f2.metric("EBITDA", format_large_number(i.get('ebitda')))
            f3.metric("Total Cash", format_large_number(i.get('totalCash')))
            
            st.divider()
            
            # --- WALL STREET SECTION ---
            st.markdown("##### 🏦 WALL STREET")
            w1, w2, w3 = st.columns(3)
            w1.metric("Consensus", str(target.get('Consensus', 'N/A')))
            w2.metric("Target Price", f"${safe_float(target.get('TargetPrice', 0)):.2f}")
            
            # Show Next Earnings Date
            edate = i.get('earningsDate', [])
            if isinstance(edate, list) and len(edate) > 0:
                next_earn = str(edate[0])[:10] 
            else: next_earn = "N/A"
            w3.metric("Next Earnings", next_earn)
            
            st.divider()
            st.markdown("##### 📊 KEY METRICS")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", format_large_number(i.get('marketCap')))
            c1.metric("P/E Ratio", i.get('trailingPE', '-'))
            
            # SAFE DISPLAY FOR PERCENTAGES
            div = safe_float(i.get('dividendYield', 0))
            pm = safe_float(i.get('profitMargins', 0))
            roe = safe_float(i.get('returnOnEquity', 0))
            
            c2.metric("Dividend Yield", f"{div*100:.2f}%" if div else '-')
            c2.metric("PEG Ratio", i.get('pegRatio', '-'))
            c3.metric("Profit Margin", f"{pm*100:.2f}%" if pm else '-')
            c3.metric("ROE", f"{roe*100:.2f}%" if roe else '-')
            c4.metric("Free Cash Flow", format_large_number(i.get('freeCashflow')))
            c4.metric("Debt/Equity", i.get('debtToEquity', '-'))
            
        with t3: 
            st.markdown("### 🧠 AI ANALYST BRIEFING")
            st.markdown(f"<div class='ai-box'>{target['AISummary']}</div>", unsafe_allow_html=True)
            
            report_text = generate_ceo_report(target)
            st.download_button(label="📄 DOWNLOAD INTELLIGENCE REPORT", data=report_text, file_name=f"{target['Ticker']}_Intelligence_Report.txt", mime="text/plain", type="primary")
            
            st.markdown("#### 📰 LATEST HEADLINES")
            news = target.get('News', [])
            if news:
                for n in news[:5]:
                    t_title = n.get('title', 'No Title')
                    t_link = n.get('link', '#')
                    st.markdown(f"• <a href='{t_link}' target='_blank' style='color: #00d4ff; text-decoration: none;'>{t_title}</a>", unsafe_allow_html=True)
            else: st.write("No news found.")
            
        with t4: 
            i = target['Info']
            c1, c2 = st.columns(2)
            c1.metric("Beta (Volatility)", i.get('beta', 'N/A'))
            c2.metric("Short Ratio", i.get('shortRatio', 'N/A'))
            
            st.markdown("---")
            st.markdown("##### 🏛️ INSTITUTIONAL HOLDINGS")
            
            # ATTEMPT 1: STOCKANALYSIS.COM (BEST)
            found_holdings = False
            try:
                sa_df = fetch_stockanalysis_holdings(sel_t)
                if sa_df is not None and not sa_df.empty:
                    st.table(sa_df)
                    found_holdings = True
            except: pass
            
            # ATTEMPT 2: YAHOO QUERY
            if not found_holdings:
                try: 
                    yq = YQTicker(sel_t)
                    holders = yq.institution_ownership
                    if holders is not None and isinstance(holders, pd.DataFrame) and not holders.empty:
                        st.table(holders.head(10))
                        found_holdings = True
                except: pass
            
            # ATTEMPT 3: MARKETWATCH (BACKUP)
            if not found_holdings:
                try: 
                    mw_df = fetch_marketwatch_holdings(sel_t)
                    if mw_df is not None and not mw_df.empty:
                        st.table(mw_df)
                        found_holdings = True
                except: pass
            
            if not found_holdings:
                st.info("Institutional data unavailable (Feed restricted).")

        with t5:
            st.markdown("### 🕵️‍♂️ INSIDER TRADING ACTIVITY")
            st.caption("Recent transactions by company executives and directors.")
            
            found_insiders = False
            # Using our new StockAnalysis Insider Scraper
            try:
                insider_df = fetch_insiders_stockanalysis(sel_t)
                if insider_df is not None and not insider_df.empty:
                    st.table(insider_df)
                    found_insiders = True
            except: pass
            
            # Backup: Finviz
            if not found_insiders:
                try:
                    fv_in = fetch_finviz_insiders(sel_t)
                    if fv_in is not None and not fv_in.empty:
                        st.table(fv_in)
                        found_insiders = True
                except: pass
                
            if not found_insiders:
                st.info("No recent insider trading activity found.")

        with t6:
            st.markdown("### 📈 DERIVATIVES & OPTIONS")
            
            deriv = target.get('Derivatives')
            if deriv:
                c1, c2 = st.columns(2)
                c1.metric("Put/Call Ratio", f"{deriv['pcr']:.2f}")
                c2.metric("Next Expiry", deriv['expiry'])
                
                st.subheader("🔥 Top Active Options")
                c_col, p_col = st.columns(2)
                with c_col:
                    st.markdown("#### CALLS (Bullish)")
                    st.table(deriv['calls'])
                with p_col:
                    st.markdown("#### PUTS (Bearish)")
                    st.table(deriv['puts'])
            else:
                st.info("Options data unavailable for this asset.")

    elif not run_scan:
        st.info("Enter tickers above and press INITIATE SCAN.")
