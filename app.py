import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

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
            background-color: #1f2833;
            color: #66fcf1;
        }
        .stButton>button:hover {
            background-color: #45a29e;
            color: black;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem;
            color: #00FFCC; 
            font-family: 'Courier New', monospace;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #aaa;
        }
        
        .reason-box {
            background-color: #111;
            padding: 10px;
            border-left: 3px solid #00FFCC;
            margin-bottom: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #eee;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. LOGIC ---
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
    if not news_items: return "NEUTRAL", 0
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
        
    if count == 0: return "NEUTRAL", 0
    avg = score / count
    if avg > 0.05: return "BULLISH", avg
    if avg < -0.05: return "BEARISH", avg
    return "NEUTRAL", avg

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

# ==========================================
# --- 3. SESSION ---
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_status' not in st.session_state: st.session_state['user_status'] = "active" # Default active for test

# ==========================================
# --- 4. LANDING PAGE (Simple Bypass for Now) ---
# ==========================================
if not st.session_state['logged_in']:
    # Direct Login for testing stability
    st.session_state['logged_in'] = True
    st.session_state['user_email'] = "admin"
    st.rerun()

# ==========================================
# --- 5. TERMINAL APP ---
# ==========================================
if st.session_state['logged_in']:
    
    with st.sidebar:
        st.title("WARP SPEED")
        st.caption(f"User: {st.session_state.get('user_email', 'Guest')}")
        if st.button("RESET"): 
            st.session_state.clear()
            st.rerun()

    # --- MACRO BAR ---
    with st.container():
        try:
            macro_ticks = ["^VIX", "^TNX", "BTC-USD", "CL=F"]
            m_data = yf.download(macro_ticks, period="5d", progress=False, auto_adjust=True)['Close']
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            names = {"^VIX": "VIX (Fear)", "^TNX": "10Y Bond", "BTC-USD": "Bitcoin", "CL=F": "Oil"}
            
            if not m_data.empty:
                last_row = m_data.ffill().iloc[-1]
                prev_row = m_data.ffill().iloc[-2]
                for idx, (sym, name) in enumerate(names.items()):
                    val = last_row.get(sym, np.nan)
                    prev_val = prev_row.get(sym, np.nan)
                    if pd.notna(val) and pd.notna(prev_val) and prev_val != 0:
                        chg = ((val - prev_val) / prev_val) * 100
                        cols = [mc1, mc2, mc3, mc4]
                        cols[idx].metric(name, f"{val:.2f}", f"{chg:+.2f}%")
                    else:
                        cols = [mc1, mc2, mc3, mc4]
                        cols[idx].metric(name, "N/A", "N/A")
        except: st.caption("Macro Data Offline")
            
    st.divider()

    # --- SCANNER ENGINE (ROBUST LOOP) ---
    def scan_market_safe(tickers):
        results = []
        progress_text = "Scanning assets..."
        my_bar = st.progress(0, text=progress_text)
        total = len(tickers)
        
        for idx, t in enumerate(tickers):
            try:
                # Update progress
                my_bar.progress(int((idx + 1) / total * 100), text=f"Scanning {t}...")
                
                # Fetch Data
                stock = yf.Ticker(t)
                df = stock.history(period="1y")
                
                # Fallback if history is empty
                if df.empty:
                    df = yf.download(t, period="1y", progress=False, auto_adjust=True)
                
                if df.empty or len(df) < 50: 
                    print(f"Skipping {t}: Not enough data")
                    continue
                
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
                    reasons.append(f"✓ RSI ({rsi:.0f}) is Oversold")
                
                # Score
                score = 50
                if verdict == "BUY": score += 20
                if verdict == "STRONG BUY": score += 35
                
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
                
                peg = info.get('pegRatio', 'N/A')
                
                # Sentiment
                news = stock.news
                sent, sent_score = analyze_sentiment(news)
                
                results.append({
                    "Ticker": t, "Price": curr, "Change": chg, "Verdict": verdict, "Sniper": score, 
                    "RVOL": rvol, "Bubble": bubble, "PEG": peg, "RSI": rsi, "Sentiment": sent,
                    "History": df, "Info": info, "News": news, "Reasons": reasons,
                    "TargetPrice": info.get('targetMeanPrice', 'N/A'),
                    "Consensus": info.get('recommendationKey', 'N/A')
                })
            except Exception as e:
                print(f"Error scanning {t}: {e}")
                continue
            
        my_bar.empty()
        return results

    # --- MAIN INTERFACE ---
    with st.form("scanner"):
        c1, c2 = st.columns([3, 1])
        with c1: query = st.text_input("ENTER ASSETS", "AAPL TSLA NVDA BTC-USD JPM")
        with c2: run_scan = st.form_submit_button("INITIATE SCAN 🔎", type="primary")

    if run_scan:
        ticks = [t.strip().upper() for t in query.replace(",", " ").split() if t.strip()]
        if ticks:
            st.session_state['data'] = scan_market_safe(ticks)
            if not st.session_state['data']:
                st.error("No data found for these symbols. Check spelling or try US tickers (e.g. AAPL).")
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
            "RSI": f"{d['RSI']:.1f}",
            "SENTIMENT": d['Sentiment']
        } for d in st.session_state['data']])
        
        def highlight_verdict(val):
            color = '#00FFCC' if 'BUY' in val else '#ff4b4b' if 'SELL' in val else 'white'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(df_view.style.map(highlight_verdict, subset=['VERDICT']), width="stretch", hide_index=True)
        
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

        # 3. DEEP DIVE
        st.divider()
        st.subheader("🔬 DEEP DIVE ANALYSIS")
        sel_t = st.selectbox("Select Asset", [d['Ticker'] for d in st.session_state['data']])
        target = next(d for d in st.session_state['data'] if d['Ticker'] == sel_t)
        
        t1, t2, t3, t4 = st.tabs(["CHART & ORACLE", "FUNDAMENTALS & WALL ST", "NEWS AI", "RISK"])
        
        with t1: 
            hist = target['History']
            ghost = find_oracle_pattern(hist['Close'])
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['UpperBB'], line=dict(color='cyan', width=1), name='Upper BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['LowerBB'], line=dict(color='cyan', width=1), name='Lower BB'), row=1, col=1)
            
            if ghost is not None:
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(len(ghost))]
                fig.add_trace(go.Scatter(x=future_dates, y=ghost, line=dict(color='magenta', dash='dash', width=2), name='Oracle Ghost'), row=1, col=1)

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
            
        with t2: 
            i = target['Info']
            st.markdown("##### 🏦 WALL STREET")
            w1, w2 = st.columns(2)
            w1.metric("Consensus", str(target.get('Consensus', 'N/A')).upper())
            w2.metric("Target Price", f"${target.get('TargetPrice', 'N/A')}")
            
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", format_large_number(i.get('marketCap')))
            c1.metric("P/E Ratio", i.get('trailingPE', '-'))
            c2.metric("Dividend Yield", f"{i.get('dividendYield', 0)*100:.2f}%" if i.get('dividendYield') else '-')
            c3.metric("Profit Margin", f"{i.get('profitMargins', 0)*100:.2f}%" if i.get('profitMargins') else '-')
            c4.metric("Free Cash Flow", format_large_number(i.get('freeCashflow')))
            
        with t3: 
            st.write("Recent News Sentiment:")
            news = target.get('News', [])
            if news:
                for n in news[:5]:
                    sent, score = analyze_sentiment([n])
                    color = "green" if sent == "BULLISH" else "red" if sent == "BEARISH" else "gray"
                    t_title = n.get('title', 'No Title')
                    t_link = n.get('link', '#')
                    st.markdown(f"**:{color}[{sent}]** [{t_title}]({t_link})")
            else: st.write("No news found.")
            
        with t4: 
            i = target['Info']
            c1, c2 = st.columns(2)
            c1.metric("Beta (Volatility)", i.get('beta', '-'))
            c2.metric("Short Ratio", i.get('shortRatio', '-'))
            st.caption("Institutional Holders:")
            try: st.dataframe(yf.Ticker(sel_t).institutional_holders.head())
            except: st.write("Data hidden")
