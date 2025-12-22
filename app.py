import yfinance as yf
import tkinter as tk
from tkinter import ttk, Toplevel, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
import json
import os
import webbrowser
import requests
import xml.etree.ElementTree as ET
import sys 

# --- 1. SAFE IMPORTS ---
try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("NOTE: TextBlob not installed. Run 'pip3 install textblob'")

try:
    import seaborn as sns
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False
    print("NOTE: Seaborn not installed. Matrix will use basic plotting.")

# --- 2. CONFIGURATION ---
WATCHLIST_FILE = "watchlist.json"
REFRESH_INTERVAL = 300 * 1000 # 5 minutes
PE_THRESHOLD = 35.0
MA_MULTIPLIER = 1.4
GHOST_LOOKBACK = 30
GHOST_PROJECTION = 15

# --- THEME COLORS ---
COLOR_BG = "#0b0c10"          
COLOR_PANEL = "#1f2833"       
COLOR_TEXT = "#c5c6c7"        
COLOR_ACCENT = "#66fcf1"      
COLOR_CHART_BG = "#1f2833"    
COLOR_GRID = "#45a29e"        
COLOR_ORACLE = "#d500f9"      
COLOR_UP = "#00e676"          
COLOR_DOWN = "#ff1744"        
COLOR_SPY = "#555555"
COLOR_TARGET = "#ffea00"      
COLOR_RESISTANCE = "#ff5252"  
COLOR_SUPPORT = "#69f0ae"     
COLOR_FIB = "#ff9800"         

# --- MAC COMPATIBILITY ---
if sys.platform == 'darwin':
    BTN_FG = "black"
    BTN_BG = "white"
else:
    BTN_FG = "white"
    BTN_BG = "#333333"

# --- GLOBAL STATE ---
current_stocks = [] 
stock_data = [] 
spy_history = None
macro_data = {}

# ==========================================
# --- PART A: HELPER FUNCTIONS (LOGIC) ---
# ==========================================

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)
                return [s for s in data if 1 <= len(s) <= 6]
        except: return []
    return []

def save_watchlist(stocks):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            clean = list(set([s.strip().upper() for s in stocks if s.strip()]))
            json.dump(clean, f)
    except: pass

def get_rss_news(ticker):
    news_items = []
    try:
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall('.//item')[:5]:
                title = item.find('title').text
                link = item.find('link').text
                if title and link:
                    news_items.append({'title': title, 'link': link})
    except: pass
    return news_items

def analyze_sentiment(news_items):
    if not SENTIMENT_AVAILABLE: return 0, "NO AI"
    score = 0
    count = 0
    for item in news_items:
        blob = TextBlob(item['title'])
        score += blob.sentiment.polarity
        count += 1
    if count == 0: return 0, "NEUTRAL"
    avg = score / count
    if avg > 0.05: return avg, "BULLISH"
    if avg < -0.05: return avg, "BEARISH"
    return avg, "NEUTRAL"

# --- MATH & INDICATORS ---
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

def calculate_support_resistance(hist, window=50):
    recent = hist.iloc[-window:]
    return recent['Low'].min(), recent['High'].max()

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
    return levels, high, low

# --- SNIPER & VERDICT ---
def calculate_sniper_score(stock):
    score = 0
    reasons = []
    
    hist = stock['hist']
    upper = hist['UpperBB'].iloc[-1]
    lower = hist['LowerBB'].iloc[-1]
    price = stock['current_price']
    bandwidth = (upper - lower) / price
    
    if bandwidth < 0.10: 
        score += 25; reasons.append("⚡ VOLATILITY SQUEEZE")
    
    try:
        rsi = float(stock['rsi_14'])
        if 45 <= rsi <= 65: score += 15; reasons.append("⚡ RSI Accumulation")
    except: pass
         
    try:
        rvol = float(stock['rvol'])
        if rvol > 1.5: score += 20; reasons.append("⚡ Institutional Vol")
    except: pass
    
    if price > stock['ma50']: score += 20; reasons.append("⚡ Bullish Trend")
         
    try:
        target = float(stock['full_info']['targetMeanPrice'])
        if target > price * 1.1: score += 20; reasons.append("⚡ Analyst Backing")
    except: pass
    
    return score, reasons

def generate_verdict(stock):
    score = 0
    reasons = []
    
    if stock['current_price'] > stock['ma50']:
        score += 20; reasons.append("✓ Trend: Bullish (>50MA)")
    else: reasons.append("✗ Trend: Bearish (<50MA)")
    
    if stock['current_price'] > stock['ma200']: score += 10
    
    try:
        pe = float(stock['pe'])
        if 0 < pe < 25: score += 15; reasons.append("✓ Undervalued (P/E < 25)")
    except: pass

    try:
        target = float(stock['full_info']['targetMeanPrice'])
        if target > stock['current_price']:
            upside = ((target - stock['current_price'])/stock['current_price'])*100
            score += 20; reasons.append(f"✓ Analyst Target: +{upside:.0f}%")
        else: reasons.append("✗ Price > Analyst Target")
    except: reasons.append("- No Analyst Targets")

    if stock['sentiment_label'] == "BULLISH": score += 15; reasons.append("✓ News Sentiment: Positive")
    
    try:
        rsi = float(stock['rsi_14'])
        if 40 <= rsi <= 70: score += 10
        elif rsi < 30: score += 20; reasons.append("! Oversold (Bounce Play)")
    except: pass

    if score >= 75: return score, "STRONG BUY", COLOR_UP, reasons
    if score >= 55: return score, "BUY", COLOR_UP, reasons
    if score <= 30: return score, "SELL", COLOR_DOWN, reasons
    return score, "HOLD", "white", reasons

# --- ORACLE (Pattern Matching) ---
def find_similar_pattern(hist_series, lookback_window=30):
    if len(hist_series) < (lookback_window * 4): return None, None, 0
    current_pattern = hist_series.iloc[-lookback_window:].values
    if current_pattern.max() == current_pattern.min(): return None, None, 0
    
    c_min, c_max = current_pattern.min(), current_pattern.max()
    current_norm = (current_pattern - c_min) / (c_max - c_min)
    
    best_score = -1
    best_idx = -1
    search_range = len(hist_series) - lookback_window - GHOST_PROJECTION - 1
    
    for i in range(0, search_range, 2):
        candidate = hist_series.iloc[i : i+lookback_window].values
        if candidate.max() == candidate.min(): continue
        cand_norm = (candidate - candidate.min()) / (candidate.max() - candidate.min())
        try:
            score = np.corrcoef(current_norm, cand_norm)[0, 1]
            if score > best_score:
                best_score = score
                best_idx = i
        except: continue

    if best_score > 0.80:
        ghost_data = hist_series.iloc[best_idx : best_idx + lookback_window + GHOST_PROJECTION].copy()
        match_date = hist_series.index[best_idx]
        return ghost_data, match_date, best_score
    return None, None, 0

# --- DATA FETCHING ---
def fetch_macro_data():
    global macro_data
    tickers = {"VIX": "^VIX", "10Y": "^TNX", "BTC": "BTC-USD", "Oil": "CL=F"}
    try:
        data = yf.download(list(tickers.values()), period="5d", progress=False, auto_adjust=True)
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        results = {}
        for name, tick in tickers.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    series = data['Close'][tick] if tick in data['Close'].columns else None
                else:
                    series = data['Close']
                if series is not None:
                    last, prev = series.iloc[-1], series.iloc[-2]
                    change = ((last - prev) / prev) * 100
                    results[name] = (last, change)
            except: results[name] = (0, 0)
        macro_data = results
    except: pass

def fetch_spy():
    global spy_history
    try:
        spy = yf.Ticker("SPY")
        spy_history = spy.history(period="5y")
        if spy_history.index.tz is not None: spy_history.index = spy_history.index.tz_localize(None)
    except: pass

def get_stock_data(stocks):
    data = []
    if spy_history is None: fetch_spy()
    fetch_macro_data()
    
    unique_tickers = list({t.strip().upper() for t in stocks if t.strip()})
    if not unique_tickers: return data
    
    try:
        hist_data = yf.download(unique_tickers, period="2y", interval="1d", progress=False, auto_adjust=True)
    except: return data

    if hist_data.empty: return data

    for ticker in unique_tickers:
        try:
            if isinstance(hist_data.columns, pd.MultiIndex):
                if ticker not in hist_data['Close'].columns: continue
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

            if h_close.index.tz is not None: h_close.index = h_close.index.tz_localize(None)
            if h_high.index.tz is not None: h_high.index = h_high.index.tz_localize(None)
            if h_low.index.tz is not None: h_low.index = h_low.index.tz_localize(None)
            if h_vol.index.tz is not None: h_vol.index = h_vol.index.tz_localize(None)

            current_price = h_close.iloc[-1]
            prev_close = h_close.iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            avg_vol = h_vol.rolling(50).mean().iloc[-1]
            curr_vol = h_vol.iloc[-1]
            rvol = curr_vol / avg_vol if avg_vol > 0 else 1.0
            
            ma50 = h_close.rolling(50).mean().iloc[-1]
            ma200 = h_close.rolling(200).mean().iloc[-1]
            rsi_series = calculate_rsi(h_close)
            macd, macd_sig = calculate_macd(h_close)
            upper_bb, lower_bb = calculate_bollinger_bands(h_close)
            
            t_obj = yf.Ticker(ticker)
            info = t_obj.info
            pe = info.get("trailingPE", None)
            
            news_items = get_rss_news(ticker)
            sent_score, sent_label = analyze_sentiment(news_items)
            supp, res = calculate_support_resistance(pd.DataFrame({'High': h_high, 'Low': h_low}))
            fib_levels, fib_high, fib_low = calculate_fibonacci(pd.DataFrame({'High': h_high, 'Low': h_low}))

            peg = info.get("pegRatio", None)
            if peg is None:
                try:
                    growth = info.get("earningsGrowth", 0)
                    if pe and growth and growth > 0:
                        peg = pe / (growth * 100)
                except: pass
            
            peg_disp = f"{peg:.2f}" if peg is not None else "N/A"

            full_info = {
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "beta": info.get("beta", "N/A"),
                "shortRatio": info.get("shortRatio", "N/A"),
                "dividendYield": info.get("dividendYield", "N/A"),
                "profitMargins": info.get("profitMargins", "N/A"), 
                "52WeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
                "52WeekLow": info.get("fiftyTwoWeekLow", "N/A"),
                "targetMeanPrice": info.get("targetMeanPrice", "N/A"),
                "recommendationKey": info.get("recommendationKey", "N/A"),
                "pegRatio": peg_disp, 
                "heldPercentInstitutions": info.get("heldPercentInstitutions", "N/A"),
                "heldPercentInsiders": info.get("heldPercentInsiders", "N/A")
            }

            bubble = False
            if pe and pe > PE_THRESHOLD and current_price > ma200 * MA_MULTIPLIER: 
                bubble = True

            hist_df = h_close.to_frame(name='Close')
            hist_df['Volume'] = h_vol
            hist_df['MACD'] = macd
            hist_df['Signal'] = macd_sig
            hist_df['UpperBB'] = upper_bb
            hist_df['LowerBB'] = lower_bb
            hist_df['RSI'] = rsi_series

            stock_obj = {
                "ticker": ticker,
                "current_price": current_price,
                "change_pct": change_pct,
                "ma50": ma50,
                "ma200": ma200,
                "rsi_14": f"{rsi_series.iloc[-1]:.1f}",
                "rvol": f"{rvol:.1f}",
                "pe": f"{pe:.1f}" if pe else "N/A",
                "bubble": bubble,
                "hist": hist_df,
                "full_info": full_info,
                "news": news_items,
                "sentiment_label": sent_label,
                "support": supp,
                "resistance": res,
                "yfinance_obj": t_obj,
                "fib_levels": fib_levels,
                "fib_high": fib_high,
                "fib_low": fib_low
            }
            
            v_score, v_rating, v_color, v_reasons = generate_verdict(stock_obj)
            s_score, s_reasons = calculate_sniper_score(stock_obj)
            
            stock_obj.update({
                'verdict_score': v_score, 
                'verdict_rating': v_rating, 
                'verdict_color': v_color, 
                'verdict_reasons': v_reasons,
                'sniper_score': s_score,
                'sniper_reasons': s_reasons
            })
            
            data.append(stock_obj)
            
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue
    return data

# ==========================================
# --- PART B: THE TERMINAL APP (GUI) ---
# ==========================================

def launch_terminal():
    """Starts the Main Warp Speed Terminal Window"""
    global root, tree, status_label, search_entry, macro_label, current_stocks
    
    current_stocks = load_watchlist()
    
    # Create Main Window
    root = tk.Tk()
    root.title("WARP SPEED TERMINAL V17")
    root.geometry("1500x850")
    root.configure(bg=COLOR_BG)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Treeview", background=COLOR_PANEL, foreground=COLOR_TEXT, fieldbackground=COLOR_PANEL, rowheight=30, font=("Segoe UI", 10))
    style.configure("Treeview.Heading", background="#333333", foreground="white", font=("Arial", 10, "bold"), borderwidth=0)
    style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])

    # MACRO BAR
    macro_label = tk.Label(root, text="INITIALIZING MARKETS...", bg="#222", fg=COLOR_ORACLE, font=("Consolas", 10))
    macro_label.pack(side=tk.TOP, fill=tk.X)

    top_frame = tk.Frame(root, bg=COLOR_BG)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=15, pady=15)
    tk.Label(top_frame, text="ASSETS >", bg=COLOR_BG, fg=COLOR_ACCENT, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
    search_entry = tk.Entry(top_frame, width=40, bg=COLOR_PANEL, fg="white", insertbackground="white", relief="flat")
    if current_stocks: search_entry.insert(0, " ".join(current_stocks))
    search_entry.pack(side=tk.LEFT, padx=10, ipady=4)
    tk.Button(top_frame, text="INITIATE SCAN", command=search_stocks, bg=COLOR_ACCENT, fg="black", font=("Arial", 9, "bold"), relief="flat").pack(side=tk.LEFT, padx=10)

    # EXTRA BUTTONS
    tk.Button(top_frame, text="[ MATRIX ]", command=open_correlation_matrix, bg=BTN_BG, fg=BTN_FG, relief="flat").pack(side=tk.LEFT, padx=5)
    tk.Button(top_frame, text="[ EXPORT ]", command=export_data, bg=BTN_BG, fg=BTN_FG, relief="flat").pack(side=tk.LEFT, padx=5)

    status_label = tk.Label(top_frame, text="SYSTEM READY", bg=COLOR_BG, fg="#555555", font=("Consolas", 10))
    status_label.pack(side=tk.RIGHT)

    cols = ("Ticker", "Price", "Change %", "VERDICT", "SNIPER", "RVOL", "Bubble?", "PEG", "RSI", "Sentiment")
    tree = ttk.Treeview(root, columns=cols, show="headings")
    for col in cols: tree.heading(col, text=col.upper())
    tree.column("Ticker", width=60, anchor="center")
    tree.column("Price", width=80, anchor="e")
    tree.column("Change %", width=70, anchor="e")
    tree.column("VERDICT", width=90, anchor="center")
    tree.column("SNIPER", width=50, anchor="center")
    tree.column("RVOL", width=50, anchor="center")
    tree.column("Bubble?", width=60, anchor="center")
    tree.column("PEG", width=50, anchor="center")
    tree.column("RSI", width=50, anchor="e")
    tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    tree.bind("<Double-1>", on_double_click)

    # Start first update
    if current_stocks: update_dashboard()
    root.mainloop()

# --- GUI HELPER FUNCTIONS FOR TERMINAL ---
def update_dashboard():
    global stock_data
    status_label.config(text="Status: Syncing...", fg=COLOR_ACCENT)
    root.update()
    
    # Update Macro
    macro_txt = ""
    if macro_data:
        macro_txt = " | ".join([f"{k}: {v[0]:.2f} ({v[1]:+.2f}%)" for k,v in macro_data.items()])
    macro_label.config(text=f"MARKET CLIMATE: {macro_txt}")

    current_stocks = load_watchlist()
    stock_data = get_stock_data(current_stocks)
    
    for row in tree.get_children(): tree.delete(row)
    for i, stock in enumerate(stock_data):
        v_rat = stock['verdict_rating']
        chg_str = f"{stock['change_pct']:+.2f}%"
        bubble_str = "🚨 YES" if stock['bubble'] else "NO"
        
        tree.insert("", "end", values=(
            stock["ticker"], 
            f"{stock['current_price']:.2f}", 
            chg_str, 
            v_rat, 
            stock['sniper_score'],
            stock['rvol'],
            bubble_str,
            stock['full_info']['pegRatio'],
            stock['rsi_14'],
            stock['sentiment_label']
        ), tags=(str(i),))
        
        if stock['bubble']: tree.tag_configure(str(i), background="#330000") 
        elif "BUY" in v_rat: tree.tag_configure(str(i), foreground=COLOR_UP)
        elif "SELL" in v_rat: tree.tag_configure(str(i), foreground=COLOR_DOWN)
        else: tree.tag_configure(str(i), foreground="white")

    status_label.config(text=f"Live: {datetime.now().strftime('%H:%M')}", fg=COLOR_UP)
    root.after(REFRESH_INTERVAL, update_dashboard)

def on_double_click(event):
    selected = tree.selection()
    if not selected: return
    index = int(tree.item(selected)["tags"][0])
    open_deep_dive(stock_data[index])

def search_stocks():
    query = search_entry.get()
    new = [s.strip().upper() for s in query.replace(",", " ").split() if s.strip()]
    if new:
        save_watchlist(new)
        update_dashboard()

def open_correlation_matrix():
    if len(stock_data) < 2:
        messagebox.showwarning("WAIT!", "Not enough stocks scanned.")
        return
    try:
        closes = {}
        for s in stock_data:
            if 'hist' in s and not s['hist'].empty:
                closes[s['ticker']] = s['hist']['Close']
        
        df = pd.DataFrame(closes)
        corr = df.corr()
        
        plt.figure(figsize=(10, 8))
        if HEATMAP_AVAILABLE:
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        else:
            plt.imshow(corr, cmap="coolwarm")
            plt.colorbar()
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Matrix failed:\n{e}")

def export_data():
    if not stock_data: return
    try:
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not f: return
        rows = []
        for s in stock_data:
            rows.append({
                "Ticker": s['ticker'], "Price": s['current_price'], "Verdict": s['verdict_rating'],
                "Sniper Score": s['sniper_score'], "RVOL": s['rvol'], "P/E": s['pe'], "PEG": s['full_info']['pegRatio']
            })
        pd.DataFrame(rows).to_csv(f, index=False)
        messagebox.showinfo("Success", "Data Exported.")
    except: pass

def open_deep_dive(stock):
    top = Toplevel(root)
    top.title(f"WARP SPEED // {stock['ticker']}")
    top.geometry("1600x950")
    top.configure(bg=COLOR_BG)
    
    left_notebook = ttk.Notebook(top)
    left_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    
    tab_analysis = tk.Frame(left_notebook, bg=COLOR_PANEL)
    tab_fund = tk.Frame(left_notebook, bg=COLOR_PANEL)
    tab_risk = tk.Frame(left_notebook, bg=COLOR_PANEL)
    left_notebook.add(tab_analysis, text="  Analysis  ")
    left_notebook.add(tab_fund, text="  Fundamentals  ")
    left_notebook.add(tab_risk, text="  Insiders & Risk  ")

    def add_lbl(parent, txt, size=10, color="white", bold=False, click_url=None):
        font = ("Consolas", size, "bold" if bold else "normal")
        lbl = tk.Label(parent, text=txt, bg=COLOR_PANEL, fg=color, font=font, anchor="w", justify="left")
        lbl.pack(fill=tk.X, pady=1, padx=10)
        if click_url:
            lbl.bind("<Button-1>", lambda e: webbrowser.open(click_url))
            lbl.config(cursor="hand2")

    # TAB 1: ANALYSIS
    chg_col = COLOR_UP if stock['change_pct'] >= 0 else COLOR_DOWN
    add_lbl(tab_analysis, f"{stock['ticker']}", 32, COLOR_ACCENT, True)
    add_lbl(tab_analysis, f"${stock['current_price']:.2f}", 22, "white", True)
    add_lbl(tab_analysis, f"{stock['change_pct']:+.2f}%", 16, chg_col, True)
    
    tk.Frame(tab_analysis, bg=COLOR_ACCENT, height=2).pack(fill=tk.X, padx=10, pady=15)
    tk.Label(tab_analysis, text=f"{stock['verdict_rating']}", bg=COLOR_PANEL, fg=stock['verdict_color'], font=("Arial", 26, "bold")).pack()
    
    if stock['sniper_score'] > 50:
        tk.Label(tab_analysis, text=f"SNIPER SCORE: {stock['sniper_score']}", bg=COLOR_PANEL, fg="#ff00ff", font=("Arial", 16, "bold")).pack(pady=5)
    
    for reason in stock['verdict_reasons']:
        r_col = COLOR_UP if "✓" in reason else (COLOR_DOWN if "✗" in reason else "white")
        add_lbl(tab_analysis, reason, 9, r_col)

    if stock['bubble']:
        tk.Label(tab_analysis, text="⚠️ BUBBLE SIGNAL", bg=COLOR_PANEL, fg=COLOR_DOWN, font=("Arial", 12, "bold")).pack(pady=5)

    tk.Frame(tab_analysis, bg="#333", height=1).pack(fill=tk.X, padx=20, pady=10)
    add_lbl(tab_analysis, f"Sentiment: {stock['sentiment_label']}", 10, "white", True)
    if stock['news']:
        for item in stock['news']:
            t = item['title'][:30]+"..." if len(item['title'])>30 else item['title']
            add_lbl(tab_analysis, f"> {t}", 8, "#888", False, item['link'])

    # TAB 2: FUNDAMENTALS
    info = stock['full_info']
    def fmt_num(n):
        if n == "N/A": return "N/A"
        try:
            if n > 1e12: return f"${n/1e12:.2f}T"
            if n > 1e9: return f"${n/1e9:.2f}B"
            return f"${n/1e6:.2f}M"
        except: return "N/A"

    add_lbl(tab_fund, ":: VALUATION ::", 11, COLOR_ORACLE, True)
    add_lbl(tab_fund, f"Mkt Cap: {fmt_num(info['marketCap'])}")
    add_lbl(tab_fund, f"P/E Ratio: {stock['pe']}")
    add_lbl(tab_fund, f"PEG Ratio: {info['pegRatio']}")
    
    div = info['dividendYield']
    try:
        div_str = f"{float(div)*100:.2f}%" if div != "N/A" else "0.00%"
    except: div_str = "N/A"
    add_lbl(tab_fund, f"Div Yield: {div_str}")
    
    # --- NEW FUNDAMENTAL METRICS DISPLAY ---
    add_lbl(tab_fund, " ")
    add_lbl(tab_fund, ":: HEALTH & MOAT ::", 11, COLOR_ORACLE, True)
    
    # ROE
    roe = info['returnOnEquity']
    try:
        roe_str = f"{roe*100:.2f}%" if roe != "N/A" else "N/A"
    except: roe_str = "N/A"
    add_lbl(tab_fund, f"ROE: {roe_str}")

    # Debt to Equity
    dte = info['debtToEquity']
    add_lbl(tab_fund, f"Debt/Equity: {dte}")

    # Profit Margins
    pm = info['profitMargins']
    try:
        pm_str = f"{pm*100:.2f}%" if pm != "N/A" else "N/A"
    except: pm_str = "N/A"
    add_lbl(tab_fund, f"Profit Margin: {pm_str}")

    # FCF
    fcf = info['freeCashflow']
    add_lbl(tab_fund, f"Free Cash Flow: {fmt_num(fcf)}")
    
    add_lbl(tab_fund, " ")
    add_lbl(tab_fund, ":: WALL STREET ::", 11, COLOR_ORACLE, True)
    rec = str(info['recommendationKey']).upper().replace("_", " ")
    add_lbl(tab_fund, f"Consensus: {rec}")
    add_lbl(tab_fund, f"Target Px: ${info['targetMeanPrice']}")

    # TAB 3: RISK
    add_lbl(tab_risk, ":: RISK PROFILE ::", 11, COLOR_ORACLE, True)
    add_lbl(tab_risk, f"Beta: {info['beta']}")
    add_lbl(tab_risk, f"Short Float: {info['shortRatio']}")
    
    tk.Label(tab_risk, text="MAJOR HOLDERS", bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(pady=10)
    try:
        holders = stock['yfinance_obj'].institutional_holders
        if holders is not None and not holders.empty:
            for idx, row in holders.head(5).iterrows():
                add_lbl(tab_risk, f"{row.get('Holder','N/A')}", 8, "#ccc")
    except: pass

    # CHARTS
    chart_frame = tk.Frame(top, bg=COLOR_BG)
    chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    tf_frame = tk.Frame(chart_frame, bg=COLOR_BG)
    tf_frame.pack(fill=tk.X, pady=(0, 5))
    
    chart_content = tk.Frame(chart_frame, bg=COLOR_BG)
    chart_content.pack(fill=tk.BOTH, expand=True)

    def plot_charts(period_days=365): 
        for widget in chart_content.winfo_children(): widget.destroy()
        fig_dd = plt.Figure(figsize=(10, 8), facecolor=COLOR_CHART_BG)
        gs = fig_dd.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.15)
        ax_p = fig_dd.add_subplot(gs[0])
        ax_m = fig_dd.add_subplot(gs[1], sharex=ax_p)
        ax_v = fig_dd.add_subplot(gs[2], sharex=ax_p)

        hist = stock['hist']
        if len(hist) > period_days: hist = hist.iloc[-period_days:]
        
        ax_p.plot(hist.index, hist['Close'], color="white", linewidth=1.5)
        ax_p.fill_between(hist.index, hist['Close'], hist['Close'].min(), color=COLOR_ACCENT, alpha=0.1)
        
        ax_p.plot(hist.index, hist['UpperBB'], color=COLOR_ACCENT, alpha=0.3)
        ax_p.plot(hist.index, hist['LowerBB'], color=COLOR_ACCENT, alpha=0.3)
        
        fibs = stock['fib_levels']
        for level, price in fibs.items():
            if price > hist['Close'].min():
                ax_p.axhline(price, color=COLOR_FIB, linestyle="--", linewidth=0.8)
                ax_p.text(hist.index[0], price, f"Fib {level}", color=COLOR_FIB, fontsize=8)

        ax_p.axhline(stock['resistance'], color=COLOR_RESISTANCE, linestyle="-", linewidth=1)
        ax_p.axhline(stock['support'], color=COLOR_SUPPORT, linestyle="-", linewidth=1)
        
        target_price = stock['full_info']['targetMeanPrice']
        if target_price != "N/A":
             ax_p.axhline(target_price, color=COLOR_TARGET, linestyle=":", linewidth=1.5, label=f"Target ${target_price}")

        if spy_history is not None:
            spy_slice = spy_history['Close'].reindex(hist.index, method='nearest')
            if len(spy_slice) > 0 and len(hist) > 0:
                scale = hist['Close'].iloc[0] / spy_slice.iloc[0]
                ax_p.plot(hist.index, spy_slice * scale, color=COLOR_SPY, linestyle="--", linewidth=1, label="S&P 500", alpha=0.5)
        
        ax_p.legend(facecolor=COLOR_CHART_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)

        ghost_data, match_date, score = find_similar_pattern(stock['hist']['Close'])
        if ghost_data is not None:
            last_date = stock['hist'].index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, GHOST_PROJECTION + 1)]
            scale = stock['hist']['Close'].iloc[-1] / ghost_data.iloc[GHOST_LOOKBACK-1]
            ghost_future = ghost_data.iloc[GHOST_LOOKBACK:] * scale
            ax_p.plot(future_dates, ghost_future, color=COLOR_ORACLE, linestyle="--", linewidth=2)

        ax_m.plot(hist.index, hist['MACD'], color=COLOR_ACCENT)
        ax_m.plot(hist.index, hist['Signal'], color=COLOR_DOWN)
        hist_vals = hist['MACD'] - hist['Signal']
        colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in hist_vals]
        ax_m.bar(hist.index, hist_vals, color=colors, alpha=0.5)

        vol_colors = [COLOR_UP if hist['Close'].iloc[i] >= hist['Close'].iloc[i-1] else COLOR_DOWN for i in range(len(hist))]
        ax_v.bar(hist.index, hist['Volume'], color=vol_colors, alpha=0.8)

        for ax in [ax_p, ax_m, ax_v]:
            ax.set_facecolor(COLOR_CHART_BG)
            ax.tick_params(colors=COLOR_TEXT)
            ax.grid(True, color=COLOR_GRID, linestyle=':', alpha=0.5)
            for spine in ax.spines.values(): spine.set_color(COLOR_GRID)

        plt.setp(ax_p.get_xticklabels(), visible=False)
        plt.setp(ax_m.get_xticklabels(), visible=False)
        
        canvas_dd = FigureCanvasTkAgg(fig_dd, master=chart_content)
        canvas_dd.draw()
        canvas_dd.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def make_btn(txt, days):
        b = tk.Button(tf_frame, text=txt, bg=BTN_BG, fg=BTN_FG, activebackground=COLOR_ACCENT, relief="flat", command=lambda: plot_charts(days))
        b.pack(side=tk.LEFT, padx=2)

    make_btn("1M", 30); make_btn("3M", 90); make_btn("6M", 180); make_btn("1Y", 365); make_btn("MAX", 1000)
    plot_charts(365)


# ==========================================
# --- PART C: LOGIN SYSTEM ---
# ==========================================

class LoginWindow:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("SYSTEM ACCESS")
        self.win.geometry("400x500")
        self.win.configure(bg=COLOR_BG)
        
        # Center the window
        screen_width = self.win.winfo_screenwidth()
        screen_height = self.win.winfo_screenheight()
        x = int((screen_width/2) - (400/2))
        y = int((screen_height/2) - (500/2))
        self.win.geometry(f"400x500+{x}+{y}")

        # UI Elements
        tk.Label(self.win, text="WARP SPEED", font=("Arial", 24, "bold"), bg=COLOR_BG, fg=COLOR_ACCENT).pack(pady=(60, 5))
        tk.Label(self.win, text="TERMINAL ACCESS", font=("Arial", 12), bg=COLOR_BG, fg=COLOR_TEXT).pack(pady=(0, 40))

        tk.Label(self.win, text="IDENTITY", font=("Arial", 10), bg=COLOR_BG, fg="#555").pack()
        self.entry_user = tk.Entry(self.win, font=("Arial", 12), bg=COLOR_PANEL, fg="white", justify="center", relief="flat", insertbackground="white")
        self.entry_user.pack(pady=5, ipady=5, padx=50, fill=tk.X)

        tk.Label(self.win, text="PASSPHRASE", font=("Arial", 10), bg=COLOR_BG, fg="#555").pack(pady=(20, 0))
        self.entry_pass = tk.Entry(self.win, font=("Arial", 12), bg=COLOR_PANEL, fg="white", justify="center", show="•", relief="flat", insertbackground="white")
        self.entry_pass.pack(pady=5, ipady=5, padx=50, fill=tk.X)

        self.btn_login = tk.Button(self.win, text="AUTHENTICATE", command=self.check_login, 
                                   bg=COLOR_ACCENT, fg="black", font=("Arial", 11, "bold"), relief="flat", cursor="hand2")
        self.btn_login.pack(pady=40, ipady=5, ipadx=20)
        
        self.lbl_status = tk.Label(self.win, text="", bg=COLOR_BG, fg=COLOR_DOWN)
        self.lbl_status.pack()

        # Bind Enter key
        self.win.bind('<Return>', lambda event: self.check_login())
        
        self.win.mainloop()

    def check_login(self):
        username = self.entry_user.get()
        password = self.entry_pass.get()

        # --- LOGIN LOGIC ---
        if username == "admin" and password == "1234":  # Change these!
            self.lbl_status.config(text="ACCESS GRANTED", fg=COLOR_UP)
            self.win.after(500, self.start_app)
        else:
            self.lbl_status.config(text="ACCESS DENIED", fg=COLOR_DOWN)

    def start_app(self):
        self.win.destroy()  # Close login window
        launch_terminal()   # START THE USER'S TERMINAL


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Start the Login Window first
    app = LoginWindow()
