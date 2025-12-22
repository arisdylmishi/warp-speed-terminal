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
    """The Oracle Ghost Algorithm"""
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
        if num >= 1e6: return f"${num/1e6:.2f}
        
