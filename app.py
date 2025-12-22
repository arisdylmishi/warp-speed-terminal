import streamlit as st
import yfinance as yf

st.title("🕵️‍♂️ YAHOO BANNED ME?")

if st.button("TEST CONNECTION"):
    try:
        st.write("Trying to fetch AAPL...")
        # Δοκιμή με τον πιο απλό τρόπο
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1d")
        
        if data.empty:
            st.error("❌ ΑΠΟΤΥΧΙΑ: Το Yahoo επέστρεψε κενά δεδομένα. (Πιθανό Ban/Rate Limit)")
        else:
            st.success("✅ ΕΠΙΤΥΧΙΑ: Η σύνδεση λειτουργεί!")
            st.write(data)
            
    except Exception as e:
        st.error(f"❌ CRASH: {e}")
