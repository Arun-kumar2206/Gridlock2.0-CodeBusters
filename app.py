"""
Main Streamlit app entry point.
Run this file using: streamlit run app.py
"""

import streamlit as st

# Configure page settings before anything else
st.set_page_config(
    page_title="Gridlock 2.0 - Bengaluru Traffic Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.dashboard import render_dashboard

if __name__ == "__main__":
    render_dashboard()
