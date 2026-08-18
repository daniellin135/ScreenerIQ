"""
app.py - Main Entrypoint Launcher for ScreenerIQ
Stock & ETF Screener powered by Quantitative Technicals, Fundamentals & Google Gemini AI Engine.
"""

import os
import sys
import streamlit as st

# Ensure repository root is on sys.path for package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page Configuration
st.set_page_config(
    page_title="ScreenerIQ | Stock & ETF Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Multi-Page Navigation Structure pointing to package pages
home_page = st.Page("screener_iq/pages/0_Home.py", title="Home & KPI Overview", icon="⚡", default=True)
grid_page = st.Page("screener_iq/pages/1_Screener_Grid.py", title="Screener Data Grid", icon="📋")
chart_page = st.Page("screener_iq/pages/2_Technical_DeepDive.py", title="Technical Deep-Dive", icon="📈")
gemini_page = st.Page("screener_iq/pages/3_Gemini_AI_Hub.py", title="Gemini AI Intelligence", icon="🤖")

pg = st.navigation({
    "Overview": [home_page],
    "Analytics & Screening": [grid_page, chart_page, gemini_page]
})

pg.run()
