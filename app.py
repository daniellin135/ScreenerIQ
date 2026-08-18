"""
app.py - Main Multi-Page Navigation Entry Point for ScreenerIQ
Stock & ETF Screener powered by Quantitative Technicals, Fundamentals & Google Gemini AI Engine.
"""

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="ScreenerIQ | Stock & ETF Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Multi-Page Navigation Structure
home_page = st.Page("pages/0_Home.py", title="Home & KPI Overview", icon="⚡", default=True)
grid_page = st.Page("pages/1_Screener_Grid.py", title="Screener Data Grid", icon="📋")
chart_page = st.Page("pages/2_Technical_DeepDive.py", title="Technical Deep-Dive", icon="📈")
gemini_page = st.Page("pages/3_Gemini_AI_Hub.py", title="Gemini AI Intelligence", icon="🤖")

pg = st.navigation({
    "Overview": [home_page],
    "Analytics & Screening": [grid_page, chart_page, gemini_page]
})

pg.run()
