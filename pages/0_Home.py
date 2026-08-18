"""
pages/0_Home.py - Landing Page & Summary Overview for ScreenerIQ
Displays KPI cards (Total Scanned, Matches Found, Top Pick, FCF Yield) and quick navigation cards.
"""

import streamlit as st
from common_ui import inject_custom_css, render_shared_sidebar

st.set_page_config(page_title="ScreenerIQ | Home", page_icon="⚡", layout="wide")
inject_custom_css()

# Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">⚡ ScreenerIQ</div>
    <div class="header-subtitle">
        Production-grade Stock & ETF Quantitative Screener integrated with Google Gemini 3.6 Flash AI Engine
    </div>
</div>
""", unsafe_allow_html=True)

# Render Shared Sidebar
full_df, screened_df, selected_api_key, filter_params = render_shared_sidebar()

if not full_df.empty:
    # KPI Summary Cards
    total_scanned = len(full_df)
    matches_found = len(screened_df)
    timeframe = filter_params.get("timeframe", "1Y")
    
    top_momentum_ticker = "N/A"
    if not screened_df.empty:
        tf_col = f"ret_{timeframe.lower()}"
        if tf_col in screened_df.columns:
            top_row = screened_df.sort_values(by=tf_col, ascending=False).iloc[0]
            top_momentum_ticker = f"{top_row['ticker']} ({top_row[tf_col]:+.1f}%)"

    avg_fcf_yield = screened_df["fcf_yield"].mean() if not screened_df.empty else 0.0

    st.markdown("### 📊 Market Scan Summary Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Scanned Universe</div>
            <div class="kpi-value">{total_scanned}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Matches Found</div>
            <div class="kpi-value" style="color: #38bdf8;">{matches_found}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Momentum Pick ({timeframe})</div>
            <div class="kpi-value" style="color: #4ade80; font-size: 1.4rem;">{top_momentum_ticker}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg FCF Yield</div>
            <div class="kpi-value" style="color: #c084fc;">{avg_fcf_yield:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Navigation Cards & Quick Buttons
    st.markdown("### 🚀 Feature Navigation & Deep Analysis")
    st.markdown("Select a module below to inspect screened assets or jump via the sidebar navigation:")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">📋 Screener Data Grid</div>
            <div class="nav-desc">
                View sortable financial data grid with conditional formatting, Market Cap, FCF yield, Profit Margins, and download CSV / Excel reports.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Data Grid →", key="btn_grid", type="primary", use_container_width=True):
            st.switch_page("pages/1_Screener_Grid.py")

    with c2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">📈 Technical Deep-Dive</div>
            <div class="nav-desc">
                Interactive Plotly candlestick charts featuring 252-day SMA (12M) gold overlay, 50-day SMA cyan overlay, volume bars, and quick metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Technical Charts →", key="btn_chart", type="primary", use_container_width=True):
            st.switch_page("pages/2_Technical_DeepDive.py")

    with c3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">🤖 Gemini AI Thesis Hub</div>
            <div class="nav-desc">
                Qualitative investment synthesis powered by Google Gemini 3.6 Flash. Generates AI sentiment scores, growth catalysts, and risk factors.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Gemini AI Analysis →", key="btn_ai", type="primary", use_container_width=True):
            st.switch_page("pages/3_Gemini_AI_Hub.py")
