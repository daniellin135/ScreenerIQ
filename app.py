"""
app.py - Main Interactive Streamlit Dashboard for ScreenerIQ
Stock & ETF Screener powered by Quantitative Technicals, Fundamentals & Google Gemini AI Engine.
"""

import io
import os
import pandas as pd
import plotly.graph_objects as gr
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

from screener import (
    load_screener_dataset,
    filter_dataset,
    get_ticker_historical_chart_data,
    DEFAULT_STOCKS,
    DEFAULT_ETFS
)
from gemini_analyst import batch_analyze_top_assets, InvestmentAnalysis

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="ScreenerIQ | Stock & ETF Screener with Gemini AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark CSS Styling
st.markdown("""
<style>
    /* Dark Theme Core */
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Title Header Styling */
    .header-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    /* AI Card Container */
    .ai-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    
    /* Sentiment Score Badges */
    .badge-score-high {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-score-med {
        background: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-score-low {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-suitability {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    /* Custom Scrollbar & Table tweak */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header Banner
    st.markdown("""
    <div class="header-container">
        <div class="header-title">⚡ ScreenerIQ</div>
        <div class="header-subtitle">
            Production-grade Stock & ETF Quantitative Screener integrated with Google Gemini 3.6 Flash AI Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("### ⚙️ Screener Controls")
    
    asset_type = st.sidebar.radio(
        "Asset Type Filter",
        options=["Both", "Stocks", "ETFs"],
        index=0,
        help="Select universe type: Stocks, ETFs, or combined"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Market Cap / AUM Filter")
    
    market_cap_range = st.sidebar.slider(
        "Market Cap ($ Billion)",
        min_value=0.0,
        max_value=300.0,
        value=(2.0, 10.0),  # Mid-cap default: $2B to $10B
        step=1.0,
        help="Default range ($2B - $10B) targets Mid-Cap growth assets"
    )
    min_market_cap, max_market_cap = market_cap_range

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📈 Technical & Fundamental Filters")
    
    above_sma_252 = st.sidebar.toggle(
        "Price > 12-Month SMA (SMA 252)",
        value=True,
        help="Filter assets currently trading above their 252-day simple moving average"
    )
    
    positive_fcf = st.sidebar.toggle(
        "Positive Free & Operating Cash Flow",
        value=True,
        help="Require Free Cash Flow > 0 and Operating Cash Flow > 0 for stocks (or Low Expense Ratio for ETFs)"
    )

    min_profit_margin = st.sidebar.slider(
        "Min Net Profit Margin (%)",
        min_value=-20.0,
        max_value=40.0,
        value=0.0,
        step=2.0,
        help="Minimum required profit margin for stocks"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⏳ Rolling Timeframe Return Filter")
    
    timeframe = st.sidebar.selectbox(
        "Historical Timeframe",
        options=["1M", "3M", "6M", "1Y", "3Y", "YTD"],
        index=3,
        help="Rolling timeframe for cumulative return calculation"
    )
    
    min_timeframe_return = st.sidebar.slider(
        f"Min Cumulative Return ({timeframe}) %",
        min_value=-50.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        help=f"Filter assets with cumulative return >= threshold over selected {timeframe} timeframe"
    )

    # Optional Custom Tickers Input
    st.sidebar.markdown("---")
    custom_ticker_str = st.sidebar.text_input(
        "Add Custom Tickers (comma separated)",
        value="",
        placeholder="e.g. PLTR, HOOD, SOFI",
        help="Add additional tickers to scan"
    )
    custom_tickers = tuple(
        [t.strip().upper() for t in custom_ticker_str.split(",") if t.strip()]
    )

    # Universe Preset Selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌐 Target Ticker Universe")
    universe_preset = st.sidebar.selectbox(
        "Select Universe Preset",
        options=[
            "Full Extended Universe (350+ Assets)",
            "S&P 500 & Nasdaq 100 Leaders (~250 Assets)",
            "Mid-Cap & High Growth (~150 Assets)",
            "Dividend & Value Aristocrats (~100 Assets)",
            "Major ETFs Universe (~50 ETFs)",
            "Core Benchmark (~84 Assets)"
        ],
        index=0,
        help="Choose target asset universe for quantitative scanning"
    )

    # Gemini API Key Input
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🤖 Gemini AI Config")
    
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Enter your Gemini API key or set GEMINI_API_KEY in .env file"
    )
    
    selected_api_key = api_key_input if api_key_input else None
    
    if not selected_api_key:
        st.sidebar.info("💡 No Gemini API key detected. App will use deterministic qualitative fallback mode.")

    # Fetch and Load Dataset
    with st.spinner(f"Fetching market data for '{universe_preset}' via multi-threaded pipeline..."):
        full_df = load_screener_dataset(preset_name=universe_preset, custom_tickers=custom_tickers)

    if full_df.empty:
        st.error("Unable to load financial market data. Please verify network connection or ticker symbols.")
        return

    # Apply Filters
    screened_df = filter_dataset(
        df=full_df,
        asset_type=asset_type,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        above_sma_252=above_sma_252,
        positive_fcf=positive_fcf,
        min_profit_margin=min_profit_margin,
        timeframe=timeframe,
        min_timeframe_return=min_timeframe_return
    )

    # Top KPI Metrics Display
    total_scanned = len(full_df)
    matches_found = len(screened_df)
    
    top_momentum_ticker = "N/A"
    top_momentum_val = 0.0
    if not screened_df.empty:
        tf_col = f"ret_{timeframe.lower()}"
        if tf_col in screened_df.columns:
            top_row = screened_df.sort_values(by=tf_col, ascending=False).iloc[0]
            top_momentum_ticker = f"{top_row['ticker']} ({top_row[tf_col]:+.1f}%)"
            top_momentum_val = top_row[tf_col]

    avg_fcf_yield = screened_df["fcf_yield"].mean() if not screened_df.empty else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Scanned</div>
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

    # Main Tabs Layout
    tab1, tab2, tab3 = st.tabs([
        "📋 Screener Data Grid", 
        "📈 Technical Deep-Dive", 
        "🤖 Gemini AI Thesis & Risk Hub"
    ])

    # ----------------------------------------------------
    # TAB 1: Screener Results Data Grid
    # ----------------------------------------------------
    with tab1:
        st.subheader("Filtered Screen Results")
        
        if screened_df.empty:
            st.warning("No assets matched your exact screening criteria. Try adjusting the sliders in the sidebar.")
        else:
            # Reorder and format display dataframe
            display_cols = [
                "ticker", "name", "asset_type", "price", "pct_above_sma252", 
                "market_cap_b", "fcf_m", "profit_margin_pct", "pe_ratio", 
                f"ret_{timeframe.lower()}"
            ]
            
            # Select available columns
            valid_cols = [c for c in display_cols if c in screened_df.columns]
            grid_df = screened_df[valid_cols].copy()
            
            # Rename for display
            column_rename = {
                "ticker": "Ticker",
                "name": "Company / Name",
                "asset_type": "Type",
                "price": "Price ($)",
                "pct_above_sma252": "% Above SMA 252",
                "market_cap_b": "Market Cap ($B)",
                "fcf_m": "FCF ($M)",
                "profit_margin_pct": "Net Margin %",
                "pe_ratio": "P/E",
                f"ret_{timeframe.lower()}": f"{timeframe} Return %"
            }
            grid_df = grid_df.rename(columns=column_rename)

            st.dataframe(
                grid_df.style.format({
                    "Price ($)": "${:.2f}",
                    "% Above SMA 252": "{:+.2f}%",
                    "Market Cap ($B)": "${:.2f}B",
                    "FCF ($M)": "${:,.1f}M",
                    "Net Margin %": "{:.1f}%",
                    "P/E": "{:.1f}",
                    f"{timeframe} Return %": "{:+.2f}%"
                }, na_rep="-"),
                width="stretch",
                height=450
            )

            # Export Section
            st.markdown("##### 📥 Export Screener Results")
            col_csv, col_excel = st.columns([1, 1])
            
            with col_csv:
                csv_data = grid_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name="screener_iq_results.csv",
                    mime="text/csv"
                )
            
            with col_excel:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    grid_df.to_excel(writer, index=False, sheet_name='Screener IQ')
                excel_data = buffer.getvalue()
                st.download_button(
                    label="📊 Download Excel (.xlsx)",
                    data=excel_data,
                    file_name="screener_iq_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # ----------------------------------------------------
    # TAB 2: Technical Deep-Dive
    # ----------------------------------------------------
    with tab2:
        st.subheader("Interactive Technical Analysis Chart")
        
        selectable_df = screened_df if not screened_df.empty else full_df
        available_tickers = selectable_df["ticker"].tolist()
        
        selected_ticker = st.selectbox(
            "Select Ticker for Candlestick & Moving Average Overlay",
            options=available_tickers,
            index=0
        )

        chart_period = st.radio(
            "Chart Period",
            options=["6mo", "1y", "2y"],
            index=1,
            horizontal=True
        )

        if selected_ticker:
            with st.spinner(f"Loading chart history for {selected_ticker}..."):
                hist_data = get_ticker_historical_chart_data(selected_ticker, period=chart_period)

            if not hist_data.empty:
                # Build Plotly Subplot (Price & Volume)
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(f"{selected_ticker} Daily Price with 252-Day & 50-Day SMA", "Volume"),
                    row_heights=[0.75, 0.25]
                )

                # Candlesticks
                fig.add_trace(
                    gr.Candlestick(
                        x=hist_data['Date'],
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )

                # SMA 252 Line (Gold)
                fig.add_trace(
                    gr.Scatter(
                        x=hist_data['Date'],
                        y=hist_data['SMA_252'],
                        mode='lines',
                        name='SMA 252 (12M)',
                        line=dict(color='#eab308', width=2)
                    ),
                    row=1, col=1
                )

                # SMA 50 Line (Cyan)
                fig.add_trace(
                    gr.Scatter(
                        x=hist_data['Date'],
                        y=hist_data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='#38bdf8', width=1.5, dash='dash')
                    ),
                    row=1, col=1
                )

                # Volume Bars
                colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(hist_data['Close'], hist_data['Open'])]
                fig.add_trace(
                    gr.Bar(
                        x=hist_data['Date'],
                        y=hist_data['Volume'],
                        name="Volume",
                        marker_color=colors
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(15, 23, 42, 0.5)',
                    plot_bgcolor='rgba(15, 23, 42, 0.5)'
                )

                st.plotly_chart(fig, width="stretch")

                # Ticker Stats Summary
                meta_row = full_df[full_df["ticker"] == selected_ticker].iloc[0].to_dict()
                st.markdown("#### 📌 Key Asset Quick Stats")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Current Price", f"${meta_row.get('price', 0):.2f}")
                sc2.metric("SMA 252 Level", f"${meta_row.get('sma_252', 0):.2f}")
                sc3.metric("% Above SMA 252", f"{meta_row.get('pct_above_sma252', 0):+.2f}%")
                sc4.metric("Market Cap", f"${meta_row.get('market_cap_b', 0):.2f}B")
                sc5.metric("1-Year Return", f"{meta_row.get('ret_1y', 0):+.2f}%")
            else:
                st.error("Historical price data unavailable for selected ticker.")

    # ----------------------------------------------------
    # TAB 3: Gemini AI Breakdown Hub
    # ----------------------------------------------------
    with tab3:
        st.subheader("🤖 Google Gemini AI qualitative thesis & risk breakdown")
        st.markdown(
            "Gemini AI analyzes the top screened assets by synthesizing fundamentals, "
            "moving average technical trends, and free cash flow generation."
        )

        target_df = screened_df if not screened_df.empty else full_df
        top_n = st.slider("Select number of top assets to analyze with Gemini", min_value=1, max_value=10, value=4)

        if st.button("🚀 Run Gemini AI Analysis", type="primary"):
            with st.spinner(f"Synthesizing qualitative analysis for top {top_n} assets via Gemini 3.6 Flash..."):
                analyses: list[InvestmentAnalysis] = batch_analyze_top_assets(
                    screened_df=target_df,
                    top_n=top_n,
                    api_key=selected_api_key
                )

            for item in analyses:
                # Score Badge Styling
                score = item.sentiment_score
                if score >= 8:
                    badge_class = "badge-score-high"
                elif score >= 5:
                    badge_class = "badge-score-med"
                else:
                    badge_class = "badge-score-low"

                st.markdown(f"""
                <div class="ai-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <span style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">{item.ticker}</span>
                            <span style="font-size: 1.1rem; color: #94a3b8; margin-left: 8px;">({item.company_name})</span>
                        </div>
                        <div>
                            <span class="{badge_class}">AI Sentiment: {score}/10</span>
                            <span class="badge-suitability" style="margin-left: 8px;">{item.suitability}</span>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 16px;">
                        <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">💡 Investment Thesis & Growth Catalysts</div>
                        <div style="color: #cbd5e1; line-height: 1.6; font-size: 0.98rem;">{item.investment_thesis}</div>
                    </div>
                    
                    <div>
                        <div style="font-weight: 700; color: #f87171; margin-bottom: 4px;">⚠️ Key Headwinds & Risk Factors</div>
                        <ul style="color: #cbd5e1; margin-top: 4px; padding-left: 20px; font-size: 0.95rem;">
                            {''.join([f'<li>{r}</li>' for r in item.risk_factors])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
