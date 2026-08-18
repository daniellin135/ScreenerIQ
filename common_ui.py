"""
common_ui.py - Shared UI Components & Sidebar Controls for ScreenerIQ
Provides unified styling, shared sidebar filters, and session state management across all pages.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from screener import load_screener_dataset, filter_dataset

# Load environment variables
load_dotenv()


def inject_custom_css():
    """Injects modern dark glassmorphic CSS rules into the active page."""
    st.markdown("""
    <style>
        /* Dark Theme Core */
        .main {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        
        /* Header Container */
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
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }
        .kpi-value {
            font-size: 1.9rem;
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

        /* Nav Cards for Landing Page */
        .nav-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            transition: all 0.2s ease;
        }
        .nav-card:hover {
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
        }
        .nav-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 8px;
        }
        .nav-desc {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 16px;
        }

        /* AI Thesis Container */
        .ai-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(129, 140, 248, 0.2);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        
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

        /* Custom Scrollbar */
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


def render_shared_sidebar():
    """
    Renders the unified quantitative controls sidebar across all multi-page views.
    Returns:
        tuple: (full_df, screened_df, selected_api_key, filter_params)
    """
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
        value=(2.0, 10.0),
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
        return pd.DataFrame(), pd.DataFrame(), selected_api_key, {}

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

    filter_params = {
        "asset_type": asset_type,
        "min_market_cap": min_market_cap,
        "max_market_cap": max_market_cap,
        "above_sma_252": above_sma_252,
        "positive_fcf": positive_fcf,
        "min_profit_margin": min_profit_margin,
        "timeframe": timeframe,
        "min_timeframe_return": min_timeframe_return,
        "universe_preset": universe_preset
    }

    return full_df, screened_df, selected_api_key, filter_params
