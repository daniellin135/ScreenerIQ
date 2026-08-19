"""
screener_iq/pages/2_Technical_DeepDive.py - Interactive Candlestick Charts & Technical Analysis for ScreenerIQ
"""

import pandas as pd
import plotly.graph_objects as gr
from plotly.subplots import make_subplots
import streamlit as st

from screener_iq.common_ui import inject_custom_css, get_current_state
from screener_iq.screener import get_ticker_historical_chart_data

st.set_page_config(page_title="ScreenerIQ | Technical Deep-Dive", page_icon="📈", layout="wide")
inject_custom_css()

# Retrieve current state from session state
full_df, screened_df, selected_api_key, filter_params = get_current_state()

st.subheader("📈 Interactive Technical Analysis Chart")
st.markdown("Inspect OHLC candlesticks, 252-day SMA (12M), 50-day SMA, and volume histogram for any scanned ticker.")

selectable_df = screened_df if not screened_df.empty else full_df

if selectable_df.empty:
    st.error("No market data available to chart.")
else:
    available_tickers = selectable_df["ticker"].tolist()
    
    col_select, col_period = st.columns([2, 1])
    with col_select:
        selected_ticker = st.selectbox(
            "Select Ticker for Candlestick & Moving Average Overlay",
            options=available_tickers,
            index=0
        )
    with col_period:
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
