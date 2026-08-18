"""
pages/3_Gemini_AI_Hub.py - Google Gemini AI Qualitative Synthesis Hub for ScreenerIQ
"""

import textwrap
import streamlit as st
from common_ui import inject_custom_css, render_shared_sidebar
from gemini_analyst import batch_analyze_top_assets, InvestmentAnalysis

st.set_page_config(page_title="ScreenerIQ | Gemini AI Hub", page_icon="🤖", layout="wide")
inject_custom_css()

# Render Shared Sidebar
full_df, screened_df, selected_api_key, filter_params = render_shared_sidebar()

st.subheader("🤖 Google Gemini AI qualitative thesis & risk breakdown")
st.markdown(
    "Gemini AI evaluates the top screened assets by synthesizing fundamental metrics, "
    "moving average technical trends, and free cash flow generation."
)

target_df = screened_df if not screened_df.empty else full_df

if target_df.empty:
    st.warning("No market data available for AI evaluation.")
else:
    top_n = st.slider("Select number of top assets to analyze with Gemini", min_value=1, max_value=10, value=4)

    if st.button("🚀 Run Gemini AI Analysis", type="primary"):
        with st.spinner(f"Synthesizing qualitative analysis for top {top_n} assets via Gemini 3.6 Flash..."):
            analyses: list[InvestmentAnalysis] = batch_analyze_top_assets(
                screened_df=target_df,
                top_n=top_n,
                api_key=selected_api_key
            )
            st.session_state["gemini_analyses"] = analyses

    # Display cached or newly generated AI analyses
    if "gemini_analyses" in st.session_state and st.session_state["gemini_analyses"]:
        for item in st.session_state["gemini_analyses"]:
            score = item.sentiment_score
            if score >= 8:
                score_color = "#4ade80"
                score_bg = "rgba(34, 197, 94, 0.2)"
            elif score >= 5:
                score_color = "#facc15"
                score_bg = "rgba(234, 179, 8, 0.2)"
            else:
                score_color = "#f87171"
                score_bg = "rgba(239, 68, 68, 0.2)"

            with st.container(border=True):
                # Header row: Ticker, Company Name, Sentiment Badge, Suitability Tag
                col_header1, col_header2 = st.columns([3, 2])
                with col_header1:
                    st.markdown(
                        f"### {item.ticker} <span style='color: #94a3b8; font-size: 1.1rem; font-weight: normal;'>({item.company_name})</span>",
                        unsafe_allow_html=True
                    )
                with col_header2:
                    st.markdown(
                        f"<div style='text-align: right; padding-top: 8px;'>"
                        f"<span style='background: {score_bg}; color: {score_color}; border: 1px solid {score_color}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.95rem; margin-right: 8px;'>AI Sentiment: {score}/10</span>"
                        f"<span style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 600;'>{item.suitability}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<hr style='margin: 12px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

                # Investment Thesis
                st.markdown("<div style='font-weight: 700; color: #38bdf8; margin-bottom: 4px;'>💡 Investment Thesis & Growth Catalysts</div>", unsafe_allow_html=True)
                st.markdown(f"> {item.investment_thesis}")

                # Key Downside Risks
                st.markdown("<div style='font-weight: 700; color: #f87171; margin-top: 12px; margin-bottom: 4px;'>⚠️ Key Headwinds & Risk Factors</div>", unsafe_allow_html=True)
                for risk in item.risk_factors:
                    st.markdown(f"- {risk}")
