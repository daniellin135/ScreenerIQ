"""
pages/3_Gemini_AI_Hub.py - Google Gemini AI Qualitative Synthesis Hub for ScreenerIQ
"""

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

        for item in analyses:
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
