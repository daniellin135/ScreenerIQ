"""
screener_iq/pages/3_Gemini_AI_Hub.py - Gemini AI Deep Research & Best Pick Recommendation Engine
Integrates Google Gemini (gemini-3.6-flash, gemini-3.7-flash, antigravity) with Pydantic structured outputs,
Google Search Grounding, Trade Blueprint Cards, Catalyst Radar, and downloadable dossiers.
"""

import streamlit as st
from screener_iq.common_ui import inject_custom_css, get_current_state
from screener_iq.data_enricher import enrich_ticker_data
from screener_iq.best_pick_analyst import (
    select_best_candidate,
    generate_best_pick_report,
    BestPickReport,
    compute_best_pick_score
)
from screener_iq.gemini_analyst import batch_analyze_top_assets, InvestmentAnalysis

st.set_page_config(page_title="ScreenerIQ | Gemini AI Hub", page_icon="🤖", layout="wide")
inject_custom_css()

# Retrieve current state from session state
full_df, screened_df, selected_api_key, filter_params = get_current_state()

st.subheader("🤖 Google Gemini AI Intelligence Hub")
st.markdown("Autonomous 'Best Pick' Recommendation, Trade Blueprinting, and Qualitative Synthesis Engine.")

target_df = screened_df if not screened_df.empty else full_df

if target_df.empty:
    st.warning("No market data available for AI evaluation. Adjust sidebar filters to scan assets.")
else:
    # Model Selector Bar
    col_model_info, col_model_choice = st.columns([2, 1])
    with col_model_info:
        st.markdown("Select Gemini model tier for AI deep research and thesis synthesis:")
    with col_model_choice:
        selected_model = st.selectbox(
            "Gemini Model Tier",
            options=["gemini-3.6-flash", "gemini-3.7-flash", "antigravity"],
            index=0,
            key="hub_model_selector",
            help="Choose model tier: gemini-3.6-flash (Fast & Structured), gemini-3.7-flash (Advanced Reasoning), or antigravity (Flagship Tier)"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-tab layout
    tab_best_pick, tab_quick_thesis = st.tabs([
        "🏆 Autonomous AI Best Pick & Deep Dossier",
        "📋 Quick Multi-Asset AI Thesis"
    ])

    # =========================================================================
    # TAB 1: AUTONOMOUS AI BEST PICK & DEEP RESEARCH DOSSIER
    # =========================================================================
    with tab_best_pick:
        # Candidate ranking and top pick detection
        best_candidate = select_best_candidate(target_df)
        best_ticker = best_candidate.get("ticker") if best_candidate else target_df.iloc[0]["ticker"]
        best_score = compute_best_pick_score(best_candidate) if best_candidate else 0.0

        # Candidate selection banner
        st.markdown("### 🏆 Top Quantitative Candidate Selection")
        
        col_cand1, col_cand2 = st.columns([3, 2])
        with col_cand1:
            st.info(
                f"**Auto-Detected #1 Opportunity:** **{best_ticker}** ({best_candidate.get('name', best_ticker)}) "
                f"with quantitative composite score of **{best_score}**."
            )
        with col_cand2:
            all_tickers = target_df["ticker"].tolist()
            default_idx = all_tickers.index(best_ticker) if best_ticker in all_tickers else 0
            chosen_ticker = st.selectbox(
                "Or manually pick any candidate symbol to analyze:",
                options=all_tickers,
                index=default_idx,
                key="select_best_pick_symbol"
            )

        # Trigger AI Generation Button
        if st.button("🚀 Generate Autonomous AI Deep Research Dossier", type="primary", use_container_width=True):
            with st.spinner(f"Enriching financial data & synthesizing deep dossier for {chosen_ticker} via {selected_model}..."):
                enriched = enrich_ticker_data(chosen_ticker)
                report: BestPickReport = generate_best_pick_report(
                    enriched_data=enriched,
                    model_name=selected_model,
                    api_key=selected_api_key
                )
                st.session_state["active_best_pick_report"] = report
                st.session_state["active_enriched_data"] = enriched

        # Check if report exists in session state
        active_report: BestPickReport = st.session_state.get("active_best_pick_report")
        active_enriched = st.session_state.get("active_enriched_data", {})

        if active_report and active_report.ticker == chosen_ticker:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- HEADER BADGES & CONFIDENCE / RISK GAUGE ---
            conf = active_report.overall_confidence_score
            risk = active_report.risk_score

            conf_color = "#4ade80" if conf >= 75 else "#facc15" if conf >= 50 else "#f87171"
            risk_color = "#4ade80" if risk <= 3 else "#facc15" if risk <= 6 else "#f87171"

            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Analyzed Asset</div>
                    <div class="kpi-value" style="color: #38bdf8;">{active_report.ticker}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">{active_report.company_name}</div>
                </div>
                """, unsafe_allow_html=True)

            with h2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Current Price / 52W High</div>
                    <div class="kpi-value" style="color: #f8fafc;">${active_enriched.get('current_price', 0):.2f}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">High: ${active_enriched.get('high_52w', 0):.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with h3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">AI Conviction Score</div>
                    <div class="kpi-value" style="color: {conf_color};">{conf}/100</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Confidence Gauge</div>
                </div>
                """, unsafe_allow_html=True)

            with h4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Risk Rating</div>
                    <div class="kpi-value" style="color: {risk_color};">{risk}/10</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Risk Assessment Meter</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- TRADE BLUEPRINT CARD ---
            strat = active_report.trade_strategy
            st.markdown("### 🎯 Trade Blueprint & Execution Parameters")
            
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Optimal Entry Range", strat.recommended_entry_range)
            b2.metric("Stop-Loss Target", strat.stop_loss_target)
            b3.metric("Short-Term Target (1-3M)", strat.target_price_short_term)
            b4.metric("Long-Term Target (6-12M)", strat.target_price_long_term)
            b5.metric("Risk : Reward Ratio", strat.risk_reward_ratio)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- CATALYST RADAR & INSTITUTIONAL SNAPSHOT ---
            col_cat, col_inst = st.columns([1, 1])

            with col_cat:
                with st.container(border=True):
                    st.markdown("#### 📡 Catalyst Radar & Event Timeline")
                    for cat in active_report.catalysts:
                        impact_color = "#4ade80" if "bullish" in cat.potential_impact.lower() else "#f87171"
                        st.markdown(
                            f"**• {cat.event_type}** (`{cat.expected_date}`)\n"
                            f"<span style='color: {impact_color}; font-size: 0.9rem;'>Impact: {cat.potential_impact}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("---")

            with col_inst:
                with st.container(border=True):
                    st.markdown("#### 🏛️ Institutional & Insider Snapshot")
                    inst_info = active_report.institutional_sentiment
                    st.markdown(f"**Verdict:** {inst_info.institutional_backing_verdict}")
                    st.markdown(f"**Top Institutional Holders:** {inst_info.key_holders_overview}")
                    st.markdown(f"**Institutional Float:** `{active_enriched.get('inst_float_pct', 0):.1f}%` float held")
                    st.markdown(f"**Insider Activity:** `{active_enriched.get('insider_activity', 'Steady')}`")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- COMPREHENSIVE AI THESIS & RISKS ---
            st.markdown("### 💡 Executive Investment Thesis & Risk Breakdown")
            
            with st.container(border=True):
                st.markdown("#### 📝 Executive Summary")
                st.markdown(active_report.executive_summary)
                
                st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
                
                c_bull, c_bear = st.columns(2)
                with c_bull:
                    st.markdown("#### 🚀 Primary Bull Case Drivers")
                    for driver in active_report.bull_case_drivers:
                        st.markdown(f"- {driver}")
                
                with c_bear:
                    st.markdown("#### ⚠️ Key Downside Risks & Headwinds")
                    for rk in active_report.bear_case_risks:
                        st.markdown(f"- {rk}")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- ONE-CLICK DOSSIER DOWNLOAD ---
            dossier_md = f"""# ScreenerIQ AI Deep Research Dossier: {active_report.company_name} ({active_report.ticker})
Generated via Google Gemini ({selected_model})

## Overview
- **Ticker:** {active_report.ticker}
- **Current Price:** ${active_enriched.get('current_price', 0):.2f}
- **AI Conviction Score:** {conf}/100
- **Risk Score:** {risk}/10

## Trade Blueprint
- **Entry Range:** {strat.recommended_entry_range}
- **Stop-Loss Target:** {strat.stop_loss_target}
- **Short-Term Target (1-3M):** {strat.target_price_short_term}
- **Long-Term Target (6-12M):** {strat.target_price_long_term}
- **Risk : Reward:** {strat.risk_reward_ratio}

## Executive Summary
{active_report.executive_summary}

## Bull Case Drivers
{chr(10).join(['- ' + d for d in active_report.bull_case_drivers])}

## Downside Risks
{chr(10).join(['- ' + r for r in active_report.bear_case_risks])}

## Institutional & Insider Activity
- {active_report.institutional_sentiment.institutional_backing_verdict}
- Key Holders: {active_report.institutional_sentiment.key_holders_overview}
"""

            st.download_button(
                label=f"📄 Download Complete {active_report.ticker} Research Dossier (.md)",
                data=dossier_md.encode("utf-8"),
                file_name=f"{active_report.ticker}_Gemini_Deep_Research_Dossier.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )

    # =========================================================================
    # TAB 2: QUICK MULTI-ASSET AI THESIS
    # =========================================================================
    with tab_quick_thesis:
        st.markdown("### 📋 Quick Qualitative Synthesis Cards")
        top_n = st.slider("Select number of top assets to analyze", min_value=1, max_value=10, value=4, key="quick_slider")

        if st.button("🚀 Run Multi-Asset Gemini Analysis", type="primary", key="btn_quick_run"):
            with st.spinner(f"Synthesizing qualitative analysis for top {top_n} assets via {selected_model}..."):
                analyses: list[InvestmentAnalysis] = batch_analyze_top_assets(
                    screened_df=target_df,
                    top_n=top_n,
                    api_key=selected_api_key,
                    model_name=selected_model
                )
                st.session_state["gemini_analyses"] = analyses

        # Display cached or newly generated AI analyses
        if "gemini_analyses" in st.session_state and st.session_state["gemini_analyses"]:
            for item in st.session_state["gemini_analyses"]:
                score = item.sentiment_score
                score_color = "#4ade80" if score >= 8 else "#facc15" if score >= 5 else "#f87171"
                score_bg = "rgba(34, 197, 94, 0.2)" if score >= 8 else "rgba(234, 179, 8, 0.2)" if score >= 5 else "rgba(239, 68, 68, 0.2)"

                with st.container(border=True):
                    col_h1, col_h2 = st.columns([3, 2])
                    with col_h1:
                        st.markdown(
                            f"### {item.ticker} <span style='color: #94a3b8; font-size: 1.1rem; font-weight: normal;'>({item.company_name})</span>",
                            unsafe_allow_html=True
                        )
                    with col_h2:
                        st.markdown(
                            f"<div style='text-align: right; padding-top: 8px;'>"
                            f"<span style='background: {score_bg}; color: {score_color}; border: 1px solid {score_color}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.95rem; margin-right: 8px;'>AI Sentiment: {score}/10</span>"
                            f"<span style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 600;'>{item.suitability}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("<hr style='margin: 12px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-weight: 700; color: #38bdf8; margin-bottom: 4px;'>💡 Investment Thesis</div>", unsafe_allow_html=True)
                    st.markdown(f"> {item.investment_thesis}")

                    st.markdown("<div style='font-weight: 700; color: #f87171; margin-top: 12px; margin-bottom: 4px;'>⚠️ Key Headwinds & Risk Factors</div>", unsafe_allow_html=True)
                    for rk in item.risk_factors:
                        st.markdown(f"- {rk}")
