"""
screener_iq/pages/1_Screener_Grid.py - Interactive Data Grid & Export Page for ScreenerIQ
"""

import io
import pandas as pd
import streamlit as st
from screener_iq.common_ui import inject_custom_css, get_current_state

st.set_page_config(page_title="ScreenerIQ | Screener Grid", page_icon="📋", layout="wide")
inject_custom_css()

# Retrieve current state from session state
full_df, screened_df, selected_api_key, filter_params = get_current_state()

st.subheader("📋 Filtered Screener Data Grid")
st.markdown("Dynamic data table featuring current market parameters and quantitative indicators.")

timeframe = filter_params.get("timeframe", "1Y")

if screened_df.empty:
    st.warning("No assets matched your exact screening criteria. Try adjusting the sliders in the sidebar.")
else:
    # Reorder and format display dataframe
    display_cols = [
        "ticker", "name", "asset_type", "price", "pct_above_sma252", 
        "market_cap_b", "fcf_m", "profit_margin_pct", "pe_ratio", 
        f"ret_{timeframe.lower()}"
    ]
    
    valid_cols = [c for c in display_cols if c in screened_df.columns]
    grid_df = screened_df[valid_cols].copy()
    
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
        height=520
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
            mime="text/csv",
            use_container_width=True
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
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # Deep Research Action
    st.markdown("---")
    st.markdown("##### 🤖 Launch Gemini Deep Research")
    col_sym, col_btn = st.columns([3, 1])
    with col_sym:
        target_sym = st.selectbox(
            "Select screened ticker for Gemini Deep Research Dossier & Trade Blueprint:",
            options=screened_df["ticker"].tolist(),
            key="grid_deep_sym"
        )
    with col_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Analyze with Gemini →", type="primary", use_container_width=True):
            st.switch_page("screener_iq/pages/3_Gemini_AI_Hub.py")
