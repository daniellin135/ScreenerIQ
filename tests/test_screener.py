"""
tests/test_screener.py - Automated Unit Tests for ScreenerIQ Data Engine & Gemini Integration
"""

import pytest
import pandas as pd
import streamlit as st

from screener_iq.screener import (
    fetch_single_ticker_data,
    filter_dataset,
    CORE_STOCKS,
    CORE_ETFS
)
from screener_iq.gemini_analyst import (
    InvestmentAnalysis,
    generate_mock_analysis,
    analyze_asset_with_gemini,
    batch_analyze_top_assets
)
from screener_iq.common_ui import get_current_state


def test_fetch_single_stock():
    """Test fetching data for a liquid stock symbol."""
    data = fetch_single_ticker_data("AAPL")
    assert data is not None
    assert data["ticker"] == "AAPL"
    assert data["asset_type"] == "Stock"
    assert data["price"] > 0
    assert data["market_cap_b"] > 0
    assert data["sma_252"] > 0
    assert "ret_1y" in data


def test_fetch_single_etf():
    """Test fetching data for an ETF symbol."""
    data = fetch_single_ticker_data("SPY")
    assert data is not None
    assert data["ticker"] == "SPY"
    assert data["asset_type"] == "ETF"
    assert data["price"] > 0
    assert data["market_cap_b"] > 0


def test_filter_dataset():
    """Test dynamic quantitative screening filters."""
    # Create sample dataframe
    sample_data = pd.DataFrame([
        {
            "ticker": "AAPL", "asset_type": "Stock", "price": 180.0, "sma_252": 170.0,
            "pct_above_sma252": 5.88, "market_cap_b": 2800.0, "fcf_m": 100000.0,
            "ocf_m": 110000.0, "profit_margin_pct": 25.0, "expense_ratio": 0.0,
            "ret_1y": 15.0, "ret_3m": 5.0
        },
        {
            "ticker": "MID1", "asset_type": "Stock", "price": 50.0, "sma_252": 45.0,
            "pct_above_sma252": 11.11, "market_cap_b": 5.0, "fcf_m": 300.0,
            "ocf_m": 400.0, "profit_margin_pct": 12.0, "expense_ratio": 0.0,
            "ret_1y": 20.0, "ret_3m": 8.0
        },
        {
            "ticker": "WEAK", "asset_type": "Stock", "price": 20.0, "sma_252": 25.0,
            "pct_above_sma252": -20.0, "market_cap_b": 1.0, "fcf_m": -50.0,
            "ocf_m": -20.0, "profit_margin_pct": -10.0, "expense_ratio": 0.0,
            "ret_1y": -30.0, "ret_3m": -15.0
        },
        {
            "ticker": "SPY", "asset_type": "ETF", "price": 500.0, "sma_252": 480.0,
            "pct_above_sma252": 4.17, "market_cap_b": 500.0, "fcf_m": 0.0,
            "ocf_m": 0.0, "profit_margin_pct": 0.0, "expense_ratio": 0.09,
            "ret_1y": 22.0, "ret_3m": 6.0
        }
    ])

    # Filter 1: Mid-cap range ($2B to $10B)
    mid_cap_filtered = filter_dataset(
        sample_data, min_market_cap=2.0, max_market_cap=10.0, above_sma_252=True, positive_fcf=True
    )
    assert len(mid_cap_filtered) == 1
    assert mid_cap_filtered.iloc[0]["ticker"] == "MID1"

    # Filter 2: Price > SMA 252 and Positive FCF
    strong_filtered = filter_dataset(
        sample_data, min_market_cap=0.0, max_market_cap=5000.0, above_sma_252=True, positive_fcf=True
    )
    assert len(strong_filtered) == 3
    assert "WEAK" not in strong_filtered["ticker"].values


def test_gemini_analysis_fallback():
    """Test qualitative Gemini analysis fallback generator."""
    sample_row = {
        "ticker": "NVDA", "name": "NVIDIA Corporation", "price": 120.0,
        "pct_above_sma252": 25.0, "profit_margin_pct": 55.0, "ret_1y": 150.0,
        "asset_type": "Stock", "fcf_m": 25000.0
    }
    
    result = generate_mock_analysis(sample_row)
    assert isinstance(result, InvestmentAnalysis)
    assert result.ticker == "NVDA"
    assert 1 <= result.sentiment_score <= 10
    assert len(result.investment_thesis) > 20
    assert len(result.risk_factors) >= 2
    assert result.suitability != ""


def test_batch_analyze_top_assets_single_request():
    """Test batch qualitative analysis generator returns structured list for multi-asset queries."""
    sample_df = pd.DataFrame([
        {"ticker": "NVDA", "name": "NVIDIA Corp", "price": 120.0, "pct_above_sma252": 25.0, "profit_margin_pct": 55.0, "ret_1y": 150.0, "asset_type": "Stock", "fcf_m": 25000.0},
        {"ticker": "AAPL", "name": "Apple Inc", "price": 180.0, "pct_above_sma252": 10.0, "profit_margin_pct": 25.0, "ret_1y": 20.0, "asset_type": "Stock", "fcf_m": 100000.0}
    ])

    results = batch_analyze_top_assets(sample_df, top_n=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].ticker in ["NVDA", "AAPL"]
    assert results[1].ticker in ["NVDA", "AAPL"]


def test_sidebar_filter_state_persistence():
    """Test that user-configured quantitative filters persist across page views."""
    custom_params = {
        "asset_type": "Stocks",
        "min_market_cap": 10.0,
        "max_market_cap": 250.0,
        "above_sma_252": True,
        "positive_fcf": True,
        "min_profit_margin": 15.0,
        "timeframe": "3M",
        "min_timeframe_return": 10.0,
        "universe_preset": "S&P 500 & Nasdaq 100 Leaders (~250 Assets)"
    }
    
    sample_full_df = pd.DataFrame([{"ticker": "NVDA", "price": 120.0, "market_cap_b": 2800.0}])
    sample_screened_df = pd.DataFrame([{"ticker": "NVDA", "price": 120.0, "market_cap_b": 2800.0}])
    
    # Store in session state as common_ui root level does
    st.session_state["state_full_df"] = sample_full_df
    st.session_state["state_screened_df"] = sample_screened_df
    st.session_state["state_api_key"] = "test_key_123"
    st.session_state["state_filter_params"] = custom_params

    # Simulate child page 1 reading state
    full1, screened1, key1, params1 = get_current_state()
    assert params1["asset_type"] == "Stocks"
    assert params1["min_market_cap"] == 10.0
    assert params1["max_market_cap"] == 250.0
    assert params1["timeframe"] == "3M"
    assert params1["universe_preset"] == "S&P 500 & Nasdaq 100 Leaders (~250 Assets)"
    assert key1 == "test_key_123"
    assert len(screened1) == 1

    # Simulate navigating to child page 2 reading state
    full2, screened2, key2, params2 = get_current_state()
    assert params2 == custom_params
    assert key2 == "test_key_123"
    assert len(screened2) == 1
