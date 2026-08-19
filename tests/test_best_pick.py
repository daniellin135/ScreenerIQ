"""
tests/test_best_pick.py - Automated Unit Tests for Best Pick Recommendation & Deep Research Engine
"""

import pytest
import pandas as pd
from screener_iq.data_enricher import enrich_ticker_data, safe_float
from screener_iq.best_pick_analyst import (
    BestPickReport,
    TradeStrategy,
    CatalystEvent,
    InstitutionalSentiment,
    compute_best_pick_score,
    select_best_candidate,
    generate_mock_best_pick_report,
    generate_best_pick_report
)


def test_data_enricher():
    """Test deep data enrichment pipeline for liquid symbol."""
    enriched = enrich_ticker_data("AAPL")
    assert enriched is not None
    assert enriched["ticker"] == "AAPL"
    assert enriched["current_price"] > 0
    assert enriched["high_52w"] > 0
    assert 0 <= enriched["rsi_14"] <= 100
    assert len(enriched["fcf_quarterly_m"]) > 0
    assert len(enriched["top_institutional_holders"]) > 0


def test_candidate_scoring_and_ranking():
    """Test quantitative candidate scoring and #1 pick selection."""
    sample_df = pd.DataFrame([
        {
            "ticker": "HIGH_MOMENTUM", "name": "High Momentum Corp", "price": 100.0,
            "pct_above_sma252": 25.0, "fcf_yield": 8.0, "ret_1y": 40.0, "profit_margin_pct": 20.0
        },
        {
            "ticker": "MED_MOMENTUM", "name": "Med Momentum Corp", "price": 50.0,
            "pct_above_sma252": 5.0, "fcf_yield": 2.0, "ret_1y": 10.0, "profit_margin_pct": 8.0
        }
    ])

    best = select_best_candidate(sample_df)
    assert best is not None
    assert best["ticker"] == "HIGH_MOMENTUM"
    score_high = compute_best_pick_score(sample_df.iloc[0])
    score_med = compute_best_pick_score(sample_df.iloc[1])
    assert score_high > score_med


def test_mock_best_pick_report_pydantic_schema():
    """Test Pydantic schema validation for mock Best Pick report."""
    enriched_sample = {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "current_price": 120.0,
        "high_52w": 140.0,
        "low_52w": 80.0,
        "sma_50": 115.0,
        "sma_200": 95.0,
        "inst_float_pct": 78.5,
        "insider_activity": "Net Insider Accumulation",
        "top_institutional_holders": ["Vanguard Group", "BlackRock"]
    }

    report = generate_mock_best_pick_report(enriched_sample)
    assert isinstance(report, BestPickReport)
    assert report.ticker == "NVDA"
    assert 1 <= report.overall_confidence_score <= 100
    assert 1 <= report.risk_score <= 10
    assert isinstance(report.trade_strategy, TradeStrategy)
    assert len(report.catalysts) >= 1
    assert isinstance(report.institutional_sentiment, InstitutionalSentiment)
    assert len(report.bull_case_drivers) >= 2
    assert len(report.bear_case_risks) >= 1


def test_best_pick_report_model_tiers():
    """Test generating report with model tier options (gemini-3.6-flash, gemini-3.7-flash, gemini-3.5-flash-lite)."""
    enriched_sample = {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "current_price": 420.0,
        "high_52w": 460.0,
        "sma_50": 410.0
    }

    for model_tier in ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"]:
        report = generate_best_pick_report(enriched_sample, model_name=model_tier)
        assert isinstance(report, BestPickReport)
        assert report.ticker == "MSFT"
