"""
gemini_analyst.py - Gemini AI Integration Engine for ScreenerIQ
Integrated with official google-genai SDK using structured Pydantic outputs.
Provides quantitative-to-qualitative analysis, investment theses, and risk profiling.
Consolidates multi-asset queries into a single API call to save bandwidth and rate limits.
"""

import json
import logging
import os
import time
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvestmentAnalysis(BaseModel):
    ticker: str
    company_name: str
    sentiment_score: int = Field(
        ...,
        description="Rating from 1 to 10 based on asset fundamentals, technical momentum, and financial health (10 = strongest conviction)."
    )
    investment_thesis: str = Field(
        ...,
        description="2-3 concise, high-impact sentences outlining core growth catalysts and market positioning."
    )
    risk_factors: List[str] = Field(
        ...,
        description="2 to 4 bullet points detailing specific downside risks, headwinds, or valuation concerns."
    )
    suitability: str = Field(
        ...,
        description="Target investor profile (e.g., 'Aggressive Growth', 'Core Value', 'Dividend & Defensive', 'Tech Momentum')."
    )


class MultiAssetAnalysisResponse(BaseModel):
    analyses: List[InvestmentAnalysis] = Field(
        ...,
        description="List of structured qualitative investment analyses for each candidate asset."
    )


def generate_mock_analysis(row: dict) -> InvestmentAnalysis:
    """
    Fallback analysis generator when Gemini API Key is missing or rate limited.
    Uses asset metrics to construct a high-quality deterministic response.
    """
    ticker = row.get("ticker", "ASSET")
    name = row.get("name", ticker)
    price = row.get("price", 0.0)
    pct_sma = row.get("pct_above_sma252", 0.0)
    margin = row.get("profit_margin_pct", 0.0)
    ret_1y = row.get("ret_1y", 0.0)
    asset_type = row.get("asset_type", "Stock")

    # Score calculation logic for mock fallback
    base_score = 5
    if pct_sma > 0: base_score += 1
    if pct_sma > 15: base_score += 1
    if ret_1y > 10: base_score += 1
    if margin > 15: base_score += 1
    if row.get("fcf_m", 0) > 0: base_score += 1
    score = min(10, max(1, base_score))

    if asset_type == "ETF":
        thesis = (
            f"{name} ({ticker}) offers diversified market exposure currently trading at ${price:.2f}, "
            f"up {ret_1y:.1f}% over the trailing year. The fund maintains strong momentum holding {pct_sma:.1f}% "
            f"above its 252-day moving average, making it an attractive core portfolio compounder."
        )
        risks = [
            "Macroeconomic volatility and broader market drawdown risk",
            "Sector concentration sensitivity",
            "Interest rate policy shift impact"
        ]
        suitability = "Core Wealth Accumulation / Broad Market"
    else:
        thesis = (
            f"{name} ({ticker}) demonstrates strong quantitative characteristics with a {margin:.1f}% net profit margin "
            f"and positive free cash flow. Currently trading {pct_sma:.1f}% above its 252-day SMA with a 1-Year return "
            f"of {ret_1y:.1f}%, the stock shows robust operational efficiency and multi-quarter momentum."
        )
        risks = [
            "Potential multiple compression if earnings growth decelerates",
            "Macro supply chain and sector-wide competitive pressure",
            "Market volatility surrounding upcoming earnings releases"
        ]
        if margin > 20 and ret_1y > 20:
            suitability = "High Quality Tech Momentum"
        elif margin > 10:
            suitability = "Quality Growth & Value"
        else:
            suitability = "Tactical Momentum / Turnaround"

    return InvestmentAnalysis(
        ticker=ticker,
        company_name=name,
        sentiment_score=score,
        investment_thesis=thesis,
        risk_factors=risks,
        suitability=suitability
    )


def analyze_asset_with_gemini(
    row: dict,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash"
) -> InvestmentAnalysis:
    """
    Calls Google Gemini API for a single asset using structured Pydantic output.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or not GENAI_AVAILABLE:
        logger.info(f"Using mock fallback analysis for {row.get('ticker')} (No valid Gemini API key or SDK)")
        return generate_mock_analysis(row)

    ticker = row.get("ticker", "N/A")
    name = row.get("name", ticker)
    price = row.get("price", 0.0)
    market_cap = row.get("market_cap_b", 0.0)
    pct_sma252 = row.get("pct_above_sma252", 0.0)
    fcf_m = row.get("fcf_m", 0.0)
    margin = row.get("profit_margin_pct", 0.0)
    pe = row.get("pe_ratio", "N/A")
    ret_1y = row.get("ret_1y", 0.0)
    ret_3m = row.get("ret_3m", 0.0)
    asset_type = row.get("asset_type", "Stock")

    prompt = f"""
You are a top Wall Street equity research analyst evaluating {name} ({ticker}).
Review the following quantitative indicators and provide a structured investment synthesis:

Key Asset Metrics:
- Asset Type: {asset_type}
- Current Price: ${price:.2f}
- Market Cap / AUM: ${market_cap:.2f} Billion
- Technical Position: {pct_sma252:.2f}% above 252-day SMA (12-Month Moving Average)
- Free Cash Flow: ${fcf_m:.2f} Million
- Net Profit Margin: {margin:.2f}%
- P/E Ratio: {pe}
- 1-Year Cumulative Return: {ret_1y:.2f}%
- 3-Month Momentum Return: {ret_3m:.2f}%

Deliver a rigorous investment analysis following the requested schema:
1. sentiment_score: 1 to 10 rating based on financial strength and momentum.
2. investment_thesis: Exactly 2 to 3 concise, impactful sentences outlining growth catalysts and qualitative edge.
3. risk_factors: 2 to 4 bullet points outlining key downside risks.
4. suitability: Recommended investor profile.
"""

    max_retries = 3
    backoff = 2

    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=key)
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InvestmentAnalysis,
                    temperature=0.3,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )

            if response and response.text:
                data = json.loads(response.text)
                return InvestmentAnalysis(**data)
        except Exception as e:
            err_msg = str(e).lower()
            logger.warning(f"Gemini API call failed for {ticker} using {model_name}: {e}")
            if any(k in err_msg for k in ["429", "503", "resource_exhausted", "unavailable", "rate limit", "quota"]):
                logger.info(f"Immediate fallback to mock synthesis for {ticker} due to rate limit/capacity error.")
                return generate_mock_analysis(row)
            time.sleep(backoff)
            backoff *= 2

    logger.error(f"All Gemini API attempts failed for {ticker}. Falling back to deterministic synthesis.")
    return generate_mock_analysis(row)


def batch_analyze_top_assets(
    screened_df: pd.DataFrame,
    top_n: int = 5,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash"
) -> List[InvestmentAnalysis]:
    """
    Runs Gemini AI analysis on top N screened assets using a SINGLE consolidated API request
    to save API bandwidth and stay strictly within free-tier Rate Limits (RPM).
    """
    if screened_df.empty:
        return []

    top_assets = screened_df.head(top_n).to_dict(orient="records")
    key = api_key or os.getenv("GEMINI_API_KEY")

    # If key missing or SDK unavailable, return deterministic mock analysis for each
    if not key or not GENAI_AVAILABLE:
        logger.info(f"Using mock fallback batch analysis for top {len(top_assets)} assets (No valid Gemini API key)")
        return [generate_mock_analysis(asset) for asset in top_assets]

    # Build consolidated multi-asset payload prompt
    asset_summaries = []
    for idx, row in enumerate(top_assets, 1):
        asset_summaries.append(
            f"Asset #{idx}: {row.get('name', row.get('ticker'))} ({row.get('ticker')})\n"
            f"- Asset Type: {row.get('asset_type', 'Stock')}\n"
            f"- Price: ${row.get('price', 0.0):.2f}\n"
            f"- Market Cap: ${row.get('market_cap_b', 0.0):.2f}B\n"
            f"- Technical Position: {row.get('pct_above_sma252', 0.0):.2f}% above 252-day SMA\n"
            f"- Free Cash Flow: ${row.get('fcf_m', 0.0):.2f}M\n"
            f"- Profit Margin: {row.get('profit_margin_pct', 0.0):.2f}%\n"
            f"- P/E Ratio: {row.get('pe_ratio', 'N/A')}\n"
            f"- 1Y Return: {row.get('ret_1y', 0.0):.2f}% | 3M Return: {row.get('ret_3m', 0.0):.2f}%\n"
        )

    combined_prompt = f"""
You are a senior institutional equity research analyst evaluating a portfolio of candidate assets.
Review the following {len(top_assets)} candidate assets and provide a structured investment synthesis for EACH asset:

{'='*40}
{chr(10).join(asset_summaries)}
{'='*40}

For EACH asset, deliver a rigorous InvestmentAnalysis complying strictly with the requested schema:
1. ticker: Symbol of the asset.
2. company_name: Name of the company/fund.
3. sentiment_score: 1 to 10 rating based on financial strength and technical momentum.
4. investment_thesis: Exactly 2 to 3 concise sentences detailing core growth catalysts.
5. risk_factors: 2 to 4 bullet points detailing specific downside risks.
6. suitability: Recommended investor profile.
"""

    max_retries = 3
    backoff = 2
    effective_model = model_name

    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=effective_model,
                contents=combined_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MultiAssetAnalysisResponse,
                    temperature=0.3,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )

            if response and response.text:
                data = json.loads(response.text)
                parsed = MultiAssetAnalysisResponse(**data)
                if parsed.analyses and len(parsed.analyses) > 0:
                    return parsed.analyses
        except Exception as e:
            err_msg = str(e).lower()
            logger.warning(f"Batch Gemini API attempt failed using {effective_model}: {e}")
            if any(k in err_msg for k in ["429", "503", "resource_exhausted", "unavailable", "rate limit", "quota"]):
                logger.info(f"Immediate fallback to mock batch synthesis due to rate limit/capacity error.")
                return [generate_mock_analysis(asset) for asset in top_assets]
            time.sleep(backoff)
            backoff *= 2

    logger.error("All batch Gemini API attempts failed. Falling back to deterministic synthesis.")
    return [generate_mock_analysis(asset) for asset in top_assets]
