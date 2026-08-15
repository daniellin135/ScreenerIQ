"""
screener.py - Dynamic Stock & ETF Screener Engine
Handles market data retrieval via yfinance, multi-threaded batch execution,
indicator calculation (SMA 252, Cash Flows, Margins, Timeframe Returns), and universe filtering.
"""

import concurrent.futures
import datetime
import logging
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined universe lists
DEFAULT_STOCKS = [
    # Tech & Growth
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "ORCL",
    "CRM", "ADBE", "QCOM", "PLTR", "INTU", "NOW", "AMAT", "TXN", "MU", "PANW",
    # Financials & Industrials
    "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "CAT", "GE", "UNP", "HON", "DE",
    # Consumer & Healthcare
    "WMT", "COST", "PG", "KO", "PEP", "LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO", "HD",
    # Energy & Materials
    "XOM", "CVX", "COP", "SLB", "LIN",
    # Mid-Caps / High Growth
    "NET", "CRWD", "DDOG", "SNOW", "SHOP", "SPOT", "UBER", "MDB", "SMCI", "ENPH"
]

DEFAULT_ETFS = [
    "SPY", "QQQ", "VTI", "SCHD", "IWM", "XLK", "XLE", "XLF", "XLV", "XLI", 
    "XLY", "XLP", "XLU", "XLB", "XLC", "ARKK", "VUG", "VTV", "SMH", "DIA", 
    "AGG", "TLT", "VNQ", "GLD", "JEPI"
]


def fetch_single_ticker_data(ticker: str) -> dict | None:
    """
    Fetch fundamental and historical price data for a single ticker.
    Calculates 252-day SMA, cash flows, profit margins, and rolling timeframe returns.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        
        # Determine asset type
        quote_type = info.get("quoteType", "EQUITY").upper()
        is_etf = quote_type == "ETF" or ticker in DEFAULT_ETFS

        # Extract name and market cap / AUM
        name = info.get("longName") or info.get("shortName") or ticker
        market_cap = info.get("marketCap") or info.get("netAssets") or info.get("totalAssets") or 0.0
        market_cap_b = market_cap / 1e9 if market_cap else 0.0

        # Fundamentals for stocks
        free_cash_flow = info.get("freeCashflow") or 0.0
        fcf_m = free_cash_flow / 1e6 if free_cash_flow else 0.0
        
        operating_cash_flow = info.get("operatingCashflow") or 0.0
        ocf_m = operating_cash_flow / 1e6 if operating_cash_flow else 0.0
        
        profit_margin = info.get("profitMargins") or 0.0
        profit_margin_pct = profit_margin * 100 if profit_margin else 0.0

        pe_ratio = info.get("trailingPE") or info.get("forwardPE") or np.nan

        # Expense ratio for ETFs
        expense_ratio = info.get("netExpenseRatio") or info.get("expenseRatio") or 0.0
        if expense_ratio and expense_ratio > 1:
            expense_ratio = expense_ratio / 100.0  # normalize if returned as percentage

        # Fetch 3 years of historical prices for accurate SMA 252 and 3Y return
        hist = t.history(period="3y")
        if hist.empty or len(hist) < 20:
            logger.warning(f"Insufficient history for {ticker}")
            return None

        # Clean historical data
        close_prices = hist["Close"].dropna()
        if len(close_prices) == 0:
            return None

        current_price = float(close_prices.iloc[-1])

        # Technical Indicators: SMA 252 (~12 Months) and SMA 50 (~50 Days)
        sma_252 = float(close_prices.rolling(window=252).mean().iloc[-1]) if len(close_prices) >= 252 else float(close_prices.mean())
        sma_50 = float(close_prices.rolling(window=50).mean().iloc[-1]) if len(close_prices) >= 50 else float(close_prices.mean())

        pct_above_sma252 = ((current_price - sma_252) / sma_252) * 100 if sma_252 > 0 else 0.0
        pct_above_sma50 = ((current_price - sma_50) / sma_50) * 100 if sma_50 > 0 else 0.0

        # Calculate Rolling Timeframe Returns (%)
        def calc_return(trading_days: int) -> float:
            if len(close_prices) > trading_days:
                past_price = float(close_prices.iloc[-(trading_days + 1)])
                return ((current_price - past_price) / past_price) * 100 if past_price > 0 else 0.0
            elif len(close_prices) > 1:
                first_price = float(close_prices.iloc[0])
                return ((current_price - first_price) / first_price) * 100 if first_price > 0 else 0.0
            return 0.0

        ret_1m = calc_return(21)
        ret_3m = calc_return(63)
        ret_6m = calc_return(126)
        ret_1y = calc_return(252)
        ret_3y = calc_return(756)

        # YTD Return calculation
        current_year = datetime.datetime.now().year
        ytd_prices = close_prices[close_prices.index.year == current_year]
        if not ytd_prices.empty:
            start_ytd_price = float(ytd_prices.iloc[0])
            ret_ytd = ((current_price - start_ytd_price) / start_ytd_price) * 100 if start_ytd_price > 0 else 0.0
        else:
            ret_ytd = ret_1m

        # FCF Yield (%) calculation
        fcf_yield = (free_cash_flow / market_cap) * 100 if (market_cap > 0 and free_cash_flow > 0) else 0.0

        return {
            "ticker": ticker,
            "name": name,
            "asset_type": "ETF" if is_etf else "Stock",
            "price": round(current_price, 2),
            "market_cap_b": round(market_cap_b, 2),
            "sma_252": round(sma_252, 2),
            "sma_50": round(sma_50, 2),
            "pct_above_sma252": round(pct_above_sma252, 2),
            "pct_above_sma50": round(pct_above_sma50, 2),
            "fcf_m": round(fcf_m, 2),
            "ocf_m": round(ocf_m, 2),
            "fcf_yield": round(fcf_yield, 2),
            "profit_margin_pct": round(profit_margin_pct, 2),
            "pe_ratio": round(pe_ratio, 2) if not np.isnan(pe_ratio) else np.nan,
            "expense_ratio": round(expense_ratio, 4),
            "ret_1m": round(ret_1m, 2),
            "ret_3m": round(ret_3m, 2),
            "ret_6m": round(ret_6m, 2),
            "ret_1y": round(ret_1y, 2),
            "ret_3y": round(ret_3y, 2),
            "ret_ytd": round(ret_ytd, 2),
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_universe_batch(tickers: list[str], max_workers: int = 12) -> pd.DataFrame:
    """
    Fetch batch market & financial data concurrently using ThreadPoolExecutor.
    """
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker_data, ticker): ticker for ticker in set(tickers)}
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            if data:
                records.append(data)
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    # Sort by Market Cap descending by default
    df = df.sort_values(by="market_cap_b", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_screener_dataset(custom_tickers: tuple = ()) -> pd.DataFrame:
    """
    Cached wrapper to load and compile full screener dataset for default + custom tickers.
    """
    universe = list(set(DEFAULT_STOCKS + DEFAULT_ETFS + list(custom_tickers)))
    return fetch_universe_batch(universe)


def filter_dataset(
    df: pd.DataFrame,
    asset_type: str = "Both",
    min_market_cap: float = 2.0,
    max_market_cap: float = 200.0,
    above_sma_252: bool = True,
    positive_fcf: bool = True,
    min_profit_margin: float = 0.0,
    timeframe: str = "1Y",
    min_timeframe_return: float = 0.0,
    max_expense_ratio: float = 0.50
) -> pd.DataFrame:
    """
    Applies dynamic quantitative filters based on user criteria.
    """
    if df.empty:
        return df

    filtered = df.copy()

    # 1. Asset Type Filter
    if asset_type == "Stocks":
        filtered = filtered[filtered["asset_type"] == "Stock"]
    elif asset_type == "ETFs":
        filtered = filtered[filtered["asset_type"] == "ETF"]

    # 2. Market Cap / AUM Filter
    filtered = filtered[
        (filtered["market_cap_b"] >= min_market_cap) & 
        (filtered["market_cap_b"] <= max_market_cap)
    ]

    # 3. Technical Filter: Price > SMA 252
    if above_sma_252:
        filtered = filtered[filtered["price"] > filtered["sma_252"]]

    # 4. Fundamental Filter: Free Cash Flow & Operating Cash Flow
    if positive_fcf:
        # For stocks: FCF > 0 and OCF > 0
        # For ETFs: Expense ratio <= max_expense_ratio
        stock_mask = (filtered["asset_type"] == "Stock") & (filtered["fcf_m"] > 0) & (filtered["ocf_m"] > 0)
        etf_mask = (filtered["asset_type"] == "ETF") & (filtered["expense_ratio"] <= max_expense_ratio)
        filtered = filtered[stock_mask | etf_mask]

    # 5. Profit Margin Filter (for stocks)
    if min_profit_margin != 0.0:
        stock_margin_mask = (filtered["asset_type"] == "Stock") & (filtered["profit_margin_pct"] >= min_profit_margin)
        etf_mask = (filtered["asset_type"] == "ETF")
        filtered = filtered[stock_margin_mask | etf_mask]

    # 6. Timeframe Cumulative Return Filter
    tf_col_map = {
        "1M": "ret_1m",
        "3M": "ret_3m",
        "6M": "ret_6m",
        "1Y": "ret_1y",
        "3Y": "ret_3y",
        "YTD": "ret_ytd"
    }
    col_name = tf_col_map.get(timeframe, "ret_1y")
    if col_name in filtered.columns:
        filtered = filtered[filtered[col_name] >= min_timeframe_return]

    return filtered.reset_index(drop=True)


def get_ticker_historical_chart_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical daily OHLCV price data for Plotly charting.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        # Compute SMA 252 and SMA 50 for overlay
        hist["SMA_252"] = hist["Close"].rolling(window=252, min_periods=10).mean()
        hist["SMA_50"] = hist["Close"].rolling(window=50, min_periods=5).mean()
        return hist
    except Exception as e:
        logger.error(f"Error getting chart data for {ticker}: {e}")
        return pd.DataFrame()
