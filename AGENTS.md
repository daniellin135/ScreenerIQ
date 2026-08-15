# Developer & AI Agent Guidelines for ScreenerIQ

Welcome to **ScreenerIQ**! This document provides essential instructions for developers and AI agents setting up the development environment, running tests, contributing to the codebase, and submitting Pull Requests (PRs).

---

## 1. Project Architecture & Tech Stack

ScreenerIQ is built using Python 3.11+ and follows a modular multi-tier structure:

* **Backend & Data Pipeline (`screener.py`)**: Handles asynchronous multi-threaded market data fetching via `yfinance` (`ThreadPoolExecutor`), calculates moving averages (SMA 252), cash flows, profit margins, and rolling cumulative timeframe returns (1M, 3M, 6M, 1Y, 3Y, YTD).
* **AI Engine (`gemini_analyst.py`)**: Uses the official `google-genai` SDK (`from google import genai`) with structured Pydantic outputs (`InvestmentAnalysis` schema) to generate qualitative investment theses, risk profiles, and sentiment scores using `gemini-3.6-flash`. Includes exponential backoff and mock fallback.
* **Frontend Web Dashboard (`app.py`)**: Streamlit interactive UI featuring dark glassmorphism styling, KPI metric cards, filter controls, Plotly candlestick charts with SMA overlays, data export (CSV/Excel), and an AI breakdown hub.
* **Configuration (`requirements.txt`, `.env`)**: Package management and environment variables.

---

## 2. Dev Environment Setup

### Prerequisites
* Python 3.11 or higher installed.
* Git.

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/daniellin135/ScreenerIQ.git
   cd ScreenerIQ
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy `.env.example` to `.env` and add your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Add your key inside `.env`:
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```

---

## 3. Running the Dashboard

To launch the Streamlit dashboard locally:
```bash
streamlit run app.py
```
The application will open automatically at `http://localhost:8501`.

---

## 4. Testing Instructions

Automated testing ensures data fetching, indicator math, and Gemini AI Pydantic structured output models function correctly.

### Run Unit Tests
```bash
pytest test_screener.py -v
```

### Key Areas to Test
1. **Indicator Calculations**: Verify SMA 252 (252 trading days) and percentage distance `% Above SMA 252`.
2. **Cash Flow Filters**: Verify stocks filtered by `Free Cash Flow > 0` and `Operating Cash Flow > 0`.
3. **Structured Gemini Output**: Verify `InvestmentAnalysis` Pydantic model parses sentiment scores (1-10), thesis strings, and risk bullet lists.
4. **Fallback Mechanism**: Test that `analyze_asset_with_gemini` gracefully returns deterministic mock analysis if `GEMINI_API_KEY` is absent or rate-limited.

---

## 5. Pull Request (PR) & Code Standards

### Branch Naming Conventions
* `feature/<feature-name>` for new features.
* `fix/<bug-name>` for bug fixes.
* `docs/<topic>` for documentation updates.

### Coding Guidelines
* **Type Annotations**: Always include type hints for function parameters and return types.
* **Error Handling**: Never fail silently on API calls or data fetching; log warnings and return clean fallback data.
* **Performance Caching**: Use `@st.cache_data` for heavy market data processing functions to keep the Streamlit interface responsive.
* **UI Consistency**: Maintain the dark modern CSS aesthetic and use predefined components.

### PR Checklist
- [ ] Code passes all `pytest` unit tests.
- [ ] `streamlit run app.py` executes without errors or warnings.
- [ ] Requirements in `requirements.txt` are up to date.
- [ ] Code is formatted cleanly with clear comments and docstrings.
