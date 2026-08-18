# ScreenerIQ — Stock & ETF Screener with Gemini AI Engine

ScreenerIQ is a production-grade, interactive Python web application that screens US Stocks and ETFs using quantitative technical (12-Month Simple Moving Average) and fundamental indicators (Free Cash Flow, Operating Cash Flow, Profit Margins, Timeframe Cumulative Returns), integrated with the Google Gemini API (`google-genai` SDK) to provide deep qualitative investment theses and downside risk profiling.

---

## Key Features

* **Dynamic Screener Engine**: Filters S&P 500, Nasdaq 100, Russell 2000 mid-caps, and major ETFs (SPY, QQQ, VTI, SCHD, IWM, XLK, XLE, etc.).
* **Technical Indicators**: Calculates 252-day Simple Moving Average (SMA 252, ~12 Months) and distance `% Above SMA 252`.
* **Fundamental Filtering**: Filters assets by Market Cap ranges ($2B–$10B mid-caps up to $300B+ mega-caps), Free Cash Flow (`FCF > 0`), Operating Cash Flow (`OCF > 0`), Net Profit Margins, and ETF Expense Ratios.
* **Rolling Timeframe Returns**: Multi-period historical return analysis across 1M, 3M, 6M, 1Y, 3Y, and YTD timeframes.
* **Google Gemini AI Engine (`gemini-3.6-flash`)**: Synthesizes quantitative data into structured qualitative analyses returning:
  * **Sentiment Score**: 1 to 10 rating based on financial health and momentum.
  * **Investment Thesis**: 2–3 concise sentences outlining core growth catalysts.
  * **Key Downside Risks**: Bullet points detailing potential headwinds.
  * **Investor Suitability**: Recommended profile (e.g. *High Quality Tech Momentum*, *Core Value*, *Defensive Dividend*).
* **Technical Deep-Dive**: Interactive Plotly candlestick charts featuring 252-day SMA (gold) and 50-day SMA (cyan) overlays with volume histograms.
* **One-Click Data Export**: Download filtered screening results to CSV or formatted Excel (`.xlsx`).

---

## System Architecture & Tech Stack

* **Language & Runtime**: Python 3.11+
* **Market Data Pipeline**: `yfinance`, `pandas`, `numpy`
* **Concurrency & Speed**: `concurrent.futures.ThreadPoolExecutor` multi-threading with Streamlit in-memory caching (`st.cache_data`)
* **AI Integration**: `google-genai` SDK (`gemini-3.6-flash`) with Pydantic structured output validation
* **Frontend Web Dashboard**: Streamlit with custom CSS dark glassmorphic layout
* **Visualization**: Plotly Interactive Subplots

---

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/daniellin135/ScreenerIQ.git
cd ScreenerIQ
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Note: If no key is configured, ScreenerIQ runs in offline preview mode with deterministic qualitative analysis).*

### 5. Launch the Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Running Automated Unit Tests

Run the pytest test suite to verify data fetching, indicator math, and Gemini Pydantic schemas:
```bash
python -m pytest tests/test_screener.py -v
```

---

## Project Structure

```
ScreenerIQ/
├── app.py                      # Application entrypoint launcher & navigation
├── screener_iq/                # Main Python package
│   ├── __init__.py
│   ├── screener.py             # Quantitative screening engine & yfinance pipeline
│   ├── gemini_analyst.py       # Google Gemini AI engine & Pydantic structured output
│   ├── common_ui.py            # Shared CSS styling & sidebar filters
│   └── pages/                  # Multi-page views
│       ├── 0_Home.py           # Landing page & KPI overview
│       ├── 1_Screener_Grid.py   # Screener data grid & CSV/Excel exports
│       ├── 2_Technical_DeepDive.py # Candlestick charts & SMA overlays
│       └── 3_Gemini_AI_Hub.py  # Gemini AI qualitative synthesis cards
├── tests/                      # Automated unit test suite
│   ├── __init__.py
│   └── test_screener.py        # Pytest test suite
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template file
├── AGENTS.md                   # Developer & AI Agent contributor guidelines
└── README.md                   # Project documentation
```