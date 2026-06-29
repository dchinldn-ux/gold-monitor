# Gold Monitor Dashboard

A free Streamlit dashboard to monitor:

- GLD ETF in USD
- COMEX gold futures (`GC=F`)
- Spot gold vs USD (`XAUUSD=X`, if available)
- US Dollar Index (`DX-Y.NYB`)
- US 10-year yield (`^TNX`)
- S&P 500 risk sentiment

It includes moving averages, RSI, normalized comparison charts, and a simple rule-based gold analysis panel.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy for free

1. Create a free GitHub account.
2. Create a new repository, for example `gold-monitor-dashboard`.
3. Upload `app.py` and `requirements.txt`.
4. Go to Streamlit Community Cloud.
5. Sign in with GitHub.
6. Choose your repository and set the main file as `app.py`.
7. Click Deploy.

## Notes

- This app uses `yfinance`, which pulls Yahoo Finance market data. Free data may be delayed or temporarily unavailable.
- This is for monitoring and education only, not financial advice.
