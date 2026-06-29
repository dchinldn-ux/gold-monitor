import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Monitor Dashboard", page_icon="🥇", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
@dataclass
class Asset:
    label: str
    ticker: str
    note: str = ""

ASSETS = {
    "GLD ETF (USD)": Asset("SPDR Gold Shares ETF", "GLD", "Gold ETF traded in USD"),
    "Gold Futures": Asset("COMEX Gold Futures", "GC=F", "Futures price; often delayed on free data"),
    "Spot Gold USD": Asset("Spot Gold vs USD", "XAUUSD=X", "May be unavailable depending on Yahoo data"),
    "US Dollar Index": Asset("US Dollar Index", "DX-Y.NYB", "Gold usually faces pressure when USD rises"),
    "US 10Y Yield": Asset("US 10-Year Treasury Yield", "^TNX", "Yahoo quotes ^TNX as yield x10; app converts it to %"),
    "S&P 500": Asset("S&P 500", "^GSPC", "Risk appetite reference"),
}

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]

@st.cache_data(ttl=900)
def load_data(tickers, period="1y"):
    """Download adjusted market data using yfinance. Cached for 15 minutes."""
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(tickers, str):
        tickers = [tickers]

    out = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            if df.empty:
                continue
            df = df.dropna(how="all")
            if "Close" not in df.columns:
                continue
            df = df.rename(columns=str.title)
            out[t] = df
        except Exception:
            continue
    return out


def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["Daily Return %"] = close.pct_change() * 100
    df["Drawdown %"] = (close / close.cummax() - 1) * 100
    return df


def fmt(x, digits=2):
    if pd.isna(x):
        return "n/a"
    return f"{x:,.{digits}f}"


def pct(x, digits=2):
    if pd.isna(x):
        return "n/a"
    return f"{x:,.{digits}f}%"


def latest_snapshot(df):
    df = add_indicators(df)
    last = df.dropna(subset=["Close"]).iloc[-1]
    prev = df.dropna(subset=["Close"]).iloc[-2] if len(df.dropna(subset=["Close"])) > 1 else last
    return {
        "price": last["Close"],
        "change": last["Close"] - prev["Close"],
        "change_pct": (last["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else np.nan,
        "ma20": last.get("MA20", np.nan),
        "ma50": last.get("MA50", np.nan),
        "ma200": last.get("MA200", np.nan),
        "rsi": last.get("RSI14", np.nan),
        "date": df.index[-1].date(),
    }


def chart_price(df, title):
    df = add_indicators(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="20D MA", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="50D MA", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="200D MA", line=dict(width=1)))
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_rsi(df):
    df = add_indicators(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI 14", line=dict(width=2)))
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold 30")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
    return fig


def normalize(df):
    s = df["Close"].dropna()
    return s / s.iloc[0] * 100


def generate_gold_analysis(gld_df, data):
    gld = add_indicators(gld_df)
    snap = latest_snapshot(gld_df)
    lines = []

    price = snap["price"]
    ma20, ma50, ma200 = snap["ma20"], snap["ma50"], snap["ma200"]
    rsi = snap["rsi"]

    if price > ma50 > ma200:
        trend = "Bullish trend"
        lines.append("GLD is above both its 50-day and 200-day moving averages, which suggests upward momentum is still intact.")
    elif price < ma50 < ma200:
        trend = "Bearish trend"
        lines.append("GLD is below both its 50-day and 200-day moving averages, which suggests the trend is weak.")
    elif price > ma50 and price < ma200:
        trend = "Short-term rebound, long-term still weak"
        lines.append("GLD is above the 50-day average but below the 200-day average, so the rebound may still need confirmation.")
    elif price < ma50 and price > ma200:
        trend = "Short-term pullback, long-term support still possible"
        lines.append("GLD is below the 50-day average but still above the 200-day average, so this may be a pullback rather than a full breakdown.")
    else:
        trend = "Mixed trend"
        lines.append("The moving-average signals are mixed, so GLD does not have a clean trend signal yet.")

    if not pd.isna(rsi):
        if rsi >= 70:
            lines.append("RSI is above 70, meaning GLD may be overbought in the short term.")
        elif rsi <= 30:
            lines.append("RSI is below 30, meaning GLD may be oversold in the short term.")
        else:
            lines.append("RSI is in the neutral zone, so price momentum is not extreme.")

    # Macro driver checks: USD index and 10Y yield
    driver_notes = []
    for ticker, name in [("DX-Y.NYB", "US Dollar Index"), ("^TNX", "US 10-year yield")]:
        if ticker in data and not data[ticker].empty:
            df = data[ticker].copy()
            s = df["Close"].dropna()
            if len(s) > 22:
                one_month = s.iloc[-1] / s.iloc[-22] - 1
                if ticker == "^TNX":
                    direction = "higher" if one_month > 0 else "lower"
                    driver_notes.append(f"{name} is {direction} over roughly 1 month. Higher yields usually pressure gold because gold has no yield.")
                else:
                    direction = "stronger" if one_month > 0 else "weaker"
                    driver_notes.append(f"{name} is {direction} over roughly 1 month. A stronger USD usually pressures gold because gold is priced in dollars.")

    return trend, lines, driver_notes


# -----------------------------
# App UI
# -----------------------------
st.title("🥇 Gold Monitor Dashboard")
st.caption("Tracks GLD in USD, gold futures, USD index, US yields, and technical signals. Free data via Yahoo Finance/yfinance; quotes may be delayed.")

with st.sidebar:
    st.header("Settings")
    main_choice = st.selectbox("Main asset", ["GLD ETF (USD)", "Gold Futures", "Spot Gold USD"], index=0)
    period = st.selectbox("History period", PERIODS, index=3)
    show_comparison = st.checkbox("Show macro comparison", True)
    st.markdown("---")
    st.write("Useful tickers:")
    st.code("GLD, GC=F, XAUUSD=X, DX-Y.NYB, ^TNX")

main_asset = ASSETS[main_choice]
required = [main_asset.ticker, "GLD", "GC=F", "DX-Y.NYB", "^TNX", "^GSPC"]
data = load_data(sorted(set(required)), period=period)

if main_asset.ticker not in data or data[main_asset.ticker].empty:
    st.error(f"No data returned for {main_asset.ticker}. Try GLD or another period.")
    st.stop()

main_df = data[main_asset.ticker]
main_df = add_indicators(main_df)
snap = latest_snapshot(main_df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{main_asset.ticker} latest", fmt(snap["price"]), f"{fmt(snap['change'])} / {pct(snap['change_pct'])}")
c2.metric("20D MA", fmt(snap["ma20"]))
c3.metric("50D MA", fmt(snap["ma50"]))
c4.metric("200D MA", fmt(snap["ma200"]))
c5.metric("RSI 14", fmt(snap["rsi"]))

st.plotly_chart(chart_price(main_df, f"{main_asset.label} ({main_asset.ticker})"), use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Momentum: RSI")
    st.plotly_chart(chart_rsi(main_df), use_container_width=True)

with right:
    st.subheader("Automatic Gold Analysis")
    gld_df = data.get("GLD", main_df)
    trend, tech_lines, driver_notes = generate_gold_analysis(gld_df, data)
    st.info(trend)
    for line in tech_lines:
        st.write("• " + line)
    st.markdown("**Macro drivers to watch**")
    for line in driver_notes:
        st.write("• " + line)
    st.caption("This is rule-based market analysis, not financial advice.")

if show_comparison:
    st.subheader("Gold vs Macro Drivers, normalized to 100")
    fig = go.Figure()
    comparison = [
        ("GLD", "GLD ETF"),
        ("GC=F", "Gold Futures"),
        ("DX-Y.NYB", "US Dollar Index"),
        ("^TNX", "US 10Y Yield"),
        ("^GSPC", "S&P 500"),
    ]
    for ticker, label in comparison:
        if ticker in data and not data[ticker].empty:
            s = normalize(data[ticker])
            fig.add_trace(go.Scatter(x=s.index, y=s, name=label, line=dict(width=2)))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Price Table")
cols = []
for ticker, label in [("GLD", "GLD ETF"), ("GC=F", "Gold Futures"), ("DX-Y.NYB", "US Dollar Index"), ("^TNX", "US 10Y Yield"), ("^GSPC", "S&P 500")]:
    if ticker in data and not data[ticker].empty:
        temp = add_indicators(data[ticker])
        s = latest_snapshot(temp)
        latest_price = s["price"] / 10 if ticker == "^TNX" else s["price"]
        cols.append({
            "Asset": label,
            "Ticker": ticker,
            "Latest": latest_price,
            "Daily Change %": s["change_pct"],
            "50D MA": s["ma50"] / 10 if ticker == "^TNX" and not pd.isna(s["ma50"]) else s["ma50"],
            "200D MA": s["ma200"] / 10 if ticker == "^TNX" and not pd.isna(s["ma200"]) else s["ma200"],
            "Last Date": str(s["date"]),
        })

table = pd.DataFrame(cols)
st.dataframe(table, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Data source: yfinance/Yahoo Finance. Free data can be delayed, interrupted, or revised. Use a paid data provider for trading-critical decisions.")
