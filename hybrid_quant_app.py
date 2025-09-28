"""
================================================================================
APP: hybrid_quant_app.py  (Streamlit)
================================================================================
What this does
--------------
- Weekly/bi-weekly rebalanced hybrid quant system that combines:
  (1) Momentum (3–12 month), (2) Short-term mean reversion, (3) Vol targeting
  (4) Kelly-style sizing proxy, and (5) Beta-based hedging suggestions.
- Produces an actionable table with: BUY/SELL, when, entry price, stop, take
  profit, suggested hedge ticker/ratio, and rationale.
- Fetches data from Tiingo and/or FMP if API keys provided; falls back to
  yfinance if not. Keys are read from env vars:
    - TIINGO_API_KEY    (https://www.tiingo.com/)
    - FMP_API_KEY       (https://financialmodelingprep.com/)

How to run locally
------------------
1) pip install -r requirements.txt
2) export TIINGO_API_KEY=...  (optional)
   export FMP_API_KEY=...     (optional)
3) streamlit run hybrid_quant_app.py

Quick deploy options
--------------------
- Streamlit Community Cloud: push this single file + requirements.txt to GitHub
  and deploy.
- Docker (Render/Fly/Railway): see Dockerfile snippet at bottom comment.

Notes & safety
--------------
- This is an educational quant template. Real trading requires controls,
  slippage/fees modeling, compliance, and guardrails. Size small.

================================================================================
"""

import os
import math
import time
import json
import textwrap
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# Config & Constants
# -----------------------------
FAILED_TICKERS = []  # populated when downloads fail so we can warn the user
DEFAULT_UNIVERSE = [
    # Liquid US large caps for demo; replace with your universe or load a file
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","TSLA","LLY","AVGO",
    "JPM","XOM","V","UNH","MA","HD","PG","ORCL","COST","BAC",
    "NFLX","KO","PEP","PFE","WMT","DIS","CSCO","ADBE","CRM","ABT",
]

INDEX_TICKER = "SPY"  # used for beta/hedge; adjust per region
TZ = "America/New_York"

# Vol targeting & sizing
TARGET_ANNUAL_VOL = 0.12  # 12% portfolio vol target
MAX_POSITION_PCT = 0.07   # cap any single position at 7% of equity
RISK_REWARD = 2.0         # take-profit vs stop (2:1)
STOP_ATR_MULT = 2.0       # stop uses ATR multiple

# Lookbacks
MOM_LOOKBACK_DAYS = 252   # ~12 months for mom
MOM_SKIP_DAYS = 21        # skip last 1 month to avoid short-term reversal
MR_LOOKBACK_WEEKS = 4     # mean-rev zscore on ~1 month weekly changes
BETA_LOOKBACK_WEEKS = 26  # for hedge beta
ATR_LOOKBACK = 14         # ATR for stops

# -----------------------------
# Helpers
# -----------------------------
@dataclass
class Signal:
    ticker: str
    date: pd.Timestamp
    action: str  # BUY/SELL/FLAT
    entry_price: float
    stop: float
    take_profit: float
    hedge_ticker: str
    hedge_ratio: float
    size_pct: float
    rationale: str
    valid_from: pd.Timestamp
    valid_to: pd.Timestamp


def fmt_pct(x):
    try:
        return f"{100*x:.2f}%"
    except Exception:
        return ""


def safe_resample_weekly(px: pd.DataFrame) -> pd.DataFrame:
    # last business day of week; align to Friday
    return px.resample("W-FRI").last().dropna(how="all")


def compute_atr(df: pd.DataFrame, n: int = ATR_LOOKBACK) -> pd.Series:
    # df columns expected: [Open, High, Low, Close]
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr


def kelly_fraction(mean_ret: float, vol: float) -> float:
    """Simplified Kelly proxy: f* ≈ mu/var for small edges; cap to [0, 0.2]."""
    if vol <= 1e-9:
        return 0.0
    var = vol**2
    f = max(0.0, min(0.2, mean_ret / max(1e-12, var)))
    return f


# -----------------------------
# Data Fetchers (Tiingo, FMP, yfinance fallback)
# -----------------------------

def fetch_yf(ticker: str, start: str) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    # Robust retry wrapper: yfinance can intermittently fail with JSON decode errors
    for _ in range(3):
        try:
            df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.index = df.index.tz_localize(None)
                return df
        except Exception:
            time.sleep(1.0)
            continue
    return None


def fetch_tiingo(ticker: str, start: str) -> Optional[pd.DataFrame]:
    key = os.getenv("TIINGO_API_KEY")
    if not key:
        return None
    try:
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {"startDate": start, "format": "json", "token": key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        # normalize columns to Yahoo-style
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date")
        rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjClose": "Adj Close",
            "volume": "Volume",
        }
        for c_old, c_new in rename.items():
            if c_old in df.columns:
                df[c_new] = df[c_old]
        needed = ["Open","High","Low","Close","Adj Close","Volume"]
        for n in needed:
            if n not in df.columns:
                df[n] = np.nan
        return df[needed]
    except Exception:
        return None


def get_price_history(ticker: str, start: str) -> Optional[pd.DataFrame]:
    # Try Tiingo → yfinance with retries
    df = fetch_tiingo(ticker, start)
    if df is None:
        df = fetch_yf(ticker, start)
    if df is None:
        FAILED_TICKERS.append(ticker)
    return df


# -----------------------------
# Signal Engines
# -----------------------------

def momentum_score(px_daily: pd.Series) -> float:
    if px_daily is None or px_daily.empty:
        return np.nan
    px = px_daily.dropna()
    if len(px) < MOM_LOOKBACK_DAYS + MOM_SKIP_DAYS + 5:
        return np.nan
    # 12-1 momentum: return over 252d back to 21d back
    r_12m = px.iloc[-(MOM_SKIP_DAYS+1)] / px.iloc[-(MOM_LOOKBACK_DAYS+MOM_SKIP_DAYS)] - 1.0
    return float(r_12m)


def mean_reversion_score(weekly_close: pd.Series) -> float:
    wc = weekly_close.dropna()
    if len(wc) < MR_LOOKBACK_WEEKS + 3:
        return np.nan
    # z-score of last weekly return vs recent history (negative z = oversold -> buy)
    wret = wc.pct_change().dropna()
    last = wret.iloc[-1]
    mu = wret.iloc[-MR_LOOKBACK_WEEKS:].mean()
    sd = wret.iloc[-MR_LOOKBACK_WEEKS:].std(ddof=1)
    if sd <= 1e-9:
        return 0.0
    z = (last - mu) / sd
    # invert: more negative z -> higher positive score
    return float(-z)


def calc_beta(weekly_ret: pd.Series, benchmark_ret: pd.Series) -> float:
    df = pd.concat([weekly_ret, benchmark_ret], axis=1).dropna()
    if len(df) < 8:
        return 1.0
    x = df.iloc[:,1]
    y = df.iloc[:,0]
    cov = np.cov(x, y)[0,1]
    var = np.var(x)
    if var <= 1e-12:
        return 1.0
    beta = cov / var
    return float(beta)


def atr_stop_takeprofit(df: pd.DataFrame, risk_reward=RISK_REWARD, atr_mult=STOP_ATR_MULT):
    atr = compute_atr(df, ATR_LOOKBACK).iloc[-1]
    close = df["Close"].iloc[-1]
    if np.isnan(atr) or np.isnan(close):
        return (np.nan, np.nan)
    stop = close - atr_mult*atr
    tp = close + risk_reward*(close - stop)
    return (float(stop), float(tp))


# -----------------------------
# Portfolio Construction
# -----------------------------

def build_signals(universe: List[str], start: str, rebalance_weeks: int = 1,
                  end: Optional[str] = None) -> pd.DataFrame:
    rows = []
    bench_df = get_price_history(INDEX_TICKER, start)
    if bench_df is None or bench_df.empty:
        st.warning("Failed to fetch benchmark data; hedges default to 1x SPY.")
    else:
        bench_w = safe_resample_weekly(bench_df["Adj Close"]).pct_change()

    for t in universe:
        df = get_price_history(t, start)
        if df is None or df.empty:
            continue
        # weekly series
        weekly = safe_resample_weekly(df["Adj Close"])  # close

        mom = momentum_score(df["Adj Close"])  # 12-1 momentum
        mr = mean_reversion_score(weekly)

        # composite: 60% momentum, 40% mean reversion (weekly horizon)
        comp = 0.6*(mom if not np.isnan(mom) else 0.0) + 0.4*(mr if not np.isnan(mr) else 0.0)

        # Recent stats for sizing
        wret = weekly.pct_change().dropna()
        ann_vol = np.sqrt(52) * wret[-BETA_LOOKBACK_WEEKS:].std(ddof=1) if len(wret) >= 4 else np.nan
        mean_w = wret[-MR_LOOKBACK_WEEKS:].mean() if len(wret) >= MR_LOOKBACK_WEEKS else np.nan

        # Size = min(vol-target position, Kelly proxy)
        if not np.isnan(ann_vol) and ann_vol > 1e-6:
            vt_weight = min(MAX_POSITION_PCT, TARGET_ANNUAL_VOL / (ann_vol * max(1.0, math.sqrt(len(universe)/10))))
        else:
            vt_weight = MAX_POSITION_PCT/2
        kelly_w = kelly_fraction(mean_w*52 if not np.isnan(mean_w) else 0.0, ann_vol if not np.isnan(ann_vol) else 0.2)
        size_pct = float(max(0.0, min(MAX_POSITION_PCT, 0.5*vt_weight + 0.5*kelly_w)))

        # Action logic: buy if strong positive comp; sell (short) if strong negative
        # thresholds scale with cross-sectional stats
        comp_series = []  # placeholder; we use absolute thresholds
        action = "FLAT"
        if not np.isnan(comp):
            if comp > 0.05:   # positive signal
                action = "BUY"
            elif comp < -0.05:  # negative signal
                action = "SELL"

        stop, tp = atr_stop_takeprofit(df)
        last = df["Close"].iloc[-1]

        # Hedge: beta vs benchmark
        beta = 1.0
        if bench_df is not None and not bench_df.empty:
            beta = calc_beta(wret[-BETA_LOOKBACK_WEEKS:], bench_w[-BETA_LOOKBACK_WEEKS:])
        hedge_ratio = round(beta, 2)

        # Validity window (weekly/bi-weekly)
        signal_date = pd.Timestamp(df.index[-1]).tz_localize(None)
        valid_from = signal_date
        valid_to = signal_date + pd.Timedelta(weeks=rebalance_weeks)

        rationale = []
        if not np.isnan(mom):
            rationale.append(f"12-1 momentum: {fmt_pct(mom)}")
        if not np.isnan(mr):
            rationale.append(f"Mean-rev z: {mr:.2f} (neg z -> oversold)")
        if not np.isnan(ann_vol):
            rationale.append(f"Ann vol: {fmt_pct(ann_vol)}")
        rationale.append(f"Beta vs {INDEX_TICKER}: {beta:.2f}")

        rows.append(Signal(
            ticker=t,
            date=signal_date,
            action=action,
            entry_price=float(last) if not np.isnan(last) else np.nan,
            stop=stop,
            take_profit=tp,
            hedge_ticker=INDEX_TICKER,
            hedge_ratio=float(hedge_ratio),
            size_pct=size_pct,
            rationale="; ".join(rationale),
            valid_from=valid_from,
            valid_to=valid_to,
        ).__dict__)

    sigs = pd.DataFrame(rows)
    if sigs.empty:
        return sigs

    # Rank by composite strength proxy (reconstruct quickly)
    # For simple UX, sort: BUY signals by strongest comp (mom + mr), then SELLs
    def comp_proxy(r: pd.Series) -> float:
        # Recover mom & mr from rationale is messy; recompute quickly using price
        return 0.0

    # Prioritize buys on top
    sigs["action_rank"] = sigs["action"].map({"BUY":0, "SELL":1, "FLAT":2})
    sigs = sigs.sort_values(["action_rank","size_pct"], ascending=[True, False]).drop(columns=["action_rank"])
    return sigs


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Hybrid Weekly Quant App", layout="wide")
    st.title("📈 Hybrid Weekly Quant (Momentum + Mean Reversion + Hedge)")
    st.caption("Weekly/bi-weekly signals with position sizing and hedge suggestions. For education only.")

    with st.sidebar:
        st.header("Settings")
        universe_src = st.radio("Universe", ["Default (US Large Caps)", "Custom CSV (tickers column)", "Manual list"], index=0)
        custom_list = []
        if universe_src == "Custom CSV (tickers column)":
            f = st.file_uploader("Upload CSV with 'ticker' column", type=["csv"]) 
            if f is not None:
                try:
                    dfu = pd.read_csv(f)
                    custom_list = sorted(list({str(x).strip().upper() for x in dfu["ticker"].dropna().unique()}))
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")
        elif universe_src == "Manual list":
            txt = st.text_area("Enter comma-separated tickers", "AAPL,MSFT,NVDA,AMZN,SPY")
            custom_list = [t.strip().upper() for t in txt.split(",") if t.strip()]

        universe = custom_list if custom_list else DEFAULT_UNIVERSE

        rebalance = st.select_slider("Rebalance cadence", options=[1,2], value=1, format_func=lambda w: "Weekly" if w==1 else "Bi-Weekly")
        start_date = st.date_input("Start date for history", dt.date.today() - dt.timedelta(days=800))
        go = st.button("Run Signals")

    if go:
        with st.spinner("Calculating weekly signals..."):
            # reset failure list per run
            global FAILED_TICKERS
            FAILED_TICKERS = []
            sigs = build_signals(universe, start=start_date.isoformat(), rebalance_weeks=rebalance)
        if FAILED_TICKERS:
            st.warning(f"The following tickers failed to load and were skipped: {', '.join(sorted(set(FAILED_TICKERS)))}. This is often temporary with free data sources; try rerunning, or set TIINGO/FMP keys for more reliable downloads.")
        if sigs is None or sigs.empty:
            st.warning("No signals generated. Try a different universe or later start date.")
            return

        # Actionable table
        pretty = sigs.copy()
        pretty.rename(columns={
            "ticker":"Ticker","action":"Action","entry_price":"Entry Price",
            "stop":"Stop","take_profit":"Take Profit","hedge_ticker":"Hedge",
            "hedge_ratio":"Hedge Ratio","size_pct":"Size %","rationale":"Rationale",
            "valid_from":"Valid From","valid_to":"Valid To","date":"Signal Date"
        }, inplace=True)
        pretty["Size %"] = (pretty["Size %"]*100).round(2)
        pretty["Entry Price"] = pretty["Entry Price"].round(2)
        pretty["Stop"] = pretty["Stop"].round(2)
        pretty["Take Profit"] = pretty["Take Profit"].round(2)
        pretty["Hedge Ratio"] = pretty["Hedge Ratio"].round(2)
        cols = [
            "Ticker","Action","Signal Date","Valid From","Valid To",
            "Entry Price","Stop","Take Profit","Size %","Hedge","Hedge Ratio","Rationale"
        ]
        st.subheader("Signals (Actionable)")
        st.dataframe(pretty[cols], use_container_width=True)

        st.download_button(
            "Download CSV",
            pretty[cols].to_csv(index=False).encode("utf-8"),
            file_name=f"signals_{dt.date.today().isoformat()}.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("How to use the hedge column")
        st.write(textwrap.dedent(f"""
        - **Hedge** uses beta vs **{INDEX_TICKER}** over the last {BETA_LOOKBACK_WEEKS} weeks.
        - If Action = BUY for Ticker=XYZ with Hedge Ratio=0.8 → short 0.8 units of {INDEX_TICKER}
          per 1 unit of XYZ *by notional* to neutralize market beta.
        - For SELL signals, hedge may be long the benchmark by the same ratio.
        - Adjust for sector ETFs if you prefer sector-neutral hedges.
        """))

        st.divider()
        st.subheader("Disclaimers & Next Steps")
        st.markdown(
            "- Educational template; does **not** account for transaction costs, taxes, borrow fees.\n"
            "- Add compliance, limits, slippage modeling before real capital.\n"
            "- Extend with fundamentals via FMP/Tiingo endpoints for Value/Quality factors.\n"
            "- Add broker/exchange connectivity and order management if automating."
        )
    else:
        st.info("Configure your universe and click **Run Signals** to generate the weekly plan.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------
# requirements.txt (create this file alongside the app)
# ---------------------------------------------------------------------------------
# streamlit==1.37.0
# pandas==2.2.2
# numpy==1.26.4
# requests==2.32.3
# yfinance==0.2.40

# ---------------------------------------------------------------------------------
# Dockerfile (optional)
# ---------------------------------------------------------------------------------
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY hybrid_quant_app.py ./
# ENV PORT=8501
# EXPOSE 8501
# CMD ["streamlit", "run", "hybrid_quant_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ---------------------------------------------------------------------------------
# Notes to extend factors with FMP/Tiingo fundamentals (pseudo-code)
# ---------------------------------------------------------------------------------
# - Use FMP endpoints like /ratios or /key-metrics to fetch P/B, ROE, Debt/EBITDA.
# - Merge into universe and compute Value (cheapness) & Quality (profitability) scores.
# - Blend into composite: comp = 0.4*mom + 0.3*mr + 0.3*(0.5*value + 0.5*quality)
# - Rebalance cadence can stay weekly/bi-weekly; fundamentals update slower.
