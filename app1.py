# stock_dashboard_full.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import feedparser
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from urllib.parse import quote_plus
from textblob import TextBlob
import logging
from chart_tools import (
    plot_fibonacci_retracement,
    plot_fibonacci_projection,
    plot_fibonacci_fan,
    plot_elliott_wave,
    plot_gartley_pattern,
    plot_trend_lines
)
# Optional libraries guarded when used
try:
    import pandas_ta as pta
    has_pandas_ta = True
except Exception:
    has_pandas_ta = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    has_transformers = True
except Exception:
    has_transformers = False

# Guard heavy ML libraries for LSTM/TF
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    has_tf = True
except Exception:
    has_tf = False

# ARIMA (statsmodels) guard
try:
    from statsmodels.tsa.arima.model import ARIMA
    has_arima = True
except Exception:
    has_arima = False

logging.getLogger('numexpr').setLevel(logging.WARNING)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📂 Sections")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Financial Scoring",
        "Peer Comparison",
        "Revenue & Income Trends",
        "Price Trends & Sharpe Ratio",
        "ARIMA Forecast",
        "LSTM Forecast",
        "News Sentiment",
        "Technical & Sentiment Analyzer",
        "Capital Allocation Matrix",
        "Chart Toolbox",
    ]
)

# ---------------------------
# Inputs
# ---------------------------
raw_tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL, MSFT, GOOGL")
tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
benchmark = st.sidebar.text_input("Enter benchmark ticker", "SPY").strip().upper()

# ---------------------------
# Helper: Financial fetch + fallbacks
# ---------------------------
@st.cache_data(show_spinner=False)
def get_financials_all(ticker):
    """Return dict with annual/quarterly financials, balance, cashflow and computed fallbacks."""
    try:
        t = yf.Ticker(ticker)
        fin = t.financials.T if hasattr(t, 'financials') and not t.financials.empty else pd.DataFrame()
        qfin = t.quarterly_financials.T if hasattr(t, 'quarterly_financials') and not t.quarterly_financials.empty else pd.DataFrame()
        bal = t.balance_sheet.T if hasattr(t, 'balance_sheet') and not t.balance_sheet.empty else pd.DataFrame()
        qbal = t.quarterly_balance_sheet.T if hasattr(t, 'quarterly_balance_sheet') and not t.quarterly_balance_sheet.empty else pd.DataFrame()
        cf = t.cashflow.T if hasattr(t, 'cashflow') and not t.cashflow.empty else pd.DataFrame()
        qcf = t.quarterly_cashflow.T if hasattr(t, 'quarterly_cashflow') and not t.quarterly_cashflow.empty else pd.DataFrame()

        annual = pd.concat([fin, bal, cf], axis=1) if not fin.empty or not bal.empty or not cf.empty else pd.DataFrame()
        quarterly = pd.concat([qfin, qbal, qcf], axis=1) if not qfin.empty or not qbal.empty or not qcf.empty else pd.DataFrame()

        # FCF fallback
        def compute_fcf(df_cash):
            if df_cash is None or df_cash.empty:
                return None
            if "Free Cash Flow" in df_cash.columns:
                return df_cash["Free Cash Flow"]
            op = None
            capex = None
            for cand in ["Total Cash From Operating Activities", "Net Cash Provided by Operating Activities", "Operating Cash Flow"]:
                if cand in df_cash.columns:
                    op = df_cash[cand]
                    break
            for cand in ["Capital Expenditures", "Capital Expenditure", "Capital Expenditures - Investing"]:
                if cand in df_cash.columns:
                    capex = df_cash[cand]
                    break
            if op is not None and capex is not None:
                return op + capex
            return None

        if "Free Cash Flow" not in annual.columns:
            fcf_ann = compute_fcf(cf)
            if fcf_ann is not None:
                annual["Free Cash Flow"] = fcf_ann
        if "Free Cash Flow" not in quarterly.columns:
            fcf_q = compute_fcf(qcf)
            if fcf_q is not None:
                quarterly["Free Cash Flow"] = fcf_q

        # ROE fallback
        if "ROE" not in annual.columns:
            if ("Net Income" in fin.columns) and ("Total Stockholder Equity" in bal.columns):
                eq = bal["Total Stockholder Equity"].replace(0, np.nan)
                annual["ROE"] = fin["Net Income"] / eq
        if "ROE" not in quarterly.columns:
            if ("Net Income" in qfin.columns) and ("Total Stockholder Equity" in qbal.columns):
                eq_q = qbal["Total Stockholder Equity"].replace(0, np.nan)
                quarterly["ROE"] = qfin["Net Income"] / eq_q

        # Debt/Equity fallback
        if "Debt/Equity" not in annual.columns:
            if ("Long Term Debt" in bal.columns or "Total Liab" in bal.columns) and ("Total Stockholder Equity" in bal.columns):
                liabilities = bal.get("Total Liab", bal.get("Long Term Debt", pd.Series(index=bal.index)))
                equity = bal["Total Stockholder Equity"].replace(0, np.nan)
                try:
                    annual["Debt/Equity"] = liabilities / equity
                except Exception:
                    pass
        if "Debt/Equity" not in quarterly.columns:
            if ("Long Term Debt" in qbal.columns or "Total Liab" in qbal.columns) and ("Total Stockholder Equity" in qbal.columns):
                liabilities_q = qbal.get("Total Liab", qbal.get("Long Term Debt", pd.Series(index=qbal.index)))
                equity_q = qbal["Total Stockholder Equity"].replace(0, np.nan)
                try:
                    quarterly["Debt/Equity"] = liabilities_q / equity_q
                except Exception:
                    pass

        return {
            "financials_annual": annual,
            "financials_quarterly": quarterly,
            "balance_annual": bal,
            "balance_quarterly": qbal,
            "cashflow_annual": cf,
            "cashflow_quarterly": qcf,
            "info": t.info
        }
    except Exception as e:
        st.warning(f"⚠️ Could not fetch {ticker}: {e}")
        return {}

# ---------------------------
# Utilities
# ---------------------------
def safe_pct_change_df(df):
    # returns percent change safely
    return df.pct_change()

def compute_scores(fin_annual, fin_quarterly, metrics):
    scores = {}
    if fin_annual is None or fin_quarterly is None:
        return {m: 0 for m in metrics}
    fin_annual = fin_annual.sort_index()
    fin_quarterly = fin_quarterly.sort_index()
    for metric in metrics:
        try:
            if metric in fin_annual.columns and metric in fin_quarterly.columns:
                ann = fin_annual[metric].dropna()
                qtr = fin_quarterly[metric].dropna()
                if len(ann) >= 2 and len(qtr) >= 2:
                    yoy = ann.iloc[-1] - ann.iloc[-2]
                    qoq = qtr.iloc[-1] - qtr.iloc[-2]
                    if yoy > 0 and qoq > 0:
                        score = 3
                    elif yoy > 0 and qoq < 0:
                        score = 2
                    elif yoy < 0 and qoq > 0:
                        score = 1
                    elif yoy < 0 and qoq < 0:
                        score = -1
                    else:
                        score = 1
                else:
                    score = 0
            else:
                score = 0
        except Exception:
            score = 0
        scores[metric] = score
    return scores

def format_long_form(df, ticker_names):
    df = df.sort_index()
    df_growth = df.pct_change() * 100
    df_long = df.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Value')
    df_long['Growth %'] = df_growth.reset_index().melt(id_vars='index', var_name='Ticker', value_name='Growth')['Growth']
    df_long.rename(columns={'index': 'Period'}, inplace=True)
    df_long['Company'] = df_long['Ticker'].map(ticker_names)
    return df_long

# ---------------------------
# Technical helper functions
# ---------------------------
def plot_fibonacci_retracement(data, ticker):
    high = float(data['High'].max())
    low = float(data['Low'].min())
    diff = high - low
    fib_levels = {
        '0.0%': (high, 'red'),
        '23.6%': (high - 0.236 * diff, 'orange'),
        '38.2%': (high - 0.382 * diff, 'yellow'),
        '50.0%': (high - 0.5 * diff, 'green'),
        '61.8%': (high - 0.618 * diff, 'blue'),
        '78.6%': (high - 0.786 * diff, 'magenta'),
        '100.0%': (low, 'purple')
    }

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(data['Close'].values, label='Close Price', color='black')
    for level, (value, color) in fib_levels.items():
        ax.axhline(value, linestyle='--', label=level, color=color)
        ax.text(len(data)*0.99, value, level, va='center', ha='right', fontsize=8, color=color)
    ax.set_title(f"{ticker} - Fibonacci Retracement")
    ax.legend(fontsize=6)
    plt.tight_layout()
    return fig

def plot_fibonacci_fan(data, ticker):
    high = float(data['High'].max())
    low = float(data['Low'].min())

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(data['Close'].values, label='Close Price', color='black')
    base_x = [0, len(data)-1]
    for ratio, color in zip([0.382, 0.5, 0.618], ['blue', 'green', 'red']):
        slope = (high - low) * ratio / len(data)
        fan_y = [low, low + slope * (len(data)-1)]
        ax.plot(base_x, fan_y, linestyle='--', label=f'Fan {ratio}', color=color)
    ax.set_title(f"{ticker} - Fibonacci Fan")
    ax.legend(fontsize=6)
    plt.tight_layout()
    return fig

def plot_fibonacci_projection(data, ticker):
    high = float(data['High'].max())
    low = float(data['Low'].min())
    diff = high - low
    latest_close = float(data['Close'].iloc[-1])
    next_points = [latest_close + diff * r for r in [0.382, 0.618, 1.0]]

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(data['Close'].values, label='Close Price', color='black')
    for r, value in zip([0.382, 0.618, 1.0], next_points):
        ax.axhline(value, linestyle=':', label=f"Projection {r}", color='gray')
        ax.text(len(data)*0.99, value, f"Proj {r}", va='center', ha='right', fontsize=8, color='gray')
    ax.set_title(f"{ticker} - Fibonacci Projection")
    ax.legend(fontsize=6)
    plt.tight_layout()
    return fig

def plot_elliott_wave(price, ticker):
    if len(price) >= 35:
        labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5', 'Wave A', 'Wave B', 'Wave C']
        slices = [(35, 30), (30, 25), (25, 20), (20, 15), (15, 10), (10, 7), (7, 4), (4, 1)]
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'brown']

        plt.figure(figsize=(8, 3))
        plt.plot(price.values, label='Close Price', color='gray', linewidth=1)

        for ((start, end), label, color) in zip(slices, labels, colors):
            seg = price.iloc[-start:-end]
            idxs = range(len(price)-start, len(price)-end)
            plt.plot(idxs, seg, label=label, color=color, linewidth=2)
            plt.text(list(idxs)[0], seg.iloc[0], label, fontsize=8, color=color)

        plt.title(f"Elliott Waves (1–5 + A–C) - {ticker}")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.close()  # return as plot object via plt.gcf()
        return 1
    return 0

def plot_gartley(price, ticker):
    if len(price) > 10:
        try:
            x = float(price.iloc[-10])
            a = float(price.iloc[-8])
            b = float(price.iloc[-6])
            c = float(price.iloc[-4])
            d = float(price.iloc[-2])
        except Exception:
            return 0
        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        cd = abs(d - c)
        tol = 0.05  # 5% tolerance
        is_gartley = False
        if (0.61 - tol) * xa < ab < (0.786 + tol) * xa and \
           (0.382 - tol) * ab < bc < (0.886 + tol) * ab and \
           (0.786 - tol) * bc < cd < (1.27 + tol) * bc:
            is_gartley = True
            plt.figure(figsize=(8,3))
            plt.plot(price.values, label='Close Price', color='black')
            points = [x, a, b, c, d]
            indices = [len(price)-10, len(price)-8, len(price)-6, len(price)-4, len(price)-2]
            labels = ['X', 'A', 'B', 'C', 'D']
            plt.scatter(indices, points, color='purple')
            for idx, val, label in zip(indices, points, labels):
                plt.text(idx, val, label, fontsize=8, color='purple')
            plt.title(f"Gartley Pattern - {ticker} ({'Detected' if is_gartley else 'Not Detected'})")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.close()
        return 1 if is_gartley else 0
    return 0

# Fallback RSI and MACD (if pandas_ta not available)
def rsi_fallback(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing via EMA-like
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd_fallback(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df = pd.DataFrame({'MACD': macd, 'MACD_signal': signal, 'MACD_hist': hist})
    # use naming similar to pandas_ta
    df = df.rename(columns={'MACD':'MACD_12_26_9', 'MACD_signal':'MACDs_12_26_9', 'MACD_hist':'MACDh_12_26_9'})
    return df

def calculate_technical_score(ticker):
    data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    if data.empty or 'Close' not in data:
        return 0.0

    # RSI
    if has_pandas_ta:
        try:
            data['RSI'] = pta.rsi(data['Close'])
        except Exception:
            data['RSI'] = rsi_fallback(data['Close'])
    else:
        data['RSI'] = rsi_fallback(data['Close'])

    # MACD
    if has_pandas_ta:
        try:
            macd_df = pta.macd(data['Close'])
            data = pd.concat([data, macd_df], axis=1)
        except Exception:
            macd_df = macd_fallback(data['Close'])
            data = pd.concat([data, macd_df], axis=1)
    else:
        macd_df = macd_fallback(data['Close'])
        data = pd.concat([data, macd_df], axis=1)

    # Ichimoku - try pandas_ta if available
    ichimoku_present = False
    if has_pandas_ta:
        try:
            ichi = pta.ichimoku(data['High'], data['Low'], data['Close'])
            # pandas_ta returns tuple sometimes; just try to concat any DataFrames
            if isinstance(ichi, pd.DataFrame):
                data = pd.concat([data, ichi], axis=1)
                ichimoku_present = True
        except Exception:
            ichimoku_present = False

    # Aroon
    try:
        if has_pandas_ta:
            aroon = pta.aroon(data['High'], data['Low'])
            data = pd.concat([data, aroon], axis=1)
    except Exception:
        pass

    price = data['Close'].reset_index(drop=True)
    # Chart-based patterns — we don't show their charts here to save execution time; functions still detect patterns
    elliott_signal = plot_elliott_wave(price, ticker)
    gartley_signal = plot_gartley(price, ticker)

    score = 0.0
    # Grab values safely
    rsi_val = None
    try:
        if 'RSI' in data.columns:
            rsi_val = data['RSI'].dropna().iloc[-1]
    except Exception:
        rsi_val = None

    macd_val = None
    try:
        if 'MACD_12_26_9' in data.columns:
            macd_val = data['MACD_12_26_9'].dropna().iloc[-1]
        elif 'MACD' in data.columns:
            macd_val = data['MACD'].dropna().iloc[-1]
    except Exception:
        macd_val = None

    ichimoku_val = None
    try:
        # many ichimoku column names; try some common ones
        for c in ['ITS_9', 'ISA_9', 'ICH_SENKOUSPAN_A', 'ICH_SENKOUSPAN_B']:
            if c in data.columns:
                ichimoku_val = data[c].dropna().iloc[-1]
                break
    except Exception:
        ichimoku_val = None

    aroon_up = None
    try:
        for c in ['AROONU_14','AROONU_25']:
            if c in data.columns:
                aroon_up = data[c].dropna().iloc[-1]
                break
    except Exception:
        aroon_up = None

    # Scoring rules (kept similar to your standalone)
    if rsi_val is not None and 30 < rsi_val < 70:
        score += 1.0
    if macd_val is not None and macd_val > 0:
        score += 1.0
    if ichimoku_val is not None and ichimoku_val > data['Close'].iloc[-1]:
        score += 0.5
    if aroon_up is not None and aroon_up > 70:
        score += 0.5
    if elliott_signal:
        score += 0.5
    if gartley_signal:
        score += 0.5

    # Normalize range approx 0-4 (we'll normalize later to 0-1)
    return round(score, 2)

def calculate_sentiment_score(ticker):
    query = quote_plus(ticker.replace(".NS", "") + " stock news")
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    sentiments = []
    for entry in feed.entries[:5]:
        blob = TextBlob(entry.title + ". " + entry.get('summary', ''))
        sentiments.append(blob.sentiment.polarity)
    if sentiments:
        # Convert polarity (-1..1) to 0..5 scale like your earlier approach
        return round((sum(sentiments)/len(sentiments) + 1) * 2.5, 2)
    return 2.5

def calculate_fundamental_score(sample_fundamentals):
    score = 0
    if sample_fundamentals.get('revenue_growth', 0) > 15:
        score += 1
    if sample_fundamentals.get('profit_growth', 0) > 15:
        score += 1
    if sample_fundamentals.get('debt_equity', 999) < 0.5:
        score += 1
    if sample_fundamentals.get('roce', 0) > 15:
        score += 1
    if sample_fundamentals.get('promoter_holding', 0) > 50:
        score += 1
    return round((score / 5) * 5, 1)

# ---------------------------
# Sections
# ---------------------------
if section == "Financial Scoring":
    st.subheader("📊 Stock Ranking by Financial & Earnings Score")

    financial_metrics = [
        "Total Revenue", "Net Income", "Free Cash Flow",
        "Operating Income", "Operating Expense", "Gross Profit",
        "Operating Cash Flow", "Capital Expenditures"
    ]

    earnings_metrics = [
        "Total Revenue", "Net Income", "Free Cash Flow",
        "ROE", "Return on Assets", "Trailing P/E", "Price To Sales Ratio"
    ]

    result = []
    for t in tickers:
        fin_data = get_financials_all(t)
        if not fin_data:
            continue

        annual = fin_data.get("financials_annual", pd.DataFrame())
        quarterly = fin_data.get("financials_quarterly", pd.DataFrame())

        if annual.empty and quarterly.empty:
            continue

        f_score = compute_scores(annual, quarterly, financial_metrics)
        e_score = compute_scores(annual, quarterly, earnings_metrics)
        total = sum(f_score.values()) * 10 + sum(e_score.values()) * 8
        result.append({
            "Ticker": t,
            "Financial Score": sum(f_score.values()) * 10,
            "Earnings Score": sum(e_score.values()) * 8,
            "Total Score": total
        })

    if result:
        df_score = pd.DataFrame(result).sort_values("Total Score", ascending=False)
        st.dataframe(df_score, use_container_width=True)
        st.session_state["df_score"] = df_score
    else:
        st.warning("⚠️ No valid financial data found for given tickers.")

elif section == "Peer Comparison":
    st.subheader("📌 Peer Valuation & Ratios")
    peer_data = []
    ni_df = pd.DataFrame()

    for t in tickers:
        fin_data = get_financials_all(t)
        if not fin_data:
            continue

        info = fin_data.get("info", {})
        annual = fin_data.get("financials_annual", pd.DataFrame())

        roe = info.get('returnOnEquity')
        if roe is None and 'ROE' in annual.columns:
            try:
                roe = annual['ROE'].dropna().iloc[-1]
            except Exception:
                roe = None

        de = info.get('debtToEquity')
        if de is None and 'Debt/Equity' in annual.columns:
            try:
                de = annual['Debt/Equity'].dropna().iloc[-1]
            except Exception:
                de = None

        peer_data.append({
            'Ticker': t,
            'Sector': info.get('sector'),
            'P/E': info.get('trailingPE'),
            'P/B': info.get('priceToBook'),
            'ROE': roe,
            'ROA': info.get('returnOnAssets'),
            'Debt/Equity': (de or 0) / (100 if de is not None and abs(de) > 1 else 1),
            'Beta': info.get('beta'),
            'Dividend Yield': info.get('dividendYield')
        })

        if 'Net Income' in annual.columns:
            try:
                ni_df[t] = annual['Net Income'].dropna()
            except Exception:
                pass

    if peer_data:
        peer_df = pd.DataFrame(peer_data)
        st.dataframe(peer_df, use_container_width=True)
    else:
        st.warning("⚠️ No peer data found.")

    # RPN + DCF
    growth = 0.05
    discount = 0.10

    rpn_list, dcf_list = [], []
    if 'peer_df' in locals() and not peer_df.empty:
        for _, row in peer_df.iterrows():
            if row['Ticker'] == benchmark:
                continue
            risk = min(row.get('Debt/Equity') or 1, 5)
            prob = min(row.get('Beta') or 1, 5)
            detect = 5 if (row.get('ROE') or 0) < 0.10 else 1
            rpn = risk * prob * detect

            ni = 0
            if not ni_df.empty and row['Ticker'] in ni_df.columns:
                try:
                    ni = ni_df[row['Ticker']].dropna().iloc[-1]
                except Exception:
                    ni = 0

            dcf_val = (ni * (1 + growth)) / (discount - growth) if (discount - growth) != 0 else np.nan

            rpn_list.append({'Ticker': row['Ticker'], 'RPN': round(rpn, 2)})
            dcf_list.append({'Ticker': row['Ticker'], 'Last Net Income': round(ni, 2), 'Approx. DCF Value': round(dcf_val, 2)})

    st.subheader("🔍 RPN (Risk Priority Number)")
    st.dataframe(pd.DataFrame(rpn_list), use_container_width=True)
    st.session_state["rpn_list"] = rpn_list

elif section == "Revenue & Income Trends":
    st.subheader("📽️ Growth Charts")
    rev_df, ni_df = pd.DataFrame(), pd.DataFrame()
    st.session_state["tickers"] = tickers
    st.session_state["benchmark"] = benchmark

    for t in tickers:
        fin_data = get_financials_all(t)
        annual = fin_data.get("financials_annual", pd.DataFrame())
        if "Total Revenue" in annual.columns:
            try:
                rev_df[t] = annual["Total Revenue"].dropna()
            except Exception:
                pass
        if "Net Income" in annual.columns:
            try:
                ni_df[t] = annual["Net Income"].dropna()
            except Exception:
                pass

    growth_type = st.selectbox("Select Growth Type", ["YoY (Annual)", "QoQ (Quarterly)"])
    period = 'annual' if "YoY" in growth_type else 'quarterly'

    ticker_names = {t: yf.Ticker(t).info.get("shortName", t) for t in tickers}

    metrics = {
        "Total Revenue": "📊 Revenue Growth",
        "Net Income": "📉 Net Income Growth"
    }

    @st.cache_data(show_spinner=False)
    def get_financial_data_local(tickers):
        data = {}
        for t in tickers:
            ticker_obj = yf.Ticker(t)
            try:
                data[t] = {
                    'financials': ticker_obj.financials.T if not ticker_obj.financials.empty else pd.DataFrame(),
                    'balance_sheet': ticker_obj.balance_sheet.T if not ticker_obj.balance_sheet.empty else pd.DataFrame(),
                    'cashflow': ticker_obj.cashflow.T if not ticker_obj.cashflow.empty else pd.DataFrame()
                }
            except Exception as e:
                st.error(f"Error loading data for {t}: {e}")
        return data

    financials = get_financial_data_local(tickers)

    def get_metric_df(metric, period='annual'):
        df_dict = {}
        for t in tickers:
            try:
                f = financials[t]
                if metric in f['financials'].columns:
                    base_df = f['financials']
                elif metric in f['balance_sheet'].columns:
                    base_df = f['balance_sheet']
                elif metric in f['cashflow'].columns:
                    base_df = f['cashflow']
                else:
                    continue

                df = base_df[[metric]].copy()
                df.index = df.index.strftime('%Y' if period == 'annual' else '%Y-%m')
                df_dict[t] = df[metric]

            except Exception as e:
                print(f"❌ {metric} for {t}: {e}")
        return pd.DataFrame(df_dict)

    metric_dfs = {m: get_metric_df(m, period) for m in metrics}

    st.subheader(f"📽️ {growth_type} Growth Charts (with % change)")
    metric_items = list(metrics.items())
    for i in range(0, len(metric_items), 2):
        cols = st.columns(2)
        for j, (metric, title) in enumerate(metric_items[i:i+2]):
            with cols[j]:
                df = metric_dfs.get(metric, pd.DataFrame())
                if not df.empty:
                    long_df = format_long_form(df, ticker_names)
                    fig = px.bar(long_df, x='Period', y='Value', color='Company',
                                 barmode='group', title=f"{title} — {growth_type}", height=450)
                    st.plotly_chart(fig, use_container_width=True)

                    fig2 = px.line(long_df, x='Period', y='Growth %', color='Company',
                                   markers=True, title=f"{title} % Change — {growth_type}", height=350)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning(f"⚠️ No data available for {metric}")

elif section == "Price Trends & Sharpe Ratio":
    st.subheader("📈 Price Trends & Sharpe Ratio")
    period = st.sidebar.selectbox("Select period:", ["1y", "3y", "5y"], index=1)
    interval = st.sidebar.selectbox("Select interval:", ["1d", "1wk", "1mo"], index=1)
    st.session_state["tickers"] = tickers
    st.session_state["benchmark"] = benchmark
    st.session_state["period"] = period
    st.session_state["interval"] = interval

    if "tickers" not in st.session_state:
        st.session_state["tickers"] = ["AAPL", "MSFT"]
    if "benchmark" not in st.session_state:
        st.session_state["benchmark"] = "SPY"

    tickers_local = st.session_state["tickers"]
    benchmark_local = st.session_state["benchmark"]
    all_tickers = tickers_local + [benchmark_local]

    price_df = yf.download(all_tickers, period="3y", interval="1wk", auto_adjust=True)['Close']
    price_df = price_df.dropna(how="all")

    returns = price_df[tickers_local].pct_change().dropna()
    daily_rf = 0.00026
    sharpe_ratios = [{
        "Stock": s,
        "Sharpe Ratio": round(((returns[s].mean() - daily_rf) / returns[s].std()) * np.sqrt(252), 2)
    } for s in tickers_local]

    st.subheader("📊 Sharpe Ratios (3Y Weekly)")
    st.dataframe(pd.DataFrame(sharpe_ratios), use_container_width=True)
    st.session_state["sharpe_ratios"] = sharpe_ratios

    st.subheader("📈 Price Trend vs Benchmark (3Y Weekly)")
    for i in range(0, len(tickers_local), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(tickers_local):
                ticker = tickers_local[i + j]
                with cols[j]:
                    fig = go.Figure()
                    df_pair = price_df[[ticker, benchmark_local]].dropna()
                    if not df_pair.empty:
                        fig.add_trace(go.Scatter(x=df_pair.index, y=df_pair[ticker], mode='lines', name=ticker))
                        fig.add_trace(go.Scatter(x=df_pair.index, y=df_pair[benchmark_local], mode='lines', name=f"Benchmark ({benchmark_local})", line=dict(dash='dot')))
                        fig.update_layout(title=f"{ticker} vs {benchmark_local}", xaxis_title="Date", yaxis_title="Price", showlegend=True, height=300)
                        st.plotly_chart(fig, use_container_width=True)

elif section == "ARIMA Forecast":
    st.subheader("🔮 ARIMA Forecast")
    if not has_arima:
        st.error("statsmodels ARIMA not available in this environment. Install statsmodels to use this feature.")
    else:
        st.title("📈 ARIMA Forecast with Confidence Interval & Fibonacci Fan")
        tickers_local = st.session_state.get("tickers", tickers)
        if not tickers_local:
            st.warning("⚠️ No tickers found. Please enter tickers.")
            st.stop()

        period = st.selectbox("Select history period:", ["1y", "3y", "5y"], index=1)
        interval = st.selectbox("Select interval:", ["1d", "1wk"], index=1)

        try:
            price_df = yf.download(tickers_local, period=period, interval=interval, auto_adjust=True)["Close"]
            if isinstance(price_df, pd.Series):
                price_df = price_df.to_frame()
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            st.stop()

        for s in tickers_local:
            st.subheader(f"📊 {s} Forecast")
            if s not in price_df.columns:
                st.warning(f"No price series for {s}")
                continue
            series = price_df[s].dropna()
            if len(series) < 60:
                st.warning(f"⚠️ Not enough data for {s} (need at least 60 points)")
                continue

            try:
                model = ARIMA(series, order=(5, 1, 0))
                model_fit = model.fit()
                pred = model_fit.get_forecast(steps=12)
                forecast = pred.predicted_mean
                conf_int = pred.conf_int()
                future_index = pd.date_range(start=series.index[-1] + timedelta(days=7), periods=12, freq="W")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Actual"))
                fig.add_trace(go.Scatter(x=future_index, y=forecast, mode="lines", name="ARIMA Forecast"))

                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
                if len(lower) == len(future_index):
                    fig.add_trace(go.Scatter(
                        x=list(future_index) + list(future_index[::-1]),
                        y=list(lower) + list(upper[::-1]),
                        fill='toself',
                        fillcolor='rgba(255,200,200,0.3)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        name='95% CI'
                    ))

                last_window = series[-60:]
                last_low = last_window.min()
                last_high = last_window.max()
                trend_start = last_window.index[0]
                trend_end = last_window.index[-1]
                fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
                for ratio in fib_ratios:
                    level = last_low + (last_high - last_low) * ratio
                    fig.add_trace(go.Scatter(x=[trend_start, trend_end + timedelta(days=90)], y=[last_low, level], mode='lines', line=dict(dash='dot'), name=f'Fib {ratio}'))

                fig.update_layout(title=f"{s} — ARIMA Forecast + 95% CI + Fib Fan", xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ ARIMA failed for {s}: {e}")

elif section == "LSTM Forecast":
    st.subheader("🤖 LSTM Forecast")
    if not has_tf:
        st.error("TensorFlow/Keras not available in this environment. Install tensorflow to use LSTM forecasts.")
    else:
        tickers_local = st.session_state.get("tickers", tickers)
        try:
            price_df = yf.download(tickers_local, period="3y", interval="1wk", auto_adjust=True)["Close"]
            if isinstance(price_df, pd.Series):
                price_df = price_df.to_frame()
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            st.stop()

        from sklearn.preprocessing import MinMaxScaler
        for s in tickers_local:
            st.subheader(f"📈 {s} — LSTM")
            if s not in price_df.columns:
                st.warning(f"No price for {s}")
                continue
            series = price_df[s].dropna().values.reshape(-1, 1)
            if len(series) < 70:
                st.warning(f"⚠️ Not enough data for {s} to train LSTM.")
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series)
            seq_len = min(60, len(scaled) - 1)
            X, y = [], []
            for i in range(seq_len, len(scaled)):
                X.append(scaled[i-seq_len:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=5, batch_size=16, verbose=0)

            last_seq = scaled[-seq_len:]
            forecast_scaled = []
            cur = last_seq.copy()
            for _ in range(12):
                p = model.predict(cur.reshape(1, seq_len, 1), verbose=0)
                forecast_scaled.append(p[0, 0])
                cur = np.append(cur[1:], p).reshape(seq_len, 1)

            forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
            future_index = pd.date_range(start=price_df.index[-1] + timedelta(days=7), periods=12, freq='W')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[s], name='Actual'))
            fig.add_trace(go.Scatter(x=future_index, y=forecast, name='LSTM Forecast'))
            fig.update_layout(title=f"{s} — LSTM Forecast", xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)

elif section == "News Sentiment":
    st.subheader("📰 News Sentiment Analysis")

    # try to load FinBERT (optional)
    if has_transformers:
        try:
            tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            nb_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            use_finbert = True
        except Exception as e:
            st.warning(f"Could not load FinBERT model: {e}. Falling back to simple sentiment.")
            use_finbert = False
    else:
        use_finbert = False

    def classify_news(text):
        if use_finbert:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = nb_model(**inputs)
                probs = softmax(outputs.logits.detach().numpy()[0])
                labels = ['neutral', 'positive', 'negative']
                sentiment = labels[probs.argmax()]
                if sentiment == "positive" and probs.max() > 0.75:
                    return 3
                elif sentiment == "positive":
                    return 2
                elif sentiment == "neutral":
                    return 1
                else:
                    return 0
            except Exception:
                return 1
        else:
            txt = text.lower()
            if any(w in txt for w in ["beat", "profit", "growth", "raise"]):
                return 2
            if any(w in txt for w in ["miss", "fall", "drop", "decline", "loss"]):
                return 0
            return 1

    def get_google_news(query, max_items=5):
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'summary': entry.get('summary', '')
            })
        return items

    tickers_session = tickers if tickers else st.session_state.get('tickers', [])
    if not tickers_session:
        st.warning("⚠️ No tickers found. Please enter tickers.")
        st.stop()

    sentiment_stats = []
    for t in tickers_session:
        info = yf.Ticker(t).info
        name = info.get('shortName', t)
        news = get_google_news(name, max_items=5)

        with st.expander(f"🗞️ {name}"):
            scores = []
            for item in news:
                title = item.get('title', '')
                summary = item.get('summary', '')
                full = f"{title} {summary}".strip()
                score = classify_news(full)
                scores.append(score)
                label = {3: '📈 Bullish', 2: '↗️ Mild Bullish', 1: '➡️ Neutral', 0: '📉 Bearish'}[score]
                st.markdown(f"- [{title}]({item.get('link','')}) — *{item.get('published','')}* — **Score: {score} ({label})**")

        if scores:
            sentiment_stats.append({
                'Ticker': t,
                'Company': name,
                'Bullish (%)': round((scores.count(3)/len(scores))*100,1),
                'Weak Bullish (%)': round((scores.count(2)/len(scores))*100,1),
                'Neutral (%)': round((scores.count(1)/len(scores))*100,1),
                'Bearish (%)': round((scores.count(0)/len(scores))*100,1),
                'Avg Sentiment Score': round(np.mean(scores),2)
            })

    sentiment_df = pd.DataFrame(sentiment_stats)
    st.subheader("📊 News Sentiment Summary Table")
    st.dataframe(sentiment_df, use_container_width=True)
    st.session_state['sentiment_df'] = sentiment_df

    if not sentiment_df.empty:
        df_long = sentiment_df.melt(id_vars=['Ticker','Company','Avg Sentiment Score'], value_vars=['Bullish (%)','Weak Bullish (%)','Neutral (%)','Bearish (%)'], var_name='Sentiment', value_name='Percentage')
        fig = px.bar(df_long, x='Company', y='Percentage', color='Sentiment', barmode='stack', title='🧠 News Sentiment Distribution per Stock')
        st.plotly_chart(fig, use_container_width=True)

elif section == "Technical & Sentiment Analyzer":
    st.title("📈 Multi-Ticker Technical & Sentiment Analyzer")
    st.info("Each ticker shows: Retracement | Fan | Projection plots and returns Technical + Sentiment + Fundamental scores used in allocation")

    sample_fundamentals = {
        'revenue_growth': 22,
        'profit_growth': 18,
        'debt_equity': 0.3,
        'roce': 20,
        'promoter_holding': 65
    }

    technical_scores = []
    sentiment_scores = []
    fundamental_scores = []

    for ticker in tickers:
        st.markdown(f"### 🔍 {ticker}")
        data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

        if data.empty:
            st.error(f"No data for {ticker}")
            technical_scores.append({'Ticker': ticker, 'Technical Score': 0.0})
            sentiment_scores.append({'Ticker': ticker, 'Sentiment Score': 2.5})
            fundamental_scores.append({'Ticker': ticker, 'Fundamental Score': 0.0})
            continue

        col1, col2, col3 = st.columns(3)
        with col1:
            fig1 = plot_fibonacci_retracement(data, ticker)
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            fig2 = plot_fibonacci_fan(data, ticker)
            st.pyplot(fig2)
            plt.close(fig2)
        with col3:
            fig3 = plot_fibonacci_projection(data, ticker)
            st.pyplot(fig3)
            plt.close(fig3)

        sentiment = calculate_sentiment_score(ticker)
        tech_score = calculate_technical_score(ticker)
        fund_score = calculate_fundamental_score(sample_fundamentals)

        # Normalize/scale scores into 0-1 for allocation later
        tech_norm = min(max(tech_score / 4.0, 0), 1)  # tech max ~4
        sent_norm = min(max(sentiment / 5.0, 0), 1)   # sentiment scaled 0-5
        fund_norm = min(max(fund_score / 5.0, 0), 1)  # fundamental 0-5 -> 0-1

        technical_scores.append({'Ticker': ticker, 'Technical Score': tech_norm})
        sentiment_scores.append({'Ticker': ticker, 'Sentiment Score': sent_norm})
        fundamental_scores.append({'Ticker': ticker, 'Fundamental Score': fund_norm})

        total = sentiment + tech_score + fund_score
        decision = "Buy" if total >= 8.0 else "Hold" if total >= 5 else "Sell"

        st.metric(label="📊 Total Score", value=round(total, 2), delta=decision)
        st.write(f"- **Fundamental Score**: {fund_score}/5 (norm {fund_norm})")
        st.write(f"- **Technical Score**: {tech_score}/4 (norm {tech_norm})")
        st.write(f"- **Sentiment Score**: {sentiment}/5 (norm {sent_norm})")
        st.markdown("---")

    # Save for allocation
    st.session_state['technical_scores'] = technical_scores
    st.session_state['sentiment_scores_local'] = sentiment_scores
    st.session_state['fundamental_scores_local'] = fundamental_scores

elif section == "Capital Allocation Matrix":
    st.subheader("📊 Sector & Capital Allocation")

    df_score = st.session_state.get('df_score')
    sharpe_ratios = st.session_state.get('sharpe_ratios')
    rpn_list = st.session_state.get('rpn_list')
    sentiment_df = st.session_state.get('sentiment_df')
    technical_scores = st.session_state.get('technical_scores')

    if any(x is None for x in [df_score, sharpe_ratios, rpn_list, technical_scores]):
        st.error("⚠️ Missing inputs from other pages. Please run Score, Sharpe, RPN, Technical Analyzer pages first.")
        st.stop()

    # Merge base metrics
    score_df = df_score.merge(pd.DataFrame(rpn_list), on='Ticker')
    score_df = score_df.merge(pd.DataFrame(sharpe_ratios), left_on='Ticker', right_on='Stock').drop(columns=['Stock'])

    # Add sentiment
    if sentiment_df is not None and 'Avg Sentiment Score' in sentiment_df.columns:
        score_df = score_df.merge(sentiment_df[['Ticker','Avg Sentiment Score']], on='Ticker', how='left')
    else:
        local_sent = pd.DataFrame(st.session_state.get('sentiment_scores_local', []))
        if not local_sent.empty:
            score_df = score_df.merge(local_sent.rename(columns={'Sentiment Score':'Avg Sentiment Score'}), on='Ticker', how='left')
        else:
            score_df['Avg Sentiment Score'] = 0

    # Add technical scores
    tech_df = pd.DataFrame(technical_scores)
    if tech_df.empty:
        score_df['Technical Score'] = 0
    else:
        score_df = score_df.merge(tech_df, on='Ticker', how='left').fillna(0)

    # Normalize metrics
    scaler = MinMaxScaler()
    cols_to_norm = ['Total Score','RPN','Sharpe Ratio','Avg Sentiment Score','Technical Score']
    for c in cols_to_norm:
        if c not in score_df.columns:
            score_df[c] = 0

    score_df[cols_to_norm] = scaler.fit_transform(score_df[cols_to_norm])

    # Invert RPN (lower better)
    score_df['RPN'] = 1 - score_df['RPN']

    # Final Score weights (include technical)
    score_df['Final Score'] = (
        0.35 * score_df['Total Score'] +
        0.18 * score_df['Sharpe Ratio'] +
        0.17 * score_df['Avg Sentiment Score'] +
        0.15 * score_df['RPN'] +
        0.15 * score_df['Technical Score']
    )

    total_capital = st.sidebar.number_input("Enter Total Capital to Allocate (₹)", value=10_00_000, step=50_000)
    score_df['Allocation %'] = (score_df['Final Score']/score_df['Final Score'].sum()*100).round(2)
    score_df['Capital Allocation (₹)'] = ((score_df['Allocation %']/100)*total_capital).round(0)

    st.subheader("📊 Final Allocation Matrix (includes Technical Score)")
    st.dataframe(score_df[["Ticker","Total Score","Sharpe Ratio","Avg Sentiment Score","RPN","Technical Score","Final Score","Allocation %","Capital Allocation (₹)"]], use_container_width=True)
    fig = px.pie(score_df, names='Ticker', values='Capital Allocation (₹)', title='💹 Capital Allocation')
    st.plotly_chart(fig, use_container_width=True)

elif section == "Chart Toolbox":
    st.subheader("🧰 Chart Drawing Toolbox")

    import io, base64, gc
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    import streamlit as st
    import matplotlib.pyplot as plt
    
    # ----------------------------
    # Select ticker
    # ----------------------------
    tickers = st.session_state.get("tickers", ["AAPL", "MSFT", "GOOG"])
    selected_ticker = st.selectbox("Select ticker to annotate:", tickers)
    
    # ----------------------------
    # Prepare background image for the selected ticker
    # ----------------------------
    if "ticker_figs" in st.session_state and selected_ticker in st.session_state["ticker_figs"]:
        fig = st.session_state["ticker_figs"][selected_ticker]
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)
        bg_img = Image.open(buf).convert("RGB")
        plt.close(fig)
    else:
        bg_img = None
    
    # ----------------------------
    # Dynamic canvas sizing
    # ----------------------------
    if bg_img:
        orig_h, orig_w = bg_img.height, bg_img.width
    else:
        orig_h, orig_w = 400, 500
    
    container_width = st.session_state.get("container_width", 700)
    aspect_ratio = orig_h / orig_w
    canvas_w = container_width
    canvas_h = int(container_width * aspect_ratio)
    
    # Convert bg_img to base64 for canvas
    if bg_img is not None:
        buf = io.BytesIO()
        bg_img.save(buf, format="PNG")
        buf.seek(0)
        bg_img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        bg_img_data = f"data:image/png;base64,{bg_img_b64}"
    else:
        bg_img_data = None
    
    # ----------------------------
    # Initialize per-ticker canvas state
    # ----------------------------
    if "ticker_canvas_objects" not in st.session_state:
        st.session_state["ticker_canvas_objects"] = {}
    
    # Load previous objects for this ticker if they exist
    initial_objects = st.session_state["ticker_canvas_objects"].get(selected_ticker, [])
    
    # ----------------------------
    # Display canvas
    # ----------------------------
    canvas_result = st_canvas(
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=bg_img_data,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=st.selectbox("Drawing mode", ["line", "point"]),
        key=f"canvas_{selected_ticker}",
        display_toolbar=True,
        initial_drawing=initial_objects  # pre-load previous drawings
    )
    
    # ----------------------------
    # Save canvas objects per ticker
    # ----------------------------
    if canvas_result.json_data:
        st.session_state["ticker_canvas_objects"][selected_ticker] = canvas_result.json_data.get("objects", [])
    
    objs = st.session_state["ticker_canvas_objects"].get(selected_ticker, [])
    
    # ----------------------------
    # Helper functions: pixel → date/price
    # ----------------------------
    def px_to_price(px_y):
        ymin, ymax = df['Close'].min(), df['Close'].max()
        return ymax - (px_y / canvas_h) * (ymax - ymin)
    
    def px_to_date(px_x):
        idx = int((px_x / canvas_w) * (len(df.index) - 1))
        return df.index[idx]
    
    # ----------------------------
    # Drawing tools
    # ----------------------------
    tool_buttons = {
        "Fibonacci": f"fib_{selected_ticker}",
        "Elliott": f"elliott_{selected_ticker}",
        "Gartley": f"gartley_{selected_ticker}",
        "Trend Lines": f"trend_{selected_ticker}"
    }
    
    # Fibonacci Retracement
    if tool_choice == "Fibonacci Retracement" and st.button("Apply Fibonacci", key=tool_buttons["Fibonacci"]):
        last_line = next((o for o in reversed(objs) if o.get("type")=="line"), None)
        if last_line:
            x1, y1 = last_line["left"], last_line["top"]
            x2, y2 = x1 + last_line["width"], y1 + last_line["height"]
            high, low = max(px_to_price(y1), px_to_price(y2)), min(px_to_price(y1), px_to_price(y2))
            fig = plot_fibonacci_retracement(df, high, low)
            st.pyplot(fig)
        else:
            st.warning("Draw a line first.")
    
    # Elliott Wave
    elif tool_choice == "Elliott Wave" and st.button("Apply Elliott", key=tool_buttons["Elliott"]):
        pts = [o for o in objs if o.get("type")=="circle"]
        if len(pts) >= 5:
            prices = [px_to_price(o["top"]) for o in pts[:5]]
            fig = plot_elliott_wave(df, prices)
            st.pyplot(fig)
        else:
            st.warning("Mark at least 5 points.")
    
    # Gartley Pattern
    elif tool_choice == "Gartley Pattern" and st.button("Apply Gartley", key=tool_buttons["Gartley"]):
        pts = [o for o in objs if o.get("type")=="circle"]
        if len(pts) >= 5:
            prices = [px_to_price(o["top"]) for o in pts[:5]]
            fig = plot_gartley_pattern(df, prices)
            st.pyplot(fig)
        else:
            st.warning("Mark 5 points (X,A,B,C,D).")
    
    # Trend Lines
    elif tool_choice == "Trend Lines" and st.button("Apply Trend Lines", key=tool_buttons["Trend Lines"]):
        lines = []
        for o in objs:
            if o.get("type")=="line":
                x1, y1 = o["left"], o["top"]
                x2, y2 = x1 + o["width"], y1 + o["height"]
                lines.append((
                    int(px_to_date(x1).day), px_to_price(y1),
                    int(px_to_date(x2).day), px_to_price(y2)
                ))
        if lines:
            fig = plot_trend_lines(df, lines)
            st.pyplot(fig)
        else:
            st.warning("Draw at least one line.")
    
    # ----------------------------
    # Cleanup
    # ----------------------------
    gc.collect()

