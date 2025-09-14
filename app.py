# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import feedparser
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide", page_title="Stock Analyzer")

# ========== SETTINGS ==========
st.sidebar.header("Select Tickers")
default_tickers = ["BIOCON.NS", "ERIS.NS", "FORTIS.NS", "VIMTALABS.NS", "SUPRIYA.NS"]
user_tickers = st.sidebar.text_area("Enter NSE tickers (comma-separated)", value=", ".join(default_tickers)).split(",")
user_benchmark = st.sidebar.text_input("Enter benchmark ticker", value="^CNXPHARMA")


tickers = [t.strip().upper() for t in user_tickers if t.strip()]
benchmark = user_benchmark.strip().upper()

# =============================
# SECTION 1: Financial Scoring
# =============================

@st.cache_data
def load_financials(tickers):
    stocks = {t: yf.Ticker(t) for t in tickers}
    return {
        t: {
            'annual': stocks[t].financials.T,
            'quarterly': stocks[t].quarterly_financials.T
        }
        for t in tickers
    }

financials = load_financials(tickers)

def compute_scores(fin_annual, fin_quarterly, metrics):
    scores = {}
    fin_annual = fin_annual.sort_index()
    fin_quarterly = fin_quarterly.sort_index()

    for metric in metrics:
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
        scores[metric] = score
    return scores

financial_metrics = [
    "Total Revenue", "Net Income", "Free Cash Flow",
    "Operating Income", "Operating Expense", "Gross Profit",
    "Operating Cash Flow", "Capital Expenditures"
]

earnings_metrics = [
    "Total Revenue", "Net Income", "Free Cash Flow",
    "Return On Equity", "Return on assest", "Trailing P/E", "Price To Sales Ratio"
]

# Compute scores
result = []
for t in tickers:
    fin = financials[t]
    f_score = compute_scores(fin['annual'], fin['quarterly'], financial_metrics)
    e_score = compute_scores(fin['annual'], fin['quarterly'], earnings_metrics)
    total = sum(f_score.values()) * 10 + sum(e_score.values()) * 8
    result.append({
        "Ticker": t,
        "Financial Score": sum(f_score.values()) * 10,
        "Earnings Score": sum(e_score.values()) * 8,
        "Total Score": total
    })

df_score = pd.DataFrame(result).sort_values("Total Score", ascending=False)

st.subheader("📊 Stock Ranking by Financial & Earnings Score")
st.dataframe(df_score, use_container_width=True)

# =============================
# SECTION 2: Peer Comparison
# =============================

st.subheader("📌 Peer Valuation & Ratios")

peer_data = []
for t in tickers:
    info = yf.Ticker(t).info
    peer_data.append({
        'Ticker': t,
        'Sector': info.get('sector'),
        'P/E': info.get('trailingPE'),
        'P/B': info.get('priceToBook'),
        'ROE': info.get('returnOnEquity'),
        'ROA': info.get('returnOnAssets'),
        'Debt/Equity': info.get('debtToEquity', 0)/100,
        'Beta': info.get('beta'),
        'Dividend Yield': info.get('dividendYield')
    })

peer_df = pd.DataFrame(peer_data)
st.dataframe(peer_df, use_container_width=True)

# =============================
# 🔺 RPN + DCF Valuation Table
# =============================
growth = 0.05
discount = 0.10

ni_df = pd.DataFrame()
for t in tickers:
    f = yf.Ticker(t).financials.T
    if 'Net Income' in f.columns:
        ni_df[t] = f['Net Income']

rpn_list, dcf_list = [], []
for _, row in peer_df.iterrows():
    if row['Ticker'] == benchmark: continue
    risk = min(row['Debt/Equity'] or 1, 5)
    prob = min(row['Beta'] or 1, 5)
    detect = 5 if (row['ROE'] or 0) < 0.10 else 1
    rpn = risk * prob * detect

    ni = ni_df[row['Ticker']].iloc[-1] if not ni_df.empty and row['Ticker'] in ni_df.columns else 0
    dcf_val = (ni * (1 + growth)) / (discount - growth) if (discount - growth) != 0 else np.nan

    rpn_list.append({'Ticker': row['Ticker'], 'RPN': round(rpn, 2)})
    dcf_list.append({'Ticker': row['Ticker'], 'Last Net Income': round(ni, 2), 'Approx. DCF Value': round(dcf_val, 2)})

st.subheader("🔍 RPN (Risk Priority Number)")
st.dataframe(pd.DataFrame(rpn_list), use_container_width=True)

# =============================
# SECTION 3: Revenue & Income Trends
# =============================

rev_df, ni_df = pd.DataFrame(), pd.DataFrame()
for t in tickers:
    fin = yf.Ticker(t).financials.T
    if "Total Revenue" in fin.columns:
        rev_df[t] = fin["Total Revenue"]
    if "Net Income" in fin.columns:
        ni_df[t] = fin["Net Income"]

import plotly.express as px
import plotly.express as px
import streamlit as st

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

# ✅ 1. Select Growth Type
growth_type = st.selectbox("Select Growth Type", ["YoY (Annual)", "QoQ (Quarterly)"])
period = 'annual' if "YoY" in growth_type else 'quarterly'

# ✅ 2. Your selected tickers
tickers = tickers
ticker_names = {t: yf.Ticker(t).info.get("shortName", t) for t in tickers}

# ✅ 3. Metrics dictionary
metrics = {
    "Total Revenue": "📊 Revenue Growth",
    "Net Income": "📉 Net Income Growth",
    "Cost Of Revenue": "🧾 COGS",
    "Long Term Debt": "🏦 Debt",
    "Normalized EBITDA": "📈 EBITDA Growth"
}

# ✅ 4. Load yfinance statements into structured dict
@st.cache_data(show_spinner=False)
def get_financial_data(tickers):
    data = {}
    for t in tickers:
        ticker_obj = yf.Ticker(t)
        try:
            data[t] = {
                'financials': ticker_obj.financials.T,
                'balance_sheet': ticker_obj.balance_sheet.T,
                'cashflow': ticker_obj.cashflow.T
            }
        except Exception as e:
            st.error(f"Error loading data for {t}: {e}")
    return data

financials = get_financial_data(tickers)

# ✅ 5. Extract individual metric dataframe
def get_metric_df(metric, period='annual'):
    df_dict = {}
    for t in tickers:
        try:
            f = financials[t]
            # Decide which statement to pull from
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

# ✅ 6. Format for long-form plotting
def format_long_form(df, metric_name):
    df_long = df.T.reset_index().melt(id_vars='index', var_name='Period', value_name=metric_name)
    df_long.columns = ['Ticker', 'Period', metric_name]
    df_long['Company'] = df_long['Ticker'].map(ticker_names)
    return df_long

# ✅ 7. Build metric_dfs
metric_dfs = {m: get_metric_df(m, period) for m in metrics}

# ✅ 8. Plotting layout
st.subheader(f"📽️ {growth_type} Animated Growth Charts")
metric_items = list(metrics.items())
for i in range(0, len(metric_items), 2):
    cols = st.columns(2)
    for j, (metric, title) in enumerate(metric_items[i:i+2]):
        with cols[j]:
            df = metric_dfs.get(metric, pd.DataFrame())
            if not df.empty:
                long_df = format_long_form(df, metric)
                fig = px.bar(
                    long_df, x='Company', y=metric, color='Company',
                    animation_frame=long_df['Period'].astype(str),
                    title=f"{title} — {growth_type}", height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ No data available for {metric}")


# =============================
# SECTION 4: Price Trends & Sharpe Ratio
# =============================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ✅ Tickers and Benchmark
tickers = tickers
benchmark = benchmark
all_tickers = tickers + [benchmark]

# ✅ Download price data
price_df = yf.download(all_tickers, period="3y", interval="1wk", auto_adjust=True)['Close']

# ✅ Sharpe Ratio Calculation
returns = price_df[tickers].pct_change().dropna()
daily_rf = 0.00026  # ~6.5% annualized / 252
sharpe_ratios = [{
    "Stock": s,
    "Sharpe Ratio": round(((returns[s].mean() - daily_rf) / returns[s].std()) * np.sqrt(252), 2)
} for s in tickers]

# ✅ Display Sharpe Ratios
st.subheader("📊 Sharpe Ratios (3Y Weekly)")
st.dataframe(pd.DataFrame(sharpe_ratios), use_container_width=True)

# ✅ Price Trend: Grid Layout
st.subheader("📈 Price Trend vs Benchmark (3Y Weekly)")

# Grid: 2 columns
for i in range(0, len(tickers), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(tickers):
            ticker = tickers[i + j]
            with cols[j]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_df.index, y=price_df[ticker],
                                         mode='lines', name=ticker))
                fig.add_trace(go.Scatter(x=price_df.index, y=price_df[benchmark],
                                         mode='lines', name='Benchmark (Nifty)', line=dict(dash='dot')))
                fig.update_layout(
                    title=f"{ticker} vs Nifty",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    showlegend=True,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)


# =============================
# SECTION 5: ARIMA Forecast with CI
# =============================
st.subheader("🔮 ARIMA Forecast + Confidence Interval (Grid View)")

from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from datetime import timedelta

# Group tickers in pairs for 2-column layout
ticker_pairs = [tickers[i:i+2] for i in range(0, len(tickers), 2)]

for pair in ticker_pairs:
    cols = st.columns(len(pair))
    for idx, s in enumerate(pair):
        with cols[idx]:
            series = price_df[s].dropna()

            # ARIMA model fitting
            model = ARIMA(series, order=(5, 1, 0))
            fit = model.fit()
            pred = fit.get_forecast(steps=12)
            forecast = pred.predicted_mean
            conf_int = pred.conf_int()
            future_index = pd.date_range(start=series.index[-1] + timedelta(days=7), periods=12, freq='W')

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Actual"))
            fig.add_trace(go.Scatter(x=future_index, y=forecast, name="Forecast"))
            fig.add_trace(go.Scatter(
                x=list(future_index) + list(future_index[::-1]),
                y=list(conf_int.iloc[:, 0]) + list(conf_int.iloc[:, 1][::-1]),
                fill='toself', fillcolor='rgba(200,200,255,0.3)', name='95% CI'
            ))

            fig.update_layout(
                title=f"📈 {s} — ARIMA 12-Week Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

# =============================
# SECTION 5: LSTM Forecast with CI
# =============================

st.subheader("🤖 LSTM Forecast (12 Weeks)")
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import numpy as np

# Group tickers in pairs for 2-column layout
ticker_pairs = [tickers[i:i+2] for i in range(0, len(tickers), 2)]

for pair in ticker_pairs:
    cols = st.columns(len(pair))  # One column per ticker
    for idx, s in enumerate(pair):
        with cols[idx]:
            series = price_df[s].dropna().values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(series)
            max_seq_len = 60
            seq_len = min(max_seq_len, len(scaled_series) - 1)

            X, y = [], []
            
            for i in range(seq_len, len(scaled_series)):
                X.append(scaled_series[i-seq_len:i])
                y.append(scaled_series[i])
            X, y = np.array(X), np.array(y)

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=5, batch_size=16, verbose=0)

            # Forecast next 12 weeks
            last_seq = scaled_series[-seq_len:]
            forecast_scaled = []
            cur_seq = last_seq
            for _ in range(12):
                pred = model.predict(cur_seq.reshape(1, seq_len, 1), verbose=0)
                forecast_scaled.append(pred[0, 0])
                cur_seq = np.append(cur_seq[1:], pred).reshape(seq_len, 1)

            forecast_lstm = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
            future_index = pd.date_range(start=price_df.index[-1] + timedelta(days=7), periods=12, freq='W')

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[s], name='Actual'))
            fig.add_trace(go.Scatter(x=future_index, y=forecast_lstm, name='LSTM Forecast'))

            fig.update_layout(
                title=f"📈 {s} — LSTM Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================
# SECTION 6: News
# =============================

import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# =============================
# FinBERT: Load Model & Tokenizer
# =============================
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits.detach().numpy()[0])
    labels = ['neutral', 'positive', 'negative']
    sentiment = labels[probs.argmax()]

    if sentiment == "positive" and probs.max() > 0.75:
        return 3  # Strong bullish
    elif sentiment == "positive":
        return 2  # Weak/potential bullish
    elif sentiment == "neutral":
        return 1  # Horizontal
    else:
        return 0  # Bearish

# =============================
# Function: Get News from Google RSS
# =============================
def get_google_news(query, max_items=3):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [{
        'title': entry.title,
        'link': entry.link,
        'published': entry.published,
        'summary': entry.get('summary', '')  # ✅ Added summary safely
    } for entry in feed.entries[:max_items]]

# =============================
# News Summary with Sentiment Score
# =============================
st.subheader("📰 News Sentiment Summary")

for t in tickers:
    info = yf.Ticker(t).info
    name = info.get("shortName", t)
    news = get_google_news(name)

    with st.expander(f"🗞️ {name}"):
        for item in news:
            title = item.get('title', '')
            summary = item.get('summary', '')
            full_text = f"{title} {summary}"
            score = classify_news(full_text.strip())
            sentiment_label = {3: "📈 Bullish", 2: "↗️ Mild Bullish", 1: "➡️ Neutral", 0: "📉 Bearish"}[score]

            st.markdown(
                f"- [{title}]({item['link']}) — *{item['published']}* — **Score: {score} ({sentiment_label})**"
            )
import plotly.express as px

# Collect sentiment stats
sentiment_stats = []

for t in tickers:
    info = yf.Ticker(t).info
    name = info.get("shortName", t)
    news = get_google_news(name)

    score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    scores = []

    for item in news:
        title = item.get('title', '')
        summary = item.get('summary', '')
        full_text = f"{title} {summary}"
        score = classify_news(full_text.strip())
        score_counts[score] += 1
        scores.append(score)

    total = sum(score_counts.values())
    if total == 0:
        continue

    sentiment_stats.append({
        "Ticker": t,
        "Company": name,
        "Bullish (%)": round((score_counts[3] / total) * 100, 1),
        "Weak Bullish (%)": round((score_counts[2] / total) * 100, 1),
        "Neutral (%)": round((score_counts[1] / total) * 100, 1),
        "Bearish (%)": round((score_counts[0] / total) * 100, 1),
        "Avg Sentiment Score": round(np.mean(scores), 2)
    })

# Show Summary Table
sentiment_df = pd.DataFrame(sentiment_stats)
st.subheader("📊 News Sentiment Summary Table")
st.dataframe(sentiment_df, use_container_width=True)

# Plot Summary Chart
if not sentiment_df.empty:
    df_long = sentiment_df.melt(id_vars=["Ticker", "Company", "Avg Sentiment Score"], 
                                value_vars=["Bullish (%)", "Weak Bullish (%)", "Neutral (%)", "Bearish (%)"],
                                var_name="Sentiment", value_name="Percentage")
    fig = px.bar(df_long, x="Company", y="Percentage", color="Sentiment", barmode="stack", 
                 title="🧠 News Sentiment Distribution per Stock")
    st.plotly_chart(fig, use_container_width=True)


# =============================
# SECTION 7: Capital Allocation Matrix
# =============================
st.subheader("💰 Capital Allocation Based on Composite Score")

# Input total capital
total_capital = st.number_input("Enter total capital (₹)", min_value=10000, value=1000000, step=10000)

# Merge all metrics
df_allocation = df_score.merge(pd.DataFrame(sharpe_ratios), left_on="Ticker", right_on="Stock", how="left")
df_allocation = df_allocation.merge(pd.DataFrame(rpn_list), on="Ticker", how="left")
df_allocation.drop(columns=["Stock"], inplace=True)

# Handle missing or zero std Sharpe
df_allocation.fillna(0, inplace=True)

# Normalize inputs
df_allocation["Score_Norm"] = (df_allocation["Total Score"] - df_allocation["Total Score"].min()) / (df_allocation["Total Score"].max() - df_allocation["Total Score"].min())
df_allocation["Sharpe_Norm"] = (df_allocation["Sharpe Ratio"] - df_allocation["Sharpe Ratio"].min()) / (df_allocation["Sharpe Ratio"].max() - df_allocation["Sharpe Ratio"].min())
df_allocation["RPN_Norm"] = 1 - ((df_allocation["RPN"] - df_allocation["RPN"].min()) / (df_allocation["RPN"].max() - df_allocation["RPN"].min()))

# Composite Score
w_score, w_sharpe, w_rpn = 0.4, 0.3, 0.3
df_allocation["Composite Score"] = (
    w_score * df_allocation["Score_Norm"] +
    w_sharpe * df_allocation["Sharpe_Norm"] +
    w_rpn * df_allocation["RPN_Norm"]
)

# Final weights
df_allocation["Weight %"] = df_allocation["Composite Score"] / df_allocation["Composite Score"].sum()
df_allocation["Capital Allocated"] = df_allocation["Weight %"] * total_capital

# Display
st.dataframe(
    df_allocation[["Ticker", "Total Score", "Sharpe Ratio", "RPN", "Composite Score", "Weight %", "Capital Allocated"]]
    .sort_values("Composite Score", ascending=False)
    .style.format({"Weight %": "{:.2%}", "Capital Allocated": "₹{:,.0f}"}), 
    use_container_width=True
)
# Merge all metrics
score_df = df_score.merge(pd.DataFrame(rpn_list), on="Ticker")
score_df = score_df.merge(pd.DataFrame(sharpe_ratios), left_on="Ticker", right_on="Stock").drop(columns=["Stock"])
score_df = score_df.merge(sentiment_df[["Ticker", "Avg Sentiment Score"]], on="Ticker", how="left").fillna(0)

# Normalize each score
scaler = MinMaxScaler()
score_df[['Total Score', 'RPN', 'Sharpe Ratio', 'Avg Sentiment Score']] = scaler.fit_transform(
    score_df[['Total Score', 'RPN', 'Sharpe Ratio', 'Avg Sentiment Score']]
)

# Invert RPN because lower is better
score_df['RPN'] = 1 - score_df['RPN']

# Composite Final Score (weighted)
score_df['Final Score'] = (
    0.4 * score_df['Total Score'] + 
    0.2 * score_df['Sharpe Ratio'] +
    0.2 * score_df['Avg Sentiment Score'] +
    0.2 * score_df['RPN']
)

score_df = score_df.sort_values("Final Score", ascending=False)

# Capital Allocation (e.g., ₹10L)
total_capital = st.sidebar.number_input("Enter Total Capital to Allocate (₹)", value=10_00_000)
score_df["Allocation %"] = (score_df['Final Score'] / score_df['Final Score'].sum() * 100).round(2)
score_df["Capital Allocation (₹)"] = ((score_df["Allocation %"] / 100) * total_capital).round(0)

st.subheader("💰 Final Allocation Matrix")
st.dataframe(score_df[[
    "Ticker", "Total Score", "Sharpe Ratio", "Avg Sentiment Score", "RPN", 
    "Final Score", "Allocation %", "Capital Allocation (₹)"
]], use_container_width=True)

# Optional Pie Chart
fig = px.pie(df_allocation, names="Ticker", values="Capital Allocated", title="💹 Capital Allocation")
st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import quote_plus
from textblob import TextBlob
import pandas_ta as ta

# -----------------------------
# Functions
# -----------------------------
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

    fig, ax = plt.subplots()
    ax.plot(data['Close'].values, label='Close Price', color='black')
    for level, (value, color) in fib_levels.items():
        ax.axhline(value, linestyle='--', label=level, color=color)
        ax.text(len(data)*0.99, value, level, va='center', ha='right', fontsize=8, color=color)
    ax.set_title(f"{ticker} - Fibonacci Retracement")
    ax.legend()
    return fig

def plot_fibonacci_fan(data, ticker):
    high = float(data['High'].max())
    low = float(data['Low'].min())

    fig, ax = plt.subplots()
    ax.plot(data['Close'].values, label='Close Price', color='black')
    base_x = [0, len(data)-1]
    for ratio, color in zip([0.382, 0.5, 0.618], ['blue', 'green', 'red']):
        slope = (high - low) * ratio / len(data)
        fan_y = [low, low + slope * (len(data)-1)]
        ax.plot(base_x, fan_y, linestyle='--', label=f'Fan {ratio}', color=color)
    ax.set_title(f"{ticker} - Fibonacci Fan")
    ax.legend()
    return fig

def plot_fibonacci_projection(data, ticker):
    high = float(data['High'].max())
    low = float(data['Low'].min())
    diff = high - low
    latest_close = float(data['Close'].iloc[-1])
    next_points = [latest_close + diff * r for r in [0.382, 0.618, 1.0]]

    fig, ax = plt.subplots()
    ax.plot(data['Close'].values, label='Close Price', color='black')
    for r, value in zip([0.382, 0.618, 1.0], next_points):
        ax.axhline(value, linestyle=':', label=f"Projection {r}", color='gray')
        ax.text(len(data)*0.99, value, f"Proj {r}", va='center', ha='right', fontsize=8, color='gray')
    ax.set_title(f"{ticker} - Fibonacci Projection")
    ax.legend()
    return fig

def calculate_sentiment_score(ticker):
    query = quote_plus(ticker.replace(".NS", "") + " stock news")
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    sentiments = []
    for entry in feed.entries[:5]:
        blob = TextBlob(entry.title + ". " + entry.get('summary', ''))
        sentiments.append(blob.sentiment.polarity)
    if sentiments:
        return round((sum(sentiments)/len(sentiments) + 1) * 2.5, 2)
    return 2.5

def plot_elliott_wave(price, ticker):
    if len(price) >= 35:
        labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5', 'Wave A', 'Wave B', 'Wave C']
        slices = [(35, 30), (30, 25), (25, 20), (20, 15), (15, 10), (10, 7), (7, 4), (4, 1)]
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'brown']

        plt.figure(figsize=(10, 4))
        plt.plot(price.values, label='Close Price', color='gray', linewidth=1)

        for ((start, end), label, color) in zip(slices, labels, colors):
            seg = price.iloc[-start:-end]
            idxs = range(len(price)-start, len(price)-end)
            plt.plot(idxs, seg, label=label, color=color, linewidth=2)
            plt.text(list(idxs)[0], seg.iloc[0], label, fontsize=8, color=color)

        plt.title(f"Elliott Waves (1–5 + A–C) - {ticker}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return 1
    return 0

def plot_gartley(price, ticker):
    if len(price) > 10:
        x = price.iloc[-10].item()
        a = price.iloc[-8].item()
        b = price.iloc[-6].item()
        c = price.iloc[-4].item()
        d = price.iloc[-2].item()
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
            plt.figure(figsize=(10, 4))
            plt.plot(price.values, label='Close Price', color='black')
            points = [x, a, b, c, d]
            indices = [len(price)-10, len(price)-8, len(price)-6, len(price)-4, len(price)-2]
            labels = ['X', 'A', 'B', 'C', 'D']
            plt.scatter(indices, points, color='purple')
            for idx, val, label in zip(indices, points, labels):
                plt.text(idx, val, label, fontsize=8, color='purple')
            plt.title(f"Gartley Pattern - {ticker} ({'Detected' if is_gartley else 'Not Detected'})")
            plt.legend()
            plt.show()
        return 1 if is_gartley else 0


def calculate_technical_score(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    if data.empty or 'Close' not in data:
        return 0.0

    data['RSI'] = ta.rsi(data['Close'])
    macd_df = ta.macd(data['Close'])
    try:
        ichimoku_tuple = ta.ichimoku(data['High'], data['Low'], data['Close'])
        ichimoku_df = pd.concat([df for df in ichimoku_tuple if isinstance(df, pd.DataFrame)], axis=1)
    except Exception:
        ichimoku_df = None
    aroon_df = ta.aroon(high=data['High'], low=data['Low'])

    if macd_df is not None:
        data = pd.concat([data, macd_df], axis=1)
    if ichimoku_df is not None:
        data = pd.concat([data, ichimoku_df], axis=1)
    if aroon_df is not None:
        data = pd.concat([data, aroon_df], axis=1)

    price = data['Close'].reset_index(drop=True)
    elliott_signal = plot_elliott_wave(price, ticker)
    gartley_signal = plot_gartley(price, ticker)

    score = 0.0
    rsi_val = data['RSI'].dropna().iloc[-1] if 'RSI' in data and not data['RSI'].isna().all() else None
    macd_val = data['MACD_12_26_9'].dropna().iloc[-1] if 'MACD_12_26_9' in data and not data['MACD_12_26_9'].dropna().empty else None
    ichimoku_val = data['ITS_9'].dropna().iloc[-1] if 'ITS_9' in data and not data['ITS_9'].dropna().empty else None
    aroon_up = data['AROONU_14'].dropna().iloc[-1] if 'AROONU_14' in data and not data['AROONU_14'].dropna().empty else None

    if rsi_val and 30 < rsi_val < 70:
        score += 1.0
    if macd_val and macd_val > 0:
        score += 1.0
    if ichimoku_val and ichimoku_val > data['Close'].iloc[-1]:
        score += 0.5
    if aroon_up and aroon_up > 70:
        score += 0.5
    if elliott_signal:
        score += 0.5
    if gartley_signal:
        score += 0.5

    return round(score, 2)

def calculate_fundamental_score(fundamentals):
    score = 0
    if fundamentals['revenue_growth'] > 15: score += 1
    if fundamentals['profit_growth'] > 15: score += 1
    if fundamentals['debt_equity'] < 0.5: score += 1
    if fundamentals['roce'] > 15: score += 1
    if fundamentals['promoter_holding'] > 50: score += 1
    return round((score / 5) * 5, 1)

# -----------------------------
# UI
# -----------------------------
st.title("📈 Multi-Ticker Technical & Sentiment Analyzer")

tickers = user_tickers
tickers = [t.strip().upper() for t in tickers if t.strip()]
st.info("Each ticker shows: Retracement | Fan | Projection plots")

sample_fundamentals = {
    'revenue_growth': 22,
    'profit_growth': 18,
    'debt_equity': 0.3,
    'roce': 20,
    'promoter_holding': 65
}

for ticker in tickers:
    st.markdown(f"### 🔍 {ticker}")
    data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

    if data.empty:
        st.error(f"No data for {ticker}")
        continue

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(plot_fibonacci_retracement(data, ticker))
    with col2:
        st.pyplot(plot_fibonacci_fan(data, ticker))
    with col3:
        st.pyplot(plot_fibonacci_projection(data, ticker))

    sentiment = calculate_sentiment_score(ticker)
    tech_score = calculate_technical_score(ticker)
    fund_score = calculate_fundamental_score(sample_fundamentals)

    total = sentiment + tech_score + fund_score
    decision = "Buy" if total >= 8.0 else "Hold" if total >= 5 else "Sell"

    st.metric(label="📊 Total Score", value=round(total, 2), delta=decision)
    st.write(f"- **Fundamental Score**: {fund_score}/5")
    st.write(f"- **Technical Score**: {tech_score}/2")
    st.write(f"- **Sentiment Score**: {sentiment}/5")
    st.markdown("---")

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

st.title("📈 Multi-Ticker Manual Elliott Wave Drawer")

# --- User enters tickers ---
tickers = tickers

# --- Session state for each ticker's Elliott points ---
if "elliott_points" not in st.session_state or not isinstance(st.session_state.elliott_points, dict):
    st.session_state.elliott_points = {}

# Ensure storage exists for each ticker
for t in tickers:
    if t not in st.session_state.elliott_points:
        st.session_state.elliott_points[t] = []

# --- Layout tickers in 3-column rows ---
ticker_rows = [tickers[i:i+3] for i in range(0, len(tickers), 3)]

for row in ticker_rows:
    cols = st.columns(len(row))
    for i, ticker in enumerate(row):
        with cols[i]:
            st.subheader(f"📊 {ticker}")

            # Reset button for this ticker
            if st.button(f"🔄 Reset {ticker}"):
                st.session_state.elliott_points[ticker] = []
                st.rerun()

            # 📌 Expander for drawing points
            with st.expander(f"✍️ Draw Elliott Wave on {ticker}", expanded=False):
                # Download data
                data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
                if data.empty:
                    st.error("⚠️ No data found")
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data.reset_index(inplace=True)
                data[['Open','High','Low','Close']] = data[['Open','High','Low','Close']].astype(float)

                # Base candlestick
                base_fig = go.Figure(data=[go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Candlestick"
                )])

                base_fig.update_layout(
                    title=f"{ticker} (Click to Select Points)",
                    xaxis_rangeslider_visible=False,
                    yaxis_title="Price",
                    height=500
                )

                # Capture clicks
                new_points = plotly_events(
                    base_fig,
                    click_event=True,
                    hover_event=False,
                    override_height=500,
                    override_width="100%"
                )

                # Save clicks
                if new_points:
                    st.session_state.elliott_points[ticker].extend(new_points)

            # 📌 Show final wave plot outside expander
            if st.session_state.elliott_points[ticker]:
                # Re-download data (needed for mapping indices)
                data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
                data.reset_index(inplace=True)

                xs, ys = [], []
                for pt in st.session_state.elliott_points[ticker]:
                    if "pointNumber" in pt:
                        idx = pt["pointNumber"]
                        if idx < len(data):
                            xs.append(data["Date"].iloc[idx])
                            ys.append(data["Close"].iloc[idx])

                if len(xs) >= 2:
                    elliott_labels = ['0','1','2','3','4','5','A','B','C']

                    fig = go.Figure(data=[go.Candlestick(
                        x=data['Date'],
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Candlestick"
                    )])

                    for x, y, label in zip(xs, ys, elliott_labels[:len(xs)]):
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            marker=dict(size=10, color="blue" if label.isnumeric() else "red"),
                            text=[label],
                            textposition="top center",
                            name=label
                        ))

                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="blue", dash="dot"),
                        name="Elliott Wave"
                    ))

                    fig.update_layout(
                        title=f"{ticker} Elliott Wave Progress ({len(xs)} points)",
                        xaxis_rangeslider_visible=False,
                        yaxis_title="Price",
                        height=500
                    )

                    if len(xs) == 8:
                        st.success(f"✅ {ticker}: Elliott Wave completed (1–5, A–C)")

                    st.plotly_chart(fig, use_container_width=True)

            st.write("📍 Selected Points:", st.session_state.elliott_points[ticker])

# =============================
# ✅ Summary
# =============================
st.success("Dashboard loaded successfully! Use scores + news + forecasts to inform your investment view.")
