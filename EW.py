import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Elliott Wave Dashboard", layout="wide")
st.title("📈 Elliott Wave Dashboard with Multi-Cycle Support & Validation")

# ---------------------- Session State ----------------------
if "cycles" not in st.session_state:
    st.session_state.cycles = {}

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    cycle_color = st.color_picker("Cycle Color", "#FF0000")
    stroke_width = st.slider("Line Width", 1, 5, 2)
    
    enable_fib = st.checkbox("Show Fibonacci Retracement", True)
    project_future = st.checkbox("Enable Future Projection", True)
    
    wave_labels = ['0','1','2','3','4','5','A','B','C']

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    for col in ['Open','High','Low','Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open','High','Low','Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

data = load_data(ticker)
if data.empty:
    st.error("No data found for this ticker")
    st.stop()
max_index = len(data)-1

# ---------------------- Sidebar: Historical Wave Points ----------------------
st.sidebar.subheader("Wave Points (Manual Entry)")
points, dates, prices = [], [], []
for label in wave_labels:
    cols = st.sidebar.columns([1,1])
    with cols[0]:
        idx = st.number_input(f"{label} idx", 0, max_index, max_index, step=1, key=f"idx_{label}")
    with cols[1]:
        price_default = float(data['Close'].iloc[idx])
        price = st.number_input(f"{label} price", value=price_default, step=0.1, key=f"price_{label}")
    points.append(idx)
    dates.append(data['Date'].iloc[idx])
    prices.append(price)

# ---------------------- Sidebar: Future Projection ----------------------
future_dates, future_prices, future_labels = [], [], []
if project_future:
    st.sidebar.subheader("Future Projection Points")
    num_future = st.sidebar.number_input("Number of projection points", 1, 9, 3)
    for i in range(1, num_future+1):
        cols = st.sidebar.columns([2,2])
        with cols[0]:
            proj_date = st.date_input(f"P{i} Date", pd.Timestamp.today(), key=f"proj_date_{i}")
        with cols[1]:
            proj_price = st.number_input(f"P{i} Price", 0.0, step=0.1, key=f"proj_price_{i}")
        future_dates.append(pd.Timestamp(proj_date))
        future_prices.append(proj_price)
        future_labels.append(f"P{i}")

all_dates = dates + future_dates
all_prices = prices + future_prices
all_labels = wave_labels + future_labels

# ---------------------- Elliott Wave Validation ----------------------
def validate_wave(prices):
    results = {}
    suggestions = []

    if len(prices) < 6:
        results['error'] = "Not enough points to validate Waves 0-5."
        return results, suggestions

    wave_lengths = [
        abs(prices[1] - prices[0]),  # Wave1
        abs(prices[2] - prices[1]),  # Wave2
        abs(prices[3] - prices[2]),  # Wave3
        abs(prices[4] - prices[3]),  # Wave4
        abs(prices[5] - prices[4])   # Wave5
    ]

    # Wave 2 retracement check
    retrace_w2 = abs(prices[2] - prices[1])
    wave1_len = wave_lengths[0]
    if retrace_w2 > wave1_len:
        results['Wave2'] = f"❌ Wave 2 retraces more than Wave 1 ({retrace_w2:.2f} > {wave1_len:.2f})"
        suggestions.append(f"Adjust Wave 2 price between {prices[1]-wave1_len:.2f} and {prices[1]+wave1_len:.2f}")
    else:
        results['Wave2'] = f"✅ Wave 2 valid retracement ({retrace_w2:.2f} <= {wave1_len:.2f})"

    # Wave 3 cannot be shortest
    wave3_len = wave_lengths[2]
    if wave3_len < min(wave_lengths[0], wave_lengths[4]):  # Compare with Wave1 and Wave5
        results['Wave3'] = f"❌ Wave 3 is shortest (Wave3={wave3_len:.2f}, Wave1={wave_lengths[0]:.2f}, Wave5={wave_lengths[4]:.2f})"
        suggestions.append(f"Increase Wave 3 length above {min(wave_lengths[0], wave_lengths[4]):.2f}")
    else:
        results['Wave3'] = f"✅ Wave 3 length valid"

    # Wave 4 cannot enter Wave 1 territory
    wave4_price = prices[4]
    wave1_start, wave1_end = prices[0], prices[1]
    if wave1_start < wave1_end:  # Uptrend
        if wave4_price <= wave1_end:
            results['Wave4'] = f"❌ Wave 4 enters Wave 1 territory ({wave4_price:.2f} <= {wave1_end:.2f})"
            suggestions.append(f"Adjust Wave 4 above {wave1_end:.2f}")
        else:
            results['Wave4'] = "✅ Wave 4 valid"
    else:  # Downtrend
        if wave4_price >= wave1_end:
            results['Wave4'] = f"❌ Wave 4 enters Wave 1 territory ({wave4_price:.2f} >= {wave1_end:.2f})"
            suggestions.append(f"Adjust Wave 4 below {wave1_end:.2f}")
        else:
            results['Wave4'] = "✅ Wave 4 valid"

    return results, suggestions

validation_results, validation_suggestions = validate_wave(prices[:6])

# ---------------------- Save / Delete Cycles ----------------------
st.sidebar.subheader("Save / Manage Cycles")
if st.sidebar.button("💾 Save Current Cycle"):
    cycle_data = {"dates": all_dates.copy(), "prices": all_prices.copy(), "labels": all_labels.copy(), "color": cycle_color}
    if ticker not in st.session_state.cycles:
        st.session_state.cycles[ticker] = []
    st.session_state.cycles[ticker].append(cycle_data)
    st.sidebar.success(f"Cycle saved for {ticker}! Total: {len(st.session_state.cycles[ticker])}")

saved_cycles = st.session_state.cycles.get(ticker, [])
if saved_cycles:
    selected_idx = st.sidebar.multiselect(
        "Select Cycles to Display",
        options=list(range(len(saved_cycles))),
        format_func=lambda i: f"Cycle {i+1}",
        default=list(range(len(saved_cycles)))
    )
else:
    selected_idx = []

# ---------------------- Base Chart ----------------------
fig = go.Figure(data=[go.Candlestick(
    x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name=f"{ticker} Candlestick"
)])

# Plot current cycle
fig.add_trace(go.Scatter(
    x=all_dates, y=all_prices, mode="lines+markers+text",
    text=all_labels, textposition="top center",
    line=dict(color=cycle_color, width=stroke_width, dash="dot"),
    marker=dict(size=8, color=cycle_color),
    name="Current Wave"
))

# Overlay saved cycles
colors = ["green", "orange", "purple", "brown", "pink", "cyan"]
for i in selected_idx:
    cycle = saved_cycles[i]
    fig.add_trace(go.Scatter(
        x=cycle["dates"], y=cycle["prices"], text=cycle["labels"], textposition="top center",
        mode="lines+markers+text",
        line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
        marker=dict(size=8, color=colors[i % len(colors)]),
        name=f"Cycle {i+1}"
    ))

# ---------------------- Layout ----------------------
fig.update_layout(height=650, xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)

# ---------------------- Tabs ----------------------
tab1, tab2, tab3 = st.tabs(["📈 Chart", "💾 Saved Cycles", "❓ Help / Guide"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)
    # Show validation
    st.subheader("🔍 Elliott Wave Validation")
    for k, v in validation_results.items():
        if "❌" in v:
            st.error(v)
        else:
            st.success(v)
    if validation_suggestions:
        st.subheader("💡 Suggestions to fix invalid waves:")
        for s in validation_suggestions:
            st.write(f"- {s}")

with tab2:
    st.subheader(f"Saved Elliott Wave Cycles for {ticker}")
    if not saved_cycles:
        st.info("No cycles saved yet.")
    else:
        for i, cycle in enumerate(saved_cycles, 1):
            st.write(f"Cycle {i}: Labels {cycle['labels']} | Dates {cycle['dates']} | Prices {cycle['prices']} | Color: {cycle['color']}")
            delete_key = f"delete_{i}"
            if st.button(f"🗑️ Delete Cycle {i}", key=delete_key):
                st.session_state.cycles[ticker].pop(i-1)
                st.experimental_rerun()

with tab3:
    st.title("❓ Elliott Wave Help / Guide")
    st.write("""
    - Elliott Waves are used to predict price trends and reversals.
    - Waves 1-5: Impulse, Waves A-C: Corrective.
    - Rules: Wave 2 never retraces 100% of Wave 1, Wave 3 cannot be shortest, Wave 4 cannot overlap Wave 1.
    - Use multiple cycles to visualize historical patterns and future projections.
    """)
    sample_dates = pd.date_range("2024-01-01", periods=9)
    sample_prices = [100,105,102,108,103,107,101,106,104]
    fig_help = go.Figure()
    fig_help.add_trace(go.Candlestick(
        x=sample_dates, open=[99,104,101,107,102,106,100,105,103],
        high=[101,106,103,109,104,108,102,107,105],
        low=[98,103,100,106,101,105,99,104,102],
        close=sample_prices, name="Sample Candlestick"
    ))
    fig_help.add_trace(go.Scatter(
        x=sample_dates, y=sample_prices, text=['0','1','2','3','4','5','A','B','C'],
        mode="lines+markers+text", marker=dict(size=8,color="red")
    ))
    st.plotly_chart(fig_help, use_container_width=True)
    st.markdown("[📘 Learn more about Elliott Waves](https://www.investopedia.com/terms/e/elliottwave.asp)")
