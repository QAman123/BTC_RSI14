import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="BTC Trading Dashboard")

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(seconds=61)  # Force initial load
if 'data' not in st.session_state:
    st.session_state.data = None
if 'indicators' not in st.session_state:
    st.session_state.indicators = None

REFRESH_INTERVAL = 60  # seconds


# Remove TTL cache and make it a regular function
def fetch_btc_data():
    try:
        url = 'https://api.binance.com/api/v3/klines'
        params = {'symbol': 'BTCUSDT', 'interval': '1m', 'limit': 500}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].astype(float)
        return df[['close']]
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def calculate_indicators(df):
    if df is None or df.empty:
        return None

    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma20 = df['close'].rolling(20).mean()
    stddev = df['close'].rolling(20).std()
    df['Upper Band'] = sma20 + (2 * stddev)
    df['Lower Band'] = sma20 - (2 * stddev)

    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df.dropna()


def get_advice(df):
    if df is None or df.empty:
        return ["No data available for analysis."]

    latest = df.iloc[-1]
    advice = []

    if latest['RSI'] < 30:
        advice.append("RSI indicates **BUY** (oversold).")
    elif latest['RSI'] > 70:
        advice.append("RSI indicates **SELL** (overbought).")
    else:
        advice.append("RSI is neutral.")

    if latest['MACD'] > latest['Signal']:
        advice.append("MACD indicates **BUY** (bullish crossover).")
    else:
        advice.append("MACD indicates **SELL** (bearish crossover).")

    if latest['close'] < latest['Lower Band']:
        advice.append("Price below lower Bollinger Band → **BUY**.")
    elif latest['close'] > latest['Upper Band']:
        advice.append("Price above upper Bollinger Band → **SELL**.")
    else:
        advice.append("Price within Bollinger Bands.")

    if latest['close'] > latest['EMA20']:
        advice.append("Price above EMA20 → **BUY**.")
    else:
        advice.append("Price below EMA20 → **SELL**.")

    return advice


def plot_indicator(title, y_col, df, secondary_cols=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[y_col], name=y_col, line=dict(color='blue')))
    if secondary_cols:
        colors = ['red', 'green', 'orange', 'purple']
        for i, col in enumerate(secondary_cols):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color)))

    fig.update_layout(
        title=title,
        height=300,
        margin=dict(t=40, b=10),
        legend=dict(orientation="h"),
        xaxis_title="Time",
        yaxis_title="Value"
    )
    return fig


# Calculate time since last update
time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
countdown = max(0, REFRESH_INTERVAL - int(time_since_update))

# Display header and countdown
st.title("🪙 BTC Trading Dashboard")
st.markdown(f"⏱ Next refresh in: **{countdown} seconds**")

# Add manual refresh button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now() - timedelta(seconds=61)  # Force refresh

# Check if we need to fetch new data
if time_since_update >= REFRESH_INTERVAL:
    with st.spinner("Fetching latest BTC data..."):
        st.session_state.data = fetch_btc_data()
        st.session_state.indicators = calculate_indicators(st.session_state.data)
        st.session_state.last_update = datetime.now()
        st.rerun()

# Display data if available
if st.session_state.indicators is not None and not st.session_state.indicators.empty:
    advice = get_advice(st.session_state.indicators)

    # Current price display
    current_price = st.session_state.indicators['close'].iloc[-1]
    st.metric("Current BTC Price", f"${current_price:,.2f}")

    st.markdown("### 📌 Strategy Advice")
    for line in advice:
        st.markdown(f"- {line}")

    # Get last 100 data points for charts
    last_100 = st.session_state.indicators.tail(100)

    # Display charts in a 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_indicator("RSI-14", "RSI", last_100), use_container_width=True)
        st.plotly_chart(plot_indicator("Bollinger Bands", "close", last_100, ["Upper Band", "Lower Band"]),
                        use_container_width=True)

    with col2:
        st.plotly_chart(plot_indicator("MACD vs Signal", "MACD", last_100, ["Signal"]), use_container_width=True)
        st.plotly_chart(plot_indicator("Price vs EMA-20", "close", last_100, ["EMA20"]), use_container_width=True)

    # Raw data section
    with st.expander("📊 Raw Data (Last 10 rows)", expanded=False):
        st.dataframe(st.session_state.indicators.tail(10))

    # Last update info
    st.markdown(f"*Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}*")

else:
    st.info("Loading BTC data...")
    # Force initial load if no data
    if st.session_state.data is None:
        st.rerun()

# Auto-refresh mechanism using query params (works better in Streamlit Cloud)
if countdown == 0:
    st.rerun()