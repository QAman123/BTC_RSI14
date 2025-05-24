import streamlit as st
from streamlit_autorefresh import st_autorefresh
from fetch_data import fetch_btc_data
from analyze_data import calculate_rsi14
import plotly.graph_objects as go

st.set_page_config(page_title="BTC RSI14", layout="wide")

st.title("📈 BTC/USDT RSI-14 Monitor")

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="rsi_refresh")

with st.spinner("Fetching data and calculating RSI..."):
    csv_path = fetch_btc_data()
    rsi_df = calculate_rsi14(csv_path)

# Plotly chart with dual y-axis
fig = go.Figure()

# Price on y-axis 1
fig.add_trace(go.Scatter(
    x=rsi_df.index,
    y=rsi_df['close'],
    name='BTC Price',
    yaxis='y1',
    line=dict(color='blue')
))

# RSI on y-axis 2
fig.add_trace(go.Scatter(
    x=rsi_df.index,
    y=rsi_df['RSI14'],
    name='RSI-14',
    yaxis='y2',
    line=dict(color='orange')
))

fig.update_layout(
    title="BTC Price and RSI-14 (Last 30 Minutes)",
    xaxis=dict(title='Timestamp'),
    yaxis=dict(
        title='Price (USD)',
        side='left'
    ),
    yaxis2=dict(
        title='RSI-14',
        overlaying='y',
        side='right',
        range=[0, 100]
    ),
    legend=dict(x=0.01, y=0.99),
    height=500
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(rsi_df)
