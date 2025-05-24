import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf

st.set_page_config(layout="wide", page_title="BTC Trading Dashboard")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'indicators' not in st.session_state:
    st.session_state.indicators = None
if 'advice_history' not in st.session_state:
    st.session_state.advice_history = []
if 'last_refresh_minute' not in st.session_state:
    st.session_state.last_refresh_minute = -1  # Force initial load
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'last_update_second' not in st.session_state:
    st.session_state.last_update_second = -1




def fetch_btc_data():
    try:
        # Download 1-minute interval BTC-USD data for the past day
        df = yf.download(tickers='BTC-USD', interval='1m', period='1d', progress=False)

        # Reset index so timestamp becomes a column
        df.reset_index(inplace=True)
        df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)

        # Ensure correct format and column names
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        return df.dropna()
    except Exception as e:
        st.error(f"Failed to fetch BTC-USD data from yfinance: {str(e)}")
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
    df['Upper_Band'] = sma20 + (2 * stddev)
    df['Lower_Band'] = sma20 - (2 * stddev)

    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df.dropna()


def get_advice_with_signals(df):
    if df is None or df.empty:
        return {}, ["No data available for analysis."], 'NEUTRAL'

    latest = df.iloc[-1]
    signals = {}
    advice = []

    # RSI Analysis
    if latest['RSI'] < 30:
        signals['RSI'] = 'BUY'
        advice.append("RSI indicates **BUY** (oversold).")
    elif latest['RSI'] > 70:
        signals['RSI'] = 'SELL'
        advice.append("RSI indicates **SELL** (overbought).")
    else:
        signals['RSI'] = 'NEUTRAL'
        advice.append("RSI is neutral.")

    # MACD Analysis
    if latest['MACD'] > latest['Signal']:
        signals['MACD'] = 'BUY'
        advice.append("MACD indicates **BUY** (bullish crossover).")
    else:
        signals['MACD'] = 'SELL'
        advice.append("MACD indicates **SELL** (bearish crossover).")

    # Bollinger Bands Analysis
    if latest['close'] < latest['Lower_Band']:
        signals['Bollinger'] = 'BUY'
        advice.append("Price below lower Bollinger Band → **BUY**.")
    elif latest['close'] > latest['Upper_Band']:
        signals['Bollinger'] = 'SELL'
        advice.append("Price above upper Bollinger Band → **SELL**.")
    else:
        signals['Bollinger'] = 'NEUTRAL'
        advice.append("Price within Bollinger Bands.")

    # EMA Analysis
    if latest['close'] > latest['EMA20']:
        signals['EMA20'] = 'BUY'
        advice.append("Price above EMA20 → **BUY**.")
    else:
        signals['EMA20'] = 'SELL'
        advice.append("Price below EMA20 → **SELL**.")

    # Calculate overall signal
    buy_count = sum(1 for s in signals.values() if s == 'BUY')
    sell_count = sum(1 for s in signals.values() if s == 'SELL')

    if buy_count > sell_count:
        overall_signal = "BUY"
    elif sell_count > buy_count:
        overall_signal = "SELL"
    else:
        overall_signal = "NEUTRAL"

    return signals, advice, overall_signal


def store_prediction(signals, current_price, timestamp, overall_signal):
    """Store prediction for accuracy tracking"""
    if not signals:
        return

    prediction = {
        'timestamp': timestamp,
        'price': current_price,
        'signals': signals.copy(),
        'overall_signal': overall_signal,
        'future_prices': {},
        'results': {}
    }

    st.session_state.advice_history.append(prediction)

    # Keep only last 50 predictions
    if len(st.session_state.advice_history) > 50:
        st.session_state.advice_history = st.session_state.advice_history[-50:]


def update_prediction_results(current_df):
    """Update past predictions with current price data"""
    if not st.session_state.advice_history or current_df is None:
        return

    current_time = current_df.index[-1]
    current_price = current_df['close'].iloc[-1]

    for prediction in st.session_state.advice_history:
        pred_time = prediction['timestamp']
        time_diff_minutes = (current_time - pred_time).total_seconds() / 60

        # Store prices at 5, 10, 15 minute intervals
        for interval in [5, 10, 15]:
            if interval - 0.5 <= time_diff_minutes <= interval + 0.5:
                if f'{interval}min' not in prediction['future_prices']:
                    prediction['future_prices'][f'{interval}min'] = current_price

                    # Calculate if prediction was correct
                    original_price = prediction['price']
                    price_change_pct = ((current_price - original_price) / original_price) * 100

                    # Check accuracy for each indicator
                    for indicator, signal in prediction['signals'].items():
                        if indicator not in prediction['results']:
                            prediction['results'][indicator] = {}

                        correct = False
                        if signal == 'BUY' and price_change_pct > 0.05:  # Price went up >0.1%
                            correct = True
                        elif signal == 'SELL' and price_change_pct < -0.05:  # Price went down >0.1%
                            correct = True
                        elif signal == 'NEUTRAL' and abs(price_change_pct) <= 0.05:  # Price stayed within ±0.5%
                            correct = True

                        prediction['results'][indicator][f'{interval}min'] = {
                            'correct': correct,
                            'price_change': price_change_pct
                        }

                    # Check overall signal accuracy
                    if 'overall' not in prediction['results']:
                        prediction['results']['overall'] = {}

                    overall_correct = False
                    overall_signal = prediction['overall_signal']
                    if overall_signal == 'BUY' and price_change_pct > 0.1:
                        overall_correct = True
                    elif overall_signal == 'SELL' and price_change_pct < -0.1:
                        overall_correct = True
                    elif overall_signal == 'NEUTRAL' and abs(price_change_pct) <= 0.5:
                        overall_correct = True

                    prediction['results']['overall'][f'{interval}min'] = {
                        'correct': overall_correct,
                        'price_change': price_change_pct
                    }


def calculate_accuracy_stats():
    """Calculate accuracy statistics"""
    if not st.session_state.advice_history:
        return {}

    stats = {}
    indicators = ['RSI', 'MACD', 'Bollinger', 'EMA20', 'overall']

    for indicator in indicators:
        stats[indicator] = {}
        for interval in ['5min', '10min', '15min']:
            correct_count = 0
            total_count = 0

            for prediction in st.session_state.advice_history:
                if indicator in prediction.get('results', {}):
                    if interval in prediction['results'][indicator]:
                        total_count += 1
                        if prediction['results'][indicator][interval]['correct']:
                            correct_count += 1

            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                stats[indicator][interval] = {
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': total_count
                }
            else:
                stats[indicator][interval] = {
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0
                }

    return stats


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


def create_accuracy_chart(stats):
    """Create accuracy visualization"""
    if not stats:
        return None

    data = []
    for indicator in stats:
        display_name = 'Overall' if indicator == 'overall' else indicator
        for interval in stats[indicator]:
            if stats[indicator][interval]['total'] > 0:
                data.append({
                    'Indicator': display_name,
                    'Interval': interval,
                    'Accuracy': stats[indicator][interval]['accuracy'],
                    'Total': stats[indicator][interval]['total']
                })

    if not data:
        return None

    df_acc = pd.DataFrame(data)
    fig = px.bar(df_acc, x='Indicator', y='Accuracy', color='Interval',
                 title='Prediction Accuracy by Indicator and Time Interval',
                 barmode='group', hover_data=['Total'])
    fig.update_layout(height=400)
    return fig


# Get current time
now = datetime.now()
current_minute = now.minute
current_second = now.second

# Only update time display every 5 seconds to reduce flashing
should_update_display = (current_second != st.session_state.last_update_second and current_second % 5 == 0)
if should_update_display:
    st.session_state.last_update_second = current_second

# Display current time with seconds (using container to prevent flashing)
st.title("🪙 BTC Trading Dashboard")

# Create a placeholder for time display
time_placeholder = st.empty()
with time_placeholder.container():
    time_col1, time_col2, time_col3 = st.columns([2, 2, 1])

    with time_col1:
        st.markdown(f"🕐 **Current Time:** {now.strftime('%H:%M:%S')}")

    with time_col2:
        next_refresh = 60 - now.second
        st.markdown(f"⏱ **Next Refresh:** {next_refresh} seconds")

    with time_col3:
        if st.button("🔄 Manual Refresh"):
            with st.spinner("Fetching fresh data..."):
                st.session_state.data = fetch_btc_data()
                st.session_state.indicators = calculate_indicators(st.session_state.data)
                st.session_state.last_refresh_minute = current_minute
                st.rerun()

# Check if we need to refresh (every full minute or first load)
should_refresh = (current_minute != st.session_state.last_refresh_minute) or not st.session_state.initialized

if should_refresh:
    with st.spinner("Updating data..."):
        new_data = fetch_btc_data()
        if new_data is not None:
            st.session_state.data = new_data
            st.session_state.indicators = calculate_indicators(new_data)
            st.session_state.last_refresh_minute = current_minute
            st.session_state.initialized = True

            # Store new prediction only if we have valid data
            if st.session_state.indicators is not None:
                current_price = st.session_state.indicators['close'].iloc[-1]
                current_time = st.session_state.indicators.index[-1]
                signals, _, overall_signal = get_advice_with_signals(st.session_state.indicators)
                store_prediction(signals, current_price, current_time, overall_signal)

# Update existing predictions with current data
if st.session_state.indicators is not None:
    update_prediction_results(st.session_state.indicators)

# Display dashboard content (this section remains static unless data changes)
if st.session_state.indicators is not None and not st.session_state.indicators.empty:
    current_price = st.session_state.indicators['close'].iloc[-1]

    # Price change calculation
    price_change = 0
    if len(st.session_state.indicators) > 1:
        prev_price = st.session_state.indicators['close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100

    # Get current signals and advice
    signals, advice, overall_signal = get_advice_with_signals(st.session_state.indicators)

    # Display current price and overall signal
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Current BTC Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")

    with col2:
        if overall_signal == "BUY":
            signal_display = "🟢 BUY"
        elif overall_signal == "SELL":
            signal_display = "🔴 SELL"
        else:
            signal_display = "🟡 NEUTRAL"
        st.metric("Overall Signal", signal_display, "")

    with col3:
        prediction_count = len(st.session_state.advice_history)
        st.metric("Predictions Tracked", str(prediction_count), "")

    # Display current advice
    st.markdown("### 📌 Current Strategy Advice")
    for line in advice:
        st.markdown(f"- {line}")

    # Display accuracy statistics if we have predictions
    if st.session_state.advice_history:
        accuracy_stats = calculate_accuracy_stats()

        if any(stats for stats in accuracy_stats.values() if any(interval['total'] > 0 for interval in stats.values())):
            st.markdown("### 🎯 Prediction Accuracy Analysis")

            # Create accuracy table
            acc_data = []
            for indicator in ['RSI', 'MACD', 'Bollinger', 'EMA20', 'overall']:
                display_name = 'Overall' if indicator == 'overall' else indicator
                row = {'Indicator': display_name}
                for interval in ['5min', '10min', '15min']:
                    stats = accuracy_stats.get(indicator, {}).get(interval, {'accuracy': 0, 'total': 0})
                    if stats['total'] > 0:
                        row[interval] = f"{stats['accuracy']:.1f}% ({stats['total']})"
                    else:
                        row[interval] = "No data"
                acc_data.append(row)

            acc_df = pd.DataFrame(acc_data)
            st.dataframe(acc_df, use_container_width=True)

            # Accuracy chart
            acc_chart = create_accuracy_chart(accuracy_stats)
            if acc_chart:
                st.plotly_chart(acc_chart, use_container_width=True)

    # Technical indicator charts
    st.markdown("### 📊 Technical Indicators")
    last_100 = st.session_state.indicators.tail(100)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_indicator("RSI-14", "RSI", last_100), use_container_width=True)
        st.plotly_chart(plot_indicator("Bollinger Bands", "close", last_100, ["Upper_Band", "Lower_Band"]),
                        use_container_width=True)

    with col2:
        st.plotly_chart(plot_indicator("MACD vs Signal", "MACD", last_100, ["Signal"]), use_container_width=True)
        st.plotly_chart(plot_indicator("Price vs EMA-20", "close", last_100, ["EMA20"]), use_container_width=True)

    # Historical predictions
    if st.session_state.advice_history:
        with st.expander("📜 Recent Predictions & Results", expanded=False):
            recent_predictions = []
            for pred in reversed(st.session_state.advice_history[-20:]):  # Last 10 predictions
                row = {
                    'Time': pred['timestamp'].strftime('%H:%M'),
                    'Price': f"${pred['price']:.2f}",
                    'Overall': pred['overall_signal'],
                    'RSI': pred['signals'].get('RSI', '-'),
                    'MACD': pred['signals'].get('MACD', '-'),
                    'Bollinger': pred['signals'].get('Bollinger', '-'),
                    'EMA20': pred['signals'].get('EMA20', '-')
                }

                # Add results for each time interval
                for interval in ['5min', '10min', '15min']:
                    if interval in pred.get('future_prices', {}):
                        price_change = 0
                        if 'overall' in pred.get('results', {}) and interval in pred['results']['overall']:
                            price_change = pred['results']['overall'][interval]['price_change']
                        row[f'After {interval}'] = f"{price_change:+.2f}%"
                    else:
                        row[f'After {interval}'] = "Pending"

                recent_predictions.append(row)

            if recent_predictions:
                pred_df = pd.DataFrame(recent_predictions)
                st.dataframe(pred_df, use_container_width=True)

    # Raw data (single instance only)
    with st.expander("📊 Raw Data (Last 10 rows)", expanded=False):
        display_cols = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'EMA20']
        available_cols = [col for col in display_cols if col in st.session_state.indicators.columns]
        st.dataframe(st.session_state.indicators[available_cols].tail(20).round(2))

else:
    st.info("Loading BTC data... Please wait.")

# Auto-refresh every 5 seconds instead of 2 to reduce flashing
time.sleep(5)
st.rerun()