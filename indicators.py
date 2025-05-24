import pandas as pd

def calculate_indicators(df):
    result = df.copy()

    # RSI-14
    delta = result['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    result['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = result['close'].ewm(span=12, adjust=False).mean()
    ema26 = result['close'].ewm(span=26, adjust=False).mean()
    result['MACD'] = ema12 - ema26
    result['Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = result['close'].rolling(20).mean()
    stddev = result['close'].rolling(20).std()
    result['Upper Band'] = sma20 + (2 * stddev)
    result['Lower Band'] = sma20 - (2 * stddev)

    # EMA-20
    result['EMA20'] = result['close'].ewm(span=20, adjust=False).mean()

    return result.dropna()

def get_advice(df):
    latest = df.iloc[-1]
    advice = []

    # RSI
    if latest['RSI'] < 30:
        advice.append("RSI indicates **BUY** (oversold).")
    elif latest['RSI'] > 70:
        advice.append("RSI indicates **SELL** (overbought).")
    else:
        advice.append("RSI is neutral.")

    # MACD
    if latest['MACD'] > latest['Signal']:
        advice.append("MACD indicates **BUY** (bullish crossover).")
    else:
        advice.append("MACD indicates **SELL** (bearish crossover).")

    # Bollinger Bands
    if latest['close'] < latest['Lower Band']:
        advice.append("Price is below lower Bollinger Band → **BUY**.")
    elif latest['close'] > latest['Upper Band']:
        advice.append("Price is above upper Bollinger Band → **SELL**.")
    else:
        advice.append("Price is within Bollinger Bands.")

    # EMA
    if latest['close'] > latest['EMA20']:
        advice.append("Price above EMA20 → **BUY**.")
    else:
        advice.append("Price below EMA20 → **SELL**.")

    return advice
