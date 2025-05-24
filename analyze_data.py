import pandas as pd

def calculate_rsi14(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI14'] = rsi
    return df[['close', 'RSI14']].dropna().tail(30)
