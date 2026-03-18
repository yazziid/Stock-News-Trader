import pandas as pd
import numpy as np

def fixing_row_problems(df):
    mask = pd.to_numeric(df['Volume'], errors='coerce').isna()
    problem_rows = df[mask]
    good_rows = df[~ mask]

    problem_rows.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume", "ticker"]
    problem_rows = problem_rows[["ticker", "Open", "High", "Low", "Close", "Volume", "Datetime"]]
    combined_df = pd.concat([good_rows, problem_rows], ignore_index=True)

    return combined_df

def get_ticker_data(Ticker, df):
    ticker_df = df[df["ticker"] == Ticker]
    ticker_df = ticker_df.sort_values("Datetime")
    ticker_df = ticker_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    ticker_df["Volume"] = ticker_df["Volume"].astype(int) 
    ticker_df["Open"] = ticker_df["Open"].astype(float) 
    ticker_df['Datetime'] = pd.to_datetime(ticker_df['Datetime'], utc=True)
    
    ticker_df = (ticker_df.sort_values("Datetime").drop_duplicates(subset="Datetime", keep="last"))
    
    return ticker_df


def set_features(ticker_df):
    # --- RETURNS ---
    ticker_df['Return'] = ticker_df['Close'].pct_change()

    # --- FUTURE TARGET ---
    ticker_df['Target_Next_Return'] = ticker_df['Return'].shift(-1)

    # --- ADD FEATURES ---
    ticker_df['EMA_Spread'] = ticker_df['EMA_20'] - ticker_df['EMA_50']
    ticker_df["return_5"] = ticker_df["Close"].pct_change(5)
    ticker_df["volatility_10"] = ticker_df["Return"].rolling(10).std()
    ticker_df["momentum"] = ticker_df["Close"] - ticker_df["Close"].shift(10)
    return ticker_df
    
    
def set_log_data(ticker_df):
    ticker_df = ticker_df.copy()

    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols: ticker_df[f'Log_{col}'] = np.log(ticker_df[col])

    ticker_df['Log_Volume'] = np.log(ticker_df['Volume'] + 1)
    ticker_df['Log_Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
    ticker_df['Log_Target_Next_Return'] = ticker_df['Log_Return'].shift(-1)
    ticker_df['volume_change'] = np.log(ticker_df['Volume'] + 1) - np.log(ticker_df['Volume'].shift(1) + 1)
    ticker_df.dropna(inplace=True)
    
    return ticker_df

def get_data_with_features(Ticker, features = True):
    df = fixing_row_problems(df)    
    ticker_df = get_ticker_data(Ticker, df)

    if features:
        ticker_df = set_features(ticker_df)
        ticker_df = set_log_data(ticker_df)
        
    return ticker_df

if __name__ == '__main__':
    df = pd.read_csv("C:/Users/yazid/Documents/projects/main projects/technical analysis/data/raw/prices/stock_data.csv", low_memory=False)

    df = fixing_row_problems(df)