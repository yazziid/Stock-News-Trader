import pandas as pd


def match_news_and_price_period(ticker_df, ticker_news_df):
    # Truncate Price Data to match News Data
    earliest_news_date = ticker_news_df['Datetime'].min()
    
    # Keep only the price rows that occur on or after the first news article
    ticker_df = ticker_df[ticker_df['Datetime'] >= earliest_news_date].copy()

    # Sort Chronologically
    ticker_df = ticker_df.sort_values('Datetime').reset_index(drop=True)
    ticker_news_df = ticker_news_df.sort_values('Datetime').reset_index(drop=True)
    return ticker_df, ticker_news_df 
    

def add_sentiment_to_news(ticker_news_df):
    from FT_finBert import get_sentiment

    ticker_news_df["sentiment"] = ticker_news_df["text"].progress_apply(get_sentiment)
    return ticker_news_df

def merge_news_and_price_data(ticker_df, ticker_news_df):
    ticker_df, ticker_news_df = match_news_and_price_period(ticker_df, ticker_news_df)
    ticker_news_df = add_sentiment_to_news(ticker_news_df)
    ticker_pn_df = pd.merge_asof(
        ticker_df,
        ticker_news_df,
        on='Datetime',
        direction='backward'
    )
    
    # Number of 5-min bars per window
    bars_1h = 12   # 12*5min = 1h
    bars_4h = 48   # 48*5min = 4h
    bars_1d = 288  # 288*5min = 24h

    ticker_pn_df['sentiment_1h'] = ticker_pn_df['sentiment'].rolling(window=bars_1h, min_periods=1).mean()
    ticker_pn_df['sentiment_4h'] = ticker_pn_df['sentiment'].rolling(window=bars_4h, min_periods=1).mean()
    ticker_pn_df['sentiment_1d'] = ticker_pn_df['sentiment'].rolling(window=bars_1d, min_periods=1).mean()
    ticker_pn_df.dropna(inplace=True)
    ticker_pn_df = ticker_pn_df.sort_values('Datetime')

    return ticker_pn_df

    
    
    