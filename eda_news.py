import pandas as pd

def get_ticker_news(file, Ticker = "META"):
    meta_news_df = file[file["Ticker"] == Ticker] 
    meta_news_df = meta_news_df.rename(columns={'Publication Date': 'Datetime'})

    meta_news_df['Datetime'] = pd.to_datetime(meta_news_df['Datetime'], utc=True)
    meta_news_df = meta_news_df.drop_duplicates(subset=["Title"])
    meta_news_df = meta_news_df.drop_duplicates(subset=["ID"])
    meta_news_df = meta_news_df.sort_values("Datetime")

    meta_news_df["text"] = (
        meta_news_df["Title"].fillna("") + ". " +
        meta_news_df["Summary"].fillna("")
    )
    
    return meta_news_df

