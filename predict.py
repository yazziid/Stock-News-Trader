import os
import torch
import joblib
import pandas as pd

from dotenv import load_dotenv
from eda import get_ticker_data, fixing_row_problems, set_features, set_log_data
from strategy import apply_strategy
from eda_news import get_ticker_news
from news_price import merge_news_and_price_data
from model_training_with_sentiments import StockLSTM
from inference import predict_next_price
from tqdm import tqdm

tqdm.pandas()
load_dotenv()
price_file_path = os.getenv("price_file_path")    
news_path_file = os.getenv("news_path_file")    

Ticker = "META"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

multimodal_features = [
    'Volume', 'volume_change', 'volatility_10', 'Log_Open', 
    'MACD_8_21_9', 'return_5', 'RSI_14', 'EMA_Spread', 'momentum',
    'sentiment', 'sentiment_1h', 'sentiment_4h'
]

def prep_latest_data():
    """
    Recreates the data preprocessing pipeline to fetch the latest sequence for inference.
    In a fully live system, this would pull from live APIs instead of CSVs.
    """
    print("Loading and preprocessing latest market data...")
    
    # Process Prices
    df = pd.read_csv(price_file_path, low_memory=False)
    df = fixing_row_problems(df)
    ticker_df = get_ticker_data(Ticker, df)
    ticker_df = apply_strategy(ticker_df)
    ticker_df = set_features(ticker_df)
    ticker_df = set_log_data(ticker_df)
    
    file_news = pd.read_csv(news_path_file, low_memory=False)
    meta_news_df = get_ticker_news(file_news)
    
    meta_pn_df = merge_news_and_price_data(ticker_df, meta_news_df)
    
    meta_pn_df.dropna(subset=multimodal_features, inplace=True)
    
    return meta_pn_df

if __name__ == "__main__":
    print("--- Initializing Prediction Engine ---")
    
    if not os.path.exists("artifacts/lstm_model_weights.pth") or not os.path.exists("artifacts/feature_scaler.joblib"):
        raise FileNotFoundError("Artifacts not found! Please run train.py first to generate the scaler and model weights.")

    print("Loading feature scaler...")
    scaler = joblib.load("artifacts/feature_scaler.joblib")
    
    print("Loading PyTorch model weights...")
    num_features = len(multimodal_features)
    model = StockLSTM(input_size=num_features, hidden_size=64, num_layers=1, output_size=1).to(device)
    
    model.load_state_dict(torch.load("artifacts/lstm_model_weights.pth"))
    model.eval() 
    
    meta_pn_df = prep_latest_data()
    last_12_bars = meta_pn_df.tail(12)
    
    if len(last_12_bars) < 12:
        raise ValueError("Not enough data to form a 12-bar sequence for inference.")
        
    current_actual_price = last_12_bars['Close'].iloc[-1]
    
    predict_next_price(
        model=model, 
        scaler=scaler, 
        latest_data_df=last_12_bars, 
        multimodal_features=multimodal_features, 
        current_price=current_actual_price,
        device=device
    )