from eda import get_ticker_data, fixing_row_problems, set_features, set_log_data, get_data_with_features
from strategy import apply_strategy
from baseline_model_training_evaluation import train_lr, split_test_train, train_xgb, train_rf
from eda_news import get_ticker_news
from news_price import merge_news_and_price_data
from model_training_with_sentiments import extract_raw_arrays, feature_scaling, create_sequences, StockLSTM, Chronological_Train_Test_Split, training_LSTM, model_evaluation, device
from inference import predict_next_price

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

Ticker = "META"

features = ['Log_Open', 'Log_High', 'Log_Low', 'Log_Close', 'Volume', 'EMA_50', 'EMA_20', 
            'MACD_8_21_9', 'MACDh_8_21_9', 'MACDs_8_21_9', 'RSI_14', 
            'EMA_Spread', 'volatility_10', 'return_5', 'volume_change', 'momentum']


def data_preprocessing(df, Ticker):
    df = fixing_row_problems(df)
    ticker_df = get_ticker_data(Ticker, df)

    print("Applying strategies")
    ticker_df = apply_strategy(ticker_df)

    print("Adding features and logging numerical data")
    ticker_df = set_features(ticker_df)
    ticker_df = set_log_data(ticker_df)
    
    return ticker_df
    


if __name__ == "__main__":
    from info import price_file_path, news_path_file
    df = pd.read_csv(price_file_path, low_memory=False)
    
    print("Data loaded successfully")
    
    meta_df = data_preprocessing(df, Ticker)

    print("Splitting data into test and train")    
    X_train, X_test, y_train, y_test = split_test_train(features, meta_df)

    print("Starting training")
    train_lr(X_train, X_test, y_train, y_test)
    train_xgb(X_train, X_test, y_train, y_test)
    train_rf(X_train, X_test, y_train, y_test)
    print("Done with baseline")
        
    print("Loading news data")
    file_news = pd.read_csv(news_path_file, low_memory=False)
    meta_news_df = get_ticker_news(file_news)
    meta_pn_df = merge_news_and_price_data(meta_df, meta_news_df)
    
    print("news data loaded")
    multimodal_features = [
    'Volume', 'volume_change', 'volatility_10', 'Log_Open', 
    'MACD_8_21_9',          
    'return_5', 'RSI_14', 'EMA_Spread', 'momentum',
    'sentiment', 'sentiment_1h', 'sentiment_4h'
    ]
    
    print("Starting training")
    seq_length = 12 
    raw_features, raw_target = extract_raw_arrays(meta_pn_df, multimodal_features)
    scaled_features, scaler = feature_scaling(raw_features)
    X_seq, y_seq = create_sequences(scaled_features, raw_target, seq_length)
    train_loader, num_features, X_test_t, y_test_t = Chronological_Train_Test_Split(X_seq, y_seq)
    
    model = StockLSTM(input_size=num_features, hidden_size=64, num_layers=1, output_size=1).to(device)
    training_LSTM(train_loader, model)
    print("training done")
    model_evaluation(model, X_test_t, y_test_t)
    print("Everything run smoothly")
    
    
    last_12_bars = meta_pn_df.tail(12)
    current_actual_price = last_12_bars['Close'].iloc[-1]

    predict_next_price(
        model=model, 
        scaler=scaler, 
        latest_data_df=last_12_bars, 
        multimodal_features=multimodal_features, 
        current_price=current_actual_price,
        device=device
    )