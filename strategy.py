import pandas_ta_classic as ta

basicStrategy = ta.Strategy(name="Basic Strategy",
                            ta = [
                                {'kind' : 'ema', 'length' : 50},
                                {'kind' : 'ema', 'length' : 20},
                                {'kind' : 'macd', 'slow' : 21, 'fast': 8},
                                {'kind' : 'rsi', 'length' : 14}
                                ])

def apply_strategy(ticker_df, strategy_name=basicStrategy):
    ticker_df = ticker_df.copy()
    
    ticker_df.ta.strategy(strategy_name, cores=1) 
    
    return ticker_df

    