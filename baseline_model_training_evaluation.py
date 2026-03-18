import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def split_test_train(features, ticker_df):
    split_idx = int(len(ticker_df) * 0.80)

    train_df = ticker_df.iloc[:split_idx].copy()
    test_df = ticker_df.iloc[split_idx:].copy()

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df['Log_Target_Next_Return']
    y_test = test_df['Log_Target_Next_Return']
    
    return X_train, X_test, y_train, y_test


def train_lr(X_train, X_test, y_train, y_test):
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    print("--- Linear Regression Baseline ---")
    print(f"MAE:  {mean_absolute_error(y_test, lr_preds):.6f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.6f}")
    
    return lr_model, lr_preds


def train_xgb(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))



def train_rf(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("MAE:", mae)
    print("R2:", r2)