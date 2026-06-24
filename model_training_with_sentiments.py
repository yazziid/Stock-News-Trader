import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_raw_arrays(merged_df, multimodal_features):
    merged_df.dropna(subset=multimodal_features + ['Log_Target_Next_Return'], inplace=True)
    raw_features = merged_df[multimodal_features].values
    raw_target = merged_df['Log_Target_Next_Return'].values
    return raw_features, raw_target


def feature_scaling(raw_features):
    split_idx_raw = int(len(raw_features) * 0.80)
    scaler = StandardScaler()
    scaler.fit(raw_features[:split_idx_raw])
    scaled_features = scaler.transform(raw_features)
    
    return scaled_features, scaler 

def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length])
        
    return np.array(X), np.array(y)


def Chronological_Train_Test_Split(X_seq, y_seq):
    split_idx = int(len(X_seq) * 0.80)
    X_train_np, X_test_np = X_seq[:split_idx], X_seq[split_idx:]
    y_train_np, y_test_np = y_seq[:split_idx], y_seq[split_idx:]

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    num_features = X_train_t.shape[2] 
    
    return train_loader, num_features, X_test_t, y_test_t


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(StockLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True 
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        last_time_step_out = lstm_out[:, -1, :]
        
        x = self.dropout(last_time_step_out)
        x = self.relu(self.fc1(x))
        predictions = self.fc2(x)
        return predictions

def training_LSTM(train_loader, model, epochs = 20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Training PyTorch LSTM...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}')

def model_evaluation(model, X_test_t, y_test_t):
    model.eval()
    with torch.no_grad():
        lstm_preds_t = model(X_test_t)
        lstm_preds_np = lstm_preds_t.cpu().numpy()
        y_test_actual = y_test_t.cpu().numpy()

    mae = mean_absolute_error(y_test_actual, lstm_preds_np)
    rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_preds_np))

    print("\n--- PyTorch Multimodal LSTM Performance ---")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    return {"MAE": mae, "RMSE": rmse}