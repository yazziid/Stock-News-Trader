import torch
import numpy as np

def predict_next_price(model, scaler, latest_data_df, multimodal_features, current_price, device):
    model.eval()
    raw_live_features = latest_data_df[multimodal_features].values
    scaled_live_features = scaler.transform(raw_live_features)
    live_sequence = np.array([scaled_live_features])
    live_tensor = torch.tensor(live_sequence, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_log_return_tensor = model(live_tensor)
        predicted_log_return = predicted_log_return_tensor.cpu().numpy()[0][0]
    
    predicted_price_multiplier = np.exp(predicted_log_return)
    predicted_next_close = current_price * predicted_price_multiplier
    
    print("\n--- Live Prediction ---")
    print(f"Current Price:           ${current_price:.2f}")
    print(f"Predicted Log Return:    {predicted_log_return:.6f}")
    print(f"Predicted Next Close:    ${predicted_next_close:.2f}")
    
    if predicted_next_close > current_price:
        print("Model Signal:            BUY / LONG")
    else:
        print("Model Signal:            SELL / SHORT")
        
    return predicted_next_close