import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import fft, ifft
from scipy.optimize import differential_evolution
import requests
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress

# Constants
ALPHA = 0.0072973525693  # Fine structure constant
PI = np.pi
EULER = np.e
PHI = (1 + np.sqrt(5)) / 2

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

symbol = "BTCUSDT"
timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
candle_map = {}

# Function to get candles
def get_candles(symbol, timeframe, limit=1000):
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        return [{
            "time": k[0] / 1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        } for k in klines]
    except BinanceAPIException as e:
        print(f"Error fetching candles for {symbol} at {timeframe}: {e}")
        return []

# Fetch candles for all timeframes
for timeframe in timeframes:
    candle_map[timeframe] = get_candles(symbol, timeframe)

# Helper function to remove NaNs and zeros from arrays
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Optimization function
def optimize_nn_hyperparameters(model, X, y):
    def objective(params):
        lr, batch_size = params
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train the model
        model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)  # Ensure y is 1D
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
        
        for epoch in range(10):  # Number of epochs
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                outputs = outputs.view(-1)  # Flatten the output to match y
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Return the final loss
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            outputs = outputs.view(-1)  # Flatten the output to match y
            loss = criterion(outputs, y_tensor)
        return loss.item()

    bounds = [(1e-5, 1e-1), (16, 128)]  # Learning rate and batch size
    result = differential_evolution(objective, bounds, seed=42)
    return result.x

# Main function
def main():
    # Prepare dataset for ANN
    timeframe = "1h"  # Choose a timeframe
    candles = candle_map[timeframe]
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    
    # Use FFT and IFFT for feature engineering
    signal_fft = fft(close_prices)
    X = np.abs(signal_fft).reshape(-1, 1)  # Example feature: magnitude of FFT
    y = close_prices  # Target: original close prices
    
    # Convert y to a compatible shape for regression
    y = np.full((X.shape[0],), np.mean(y))  # Use mean close price as target
    
    # Print shapes for debugging
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    
    # Build and optimize neural network
    input_dim = X.shape[1]
    model = SimpleNN(input_dim)
    optimized_params = optimize_nn_hyperparameters(model, X, y)
    
    # Print results
    print(f"Optimized parameters: Learning Rate = {optimized_params[0]}, Batch Size = {int(optimized_params[1])}")
    
    # Forecast results using the trained model (example)
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor)
    print(f"Predictions: {predictions.numpy()}")

if __name__ == "__main__":
    main()
