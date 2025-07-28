# gemini
import os

# Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import os
import glob
import time
import talib # Make sure talib is imported for RSI

def format_time(seconds):
    """Convert seconds to MM:SS format if over 60 seconds"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} minutes and {remaining_seconds:.2f} seconds"

# Find the highest existing run number
existing_runs = glob.glob("lstm_predictions_run_*.csv")
if existing_runs:
    run_numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_runs]
    run_number = max(run_numbers) + 1
else:
    run_number = 1  # Start from 1 if no files exist

# Start timer
start_time = time.time()

# Load and preprocess data
df = pd.read_csv("forex_data.csv", delimiter=';')

# --- Correctly handle Date and Time columns ---
# Combine 'Date' and 'Time' into a single 'DateTime' column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
# Set 'DateTime' as the DataFrame index
df = df.set_index('DateTime')
# Drop the original 'Date' and 'Time' columns as they are now in the index
df = df.drop(columns=['Date', 'Time'])

# Ensure relevant columns are numeric, coercing errors
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Moving Averages ---
# Simple Moving Averages (SMA)
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Exponential Moving Averages (EMA) - more responsive
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# --- Relative Strength Index (RSI) ---
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

# Drop rows with NaN values created by moving averages and RSI calculation
# This also handles any NaNs introduced by pd.to_numeric(errors='coerce')
df_cleaned = df.dropna().copy() # .copy() to avoid SettingWithCopyWarning

# Select features for scaling and prediction
# These column names are now confirmed to be present in df_cleaned
features = ['Close', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI']
df_to_scale = df_cleaned[features]

scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df_to_scale)

# Create sequences for training
def create_sequences(data, seq_length, target_feature_index):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, target_feature_index]) # Predict only the 'Close' price
    return np.array(sequences), np.array(labels)

seq_length = 50  # Using past 50 days to predict the next day
target_feature_index = features.index('Close') # Get the index of 'Close' within your features list
X, y = create_sequences(df_scaled, seq_length, target_feature_index)

# Reshape y to be 2D for the scaler.inverse_transform later
y = y.reshape(-1, 1)

# Build and train the model
num_features = df_scaled.shape[1] # Number of features in your scaled data

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)), # Adjusted input_shape
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Generate predictions
predictions = model.predict(X)
# Inverse transform the predictions. Note: scaler was fitted on multiple features.
# To inverse transform a single column (like 'Close'), we need to create a dummy array.
# First, create an array of zeros with the correct number of features for one timestamp.
# Then, insert predictions into the 'Close' column's position.
dummy_array_for_inverse = np.zeros((predictions.shape[0], num_features))
dummy_array_for_inverse[:, target_feature_index] = predictions.flatten()
predictions_original_scale = scaler.inverse_transform(dummy_array_for_inverse)[:, target_feature_index]

# Get the actual 'Close' prices corresponding to the predictions
# These are from df_cleaned, shifted by seq_length
actual_prices_for_predictions = df_cleaned['Close'].values[seq_length : len(predictions_original_scale) + seq_length]

# Create DataFrame for predictions
df_predictions = pd.DataFrame({
    'DateTime': df_cleaned.index[seq_length : len(predictions_original_scale) + seq_length],
    'Actual': actual_prices_for_predictions,
    'Predicted': predictions_original_scale
})


# --- New Code for Trading Signals ---

# Define the trading strategy function
def generate_signals(df_pred, threshold=0.0001):
    signals = []
    for i in range(len(df_pred)):
        # Compare predicted price with the actual price for the same time point
        if df_pred['Predicted'].iloc[i] > df_pred['Actual'].iloc[i] * (1 + threshold):
            signals.append('Buy')
        elif df_pred['Predicted'].iloc[i] < df_pred['Actual'].iloc[i] * (1 - threshold):
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

# Generate signals with a small threshold (e.g., 0.05% price movement for a signal)
trading_signals = generate_signals(df_predictions, threshold=0.0005)
df_predictions['Signal'] = trading_signals

# Plotting with signals
plt.figure(figsize=(14, 7))
# Use df_predictions['DateTime'] for the x-axis now
plt.plot(df_predictions['DateTime'], df_predictions['Actual'], label="Actual Prices", color='blue')
plt.plot(df_predictions['DateTime'], df_predictions['Predicted'], label="Predicted Prices", color='orange')

# Plot Buy/Sell signals on the chart
buy_points = df_predictions[df_predictions['Signal'] == 'Buy']
sell_points = df_predictions[df_predictions['Signal'] == 'Sell']

plt.scatter(buy_points['DateTime'], buy_points['Actual'], marker='^', color='green', s=100, label='Buy Signal', alpha=0.7)
plt.scatter(sell_points['DateTime'], sell_points['Actual'], marker='v', color='red', s=100, label='Sell Signal', alpha=0.7)

plt.title('Forex Prices with LSTM Predictions and Trading Signals')
plt.xlabel('Date Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Save predictions and signals to CSV
predictions_filename = f"lstm_predictions_run_{run_number}_with_signals.csv" # Updated filename
df_predictions.to_csv(predictions_filename, index=False)

print(f"LSTM predictions and trading signals saved to '{predictions_filename}'")

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in: {format_time(elapsed_time)}")
