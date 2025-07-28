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

print("[44] Shape of df after loading:", df.shape)
print("[45] First 5 rows of df:\n", df.head())
print("[46] Number of NaNs per column after loading:\n", df.isnull().sum())

# IMPORTANT: Since 'Date' is a number, we will NOT convert it to datetime
# and will NOT set it as the index. We will keep it as a regular column or use default index.

# Ensure 'Close' is treated as a numeric value
df['Close'] = pd.to_numeric(df['Close'])

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
# These NaNs are typically at the beginning of the DataFrame
df_cleaned = df.dropna().copy() # .copy() to avoid SettingWithCopyWarning

# Select features for scaling and prediction
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
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# Get the actual 'Close' prices corresponding to the predictions
# These are from df_cleaned, shifted by seq_length
actual_prices_for_predictions = df_cleaned['Close'].values[seq_length : len(predictions) + seq_length]

# Create DataFrame for predictions
df_predictions = pd.DataFrame({
    # Use the 'Date' column from df_cleaned, adjusted for the sequence offset
    'Date_Index': df_cleaned['Date'].values[seq_length : len(predictions) + seq_length] if 'Date' in df_cleaned.columns else df_cleaned.index[seq_length : len(predictions) + seq_length],
    'Actual': actual_prices_for_predictions,
    'Predicted': predictions.flatten()
})


# --- New Code for Trading Signals ---

# Define the trading strategy function
def generate_signals(df_pred, threshold=0.0001):
    signals = []
    # Iterate through the DataFrame to compare actual and predicted prices
    for i in range(len(df_pred)):
        # To decide for today, we use the model's prediction for today's price
        # compared to the actual price of the previous day, or some other reference.
        # For this version, we will compare the *predicted* price for day `i`
        # with the *actual* price of day `i` (which the model aims to predict).
        # A buy signal if predicted is significantly higher than actual, implying future rise.
        # A sell signal if predicted is significantly lower than actual, implying future fall.

        # Note: In a real trading scenario, you'd typically predict the *next day's* price
        # based on *today's* data, and then compare that prediction to today's actual close
        # to make a decision for tomorrow's trade.
        # Your current setup has X as `data[i:i + seq_length]` and y as `data[i + seq_length]`.
        # This means predictions[i] is the prediction for Actual[i].

        if df_pred['Predicted'].iloc[i] > df_pred['Actual'].iloc[i] * (1 + threshold):
            signals.append('Buy')
        elif df_pred['Predicted'].iloc[i] < df_pred['Actual'].iloc[i] * (1 - threshold):
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

# Generate signals with a small threshold (e.g., 0.05% price movement for a signal)
# You will need to tune this threshold based on your desired sensitivity and
# typical price movements in your chosen Forex pair.
trading_signals = generate_signals(df_predictions, threshold=0.0005)
df_predictions['Signal'] = trading_signals

# Plotting with signals
plt.figure(figsize=(14, 7))
# Use df_predictions['Date_Index'] for the x-axis, which correctly accounts for the offset
plt.plot(df_predictions['Date_Index'], df_predictions['Actual'], label="Actual Prices", color='blue')
plt.plot(df_predictions['Date_Index'], df_predictions['Predicted'], label="Predicted Prices", color='orange')

# Plot Buy/Sell signals on the chart
buy_points = df_predictions[df_predictions['Signal'] == 'Buy']
sell_points = df_predictions[df_predictions['Signal'] == 'Sell']

plt.scatter(buy_points['Date_Index'], buy_points['Actual'], marker='^', color='green', s=100, label='Buy Signal', alpha=0.7)
plt.scatter(sell_points['Date_Index'], sell_points['Actual'], marker='v', color='red', s=100, label='Sell Signal', alpha=0.7)

plt.title('Forex Prices with LSTM Predictions and Trading Signals')
plt.xlabel('Date Index (Numerical)')
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
