# gemini
import os

# Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

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

# Initialize run number (you can increment this manually or automate it)
# run_number = 1  # Change this for each run or automate it (see notes below)

# Start timer
start_time = time.time()


# Load and preprocess data
df = pd.read_csv("forex_data.csv", delimiter=';')
# df = df[['Close']]  # Using closing price for prediction

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
df_cleaned = df.dropna().copy() # .copy() to avoid SettingWithCopyWarning

# Select features for scaling and prediction
features = ['Close', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI']
df_to_scale = df_cleaned[features]


scaler = MinMaxScaler(feature_range=(0,1))
# df_scaled = scaler.fit_transform(df)
df_scaled = scaler.fit_transform(df_to_scale)

# Create sequences for training
def create_sequences(data, seq_length):
    sequences, labels = [], []
    # The labels for prediction will be the 'Close' price, which is the first feature in 'data'
    # Assuming 'Close' is always the first feature in your 'features' list.
    close_price_index = features.index('Close')
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, close_price_index]) # Predict only the 'Close' price
    return np.array(sequences), np.array(labels)

seq_length = 50  # Using past 50 days to predict the next day
X, y = create_sequences(df_scaled, seq_length)

# Reshape y to be 2D for the scaler.inverse_transform later, if necessary
y = y.reshape(-1, 1)

# Update input_shape to reflect the number of features
num_features = df_scaled.shape[1]

# Build and train the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Generate predictions
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# For plotting, use the numerical index of the DataFrame since 'Date' isn't a date
# The actual prices should correspond to the `df_cleaned` DataFrame's 'Close' values
# shifted by seq_length, as that's what the model is trying to predict.
plt.plot(df_cleaned.index[seq_length:len(predictions) + seq_length],
         df_cleaned['Close'][seq_length:len(predictions) + seq_length],
         label="Actual Prices")
plt.plot(df_cleaned.index[seq_length:len(predictions) + seq_length],
         predictions, label="Predicted Prices")
plt.legend()
plt.show()

# Create DataFrame for predictions

# --- New Code for Trading Signals ---

# Define the trading strategy function
def generate_signals(df_pred, threshold=0.0001):
    signals = []
    # Iterate through the DataFrame to compare actual and predicted prices
    for i in range(len(df_pred)):
        # To decide for today, we look at the prediction for today
        # and compare it with yesterday's actual close (which is df_pred['Actual'][i-1]
        # or more simply, we compare df_pred['Predicted'][i] with df_pred['Actual'][i-1] for next day's movement.
        # However, for simplicity and directness based on your model's output
        # where predictions are for the 'Actual' value at that index,
        # we'll compare predicted vs. actual for the *same* day index for signal generation.
        # A more robust strategy would predict tomorrow's close based on today's close.
        # For now, let's assume 'Predicted' is the model's estimate for the 'Actual' at that row's Date.

        # If the predicted price is significantly higher than the actual price (of the same point in time)
        # We consider a potential "Buy" opportunity
        if df_pred['Predicted'].iloc[i] > df_pred['Actual'].iloc[i] * (1 + threshold):
            signals.append('Buy')
        # If the predicted price is significantly lower than the actual price
        # We consider a potential "Sell" opportunity
        elif df_pred['Predicted'].iloc[i] < df_pred['Actual'].iloc[i] * (1 - threshold):
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

# Generate signals with a small threshold (e.g., 0.01% price movement for a signal)
# You will need to tune this threshold based on your desired sensitivity and
# typical price movements in your chosen Forex pair.
trading_signals = generate_signals(df_predictions, threshold=0.0005)
df_predictions['Signal'] = trading_signals

# Plotting with signals (optional - you might want a different visualization for signals)
plt.figure(figsize=(14, 7))
plt.plot(df.index[:len(predictions)], df['Close'][:len(predictions)], label="Actual Prices", color='blue')
plt.plot(df.index[:len(predictions)], predictions, label="Predicted Prices", color='orange')

# Plot Buy/Sell signals on the chart
buy_points = df_predictions[df_predictions['Signal'] == 'Buy']
sell_points = df_predictions[df_predictions['Signal'] == 'Sell']

plt.scatter(buy_points['Date'], buy_points['Actual'], marker='^', color='green', s=100, label='Buy Signal')
plt.scatter(sell_points['Date'], sell_points['Actual'], marker='v', color='red', s=100, label='Sell Signal')

plt.title('Forex Prices with LSTM Predictions and Trading Signals')
plt.xlabel('Date Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Save predictions and signals to CSV
df_predictions.to_csv("lstm_predictions_with_signals.csv", index=False)

print("LSTM predictions and trading signals saved to 'lstm_predictions_with_signals.csv'")

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in: {format_time(elapsed_time)}")
