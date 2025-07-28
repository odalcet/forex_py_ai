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

# Load data, assuming the first column is the full DateTime string
# and subsequent columns are shifted (Time is Open, Open is High, etc.)
df = pd.read_csv("forex_data.csv", delimiter=';')

# --- Correctly handle column renaming and DateTime parsing ---
# Rename columns based on the actual content as per your sample data
# The first column 'Date' contains 'YYYY.MM.DD HH:MM' (the full timestamp)
# The column named 'Time' contains 'Open' price, etc.
df = df.rename(columns={
    'Date': 'DateTime_Full',  # This column contains 'YYYY.MM.DD HH:MM'
    'Time': 'Open',
    'Open': 'High',
    'High': 'Low',
    'Low': 'Close',
    'Close': 'Volume',
    'Volume': 'Extra_Column_If_Any' # Catches any remaining column if header is longer
})

# Convert the 'DateTime_Full' column to datetime objects
df['DateTime'] = pd.to_datetime(df['DateTime_Full'], format='%Y.%m.%d %H:%M', errors='coerce')

# Set 'DateTime' as the DataFrame index
df = df.set_index('DateTime')

# Drop the original raw DateTime column and the potentially extra column
df = df.drop(columns=['DateTime_Full', 'Extra_Column_If_Any'], errors='ignore') # 'errors='ignore' prevents error if col doesn't exist

# Ensure relevant columns are numeric, coercing errors
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows where DateTime conversion failed (NaT values) or where essential price/volume data is NaN
df_cleaned = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy() # Keep 'Volume' if you're using it later

# --- Moving Averages ---
# Simple Moving Averages (SMA)
df_cleaned['SMA_50'] = df_cleaned['Close'].rolling(window=50).mean()
df_cleaned['SMA_200'] = df_cleaned['Close'].rolling(window=200).mean()

# Exponential Moving Averages (EMA) - more responsive
df_cleaned['EMA_12'] = df_cleaned['Close'].ewm(span=12, adjust=False).mean()
df_cleaned['EMA_26'] = df_cleaned['Close'].ewm(span=26, adjust=False).mean()

# --- Relative Strength Index (RSI) ---
df_cleaned['RSI'] = talib.RSI(df_cleaned['Close'], timeperiod=14)

# Drop rows with NaN values created by moving averages and RSI calculation
df_cleaned = df_cleaned.dropna().copy()

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
# Inverse transform the predictions.
dummy_array_for_inverse = np.zeros((predictions.shape[0], num_features))
dummy_array_for_inverse[:, target_feature_index] = predictions.flatten()
predictions_original_scale = scaler.inverse_transform(dummy_array_for_inverse)[:, target_feature_index]

# Get the actual 'Close' prices corresponding to the predictions
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


# --- (After df_predictions is created) ---

# Merge df_predictions with the relevant columns from df_cleaned (indicators)
# We need to make sure the indices align correctly.
# df_cleaned already has 'DateTime' as index and contains indicators after dropna()
# df_predictions also has 'DateTime' as a column (which we can set as index for merging)

# Ensure df_predictions has DateTime as index for merging
df_predictions_indexed = df_predictions.set_index('DateTime')

# Select the indicator columns from df_cleaned that align with df_predictions' timeframe
# The indicators in df_cleaned already account for the seq_length offset due to dropna()
# So we need to ensure we only merge the part of df_cleaned that corresponds to df_predictions
# The index of df_predictions is df_cleaned.index[seq_length : len(predictions_original_scale) + seq_length]
# So, we can directly select those rows from df_cleaned
relevant_indicators = df_cleaned[
    df_cleaned.index.isin(df_predictions_indexed.index)
][['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI']]

# Merge them
df_combined = df_predictions_indexed.merge(
    relevant_indicators,
    left_index=True,
    right_index=True,
    how='inner'
).reset_index() # Reset index to make 'DateTime' a column again if preferred for signals function


# --- Modified generate_signals function ---
def generate_signals_with_indicators(df_data, price_threshold=0.0005, rsi_buy_threshold=30, rsi_sell_threshold=70):
    signals = []
    for i in range(len(df_data)):
        current_actual = df_data['Actual'].iloc[i]
        current_predicted = df_data['Predicted'].iloc[i]
        current_ema12 = df_data['EMA_12'].iloc[i]
        current_rsi = df_data['RSI'].iloc[i]

        signal = 'Hold' # Default to Hold

        # Example Strategy 1: Predicted vs EMA crossover with RSI confirmation
        # Buy if predicted price is significantly above short-term EMA AND RSI is oversold
        if (current_predicted > current_ema12 * (1 + price_threshold)) and (current_rsi < rsi_buy_threshold):
            signal = 'Buy'
        # Sell if predicted price is significantly below short-term EMA AND RSI is overbought
        elif (current_predicted < current_ema12 * (1 - price_threshold)) and (current_rsi > rsi_sell_threshold):
            signal = 'Sell'

        # You can add more complex conditions here
        # For example, a simple golden cross / death cross on actuals (though not predictive)
        # if df_data['SMA_50'].iloc[i] > df_data['SMA_200'].iloc[i]:
        #     # Potential bullish sign, but you'd need to check for *crossovers* specifically
        #     pass

        signals.append(signal)
    return signals

# Generate signals using the new function and combined DataFrame
trading_signals = generate_signals_with_indicators(
    df_combined,
    price_threshold=0.0005, # How much difference between predicted and EMA to trigger
    rsi_buy_threshold=30,
    rsi_sell_threshold=70
)
df_combined['Signal'] = trading_signals

# --- Update df_predictions to be df_combined for plotting and saving ---
df_predictions = df_combined # Now df_predictions contains signals AND indicators

# ... (rest of your plotting and saving code will use df_predictions) ...


# Plotting with signals
plt.figure(figsize=(14, 7))
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
