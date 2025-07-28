# gemini
import os

# Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import random
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
seed_value = 42 # You can choose any integer here
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import os
import glob
import time
import talib # Make sure talib is imported for RSI
import datetime # <--- ADD THIS LINE

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
    run_numbers = []
    for f in existing_runs:
        try:
            parts = f.split("_")
            if 'with' in parts:
                run_num_str = parts[parts.index('with') - 1]
                run_numbers.append(int(run_num_str))
            else:
                run_numbers.append(int(f.split("_")[-1].split(".")[0]))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse run number from filename: {f}")
            continue

    if run_numbers:
        run_number = max(run_numbers) + 1
    else:
        run_number = 1
else:
    run_number = 1

# Start timer
start_time = time.time()

# Load data, assuming the first column is the full DateTime string
df = pd.read_csv("forex_data.csv", delimiter=';')

# --- Correctly handle column renaming and DateTime parsing ---
df = df.rename(columns={
    'Date': 'DateTime_Full',
    'Time': 'Open',
    'Open': 'High',
    'High': 'Low',
    'Low': 'Close',
    'Close': 'Volume',
    'Volume': 'Extra_Column_If_Any'
})

df['DateTime'] = pd.to_datetime(df['DateTime_Full'], format='%Y.%m.%d %H:%M', errors='coerce')
df = df.set_index('DateTime')
df = df.drop(columns=['DateTime_Full', 'Extra_Column_If_Any'], errors='ignore')

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_cleaned = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()

# --- Moving Averages ---
df_cleaned['SMA_50'] = df_cleaned['Close'].rolling(window=50).mean()
df_cleaned['SMA_200'] = df_cleaned['Close'].rolling(window=200).mean()
df_cleaned['EMA_12'] = df_cleaned['Close'].ewm(span=12, adjust=False).mean()
df_cleaned['EMA_26'] = df_cleaned['Close'].ewm(span=26, adjust=False).mean()

# --- Relative Strength Index (RSI) ---
df_cleaned['RSI'] = talib.RSI(df_cleaned['Close'], timeperiod=14)

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
        labels.append(data[i + seq_length, target_feature_index])
    return np.array(sequences), np.array(labels)

seq_length = 50
target_feature_index = features.index('Close')
X, y = create_sequences(df_scaled, seq_length, target_feature_index)

y = y.reshape(-1, 1)

# Build and train the model
num_features = df_scaled.shape[1]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Output Variables:
#   model.fit() returns a History object. While not explicitly 
#         assigned to a variable in your provided snippet (e.g., 
#         history = model.fit(...)), this object contains 
#         valuable information about the training process, such as:
#   The loss value at the end of each epoch.
#   (If you use validation data) The validation loss at the end of each epoch.

history = model.fit(X, y, epochs=20, batch_size=32)
print(history.history['loss']) # To see the loss values over epochs

# Generate predictions
predictions = model.predict(X)
dummy_array_for_inverse = np.zeros((predictions.shape[0], num_features))
dummy_array_for_inverse[:, target_feature_index] = predictions.flatten()
predictions_original_scale = scaler.inverse_transform(dummy_array_for_inverse)[:, target_feature_index]

actual_prices_for_predictions = df_cleaned['Close'].values[seq_length : len(predictions_original_scale) + seq_length]

# Create initial df_predictions DataFrame
df_predictions = pd.DataFrame({
    'DateTime': df_cleaned.index[seq_length : len(predictions_original_scale) + seq_length],
    'Actual': actual_prices_for_predictions,
    'Predicted': predictions_original_scale
})

# Ensure df_predictions has DateTime as index for merging
df_predictions_indexed = df_predictions.set_index('DateTime')

# Select the indicator columns from df_cleaned that align with df_predictions' timeframe
relevant_indicators = df_cleaned[
    df_cleaned.index.isin(df_predictions_indexed.index)
][['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI']]

# Merge them
df_combined = df_predictions_indexed.merge(
    relevant_indicators,
    left_index=True,
    right_index=True,
    how='inner'
).reset_index()


# --- Modified generate_signals function ---
def generate_signals_with_indicators(df_data, price_threshold=0.0005, rsi_buy_threshold=30, rsi_sell_threshold=70):
    signals = []
    for i in range(len(df_data)):
        current_actual = df_data['Actual'].iloc[i]
        current_predicted = df_data['Predicted'].iloc[i]
        current_ema12 = df_data['EMA_12'].iloc[i]
        current_rsi = df_data['RSI'].iloc[i]

        signal = 'Hold'

        # Example Strategy: Predicted vs EMA crossover with RSI confirmation
        if (current_predicted > current_ema12 * (1 + price_threshold)) and (current_rsi < rsi_buy_threshold):
            signal = 'Buy'
        elif (current_predicted < current_ema12 * (1 - price_threshold)) and (current_rsi > rsi_sell_threshold):
            signal = 'Sell'

        signals.append(signal)
    return signals

# Generate signals using the new function and combined DataFrame
trading_signals = generate_signals_with_indicators(
    df_combined,
    price_threshold=0.0005,
    rsi_buy_threshold=30,
    rsi_sell_threshold=70
)
df_combined['Signal'] = trading_signals

# Update df_predictions to be df_combined for plotting and saving
df_predictions = df_combined


# --- NEW: Backtesting Function ---
def backtest_strategy(df_data, initial_capital=100000):
    capital = initial_capital
    position = 0 # 0 for no position, 1 for long (holding asset)
    portfolio_value = []
    cash_history = []
    position_history = []

    # Iterate through the DataFrame to simulate trades
    for i in range(len(df_data)):
        current_price = df_data['Actual'].iloc[i]
        signal = df_data['Signal'].iloc[i]

        if signal == 'Buy':
            if position == 0: # Only buy if not already in a position
                # Buy as many units as possible with current capital
                units_to_buy = capital / current_price
                position += units_to_buy
                capital = 0 # All capital invested
                print(f"Buy signal at {df_data['DateTime'].iloc[i]}: Bought {units_to_buy:.2f} units at {current_price:.5f}")
        elif signal == 'Sell':
            if position > 0: # Only sell if holding a position
                # Sell all units
                capital += position * current_price
                print(f"Sell signal at {df_data['DateTime'].iloc[i]}: Sold {position:.2f} units at {current_price:.5f}")
                position = 0 # No longer holding position

        # Calculate portfolio value at the end of each day
        current_portfolio_value = capital + (position * current_price)
        portfolio_value.append(current_portfolio_value)
        cash_history.append(capital)
        position_history.append(position)

    # If still in a position at the end, liquidate it (for final portfolio value calculation)
    if position > 0:
        capital += position * df_data['Actual'].iloc[-1]
        position = 0
        print(f"Liquidating remaining position at {df_data['DateTime'].iloc[-1]} for {df_data['Actual'].iloc[-1]:.5f}")

    # Create a DataFrame for backtest results
    backtest_results = pd.DataFrame({
        'DateTime': df_data['DateTime'],
        'Portfolio_Value': portfolio_value,
        'Cash': cash_history,
        'Position_Units': position_history
    })
    return backtest_results

# Run the backtest
backtest_results_df = backtest_strategy(df_predictions, initial_capital=100000)

# --- Plotting Backtest Results ---
plt.figure(figsize=(14, 7))
plt.plot(backtest_results_df['DateTime'], backtest_results_df['Portfolio_Value'], label="Portfolio Value", color='purple')
plt.title('Backtesting Strategy Performance')
plt.xlabel('Date Time')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()

# Print final performance
initial_capital = backtest_results_df['Portfolio_Value'].iloc[0]
final_capital = backtest_results_df['Portfolio_Value'].iloc[-1]
total_return = ((final_capital - initial_capital) / initial_capital) * 100

print(f"\n--- Backtesting Results ---")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Capital: ${final_capital:,.2f}")
print(f"Total Return: {total_return:.2f}%")

# Save backtest results
backtest_filename = f"lstm_backtest_run_{run_number}.csv"
backtest_results_df.to_csv(backtest_filename, index=False)
print(f"Backtest results saved to '{backtest_filename}'")


# Plotting original signals (kept for reference, using df_predictions)
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

# Save predictions and signals to CSV (this was already there)
predictions_filename = f"lstm_predictions_run_{run_number}_with_signals.csv"
df_predictions.to_csv(predictions_filename, index=False)

print(f"LSTM predictions and trading signals saved to '{predictions_filename}'")

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in: {format_time(elapsed_time)}")
# Get the current local date and time
current_local_time = datetime.datetime.now()

# Format it as "Monday, July 23 2025 20:39"
formatted_time = current_local_time.strftime("%A, %B %d %Y %H:%M")

# Print the formatted time
print(formatted_time)
