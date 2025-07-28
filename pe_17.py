# gemini 28jl25 1534  lunes
# parece que Git funciona 
import os

# Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # <--- Keep this line to prevent plot display

# from keras.models import Sequential
# from keras.layers import LSTM, Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


from sklearn.preprocessing import MinMaxScaler

import os
import glob
import time
import talib
import datetime
import json
import random
import tensorflow as tf


# --- Load Configuration from JSON File ---
config_file_path = 'config.json'
try:
    with open(config_file_path, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{config_file_path}' not found. Please create it.")
    exit() # Exit the script if config file is missing

# Extract parameters from config
model_params = config.get('model_params', {})
strategy_params = config.get('strategy_params', {})
backtest_params = config.get('backtest_params', {})
general_settings = config.get('general_settings', {})

# Assign parameters to variables, using default values if not found in config
seed_value = general_settings.get('random_seed', 42)
seq_length = model_params.get('seq_length', 50)
epochs = model_params.get('epochs', 20)
batch_size = model_params.get('batch_size', 32)
lstm_units = model_params.get('lstm_units', 50)
dense_units = model_params.get('dense_units', 25)

price_threshold = strategy_params.get('price_threshold', 0.0005)
rsi_buy_threshold = strategy_params.get('rsi_buy_threshold', 30)
rsi_sell_threshold = strategy_params.get('rsi_sell_threshold', 70)
stop_loss_pct = strategy_params.get('stop_loss_pct', 0.01)
take_profit_pct = strategy_params.get('take_profit_pct', 0.02)
transaction_cost_pct = strategy_params.get('transaction_cost_pct', 0.0002)

initial_capital = backtest_params.get('initial_capital', 100000)


# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Optional: For even greater determinism on GPU (may slow down training slightly)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# --- DEBUG PRINTS (KEEP THESE FOR NOW TO VERIFY) ---
print(f"DEBUG (p_17.py): Loaded epochs: {epochs}")
print(f"DEBUG (p_17.py): Loaded price_threshold: {price_threshold}")
print(f"DEBUG (p_17.py): Loaded stop_loss_pct: {stop_loss_pct}")
print(f"DEBUG (p_17.py): Loaded take_profit_pct: {take_profit_pct}")
# --- END DEBUG PRINTS ---


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
# df = pd.read_csv("forex_data.csv", delimiter=';')
df = pd.read_csv("forex_data_m8.csv", delimiter=';')

# --- Correctly handle column renaming and DateTime parsing ---
df = df.rename(columns={
    'Timestamp': 'DateTime_Full'
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

# Use seq_length from config
target_feature_index = features.index('Close')
X, y = create_sequences(df_scaled, seq_length, target_feature_index)

y = y.reshape(-1, 1)

# Build and train the model
num_features = df_scaled.shape[1]

model = Sequential([
    LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, num_features)),
    LSTM(lstm_units, return_sequences=False),
    Dense(dense_units),
    Dense(1)
])

# model.compile(optimizer='adam', loss='mean_squared_squared_error')
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=epochs, batch_size=batch_size) # <--- CORRECTED: Using 'epochs' and 'batch_size' variables

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
def generate_signals_with_indicators(df_data, price_threshold, rsi_buy_threshold, rsi_sell_threshold):
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
    price_threshold=price_threshold, # <--- CORRECTED: Using 'price_threshold' variable
    rsi_buy_threshold=rsi_buy_threshold, # <--- CORRECTED: Using 'rsi_buy_threshold' variable
    rsi_sell_threshold=rsi_sell_threshold # <--- CORRECTED: Using 'rsi_sell_threshold' variable
)
df_combined['Signal'] = trading_signals

# Update df_predictions to be df_combined for plotting and saving
df_predictions = df_combined


# --- NEW: Backtesting Function with Stop-Loss and Take-Profit ---
def backtest_strategy(df_data, initial_capital, stop_loss_pct, take_profit_pct, transaction_cost_pct):
    capital = initial_capital
    position_units = 0
    in_position = False
    entry_price = 0.0

    portfolio_value = []
    cash_history = []
    position_units_history = []
    trade_log = []

    for i in range(len(df_data)):
        current_price = df_data['Actual'].iloc[i]
        signal = df_data['Signal'].iloc[i]
        current_datetime = df_data['DateTime'].iloc[i]

        current_portfolio_val = capital + (position_units * current_price)
        portfolio_value.append(current_portfolio_val)
        cash_history.append(capital)
        position_units_history.append(position_units)

        if in_position:
            if current_price <= entry_price * (1 - stop_loss_pct):
                profit_loss = (current_price - entry_price) * position_units
                capital += position_units * current_price * (1 - transaction_cost_pct)
                trade_log.append({
                    'DateTime': current_datetime,
                    'Type': 'Sell (Stop-Loss)',
                    'Price': current_price,
                    'Units': position_units,
                    'PnL': profit_loss,
                    'Reason': 'Stop-Loss Hit'
                })
                position_units = 0
                in_position = False
                continue

            if current_price >= entry_price * (1 + take_profit_pct):
                profit_loss = (current_price - entry_price) * position_units
                capital += position_units * current_price * (1 - transaction_cost_pct)
                trade_log.append({
                    'DateTime': current_datetime,
                    'Type': 'Sell (Take-Profit)',
                    'Price': current_price,
                    'Units': position_units,
                    'PnL': profit_loss,
                    'Reason': 'Take-Profit Hit'
                })
                position_units = 0
                in_position = False
                continue

        if signal == 'Buy' and not in_position:
            if capital > 0:
                units_to_buy = capital / current_price
                position_units += units_to_buy
                capital = 0
                in_position = True
                entry_price = current_price
                trade_log.append({
                    'DateTime': current_datetime,
                    'Type': 'Buy',
                    'Price': current_price,
                    'Units': units_to_buy,
                    'PnL': 0,
                    'Reason': 'Buy Signal'
                })

        elif signal == 'Sell' and in_position:
            profit_loss = (current_price - entry_price) * position_units
            capital += position_units * current_price * (1 - transaction_cost_pct)
            trade_log.append({
                'DateTime': current_datetime,
                'Type': 'Sell (Signal)',
                'Price': current_price,
                'Units': position_units,
                'PnL': profit_loss,
                'Reason': 'Sell Signal'
            })
            position_units = 0
            in_position = False

    if in_position:
        final_price = df_data['Actual'].iloc[-1]
        profit_loss = (final_price - entry_price) * position_units
        capital += position_units * final_price * (1 - transaction_cost_pct)
        trade_log.append({
            'DateTime': df_data['DateTime'].iloc[-1],
            'Type': 'Liquidate (End)',
            'Price': final_price,
            'Units': position_units,
            'PnL': profit_loss,
            'Reason': 'End of Backtest Liquidation'
        })

    final_portfolio_val = capital + (position_units * df_data['Actual'].iloc[-1] if position_units > 0 else 0)
    if len(portfolio_value) > 0:
        portfolio_value[-1] = final_portfolio_val
        cash_history[-1] = capital
        position_units_history[-1] = position_units
    else:
        portfolio_value.append(final_portfolio_val)
        cash_history.append(capital)
        position_units_history.append(position_units)


    backtest_results = pd.DataFrame({
        'DateTime': df_data['DateTime'],
        'Portfolio_Value': portfolio_value[:len(df_data)],
        'Cash': cash_history[:len(df_data)],
        'Position_Units': position_units_history[:len(df_data)]
    })
    trade_log_df = pd.DataFrame(trade_log)

    return backtest_results, trade_log_df

backtest_results_df, trade_log_df = backtest_strategy(
    df_predictions,
    initial_capital=initial_capital, # <--- CORRECTED: Using 'initial_capital' variable
    stop_loss_pct=stop_loss_pct,       # <--- CORRECTED: Using 'stop_loss_pct' variable
    take_profit_pct=take_profit_pct,     # <--- CORRECTED: Using 'take_profit_pct' variable
    transaction_cost_pct=transaction_cost_pct # <--- CORRECTED: Using 'transaction_cost_pct' variable
)


# --- Plotting Backtest Results ---
plt.figure(figsize=(14, 7))
plt.plot(backtest_results_df['DateTime'], backtest_results_df['Portfolio_Value'], label="Portfolio Value", color='purple')
plt.title('Backtesting Strategy Performance')
plt.xlabel('Date Time')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.close('all') # <--- Keep this line to close figures in memory

# Print final performance
current_local_time = datetime.datetime.now()
formatted_time = current_local_time.strftime("%A, %B %d %Y %H:%M")

initial_capital_from_bt = backtest_results_df['Portfolio_Value'].iloc[0]
final_capital = backtest_results_df['Portfolio_Value'].iloc[-1]
total_return = ((final_capital - initial_capital_from_bt) / initial_capital_from_bt) * 100

print(f"\n--- Backtesting Results ({formatted_time}) ---")
print(f"Initial Capital: ${initial_capital_from_bt:,.2f}")
print(f"Final Capital: ${final_capital:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {len(trade_log_df[trade_log_df['Type'] == 'Buy'])}")

# Save backtest results
backtest_filename = f"lstm_backtest_run_{run_number}.csv"
backtest_results_df.to_csv(backtest_filename, index=False)
print(f"Backtest results saved to '{backtest_filename}'")

# Save trade log for detailed analysis
trade_log_filename = f"lstm_trade_log_run_{run_number}.csv"
trade_log_df.to_csv(trade_log_filename, index=False)
print(f"Trade log saved to '{trade_log_filename}'")


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
plt.close('all') # <--- Keep this line to close figures in memory

# Save predictions and signals to CSV (this was already there)
predictions_filename = f"lstm_predictions_run_{run_number}_with_signals.csv"
df_predictions.to_csv(predictions_filename, index=False)

print(f"LSTM predictions and trading signals saved to '{predictions_filename}'")

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in: {format_time(elapsed_time)}")
