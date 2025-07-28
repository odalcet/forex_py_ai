# gemini

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preparation ---
# NOTE: Replace this section with your actual data loading method.
# For example: data = pd.read_csv('your_stock_data.csv', index_col='Date', parse_dates=True)
# We will create some sample data for demonstration purposes.
data = pd.DataFrame(np.random.randn(500, 1), columns=['Close'])
data['Close'] = data['Close'].cumsum() + 50
data.index = pd.date_range('2023-01-01', periods=500)

# --- 2. Indicator Calculation ---
# This section calculates all the technical indicators.

# Moving Averages (50-day and 200-day)
data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)

# Relative Strength Index (RSI)
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

# Moving Average Convergence Divergence (MACD)
# talib.MACD returns three values: the MACD line, the signal line, and the histogram
data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)


# --- 3. Signal Generation ---
# Here we define the logic for our trading signals based on the MACD crossover.

# Create a signal column, default to 0 (no signal)
data['signal'] = 0

# Define the buy signal: MACD crosses above the signal line
# We use shift(1) to ensure we are comparing the current value to the previous one
buy_condition = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))

# Define the sell signal: MACD crosses below the signal line
sell_condition = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))

# Use np.where to assign signals: 1 for buy, -1 for sell
data['signal'] = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

# Get the actual points where a signal is generated
data['buy_signal_price'] = np.where(data['signal'] == 1, data['Close'], np.nan)
data['sell_signal_price'] = np.where(data['signal'] == -1, data['Close'], np.nan)

# Clean up data by removing rows with NaN values from indicator calculations
data.dropna(inplace=True)


# --- 4. Visualization ---
# This section plots the price data, indicators, and signals.

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('Trading Signal Generation Script')

# Plot 1: Price and Signals
ax1.plot(data['Close'], label='Close Price', color='skyblue', alpha=0.9)
ax1.plot(data['SMA_50'], label='50-Day SMA', color='orange', linestyle='--', alpha=0.8)
ax1.plot(data['SMA_200'], label='200-Day SMA', color='purple', linestyle='--', alpha=0.8)

# Plot Buy and Sell Signals
ax1.scatter(data.index, data['buy_signal_price'], label='Buy Signal', marker='^', color='green', s=100, zorder=5)
ax1.scatter(data.index, data['sell_signal_price'], label='Sell Signal', marker='v', color='red', s=100, zorder=5)

ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True)

# Plot 2: RSI
ax2.plot(data['RSI'], label='RSI', color='lightcoral')
ax2.axhline(70, linestyle='--', color='red', alpha=0.5) # Overbought line
ax2.axhline(30, linestyle='--', color='green', alpha=0.5) # Oversold line
ax2.set_ylabel('RSI')
ax2.legend()
ax2.grid(True)

plt.xlabel('Date')
plt.show()

# Display the last few rows of the DataFrame with signals
print("Data with indicators and signals:")
print(data.tail(10))