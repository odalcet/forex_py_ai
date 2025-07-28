import os

# Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


# Load data - original: "forex_data.csv"
df = pd.read_csv("forex_data.csv", delimiter=';')
# df = pd.read_csv("forex_data_m5_05.csv", delimiter=';')
df = df[['Close']]  # Use only the closing price

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 50
X, y = create_sequences(df_scaled, seq_length)

# Build the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Predict and inverse transform
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Plot actual vs predicted
plt.plot(df.index[:len(predictions)], df['Close'][:len(predictions)], label="Actual Prices")
plt.plot(df.index[:len(predictions)], predictions, label="Predicted Prices")
plt.legend()
plt.show()

# Prepare DataFrame with predictions
df_predictions = pd.DataFrame({
    'Date': df.index[seq_length:],
    'Actual': df['Close'].values[seq_length:],
    'Predicted': predictions.flatten()
})

# Generate trading signals
df_predictions['Signal'] = 0
df_predictions['Signal'][1:] = np.where(
    df_predictions['Predicted'][1:].values > df_predictions['Actual'][:-1].values, 1, -1
)

# Simulate trading
initial_balance = 10000
balance = initial_balance
position = 0  # 0 = no position, 1 = long
trades = []

for i in range(1, len(df_predictions)):
    signal = df_predictions.at[i, 'Signal']
    price = df_predictions.at[i, 'Actual']
    date = df_predictions.at[i, 'Date']

    if signal == 1 and position != 1:
        balance -= price
        position = 1
        trades.append({'Date': date, 'Action': 'BUY', 'Price': price})

    elif signal == -1 and position == 1:
        balance += price
        position = 0
        trades.append({'Date': date, 'Action': 'SELL', 'Price': price})

final_balance = balance + (df_predictions.iloc[-1]['Actual'] if position == 1 else 0)
print(f"Final Balance: {final_balance:.2f}")

# Save trade log and predictions with run number
df_predictions.to_csv(f"lstm_predictions_run_{run_number}.csv", index=False)
pd.DataFrame(trades).to_csv(f"trade_log_run_{run_number}.csv", index=False)

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in: {format_time(elapsed_time)}")
