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


# df = pd.read_csv("forex_data.csv")
df = pd.read_csv("forex_data.csv", delimiter=';')
df = df[['Close']]  # Using closing price for prediction

scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Create sequences for training
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 50  # Using past 50 days to predict the next day
X, y = create_sequences(df_scaled, seq_length)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=20, batch_size=32)

predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

plt.plot(df.index[:len(predictions)], df['Close'][:len(predictions)], label="Actual Prices")
plt.plot(df.index[:len(predictions)], predictions, label="Predicted Prices")
plt.legend()
plt.show()

df_predictions = pd.DataFrame({
    'Date': df.index[seq_length:],  # Adjust for sequence offset
    'Actual': df['Close'].values[seq_length:],
    'Predicted': predictions.flatten()
})
df_predictions.to_csv("lstm_predictions.csv", index=False)


# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in: {format_time(elapsed_time)}")
