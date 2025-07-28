# control_script.py
# gemini
import json
import subprocess
import sys
import pandas as pd
import re # For regular expression parsing
import os
import numpy as np
import time # Import the time module

# --- Configuration ---
MAIN_SCRIPT_PATH = 'pe_17.py'
CONFIG_FILE_PATH = 'config.json'
RESULTS_SUMMARY_FILE = 'backtesting_summary.csv'

# --- Utility Function for Time Formatting ---
def format_time(seconds):
    """Converts seconds to a human-readable format (e.g., MM:SS or HH:MM:SS)."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"


# --- Variables to iterate ---
# Define the range for variables you want to test
# Example: Vary epochs from 10 to 30, and price_threshold from 0.0003 to 0.0007

#    'epochs_range': {'start': 10, 'end': 30, 'step': 5},
#    'price_threshold_range': {'start': 0.0003, 'end': 0.0007, 'step': 0.0001}

ITERATION_CONFIG = {
    'epochs_range': {'start': 10, 'end': 30, 'step': 5},
    'price_threshold_range': {'start': 0.0003, 'end': 0.0005, 'step': 0.0001}
}

# original
# 'epochs_range': {'start': 10, 'end': 30, 'step': 5},
# 'price_threshold_range': {'start': 0.0003, 'end': 0.0007, 'step': 0.0001}

# backtest 01    58 minutos
# 'epochs_range': {'start': 10, 'end': 20, 'step': 5},
# 'price_threshold_range': {'start': 0.0003, 'end': 0.0005, 'step': 0.0001}


# --- Functions to manage config.json ---
def load_config(file_path):
    """Loads the JSON configuration from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        sys.exit(1) # Exit if config is missing
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check file format.")
        sys.exit(1)

def save_config(config_data, file_path):
    """Saves the updated JSON configuration to a file."""
    with open(file_path, 'w') as f:
        json.dump(config_data, f, indent=4)

# --- Function to run the main script and parse its output ---
def run_main_script_and_get_stats(script_path):
    """
    Runs the main script as a subprocess and captures its output to extract statistics.
    Assumes the main script prints Final Capital, Total Return, and Total Trades.
    """
    print(f"Running {script_path}...")
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        process = subprocess.run([sys.executable, script_path],
                                 capture_output=True, text=True, check=True)
        output = process.stdout
        # print("--- Main Script Output ---")
        # print(output) # Uncomment for debugging script output
        # print("--------------------------")

        # --- Parse statistics using regular expressions ---
        # Look for lines like:
        # Final Capital: $97,839.99
        # Total Return: -2.16%
        # Total Trades: X

        final_capital_match = re.search(r"Final Capital: \$([0-9,.]+)", output)
        total_return_match = re.search(r"Total Return: (-?\d+\.\d+)%", output)
        total_trades_match = re.search(r"Total Trades: (\d+)", output)

        stats = {
            'final_capital': float(final_capital_match.group(1).replace(',', '')) if final_capital_match else None,
            'total_return_pct': float(total_return_match.group(1)) if total_return_match else None,
            'total_trades': int(total_trades_match.group(1)) if total_trades_match else None
        }

        if None in stats.values():
            print("Warning: Could not extract all statistics from main script output.")
            print(f"Output received:\n{output}")
            print(f"Parsed stats: {stats}")
        return stats

    except subprocess.CalledProcessError as e:
        print(f"Error running script '{script_path}':")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

# --- Main control loop ---
if __name__ == "__main__":

    # --- Timer 1: Total Elapsed Time Start ---
    total_start_time = time.time()

    all_results = []
    initial_config = load_config(CONFIG_FILE_PATH) # Load the base config once

    # Example: Iterate through epochs and price_threshold
    epochs_values = range(ITERATION_CONFIG['epochs_range']['start'],
                          ITERATION_CONFIG['epochs_range']['end'] + 1,
                          ITERATION_CONFIG['epochs_range']['step'])

    price_threshold_values = np.arange(ITERATION_CONFIG['price_threshold_range']['start'],
                                       ITERATION_CONFIG['price_threshold_range']['end'] + ITERATION_CONFIG['price_threshold_range']['step'], # Add step for inclusive end
                                       ITERATION_CONFIG['price_threshold_range']['step'])


    print(f"Starting backtesting scenarios...")
    print(f"Epochs to test: {list(epochs_values)}")
    print(f"Price Thresholds to test: {[f'{val:.4f}' for val in price_threshold_values]}")


    for current_epochs in epochs_values:
        for current_price_threshold in price_threshold_values:
            # --- Timer 2: Current Scenario Start ---
            scenario_start_time = time.time()

            print(f"\n--- Running scenario: Epochs={current_epochs}, PriceThreshold={current_price_threshold:.4f} ---")

            # Create a mutable copy of the initial configuration for this run
            current_run_config = initial_config.copy()

            # Modify variables for the current scenario
            # Ensure the keys match your config.json structure
            if 'model_params' in current_run_config:
                current_run_config['model_params']['epochs'] = current_epochs
            else:
                print("Warning: 'model_params' section not found in config.json. Epochs not updated.")

            if 'strategy_params' in current_run_config:
                current_run_config['strategy_params']['price_threshold'] = float(f'{current_price_threshold:.4f}') # Ensure float type and consistent formatting
            else:
                print("Warning: 'strategy_params' section not found in config.json. Price Threshold not updated.")

            # Save the modified config
            save_config(current_run_config, CONFIG_FILE_PATH)

            # Run the main script and get statistics
            stats = run_main_script_and_get_stats(MAIN_SCRIPT_PATH)

            # --- Timer 2: Current Scenario End and Print ---
            scenario_elapsed_time = time.time() - scenario_start_time
            print(f"    Current scenario duration: {format_time(scenario_elapsed_time)}")
            
            # --- Timer 1: Total Elapsed Time Print ---
            current_total_elapsed_time = time.time() - total_start_time
            print(f"    Total elapsed time: {format_time(current_total_elapsed_time)}")

            if stats:
                scenario_results = {
                    'epochs': int(current_epochs),
                    'price_threshold': float(f'{current_price_threshold:.4f}'),
                    # Or for SL/TP:
                    # 'stop_loss_pct': float(f'{current_stop_loss_pct:.4f}'),
                    # 'take_profit_pct': float(f'{current_take_profit_pct:.4f}'),
                    'scenario_duration_seconds': scenario_elapsed_time, # Store raw seconds for analysis
                    **stats
                }
                all_results.append(scenario_results)
            else:
                print(f"Skipping scenario due to error: Epochs={current_epochs}, PriceThreshold={current_price_threshold}")

    print("\n--- All scenarios completed ---")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_SUMMARY_FILE, index=False)
        print(f"Summary of all backtesting results saved to '{RESULTS_SUMMARY_FILE}'")
        print("\n--- Summary Table ---")
        print(results_df)
    else:
        print("No results were collected.")

    # --- Timer 1: Final Total Elapsed Time ---
    final_total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal execution time for all scenarios: {format_time(final_total_elapsed_time)}")
