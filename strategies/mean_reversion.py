# strategies/mean_reversion.py

import pandas as pd


class MeanReversionStrategy:
    def __init__(self, data_feed, window, std_dev_multiplier):
        self.data = data_feed
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier

    def generate_signals(self):
        if 'close' not in self.data.columns:
            raise ValueError("Data feed must contain a 'close' price column for Mean Reversion Strategy.")

        strategy_execution_data = pd.DataFrame(index=self.data.index)
        strategy_execution_data['close'] = self.data['close']

        # Calculate Middle Band (SMA)
        strategy_execution_data['middle_band'] = strategy_execution_data['close'].rolling(window=self.window, min_periods=1).mean()

        # Calculate Standard Deviation
        strategy_execution_data['std_dev'] = strategy_execution_data['close'].rolling(window=self.window, min_periods=1).std()

        # Calculate Upper and Lower Bands
        strategy_execution_data['upper_band'] = strategy_execution_data['middle_band'] + (strategy_execution_data['std_dev'] * self.std_dev_multiplier)
        strategy_execution_data['lower_band'] = strategy_execution_data['middle_band'] - (strategy_execution_data['std_dev'] * self.std_dev_multiplier)

        # Initialize signal column
        strategy_execution_data['final_signal'] = 0

        # --- Generate Buy Signals ---
        # A buy signal occurs when the price crosses below the lower band (oversold) and then crosses back above it.
        # Condition 1: Current close price is above the lower band
        # Condition 2: Previous close price was below or equal to the lower band
        strategy_execution_data.loc[(strategy_execution_data['close'] > strategy_execution_data['lower_band']) &
                    (strategy_execution_data['close'].shift(1) <= strategy_execution_data['lower_band'].shift(1)), 'final_signal'] = 1

        # --- Generate Sell Signals ---
        # A sell signal occurs when the price crosses above the upper band (overbought) and then crosses back below it.
        # Condition 1: Current close price is below the upper band
        # Condition 2: Previous close price was above or equal to the upper band
        strategy_execution_data.loc[(strategy_execution_data['close'] < strategy_execution_data['upper_band']) &
                    (strategy_execution_data['close'].shift(1) >= strategy_execution_data['upper_band'].shift(1)), 'final_signal'] = -1

        # Debugging: Print a sample of signals and indicators
        print("\n--- Mean Reversion Signal Generation Debug (first 5 and last 5 rows) ---")
        print(strategy_execution_data[['close', 'middle_band', 'upper_band', 'lower_band', 'final_signal']].head())
        print(strategy_execution_data[['close', 'middle_band', 'upper_band', 'lower_band', 'final_signal']].tail())
        print("-----------------------------------------------------------------------")

        # Debugging: Check value counts of final_signal
        print("\n--- Final Signal Value Counts ---")
        print(strategy_execution_data['final_signal'].value_counts())
        print("-----------------------------------")

        # Debugging: Check where actual trades should occur (non-zero signals)
        non_zero_signals = strategy_execution_data[strategy_execution_data['final_signal'] != 0]
        print(f"\n--- Non-zero signals ({len(non_zero_signals)} occurrences) ---")
        if not non_zero_signals.empty:
            print(non_zero_signals[['close', 'middle_band', 'upper_band', 'lower_band', 'final_signal']].head(10))
            print(non_zero_signals[['close', 'middle_band', 'upper_band', 'lower_band', 'final_signal']].tail(10))
        print("--------------------------------------------------")


        return strategy_execution_data