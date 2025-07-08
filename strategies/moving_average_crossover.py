# strategies/moving_average_crossover.py

import pandas as pd
import numpy as np


class MovingAverageCrossoverStrategy:
    def __init__(self, data_feed, short_window, long_window):
        self.data = data_feed.copy() # Work on a copy to avoid modifying original data_feed
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        # Ensure 'close' column exists
        if 'close' not in self.data.columns:
            raise ValueError("Data feed must contain a 'close' price column.")

        # Initialize the DataFrame that will be returned
        # It must contain 'close' and will eventually contain 'final_signal' and indicators
        strategy_execution_data = pd.DataFrame(index=self.data.index)
        strategy_execution_data['close'] = self.data['close'] # Keep close price for context

        # Calculate moving averages directly on strategy_execution_data for consistency
        strategy_execution_data['short_ma'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        strategy_execution_data['long_ma'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()

        # Debugging: Print a sample of MAs
        print("\n--- MA Calculation Debug (first 5 and last 5 rows) ---")
        print(strategy_execution_data[['close', 'short_ma', 'long_ma']].head())
        print(strategy_execution_data[['close', 'short_ma', 'long_ma']].tail())
        print("--------------------------------------------------")

        # Generate signal: 1 for buy, -1 for sell, 0 for hold
        # Initialize 'final_signal' column with 0 (hold)
        strategy_execution_data['final_signal'] = 0

        # Buy signal: when short_ma crosses above long_ma
        # This is a transition from short_ma being below or equal to long_ma, to strictly above
        strategy_execution_data.loc[(strategy_execution_data['short_ma'].shift(1) <= strategy_execution_data['long_ma'].shift(1)) &
                                    (strategy_execution_data['short_ma'] > strategy_execution_data['long_ma']), 'final_signal'] = 1

        # Sell signal: when short_ma crosses below long_ma
        # This is a transition from short_ma being above or equal to long_ma, to strictly below
        strategy_execution_data.loc[(strategy_execution_data['short_ma'].shift(1) >= strategy_execution_data['long_ma'].shift(1)) &
                                    (strategy_execution_data['short_ma'] < strategy_execution_data['long_ma']), 'final_signal'] = -1

        # Debugging: Print a sample of signals
        print("\n--- Signal Generation Debug (first 5 and last 5 rows) ---")
        print(strategy_execution_data[['short_ma', 'long_ma', 'final_signal']].head())
        print(strategy_execution_data[['short_ma', 'long_ma', 'final_signal']].tail())
        print("--------------------------------------------------")

        # Debugging: Check value counts of final_signal
        print("\n--- Final Signal Value Counts ---")
        print(strategy_execution_data['final_signal'].value_counts())
        print("-----------------------------------")

        # Debugging: Check where actual trades should occur (non-zero signals)
        non_zero_signals = strategy_execution_data[strategy_execution_data['final_signal'] != 0]
        print(f"\n--- Non-zero signals ({len(non_zero_signals)} occurrences) ---")
        if not non_zero_signals.empty:
            print(non_zero_signals[['short_ma', 'long_ma', 'final_signal']].head(10)) # Print first 10
            print(non_zero_signals[['short_ma', 'long_ma', 'final_signal']].tail(10)) # Print last 10
        else:
            print("No non-zero signals generated.")
        print("-----------------------------------\n")

        # Return the comprehensive DataFrame
        return strategy_execution_data