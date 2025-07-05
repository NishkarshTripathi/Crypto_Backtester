# strategies/moving_average_crossover.py

import pandas as pd
import numpy as np


class MovingAverageCrossoverStrategy:
    def __init__(self, data_feed, short_window, long_window):
        self.data = data_feed
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        # Ensure 'close' column exists
        if 'close' not in self.data.columns:
            raise ValueError("Data feed must contain a 'close' price column.")

        signals = pd.DataFrame(index=self.data.index)
        signals['close'] = self.data['close']  # Keep close price for context

        # Calculate moving averages
        signals['short_ma'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_ma'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()

        # Debugging: Print a sample of MAs
        print("\n--- MA Calculation Debug (first 5 and last 5 rows) ---")
        print(signals[['close', 'short_ma', 'long_ma']].head())
        print(signals[['close', 'short_ma', 'long_ma']].tail())
        print("--------------------------------------------------")

        # Generate signal: 1 for buy, -1 for sell
        # Initializing 'signal' column with 0
        signals['signal'] = 0

        # Create a 'crossover' column: 1 if short_ma crosses above long_ma, -1 if it crosses below
        # This captures the *event* of a crossover
        signals['crossover'] = 0
        signals.loc[signals['short_ma'] > signals['long_ma'], 'crossover'] = 1  # Short above Long
        signals.loc[signals['short_ma'] < signals['long_ma'], 'crossover'] = -1  # Short below Long

        # Identify actual crossover points (where the 'crossover' state *changes*)
        # We need to detect transitions from -1 to 1 (buy) and 1 to -1 (sell)
        signals['final_signal'] = 0

        # Buy signal: when short_ma crosses above long_ma
        # This is a transition from short_ma being below long_ma to above
        signals.loc[(signals['short_ma'].shift(1) <= signals['long_ma'].shift(1)) &
                    (signals['short_ma'] > signals['long_ma']), 'final_signal'] = 1

        # Sell signal: when short_ma crosses below long_ma
        # This is a transition from short_ma being above long_ma to below
        signals.loc[(signals['short_ma'].shift(1) >= signals['long_ma'].shift(1)) &
                    (signals['short_ma'] < signals['long_ma']), 'final_signal'] = -1

        # Debugging: Print a sample of signals
        print("\n--- Signal Generation Debug (first 5 and last 5 rows) ---")
        print(signals[['short_ma', 'long_ma', 'final_signal']].head())
        print(signals[['short_ma', 'long_ma', 'final_signal']].tail())
        print("--------------------------------------------------")

        # Debugging: Check value counts of final_signal
        print("\n--- Final Signal Value Counts ---")
        print(signals['final_signal'].value_counts())
        print("-----------------------------------")

        # Debugging: Check where actual trades should occur (non-zero signals)
        non_zero_signals = signals[signals['final_signal'] != 0]
        print(f"\n--- Non-zero signals ({len(non_zero_signals)} occurrences) ---")
        if not non_zero_signals.empty:
            print(non_zero_signals[['short_ma', 'long_ma', 'final_signal']].head(10))  # Print first 10
            print(non_zero_signals[['short_ma', 'long_ma', 'final_signal']].tail(10))  # Print last 10
        else:
            print("No non-zero signals generated at all!")
        print("-----------------------------------")

        return signals