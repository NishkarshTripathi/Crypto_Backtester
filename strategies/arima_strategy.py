# strategies/arima_strategy.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ValueWarning
import time # <--- Import the time module

class ARIMAStrategy:
    def __init__(self, data_feed, order=(5,1,0), prediction_period=1):
        self.data = data_feed
        self.order = order
        self.prediction_period = prediction_period

    def generate_signals(self):
        if 'close' not in self.data.columns:
            raise ValueError("Data feed must contain a 'close' price column for ARIMA Strategy.")

        strategy_execution_data = pd.DataFrame(index=self.data.index)
        strategy_execution_data['close'] = self.data['close']
        strategy_execution_data['predicted_price'] = np.nan
        strategy_execution_data['final_signal'] = 0

        p, d, q = self.order
        min_model_fit_points_actual = max(100, max(p, q) + d + 1)
        min_total_points_for_prediction = min_model_fit_points_actual + self.prediction_period - 1

        if len(self.data) < min_total_points_for_prediction:
            print(f"Not enough total data points for ARIMA with order {self.order} and prediction period {self.prediction_period}.")
            print(f"Need at least {min_total_points_for_prediction} points, but got {len(self.data)}.")
            return strategy_execution_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=HessianInversionWarning)
            warnings.filterwarnings("ignore", category=ValueWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            total_iterations = len(self.data)
            progress_interval = 100 # Print progress every 100 iterations

            # --- Variables for performance tracking ---
            total_decision_time = 0.0
            num_decisions_made = 0 # Only count iterations where ARIMA model was actually fitted

            for i in range(min_total_points_for_prediction - 1, total_iterations):
                current_timestamp = self.data.index[i]

                # --- Start timing for this iteration's decision process ---
                start_time = time.time()

                train_data = self.data['close'].iloc[:i]

                if len(train_data) < min_model_fit_points_actual:
                    strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                    # No ARIMA fit, so don't count this as a "decision made" by the model
                    continue

                if train_data.isnull().any():
                    train_data_cleaned = train_data.dropna()
                    if train_data_cleaned.empty:
                        print(f"Warning: All training data is NaN or empty after dropping NaNs at index {current_timestamp}. Skipping ARIMA fit.")
                        strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                        continue
                    else:
                        train_data = train_data_cleaned

                if train_data.nunique() < 2:
                    print(f"Warning: Training data has insufficient unique values (all values are the same or too few unique values) at index {current_timestamp}. Skipping ARIMA fit.")
                    strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                    continue

                if d >= 1:
                    differenced_train_data = train_data.diff().dropna()

                    if differenced_train_data.empty:
                         print(f"Warning: Differenced training data is empty at index {current_timestamp}. Skipping ARIMA fit.")
                         strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                         continue

                    if np.isclose(differenced_train_data.var(), 0) and len(differenced_train_data) > 1:
                        print(f"Warning: Differenced training data has near-zero variance at index {current_timestamp}. Skipping ARIMA fit.")
                        strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                        continue

                    if len(differenced_train_data) < (p + q + 1):
                        print(f"Warning: Insufficient differenced data points ({len(differenced_train_data)}) for ARIMA(p={p},q={q}) components at index {current_timestamp}. Skipping ARIMA fit.")
                        strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                        continue

                if (len(train_data) - d) < (p + q + 1):
                    print(f"Warning: Training data (after accounting for differencing) is too short for chosen ARIMA order {self.order} at index {current_timestamp}. Skipping ARIMA fit.")
                    strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                    continue

                try:
                    model = ARIMA(train_data, order=self.order)
                    model_fit = model.fit()

                    forecast_result = model_fit.predict(
                        start=len(train_data),
                        end=len(train_data) + self.prediction_period - 1
                    )
                    predicted_price_for_current_bar = forecast_result.iloc[self.prediction_period - 1]

                    strategy_execution_data.loc[current_timestamp, 'predicted_price'] = predicted_price_for_current_bar

                    current_close = strategy_execution_data['close'].loc[current_timestamp]

                    if predicted_price_for_current_bar > current_close:
                        strategy_execution_data.loc[current_timestamp, 'final_signal'] = 1 # Buy
                    elif predicted_price_for_current_bar < current_close:
                        strategy_execution_data.loc[current_timestamp, 'final_signal'] = -1 # Sell
                    else:
                        strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0 # Hold

                    # --- End timing for successful decision ---
                    end_time = time.time()
                    total_decision_time += (end_time - start_time)
                    num_decisions_made += 1

                except Exception as e:
                    print(f"ARIMA model failed at index {current_timestamp} with error: {e}. Setting signal to 0.")
                    strategy_execution_data.loc[current_timestamp, 'final_signal'] = 0
                    # Do NOT increment num_decisions_made if model failed

                # --- Progress Indicator (now includes average time) ---
                if (i + 1) % progress_interval == 0 or (i + 1) == total_iterations:
                    progress_percent = ((i + 1) / total_iterations) * 100
                    avg_time_per_decision = total_decision_time / num_decisions_made if num_decisions_made > 0 else 0.0
                    print(f"ARIMA Strategy Progress: {progress_percent:.2f}% complete ({i + 1}/{total_iterations} data points processed) at {current_timestamp}. Avg decision time: {avg_time_per_decision:.4f}s")


        # Debugging: Print a sample of signals and indicators
        print("\n--- ARIMA Signal Generation Debug (first 5 and last 5 rows) ---")
        print(strategy_execution_data[['close', 'predicted_price', 'final_signal']].head())
        print(strategy_execution_data[['close', 'predicted_price', 'final_signal']].tail())
        print("---------------------------------------------------------------")

        print("\n--- Final Signal Value Counts ---")
        print(strategy_execution_data['final_signal'].value_counts())
        print("-----------------------------------")

        non_zero_signals = strategy_execution_data[strategy_execution_data['final_signal'] != 0]
        print(f"\n--- Non-zero signals ({len(non_zero_signals)} occurrences) ---")
        if not non_zero_signals.empty:
            print(non_zero_signals[['close', 'predicted_price', 'final_signal']].head(10))
            print(non_zero_signals[['close', 'predicted_price', 'final_signal']].tail(10))
        else:
            print("No non-zero signals generated for ARIMA strategy.")

        return strategy_execution_data