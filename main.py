# main.py

import yaml
import os
import matplotlib.pyplot as plt  # Still needed for plt.show()

# Import modules from our backtester and strategies directories
from backtester.data_handler import DeltaExchangeDataHandler
from backtester.backtesting_engine import BacktestingEngine
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy  # Import our specific strategy
from strategies.mean_reversion import MeanReversionStrategy

# Import our new modules
from backtester import performance_metrics
from visualizations import plotting


def main():
    """
    Main function to run the backtesting process.
    It loads configuration, fetches data, applies the strategy, and runs the backtest.
    """
    print("Starting crypto backtester...")

    # --- 1. Load Configuration ---
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create a config.yaml file in the 'config/' directory.")
        # Optionally create a dummy config for first-time users
        sample_config = {
            'tickers': ['BTCUSD'],
            'timeframe': '1h',
            'start_date': '2025-01-01',  # Changed to a past date for actual data fetching
            'end_date': '2025-06-30',  # Changed to a past date for actual data fetching
            'initial_capital': 10000.0,
            'commission_rate': 0.001,
            'strategy_parameters': {
                'Moving_Average_Crossover': {
                    'short_window': 10,
                    'long_window': 30
                },
                'Mean_Reversion': {
                    'window': 20,
                    'std_dev_multiplier': 2.0
                }
            },
            'active_strategy': 'Mean_Reversion'  # New key to select active strategy
        }
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
        print("A sample config.yaml has been created. Please review and run again.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tickers = config.get('tickers', [])
    timeframe = config.get('timeframe', '1h')
    start_date = config.get('start_date', None)
    end_date = config.get('end_date', None)
    initial_capital = config.get('initial_capital', 10000.0)
    commission_rate = config.get('commission_rate', 0.001)
    active_strategy_name = config.get('active_strategy', None)

    if not tickers or not start_date or not end_date or not active_strategy_name:
        print("Error: Missing essential configuration parameters (tickers, start_date, end_date, active_strategy).")
        return

    strategy_parameters = config.get('strategy_parameters', {})
    current_strategy_params = strategy_parameters.get(active_strategy_name, {})

    print(f"Running backtest for tickers: {tickers} with {active_strategy_name} strategy.")
    print(f"Timeframe: {timeframe}, Start: {start_date}, End: {end_date}")
    print(f"Initial Capital: ${initial_capital}, Commission Rate: {commission_rate}")
    print(f"Strategy Parameters ({active_strategy_name}): {current_strategy_params}")

    data_handler = DeltaExchangeDataHandler()

    for ticker in tickers:
        print(f"\n--- Running backtest for {ticker} ---")
        try:
            # --- 2. Fetch Historical Data ---
            historical_data = data_handler.fetch_historical_data(ticker, timeframe, start_date, end_date)

            if historical_data.empty:
                print(f"No historical data fetched for {ticker}. Skipping backtest.")
                continue

            print(f"Successfully fetched {len(historical_data)} data points for {ticker}.")

            # --- 3. Initialize and Run Strategy ---
            # Dynamically select strategy based on config
            if active_strategy_name == 'Moving_Average_Crossover':
                strategy = MovingAverageCrossoverStrategy(
                    data_feed=historical_data,
                    short_window=current_strategy_params.get('short_window', 10),
                    long_window=current_strategy_params.get('long_window', 30)
                )
            elif active_strategy_name == 'Mean_Reversion':  # New strategy instantiation
                strategy = MeanReversionStrategy(
                    data_feed=historical_data,
                    window = current_strategy_params.get('window', 20),
                    std_dev_multiplier = current_strategy_params.get('std_dev_multiplier', 2.0)
                )
            else:
                raise ValueError(f"Unknown strategy: {active_strategy_name}")

            # --- 4. Run Backtest ---
            backtesting_engine = BacktestingEngine(
                data_feed=historical_data,  # Original historical data needed for benchmark in engine
                strategy=strategy,
                initial_capital=initial_capital,
                commission_rate=commission_rate
            )

            # Unpack the three returned DataFrames
            backtest_results = backtesting_engine.run_backtest()
            portfolio_history = backtest_results['portfolio_history']
            trades = backtest_results['trades']
            strategy_execution_data = backtest_results['strategy_execution_data']

            # --- 5. Display Results ---
            print("\n--- Backtest Summary ---")
            print(f"Initial Capital: ${backtest_results['initial_capital']:.2f}")
            print(f"Final Capital: ${backtest_results['final_capital']:.2f}")
            print(f"Total PnL: ${backtest_results['total_pnl']:.2f}")
            print(f"Total Returns: {backtest_results['total_returns_percent']:.2f}%")
            print(f"Number of Entry Signals (BUYs): {backtest_results['num_trades']}")
            num_completed_trades = backtest_results['total_completed_trades']
            print(f"Number of Completed Trades (SELLs): {num_completed_trades}")
            print(f"Winning Trades: {backtest_results['winning_trades']}")
            print(f"Losing Trades: {backtest_results['losing_trades']}")
            print(f"Win Rate: {backtest_results['win_rate_percent']:.2f}%")
            print(f"Average PnL per Completed Trade: ${backtest_results['avg_pnl_per_trade']:.2f}")
            print(f"Max Drawdown: {backtest_results['max_drawdown_percent']:.2f}%")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"Sortino Ratio: {backtest_results['sortino_ratio']:.2f}")
            print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
            print(f"Expectancy: ${backtest_results['expectancy']:.2f}")
            print(f"Up Market Capture: {backtest_results['up_market_capture']:.2f}%")
            print(f"Down Market Capture: {backtest_results['down_market_capture']:.2f}%")

            if num_completed_trades > 0:  # Check total_completed_trades instead of num_trades for trade specific prints
                print("\n--- Trades Executed (first 5) ---")
                print(trades[['Timestamp', 'Type', 'Price', 'Units', 'Commission', 'PnL']].head().to_string())
                print("\n--- Trades Executed (last 5) ---")
                print(trades[['Timestamp', 'Type', 'Price', 'Units', 'Commission', 'PnL']].tail().to_string())

                print("\n--- Portfolio Value History (first 5) ---")
                # Ensure 'timestamp' is visible in prints
                print(portfolio_history.reset_index()[[
                    'timestamp', 'cash', 'units_held', 'holdings_value', 'total_value',
                    'benchmark_value']].head().to_string())
                print("\n--- Portfolio Value History (last 5) ---")
                print(portfolio_history.reset_index()[[
                    'timestamp', 'cash', 'units_held', 'holdings_value', 'total_value',
                    'benchmark_value']].tail().to_string())

                # --- Plotting calls (using the new plotting module) ---
                plotting.plot_cumulative_returns_vs_benchmark(portfolio_history, ticker)
                plotting.plot_portfolio_performance(portfolio_history, ticker)  # Original equity curve
                plotting.plot_drawdowns(backtest_results['drawdown_series'], backtest_results['max_drawdown_percent'],
                                        ticker)
                # Pass strategy_execution_data to plot_trades_on_price_chart
                plotting.plot_trades_on_price_chart(strategy_execution_data, trades, ticker)
                plotting.plot_rolling_metrics(portfolio_history, ticker)
                plotting.plot_returns_distribution(portfolio_history, ticker)
                plotting.plot_pnl_per_trade_distribution(trades, ticker)

            else:
                print("No trades were executed during the backtest.")

        except ValueError as e:
            print(f"Error during strategy or backtest for {ticker}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {ticker}: {e}")

    print("\nBacktesting process finished.")
    # Call plt.show() here to display all figures generated across all tickers
    plt.show()


if __name__ == '__main__':
    main()