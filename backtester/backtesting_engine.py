# backtester/backtesting_engine.py

import pandas as pd

from . import performance_metrics


class BacktestingEngine:
    def __init__(self, data_feed, strategy, initial_capital=10000.0, commission_rate=0.001):
        self.data_feed = data_feed
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.portfolio_history = pd.DataFrame(index=data_feed.index)
        self.trades = pd.DataFrame(columns=['Timestamp', 'Type', 'Price', 'Units', 'Commission', 'PnL'])
        self.min_units_threshold = 0.00000001  # A very small threshold for crypto fractional units

    def run_backtest(self):
        # Generate signals and all strategy-specific data from the strategy
        strategy_execution_data = self.strategy.generate_signals()

        # Ensure the index is a DatetimeIndex and its name is 'timestamp' for consistency
        if not isinstance(strategy_execution_data.index, pd.DatetimeIndex):
            strategy_execution_data.index = pd.to_datetime(strategy_execution_data.index)
        strategy_execution_data.index.name = 'timestamp'

        # Initialize portfolio for ALL rows upfront.
        # We will populate these iteratively.
        self.portfolio_history['cash'] = self.initial_capital
        self.portfolio_history['units_held'] = 0.0
        self.portfolio_history['holdings_value'] = 0.0
        self.portfolio_history['total_value'] = self.initial_capital
        self.portfolio_history['close'] = strategy_execution_data['close'] # Add close price to portfolio history


        current_cash = self.initial_capital
        current_units = 0.0
        entry_price = 0.0
        num_trades = 0

        # Backtesting loop
        for i, (timestamp, row) in enumerate(strategy_execution_data.iterrows()):
            close_price = row['close']
            signal = row['final_signal'] # This signal can be 1 (buy), -1 (sell), or 0 (hold)

            # Update current portfolio state from previous step, or initial capital for the first step
            if i > 0:
                current_cash = self.portfolio_history.iloc[i-1]['cash']
                current_units = self.portfolio_history.iloc[i-1]['units_held']

            # Calculate holdings value based on current close price
            current_holdings_value = current_units * close_price
            current_total_value = current_cash + current_holdings_value

            # --- Execute Trades based on Signals ---
            if signal == 1:  # Buy signal
                if current_cash > 0:
                    # Buy as many units as possible with available cash
                    entry_price = current_cash
                    units_to_buy = (current_cash / close_price) * (1 - self.commission_rate) # Deduct commission from units
                    commission = current_cash * self.commission_rate
                    current_cash -= current_cash  # All cash is used
                    current_units += units_to_buy
                    num_trades += 1
                    self.trades.loc[len(self.trades)] = [timestamp, 'BUY', close_price, units_to_buy, commission, 0.0] # PnL for BUY is 0

            elif signal == -1:  # Sell signal
                if current_units > self.min_units_threshold:
                    # Sell all held units
                    commission = (current_units * close_price) * self.commission_rate

                    cash_after_exit = (current_units * close_price) - commission
                    pnl = cash_after_exit - entry_price # PnL from this sell
                    print("PnL of current trade", pnl)
                    current_cash += cash_after_exit

                    self.trades.loc[len(self.trades)] = [timestamp, 'SELL', close_price, current_units, commission, pnl]
                    current_units = 0.0 # All units are sold
                    num_trades += 1 # This counts as a completed trade cycle entry/exit


            # Update portfolio history for the current timestamp
            self.portfolio_history.loc[timestamp, 'cash'] = current_cash
            self.portfolio_history.loc[timestamp, 'units_held'] = current_units
            self.portfolio_history.loc[timestamp, 'holdings_value'] = current_units * close_price
            self.portfolio_history.loc[timestamp, 'total_value'] = current_cash + (current_units * close_price)


        # --- Backtest Results Calculation ---
        final_capital = self.portfolio_history['total_value'].iloc[-1]
        total_pnl = final_capital - self.initial_capital
        total_returns_percent = (total_pnl / self.initial_capital) * 100

        winning_trades_df = self.trades[(self.trades['Type'] == 'SELL') & (self.trades['PnL'] > 0)]
        losing_trades_df = self.trades[(self.trades['Type'] == 'SELL') & (self.trades['PnL'] < 0)]

        winning_trades_count = len(winning_trades_df)
        losing_trades_count = len(losing_trades_df)

        # Total completed trades (buy-sell cycles)
        # For a simple strategy that always closes the full position, number of sell trades
        # should equal number of buy trades unless a position is open at the end.
        total_completed_trades = len(self.trades[self.trades['Type'] == 'SELL'])

        win_rate_percent = (winning_trades_count / total_completed_trades) * 100 if total_completed_trades > 0 else 0.0
        avg_pnl_per_trade = total_pnl / total_completed_trades if total_completed_trades > 0 else 0.0

        # Calculate performance metrics using the new module
        drawdown_series, max_drawdown_percent = performance_metrics.calculate_drawdowns(self.portfolio_history)

        # Ensure 'daily_returns' is calculated in portfolio_history for Sharpe/Sortino
        self.portfolio_history['daily_returns'] = self.portfolio_history['total_value'].pct_change().fillna(0)

        # For benchmark, we'll use the simple buy-and-hold returns of the asset
        # This assumes the data_feed 'close' price is the benchmark asset's price
        benchmark_initial_value = self.data_feed['close'].iloc[0]
        benchmark_final_value = self.data_feed['close'].iloc[-1]
        benchmark_total_return = (benchmark_final_value - benchmark_initial_value) / benchmark_initial_value
        benchmark_daily_returns = self.data_feed['close'].pct_change().fillna(0)

        # Add benchmark value to portfolio history for plotting
        # This simulates buying and holding the asset with initial capital
        self.portfolio_history['benchmark_value'] = self.initial_capital * (1 + benchmark_daily_returns).cumprod()


        sharpe_ratio = performance_metrics.calculate_sharpe_ratio(self.portfolio_history)
        sortino_ratio = performance_metrics.calculate_sortino_ratio(self.portfolio_history)
        profit_factor = performance_metrics.calculate_profit_factor(self.trades)
        expectancy = performance_metrics.calculate_expectancy(self.trades)
        up_market_capture, down_market_capture = performance_metrics.calculate_capture_ratios(
            self.portfolio_history, benchmark_daily_returns
        )


        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_returns_percent': total_returns_percent,
            'num_trades': num_trades,
            'total_completed_trades': total_completed_trades,
            'winning_trades': winning_trades_count,
            'losing_trades': losing_trades_count,
            'win_rate_percent': win_rate_percent,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'portfolio_history': self.portfolio_history, # This now contains close price and benchmark value
            'trades': self.trades,
            'drawdown_series': drawdown_series,
            'max_drawdown_percent': max_drawdown_percent,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'up_market_capture': up_market_capture,
            'down_market_capture': down_market_capture,
            'strategy_execution_data': strategy_execution_data # This contains signals and all indicators
        }

        return results