# backtester/backtesting_engine.py

import pandas as pd
import numpy as np
from . import performance_metrics  # Import the new performance_metrics module


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
        # Generate signals from the strategy
        data_with_signals = self.strategy.generate_signals()

        # Ensure the index is a DatetimeIndex and its name is 'timestamp' for consistency
        if not isinstance(data_with_signals.index, pd.DatetimeIndex):
            data_with_signals.index = pd.to_datetime(data_with_signals.index)
        data_with_signals.index.name = 'timestamp'

        # Initialize portfolio for ALL rows upfront.
        self.portfolio_history['cash'] = 0.0
        self.portfolio_history['units_held'] = 0.0
        self.portfolio_history['holdings_value'] = 0.0
        self.portfolio_history['total_value'] = 0.0
        self.portfolio_history['short_ma'] = np.nan
        self.portfolio_history['long_ma'] = np.nan

        # Set initial capital for the first timestamp in the portfolio history
        if not self.portfolio_history.empty:
            self.portfolio_history.loc[self.portfolio_history.index[0], 'cash'] = self.initial_capital
            self.portfolio_history.loc[self.portfolio_history.index[0], 'total_value'] = self.initial_capital
        else:
            print("Error: Data feed is empty, cannot initialize portfolio history.")
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_pnl': 0.0,
                'total_returns_percent': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_percent': 0.0,
                'avg_pnl_per_trade': 0.0,
                'portfolio_history': self.portfolio_history,
                'trades': self.trades,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'up_market_capture': 0.0,
                'down_market_capture': 0.0,
                'drawdown_series': pd.Series(dtype=float),
                'max_drawdown_percent': 0.0
            }

        # Loop through data points, starting from the second element
        for i in range(1, len(data_with_signals)):
            current_timestamp = data_with_signals.index[i]
            prev_timestamp = data_with_signals.index[i - 1]

            trade_price = data_with_signals['close'].loc[current_timestamp]
            signal = data_with_signals['final_signal'].loc[current_timestamp]

            # Carry forward previous period's portfolio state
            self.portfolio_history.loc[current_timestamp, 'cash'] = self.portfolio_history.loc[prev_timestamp, 'cash']
            self.portfolio_history.loc[current_timestamp, 'units_held'] = self.portfolio_history.loc[
                prev_timestamp, 'units_held']
            self.portfolio_history.loc[current_timestamp, 'short_ma'] = data_with_signals['short_ma'].loc[
                current_timestamp]
            self.portfolio_history.loc[current_timestamp, 'long_ma'] = data_with_signals['long_ma'].loc[
                current_timestamp]

            # Holdings value is always based on current price
            self.portfolio_history.loc[current_timestamp, 'holdings_value'] = \
                self.portfolio_history.loc[current_timestamp, 'units_held'] * trade_price

            # Buy Logic
            if signal == 1 and self.portfolio_history.loc[prev_timestamp, 'units_held'] == 0:
                available_cash = self.portfolio_history.loc[current_timestamp, 'cash']
                units_to_buy = available_cash / (trade_price * (1 + self.commission_rate))

                if units_to_buy >= self.min_units_threshold:
                    cost_of_units = units_to_buy * trade_price
                    commission = cost_of_units * self.commission_rate
                    total_cost = cost_of_units + commission

                    self.portfolio_history.loc[current_timestamp, 'cash'] -= total_cost
                    self.portfolio_history.loc[current_timestamp, 'units_held'] += units_to_buy
                    self.portfolio_history.loc[current_timestamp, 'holdings_value'] = \
                        self.portfolio_history.loc[current_timestamp, 'units_held'] * trade_price

                    self.trades.loc[len(self.trades)] = [current_timestamp, 'BUY', trade_price, units_to_buy,
                                                         commission, 0.0]
                    # print(f"[{current_timestamp}] *** BUY EXECUTED *** Price: {trade_price:.2f}, Units: {units_to_buy:.4f}, Comm: {commission:.2f}, New Cash: {self.portfolio_history.loc[current_timestamp, 'cash']:.2f}")
                # else:
                # print(f"[{current_timestamp}] BUY Signal, but calculated units ({units_to_buy:.8f}) are below minimum threshold ({self.min_units_threshold}). Cash: {available_cash:.2f}, Price: {trade_price:.2f}")

            # Sell Logic (Close Long Position)
            elif signal == -1 and self.portfolio_history.loc[prev_timestamp, 'units_held'] > 0:
                units_to_sell = self.portfolio_history.loc[prev_timestamp, 'units_held']

                gross_proceeds = units_to_sell * trade_price
                commission = gross_proceeds * self.commission_rate
                net_proceeds = gross_proceeds - commission
                trade_pnl = 0.0

                last_buy_trade_row = self.trades[self.trades['Type'] == 'BUY'].iloc[-1:]
                if not last_buy_trade_row.empty:
                    last_buy = last_buy_trade_row.iloc[0]
                    buy_price = last_buy['Price']
                    buy_units = last_buy['Units']
                    buy_commission = last_buy['Commission']
                    trade_pnl = (units_to_sell * trade_price - commission) - (buy_units * buy_price + buy_commission)
                # else:
                # print(f"[{current_timestamp}] Warning: SELL Signal but no prior BUY trade recorded to calculate PnL. Units: {units_to_sell:.4f}. This implies an issue with trade tracking or an initial position assumed outside the backtester's scope.")

                self.portfolio_history.loc[current_timestamp, 'cash'] += net_proceeds
                self.portfolio_history.loc[current_timestamp, 'units_held'] = 0.0
                self.portfolio_history.loc[current_timestamp, 'holdings_value'] = 0.0

                self.trades.loc[len(self.trades)] = [current_timestamp, 'SELL', trade_price, units_to_sell, commission,
                                                     trade_pnl]
                # print(f"[{current_timestamp}] *** SELL EXECUTED *** Price: {trade_price:.2f}, Units: {units_to_sell:.4f}, Comm: {commission:.2f}, PnL: {trade_pnl:.2f}, New Cash: {self.portfolio_history.loc[current_timestamp, 'cash']:.2f}")

            # Update total_value regardless of trade
            self.portfolio_history.loc[current_timestamp, 'total_value'] = \
                self.portfolio_history.loc[current_timestamp, 'cash'] + \
                self.portfolio_history.loc[current_timestamp, 'holdings_value']

        # After the loop, calculate final portfolio value
        final_capital = self.portfolio_history.loc[self.portfolio_history.index[-1], 'total_value']
        total_pnl = final_capital - self.initial_capital
        total_returns_percent = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0

        # Number of trades (buy signals)
        num_trades = len(self.trades[self.trades['Type'] == 'BUY'])

        # Ensure 'PnL' column is numeric for calculations
        self.trades['PnL'] = pd.to_numeric(self.trades['PnL'], errors='coerce').fillna(0)

        # Filter trades that are 'SELL' and have a PnL calculated for final stats
        completed_trades_pnl = self.trades[self.trades['Type'] == 'SELL']['PnL']
        winning_trades_count = len(completed_trades_pnl[completed_trades_pnl > 0])
        losing_trades_count = len(completed_trades_pnl[completed_trades_pnl < 0])
        total_completed_trades = len(completed_trades_pnl)

        win_rate_percent = (winning_trades_count / total_completed_trades) * 100 if total_completed_trades > 0 else 0.0
        avg_pnl_per_trade = completed_trades_pnl.sum() / total_completed_trades if total_completed_trades > 0 else 0.0

        # --- Calculate Metrics using performance_metrics.py ---
        self.portfolio_history['daily_returns'] = self.portfolio_history['total_value'].pct_change().fillna(0)

        # Prepare benchmark daily returns (using close price of the asset)
        benchmark_daily_returns = self.data_feed['close'].pct_change().fillna(0)
        # Add normalized benchmark value to portfolio_history for plotting
        self.portfolio_history['benchmark_value'] = (1 + benchmark_daily_returns).cumprod() * self.initial_capital

        drawdown_series, max_drawdown_percent = performance_metrics.calculate_drawdowns(self.portfolio_history)
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
            'portfolio_history': self.portfolio_history,
            'trades': self.trades,
            'drawdown_series': drawdown_series,
            'max_drawdown_percent': max_drawdown_percent,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'up_market_capture': up_market_capture,
            'down_market_capture': down_market_capture
        }
        return results