# visualizations/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_portfolio_performance(portfolio_history, ticker):
    """
    Plots the total portfolio value over time (Equity Curve).
    """
    if portfolio_history.empty or 'total_value' not in portfolio_history.columns:
        print(f"No portfolio history data to plot for {ticker} performance.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_history.index, portfolio_history['total_value'], label='Total Portfolio Value', color='blue')
    plt.title(f'{ticker} Portfolio Performance (Equity Curve)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_drawdowns(drawdown_series, max_drawdown_percent, ticker):
    """
    Plots the portfolio drawdowns over time.
    """
    if drawdown_series.empty:
        print(f"No drawdown data to plot for {ticker}.")
        return

    plt.figure(figsize=(14, 7))
    plt.fill_between(drawdown_series.index, 0, drawdown_series, color='red', alpha=0.3)
    plt.title(f'{ticker} Portfolio Drawdown (Max Drawdown: {max_drawdown_percent:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()


def plot_trades_on_price_chart(historical_data, trades, ticker):
    """
    Plots the asset's close price and marks buy/sell trades.
    Assumes historical_data has 'close', and trades has 'Timestamp', 'Type', 'Price'.
    """
    if historical_data.empty or 'close' not in historical_data.columns:
        print(f"No historical data to plot for {ticker} price chart.")
        return
    if trades.empty:
        print(f"No trades to plot for {ticker}.")

    plt.figure(figsize=(14, 7))
    plt.plot(historical_data.index, historical_data['close'], label=f'{ticker} Close Price', alpha=0.7)

    buy_trades = trades[trades['Type'] == 'BUY']
    sell_trades = trades[trades['Type'] == 'SELL']

    if not buy_trades.empty:
        plt.scatter(buy_trades['Timestamp'], buy_trades['Price'], marker='^', color='green', s=100, label='Buy Signal',
                    alpha=1.0)
    if not sell_trades.empty:
        plt.scatter(sell_trades['Timestamp'], sell_trades['Price'], marker='v', color='red', s=100, label='Sell Signal',
                    alpha=1.0)

    # Optional: Plotting MAs if they exist in historical_data (assuming strategy adds them)
    if 'short_ma' in historical_data.columns and 'long_ma' in historical_data.columns:
        plt.plot(historical_data.index, historical_data['short_ma'], label='Short MA', color='orange', linestyle='--')
        plt.plot(historical_data.index, historical_data['long_ma'], label='Long MA', color='purple', linestyle='--')

    plt.title(f'{ticker} Price Chart with Buy/Sell Signals and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_cumulative_returns_vs_benchmark(portfolio_history, ticker):
    """
    Plots the cumulative returns of the strategy vs. a benchmark.
    Assumes 'total_value' and 'benchmark_value' are in portfolio_history,
    and both start from the same initial capital.
    """
    if portfolio_history.empty or 'total_value' not in portfolio_history.columns or 'benchmark_value' not in portfolio_history.columns:
        print(f"Insufficient data to plot cumulative returns vs. benchmark for {ticker}.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_history.index, portfolio_history['total_value'], label='Strategy Equity Curve', color='blue')
    plt.plot(portfolio_history.index, portfolio_history['benchmark_value'], label='Benchmark (Buy & Hold)',
             color='orange', linestyle='--')
    plt.title(f'{ticker} Cumulative Returns: Strategy vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_rolling_metrics(portfolio_history, ticker, window=30):
    """
    Plots rolling Sharpe Ratio and Rolling Volatility of daily returns.
    Assumes 'daily_returns' is in portfolio_history.
    """
    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns:
        print(f"Insufficient data for rolling metrics for {ticker}.")
        return

    daily_returns = portfolio_history['daily_returns']

    # Rolling Sharpe Ratio (using risk-free rate = 0)
    # Annualization factor of 365 for crypto daily data
    rolling_sharpe = (daily_returns.rolling(window=window).mean() /
                      daily_returns.rolling(window=window).std()) * np.sqrt(365)

    # Rolling Volatility (annualized standard deviation)
    rolling_volatility = daily_returns.rolling(window=window).std() * np.sqrt(365)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(rolling_sharpe.index, rolling_sharpe, label=f'Rolling Sharpe Ratio ({window}-day)', color='green')
    ax1.set_title(f'{ticker} Rolling Sharpe Ratio')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(rolling_volatility.index, rolling_volatility, label=f'Rolling Volatility ({window}-day)', color='purple')
    ax2.set_title(f'{ticker} Rolling Volatility')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Annualized Volatility')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()


def plot_returns_distribution(portfolio_history, ticker, bins=50):
    """
    Plots a histogram of daily/period returns.
    Assumes 'daily_returns' is in portfolio_history.
    """
    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns:
        print(f"No daily returns data to plot distribution for {ticker}.")
        return

    daily_returns = portfolio_history['daily_returns']
    if daily_returns.sum() == 0 and len(daily_returns) > 0:  # Check if all returns are zero
        print(f"All daily returns are zero for {ticker}, cannot plot distribution.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(daily_returns, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'{ticker} Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()


def plot_pnl_per_trade_distribution(trades, ticker, bins=50):
    """
    Plots a histogram of PnL per completed trade.
    Assumes 'PnL' is in trades DataFrame for 'SELL' trades.
    """
    completed_trades_pnl = trades[trades['Type'] == 'SELL']['PnL']
    if completed_trades_pnl.empty:
        print(f"No completed trades PnL data to plot distribution for {ticker}.")
        return

    if completed_trades_pnl.sum() == 0 and len(completed_trades_pnl) > 0:  # Check if all PnL are zero
        print(f"All PnL per trade are zero for {ticker}, cannot plot distribution.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(completed_trades_pnl, bins=bins, color='lightgreen', edgecolor='black')
    plt.title(f'{ticker} Distribution of PnL per Completed Trade')
    plt.xlabel('PnL per Trade ($)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()