# visualizations/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ... (other functions) ...

def plot_trades_on_price_chart(data_to_plot, trades, ticker):
    """
    Plots the close price, strategy-specific indicators, and trade signals.

    Args:
        data_to_plot (pd.DataFrame): DataFrame with 'close', 'final_signal',
                                     and all indicator columns (e.g., short_ma, long_ma,
                                     middle_band, upper_band, lower_band, predicted_price).
        trades (pd.DataFrame): DataFrame containing executed trades.
        ticker (str): The cryptocurrency ticker.
    """
    if data_to_plot.empty:
        print(f"No data to plot trades on price chart for {ticker}.")
        return

    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot Close Price
    ax.plot(data_to_plot.index, data_to_plot['close'], label='Close Price', color='gray', alpha=0.7)

    # Plot Strategy Indicators dynamically
    if 'short_ma' in data_to_plot.columns and 'long_ma' in data_to_plot.columns:
        ax.plot(data_to_plot.index, data_to_plot['short_ma'], label='Short MA', color='orange')
        ax.plot(data_to_plot.index, data_to_plot['long_ma'], label='Long MA', color='purple')
    if 'middle_band' in data_to_plot.columns:
        ax.plot(data_to_plot.index, data_to_plot['middle_band'], label='Middle Band (SMA)', color='blue', linestyle='--')
        ax.plot(data_to_plot.index, data_to_plot['upper_band'], label='Upper Band', color='red', linestyle='--')
        ax.plot(data_to_plot.index, data_to_plot['lower_band'], label='Lower Band', color='green', linestyle='--')
    if 'predicted_price' in data_to_plot.columns: # <--- NEW: Plot Predicted Price
        ax.plot(data_to_plot.index, data_to_plot['predicted_price'], label='ARIMA Predicted Price', color='cyan', linestyle=':')


    # Plot Buy and Sell Signals (from final_signal in strategy_execution_data)
    buy_signals = data_to_plot[data_to_plot['final_signal'] == 1]
    sell_signals = data_to_plot[data_to_plot['final_signal'] == -1]

    ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal', alpha=1)
    ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal', alpha=1)

    # Plot Actual Trades (from trades DataFrame)
    # Filter for 'BUY' and 'SELL' trades for plotting
    buy_trades = trades[trades['Type'] == 'BUY']
    sell_trades = trades[trades['Type'] == 'SELL']

    # Use the 'Price' column from the trades DataFrame for plotting trade execution points
    ax.scatter(buy_trades['Timestamp'], buy_trades['Price'], marker='^', color='darkgreen', s=150, label='Actual Buy Trade', alpha=0.7, edgecolors='black')
    ax.scatter(sell_trades['Timestamp'], sell_trades['Price'], marker='v', color='darkred', s=150, label='Actual Sell Trade', alpha=0.7, edgecolors='black')


    ax.set_title(f'{ticker} Price Chart with Strategy Signals and Trades')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

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

def plot_cumulative_returns_vs_benchmark(portfolio_history, ticker):
    """
    Plots the cumulative returns of the strategy against a simple buy-and-hold benchmark.
    """
    if portfolio_history.empty or 'total_value' not in portfolio_history.columns or 'benchmark_value' not in portfolio_history.columns:
        print(f"Not enough data to plot cumulative returns vs benchmark for {ticker}.")
        return

    # Normalize to initial capital for cumulative returns percentage
    strategy_returns = (portfolio_history['total_value'] / portfolio_history['total_value'].iloc[0] - 1) * 100
    benchmark_returns = (portfolio_history['benchmark_value'] / portfolio_history['benchmark_value'].iloc[0] - 1) * 100

    plt.figure(figsize=(14, 7))
    plt.plot(strategy_returns.index, strategy_returns, label='Strategy Cumulative Returns', color='purple')
    plt.plot(benchmark_returns.index, benchmark_returns, label='Benchmark Cumulative Returns', color='gray', linestyle='--')
    plt.title(f'{ticker} Cumulative Returns vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_rolling_metrics(portfolio_history, ticker, window=30):
    """
    Plots rolling Sharpe Ratio and Volatility.
    """
    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns:
        print(f"No daily returns data for rolling metrics plot for {ticker}.")
        return

    # Ensure there's enough data for the rolling window
    if len(portfolio_history) < window:
        print(f"Not enough data points ({len(portfolio_history)}) for a {window}-day rolling window for {ticker}.")
        return

    rolling_sharpe = portfolio_history['daily_returns'].rolling(window=window).apply(
        lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(365), raw=True
    )
    rolling_volatility = portfolio_history['daily_returns'].rolling(window=window).std() * np.sqrt(365)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(rolling_sharpe.index, rolling_sharpe, label=f'Rolling {window}-Day Sharpe Ratio', color='darkblue')
    ax1.set_title(f'{ticker} Rolling Sharpe Ratio (Window: {window} Days)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(rolling_volatility.index, rolling_volatility, label=f'Rolling {window}-Day Volatility', color='darkred')
    ax2.set_title(f'{ticker} Rolling Volatility (Window: {window} Days)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (Annualized)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()


def plot_returns_distribution(portfolio_history, ticker, bins=50):
    """
    Plots a histogram of daily returns.
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
    plt.hist(completed_trades_pnl, bins=bins, color='lightcoral', edgecolor='black')
    plt.title(f'{ticker} Distribution of PnL per Completed Trade')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()