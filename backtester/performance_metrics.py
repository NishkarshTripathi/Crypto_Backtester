# backtester/performance_metrics.py

import pandas as pd
import numpy as np


def calculate_drawdowns(portfolio_history):
    """
    Calculates the drawdowns from the portfolio total value history.

    Args:
        portfolio_history (pd.DataFrame): DataFrame containing 'total_value'.

    Returns:
        tuple: (drawdown_series, max_drawdown_percent)
               - drawdown_series: A pandas Series with drawdown percentages.
               - max_drawdown_percent: The maximum drawdown percentage.
    """
    if portfolio_history.empty or 'total_value' not in portfolio_history.columns:
        return pd.Series(dtype=float), 0.0

    peak_equity = portfolio_history['total_value'].expanding(min_periods=1).max()
    drawdown = (portfolio_history['total_value'] - peak_equity) / peak_equity
    max_drawdown_percent = drawdown.min() * 100
    return drawdown, max_drawdown_percent


def calculate_sharpe_ratio(portfolio_history, risk_free_rate=0.0, annualization_factor=365):
    """
    Calculates the annualized Sharpe Ratio.
    Assumes daily_returns are already calculated in portfolio_history.
    """
    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns:
        return 0.0

    daily_returns = portfolio_history['daily_returns']
    # Ensure there's enough data and variability to calculate std dev
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0

    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    sharpe_ratio = (avg_daily_return - risk_free_rate / annualization_factor) / std_daily_return
    return sharpe_ratio * np.sqrt(annualization_factor)


def calculate_sortino_ratio(portfolio_history, risk_free_rate=0.0, annualization_factor=365):
    """
    Calculates the annualized Sortino Ratio.
    Assumes daily_returns are already calculated in portfolio_history.
    """
    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns:
        return 0.0

    daily_returns = portfolio_history['daily_returns']
    downside_returns = daily_returns[daily_returns < 0]

    if len(downside_returns) < 2 or downside_returns.std() == 0:
        # If no downside returns or no variability in downside returns
        return 0.0

    avg_daily_return = daily_returns.mean()
    downside_std_dev = downside_returns.std()

    sortino_ratio = (avg_daily_return - risk_free_rate / annualization_factor) / downside_std_dev
    return sortino_ratio * np.sqrt(annualization_factor)


def calculate_profit_factor(trades):
    """
    Calculates the Profit Factor.
    Profit Factor = Gross Profit / Gross Loss
    """
    if trades.empty:
        return 0.0

    gross_profit = trades[trades['PnL'] > 0]['PnL'].sum()
    gross_loss = trades[trades['PnL'] < 0]['PnL'].sum()  # PnL is already negative for losses

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0  # Avoid division by zero

    return abs(gross_profit / gross_loss)


def calculate_expectancy(trades):
    """
    Calculates the Expectancy of the trading system.
    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    """
    if trades.empty:
        return 0.0

    winning_trades = trades[trades['PnL'] > 0]['PnL']
    losing_trades = trades[trades['PnL'] < 0]['PnL']

    total_trades = len(trades[trades['Type'] == 'SELL'])  # Consider only completed trades for expectancy

    if total_trades == 0:
        return 0.0

    win_rate = len(winning_trades) / total_trades
    loss_rate = len(losing_trades) / total_trades

    avg_win = winning_trades.mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades.mean() if not losing_trades.empty else 0.0  # Will be negative or zero

    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)  # Add because avg_loss is already negative
    return expectancy


def calculate_capture_ratios(portfolio_history, benchmark_daily_returns):
    """
    Calculates Up-Market Capture and Down-Market Capture ratios.

    Args:
        portfolio_history (pd.DataFrame): DataFrame containing 'daily_returns'.
        benchmark_daily_returns (pd.Series): Series of benchmark daily returns.

    Returns:
        tuple: (up_market_capture, down_market_capture)
    """
    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns or benchmark_daily_returns.empty:
        return 0.0, 0.0

    # Align the returns based on their index (timestamps)
    aligned_returns = pd.DataFrame({
        'strat_returns': portfolio_history['daily_returns'],
        'bench_returns': benchmark_daily_returns
    }).dropna()

    if aligned_returns.empty:
        return 0.0, 0.0

    strat_returns = aligned_returns['strat_returns']
    bench_returns = aligned_returns['bench_returns']

    up_market_days = bench_returns > 0
    down_market_days = bench_returns < 0

    up_market_capture = 0.0
    down_market_capture = 0.0

    # Up-Market Capture
    if up_market_days.any():
        strat_up_returns = strat_returns[up_market_days]
        bench_up_returns = bench_returns[up_market_days]

        # Calculate cumulative returns for strategy and benchmark on up days
        strat_cum_up_return = (1 + strat_up_returns).prod() - 1
        bench_cum_up_return = (1 + bench_up_returns).prod() - 1

        if bench_cum_up_return != 0:
            up_market_capture = (strat_cum_up_return / bench_cum_up_return) * 100
        elif strat_cum_up_return != 0:
            up_market_capture = np.inf  # Strategy performed while benchmark was flat
        # else both are 0, up_market_capture remains 0.0

    # Down-Market Capture
    if down_market_days.any():
        strat_down_returns = strat_returns[down_market_days]
        bench_down_returns = bench_returns[down_market_days]

        # Calculate cumulative returns for strategy and benchmark on down days
        strat_cum_down_return = (1 + strat_down_returns).prod() - 1
        bench_cum_down_return = (1 + bench_down_returns).prod() - 1

        if bench_cum_down_return != 0:
            down_market_capture = (strat_cum_down_return / bench_cum_down_return) * 100
        elif strat_cum_down_return != 0:
            down_market_capture = np.inf  # Strategy lost while benchmark was flat (not good)
        # else both are 0, down_market_capture remains 0.0

    return up_market_capture, down_market_capture