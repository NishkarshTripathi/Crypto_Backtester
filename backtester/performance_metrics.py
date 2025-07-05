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
    if daily_returns.std() == 0:
        return 0.0  # Avoid division by zero if no volatility

    avg_daily_return = daily_returns.mean()
    sharpe_ratio = (avg_daily_return - risk_free_rate / annualization_factor) / daily_returns.std()
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

    if downside_returns.empty or downside_returns.std() == 0:
        return 0.0  # Avoid division by zero if no downside volatility or no negative returns

    avg_daily_return = daily_returns.mean()
    sortino_ratio = (avg_daily_return - risk_free_rate / annualization_factor) / downside_returns.std()
    return sortino_ratio * np.sqrt(annualization_factor)


def calculate_profit_factor(trades):
    """
    Calculates the Profit Factor (Gross Profits / Gross Losses).
    """
    if trades.empty or 'PnL' not in trades.columns:
        return 0.0

    gross_profits = trades[trades['PnL'] > 0]['PnL'].sum()
    gross_losses = trades[trades['PnL'] < 0]['PnL'].abs().sum()

    if gross_losses == 0:
        return 1.0 if gross_profits > 0 else 0.0  # If no losses, and profits > 0, consider it highly profitable
    return gross_profits / gross_losses


def calculate_expectancy(trades):
    """
    Calculates the Expectancy per trade.
    Expectancy = (Win Rate * Average Win) - (Loss Rate * Average Loss)
    """
    if trades.empty or 'PnL' not in trades.columns:
        return 0.0

    # Filter for completed trades (SELL trades with calculated PnL)
    completed_trades_pnl = trades[trades['Type'] == 'SELL']['PnL']
    if completed_trades_pnl.empty:
        return 0.0

    winning_trades_pnl = completed_trades_pnl[completed_trades_pnl > 0]
    losing_trades_pnl = completed_trades_pnl[completed_trades_pnl < 0]

    total_completed_trades = len(completed_trades_pnl)
    winning_trades_count = len(winning_trades_pnl)
    losing_trades_count = len(losing_trades_pnl)

    win_rate = winning_trades_count / total_completed_trades if total_completed_trades > 0 else 0.0
    loss_rate = losing_trades_count / total_completed_trades if total_completed_trades > 0 else 0.0

    avg_win = winning_trades_pnl.mean() if winning_trades_count > 0 else 0.0
    avg_loss = losing_trades_pnl.mean() if losing_trades_count > 0 else 0.0  # This will be a negative value

    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    return expectancy


def calculate_capture_ratios(portfolio_history, benchmark_daily_returns):
    """
    Calculates Up-Market and Down-Market Capture Ratios.

    Args:
        portfolio_history (pd.DataFrame): DataFrame containing 'daily_returns' of the strategy.
        benchmark_daily_returns (pd.Series): Daily returns of the benchmark.

    Returns:
        tuple: (up_market_capture, down_market_capture)
    """
    up_market_capture = 0.0
    down_market_capture = 0.0

    if portfolio_history.empty or 'daily_returns' not in portfolio_history.columns or benchmark_daily_returns.empty:
        return up_market_capture, down_market_capture

    # Align strategy and benchmark returns by their common index
    common_index = portfolio_history.index.intersection(benchmark_daily_returns.index)
    strat_returns = portfolio_history['daily_returns'].loc[common_index]
    bench_returns = benchmark_daily_returns.loc[common_index]

    if strat_returns.empty or bench_returns.empty:
        return up_market_capture, down_market_capture

    # Up-market days are when benchmark returns are positive
    up_market_days = bench_returns > 0
    # Down-market days are when benchmark returns are negative
    down_market_days = bench_returns < 0

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
            down_market_capture = -np.inf if strat_cum_down_return < 0 else np.inf  # Strategy moved while benchmark was flat/zero
        # else both are 0, down_market_capture remains 0.0

    return up_market_capture, down_market_capture