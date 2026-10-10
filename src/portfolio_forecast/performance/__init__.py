"""Backtests and their measurement, and summaries of simulated paths.

1. `simulate_buy_and_hold` (`simulate.py`) values a buy-and-hold portfolio
   at every mark.
2. `calculate_period_returns` and `calculate_session_returns`
   (`returns.py`) give the return series the metrics use.
3. The metrics (`metrics.py`) and `performance_report` (`report.py`)
   measure one series; `statistical_report` summarizes simulated paths.
"""
from .metrics import (
    calculate_actual_yearly_return,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from .report import performance_report, statistical_report
from .returns import calculate_daily_return, calculate_period_returns, calculate_session_returns
from .simulate import simulate_buy_and_hold

__all__ = ["calculate_actual_yearly_return", "calculate_cagr",
           "calculate_daily_return", "calculate_max_drawdown",
           "calculate_period_returns", "calculate_session_returns",
           "calculate_sharpe_ratio", "calculate_sortino_ratio",
           "performance_report", "simulate_buy_and_hold",
           "statistical_report"]
