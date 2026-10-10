"""Monte Carlo simulation of future portfolio and strategy returns.

Flow, by subpackage:

1. ``performance``: `simulate_buy_and_hold` backtests a portfolio into a
   `BacktestResult` (values at every mark, with each return's class), and
   `performance_report` measures it.
2. ``forecast``: three Monte Carlo simulators turn a value series into a
   `SimulationResult` of future paths.
3. ``performance``, ``plotting``, ``reporting``: `statistical_report`
   summarizes the paths, the fan charts draw them, and
   `compile_statistical_reports` typesets both as a PDF with LaTeX.
4. ``utils``: the calendar and marks underneath: `next_trading_dates`
   (the future marks, as a `FutureLayout`), `classify_returns`,
   `to_session_values`, `close_times` and ``PERIODS_PER_YEAR``.

The three result types are importable from the package root.
"""
from .types import BacktestResult, FutureLayout, SimulationResult

__all__ = ["BacktestResult", "FutureLayout", "SimulationResult"]
