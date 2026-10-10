"""Calendars, marks and bar sizes shared by the backtest and the simulators.

`next_trading_dates` (`trading_dates.py`) lays out future marks;
`classify_returns` and `to_session_values` (`marks.py`) class and collapse
value series; `close_times` (`calendar.py`) gives bar close times;
``PERIODS_PER_YEAR`` (`periods.py`) annualizes risk figures.
"""
from .calendar import close_times
from .marks import classify_returns, to_session_values
from .periods import PERIODS_PER_YEAR
from .trading_dates import next_trading_dates

__all__ = ["PERIODS_PER_YEAR", "classify_returns", "close_times", "next_trading_dates", "to_session_values"]
