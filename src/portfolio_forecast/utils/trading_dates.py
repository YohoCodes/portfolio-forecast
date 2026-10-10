"""`next_trading_dates`: the future marks that continue a history.

This is the single source of future marks, for the simulators and the
fan charts. It is a thin wrapper over the marks engine:

1. `next_trading_dates` reads the history's labels (the index of a Series
   or DataFrame, the values of a `BacktestResult`, or the labels
   themselves), drops NaT and duplicates and sorts them.
2. `marks._future_layout` continues the history's grid on the calendar
   and returns the marks with each return's class and length.
"""
import pandas as pd

from ..types import BacktestResult, FutureLayout
from .marks import _future_layout


def next_trading_dates(values, n, interval=None, calendar='XNYS', tz=None) -> FutureLayout:
    """Return the future marks that continue a history on the exchange
    calendar, with the class and length of each return.

    Parameters
    ----------
    values : pandas.Series, pandas.DataFrame, BacktestResult or array-like of datetimes
        The history: a value series or a frame of any number of columns
        (only its index is used, so the result is the same as passing the
        index), or its labels. Duplicates and NaT are dropped and the rest
        sorted.
    n : int
        The horizon. On intraday history, sessions; on daily and longer
        history, bars (a daily bar is a session). Zero or less gives an
        empty layout.
    interval : str or pandas.Timedelta, optional
        Bar size, as a bar-size string (``'5 mins'``, ``'1 hour'``,
        ``'1 day'``, ``'1 week'``, ``'1 month'``), pandas spelling
        (``'5min'``, ``'1h'``, ``'1D'``), yfinance spelling (``'5m'``,
        ``'1wk'``, ``'1mo'``) or a Timedelta. None infers it from the
        history.
    calendar : str or None, default 'XNYS'
        ``exchange_calendars`` calendar code; XNYS is the NYSE. None is
        continuous mode, for 24/7 data on a constant grid.
    tz : str or tzinfo, optional
        Timezone of naive intraday labels. Defaults to this machine's local
        timezone. Ignored for dates.

    Returns
    -------
    FutureLayout
        ``dates``: the future marks, labelled like the history (naive or
        timezone-aware in its timezone; dates stay dates). ``classes``: the
        class of the return ending at each mark, ``"gap"`` at a session's
        open mark and ``"regular"`` otherwise. ``tau``: each return's length
        over the bar size, 1.0 for full bars and gaps (a gap is one unit
        whatever its clock length), less for a short bar.

    Raises
    ------
    ValueError
        If `values` is empty, `interval` can't be parsed, the interval can't
        be inferred (pass it), a mark falls outside the calendar's sessions,
        `interval` is intraday but the history is dates, or the history has
        one mark per session away from the close, so it is neither daily nor
        intraday (collapse it with `to_session_values`). In continuous
        mode, if `interval` is in months, which have no fixed length.

    Notes
    -----
    - Daily and longer history is decided on the calendar: at most one
      mark per session, at its close, plus an optional initial fill mark;
      dates are daily.
    - Intraday: the history's grid (bar size and phase) is learned from its
      marks. If the history ends mid-session, that session's remaining bars
      come first and count as the first of the `n` sessions, so the
      horizon always ends at a session close. Every other session has one
      open mark (the gap) and its bar closes, on the grid, with a short last
      bar ending at the session close (half-days included).
    - A session that opens the instant the previous one closed (weeknights
      on ``"24/5"``) has no open mark and no gap: its first bar runs close to
      close and is regular.
    - Daily and longer: n-day bars step in sessions; weekly and monthly bars
      keep the history's labelling (first or last session of the period, or
      its close).
    - Continuous mode (``calendar=None``): `n` counts bars, and the
      future marks are t_T + kΔ for k = 1..n, all regular with tau 1.0. Δ
      is the history's most common spacing, or `interval` (a fixed size;
      months raise).

    Examples
    --------
    >>> layout = next_trading_dates(values, 20)
    >>> sim_dates = values.index[-1:].append(layout.dates)
    """
    if isinstance(values, BacktestResult):
        values = values.values
    # Only the labels matter, so a frame's columns (one or many) are ignored
    labels = values.index if isinstance(values, (pd.Series, pd.DataFrame)) else values
    dates = pd.DatetimeIndex(pd.to_datetime(labels)).dropna().unique().sort_values()
    if len(dates) == 0:
        raise ValueError('values is empty')
    return _future_layout(dates, n, interval, calendar, tz)
