"""Return series of a value series: per mark, and per session for the risk
figures.

Flow:

1. `calculate_period_returns` gives the simple return between consecutive
   marks (`calculate_daily_return` is its deprecated old name).
2. `calculate_session_returns` collapses the series to one value per
   session (`utils.to_session_values`) and gives the session returns R_d
   behind volatility, Sharpe and Sortino, leaving out a partial first
   session. Daily and longer series use their own bar returns; in
   continuous mode the sessions are UTC days.
3. `_risk_periods_per_year` gives the factor that annualizes them: 252
   sessions, 52 weeks, 12 months, or 365 days in continuous mode.

All returns are simple, r = V_t / V_{t-1} - 1, as fractions, labelled with
the mark where they end.
"""
import warnings

import numpy as np
import pandas as pd

from ..types import BacktestResult
from ..utils.calendar import _check_increasing, _continuous_step
from ..utils.marks import (
    DAILY,
    _daily_bar_size,
    _first_mark_at_close,
    _labels_of,
    _series_kind,
    _session_values,
)
from ..utils.periods import TRADING_DAYS, _parse_interval

# Continuous-mode risk figures annualize calendar days
CALENDAR_DAYS = 365

_ONE_DAY = pd.Timedelta(days=1)

# Bars per year of the risk series, by parsed bar size
_RISK_PERIODS = {("session", 1): TRADING_DAYS, ("week", 1): 52, ("month", 1): 12}


def _values_of(values):
    """Return the value Series of a Series or BacktestResult."""
    return values.values if isinstance(values, BacktestResult) else values


def calculate_period_returns(values):
    """Return the simple returns between consecutive marks of a value series.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values, oldest first, one per mark.

    Returns
    -------
    pandas.Series
        r_t = V_t / V_{t-1} - 1 for each mark after the first, labelled with
        the mark where it ends, so always len(values) - 1 rows. A return
        across a missing value is measured from the last known value.
        Returns after a total loss are 0/0, undefined, and kept as NaN;
        the -100% return itself is an ordinary -1.0.

    See Also
    --------
    calculate_session_returns : The per-session series behind the risk figures.
    """
    values = _values_of(values)
    # Returns from the last known value (ffill; pct_change's own filling is deprecated), less
    # the first mark; 0/0 after a total loss stays NaN, so there are len(values) - 1 rows
    return values.ffill().pct_change(fill_method=None).iloc[1:]


def calculate_daily_return(values):
    """Return the simple returns between consecutive marks (deprecated).

    .. deprecated:: 0.4.0
        Renamed `calculate_period_returns`, since the returns are per mark,
        not per day; this name is removed in 1.0.0.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values, oldest first.

    Returns
    -------
    pandas.Series
        The same as `calculate_period_returns`.

    Warns
    -----
    DeprecationWarning
        Always, naming `calculate_period_returns`.
    """
    warnings.warn(
        "calculate_daily_return is deprecated and will be removed in 1.0.0; use calculate_period_returns, "
        "or calculate_session_returns for the per-session series",
        DeprecationWarning, stacklevel=2,
    )
    return calculate_period_returns(values)


def calculate_session_returns(values, calendar="XNYS"):
    """Return the session returns behind volatility, Sharpe and Sortino.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values, oldest first. Intraday labels must be
        timezone-aware.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.

    Returns
    -------
    pandas.Series
        One simple return per full session, labelled with the
        session's last mark: R_d = V(C_d) / V(C_{d-1}) - 1, which on intraday data is
        (1 + g_d) Π_k (1 + r_{d,k}) - 1. Daily, weekly and monthly series:
        their own bar returns. Continuous mode: calendar-day returns
        R_t = V(D_t) / V(D_{t-1}) - 1, D_t the last mark at or before 00:00
        UTC of day t, labelled at that midnight; a series with a grid step
        of a day or more uses its own bar returns.

    Raises
    ------
    ValueError
        If the labels can't be placed on the calendar (see
        `portfolio_forecast.utils.to_session_values`).

    Notes
    -----
    - Partial first session: the first return R_1 is left out when it is partial, that is when
      the series does not start at a session close: always on intraday
      data, and on daily or longer data bought at the open. Daily data
      bought at the close (or dated) keeps it. So the series is the same
      under both fills. In continuous mode the first UTC day is always
      left out; a continuous series of daily or longer bars keeps
      every return, since the fill can't be read from its labels.
    - Total loss: returns after a total loss are 0/0, undefined, and kept as NaN,
      so the length depends only on the labels and the calendar; the risk
      figures drop NaN before computing.

    See Also
    --------
    calculate_period_returns : Returns between consecutive marks.
    portfolio_forecast.utils.to_session_values : The values these returns come from.
    """
    values = _values_of(values)
    index = _labels_of(values)
    if len(index) < 2:
        return values.iloc[:0].astype(float)
    # Where the session values sit in the series, their labels, and whether R_1 is partial
    positions, labels, drop_first = _session_positions(index, calendar)
    sessions = pd.Series(values.to_numpy()[positions], index=labels, name=values.name)
    returns = calculate_period_returns(sessions)
    # Risk figures use full sessions only, so a partial R_1 goes
    if drop_first:
        returns = returns.iloc[1:]
    return returns


def _session_positions(index, calendar="XNYS", drop_live=True):
    """Return where the session returns of a series of marks come from.

    `index` holds at least two mark labels. Returns ``(positions, labels,
    drop_first)``: the positions in `index` of the session values (the
    first mark, then each session's last mark, as `to_session_values`
    picks them; every mark for daily and longer series, and for continuous
    bars of a day or more), the labels those values carry (a UTC midnight
    in continuous mode), and whether the first session return is partial
    and left out. `statistical_report` applies the same positions to
    every simulated path, so its figures follow these rules exactly; it
    passes `drop_live` False, since future marks have no session in
    progress, while a history's last session still trading is left out.
    """
    if calendar is None:
        _check_increasing(index, "mark labels")
        if _continuous_step(index) >= _ONE_DAY:
            # Daily and longer bars use their own returns, all kept
            return np.arange(len(index)), index, False
    # The positions ride through to_session_values as the values, so they
    # are picked (and relabelled) exactly as the values would be; its
    # short-session warning is left out, since only the positions are used
    picked, _ = _session_values(pd.Series(np.arange(len(index), dtype=float), index=index), calendar,
                                drop_live=drop_live)
    # A series starting at a session close keeps R_1; in continuous mode the first UTC
    # day always goes, even from 00:00 UTC, so both fills give the same series
    full_first = False if calendar is None else _first_mark_at_close(picked.index, calendar)
    return picked.to_numpy().astype(int), picked.index, not full_first


def _risk_periods_per_year(values, calendar="XNYS", interval=None):
    """Return the factor that annualizes the session returns:
    252 for intraday and daily series, 52 for weekly, 12 for monthly, 365
    in continuous mode (365 d / Δ for a grid step Δ longer than a day).

    `interval` (any spelling `_parse_interval` reads) overrides the bar size
    inferred from a daily-or-longer series. Raises ValueError for bar sizes
    with no annualization factor (3-day or quarterly bars), and for an
    intraday `interval` on a daily-or-longer series.
    """
    index = _labels_of(_values_of(values))
    if calendar is None:
        # Bars longer than a day annualize their own returns, 365 d / Δ
        step = _continuous_step(index) if len(index) > 1 else _ONE_DAY
        return CALENDAR_DAYS * _ONE_DAY / step if step > _ONE_DAY else CALENDAR_DAYS
    # Intraday series annualize their session returns with 252
    if _series_kind(index, calendar) != DAILY:
        return TRADING_DAYS
    # Daily and longer: the given bar size, or the one inferred from the bar closes
    if interval is not None:
        bar = _parse_interval(interval)
        if bar[0] == "intraday":
            raise ValueError(f"interval {interval!r} is intraday, but values is a daily or longer series")
    else:
        bar = _daily_bar_size(index, calendar)
    if bar not in _RISK_PERIODS:
        raise ValueError(
            f"No annualization factor for {bar[1]}-{bar[0]} bars; risk figures support daily, weekly "
            "and monthly bars (252, 52 and 12 a year)"
        )
    return _RISK_PERIODS[bar]
