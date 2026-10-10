"""Bar sizes: parse them, and give the periods-per-year factor that annualizes
the risk series.

Flow:

1. ``PERIODS_PER_YEAR`` tabulates periods per year for the series risk
   figures are computed on: 252 sessions, 52 weeks and 12 months.
   Intraday bars have no entry: their risk figures use session returns,
   annualized with 252, never per-bar annualization. Continuous mode
   annualizes calendar days with 365 (in `performance.returns`).
2. ``_parse_interval`` reads any accepted spelling of a bar size (IB-style
   '5 mins', pandas '5min', yfinance '5m', or a Timedelta) as a canonical
   ``(kind, step)`` pair. ``utils.trading_dates`` and the reports use it.
3. ``_PERIODS_BY_BAR`` re-keys the table by that pair, so every spelling of a
   size finds the same entry.
4. ``_periods_per_year`` parses the caller's interval and looks it up, using
   ``_describe_bar`` to name the bar size in its error message.
   `statistical_report` calls it for paths given without their marks, which
   it can only annualize as daily, weekly or monthly bars; an intraday bar
   size has no entry, since intraday risk figures need the sessions.

Conventions: a year is 252 sessions, 52 weeks or 12 months. These are
conventions, not counts of a real year: half-days, holidays and extended
hours are not modelled here.
"""
import re

import pandas as pd

# A year is 252 sessions, the annualizer of session returns and the
# session clock of CAGR
TRADING_DAYS = 252


# Periods per year of the series risk figures use, by canonical bar size (other spellings
# match by parsing); weekly and monthly bars use 52 and 12, not 252/5 or 252/21
PERIODS_PER_YEAR = {
    '1 day':    TRADING_DAYS,
    '1 week':   52,
    '1 month':  12,
}

def _parse_interval(interval):
    """Parse a bar size into a canonical ``(kind, step)`` pair.

    Parameters
    ----------
    interval : str or Timedelta
        IB-style ('30 secs', '5 mins', '1 hour', '1 day', '1 week',
        '1 month'), pandas-style ('5min', '1h', '1D'), yfinance-style ('5m',
        '1h', '1d', '1wk', '1mo'), or a ``pd.Timedelta`` or
        ``datetime.timedelta``.

    Returns
    -------
    tuple
        One of ``('intraday', Timedelta)`` for bars shorter than a day,
        ``('session', n)`` for n-day bars counted in trading sessions, or
        ``('week', n)`` / ``('month', n)``.

    Raises
    ------
    ValueError
        If `interval` is not an optional count followed by a recognized unit.

    Notes
    -----
    A whole-day Timedelta counts trading sessions, not calendar days:
    ``Timedelta(days=7)`` reads as 7 sessions, not one week.
    """
    # A Timedelta: whole days count sessions, anything shorter is intraday
    if isinstance(interval, pd.Timedelta) or hasattr(interval, 'total_seconds'):
        td = pd.Timedelta(interval)
        if td % pd.Timedelta(days=1) == pd.Timedelta(0):
            return 'session', int(td / pd.Timedelta(days=1))
        return 'intraday', td

    # A string: an optional count, then a unit
    match = re.fullmatch(r'\s*(\d+)?\s*([A-Za-z]+)\s*', str(interval))
    if not match:
        raise ValueError(f'Unrecognized interval: {interval!r}')
    n, unit = int(match.group(1) or 1), match.group(2)

    # A bare 'm' is yfinance's minute, a bare 'M' pandas' month; otherwise match the
    # unit's leading letters, any case: 'mo' is a month, 'mi' or pandas' 't' a minute
    if unit == 'm':
        return 'intraday', pd.Timedelta(minutes=n)
    u = unit.lower()
    if unit == 'M' or u.startswith('mo'):
        return 'month', n
    if u.startswith(('mi', 't')):
        return 'intraday', pd.Timedelta(minutes=n)
    if u.startswith('s'):
        return 'intraday', pd.Timedelta(seconds=n)
    if u.startswith('h'):
        return 'intraday', pd.Timedelta(hours=n)
    if u.startswith('d'):
        return 'session', n
    if u.startswith('w'):
        return 'week', n
    raise ValueError(f'Unrecognized interval unit: {unit!r}')


# PERIODS_PER_YEAR keyed by the parsed bar size, so any spelling that
# _parse_interval accepts ('1D', '5min', '5m', a Timedelta, ...) finds its entry
_PERIODS_BY_BAR = {_parse_interval(key): n
                   for key, n in PERIODS_PER_YEAR.items()}


def _periods_per_year(interval):
    """Return bars per year for a bar size, the factor that annualizes per-bar figures.

    Parameters
    ----------
    interval : str or Timedelta
        A ``PERIODS_PER_YEAR`` key ('1 day', '1 week', '1 month'), or any
        other spelling ``_parse_interval`` accepts ('1D', '1wk', '1mo').

    Returns
    -------
    int
        The table entry for that bar size: 252, 52 or 12.

    Raises
    ------
    ValueError
        If `interval` can't be parsed, or reads as a bar size with no entry
        (intraday bars, which are never annualized per bar, or 3-day
        bars); the message lists the supported keys.
    """
    # Parse, then look up; either failure names the supported keys
    expected = f"Expected one of: {', '.join(PERIODS_PER_YEAR)}"
    try:
        bar = _parse_interval(interval)
    except ValueError as e:
        raise ValueError(
            f"Unsupported interval {interval!r}: {e}. {expected}"
        ) from None
    if bar not in _PERIODS_BY_BAR:
        raise ValueError(
            f"Unsupported interval {interval!r}: it reads as {_describe_bar(bar)} "
            f"bars, which have no bars-per-year entry. {expected}"
        )
    return _PERIODS_BY_BAR[bar]


def _describe_bar(bar):
    """Return a parsed bar size in words, e.g. ``('intraday', 90 min)`` -> '90-minute'.

    Intraday steps use the largest unit (hour, minute, second) that divides
    them exactly, so 90 minutes reads '90-minute', not '1.5-hour'.
    """
    kind, step = bar
    if kind == 'intraday':
        seconds = int(step.total_seconds())
        for unit, size in (('hour', 3600), ('minute', 60), ('second', 1)):
            if seconds % size == 0:
                return f"{seconds // size}-{unit}"
    unit = {'session': 'day', 'week': 'week', 'month': 'month'}[kind]
    return f"{step}-{unit}"
