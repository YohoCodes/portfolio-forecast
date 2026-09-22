import math
import re

import pandas as pd

# US equity session: 252 trading days, 6.5 hours (390 minutes) per day
TRADING_DAYS = 252
MINUTES_PER_DAY = 6.5 * 60
SECONDS_PER_DAY = MINUTES_PER_DAY * 60


def _bars_per_year(bar_seconds):
    """Bars in a year of regular-hours sessions for a bar of `bar_seconds`.

    Bars are counted with ceil, not division: a bar size that does not divide
    the 390-minute session still opens a bar for the leftover stub. On hourly
    bars 9:30-16:00 yields 7 bars (a 30-minute stub at the open plus six full
    hours), not 6.5.
    """
    return TRADING_DAYS * math.ceil(SECONDS_PER_DAY / bar_seconds)


# Bars per year for each supported bar size, used to annualize. Keys are the
# canonical bar-size strings; other spellings of the same sizes ('1D', '5min',
# '5m', a Timedelta) are matched by parsing them (see _periods_per_year).
PERIODS_PER_YEAR = {
    '1 secs':   _bars_per_year(1),
    '5 secs':   _bars_per_year(5),
    '10 secs':  _bars_per_year(10),
    '15 secs':  _bars_per_year(15),
    '30 secs':  _bars_per_year(30),
    '1 min':    _bars_per_year(60),
    '2 mins':   _bars_per_year(2 * 60),
    '3 mins':   _bars_per_year(3 * 60),
    '5 mins':   _bars_per_year(5 * 60),
    '10 mins':  _bars_per_year(10 * 60),
    '15 mins':  _bars_per_year(15 * 60),
    '20 mins':  _bars_per_year(20 * 60),
    '30 mins':  _bars_per_year(30 * 60),
    '1 hour':   _bars_per_year(60 * 60),
    '2 hours':  _bars_per_year(2 * 60 * 60),
    '3 hours':  _bars_per_year(3 * 60 * 60),
    '4 hours':  _bars_per_year(4 * 60 * 60),
    '8 hours':  _bars_per_year(8 * 60 * 60),
    '1 day':    TRADING_DAYS,
    '1 week':   52,
    '1 month':  12,
}


def _parse_interval(interval):
    """
    Turn a bar size into (kind, step):
      ('intraday', Timedelta) for bars shorter than a day
      ('session', n)          for n-day bars, counted in trading sessions
      ('week', n) / ('month', n)
    Accepts bar-size strings ('30 secs', '5 mins', '1 hour', '1 day', '1 week', '1 month'),
    pandas-style strings ('5min', '1h', '1D'), yfinance-style strings ('5m', '1h', '1d',
    '1wk', '1mo') or a Timedelta.
    """
    if isinstance(interval, pd.Timedelta) or hasattr(interval, 'total_seconds'):
        td = pd.Timedelta(interval)
        if td % pd.Timedelta(days=1) == pd.Timedelta(0):
            return 'session', int(td / pd.Timedelta(days=1))
        return 'intraday', td

    match = re.fullmatch(r'\s*(\d+)?\s*([A-Za-z]+)\s*', str(interval))
    if not match:
        raise ValueError(f'Unrecognized interval: {interval!r}')
    n, unit = int(match.group(1) or 1), match.group(2)

    # A bare 'm' is yfinance's minute and a bare 'M' pandas' month; otherwise
    # match on the unit's leading letters
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
_PERIODS_BY_BAR = {_parse_interval(key): n for key, n in PERIODS_PER_YEAR.items()}


def _periods_per_year(interval):
    """Bars per year for `interval`, as a PERIODS_PER_YEAR key ('5 mins',
    '1 day') or any other spelling _parse_interval accepts ('5min', '5m')."""
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
    """A parsed bar size in words, e.g. ('intraday', 90 min) -> '90-minute'."""
    kind, step = bar
    if kind == 'intraday':
        seconds = int(step.total_seconds())
        for unit, size in (('hour', 3600), ('minute', 60), ('second', 1)):
            if seconds % size == 0:
                return f"{seconds // size}-{unit}"
    unit = {'session': 'day', 'week': 'week', 'month': 'month'}[kind]
    return f"{step}-{unit}"
