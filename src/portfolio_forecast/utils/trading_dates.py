import re

import pandas as pd
# Exchange calendar with NYSE holidays, special closures, and early closes
import exchange_calendars as xcals
from dateutil.tz import tzlocal


def _parse_interval(interval):
    """
    Turn a bar size into (kind, step):
      ('intraday', Timedelta) for bars shorter than a day
      ('session', n)          for n-day bars, counted in trading sessions
      ('week', n) / ('month', n)
    Accepts bar-size strings ('30 secs', '5 mins', '1 hour', '1 day', '1 week', '1 month'),
    pandas-style strings ('5min', '1h', '1D') or a Timedelta.
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

    # 'M' alone is pandas' month; otherwise match on the unit's leading letters
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


def _infer_interval(dates, cal):
    """Work out the bar size from the spacing of the input timestamps."""
    if len(dates) < 2:
        raise ValueError('Need at least two dates to infer the interval; pass interval= explicitly')

    # Intraday: some timestamp carries a time of day. The most common gap is the bar size
    # (the smallest gap can be a partial bar, and the largest spans nights/weekends)
    if not (dates == dates.normalize()).all():
        return 'intraday', dates.to_series().diff().dropna().mode().min()

    # Daily or longer: judge by the typical calendar-day gap
    day_gap = dates.to_series().diff().dropna().dt.days.median()
    if 5 <= day_gap <= 9:
        return 'week', 1
    if day_gap >= 25:
        return 'month', max(1, round(day_gap / 30.44))

    # n-day bars: count the gap in trading sessions rather than calendar days
    positions = cal.sessions.searchsorted(dates.tz_localize(None))
    return 'session', max(1, int(pd.Series(positions).diff().dropna().mode().min()))


def _next_sessions(dates, n_periods, step, cal):
    """Every step-th trading session after the last date."""
    last = dates[-1].tz_localize(None)
    after = cal.sessions[cal.sessions > last]
    # If the last date is itself a session, the next bar is `step` sessions on from it;
    # otherwise the first session after it counts as step one
    offset = step - 1 if last in cal.sessions else 0
    return after[offset::step][:n_periods]


def _next_periodic(dates, n_periods, kind, step, cal):
    """Weekly or monthly bars, labeled the same way the input labels them."""
    freq = 'W-SUN' if kind == 'week' else 'M'
    sessions = pd.Series(cal.sessions, index=cal.sessions.to_period(freq))
    firsts, lasts = sessions.groupby(level=0).min(), sessions.groupby(level=0).max()

    # Learn the labeling convention: is each bar stamped with the first or the last
    # session of its week/month? (A one-session period counts as both.)
    labels = dates.tz_localize(None)
    periods = labels.to_period(freq)
    first_votes = sum(firsts.get(p) == d for p, d in zip(periods, labels))
    last_votes = sum(lasts.get(p) == d for p, d in zip(periods, labels))
    anchor = firsts if first_votes >= last_votes else lasts

    # Periods after the last bar's, taking every step-th one
    future = anchor[anchor.index > periods[-1]]
    return pd.DatetimeIndex(future.iloc[step - 1::step].iloc[:n_periods].values)


def _next_intraday(dates, n_periods, step, cal):
    """
    Intraday bars. Everything is done in exchange wall-clock time (naive) so daylight saving
    doesn't shift the grid. The session layout is learned from the input:
      - the grid phase: where bars fall relative to midnight (e.g. hourly bars on :00)
      - whether the first bar sits at the open even when the open is off-grid (e.g. 9:30)
      - how far before the open and after the close bars run (extended hours)
    """
    wall = dates.tz_convert(cal.tz).tz_localize(None)
    df = pd.DataFrame({'t': wall, 'session': wall.normalize()})
    df = df[df['session'].isin(cal.sessions)]

    opens = cal.opens.dt.tz_convert(cal.tz).dt.tz_localize(None)
    closes = cal.closes.dt.tz_convert(cal.tz).dt.tz_localize(None)
    df['open'] = df['session'].map(opens)
    df['close'] = df['session'].map(closes)

    first = df.groupby('session')['t'].transform('min') == df['t']
    since_midnight = df['t'] - df['session']

    # Grid phase from bars that aren't a session's first bar (those may be partial);
    # fall back to the first bars when that's all there is
    later = since_midnight[~first] if (~first).any() else since_midnight
    phase = (later % step).mode().min()

    # Minutes before the open that bars start (0 for regular hours only)
    pre = min((df.loc[first, 't'] - df.loc[first, 'open']).min(), pd.Timedelta(0))
    # How far past the close the latest bar starts; bars must start before the close
    # unless the data shows extended-hours bars after it
    past_close = (df['t'] - df['close']).max()
    post_limit = past_close + pd.Timedelta(1) if past_close >= pd.Timedelta(0) else pd.Timedelta(0)

    # Does the data put a bar exactly at the session start when that start is off-grid?
    starts = df.loc[first, 'open'] + pre
    off_grid = ((starts - starts.dt.normalize()) % step) != phase
    first_off = df.loc[first, 't'][off_grid]
    bar_at_start = bool((first_off == starts[off_grid]).any()) if off_grid.any() else True

    last = wall[-1]
    out = []
    for session in cal.sessions[cal.sessions >= last.normalize()]:
        start, end = opens[session] + pre, closes[session] + post_limit
        # Grid points from the first one at/after the session start
        first_grid = session + phase + step * -(-(start - session - phase) // step)
        bars = pd.date_range(first_grid, end, freq=step, inclusive='left')
        if bar_at_start and (len(bars) == 0 or bars[0] != start):
            bars = bars.insert(0, start)
        out.extend(bars[bars > last])
        if len(out) >= n_periods:
            break
    return pd.DatetimeIndex(out[:n_periods])


def next_trading_dates(dates, n_periods, interval=None, calendar='XNYS', tz=None):
    """Future trading dates or bar times that continue a series of bars.

    Used to label simulated periods with real dates: the output starts at the
    first trading period after the last input date and follows the input's
    spacing, skipping weekends, exchange holidays and closed hours.

    Parameters
    ----------
    dates : array-like of datetimes
        Historical bar dates or times, e.g. ``prices.index``. Duplicates and
        NaT are dropped and the rest sorted.
    n_periods : int
        Number of future periods to return. Zero or less returns an empty
        index.
    interval : str or pandas.Timedelta, optional
        Bar size, as a bar-size string (``'30 secs'``, ``'5 mins'``,
        ``'1 hour'``, ``'1 day'``, ``'1 week'``, ``'1 month'``), pandas spelling
        (``'5min'``, ``'1h'``, ``'1D'``) or as a Timedelta. None infers it
        from the spacing of `dates`.
    calendar : str, default 'XNYS'
        ``exchange_calendars`` calendar code; XNYS is the NYSE.
    tz : str or tzinfo, optional
        Timezone that naive intraday timestamps are in. Defaults to this
        machine's local timezone. Ignored for daily and longer bars.

    Returns
    -------
    pandas.DatetimeIndex
        `n_periods` future trading dates (daily and longer bars) or bar start
        times (intraday), naive or timezone-aware like the input.

    Raises
    ------
    ValueError
        If `dates` is empty, `interval` cannot be parsed, or `interval` is
        None and `dates` has fewer than two entries.

    Notes
    -----
    - n-day bars are counted in trading sessions, not calendar days.
    - Weekly and monthly bars keep the input's labelling convention: each
      bar stamped with the first or the last session of its week or month.
    - Intraday bars follow the session layout learned from the input: the
      grid phase (e.g. hourly bars on the hour), whether a bar sits at an
      off-grid open such as 9:30, and whether bars extend into pre- or
      post-market hours. Early closes are respected.

    Examples
    --------
    >>> sim_dates = prices.index[-1:].append(next_trading_dates(prices.index, 100))
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).dropna().unique().sort_values()
    if len(dates) == 0:
        raise ValueError('dates is empty')
    if n_periods <= 0:
        return dates[:0]

    naive = dates.tz is None
    in_tz = (tz or tzlocal()) if naive else dates.tz

    # Calendar covering the input, extended until the horizon fits (exchange_calendars only
    # builds about a year ahead by default; future holidays come from NYSE's standing rules)
    start = dates[0].tz_localize(None).normalize() - pd.Timedelta(days=10)
    base_end = dates[-1].tz_localize(None).normalize()
    extra_days = 400
    while True:
        cal = xcals.get_calendar(calendar, start=start, end=base_end + pd.Timedelta(days=extra_days))
        kind, step = _parse_interval(interval) if interval is not None else _infer_interval(dates, cal)

        if kind == 'intraday':
            # Naive intraday times are wall-clock in `in_tz`; resolve DST ambiguity by bar order,
            # or drop the ambiguous bars when the order can't settle it
            aware = dates
            if naive:
                try:
                    aware = dates.tz_localize(in_tz, ambiguous='infer', nonexistent='shift_forward')
                except Exception:
                    aware = dates.tz_localize(in_tz, ambiguous='NaT', nonexistent='shift_forward').dropna()
            result = _next_intraday(aware, n_periods, step, cal).tz_localize(cal.tz).tz_convert(in_tz)
            result = result.tz_localize(None) if naive else result
        elif kind == 'session':
            result = pd.DatetimeIndex(_next_sessions(dates, n_periods, step, cal))
            result = result if naive else result.tz_localize(dates.tz)
        else:
            result = _next_periodic(dates, n_periods, kind, step, cal)
            result = result if naive else result.tz_localize(dates.tz)

        if len(result) >= n_periods:
            return result
        extra_days *= 2
