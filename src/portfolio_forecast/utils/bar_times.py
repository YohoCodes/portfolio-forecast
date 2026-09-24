import exchange_calendars as xcals
import pandas as pd
from dateutil.tz import tzlocal


def _daily_sessions(dates, calendar='XNYS'):
    """(opens, closes) of the sessions on naive, midnight `dates`, as times on
    the exchange's clock, or None if the dates are spaced like weekly or longer
    bars, whose date doesn't say which session they end on.

    Daily bars sit on consecutive sessions, apart from the odd missing bar, so
    the dates count as daily when the most common step between them is one
    session. Raises ValueError for a date that isn't a session.
    """
    cal = xcals.get_calendar(calendar,
                             start=dates.min() - pd.Timedelta(days=7),
                             end=dates.max() + pd.Timedelta(days=7))
    session_i = cal.sessions.get_indexer(dates)
    not_session = session_i < 0
    if not_session.any():
        raise ValueError(
            f"{not_session.sum()} bar(s) fall outside {calendar} sessions, "
            f"first {dates[not_session][0]}"
        )
    if len(dates) > 1 and pd.Series(session_i).diff().dropna().mode().min() > 1:
        return None
    opens = pd.DatetimeIndex(cal.opens.reindex(dates)).tz_convert(cal.tz)
    closes = pd.DatetimeIndex(cal.closes.reindex(dates)).tz_convert(cal.tz)
    return opens.rename(dates.name), closes.rename(dates.name)


def close_times(index, calendar='XNYS', tz=None):
    """The time each bar of a start-labelled index closes.

    yfinance and Interactive Brokers label a bar with its start, but its close
    (and anything computed from it) is only known when the bar ends. This
    gives the label that says so.

    Parameters
    ----------
    index : array-like of datetimes
        Bar start times, oldest first.
    calendar : str, default 'XNYS'
        ``exchange_calendars`` calendar code for the session closes.
    tz : str or tzinfo, optional
        Timezone that naive intraday times are in. Defaults to this machine's
        local timezone. Ignored for everything else.

    Returns
    -------
    pandas.DatetimeIndex
        One close per bar. Intraday closes keep the input's timezone, or its
        naive wall clock. Daily closes are timezone-aware, on the exchange's
        clock.

    Raises
    ------
    ValueError
        If a bar falls outside the calendar's sessions.

    Notes
    -----
    - An intraday bar closes at the next point of its bar grid, and no later
      than its session's close. The grid is learned from the index: the bar
      size is the most common gap between bars after a session's first (which
      can be partial), and its phase the most common offset from midnight of
      those bars. Hourly bars from a 9:30 open close at 10:00, 11:00, ...,
      16:00, or 13:00 on a half-day. A missing bar doesn't stretch the one
      before it.
    - An intraday index with no two bars in one session closes each bar at its
      session's close.
    - Dates spaced like daily bars close at their session's close. Dates spaced
      further apart (weekly and longer bars) are returned unchanged.
    - An unfinished bar gets its scheduled close, though its value is only the
      last trade so far.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return index
    if index.tz is None:
        if (index == index.normalize()).all():
            daily = _daily_sessions(index, calendar)
            return index if daily is None else daily[1]
        aware = index.tz_localize(tz or tzlocal(), ambiguous='infer',
                                  nonexistent='shift_forward')
        return close_times(aware, calendar).tz_localize(None)

    cal = xcals.get_calendar(calendar,
                             start=index.min().tz_convert(None).normalize() - pd.Timedelta(days=7),
                             end=index.max().tz_convert(None).normalize() + pd.Timedelta(days=7))
    wall = index.tz_convert(cal.tz).tz_localize(None)
    session = wall.normalize()

    not_session = ~session.isin(cal.sessions)
    if not_session.any():
        raise ValueError(
            f"{not_session.sum()} bar(s) fall outside {calendar} sessions, "
            f"first {index[not_session][0]}"
        )
    session_close = pd.DatetimeIndex(cal.closes.reindex(session)).tz_convert(index.tz)

    since_midnight = pd.Series(wall - session)
    same_session = pd.Series(session).diff() == pd.Timedelta(0)
    # A session's first bar can be partial (the 9:30 bar under hourly bars),
    # so the bar size comes from gaps between later bars where there are any
    gaps = pd.Series(wall).diff()
    after_first = same_session & same_session.shift(1, fill_value=False)
    step = gaps[after_first if after_first.any() else same_session].mode()
    if step.empty:
        return session_close.rename(index.name)
    step = step.min()

    # The phase comes from bars after a session's first, which can start off
    # the grid (a 9:30 open under hourly bars)
    later = since_midnight[same_session] if same_session.any() else since_midnight
    phase = (later % step).mode().min()

    # Next grid point strictly after each start, then no later than the close
    n_steps = (since_midnight - phase) // step + 1
    grid_close = pd.DatetimeIndex(session + phase + n_steps * step)
    grid_close = grid_close.tz_localize(cal.tz).tz_convert(index.tz)
    return grid_close.where(grid_close < session_close, session_close).rename(index.name)
