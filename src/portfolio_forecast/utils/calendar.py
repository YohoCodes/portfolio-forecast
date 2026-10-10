"""Bars on the exchange calendar: sessions, opens, closes.

Every bar maps to its calendar session span: an intraday bar to the one
session it trades in, a daily bar to its session, a weekly or monthly bar to
the sessions of its week or month. Times are read on the exchange's clock, so
half-days and daylight-saving changes come from the calendar, never from
clock arithmetic.

Flow:

1. `_bar_spans` maps each bar to its span: `_date_spans` for dated bars,
   `close_times` (grid from `_bar_grid`) for intraday bars, and
   `_continuous_spans` when ``calendar=None`` (continuous mode: no
   sessions, each bar closes one grid step Δ after its start).
2. `session_spans`, `close_times` and `completed_mask` are the public
   views of those spans.
3. `_off_session_mask` finds intraday bars outside every session, which the
   marks engine drops before building marks.
"""
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from dateutil.tz import tzlocal


def _daily_sessions(dates, calendar='XNYS'):
    """Return (opens, closes) of the sessions on naive, midnight `dates`, as
    times on the exchange's clock, or None if the dates are spaced like weekly
    or longer bars, whose date doesn't say which session they end on.

    Daily bars sit on consecutive sessions, apart from the odd missing bar, so
    the dates count as daily when the most common step between them is one
    session. Raises ValueError for a date that isn't a session.
    """
    # A week of padding keeps the sessions just outside the dates' range
    cal = xcals.get_calendar(calendar,
                             start=dates.min() - pd.Timedelta(days=7),
                             end=dates.max() + pd.Timedelta(days=7))
    # Each date's position among the sessions; -1 for a date that isn't one
    session_i = cal.sessions.get_indexer(dates)
    not_session = session_i < 0
    if not_session.any():
        raise ValueError(
            f"{not_session.sum()} bar(s) fall outside {calendar} sessions, "
            f"first {dates[not_session][0]}"
        )
    # A most common step of more than one session means weekly or longer bars
    if len(dates) > 1 and pd.Series(session_i).diff().dropna().mode().min() > 1:
        return None
    opens = pd.DatetimeIndex(cal.opens.reindex(dates)).tz_convert(cal.tz)
    closes = pd.DatetimeIndex(cal.closes.reindex(dates)).tz_convert(cal.tz)
    return opens.rename(dates.name), closes.rename(dates.name)


def _aligned(step, phase, open_offsets):
    """Return whether a grid of `step` and `phase` (offset from midnight) is
    aligned to the clock (phase 0) or to every session open (each open's
    offset from midnight, `open_offsets`, sits on the grid)."""
    return phase == pd.Timedelta(0) or bool(((open_offsets % step) == phase).all())


def _choose_step(later, first, on_grid, open_offsets):
    """Return (step, phase) of a bar grid from spacings within sessions.

    `later` are spacings between bars after a session's first (each a
    multiple of the bar size, more than one where bars are missing); `first`
    are spacings from a session's first bar, which can be partial;
    `on_grid` are offsets from midnight of points known to sit on the grid;
    `open_offsets` the session opens' offsets from midnight.

    The step is the most common `later` spacing. When few bars are present
    that can be a multiple of the bar size (bars 9:30, 10:30, 12:30 give
    only 10:30 -> 12:30). A real grid is aligned to the session open or to
    the clock, so if the most common spacing's grid is neither, the largest
    first spacing that divides it and whose grid is aligned is taken
    instead: the first bar was then a full bar and a later one is missing.
    """
    step = later.mode().min()

    def phase_of(size):
        """Return the most common offset of `on_grid` from a grid of `size`."""
        return (on_grid % size).mode().min() if len(on_grid) else pd.Timedelta(0)

    # A grid aligned to neither the open nor the clock: try the largest aligned
    # first spacing that divides it
    if not _aligned(step, phase_of(step), open_offsets):
        for size in sorted(set(first), reverse=True):
            if pd.Timedelta(0) < size < step and step % size == pd.Timedelta(0) \
                    and _aligned(size, phase_of(size), open_offsets):
                step = size
                break
    return step, phase_of(step)


def _bar_grid(wall, session, wall_open):
    """Return (step, phase) of the bar grid of intraday bar starts, or None.

    `wall` holds the starts on the exchange's wall clock, `session` their
    session dates and `wall_open` their sessions' opens. The step comes
    from gaps between bars after a session's first (which can be partial,
    as the 9:30 bar under hourly bars), see `_choose_step`; the phase is the
    most common offset from midnight of bars after a session's first, which
    can start off the grid. None if no session holds two bars.
    """
    since_midnight = pd.Series(wall - session)
    # Spacing from the previous bar in the same session; `after_first` skips
    # each session's first spacing, which can start from a partial bar
    same_session = pd.Series(session).diff() == pd.Timedelta(0)
    gaps = pd.Series(wall).diff()
    after_first = same_session & same_session.shift(1, fill_value=False)
    later = gaps[after_first if after_first.any() else same_session]
    if later.empty:
        return None
    # Bars after a session's first are on the grid; they fix its phase
    on_grid = since_midnight[same_session] if same_session.any() else since_midnight
    open_offsets = pd.Series(pd.DatetimeIndex(wall_open) - pd.DatetimeIndex(wall_open).normalize())
    return _choose_step(later, gaps[same_session & ~after_first], on_grid, open_offsets)


def close_times(index, calendar='XNYS', tz=None):
    """Return the time each bar of a start-labelled index closes.

    yfinance and Interactive Brokers label a bar with its start, but its close
    (and anything computed from it) is only known when the bar ends. This
    gives the label that says so.

    Parameters
    ----------
    index : array-like of datetimes
        Bar start times, oldest first.
    calendar : str or None, default 'XNYS'
        ``exchange_calendars`` calendar code for the session closes. None is
        continuous mode: each bar closes one grid step after its start.
    tz : str or tzinfo, optional
        Timezone that naive intraday times are in. Defaults to this machine's
        local timezone. Ignored for everything else.

    Returns
    -------
    pandas.DatetimeIndex
        One close per bar. Intraday closes keep the input's timezone, or its
        naive wall clock. Daily closes are timezone-aware, on the exchange's
        clock. Continuous mode keeps the input's labelling, naive or aware.

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
    - Continuous mode: the grid step Δ is the most common spacing between
      bars, and every bar closes at its start + Δ.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return index
    # Continuous mode: every bar closes one grid step after its start
    if calendar is None:
        return (index + _continuous_step(index)).rename(index.name)
    if index.tz is None:
        # Dates: daily bars close at their session's close; longer bars stay as they are
        if (index == index.normalize()).all():
            daily = _daily_sessions(index, calendar)
            return index if daily is None else daily[1]
        # Same DST handling as `_localize`
        aware = index.tz_localize(tz or tzlocal(), ambiguous='infer',
                                  nonexistent='shift_forward')
        return close_times(aware, calendar).tz_localize(None)

    cal = xcals.get_calendar(calendar,
                             start=index.min().tz_convert(None).normalize() - pd.Timedelta(days=7),
                             end=index.max().tz_convert(None).normalize() + pd.Timedelta(days=7))
    # Each bar's start on the exchange's wall clock, and its session date
    wall = index.tz_convert(cal.tz).tz_localize(None)
    session = wall.normalize()

    not_session = ~session.isin(cal.sessions)
    if not_session.any():
        raise ValueError(
            f"{not_session.sum()} bar(s) fall outside {calendar} sessions, "
            f"first {index[not_session][0]}"
        )
    # Each bar's session close (in the input's timezone) and open (on the wall clock)
    session_close = pd.DatetimeIndex(cal.closes.reindex(session)).tz_convert(index.tz)
    wall_open = pd.DatetimeIndex(cal.opens.reindex(session)).tz_convert(cal.tz).tz_localize(None)

    # The bar grid's step and phase, or None with no two bars in one session
    grid = _bar_grid(wall, session, wall_open)
    if grid is None:
        return session_close.rename(index.name)
    step, phase = grid
    since_midnight = pd.Series(wall - session)

    # A bar closes at the next grid point strictly after its start, and
    # no later than its session's close (a short last bar, or a half-day)
    n_steps = (since_midnight - phase) // step + 1
    grid_close = pd.DatetimeIndex(session + phase + n_steps * step)
    grid_close = grid_close.tz_localize(cal.tz).tz_convert(index.tz)
    return grid_close.where(grid_close < session_close, session_close).rename(index.name)


# Padding around the input when building a calendar: a few days for intraday bars,
# more than a year for dated bars, whose span can reach into neighbouring months
_INTRADAY_PAD = pd.Timedelta(days=10)
_DATE_PAD = pd.Timedelta(days=400)

_SPAN_COLUMNS = ['start', 'close', 'first_open', 'last_close', 'n_sessions', 'session', 'kind']


def _get_calendar(calendar, first, last, pad):
    """Return `calendar` covering naive dates `first`..`last`, padded by `pad`
    on each side."""
    return xcals.get_calendar(calendar, start=(first - pad).normalize(), end=(last + pad).normalize())


def _session_times(cal):
    """Return (opens, closes) of every session of `cal`, as DatetimeIndexes on
    the exchange's clock, positionally aligned with ``cal.sessions``."""
    opens = pd.DatetimeIndex(cal.opens).tz_convert(cal.tz)
    closes = pd.DatetimeIndex(cal.closes).tz_convert(cal.tz)
    return opens, closes


def _localize(index, tz):
    """Return `index` made timezone-aware.

    Naive times are wall-clock times in `tz` (default: this machine's local
    timezone). DST: a wall time repeated when clocks go back is placed by
    the order of the labels (``ambiguous='infer'``); one skipped when they
    go forward moves to the first instant after the gap.
    """
    if index.tz is not None:
        return index
    return index.tz_localize(tz or tzlocal(), ambiguous='infer', nonexistent='shift_forward')


def _is_midnight(index):
    """Return whether every label is midnight in its own timezone: a date, not
    a time of day."""
    return bool((index == index.normalize()).all())


def _check_increasing(index, what='index'):
    """Raise ValueError unless `index` is strictly increasing."""
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError(f"{what} must be strictly increasing (sorted, no duplicates)")


def _continuous_step(index):
    """Return the grid step Δ of continuous-mode labels.

    Δ is the most common spacing between consecutive labels (the smallest
    on a tie), so a missing grid point does not change it. `index` is
    sorted. Raises ValueError for fewer than two labels, whose step is
    unknown.
    """
    if len(index) < 2:
        raise ValueError(
            "Continuous mode (calendar=None) needs at least two bars or marks to infer the grid step"
        )
    return pd.Series(index).diff().dropna().mode().min()


def _continuous_spans(index, tz):
    """Return the spans of continuous-mode bars as a `_SPAN_COLUMNS` frame.

    Each bar is its own unit: it starts at its label and closes one grid
    step Δ later, since a value is labelled at its bar's close. Naive labels,
    dates included, are wall-clock times in `tz` (default: this machine's
    local timezone).
    """
    aware = _localize(index, tz)
    close = aware + _continuous_step(aware)
    return pd.DataFrame({
        'start': aware,
        'close': close,
        'first_open': aware,
        'last_close': close,
        'n_sessions': 1,
        'session': pd.NaT,
        'kind': 'continuous',
    }).set_axis(index)


def _infer_date_bars(dates, cal):
    """Return (kind, step) of date-labelled bars: ('session', n) for
    n-session bars, ('week', 1) or ('month', n), from the spacing of naive
    `dates`.

    The median spacing in calendar days decides: 5 to 9 days is weekly, 25
    or more is months (counted at 30.44 days, 365.25 / 12), anything else
    is n-session bars. A single date is read as a daily bar.
    """
    if len(dates) < 2:
        return 'session', 1
    day_gap = dates.to_series().diff().dropna().dt.days.median()
    if 5 <= day_gap <= 9:
        return 'week', 1
    if day_gap >= 25:
        return 'month', max(1, round(day_gap / 30.44))
    # n-session bars: count the step in sessions, so weekends and holidays
    # don't stretch it
    positions = cal.sessions.searchsorted(dates)
    return 'session', max(1, int(pd.Series(positions).diff().dropna().mode().min()))


def _date_spans(dates, calendar, kind=None, step=None):
    """Return the session spans of date-labelled bars (daily, n-session,
    weekly, monthly), as a frame with `_SPAN_COLUMNS`, positionally aligned
    with naive midnight `dates`.

    Daily and n-session bars must be dated on a session and span it and the
    sessions after it (start labels, as yfinance and IB use). Weekly and
    monthly bars span the sessions of their week or month, so they may be
    labelled with any day in it (yfinance dates a month by its 1st, which can
    be a weekend). Multi-period bars are anchored at their first or last
    period by majority vote over the labels.

    Two internal columns say where each bar sits on the bar grid, so that a
    return spanning a scheduled bar missing from the data can be made
    irregular:
    ``slot`` (its first session's position, or its first period's ordinal)
    and ``step`` (sessions or periods per bar). Consecutive bars whose slots
    differ by more than ``step`` have a scheduled bar missing between them.
    """
    cal = _get_calendar(calendar, dates.min(), dates.max(), _DATE_PAD)
    if kind is None:
        kind, step = _infer_date_bars(dates, cal)
    sessions = cal.sessions
    opens, closes = _session_times(cal)
    dates = dates.as_unit(sessions.unit)

    # Session bars: from the labelled session through step - 1 more
    if kind == 'session':
        first = sessions.get_indexer(dates)
        outside = first < 0
        if outside.any():
            raise ValueError(
                f"{outside.sum()} bar(s) fall outside {calendar} sessions, first {dates[outside][0]}"
            )
        last = first + step - 1
        slot = first
    else:
        # Weekly or monthly bars: the first and last session of each label's own period
        freq = 'W-SUN' if kind == 'week' else 'M'
        periods = dates.to_period(freq)
        own_first = sessions.searchsorted(periods.start_time.as_unit(sessions.unit), 'left')
        own_last = sessions.searchsorted(periods.end_time.as_unit(sessions.unit), 'right') - 1
        # Majority vote on the anchor: a label on or before its period's first
        # session votes "first period", one on its last session "last period"
        own_first_session = sessions[np.minimum(own_first, len(sessions) - 1)]
        first_votes = int((dates <= own_first_session).sum())
        last_votes = int((dates == sessions[np.maximum(own_last, 0)]).sum())
        lo, hi = (periods, periods + (step - 1)) if first_votes >= last_votes else (periods - (step - 1), periods)
        # The bar's span, from its first period's first session to its last period's last
        first = sessions.searchsorted(lo.start_time.as_unit(sessions.unit), 'left')
        last = sessions.searchsorted(hi.end_time.as_unit(sessions.unit), 'right') - 1
        slot = np.asarray(lo.asi8)
        empty = last < first
        if empty.any():
            raise ValueError(
                f"{empty.sum()} bar(s) span no {calendar} session, first {dates[empty][0]}"
            )

    return pd.DataFrame({
        'start': opens[first],
        'close': closes[last],
        'first_open': opens[first],
        'last_close': closes[last],
        'n_sessions': (last - first + 1).astype(int),
        'session': sessions[last],
        'kind': kind,
        'slot': slot,
        'step': step,
    })


def _bar_spans(index, calendar='XNYS', tz=None):
    """Return every bar of a start-labelled `index` on the calendar.

    Returns a frame indexed like `index` with columns:

    - ``start``: when the bar starts (intraday: its label; dated bars: the
      open of its first session), on the exchange's clock
    - ``close``: when it ends (intraday: its `close_times` close, no later
      than the session close; dated bars: the close of its last session)
    - ``first_open``, ``last_close``, ``n_sessions``: its session span
    - ``session``: its last session, as a naive date
    - ``kind``: ``'intraday'``, ``'session'`` (daily), ``'week'`` or
      ``'month'``; ``'session'`` with ``n_sessions > 1`` for n-day bars;
      ``'continuous'`` with ``calendar=None`` (see `_continuous_spans`)
    - intraday only, ``prev_close``: the close of the calendar session
      before the bar's, so ``first_open == prev_close`` marks a session
      that opens the instant the last one closed, which has no gap
    - dated bars only, ``slot`` and ``step`` (see `_date_spans`)

    Midnight labels are dates. Other labels are times of day (naive ones in
    `tz`); the bars are intraday if any session holds two of them, and
    otherwise each names its whole session.

    Raises ValueError if the index is not strictly increasing, a bar falls
    outside the calendar's sessions, or an intraday bar starts outside its
    session's hours (extended-hours bars are not supported).
    """
    index = pd.DatetimeIndex(index)
    _check_increasing(index)
    if len(index) == 0:
        return pd.DataFrame(columns=_SPAN_COLUMNS, index=index)
    # Continuous mode and dates have their own span builders
    if calendar is None:
        return _continuous_spans(index, tz)
    if _is_midnight(index):
        dates = index.tz_localize(None) if index.tz is not None else index
        return _date_spans(dates, calendar).set_axis(index)

    # Times of day: each bar's session, on the exchange's clock
    aware = _localize(index, tz)
    cal = _get_calendar(calendar, aware.min().tz_convert(None), aware.max().tz_convert(None), _INTRADAY_PAD)
    start = aware.tz_convert(cal.tz)
    session = start.tz_localize(None).normalize().as_unit(cal.sessions.unit)
    pos = cal.sessions.get_indexer(session)
    outside = pos < 0
    if outside.any():
        raise ValueError(
            f"{outside.sum()} bar(s) fall outside {calendar} sessions, first {index[outside][0]}"
        )
    if not session.duplicated().any():
        # One bar per session: each is that session's daily bar
        return _date_spans(session, calendar, 'session', 1).set_axis(index)

    # Intraday bars must start within their session's hours
    opens, closes = _session_times(cal)
    session_open, session_close = opens[pos], closes[pos]
    off_hours = (start < session_open) | (start >= session_close)
    if off_hours.any():
        raise ValueError(
            f"{off_hours.sum()} bar(s) start outside {calendar} session hours, first {index[off_hours][0]}. "
            "Extended-hours bars aren't supported; pass regular-hours bars only."
        )
    return pd.DataFrame({
        'start': start,
        'close': close_times(start, calendar),
        'first_open': session_open,
        'last_close': session_close,
        'n_sessions': 1,
        'session': cal.sessions[pos],
        'kind': 'intraday',
        # The calendar is padded by days on each side, so pos - 1 >= 0
        'prev_close': closes[pos - 1],
    }).set_axis(index)


def _off_session_mask(index, calendar='XNYS', tz=None):
    """Return which bars of a start-labelled `index` are intraday bars outside
    every session.

    A bar is outside if it starts on a non-session day, before its session's
    open or at or after its close (extended hours, or a 24/7 asset's
    weekend). Only intraday bars are ever outside: dates, a time-of-day index
    with at most one bar per day (read as daily bars, which raise on a
    non-session day instead) and continuous mode (``calendar=None``)
    give all False.
    """
    index = pd.DatetimeIndex(index)
    none = np.zeros(len(index), dtype=bool)
    if calendar is None or len(index) == 0 or _is_midnight(index):
        return none
    # Each bar's start and day on the exchange's clock; one bar per day means daily bars
    aware = _localize(index, tz)
    cal = _get_calendar(calendar, aware.min().tz_convert(None), aware.max().tz_convert(None), _INTRADAY_PAD)
    start = aware.tz_convert(cal.tz)
    day = start.tz_localize(None).normalize().as_unit(cal.sessions.unit)
    if not day.duplicated().any():
        return none
    pos = cal.sessions.get_indexer(day)
    opens, closes = _session_times(cal)
    # A bar is inside if it starts on a session day, at or after the
    # open and before the close
    inside = pos >= 0
    inside[inside] = (start[inside] >= opens[pos[inside]]) & (start[inside] < closes[pos[inside]])
    return ~inside


def session_spans(index, calendar='XNYS', tz=None):
    """Return the calendar session span of every bar.

    Parameters
    ----------
    index : array-like of datetimes
        Bar labels, oldest first. Midnight labels are dates (daily, weekly,
        monthly bars); other labels are bar start times.
    calendar : str or None, default 'XNYS'
        ``exchange_calendars`` calendar code. None is continuous mode:
        there are no sessions, and each bar spans itself.
    tz : str or tzinfo, optional
        Timezone of naive intraday times. Defaults to this machine's local
        timezone. Ignored for dates, except in continuous mode, where naive
        dates are midnights in `tz`.

    Returns
    -------
    pandas.DataFrame
        Indexed like `index`, with ``first_open`` (open of the bar's first
        session), ``last_close`` (close of its last session), both
        timezone-aware on the exchange's clock, and ``n_sessions``. An
        intraday bar spans the one session it trades in. In continuous mode,
        ``first_open`` is the bar's start, ``last_close`` its start plus the
        grid step, and ``n_sessions`` is 1.

    Raises
    ------
    ValueError
        If `index` is not strictly increasing, a bar falls outside the
        calendar's sessions, or an intraday bar starts outside session hours.

    Notes
    -----
    - Daily and n-day bars are dated on a session. n-day bars are read as
      start-labelled (the yfinance and Interactive Brokers convention): an
      n-day bar spans its labelled session and the next n - 1.
    - Weekly and monthly bars span every session of their week (Monday to
      Sunday) or month, holidays and half-days included, and may be dated
      with any day in it.
    - Bars are intraday if some session holds two of them; a time-of-day
      series with one bar per session is read as daily bars.
    - Intraday bars outside session hours raise here; the backtest drops
      them with a warning before building marks.
    """
    return _bar_spans(index, calendar, tz)[['first_open', 'last_close', 'n_sessions']]


def completed_mask(index, calendar='XNYS', tz=None, now=None):
    """Return which bars have closed by `now`.

    Parameters
    ----------
    index : array-like of datetimes
        Bar labels, as for `session_spans`.
    calendar : str or None, default 'XNYS'
        ``exchange_calendars`` calendar code; None is continuous mode,
        where a bar closes one grid step after its start.
    tz : str or tzinfo, optional
        Timezone of naive intraday times and of a naive `now`. Defaults to
        this machine's local timezone.
    now : datetime-like, optional
        The current time. None uses the clock; tests inject it.

    Returns
    -------
    numpy.ndarray of bool
        True where the bar's close is at or before `now`. An intraday bar
        closes at its `close_times` close (no later than the session close);
        a dated bar at the close of its last session.
    """
    # Each bar's close; a naive now is read in tz
    spans = _bar_spans(index, calendar, tz)
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize(tz or tzlocal())
    # A bar is complete only if its close is at or before now
    return (pd.DatetimeIndex(spans['close']) <= now) if len(spans) else np.zeros(0, dtype=bool)
