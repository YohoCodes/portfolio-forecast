"""Marks and return classes: the one engine behind the backtest's labels,
`classify_returns`, `to_session_values` and the future layout.

A mark is one entry of a value series: a moment and the portfolio's value
then. Intraday marks are each session's open, then each bar's close;
daily and longer marks are the fill, then each bar's close. The return
ending at a mark is classed ``"gap"`` (last close to the next session's
open), ``"regular"`` (a bar) or ``"irregular"``.

Flow:

1. `history_marks` (via `_build_marks`) turns bars into marks: it drops
   intraday bars outside every session, maps the rest to sessions
   (`calendar._bar_spans`), interleaves open marks and bar closes, records
   each symbol's freshness and classes every return.
2. `classify_returns` runs the same classing on the labels of an external
   value series, and `is_daily` / `_series_kind` decide daily against
   intraday on the calendar, never from clock times alone.
3. `to_session_values` collapses a series to one value per session (or per
   UTC day in continuous mode).
4. `_future_layout` continues a history's grid into the future,
   for `next_trading_dates`.

``calendar=None`` is continuous mode, for 24/7 assets: a constant grid
t_k = t_0 + kΔ, no sessions, gaps or open marks, every return regular
unless a grid point is missing or a held symbol is stale.
"""
import warnings
from typing import NamedTuple

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from dateutil.tz import tzlocal

from ..types import BacktestResult, FutureLayout
from .calendar import (
    _DATE_PAD,
    _INTRADAY_PAD,
    _bar_grid,
    _bar_spans,
    _check_increasing,
    _choose_step,
    _continuous_step,
    _get_calendar,
    _infer_date_bars,
    _is_midnight,
    _localize,
    _off_session_mask,
    _session_times,
)
from .periods import _parse_interval

GAP, REGULAR, IRREGULAR = "gap", "regular", "irregular"

# Continuous mode warns when more than this share of returns is irregular,
# a sign the data isn't 24/7
_IRREGULAR_SHARE_LIMIT = 0.10


def _held_symbols(held, columns):
    """Return the held symbols (w_i > 0) as an Index of `columns` labels.

    `held` is a list-like of column labels, a boolean Series by symbol, or a
    weight Series by symbol (positive weights are held).
    """
    if isinstance(held, pd.Series) and not isinstance(held.index, pd.RangeIndex):
        held = held.index[held.astype(bool) if held.dtype == bool else held > 0]
    held = pd.Index(held)
    unknown = held.difference(columns)
    if len(unknown):
        raise ValueError(f"held symbol(s) not among the price columns: {', '.join(map(str, unknown))}")
    return held


def _missing_between(labels, scheduled):
    """Return, for each pair of consecutive `labels`, whether a `scheduled`
    mark strictly between them is absent from `labels`.

    Both are timezone-aware DatetimeIndexes; they are compared as instants.
    """
    labels = labels.tz_convert("UTC")
    scheduled = scheduled.tz_convert("UTC")
    # Scheduled marks absent from the labels, then how many fall inside each pair
    absent = scheduled[~scheduled.isin(labels)].sort_values()
    lo = absent.searchsorted(labels[:-1], side="right")
    hi = absent.searchsorted(labels[1:], side="left")
    return hi > lo


def _scheduled_intraday(cal, first_pos, last_pos, step, phase):
    """Return every scheduled intraday mark of sessions `first_pos` to
    `last_pos` (positions in ``cal.sessions``), timezone-aware.

    A session's scheduled marks are its open mark, unless it opens the
    instant the previous session closed (no gap, so no open mark), then
    its bar closes on the grid `(step, phase)` (wall clock), ending at the
    session close.
    """
    opens, closes = _session_times(cal)
    wall_open, wall_close = opens.tz_localize(None), closes.tz_localize(None)
    points = []
    # The calendar is padded before the first session, so p - 1 >= 0
    for p in range(first_pos, last_pos + 1):
        if opens[p] != closes[p - 1]:
            points.append(pd.DatetimeIndex([wall_open[p]]))
        points.append(_session_grid(wall_open[p], wall_close[p], step, phase))
    wall = points[0].append(points[1:]) if points else pd.DatetimeIndex([])
    # DST: a wall time repeated when clocks go back is left out rather than guessed;
    # one skipped when they go forward moves to the first instant after the gap
    aware = wall.tz_localize(cal.tz, ambiguous="NaT", nonexistent="shift_forward")
    return aware[aware.notna()]


class _Marks(NamedTuple):
    """Everything `_build_marks` knows about a backtest's marks.

    ``labels``, ``fresh`` and ``classes`` are `history_marks`' output.
    ``rows`` are the bar labels kept (priced, in session); ``bar`` and
    ``is_open`` give, for each mark, its bar (a position in ``rows``) and
    whether it is that bar's open. ``late_open``, ``missing`` and ``stale``
    say, for each return, why it is irregular: a late open mark, a
    scheduled mark missing for every symbol or a stale held symbol.
    """

    labels: pd.DatetimeIndex
    fresh: pd.DataFrame
    classes: pd.Series
    rows: pd.Index
    bar: np.ndarray
    is_open: np.ndarray
    late_open: np.ndarray
    missing: np.ndarray
    stale: np.ndarray


def _drop_off_session(closes, opens, calendar, tz, stacklevel):
    """Return `closes` and `opens` without intraday bars outside every
    session, warning once with their count.

    Their price moves are not lost: the next mark prices every symbol
    afresh, so they land in the next session's gap. `opens` may be None.
    `stacklevel` is counted from this function. Raises ValueError if no bar
    is left.
    """
    off = _off_session_mask(closes.index, calendar, tz)
    if not off.any():
        return closes, opens
    warnings.warn(
        f"Dropped {off.sum()} intraday bar(s) outside {calendar} sessions (extended hours or "
        f"non-session days), first {closes.index[off][0]}. Values are marked in session hours only; "
        "the dropped bars' moves fall into the next session's gap",
        UserWarning, stacklevel=stacklevel,
    )
    if off.all():
        raise ValueError(f"No bars fall inside {calendar} sessions")
    return closes[~off], (opens[~off] if opens is not None else None)


def _build_marks(opens, closes, held, calendar, tz, fill, stacklevel=3):
    """Return the `_Marks` of a backtest; the engine behind `history_marks`
    and `simulate_buy_and_hold`.

    Raises ValueError as listed under `history_marks`. Warns (UserWarning)
    about intraday bars dropped for lying outside every session and,
    in continuous mode, about a high share of irregular returns.
    `stacklevel` points both warnings at the public caller.
    """
    if fill not in ("open", "close"):
        raise ValueError(f"fill must be 'open' or 'close', got {fill!r}")
    closes = pd.DataFrame(closes)
    if opens is not None:
        opens = pd.DataFrame(opens)
        if not opens.index.equals(closes.index) or set(opens.columns) != set(closes.columns):
            raise ValueError("opens and closes must have the same index and columns")
        opens = opens[closes.columns]
    held = _held_symbols(held, closes.columns)

    # A row with no price for any symbol is not a bar
    priced = closes.notna().any(axis=1)
    if opens is not None:
        priced |= opens.notna().any(axis=1)
        opens = opens[priced]
    closes = closes[priced]
    if closes.empty:
        raise ValueError("closes has no priced bars")
    _check_increasing(pd.DatetimeIndex(closes.index), "bar index")

    closes, opens = _drop_off_session(closes, opens, calendar, tz, stacklevel + 1)

    bars = _bar_spans(closes.index, calendar, tz)
    kind = bars["kind"].iloc[0]
    intraday = kind == "intraday"
    continuous = kind == "continuous"
    if opens is None and intraday:
        raise ValueError("Intraday bars need opens for each session's open mark; pass data with opens")
    if opens is None and fill == "open":
        raise ValueError("fill='open' needs opening prices; pass data with opens, or use fill='close'")

    n_bars = len(bars)
    starts = pd.DatetimeIndex(bars["start"])
    ends = pd.DatetimeIndex(bars["close"])
    if intraday:
        # Each session's open mark, at its first available bar's start; none for a
        # session that opens the instant the last one closed (no gap), apart from the fill
        session = bars["session"].to_numpy()
        first_open = pd.DatetimeIndex(bars["first_open"])
        new_session = np.r_[True, session[1:] != session[:-1]]
        has_gap = first_open != pd.DatetimeIndex(bars["prev_close"])
        open_bars = np.flatnonzero(new_session & has_gap)
        if fill == "open" and (not len(open_bars) or open_bars[0] != 0):
            open_bars = np.r_[0, open_bars]
        if fill == "close":
            # Fill="close" buys at the first bar's close, so session 1
            # has no open mark
            open_bars = open_bars[open_bars > 0]
        at_session_open = starts[open_bars] == first_open[open_bars]
    else:
        # Daily and longer, and continuous mode: the fill mark only
        open_bars = np.array([0]) if fill == "open" else np.array([], dtype=int)
        at_session_open = np.ones(len(open_bars), dtype=bool)

    # Interleave: a bar's open mark sorts just before its close
    bar = np.r_[open_bars, np.arange(n_bars)]
    is_open = np.r_[np.ones(len(open_bars), dtype=bool), np.zeros(n_bars, dtype=bool)]
    open_at_start = np.r_[at_session_open, np.zeros(n_bars, dtype=bool)]
    order = np.argsort(2 * bar + ~is_open, kind="stable")
    bar, is_open, open_at_start = bar[order], is_open[order], open_at_start[order]
    labels = starts[bar].where(is_open, ends[bar])
    if not (labels.is_monotonic_increasing and labels.is_unique):
        dup = labels[labels.duplicated()][0]
        raise ValueError(f"Two bars close at {dup}; bars must not overlap")

    # Own price at each mark: the open at an open mark, the close otherwise
    own = closes.notna().to_numpy()[bar]
    if len(open_bars):
        own[is_open] = opens.notna().to_numpy()[bar[is_open]]
    # After its last price a symbol is cash, which is always fresh
    rows = np.arange(len(labels))[:, None]
    last_price = len(labels) - 1 - np.argmax(own[::-1], axis=0)
    fresh = own | ((rows > last_price) & own.any(axis=0))

    # Why each return (ending at marks 1..n-1) might be irregular
    held_fresh = fresh[:, closes.columns.get_indexer(held)].all(axis=1)
    # A return keeps its class only if every held symbol has its own
    # price at both of its marks
    stale = ~(held_fresh[1:] & held_fresh[:-1])
    if continuous:
        # A return longer than one grid step spans a missing grid point
        late_open = np.zeros(len(labels) - 1, dtype=bool)
        missing = np.asarray((labels[1:] - labels[:-1]) != (ends[0] - starts[0]))
    else:
        # An open mark after the session open (its first bar is
        # missing) ends an irregular return
        late_open = (is_open & ~open_at_start)[1:]
        missing = _missing_bars(bars, labels, bar, is_open, calendar)
    base = np.where(is_open, GAP, REGULAR)[1:] if intraday else np.full(len(labels) - 1, REGULAR)
    classes = np.where(stale | late_open | missing, IRREGULAR, base)

    labels = labels.rename(None)
    classes = pd.Series(classes, index=labels[1:], dtype=object)
    if continuous:
        _warn_irregular_share(classes, stacklevel)
    return _Marks(
        labels=labels,
        fresh=pd.DataFrame(fresh, index=labels, columns=closes.columns),
        classes=classes,
        rows=closes.index,
        bar=bar,
        is_open=is_open,
        late_open=late_open,
        missing=missing,
        stale=stale,
    )


def _missing_bars(bars, labels, bar, is_open, calendar):
    """Return, per return, whether its span holds a scheduled mark missing
    from the data for every symbol: a mid-session bar, a session's
    open mark, or a whole non-holiday session.

    Intraday, the schedule is every session's open mark (none where it
    opens the instant the last one closed) and its bar closes on the
    history's grid. Daily and longer, it is one close per
    bar slot: consecutive bars more than one step of sessions (or weeks, or
    months) apart have a bar missing between them.
    """
    if len(labels) < 2:
        return np.zeros(0, dtype=bool)
    if bars["kind"].iloc[0] != "intraday":
        slot = bars["slot"].to_numpy()[bar]
        step = int(bars["step"].iloc[0])
        # A return from the fill (an open mark) to its own bar's close skips nothing
        return (np.diff(slot) > step) & ~is_open[:-1]
    # Intraday: the bar grid learned from the bars, then every scheduled mark it implies
    starts = pd.DatetimeIndex(bars["start"])
    first, last = pd.Timestamp(bars["session"].iloc[0]), pd.Timestamp(bars["session"].iloc[-1])
    cal = _get_calendar(calendar, first, last, _INTRADAY_PAD)
    wall = starts.tz_convert(cal.tz).tz_localize(None)
    wall_open = pd.DatetimeIndex(bars["first_open"]).tz_convert(cal.tz).tz_localize(None)
    step, phase = _bar_grid(wall, pd.DatetimeIndex(bars["session"]), wall_open)
    pos = cal.sessions.get_indexer(pd.DatetimeIndex([first, last]).as_unit(cal.sessions.unit))
    return _missing_between(labels, _scheduled_intraday(cal, pos[0], pos[1], step, phase))


def _warn_irregular_share(classes, stacklevel):
    """Warn when more than 10% of continuous-mode returns are irregular:
    the data is likely not 24/7, so the grid has holes every night or weekend."""
    n = len(classes)
    n_irregular = int((classes == IRREGULAR).sum())
    if n and n_irregular / n > _IRREGULAR_SHARE_LIMIT:
        warnings.warn(
            f"{n_irregular} of {n} returns ({n_irregular / n:.0%}) are irregular in continuous mode "
            "(calendar=None): the grid has missing points. calendar=None is for 24/7 data on a constant "
            "grid; for weekday-only data (NAVs, indices, FX) pass calendar=\"24/5\", and for exchange-traded "
            "assets the exchange's calendar (e.g. calendar=\"XNYS\")",
            UserWarning, stacklevel=stacklevel + 1,
        )


def history_marks(opens, closes, held, calendar="XNYS", tz=None, fill="open"):
    """Return the marks of a backtest, each symbol's freshness there, and the
    class of every return.

    Parameters
    ----------
    opens : pandas.DataFrame or None
        Bar opens, (bar x symbol), on the same index and columns as `closes`.
        None is allowed only for daily and longer bars, or continuous mode,
        with ``fill="close"``.
    closes : pandas.DataFrame
        Bar closes, (bar x symbol), indexed by bar label: start times for
        intraday bars, dates for daily and longer bars. Strictly increasing.
    held : list-like or pandas.Series
        The symbols held (w_i > 0): column labels, a boolean Series by
        symbol, or a weight Series by symbol.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code. None is continuous mode
        for 24/7 data: a constant grid, no sessions.
    tz : str or tzinfo, optional
        Timezone of naive intraday times (and, in continuous mode, of naive
        dates). Defaults to this machine's local timezone.
    fill : {"open", "close"}, default "open"
        Where the book is bought: at the first bar's open, which is then
        the first mark, or at its close.

    Returns
    -------
    labels : pandas.DatetimeIndex
        The marks, timezone-aware on the exchange's clock. Intraday:
        each session's open mark at its first available bar's start,
        then each bar's close (no later than the session close). Session 1
        has an open mark only with ``fill="open"``; a session that opens the
        instant the previous one closed (O_d = C_{d-1}, e.g. weeknights on
        "24/5") has none. Daily and longer: the fill mark at the open
        of the first bar's first session (``fill="open"``), then each bar's
        close at the close of its last session. Continuous mode: the fill
        mark at the first bar's start (``fill="open"``), then each bar's
        start + Δ, Δ the most common spacing of the bars.
    fresh : pandas.DataFrame
        (mark x symbol) booleans: True where the symbol has its own price at
        the mark (its open at an open mark, its close at a close mark), and
        at every mark after its last price, where it is cash.
    classes : pandas.Series
        Class of each return, indexed by the mark where it ends
        (``labels[1:]``). ``"gap"`` for a return ending at an open mark at
        the session open; ``"irregular"`` for one ending at an open mark
        after it, with a held symbol stale at either end, or
        whose span holds a scheduled mark missing for every symbol (a
        mid-session bar, or a whole non-holiday session; in continuous mode
        a grid point); else ``"regular"``. Daily and longer, and continuous
        mode, have no gaps.

    Raises
    ------
    ValueError
        If intraday bars come without opens, ``fill="open"`` has no
        opens, the frames don't line up, a held symbol isn't a column,
        a daily bar is dated on a non-session day, or continuous mode
        has a single bar (its grid step is unknown).

    Warns
    -----
    UserWarning
        With a calendar, once with the count of intraday bars dropped for
        lying outside every session. In continuous mode, when more
        than 10% of returns are irregular, naming ``calendar="24/5"`` or an
        exchange calendar as the fix.

    Notes
    -----
    - Rows with no price at all are not bars. A scheduled bar missing for
      every symbol leaves no mark, and the return spanning it is irregular;
      a holiday is not scheduled, so the gap across it stays a gap. A
      price missing for one symbol makes that symbol stale at the mark.
    - Nothing is dropped for being in progress and nothing is warned about
      late opens, stale prices or missing bars: the caller applies `completed_mask` first and counts
      the irregular returns for its warnings.
    """
    marks = _build_marks(opens, closes, held, calendar, tz, fill)
    return marks.labels, marks.fresh, marks.classes


def _labels_of(values):
    """Return the DatetimeIndex of a Series, BacktestResult or DatetimeIndex."""
    if isinstance(values, BacktestResult):
        values = values.values
    index = values.index if isinstance(values, pd.Series) else values
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("values needs a DatetimeIndex of mark labels")
    return index


def _mark_sessions(index, calendar):
    """Return the session of every mark, as a frame with columns ``session``
    (naive date), ``pos`` (position in the calendar's sessions),
    ``at_open``, ``at_close``, ``close`` and ``open`` (the session's close
    and open, on the exchange's clock) and ``zero_gap`` (the session opens
    the instant the previous one closed).

    A mark at time t belongs to the first session whose close is at or after
    t, if that session has opened by t. So a mark at an instant that is both
    a close and the next open (O_d = C_{d-1}, "24/5") is the earlier
    session's close.

    Raises ValueError for naive labels (they are times on an unknown clock)
    and for a mark outside every session's open-to-close span.
    """
    if index.tz is None:
        raise ValueError(
            "Intraday labels are naive, so their clock is unknown; localize them first, "
            "e.g. values.tz_localize('America/New_York')"
        )
    cal = _get_calendar(calendar, index.min().tz_convert(None), index.max().tz_convert(None), _INTRADAY_PAD)
    # Each mark's session: the first whose close is at or after it, if it has opened
    t = index.tz_convert(cal.tz)
    opens, closes = _session_times(cal)
    pos = closes.searchsorted(t, side="left")
    clipped = np.minimum(pos, len(closes) - 1)
    outside = (pos >= len(closes)) | (t < opens[clipped])
    if outside.any():
        raise ValueError(
            f"{outside.sum()} mark(s) fall outside {calendar} session hours, first {index[outside][0]}"
        )
    # The calendar is padded before the first mark, so pos - 1 >= 0
    return pd.DataFrame({
        "session": cal.sessions[pos],
        "pos": pos,
        "at_open": t == opens[pos],
        "at_close": t == closes[pos],
        "close": closes[pos],
        "open": opens[pos],
        "zero_gap": opens[pos] == closes[pos - 1],
    })


DAILY, INTRADAY, UNCLASSIFIABLE = "daily", "intraday", "unclassifiable"

_UNCLASSIFIABLE_MESSAGE = (
    "values has at most one mark per session but not at the session close (e.g. 12:00 every session), "
    "so it can't be classified as daily or intraday. Collapse it with to_session_values, which "
    "labels each session's value at its close"
)

_ONE_DAY = pd.Timedelta(days=1)


def _series_kind(index, calendar):
    """Return ``"daily"``, ``"intraday"`` or ``"unclassifiable"`` for a
    DatetimeIndex of marks.

    Daily: at most one mark per session, at its close, plus optionally one
    initial fill mark anywhere; midnight dates are daily. Unclassifiable: at
    most one mark per session (the fill aside), not all at the close.
    Intraday: several marks in some session. Continuous mode
    (``calendar=None``): daily iff the grid step is one day, else intraday.
    """
    if len(index) == 0:
        raise ValueError("values is empty")
    if calendar is None:
        _check_increasing(index, "mark labels")
        return DAILY if _continuous_step(index) == _ONE_DAY else INTRADAY
    if _is_midnight(index):
        return DAILY
    _check_increasing(index, "mark labels")
    info = _mark_sessions(index, calendar)
    at_close = info["at_close"].to_numpy()
    # One initial fill mark may sit anywhere; every other mark must
    # be a session close, one per session
    closes = at_close if at_close[0] else at_close[1:]
    if closes.all() and info["session"][at_close].is_unique:
        return DAILY
    # Unclassifiable is one mark per session, but not at the close,
    # leaving out a fill that shares its session with the next mark
    session = info["session"]
    if len(session) > 1 and session.iloc[0] == session.iloc[1]:
        session = session.iloc[1:]
    return UNCLASSIFIABLE if session.is_unique else INTRADAY


def _is_daily_index(index, calendar):
    """Return `is_daily` of a DatetimeIndex of marks."""
    return _series_kind(index, calendar) == DAILY


def is_daily(values, calendar="XNYS"):
    """Return whether a value series is daily (or longer) rather than intraday.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or pandas.DatetimeIndex
        The value series, or its labels, oldest first.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.

    Returns
    -------
    bool
        True if the series has at most one mark per session, at the session
        close, plus optionally one initial fill mark (such as our own
        ``fill="open"`` daily output: 09:30, then 16:00 marks). Midnight
        labels are dates, so daily. Weekly and longer series, marked at
        their last session's close, count as daily too. False for intraday
        series, and for a series with one mark per session away from the
        close (e.g. 12:00 every session), which is neither daily nor
        intraday (see `to_session_values`). In continuous mode, True
        iff the grid step Δ (the most common spacing) is one day.

    Raises
    ------
    ValueError
        If `values` is empty, its labels are naive times of day, not
        strictly increasing, or a mark falls outside every session. In
        continuous mode, if it has a single mark (no grid step).

    Notes
    -----
    - The decision is made on the calendar, never from clock times alone: a
      mark at 13:00 is a close on a half-day and mid-session on other days.
    - There is no `tz` parameter: naive times of day are on an unknown
      clock, so they raise rather than being guessed. Localize them first,
      e.g. ``values.tz_localize('America/New_York')``. Naive midnight dates
      are fine, and so is any naive series in continuous mode, which only
      looks at spacings.
    """
    return _is_daily_index(_labels_of(values), calendar)


def _bar_close_sessions(index, calendar):
    """Return the naive session dates of the bar closes of a daily-or-longer
    series of marks, leaving out a leading fill mark that is not a close."""
    if _is_midnight(index):
        return (index.tz_localize(None) if index.tz is not None else index).as_unit("ns")
    info = _mark_sessions(index, calendar)
    closes = info if info["at_close"].iloc[0] else info.iloc[1:]
    return pd.DatetimeIndex(closes["session"]).as_unit("ns")


def _daily_bar_size(index, calendar):
    """Return (kind, step) of the bars of a daily-or-longer series of marks:
    ``('session', n)``, ``('week', n)`` or ``('month', n)``."""
    dates = _bar_close_sessions(index, calendar)
    cal = _get_calendar(calendar, dates.min(), dates.max(), _DATE_PAD)
    return _infer_date_bars(dates, cal)


def _first_mark_at_close(index, calendar):
    """Return whether a series' first mark is a session close (dates count
    as closes), so its first return runs close to close."""
    if _is_midnight(index):
        return True
    return bool(_mark_sessions(index[:1], calendar)["at_close"].iloc[0])


def _intraday_step(index, calendar):
    """Return the bar size Δ of an intraday series of marks, learned from the
    marks as for `next_trading_dates` (a short bar's ρ = τ/Δ is measured against it)."""
    # The marks on the exchange's wall clock, each session's first mark flagged
    info = _mark_sessions(index, calendar)
    cal_tz = pd.DatetimeIndex(info["close"]).tz
    wall = index.tz_convert(cal_tz).tz_localize(None)
    session = info["session"].to_numpy()
    first_in_session = np.r_[True, session[1:] != session[:-1]]
    wall_open = pd.DatetimeIndex(info["open"]).tz_localize(None)
    return _grid(wall, first_in_session, info["at_close"].to_numpy(), None, wall_open)[0]


def _continuous_classes(index):
    """Return the classes of continuous-mode marks: regular when the
    return spans one grid step, irregular when it spans a missing grid point."""
    step = _continuous_step(index)
    regular = np.asarray((index[1:] - index[:-1]) == step)
    return pd.Series(np.where(regular, REGULAR, IRREGULAR), index=index[1:], dtype=object)


def _dated_missing(sessions, cal):
    """Return, for consecutive daily-or-longer bar closes on naive session
    dates `sessions`, whether a scheduled bar is missing between them:
    their slots (sessions, weeks or months) differ by more than one bar."""
    # Each close's slot: its session position, or its week or month ordinal
    kind, step = _infer_date_bars(sessions, cal)
    if kind == "session":
        slot = cal.sessions.searchsorted(sessions.as_unit(cal.sessions.unit))
    else:
        slot = sessions.to_period("W-SUN" if kind == "week" else "M").asi8
    return np.diff(np.asarray(slot)) > step


def classify_returns(values, calendar="XNYS"):
    """Return the class of every return of a value series, inferred from its
    labels.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or pandas.DatetimeIndex
        The value series, or its labels, oldest first. Intraday labels must
        be timezone-aware.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.

    Returns
    -------
    pandas.Series
        ``"gap"``, ``"regular"`` or ``"irregular"`` for each return, indexed
        by the mark where it ends (``values.index[1:]``). On intraday
        series, a return ending at a session's first mark is a gap if that
        mark is at the session open and irregular if it is later,
        unless the session opens the instant the previous one closed, when
        it is a regular bar. A return whose span holds a scheduled
        mark missing from the series (a mid-session bar, a whole
        non-holiday session) is irregular. Every other return is
        regular. Daily and longer series: regular, or irregular across a
        missing bar. Continuous mode: regular across one grid step,
        irregular across more.

    Raises
    ------
    ValueError
        If an intraday series spans several sessions but no session after
        the first that has a gap starts with a mark at its open: there are
        no open marks to infer the gaps from. Pass ``classes`` to the
        simulator, or collapse the series with `to_session_values`. If the
        series has one mark per session away from the close (e.g. 12:00
        every session): it is neither daily nor intraday; collapse it
        with `to_session_values`. Also for naive times of day and marks
        outside every session.

    Warns
    -----
    UserWarning
        In continuous mode, when more than 10% of returns are irregular,
        naming ``calendar="24/5"`` or an exchange calendar as the fix.

    Notes
    -----
    - Classes come from the labels only. A symbol priced stale inside an
      external backtest can't be seen here; `simulate_buy_and_hold`
      supplies classes that account for it.
    - The intraday schedule is learned from the labels: the bar size is the
      most common spacing within sessions, as for `next_trading_dates`.
    - There is no `tz` parameter: naive times of day raise; localize them
      first. Naive midnight dates are fine.
    """
    index = _labels_of(values)
    if len(index) < 2:
        return pd.Series(np.full(max(len(index) - 1, 0), REGULAR), index=index[1:], dtype=object)
    # Continuous mode: regular across one grid step, irregular across more
    if calendar is None:
        _check_increasing(index, "mark labels")
        classes = _continuous_classes(index)
        _warn_irregular_share(classes, stacklevel=2)
        return classes
    # Daily and longer: regular, or irregular across a missing bar
    kind = _series_kind(index, calendar)
    if kind == UNCLASSIFIABLE:
        raise ValueError(_UNCLASSIFIABLE_MESSAGE)
    if kind == DAILY:
        return pd.Series(np.where(_daily_missing(index, calendar), IRREGULAR, REGULAR),
                         index=index[1:], dtype=object)
    # Intraday: where each return's end mark sits in its session
    info = _mark_sessions(index, calendar)
    session = info["session"].to_numpy()
    new_session = session[1:] != session[:-1]
    at_open = info["at_open"].to_numpy()[1:]
    zero_gap = info["zero_gap"].to_numpy()[1:]
    # A session's first mark ends a gap only where the market was closed (O_d != C_{d-1}),
    # else its first bar is regular; a first mark after the open is irregular
    first_class = np.where(at_open, GAP, np.where(zero_gap, REGULAR, IRREGULAR))
    classes = np.where(new_session, first_class, REGULAR)
    # Gaps can be told apart only if some session that has one starts at its open
    needs_gap = new_session & ~zero_gap
    if needs_gap.any() and not (needs_gap & at_open).any():
        raise ValueError(
            "Intraday values with no open marks: no session after the first starts with a mark at its open, "
            "so the overnight gaps can't be told apart. Pass classes, or use to_session_values for a daily series"
        )
    # A return across a scheduled mark missing from the series is irregular
    classes = np.where(_intraday_missing(index, info, calendar), IRREGULAR, classes)
    return pd.Series(classes, index=index[1:], dtype=object)


def _daily_missing(index, calendar):
    """Return, per return of a daily-or-longer series, whether a bar is missing across it:
    between consecutive bar closes only (a leading fill mark skips nothing)."""
    if _is_midnight(index):
        dates = (index.tz_localize(None) if index.tz is not None else index).as_unit("ns")
        cal = _get_calendar(calendar, dates.min(), dates.max(), _DATE_PAD)
        return _dated_missing(dates, cal)
    # Times of day: compare bar closes only, leaving out a leading fill mark
    info = _mark_sessions(index, calendar)
    fill_mark = not info["at_close"].iloc[0]
    sessions = pd.DatetimeIndex(info["session"]).as_unit("ns")
    cal = _get_calendar(calendar, sessions.min(), sessions.max(), _DATE_PAD)
    closes = sessions[1:] if fill_mark else sessions
    missing = _dated_missing(closes, cal) if len(closes) > 1 else np.zeros(0, dtype=bool)
    return np.r_[False, missing] if fill_mark else missing


def _intraday_missing(index, info, calendar):
    """Return, per return of an intraday series, whether a scheduled mark is missing across it, on the
    grid learned from the marks themselves."""
    cal = _get_calendar(calendar, index.min().tz_convert(None), index.max().tz_convert(None), _INTRADAY_PAD)
    wall = index.tz_convert(cal.tz).tz_localize(None)
    session = info["session"].to_numpy()
    first_in_session = np.r_[True, session[1:] != session[:-1]]
    wall_open = pd.DatetimeIndex(info["open"]).tz_convert(cal.tz).tz_localize(None)
    step, phase = _grid(wall, first_in_session, info["at_close"].to_numpy(), None, wall_open)
    # The marks' positions were found on a calendar built for the same
    # span; re-find the two ends on this one
    ends = pd.DatetimeIndex([info["session"].iloc[0], info["session"].iloc[-1]]).as_unit(cal.sessions.unit)
    first_pos, last_pos = cal.sessions.get_indexer(ends)
    return _missing_between(index, _scheduled_intraday(cal, first_pos, last_pos, step, phase))


def _utc_day_values(values):
    """Return a continuous-mode value series collapsed to its first value and
    the value at the last mark at or before each 00:00 UTC, each
    labelled at that midnight, in the series' timezone."""
    index = values.index
    if index.tz is None:
        raise ValueError(
            "Continuous-mode labels are naive, so their 00:00 UTC boundaries are unknown; "
            "localize them first, e.g. values.tz_localize('UTC')"
        )
    # Every 00:00 UTC inside the series, and the last mark at or before each
    utc = index.tz_convert("UTC")
    boundaries = pd.date_range(utc[0].ceil("D"), utc[-1].floor("D"), freq="D")
    last_mark = utc.searchsorted(boundaries, side="right") - 1
    days = pd.Series(values.to_numpy()[last_mark], index=boundaries.tz_convert(index.tz), name=values.name)
    # Prepend the first value, unless it already sits on a midnight
    if len(boundaries) and boundaries[0] == utc[0]:
        out = days
    else:
        out = pd.concat([values.iloc[:1], days])
    out.index.name = index.name
    return out


def to_session_values(values, calendar="XNYS", *, now=None):
    """Collapse a value series to one value per session, its last mark's
    value labelled at the session's close, plus the first mark.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        The value series, oldest first. Intraday labels must be
        timezone-aware.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.
    now : datetime-like, optional, keyword only
        The current time, for leaving out a session still in progress. None
        uses the clock; a naive `now` is read in this machine's local
        timezone, as in `simulate_buy_and_hold`.

    Returns
    -------
    pandas.Series
        The first value (the fill, ``br0``), then each session's last value.
        Its returns are the session returns R_d of the risk figures, so a daily forecast
        from intraday history agrees with the risk figures. A daily or
        longer series comes back unchanged. A series with one mark per
        session away from the close (e.g. 12:00 every session) is accepted,
        and each session's last value is relabelled at that session's close,
        which makes it daily. On an intraday series, a session whose last
        mark is before its close (its final bars are missing, e.g. a
        half-day without its last bar) also has its last value labelled
        at its close. An intraday series' last session whose close is
        after `now` is still in progress and is left out, without a
        warning; only sessions closed by `now` are relabelled.

        Continuous mode: the first value, then the value at the last mark at
        or before each 00:00 UTC within the series, labelled at that
        midnight (in the series' timezone), so its returns are the
        calendar-day returns the risk figures use in continuous mode. A daily series (grid
        step one day) comes back unchanged. A UTC day after the last midnight is left out.

    Raises
    ------
    ValueError
        If its labels are naive times of day, not strictly increasing, or a
        mark falls outside every session. In continuous mode, naive labels
        of a non-daily series raise: their 00:00 UTC is unknown.

    Warns
    -----
    UserWarning
        When intraday sessions closed by `now` end before their close, with
        their count and the first one's date. The one-mark-per-session
        relabel and a session left out as in progress do not warn.

    Notes
    -----
    There is no `tz` parameter: naive times of day raise; localize them
    first. Naive midnight dates are fine.
    """
    out, short = _session_values(values, calendar, now)
    if len(short):
        warnings.warn(
            f"{len(short)} session(s) end before their close, first {short[0]:%Y-%m-%d}: "
            "each one's last value is labelled at its close",
            UserWarning, stacklevel=2,
        )
    return out


def _session_values(values, calendar, now=None, drop_live=True):
    """Return `to_session_values` without its warning, and the sessions
    that end before their close.

    The sessions are naive dates, oldest first; empty unless the series is
    intraday on a calendar and some session closed by `now` has its last
    mark (other than the first mark) before its close. With `drop_live`
    False nothing is left out as in progress and every session counts as
    closed, for marks laid out in the future (simulated paths).
    """
    no_short = pd.DatetimeIndex([])
    if isinstance(values, BacktestResult):
        values = values.values
    index = _labels_of(values)
    if len(index) < 2:
        return values.copy(), no_short
    kind = _series_kind(index, calendar)
    if kind == DAILY:
        return values.copy(), no_short
    if calendar is None:
        return _utc_day_values(values), no_short
    info = _mark_sessions(index, calendar)
    closed = np.ones(len(index), dtype=bool)
    if kind == INTRADAY and drop_live:
        # A naive now is read in local time, as in simulate_buy_and_hold
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if now.tz is None:
            now = now.tz_localize(tzlocal())
        closed = (pd.DatetimeIndex(info["close"]) <= now)
        # The last session still in progress at now is left out, the first mark kept
        live = info["session"].to_numpy() == info["session"].iloc[-1]
        if not closed[-1]:
            live[0] = False
            values, index, info = values[~live], index[~live], info[~live].reset_index(drop=True)
            closed = closed[~live]
    # Keep the first mark and each session's last
    session = info["session"].to_numpy()
    last_in_session = np.r_[session[1:] != session[:-1], True]
    keep = last_in_session.copy()
    keep[0] = True
    out = values[keep].copy()
    # Each session's last value is labelled at its close: always for one
    # mark per session, and on intraday series for the sessions whose
    # final bars are missing; the first mark keeps its label
    relabel = last_in_session.copy()
    relabel[0] = relabel[0] and kind == UNCLASSIFIABLE
    short = no_short
    if kind == INTRADAY:
        relabel &= ~info["at_close"].to_numpy() & closed
        short = pd.DatetimeIndex(info["session"][relabel])
    if relabel.any():
        closes = pd.DatetimeIndex(info["close"]).tz_convert(index.tz)
        out.index = pd.DatetimeIndex(np.where(relabel, closes, index)[keep]).tz_convert(index.tz)
        out.index.name = index.name
    return out, short


# --- Future layout ---------------------------------------------

def _empty_layout(dates):
    """Return a FutureLayout with no marks, labelled like `dates`."""
    return FutureLayout(dates[:0], np.array([], dtype=object), np.array([], dtype=float))


def _horizon_calendar(calendar, first, last, enough):
    """Return a calendar from just before naive date `first` and far enough
    past `last` that ``enough(cal)`` holds.

    exchange_calendars builds only about a year ahead by default; later
    sessions come from the exchange's standing rules, so the end is pushed
    out, doubling each time, until the horizon fits.
    """
    extra = pd.Timedelta(days=400)
    while True:
        cal = xcals.get_calendar(calendar, start=first - pd.Timedelta(days=10), end=last + extra)
        if enough(cal):
            return cal
        extra *= 2


def _next_sessions(dates, n, step, cal):
    """Return every `step`-th session after the last of naive `dates`, `n` of
    them."""
    last = dates[-1]
    after = cal.sessions[cal.sessions > last]
    # If the last date is itself a session, the next bar is `step` sessions
    # on from it; otherwise the first session after it counts as step one
    offset = step - 1 if last in cal.sessions else 0
    return pd.DatetimeIndex(after[offset::step][:n])


def _next_periodic(dates, n, kind, step, cal):
    """Return `n` weekly or monthly bars after naive `dates`, each dated with
    the first or the last session of its period, as the input is (majority
    vote; a one-session period counts for both)."""
    # Each period's first and last session, and which one the history's dates use
    freq = 'W-SUN' if kind == 'week' else 'M'
    sessions = pd.Series(cal.sessions, index=cal.sessions.to_period(freq))
    firsts, lasts = sessions.groupby(level=0).min(), sessions.groupby(level=0).max()
    periods = dates.to_period(freq)
    first_votes = sum(firsts.get(p) == d for p, d in zip(periods, dates, strict=True))
    last_votes = sum(lasts.get(p) == d for p, d in zip(periods, dates, strict=True))
    # Every step-th period after the last, dated with that session
    anchor = firsts if first_votes >= last_votes else lasts
    future = anchor[anchor.index > periods[-1]]
    return pd.DatetimeIndex(future.iloc[step - 1::step].iloc[:n].values)


def _future_bars(dates, n, kind, step, calendar):
    """Return the session dates of the next `n` daily-or-longer bars after
    naive session `dates`, and the calendar, extended until they fit."""
    def bars(cal):
        """Return the next `n` bars on `cal`, inferring the bar size if not given."""
        if kind is None:
            k, st = _infer_date_bars(dates, cal)
        else:
            k, st = kind, step
        return _next_sessions(dates, n, st, cal) if k == 'session' else _next_periodic(dates, n, k, st, cal)

    cal = _horizon_calendar(calendar, dates[0], dates[-1], lambda cal: len(bars(cal)) >= n)
    return bars(cal), cal


def _daily_layout(index, n, kind, step, calendar, tz):
    """Return the future marks of daily and longer history: `n` bars, all
    regular, each a full unit, labelled like the history (dates, or
    session closes)."""
    # Dates: the future bars' session dates, in the history's timezone if it has one
    if _is_midnight(index):
        dates = index.tz_localize(None) if index.tz is not None else index
        if kind is None and len(dates) < 2:
            raise ValueError('Need at least two dates to infer the interval; pass interval= explicitly')
        future, _ = _future_bars(dates, n, kind, step, calendar)
        out = future if index.tz is None else future.tz_localize(index.tz)
    else:
        aware = _localize(index, tz)
        info = _mark_sessions(aware, calendar)
        # Bar closes only: an initial fill mark away from a close is not a bar
        bar_marks = info if info["at_close"].iloc[0] else info.iloc[1:]
        dates = pd.DatetimeIndex(bar_marks["session"]).unique()
        if len(dates) == 0 or (kind is None and len(dates) < 2):
            raise ValueError('Need at least two bar closes to infer the interval; pass interval= explicitly')
        # Times of day: each future bar labelled at its session's close
        future, cal = _future_bars(dates, n, kind, step, calendar)
        closes = pd.DatetimeIndex(cal.closes.reindex(future.as_unit(cal.sessions.unit)))
        out = closes.tz_convert(aware.tz)
        if index.tz is None:
            out = out.tz_localize(None)
    return FutureLayout(out.rename(None), np.full(len(out), REGULAR, dtype=object), np.ones(len(out)))


def _grid(wall, first_in_session, at_close, interval, wall_open=None):
    """Return (step, phase) of the bar grid of intraday marks on the
    exchange's wall clock: the bar size, and the offset of bar closes from
    midnight.

    The step comes from spacings between consecutive marks in a session,
    leaving out spacings from a session's first mark (its first bar can be
    partial, as 9:30 to 10:00 under hourly bars) and, where others remain,
    spacings ending at a session close (a short last bar, as 12:30 to 13:00
    on a half-day); see `calendar._choose_step`, which also uses the
    sessions' opens `wall_open` (one per mark) to settle sparse data. The
    phase comes from bar closes that aren't a session close, which can be
    off the grid (16:00 under 9:30-phased hourly bars).
    """
    # Spacings within a session; `later` leaves out each session's first spacing
    spacing = pd.Series(wall).diff()
    same = ~first_in_session
    later = same & np.r_[False, same[:-1]]
    # Closes that aren't a session close sit on the grid and give its phase
    since_midnight = pd.Series(wall - wall.normalize())
    on_grid = same & ~at_close
    sample = since_midnight[on_grid] if on_grid.any() else since_midnight[same]
    # A given interval fixes the step; only the phase is learned
    if interval is not None:
        step = interval
        phase = (sample % step).mode().min() if len(sample) else pd.Timedelta(0)
        return step, phase
    # Step candidates: full-bar spacings, falling back to any within a session
    full = later & ~at_close
    candidates = spacing[full] if full.any() else spacing[later] if later.any() else spacing[same]
    if candidates.empty:
        raise ValueError("Can't infer the bar size from one mark per session after the open; pass interval=")
    if wall_open is None:
        open_offsets = pd.Series([], dtype="timedelta64[ns]")
    else:
        wall_open = pd.DatetimeIndex(wall_open)
        open_offsets = pd.Series(wall_open - wall_open.normalize())
    first = spacing[same & ~later]
    return _choose_step(candidates, first, sample, open_offsets)


def _session_grid(session_open, session_close, step, phase):
    """Return the bar closes of one session on the wall clock: grid points
    strictly after the open and before the close, then the close itself (a
    short last bar, as on a half-day, ends at the session close)."""
    # The first grid point strictly after the open, then every step until the close
    base = session_open.normalize() + phase
    first = base + step * ((session_open - base) // step + 1)
    points = pd.date_range(first, session_close, freq=step, inclusive='left')
    return points.append(pd.DatetimeIndex([session_close]))


def _intraday_layout(index, n, interval, calendar, tz):
    """Return the future marks of intraday history: `n` sessions, the first
    of them the rest of the last session's bars if the history ends
    mid-session, the others each an open mark (gap) and its bar
    closes on the history's grid; tau is each return's length over the bar
    size, 1.0 for gaps."""
    # The history's last session, and a calendar reaching n sessions past it
    aware = _localize(index, tz)
    info = _mark_sessions(aware, calendar)
    session = pd.DatetimeIndex(info["session"])
    last_session = session[-1]
    cal = _horizon_calendar(calendar, last_session, last_session, lambda cal: (cal.sessions > last_session).sum() >= n)
    # The history's bar grid (step and phase) on the exchange's wall clock
    wall = aware.tz_convert(cal.tz).tz_localize(None)
    first_in_session = np.r_[True, session[1:] != session[:-1]]
    step, phase = _grid(wall, first_in_session, info["at_close"].to_numpy(), interval,
                        pd.DatetimeIndex(info["open"]).tz_convert(cal.tz).tz_localize(None))
    opens, closes = _session_times(cal)
    wall_open, wall_close = opens.tz_localize(None), closes.tz_localize(None)
    pos = cal.sessions.get_indexer(pd.DatetimeIndex([last_session]).as_unit(cal.sessions.unit))[0]

    dates, classes, tau = [], [], []

    def emit(points, start, first_class):
        """Append marks `points`; each return runs from the mark before it,
        the first from `start`, with class `first_class`, the rest regular."""
        previous = start
        for i, t in enumerate(points):
            dates.append(t)
            classes.append(first_class if i == 0 else REGULAR)
            # A gap is one unit whatever its clock length; a bar's tau
            # is its length over the bar size
            tau.append(1.0 if (i == 0 and first_class == GAP) else (t - previous) / step)
            previous = t

    # A history ending mid-session d: its remaining bars come first and count as the
    # first of the n sessions, so the horizon ends at the close of session d + n - 1
    last = wall[-1]
    remaining = _session_grid(wall_open[pos], wall_close[pos], step, phase)
    remaining = remaining[remaining > last]
    emit(remaining, last, REGULAR)
    n_full = n - 1 if len(remaining) else n
    # Then full sessions: an open mark (the gap), then the bar closes; a session opening
    # the instant the last one closed has no gap, so its first bar is regular
    for p in range(pos + 1, pos + 1 + n_full):
        grid = _session_grid(wall_open[p], wall_close[p], step, phase)
        if opens[p] == closes[p - 1]:
            emit(grid, wall_open[p], REGULAR)
        else:
            emit(pd.DatetimeIndex([wall_open[p]]).append(grid), wall_open[p], GAP)

    # Label the marks like the history: its timezone, or naive local wall clock
    out = pd.DatetimeIndex(dates).tz_localize(cal.tz)
    out = out.tz_convert(tz or tzlocal()).tz_localize(None) if index.tz is None else out.tz_convert(index.tz)
    return FutureLayout(out, np.array(classes, dtype=object), np.array(tau, dtype=float))


def _continuous_layout(index, n, interval):
    """Return the future marks of continuous-mode history:
    t_T + kΔ for k = 1..n, all regular, each a full grid step."""
    # The grid step: learned from the history, or the given fixed bar size
    if interval is None:
        step = _continuous_step(index)
    else:
        kind, size = _parse_interval(interval)
        if kind == "month":
            raise ValueError(
                f"interval {interval!r}: months have no constant length, so continuous mode "
                "(calendar=None) can't step by them; pass a fixed bar size"
            )
        # No sessions here, so a "session" is a calendar day
        step = {"intraday": lambda: size, "session": lambda: size * _ONE_DAY,
                "week": lambda: 7 * size * _ONE_DAY}[kind]()
    dates = pd.DatetimeIndex([index[-1] + k * step for k in range(1, n + 1)])
    return FutureLayout(dates, np.full(n, REGULAR, dtype=object), np.ones(n))


def _future_layout(index, n, interval=None, calendar="XNYS", tz=None):
    """Return the future marks continuing the history `index`:
    see `next_trading_dates`. `index` is sorted, unique and non-empty."""
    if n <= 0:
        return _empty_layout(index)
    if calendar is None:
        return _continuous_layout(index, n, interval)
    # Dates are daily or longer; an intraday interval can't apply to them
    kind, step = _parse_interval(interval) if interval is not None else (None, None)
    if _is_midnight(index):
        if kind == 'intraday':
            raise ValueError(f"interval {interval!r} is intraday but the history is dates")
        return _daily_layout(index, n, kind, step, calendar, tz)
    # No layout, hence no classes, for a series that is neither daily nor intraday
    series_kind = _series_kind(_localize(index, tz), calendar)
    if series_kind == UNCLASSIFIABLE:
        raise ValueError(_UNCLASSIFIABLE_MESSAGE)
    # A given interval decides the layout; otherwise the series' own kind does
    if kind == 'intraday':
        return _intraday_layout(index, n, step, calendar, tz)
    if kind is not None or series_kind == DAILY:
        return _daily_layout(index, n, kind, step, calendar, tz)
    return _intraday_layout(index, n, None, calendar, tz)
