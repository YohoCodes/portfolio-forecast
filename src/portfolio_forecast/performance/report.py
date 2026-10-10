"""Reports: performance and risk figures of one value series
(`performance_report`), and of many simulated paths (`statistical_report`).

Flow of `performance_report`:

1. Resolve the input: a Series or `BacktestResult` (whose classes it uses),
   with the deprecated `dates` keyword as labels; fewer than two marks
   raise.
2. Returns: per mark (`calculate_period_returns`) and per session
   (`calculate_session_returns`, full sessions only, the risk series), annualized
   with `_risk_periods_per_year` (252, 52, 12, or 365 in continuous mode).
3. Figures from `metrics`: CAGR and Total Years on `cagr_clock`,
   volatility, Sharpe, downside volatility and Sortino on the session
   returns, Gap and Bar Volatility, max drawdown over every
   mark and the yearly returns.
4. Assemble the dict and optionally print it.

Flow of `statistical_report` (the same rules on every simulated path):

1. `_resolve_paths` reads the paths and their marks: a SimulationResult's
   future marks and classes, with column 0's label (the last historical
   mark) recovered by `_layout_origin`; or a plain array with `dates`, or
   with only a bar size.
2. The marks fix, once for all paths, which columns give the session
   returns (`returns._session_positions`), which returns enter Gap
   and Bar Volatility (`metrics._class_masks`), and the years Y.
3. Each figure is computed on every path at once (numpy, one row per
   path), with the definitions of the `metrics` functions.
4. `_range_summary` gives each figure's mean, median and range across
   paths, inf-safe for infinite Sortino ratios; `_warn_paths`
   warns about infinite Sortino ratios, total losses and paths left
   out of a summary.
"""
import warnings

import numpy as np
import pandas as pd

from ..types import BacktestResult, SimulationResult
from ..utils.calendar import (
    _DATE_PAD,
    _INTRADAY_PAD,
    _check_increasing,
    _continuous_step,
    _get_calendar,
    _infer_date_bars,
    _is_midnight,
    _session_times,
)
from ..utils.marks import DAILY, _grid, _mark_sessions, _series_kind, _session_grid, classify_returns
from ..utils.periods import _parse_interval, _periods_per_year
from .metrics import (
    CAGR_CLOCKS,
    DAYS_PER_YEAR,
    _class_masks,
    _class_std,
    _class_volatility,
    _scaled_bars,
    _series_class_masks,
    calculate_actual_yearly_return,
    calculate_cagr,
    calculate_downside_deviation,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_years,
    calculate_volatility,
)
from .returns import (
    _risk_periods_per_year,
    _session_positions,
    calculate_daily_return,
    calculate_period_returns,
    calculate_session_returns,
)

__all__ = [
    "calculate_actual_yearly_return", "calculate_cagr", "calculate_daily_return",
    "calculate_max_drawdown", "calculate_sharpe_ratio", "calculate_sortino_ratio",
    "performance_report", "statistical_report",
]


def _resolve_report_input(values, classes, dates):
    """Return (values, classes) for `performance_report`: the value Series
    with a DatetimeIndex, and the classes (given, a BacktestResult's, or
    None). `dates` is the deprecated 0.3.0 keyword.

    Raises ValueError for fewer than two marks or `dates` of the wrong
    length, and TypeError for values without a DatetimeIndex.
    """
    if isinstance(values, BacktestResult):
        if classes is None:
            classes = values.classes
        values = values.values
    values = pd.Series(values, dtype=float)
    if dates is not None:
        warnings.warn(
            "performance_report's dates is deprecated and will be removed in 1.0.0: the labels come from "
            "values' index (or a BacktestResult)",
            DeprecationWarning, stacklevel=3,
        )
        dates = pd.DatetimeIndex(dates)
        if len(dates) != len(values):
            raise ValueError(f"dates has {len(dates)} entries; values has {len(values)}")
        if not isinstance(values.index, pd.DatetimeIndex):
            values.index = dates
    if len(values) < 2:
        raise ValueError(f"performance_report needs at least two marks (values has {len(values)})")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("values needs a DatetimeIndex of mark labels (or pass a BacktestResult)")
    _check_increasing(values.index, "values' labels")
    return values, classes


def _utc_boundaries(index):
    """Return the number of 00:00 UTC instants from the first mark to the last,
    inclusive; continuous mode needs at least two for volatility, Sharpe and Sortino."""
    utc = index.tz_convert("UTC") if index.tz is not None else index
    return len(pd.date_range(utc[0].ceil("D"), utc[-1].floor("D"), freq="D"))


def _total_time_years(labels):
    """Return the printed Total Time: calendar years from the first label to the last.

    Display only, the same on every series kind; CAGR keeps `cagr_clock`.
    """
    return (labels[-1] - labels[0]) / pd.Timedelta(days=DAYS_PER_YEAR)


def _class_rows(masks):
    """Return (gap row, bar row): whether the printed report shows Gap and Bar Volatility.

    Decided from the classes, not from NaN values: Gap Vol needs at least
    one gap return, Bar Vol an intraday series with regular bars.
    """
    if masks is None:
        return False, False
    gap, regular, _ = masks
    return gap is not None and bool(gap.any()), bool(regular.any())


def performance_report(values, classes=None, *, interval=None, calendar="XNYS", risk_free_rate=0.0,
                       cagr_clock="session", display=False, dates=None):
    """Return performance and risk figures for one portfolio value series.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values at every mark, oldest first, labelled with
        timezone-aware moments (dates are fine for daily and longer series),
        e.g. from `simulate_buy_and_hold`. A BacktestResult also supplies the
        classes.
    classes : pandas.Series, optional
        Class of each return (``"gap"``, ``"regular"``, ``"irregular"``),
        indexed by the mark where it ends. None takes a BacktestResult's, or
        infers them with `portfolio_forecast.utils.classify_returns`. Used
        for Gap and Bar Volatility only.
    interval : str or pandas.Timedelta, optional
        Bar size, in any spelling `next_trading_dates` accepts. None infers
        it from the labels. On daily and longer series it picks the
        annualization (``'1 day'``: 252, ``'1 week'``: 52, ``'1 month'``:
        12); on intraday series it is the bar size Δ that short bars are
        scaled to.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.
    risk_free_rate : float, default 0.0
        Annual risk-free rate, as a fraction, spread evenly over the year's
        sessions (r_f / 252; r_f / 52 or r_f / 12 on weekly or monthly
        bars; r_f / 365 in continuous mode).
    cagr_clock : {"session", "calendar"}, default "session"
        Clock of CAGR and Total Years: ``"session"``, Y = N / 252 with N
        the session closes after the first mark; ``"calendar"``,
        Y = (t_T - t_0) / 365.25 days. Continuous mode needs ``"calendar"``.
    display : bool, default False
        If True, also print a formatted report.
    dates : array-like of datetimes, optional
        .. deprecated:: 0.4.0
            The labels come from `values`; removed in 1.0.0. If given, they
            label a `values` that has no DatetimeIndex.

    Returns
    -------
    dict
        ``"Period Returns"``
            pandas.Series of returns between consecutive marks, NaN
            after a total loss.
        ``"Daily Returns"``
            pandas.Series of session returns, the series behind the risk
            figures: full sessions only; weekly and monthly bar
            returns on those series; UTC-day returns labelled at 00:00 UTC
            in continuous mode. NaN after a total loss; the risk
            figures drop those rows.
        ``"Periods"``
            Number of returns between marks, ``len(values) - 1``.
        ``"Total Days"``
            Whole calendar days from the first mark to the last.
        ``"Total Years"``
            Y on `cagr_clock`.
        ``"Returns"``
            dict with ``"Initial Bankroll"``, ``"Final Bankroll"``,
            ``"Total Return"``, ``"CAGR"`` (-100% at a total loss) and
            ``"Actual Yearly Returns"`` (a DataFrame by year with columns
            ``return``, ``days`` and ``is_full``).
        ``"Risk"``
            dict with ``"Gap Volatility"`` and ``"Bar Volatility"`` (per
            unit, unannualized, NaN on daily data), ``"Annualized
            Volatility"``, ``"Downside Volatility"`` (annualized),
            ``"Max Drawdown"`` (a fraction in [0, 1] over every mark),
            ``"Sharpe Ratio"`` and ``"Sortino Ratio"``.

    Raises
    ------
    ValueError
        If `values` has fewer than two marks, `cagr_clock` is unknown
        or ``"session"`` in continuous mode, `interval` can't be parsed, has
        no annualization factor or is intraday on a daily or longer series,
        or the labels can't be placed on the calendar.
    TypeError
        If `values` has no DatetimeIndex.

    Warns
    -----
    UserWarning
        When no session return is below the risk-free rate (Sortino is then
        infinite, or NaN if every session returns exactly the risk-free
        rate); in continuous mode, when the series spans fewer than
        two 00:00 UTC boundaries, so it has no full UTC day and volatility,
        Sharpe and Sortino are NaN.
    DeprecationWarning
        If `dates` is given.

    Notes
    -----
    - Volatility, Sharpe and Sortino use session returns R_d = V(C_d) /
      V(C_{d-1}) - 1 annualized with 252 (52 or 12 for weekly or monthly
      bars, 365 in continuous mode), never per-mark returns, since gap and
      bar returns are not identically distributed. A partial first
      session is left out, so these figures are the same under both fills.
      Total return, CAGR and drawdown use every mark from br0.
    - ``cagr_clock="calendar"`` annualizes CAGR on a different clock from
      the volatility (calendar days against sessions).
    - An intraday backtest's max drawdown includes the overnight gaps, so
      it is at least the daily backtest's for the same book.

    See Also
    --------
    statistical_report : Figures across many simulated paths.
    """
    values, classes = _resolve_report_input(values, classes, dates)
    index = values.index
    # Fail on an unreadable interval before any figure is computed
    if interval is not None:
        _parse_interval(interval)

    # Returns per mark, per full session, and the factor that annualizes the latter
    period_returns = calculate_period_returns(values)
    session_returns = calculate_session_returns(values, calendar)
    periods_per_year = _risk_periods_per_year(values, calendar, interval)

    # Growth from br0 over every mark: total return, Y on the chosen clock, CAGR, yearly returns
    initial_bankroll = float(values.iloc[0])
    final_bankroll = float(values.iloc[-1])
    total_return = final_bankroll / initial_bankroll - 1
    n_periods = len(values) - 1
    total_days = (index[-1] - index[0]).days
    total_years = calculate_total_years(values, calendar, cagr_clock)
    # With Y = 0 there is no span to annualize over, so CAGR is NaN
    cagr = calculate_cagr(values, years=total_years) if total_years > 0 else np.nan
    yearly_returns = calculate_actual_yearly_return(values, calendar=calendar)

    # In continuous mode, fewer than two 00:00 UTC boundaries leave no full UTC day:
    # volatility, Sharpe and Sortino are NaN with a warning, not estimated from nothing
    no_full_day = calendar is None and _utc_boundaries(index) < 2
    if no_full_day:
        warnings.warn(
            f"values spans {_utc_boundaries(index)} 00:00 UTC boundary(ies), fewer than two, so it has no full "
            "UTC day: volatility, Sharpe and Sortino are NaN",
            UserWarning, stacklevel=2,
        )
        ann_vol = sharpe = sortino = ann_downside_vol = np.nan
    else:
        ann_vol = calculate_volatility(session_returns, periods_per_year)
        sharpe = calculate_sharpe_ratio(session_returns, periods_per_year, risk_free_rate)
        ann_downside_vol = calculate_downside_deviation(session_returns, periods_per_year, risk_free_rate) \
            * np.sqrt(periods_per_year)
        sortino = calculate_sortino_ratio(session_returns, periods_per_year, risk_free_rate)
    # Per-mark figures: gap and bar volatility per unit, and drawdown over every mark
    masks = _series_class_masks(values, classes, calendar, interval)
    class_vol = _class_volatility(values, masks)
    max_drawdown = calculate_max_drawdown(values)

    report = {
        "Period Returns": period_returns,
        "Daily Returns": session_returns,
        "Periods": n_periods,
        "Total Days": total_days,
        "Total Years": total_years,
        "Returns": {
            "Initial Bankroll": initial_bankroll,
            "Final Bankroll": final_bankroll,
            "Total Return": total_return,
            "CAGR": cagr,
            "Actual Yearly Returns": yearly_returns,
        },
        "Risk": {
            "Gap Volatility": class_vol["Gap Volatility"],
            "Bar Volatility": class_vol["Bar Volatility"],
            "Annualized Volatility": ann_vol,
            "Downside Volatility": ann_downside_vol,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
        },
    }

    if display:
        _print_performance_report(report, _total_time_years(index), *_class_rows(masks))
    return report


def _print_performance_report(report, total_time, gap_row, bar_row):
    """Print a `performance_report` dict as a formatted table.

    `total_time` is the header's calendar years; `gap_row` and `bar_row`
    say whether the series has the class behind each volatility row.
    """
    separator = "=" * 52
    sub_separator = "-" * 52
    returns, risk = report["Returns"], report["Risk"]
    print("\n" + separator)
    print(f"{'PORTFOLIO PERFORMANCE REPORT':^52}")
    print(sub_separator)
    # Calendar years, then the whole calendar days they span
    print(f"  Total Time         : {total_time:>5.2f} years ({report['Total Days']} days)")
    print(f"  Periods            : {report['Periods']:>5} returns")
    print(separator)
    print(f"{'RETURNS':^52}")
    print(sub_separator)
    print(f"  Initial Bankroll   : {returns['Initial Bankroll']:>12.2f}")
    print(f"  Final Bankroll     : {returns['Final Bankroll']:>12.2f}")
    print(f"  Total Return       : {returns['Total Return']:>12.2%}")
    print(f"  CAGR               : {returns['CAGR']:>12.2%}")
    print(sub_separator)
    print("  Actual Yearly Returns:")
    for year, row in returns["Actual Yearly Returns"].iterrows():
        span = "full year" if row["is_full"] else f"{int(row['days'])} days"
        print(f"    - {year}            : {row['return']:>12.2%}  ({span})")
    print(separator)
    print(f"{'RISK':^52}")
    print(sub_separator)
    print(f"  Volatility         : {risk['Annualized Volatility']:>12.2%}")
    print(f"  Downside Vol       : {risk['Downside Volatility']:>12.2%}")
    # Only the classes the series has: no gap row without gaps, no bar row on daily data
    if gap_row:
        print(f"  Gap Vol (per gap)  : {risk['Gap Volatility']:>12.2%}")
    if bar_row:
        print(f"  Bar Vol (per bar)  : {risk['Bar Volatility']:>12.2%}")
    print(f"  Max Drawdown       : {risk['Max Drawdown']:>12.2%}")
    print(f"  Sharpe Ratio       : {risk['Sharpe Ratio']:>12.2f}")
    print(f"  Sortino Ratio      : {risk['Sortino Ratio']:>12.2f}")
    print(separator + "\n")


# --- statistical_report ------------------------------------------------------

# Per-path figures, in the column order of "Per Path Metrics"
_RETURN_METRICS = ("Final Bankroll", "Total Return", "CAGR")
_RISK_METRICS = ("Gap Volatility", "Bar Volatility", "Annualized Volatility", "Downside Volatility",
                 "Max Drawdown", "Sharpe Ratio", "Sortino Ratio")
# Figures whose undefined (NaN) paths are counted in the "left out" warning;
# Gap and Bar Volatility are NaN by definition on daily data
_COUNTED_METRICS = ("CAGR", "Annualized Volatility", "Downside Volatility", "Max Drawdown",
                    "Sharpe Ratio", "Sortino Ratio")


def _continuous_origin(dates, bar):
    """Return t_1 − Δ, the mark before the first continuous-mode future mark.

    Δ comes from `bar` (a parsed interval) or the spacing of `dates`. Raises
    ValueError for one future mark with no `bar`, or for monthly bars,
    which have no constant length.
    """
    if bar is None:
        if len(dates) < 2:
            raise ValueError("Can't infer the grid step from one future mark; pass interval=")
        return dates[0] - _continuous_step(dates)
    kind, size = bar
    if kind == "month":
        raise ValueError("Months have no constant length, so continuous mode (calendar=None) can't step by them")
    # No sessions here, so a "session" is a calendar day (as in next_trading_dates)
    step = {"intraday": size, "session": size * pd.Timedelta(days=1), "week": 7 * size * pd.Timedelta(days=1)}[kind]
    return dates[0] - step


def _daily_origin(dates, bar, calendar):
    """Return the label of the bar before the first daily-or-longer future mark.

    It is labelled like `dates`: a date, or its session's close.

    n-session bars are start-labelled, so the previous bar starts n sessions
    earlier; weekly and monthly bars are dated with the first or the last
    session of their period, as `dates` are (majority vote, as
    `next_trading_dates` labels them).
    """
    # The future bars' session dates, and the bar size if not given
    midnight = _is_midnight(dates)
    if midnight:
        sessions = (dates.tz_localize(None) if dates.tz is not None else dates).as_unit("ns")
    else:
        sessions = pd.DatetimeIndex(_mark_sessions(dates, calendar)["session"]).as_unit("ns")
    cal = _get_calendar(calendar, sessions.min(), sessions.max(), _DATE_PAD)
    if bar is None:
        if len(sessions) < 2:
            raise ValueError("Can't infer the bar size from one future mark; pass interval=")
        bar = _infer_date_bars(sessions, cal)
    kind, step = bar
    # The previous bar: step sessions back, or the anchor session of the previous period
    all_sessions = cal.sessions.as_unit("ns")
    if kind == "session":
        previous = all_sessions[all_sessions.searchsorted(sessions[0]) - step]
    else:
        freq = "W-SUN" if kind == "week" else "M"
        by_period = pd.Series(all_sessions, index=all_sessions.to_period(freq))
        firsts, lasts = by_period.groupby(level=0).min(), by_period.groupby(level=0).max()
        periods = sessions.to_period(freq)
        first_votes = sum(firsts.get(p) == d for p, d in zip(periods, sessions, strict=True))
        last_votes = sum(lasts.get(p) == d for p, d in zip(periods, sessions, strict=True))
        anchor = firsts if first_votes >= last_votes else lasts
        previous = anchor[periods[0] - step]
    # Labelled like `dates`: a date, or that session's close
    if midnight:
        return previous if dates.tz is None else previous.tz_localize(dates.tz)
    return pd.Timestamp(cal.closes[previous]).tz_convert(dates.tz)


def _intraday_origin(dates, bar, calendar):
    """Return the mark before the first intraday future mark of `dates`.

    In the first mark's session d the marks run C_{d-1}, then O_d (the open
    mark, left out when O_d = C_{d-1}), then the bar closes of d on
    the grid of `dates`. The mark before the first future mark is the last
    of those before it: C_{d-1} for an open mark (a gap), the previous bar
    close (or O_d) for a mid-session start.
    """
    # The future marks' bar grid on the exchange's wall clock
    info = _mark_sessions(dates, calendar)
    cal = _get_calendar(calendar, dates.min().tz_convert(None), dates.max().tz_convert(None), _INTRADAY_PAD)
    opens, closes = _session_times(cal)
    wall = dates.tz_convert(cal.tz).tz_localize(None)
    session = info["session"].to_numpy()
    first_in_session = np.r_[True, session[1:] != session[:-1]]
    wall_open = pd.DatetimeIndex(info["open"]).tz_convert(cal.tz).tz_localize(None)
    step, phase = _grid(wall, first_in_session, info["at_close"].to_numpy(),
                        None if bar is None else bar[1], wall_open)
    # Session d's marks: C_{d-1}, its open mark if it has a gap, then its bar closes
    pos = cal.sessions.get_indexer(pd.DatetimeIndex(session[:1]).as_unit(cal.sessions.unit))[0]
    grid = _session_grid(opens[pos].tz_localize(None), closes[pos].tz_localize(None), step, phase)
    marks = pd.DatetimeIndex([closes[pos - 1]])
    if opens[pos] != closes[pos - 1]:
        marks = marks.append(pd.DatetimeIndex([opens[pos]]))
    marks = marks.append(grid.tz_localize(cal.tz))
    # The last of them before the first future mark
    return marks[marks < dates[0]][-1].tz_convert(dates.tz)


def _layout_origin(dates, calendar="XNYS", interval=None):
    """Return the label of column 0 of paths laid out on future marks `dates`.

    A `SimulationResult` carries the future marks of `next_trading_dates`,
    which leave out the last historical mark t_0 that column 0 stands at.
    The layout continues the history's grid from t_0, so t_0 is
    the grid point just before the first future mark; see
    `_continuous_origin`, `_daily_origin` and `_intraday_origin`. The bar
    size comes from `interval`, or is inferred from `dates`.

    Raises ValueError if the bar size or grid can't be inferred (one future
    mark: pass `interval`), or the marks can't be placed on the calendar
    (e.g. naive intraday labels).
    """
    bar = _parse_interval(interval) if interval is not None else None
    if calendar is None:
        return _continuous_origin(dates, bar)
    if bar is None and len(dates) < 2:
        raise ValueError("Can't infer the bar size from one future mark; pass interval=")
    # Daily or intraday: from the given bar size, else from the marks themselves
    daily = bar[0] != "intraday" if bar is not None else _series_kind(dates, calendar) == DAILY
    if _is_midnight(dates) or daily:
        return _daily_origin(dates, bar, calendar)
    return _intraday_origin(dates, bar, calendar)


def _resolve_paths(sims, interval, dates, calendar):
    """Return (paths, labels, classes) for `statistical_report`.

    `paths` is a float array with one path per row. `labels` are the marks
    of its columns (column 0 included), or None when only the bar size is
    known; `classes` the class of each return, or None to infer them from
    `labels`. A SimulationResult supplies its future marks and classes, and
    column 0's label is recovered with `_layout_origin`; `dates`, if given,
    labels every column instead (the 0.3.0 form).
    """
    # The paths as a 2-D array, and a SimulationResult's future marks and classes
    classes = None
    origin_dates = None
    if isinstance(sims, SimulationResult):
        paths = np.asarray(sims.paths, dtype=float)
        classes = sims.classes
        if dates is None and sims.dates is not None:
            origin_dates = pd.DatetimeIndex(sims.dates)
    else:
        paths = np.asarray(sims, dtype=float)
    if paths.ndim != 2:
        raise ValueError(f"sims needs one path per row (a 2-D array), got {paths.ndim} dimension(s)")
    n_points = paths.shape[1]
    if n_points < 2:
        raise ValueError(f"statistical_report needs at least two marks per path (sims has {n_points})")
    # Column labels: column 0's mark recovered before the future marks, or `dates`, or none
    if origin_dates is not None:
        if len(origin_dates) != n_points - 1:
            raise ValueError(
                f"sims.dates has {len(origin_dates)} future marks; expected {n_points - 1} "
                "(one per column of sims.paths after column 0)"
            )
        labels = pd.DatetimeIndex([_layout_origin(origin_dates, calendar, interval)]).append(origin_dates)
    elif dates is not None:
        labels = pd.DatetimeIndex(pd.to_datetime(dates))
        if len(labels) != n_points:
            raise ValueError(
                f"dates has {len(labels)} entries; expected {n_points} "
                "(one per column of sims, starting value included)"
            )
    else:
        if interval is None:
            raise ValueError(
                "interval is required when sims has no marks (a plain array, or a SimulationResult "
                "without dates): it is the bar size of one simulated period"
            )
        labels = None
    if labels is not None:
        _check_increasing(labels, "dates")
    if classes is not None and labels is not None:
        classes = pd.Series(np.asarray(classes, dtype=object), index=labels[1:])
        if len(classes) != n_points - 1:
            raise ValueError(f"sims.classes has {len(classes)} entries; expected {n_points - 1}")
    return paths, labels, classes


def _quantile(sorted_values, q):
    """Return the q-quantile of ascending `sorted_values`, which may hold infinities.

    Linear between order statistics, like numpy's default, but inf-safe:
    between a finite value and an infinity the quantile is that infinity,
    never the NaN that interpolating inf − inf gives. Between −inf and +inf
    it is NaN.
    """
    # The two order statistics around position h, and how far between them q falls
    h = (len(sorted_values) - 1) * q
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    a, b = sorted_values[lo], sorted_values[hi]
    frac = h - lo
    if frac == 0 or a == b:
        return float(a)
    if np.isinf(a) or np.isinf(b):
        # Opposite infinities have no value in between
        return float(a if np.isinf(a) and not np.isinf(b) else b if not np.isinf(a) else np.nan)
    return float(a + (b - a) * frac)


def _range_summary(values, confidence):
    """Return the mean, the median and the central `confidence` range of a figure across paths.

    Lower and Upper are the (1 - confidence) / 2 and (1 + confidence) / 2
    percentiles of the paths' values: where that share of simulated
    outcomes landed under the fitted model, not a confidence interval for
    the figure. NaN paths (figure undefined) are left out. Infinite
    values (a Sortino ratio with no down session) are kept: the mean is
    then infinite (`statistical_report` reports the Sortino mean as NaN),
    and the percentiles are inf-safe (`_quantile`).
    """
    # Defined values only, sorted for the percentiles
    values = np.asarray(values, dtype=float)
    values = np.sort(values[~np.isnan(values)])
    if len(values) == 0:
        return {"Mean": np.nan, "Median": np.nan, "Lower": np.nan, "Upper": np.nan}
    # With infinities, inf-safe percentiles; otherwise numpy's
    tail = (1 - confidence) / 2
    if np.isinf(values).any():
        lower, median, upper = (_quantile(values, q) for q in (tail, 0.5, 1 - tail))
        with np.errstate(invalid="ignore"):
            mean = float(values.mean())
    else:
        lower, median, upper = np.quantile(values, [tail, 0.5, 1 - tail])
        mean = float(values.mean())
    return {"Mean": mean, "Median": median, "Lower": lower, "Upper": upper}


def _row_moments(returns):
    """Return (count, mean, sample std) of each row of `returns`, ignoring NaN.

    Returns after a total loss are 0/0, undefined, so they are left out
    of the count as well as the moments.
    """
    finite = ~np.isnan(returns)
    count = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite, returns, 0.0).sum(axis=1) / count
    std = _class_std(returns)
    return count, mean, std


def _path_risk(session_returns, periods_per_year, risk_free_rate):
    """Return annualized volatility, downside volatility, Sharpe and Sortino of each path.

    `session_returns` has one row per path. The definitions are those of
    `calculate_volatility`, `calculate_downside_deviation`,
    `calculate_sharpe_ratio` and `calculate_sortino_ratio`: sample standard
    deviation (ddof=1), excess return R − r_f / periods_per_year, and every
    session in the downside average. A path with no defined return gets NaN.
    """
    # Per path: defined-return count, σ̂, mean excess return and shortfall below the target
    count, _, std = _row_moments(session_returns)
    excess = session_returns - risk_free_rate / periods_per_year
    _, mean_excess, _ = _row_moments(excess)
    shortfall = np.where(np.isnan(excess), 0.0, np.minimum(excess, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        # Every one of the n sessions enters the average
        downside = np.sqrt((shortfall ** 2).sum(axis=1) / count)
        root = np.sqrt(periods_per_year)
        sharpe = np.where(std > 0, mean_excess / std, np.nan) * root
        # No session below the target gives +inf (NaN if the mean excess is 0)
        sortino = np.where(downside > 0, mean_excess * root / downside,
                           np.where(mean_excess > 0, np.inf, np.nan))
    sortino = np.where(count > 0, sortino, np.nan)
    return std * root, downside * root, sharpe, sortino


def _path_cagr(initial, final, years):
    """Return each path's CAGR, (V_T / V_0)^(1/Y) − 1.

    −100% at a total loss; NaN for a negative growth multiple, or for
    every path when Y is not positive (no span to annualize over).
    """
    growth = final / initial
    cagr = np.full(len(growth), np.nan)
    if not years > 0:
        return cagr
    positive = growth > 0
    cagr[positive] = growth[positive] ** (1 / years) - 1
    cagr[growth == 0] = -1.0
    return cagr


def statistical_report(sims, interval=None, dates=None, risk_free_rate=0.0, confidence=0.95, display=False,
                       *, calendar="XNYS", cagr_clock="session"):
    """Return performance and risk figures summarized across simulated paths.

    Every figure of `performance_report` is computed on each path under the
    same rules, then summarized across paths by its mean, its median and
    the 95% range across paths (for the default `confidence`). Tail risk of
    the total return is added.

    Parameters
    ----------
    sims : SimulationResult or array-like of shape (n_sims, n_marks + 1)
        One value path per row, starting value in column 0, as returned by
        the `portfolio_forecast.forecast` simulators. A SimulationResult
        supplies the future marks (its ``dates``) and their classes; column
        0 stands at the last historical mark, which is recovered from the
        layout (the grid point before the first future mark). A plain
        array needs `interval`, and `dates` for intraday paths.
    interval : str or pandas.Timedelta, optional
        Bar size, in any spelling `next_trading_dates` accepts. Required for
        paths without marks (a plain array without `dates`, or a
        SimulationResult without dates), which are read as daily, weekly or
        monthly bars (annualized with 252, 52 or 12). With marks it is
        optional, as in `performance_report`: the bar size Δ that short bars
        are scaled to, or the annualization of daily and longer bars.
    dates : array-like of datetimes, optional
        ``n_marks + 1`` labels, one per column of `sims`: the last
        historical mark, then the future marks (the 0.3.0 form, e.g.
        ``values.index[-1:].append(layout.dates)``). Given with a
        SimulationResult, they replace its labels. Intraday labels must be
        timezone-aware.
    risk_free_rate : float, default 0.0
        Annual risk-free rate, as a fraction, spread evenly over the year's
        sessions (r_f / 252; r_f / 52 or r_f / 12 on weekly or monthly
        bars; r_f / 365 in continuous mode).
    confidence : float, default 0.95
        Share of paths inside each range, strictly between 0 and 1: 0.95
        spans the 2.5th to the 97.5th percentile across paths. VaR and CVaR
        use the whole 1 − confidence tail (5% by default), twice the tail
        left beyond each end of a range.
    display : bool, default False
        If True, also print a formatted report.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.
    cagr_clock : {"session", "calendar"}, default "session"
        Clock of CAGR and Total Years: ``"session"``, Y = N / 252 with N
        the session closes after column 0 (a partial first or last session
        counts whole); ``"calendar"``, Y = (t_T - t_0) / 365.25 days, which
        needs marks. Continuous mode needs ``"calendar"``. Paths without
        marks count time in bars: Y = n_marks / (252, 52 or 12).

    Returns
    -------
    dict
        ``"Per Path Metrics"``
            pandas.DataFrame, one row per path, columns ``"Final
            Bankroll"``, ``"Total Return"``, ``"CAGR"``, ``"Gap
            Volatility"``, ``"Bar Volatility"``, ``"Annualized
            Volatility"``, ``"Downside Volatility"``, ``"Max Drawdown"``,
            ``"Sharpe Ratio"``, ``"Sortino Ratio"``, each as in
            `performance_report`.
        ``"Period Returns"``
            numpy.ndarray of shape ``(n_sims, n_marks)``: returns between
            consecutive marks (NaN after a total loss).
        ``"Daily Returns"``
            numpy.ndarray, one row per path: the session returns behind the
            risk figures (a partial first session left out).
        ``"Paths"``, ``"Periods"``
            n_sims and n_marks.
        ``"Total Days"``
            Whole calendar days from column 0 to the last mark, or None
            without marks.
        ``"Total Years"``
            Y on `cagr_clock`.
        ``"Total Time (Years)"``
            Calendar years from column 0 to the last mark,
            (t_T - t_0) / 365.25 days, whatever `cagr_clock` is; None
            without marks. The printed and PDF headers show it.
        ``"Has Gaps"``, ``"Intraday"``
            Whether the marks include at least one gap return, and whether
            they are intraday (with regular bars); both False without
            marks. The printout and the PDF show the Gap and Bar Volatility
            rows only when these are True.
        ``"Confidence"``
            The `confidence` used.
        ``"Returns"``, ``"Risk"``
            pandas.DataFrames with one row per figure (Returns: Final
            Bankroll, Total Return, CAGR; Risk: Gap Volatility, Bar
            Volatility, Annualized Volatility, Downside Volatility, Max
            Drawdown, Sharpe Ratio, Sortino Ratio) and columns ``Mean``,
            ``Median``, ``Lower``, ``Upper``. Lower and Upper bound the 95%
            range across paths (for the default `confidence`): percentiles
            across paths under the fitted model, not a confidence
            interval.
        ``"Tail Risk"``
            dict with ``"Probability of Loss"`` (share of paths with a
            negative total return), ``"Value at Risk"`` (the total return at
            the ``1 - confidence`` quantile) and ``"Conditional VaR"`` (the
            mean total return at or below it).

    Raises
    ------
    ValueError
        If a path has fewer than two marks, `confidence` is not in
        (0, 1), `cagr_clock` is unknown, ``"calendar"`` without marks or
        ``"session"`` in continuous mode, continuous mode without marks,
        `interval` is missing for paths without marks, intraday for paths
        without marks (their risk figures need the sessions), or has no
        annualization factor, `dates` or the SimulationResult's dates and
        classes don't match the paths, or the marks can't be placed on the
        calendar.

    Warns
    -----
    UserWarning
        When some paths have an infinite Sortino ratio (no session below the
        risk-free rate), with their count and share and that the Sortino
        mean is NaN; when some paths
        end at a total loss, with their count; when a figure is
        undefined on some paths and left out of their summary, with the
        counts; in continuous mode, when the paths span fewer than two
        00:00 UTC boundaries, so volatility, Sharpe and Sortino are NaN.

    Notes
    -----
    - The per-path figures follow `performance_report`: volatility, Sharpe
      and Sortino on session returns annualized with 252 (52 or 12 on
      weekly or monthly bars, 365 in continuous mode), never per mark,
      leaving out a partial first session, e.g. a path that starts
      mid-session; max drawdown a fraction in [0, 1] over every mark,
      open marks included; Gap and Bar Volatility per unit and NaN on
      daily data.
    - Ruined paths (final value 0) are kept in every summary: CAGR -100%,
      risk figures on the returns up to and including the -100% one.
    - A Sortino ratio of +inf stays in its summary: the Sortino mean is
      then NaN, and the median and range use inf-safe percentiles.

    See Also
    --------
    performance_report : The same figures for one historical series.
    portfolio_forecast.plotting.plot_simulated_paths : Fan chart of `sims`.

    Examples
    --------
    >>> sims = nonparametric_monte_carlo(backtest, sim_length=20, random_state=42)
    >>> report = statistical_report(sims, display=True)
    >>> report["Tail Risk"]["Probability of Loss"]
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}")
    if cagr_clock not in CAGR_CLOCKS:
        raise ValueError(f"cagr_clock must be 'session' or 'calendar', got {cagr_clock!r}")
    # The paths, their column marks (None if only the bar size is known) and classes
    paths, labels, classes = _resolve_paths(sims, interval, dates, calendar)
    n_sims, n_points = paths.shape
    n_periods = n_points - 1

    with np.errstate(divide="ignore", invalid="ignore"):
        # After a total loss the returns are 0/0, undefined (NaN)
        period_returns = paths[:, 1:] / paths[:, :-1] - 1

    no_full_day = False
    if labels is None:
        # Paths without marks: only the bar size is known
        if calendar is None:
            raise ValueError(
                "Continuous mode (calendar=None) needs the marks, since its risk figures use calendar-day returns at "
                "00:00 UTC: pass a SimulationResult or dates"
            )
        if _parse_interval(interval)[0] == "intraday":
            raise ValueError(
                f"interval {interval!r} is intraday, but sims has no marks: intraday risk figures use session "
                "returns, so pass a SimulationResult or dates"
            )
        if cagr_clock == "calendar":
            raise ValueError("cagr_clock='calendar' counts calendar days, so it needs the marks: pass a "
                             "SimulationResult or dates")
        periods_per_year = _periods_per_year(interval)
        # Every bar return is a full bar from the last historical close
        session_returns = period_returns
        total_years = n_periods / periods_per_year
        total_days = None
        masks = None
    else:
        # Paths with marks: the same session columns, years and class masks as one series
        periods_per_year = _risk_periods_per_year(labels, calendar, interval)
        # Future marks are never in progress, so no session is left out
        positions, _, drop_first = _session_positions(labels, calendar, drop_live=False)
        session_values = paths[:, positions]
        with np.errstate(divide="ignore", invalid="ignore"):
            session_returns = session_values[:, 1:] / session_values[:, :-1] - 1
        if drop_first:
            # A partial first session (e.g. a mid-session start) is left out
            session_returns = session_returns[:, 1:]
        total_years = calculate_total_years(labels, calendar, cagr_clock)
        total_days = (labels[-1] - labels[0]).days
        if classes is None and _series_kind(labels, calendar) != DAILY:
            classes = classify_returns(labels, calendar)
        masks = _class_masks(labels, classes, calendar, interval)
        no_full_day = calendar is None and _utc_boundaries(labels) < 2

    # Every figure on every path at once, one row per path
    initial_bankroll = paths[:, 0]
    final_bankroll = paths[:, -1]
    total_return = final_bankroll / initial_bankroll - 1
    cagr = _path_cagr(initial_bankroll, final_bankroll, total_years)
    if no_full_day:
        warnings.warn(
            f"The paths span {_utc_boundaries(labels)} 00:00 UTC boundary(ies), fewer than two, so they have no "
            "full UTC day: volatility, Sharpe and Sortino are NaN",
            UserWarning, stacklevel=2,
        )
        ann_vol = ann_downside_vol = sharpe = sortino = np.full(n_sims, np.nan)
    else:
        ann_vol, ann_downside_vol, sharpe, sortino = _path_risk(session_returns, periods_per_year, risk_free_rate)
    if masks is None:
        gap_vol = bar_vol = np.full(n_sims, np.nan)
    else:
        gap, regular, scale = masks
        gap_vol = _class_std(period_returns[:, gap]) if gap is not None else np.full(n_sims, np.nan)
        bar_vol = _class_std(_scaled_bars(period_returns[:, regular], scale))
    # Max over every mark of 1 - V(t) / max_{s<=t} V(s), a fraction in [0, 1]
    max_drawdown = (1 - paths / np.maximum.accumulate(paths, axis=1)).max(axis=1)

    per_path = pd.DataFrame({
        "Final Bankroll": final_bankroll,
        "Total Return": total_return,
        "CAGR": cagr,
        "Gap Volatility": gap_vol,
        "Bar Volatility": bar_vol,
        "Annualized Volatility": ann_vol,
        "Downside Volatility": ann_downside_vol,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
    })
    _warn_paths(per_path, final_bankroll, no_full_day)

    def summarize(columns):
        """Return the range summary of each column of `per_path`, one row each."""
        return pd.DataFrame({col: _range_summary(per_path[col].to_numpy(), confidence) for col in columns}).T

    # Mean, median and range across paths of each figure
    returns_summary = summarize(_RETURN_METRICS)
    risk_summary = summarize(_RISK_METRICS)
    # One infinite Sortino ratio makes an arithmetic mean meaningless, so it
    # is NaN; the median and range are order statistics and stay
    if np.isposinf(per_path["Sortino Ratio"]).any():
        risk_summary.loc["Sortino Ratio", "Mean"] = np.nan

    # Tail risk of the total return: the chance of a loss, VaR (the 1 - confidence
    # quantile) and Conditional VaR (the mean return at or below it)
    prob_loss = float((total_return < 0).mean())
    var = float(np.quantile(total_return, 1 - confidence))
    cvar = float(total_return[total_return <= var].mean())

    # Which classes the marks have, so the printout and the PDF show only those volatility rows
    has_gaps, intraday = _class_rows(masks)
    report = {
        "Per Path Metrics": per_path,
        "Period Returns": period_returns,
        "Daily Returns": session_returns,
        "Paths": n_sims,
        "Periods": n_periods,
        "Total Days": total_days,
        "Total Years": total_years,
        "Total Time (Years)": _total_time_years(labels) if labels is not None else None,
        "Has Gaps": has_gaps,
        "Intraday": intraday,
        "Confidence": confidence,
        "Returns": returns_summary,
        "Risk": risk_summary,
        "Tail Risk": {
            "Probability of Loss": prob_loss,
            "Value at Risk": var,
            "Conditional VaR": cvar,
        },
    }
    if display:
        _print_statistical_report(report)
    return report


def _warn_paths(per_path, final_bankroll, no_full_day):
    """Warn about the paths that need a caveat.

    One warning each, with counts: infinite Sortino ratios, total losses,
    and figures undefined on some paths and left out of their summaries.
    """
    n_sims = len(per_path)
    infinite = int(np.isposinf(per_path["Sortino Ratio"]).sum())
    if infinite:
        warnings.warn(
            f"Sortino ratio is infinite on {infinite} of {n_sims} paths ({infinite / n_sims:.1%}); its mean is NaN",
            UserWarning, stacklevel=3,
        )
    ruined = int((final_bankroll == 0).sum())
    if ruined:
        warnings.warn(
            f"{ruined} of {n_sims} paths end at a total loss (value 0): their CAGR is -100% and their returns "
            "after the loss are undefined and left out. They are kept in every summary",
            UserWarning, stacklevel=3,
        )
    # With no full UTC day the risk figures are NaN on every path by
    # rule, and statistical_report has already warned about it
    counted = _COUNTED_METRICS if not no_full_day else ("CAGR", "Max Drawdown")
    undefined = {m: int(per_path[m].isna().sum()) for m in counted}
    undefined = {m: k for m, k in undefined.items() if k}
    if undefined:
        listing = ", ".join(f"{m} on {k} of {n_sims}" for m, k in undefined.items())
        warnings.warn(
            f"Figures undefined on some paths, left out of their summaries: {listing} paths (e.g. no variance, "
            "too few sessions, or a value below zero)",
            UserWarning, stacklevel=3,
        )


def _print_statistical_report(report):
    """Print a `statistical_report` dict as a formatted table, with the ranges worded as across paths.

    The Gap and Bar Vol rows print only if ``"Has Gaps"`` and
    ``"Intraday"`` say the marks have that class.
    """
    width = 80
    separator = "=" * width
    sub_separator = "-" * width
    level = f"{report['Confidence']:.0%}"
    returns_summary, risk_summary, tail = report["Returns"], report["Risk"], report["Tail Risk"]

    def row(label, stats, fmt):
        """Return one printed row: mean, median and the range across paths."""
        mean = format(stats["Mean"], fmt)
        median = format(stats["Median"], fmt)
        band = f"[{format(stats['Lower'], fmt)}, {format(stats['Upper'], fmt)}]"
        return f"  {label:<21}: {mean:>10}   {median:>10}   {band:>28}"

    # Lower/Upper are percentiles across paths, not a confidence interval
    column_header = f"  {'':<21}  {'Mean':>10}   {'Median':>10}   {level + ' range across paths':>28}"

    print("\n" + separator)
    print(f"{'SIMULATED PORTFOLIO STATISTICAL REPORT':^{width}}")
    print(sub_separator)
    print(f"  {'Simulated Paths':<21}: {report['Paths']:>5,}")
    # Paths without marks have no calendar time: only their period count
    if report["Total Time (Years)"] is not None:
        print(f"  {'Total Time':<21}: {report['Total Time (Years)']:>5.2f} years ({report['Total Days']} days)")
    print(f"  {'Horizon':<21}: {report['Periods']:>5} periods")
    print(separator)
    print(f"{'RETURNS':^{width}}")
    print(sub_separator)
    print(column_header)
    print(sub_separator)
    print(row("Final Bankroll", returns_summary.loc["Final Bankroll"], ".2f"))
    print(row("Total Return", returns_summary.loc["Total Return"], ".2%"))
    print(row("CAGR", returns_summary.loc["CAGR"], ".2%"))
    print(separator)
    print(f"{'RISK':^{width}}")
    print(sub_separator)
    print(column_header)
    print(sub_separator)
    print(row("Volatility", risk_summary.loc["Annualized Volatility"], ".2%"))
    print(row("Downside Vol", risk_summary.loc["Downside Volatility"], ".2%"))
    # Only the classes the marks have, as in performance_report
    if report["Has Gaps"]:
        print(row("Gap Vol (per gap)", risk_summary.loc["Gap Volatility"], ".2%"))
    if report["Intraday"]:
        print(row("Bar Vol (per bar)", risk_summary.loc["Bar Volatility"], ".2%"))
    print(row("Max Drawdown", risk_summary.loc["Max Drawdown"], ".2%"))
    print(row("Sharpe Ratio", risk_summary.loc["Sharpe Ratio"], ".2f"))
    print(row("Sortino Ratio", risk_summary.loc["Sortino Ratio"], ".2f"))
    print(separator)
    print(f"{'TAIL RISK (TOTAL RETURN)':^{width}}")
    print(sub_separator)
    print(f"  {'Probability of Loss':<21}: {tail['Probability of Loss']:>10.2%}")
    print(f"  {f'VaR ({level})':<21}: {tail['Value at Risk']:>10.2%}")
    print(f"  {f'CVaR ({level})':<21}: {tail['Conditional VaR']:>10.2%}")
    print(separator + "\n")
