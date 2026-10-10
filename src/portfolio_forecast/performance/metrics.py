"""Performance and risk metrics of one value series.

Pure functions: each takes values or returns and gives one figure, with no
printing. `performance_report` assembles them.

Flow:

1. Time: `count_sessions` and `calculate_total_years` give the years a
   series spans on the session or calendar clock; `calculate_cagr` turns
   them into a compound annual growth rate, -100% at a total loss.
2. Risk on the session returns of `returns.calculate_session_returns`:
   `calculate_excess_returns`, `calculate_volatility`,
   `calculate_sharpe_ratio`, `calculate_downside_deviation` and
   `calculate_sortino_ratio`.
3. Per mark: `calculate_class_volatility` (per-class volatility of gap
   and regular returns) and `calculate_max_drawdown`.
4. `calculate_actual_yearly_return` gives each year's return and
   whether the series covers the whole year.

Conventions: simple returns (r = V_t / V_{t-1} - 1) as fractions; a year is
252 sessions, 52 weeks or 12 months for annualizing risk (365 days in
continuous mode), and 365.25 days on the calendar clock. These are
conventions, not counts of a real year.
"""
import warnings

import numpy as np
import pandas as pd

from ..types import BacktestResult
from ..utils.calendar import (
    _DATE_PAD,
    _INTRADAY_PAD,
    _bar_spans,
    _get_calendar,
    _infer_date_bars,
    _is_midnight,
    _session_times,
)
from ..utils.marks import (
    DAILY,
    GAP,
    REGULAR,
    _intraday_step,
    _labels_of,
    _mark_sessions,
    _series_kind,
    classify_returns,
)
from ..utils.periods import TRADING_DAYS, _parse_interval

# The calendar clock's year
DAYS_PER_YEAR = 365.25

CAGR_CLOCKS = ("session", "calendar")


def _values_of(values):
    """Return the value Series of a Series or BacktestResult."""
    return values.values if isinstance(values, BacktestResult) else values


# --- Time and growth ----------------------------------------------

def count_sessions(values, calendar="XNYS"):
    """Return N, the number of sessions the session clock of CAGR counts.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or pandas.DatetimeIndex
        The value series, or its labels, oldest first.
    calendar : str, default "XNYS"
        ``exchange_calendars`` calendar code.

    Returns
    -------
    int
        #{d : t_0 < C_d ≤ t_T}, the session closes after the first mark up
        to and including the last, plus one if the last mark falls inside a
        session. So a partial first session counts whole (its close is after
        the fill), and so does a session in progress at the last mark.
        Weekly and monthly series count every session their bars span.
        Dates are read as their bars' closes; daily dates need not
        all be sessions of `calendar`.

    Raises
    ------
    ValueError
        If `calendar` is None: continuous mode has no sessions.
    """
    if calendar is None:
        raise ValueError(
            "The session clock needs sessions; continuous mode (calendar=None) has none: "
            "use cagr_clock='calendar'"
        )
    # The first and last marks as instants: dated bars by their closes
    index = _labels_of(_values_of(values))
    if _is_midnight(index):
        dates = (index.tz_localize(None) if index.tz is not None else index).as_unit("ns")
        cal = _get_calendar(calendar, dates[0], dates[-1], _DATE_PAD)
        if _infer_date_bars(dates, cal)[0] == "session":
            # Daily dates are session closes; an external series may carry
            # dates off this calendar, so count sessions between the ends
            sessions = cal.sessions.as_unit("ns")
            return int(((sessions > dates[0]) & (sessions <= dates[-1])).sum())
        closes = pd.DatetimeIndex(_bar_spans(index, calendar)["close"])
        first, last = closes[0], closes[-1]
    else:
        first, last = index[0], index[-1]
    cal = _get_calendar(calendar, first.tz_convert(None), last.tz_convert(None), _INTRADAY_PAD)
    # Session closes after the first mark, up to and including the last
    session_closes = _session_times(cal)[1]
    n = int(((session_closes > first) & (session_closes <= last)).sum())
    # A session in progress at the last mark counts whole. The last
    # mark's session is the first whose close is at or after it
    return n + int(session_closes[session_closes.searchsorted(last)] > last)


def calculate_total_years(values, calendar="XNYS", cagr_clock="session"):
    """Return the years a value series spans, the Y of CAGR.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or pandas.DatetimeIndex
        The value series, or its labels, oldest first.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.
    cagr_clock : {"session", "calendar"}, default "session"
        ``"session"``: Y = N / 252, N from `count_sessions`. ``"calendar"``:
        Y = (t_T - t_0) / 365.25 days.

    Returns
    -------
    float
        Y, in years.

    Raises
    ------
    ValueError
        If `cagr_clock` is unknown, or is ``"session"`` in continuous mode,
        which has no sessions.

    Notes
    -----
    The calendar clock counts nights, weekends and holidays, which the
    volatility's 252-session year does not, so a CAGR on it is annualized
    on a different clock from the volatility.
    """
    if cagr_clock not in CAGR_CLOCKS:
        raise ValueError(f"cagr_clock must be 'session' or 'calendar', got {cagr_clock!r}")
    index = _labels_of(_values_of(values))
    if cagr_clock == "calendar":
        return (index[-1] - index[0]) / pd.Timedelta(days=1) / DAYS_PER_YEAR
    return count_sessions(index, calendar) / TRADING_DAYS


def calculate_cagr(values, periods_per_year=None, *, years=None):
    """Return the compound annual growth rate of a value series.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values, oldest first.
    periods_per_year : float, optional
        Marks per year, for the 0.3.0 bar clock Y = (n - 1) / periods_per_year.
        Ignored when `years` is given.
    years : float, optional
        The span Y in years, e.g. from `calculate_total_years`.

    Returns
    -------
    float
        (V_T / V_0) ** (1 / Y) - 1. Exactly -1.0 (-100%) when V_T = 0;
        NaN when V_T < 0.

    Raises
    ------
    ZeroDivisionError
        If Y is zero (a single value, or no time elapsed).
    ValueError
        If neither `periods_per_year` nor `years` is given.
    """
    values = _values_of(values)
    # Y from the caller, or the old bar clock: marks over periods per year
    if years is None:
        if periods_per_year is None:
            raise ValueError("pass years (see calculate_total_years) or periods_per_year")
        years = (len(values) - 1) / periods_per_year
    exponent = 1 / years
    growth = float(values.iloc[-1] / values.iloc[0])
    # A total loss is a CAGR of -100% whatever the span Y
    if growth == 0:
        return -1.0
    if growth < 0:
        return np.nan
    return growth ** exponent - 1


# --- Risk on session returns --------------------------------------

def calculate_excess_returns(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS):
    """Return returns in excess of the risk-free rate, r - r_f / periods_per_year.

    Parameters
    ----------
    returns : pandas.Series
        Session returns (or bar returns of weekly and monthly series).
    risk_free_rate : float, default 0.0
        Annual risk-free rate, as a fraction.
    periods_per_year : float, default 252
        Returns per year: 252 sessions, 52 weeks, 12 months, or 365 days in
        continuous mode.

    Returns
    -------
    pandas.Series
        The excess returns, on the same index.
    """
    return returns - risk_free_rate / periods_per_year


def calculate_volatility(returns, periods_per_year=TRADING_DAYS):
    """Return the annualized volatility, σ̂(R) √periods_per_year.

    The sample standard deviation (ddof=1) of the defined returns (NaN
    returns after a total loss are dropped); NaN for fewer than two.
    """
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def calculate_sharpe_ratio(period_returns, periods_per_year=TRADING_DAYS, risk_free_rate=0.0):
    """Return the annualized Sharpe ratio of session returns.

    Parameters
    ----------
    period_returns : pandas.Series
        Session returns, e.g. from `calculate_session_returns`.
    periods_per_year : float, default 252
        Returns per year, used to annualize and to spread the risk-free rate.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.

    Returns
    -------
    float
        mean(e) / σ̂(R) √periods_per_year with e = R - r_f / periods_per_year
        and the sample standard deviation (ddof=1). NaN returns (after a
        total loss) are dropped first. NaN for fewer than two returns or
        zero variance.
    """
    # Defined returns only: NaN after a total loss is dropped
    period_returns = period_returns.dropna()
    excess = calculate_excess_returns(period_returns, risk_free_rate, periods_per_year)
    std = period_returns.std(ddof=1)
    if not std > 0:
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def calculate_downside_deviation(period_returns, periods_per_year=TRADING_DAYS, risk_free_rate=0.0):
    """Return the per-session downside deviation DD = √(Σ min(e, 0)² / n).

    Every one of the n sessions enters the average, so returns above the
    target count as zero shortfall rather than being dropped. The target is
    the risk-free rate, e = R - r_f / periods_per_year. NaN returns (after
    a total loss) are dropped first, so n counts defined returns only.
    NaN for no returns. Multiply by √periods_per_year for the annualized
    Downside Volatility.
    """
    excess = calculate_excess_returns(period_returns.dropna(), risk_free_rate, periods_per_year)
    if len(excess) == 0:
        return np.nan
    return float(np.sqrt((np.minimum(excess, 0) ** 2).sum() / len(excess)))


def calculate_sortino_ratio(period_returns, periods_per_year=TRADING_DAYS, risk_free_rate=0.0):
    """Return the annualized Sortino ratio of session returns.

    Parameters
    ----------
    period_returns : pandas.Series
        Session returns, e.g. from `calculate_session_returns`.
    periods_per_year : float, default 252
        Returns per year, used to annualize and to spread the risk-free rate.
    risk_free_rate : float, default 0.0
        Annual risk-free rate; also the target below which a return is
        downside.

    Returns
    -------
    float
        mean(e) √periods_per_year / DD, DD from `calculate_downside_deviation`.
        With no session below the target (DD = 0): +inf if mean(e) > 0, NaN
        if mean(e) = 0. NaN returns (after a total loss) are dropped
        first. NaN for no returns.

    Warns
    -----
    UserWarning
        When no session is below the target, with the count of sessions:
        "Sortino ratio is infinite: none of the n sessions is below the
        risk-free rate", or, when every session returns exactly the
        risk-free rate, "Sortino ratio is NaN: all n sessions return
        exactly the risk-free rate".
    """
    # Defined returns only: NaN after a total loss is dropped
    period_returns = period_returns.dropna()
    excess = calculate_excess_returns(period_returns, risk_free_rate, periods_per_year)
    n = len(excess)
    if n == 0:
        return np.nan
    dd = calculate_downside_deviation(period_returns, periods_per_year, risk_free_rate)
    mean = float(excess.mean())
    if dd == 0:
        # Every e_d >= 0, so the mean is too: +inf if positive, NaN if 0,
        # with a warning, since neither is a measurement
        if mean > 0:
            warnings.warn(f"Sortino ratio is infinite: none of the {n} sessions is below the risk-free rate",
                          UserWarning, stacklevel=2)
            return np.inf
        warnings.warn(f"Sortino ratio is NaN: all {n} sessions return exactly the risk-free rate",
                      UserWarning, stacklevel=2)
        return np.nan
    return mean * np.sqrt(periods_per_year) / dd


# --- Per-mark figures ---------------------------------------------

def calculate_max_drawdown(values):
    """Return the maximum drawdown, max_t (1 - V(t) / max_s≤t V(s)).

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values at every mark, oldest first. On intraday data that
        includes the open marks, so the overnight gaps count; an intraday
        backtest's drawdown is at least the daily one for the same book.

    Returns
    -------
    float
        A fraction in [0, 1]: 0.25 for a 25% fall from the high-water mark,
        0.0 if the series never falls, 1.0 at a total loss.
    """
    values = _values_of(values)
    return float((1 - values / values.cummax()).max())


def calculate_class_volatility(values, classes=None, calendar="XNYS", interval=None):
    """Return the per-unit volatility of gap and of regular returns.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values at every mark, oldest first. A BacktestResult
        supplies its own classes.
    classes : pandas.Series, optional
        The class of each return, indexed by the mark where it ends. None
        infers them with `classify_returns` (or takes a BacktestResult's).
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.
    interval : str or pandas.Timedelta, optional
        Bar size Δ of intraday data; None infers it from the marks.

    Returns
    -------
    dict
        ``"Gap Volatility"``: σ̂(g_d) over gap returns. ``"Bar Volatility"``:
        σ̂(r') over regular returns, a short bar (τ < Δ) scaled in logs and
        converted back, r' = exp(ℓ √(Δ/τ)) - 1 with ℓ = ln(1 + r).
        Both per unit (one gap, one full bar), unannualized, sample standard
        deviations (ddof=1), irregular returns left out. NaN on daily and
        longer data, continuous daily data included. Continuous mode: Gap
        Volatility NaN, Bar Volatility σ̂(r_k) of the regular returns,
        unscaled.
    """
    # The classes: given, a BacktestResult's, or inferred from intraday labels
    if isinstance(values, BacktestResult) and classes is None:
        classes = values.classes
    values = _values_of(values)
    masks = _series_class_masks(values, classes, calendar, interval)
    return _class_volatility(values, masks)


def _series_class_masks(values, classes, calendar="XNYS", interval=None):
    """Return `_class_masks` for one value series, classifying intraday
    labels when `classes` is None.

    Shared by `calculate_class_volatility` and `performance_report`, which
    also needs the masks to decide which volatility rows to print.
    """
    index = _labels_of(values)
    if classes is None and len(index) >= 2 and _series_kind(index, calendar) != DAILY:
        classes = classify_returns(values, calendar)
    return _class_masks(index, classes, calendar, interval)


def _class_volatility(values, masks):
    """Return Gap and Bar Volatility of one value series from its `_class_masks`."""
    # None: both figures are NaN by definition (daily and longer data)
    if masks is None:
        return {"Gap Volatility": np.nan, "Bar Volatility": np.nan}
    # σ̂ of the gaps, and of the bars scaled to full bars
    gap, regular, scale = masks
    returns = values.pct_change(fill_method=None).iloc[1:].to_numpy()
    return {
        "Gap Volatility": _class_std(returns[None, gap])[0] if gap is not None else np.nan,
        "Bar Volatility": _class_std(_scaled_bars(returns[None, regular], scale))[0],
    }


def _class_masks(index, classes, calendar="XNYS", interval=None):
    """Return which returns of a series of marks enter Gap and Bar Volatility,
    and how each bar is scaled.

    Shared by `calculate_class_volatility` (one series) and
    `statistical_report` (every simulated path on the same marks), so both
    select and scale returns the same way.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        The mark labels, oldest first.
    classes : pandas.Series or array-like of str
        ``"gap"``, ``"regular"`` or ``"irregular"`` for each return: a
        Series indexed by the mark where the return ends, or an array
        aligned with ``index[1:]``. Irregular returns are in neither mask.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.
    interval : str or pandas.Timedelta, optional
        Bar size Δ; None infers it from the marks.

    Returns
    -------
    tuple or None
        None when both figures are NaN by definition: fewer than two marks,
        or a daily or longer series. Otherwise ``(gap, regular,
        scale)``: boolean masks over the returns (`gap` is None in
        continuous mode, which has no gaps), and for each regular return,
        in order, the factor √(Δ/τ) that scales its log return to a full
        bar: above 1 only for a short bar (τ < Δ, such as a half-day's last
        bar); 1 for a bar as long as Δ or longer, and in continuous mode.
    """
    if len(index) < 2 or _series_kind(index, calendar) == DAILY:
        return None
    # Classes aligned with the returns; continuous mode has no gaps and no short bars
    if isinstance(classes, pd.Series):
        classes = classes.reindex(index[1:])
    classes = np.asarray(classes, dtype=object)
    regular = classes == REGULAR
    if calendar is None:
        return None, regular, np.ones(int(regular.sum()))
    # Each regular return's length τ against the bar size Δ
    step = _parse_interval(interval)[1] if interval is not None else _intraday_step(index, calendar)
    lengths = pd.TimedeltaIndex(index[1:] - index[:-1])[regular]
    # Only a short bar (τ < Δ) is rescaled; a regular return as long as
    # the bar size or longer is left as it is
    tau = np.minimum(lengths / step, 1.0)
    return classes == GAP, regular, np.sqrt(1 / np.asarray(tau, dtype=float))


def _scaled_bars(returns, scale):
    """Return regular returns scaled to full bars, r' = exp(ℓ √(Δ/τ)) - 1.

    `returns` holds simple returns, one row per path and one column per
    regular return; `scale` holds each column's √(Δ/τ) from `_class_masks`.
    The scaling is in logs, ℓ = ln(1 + r), so a scaled return stays above
    -100%.

    This is Bar Volatility's pure √ρ normalization (ρ = τ/Δ), ℓ' = ℓ/√ρ.
    The simulators use the drift-adjusted location–scale map instead,
    ℓ' = m̂ + (ℓ − ρ m̂)/√ρ; the
    two give volatilities about 2e-6 apart in relative terms.

    A -100% bar has log1p(-1) = -inf (the divide warning silenced), which
    expm1 maps back to -1. NaN returns (after a total loss) stay NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.expm1(np.log1p(returns) * scale)


def _class_std(returns):
    """Return the sample standard deviation (ddof=1) of each row of `returns`.

    `returns` is 2-D, one row per path (a single series is one row). NaN
    entries, the undefined returns after a total loss, are ignored, so
    each row uses its own count n of defined returns:
    σ̂ = √(Σ (r − r̄)² / (n − 1)). A row with n < 2 gives NaN, without the
    RuntimeWarning that ``np.nanstd(..., ddof=1)`` raises for it.
    """
    # Defined returns per row; rows with fewer than two stay NaN
    returns = np.asarray(returns, dtype=float)
    finite = ~np.isnan(returns)
    count = finite.sum(axis=1)
    out = np.full(returns.shape[0], np.nan)
    ok = count >= 2
    if ok.any():
        rows = returns[ok]
        mean = np.nansum(rows, axis=1) / count[ok]
        squares = np.where(finite[ok], (rows - mean[:, None]) ** 2, 0.0)
        out[ok] = np.sqrt(squares.sum(axis=1) / (count[ok] - 1))
    return out


# --- Yearly returns ---------------------------------------------------

def _mark_years(index, calendar):
    """Return the year of each mark: its session's year on the exchange clock,
    a date's own year, or in continuous mode the UTC day it closes
    (a mark at 00:00 UTC ends the day before; a first mark there starts its
    own day)."""
    # Continuous mode: the UTC day a mark closes, so 00:00 UTC ends the day before
    if calendar is None:
        utc = index.tz_convert("UTC") if index.tz is not None else index
        day = (utc - pd.Timedelta(1, "ns")).normalize()
        if utc[0] == utc[0].normalize():
            day = day.where(np.arange(len(day)) > 0, utc[0])
        return pd.DatetimeIndex(day).year.to_numpy()
    if _is_midnight(index):
        return index.year.to_numpy()
    return pd.DatetimeIndex(_mark_sessions(index, calendar)["session"]).year.to_numpy()


def _year_covered(index, year, calendar, first_year, last_year):
    """Return whether the series covers `year` from its first session's open
    (or 00:00 UTC on 1 January) to its last session's close."""
    start, end = index[0], index[-1]
    if calendar is None:
        jan1 = pd.Timestamp(f"{year}-01-01", tz="UTC")
        utc = (lambda t: t.tz_convert("UTC")) if index.tz is not None else (lambda t: t.tz_localize("UTC"))
        covers_start = not first_year or utc(start) <= jan1
        covers_end = not last_year or utc(end) >= jan1 + pd.DateOffset(years=1)
        return covers_start and covers_end
    # The open of the year's first session and the close of its last
    cal = _get_calendar(calendar, pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"), _DATE_PAD)
    sessions = cal.sessions[cal.sessions.year == year]
    opens, closes = _session_times(cal)
    in_year = np.flatnonzero(cal.sessions.year == year)
    first_open, last_close = opens[in_year[0]], closes[in_year[-1]]
    if _is_midnight(index):
        # A date is its session's close: it never covers that session's open
        covers_start = not first_year
        covers_end = not last_year or end.tz_localize(None) >= sessions[-1].tz_localize(None)
    else:
        covers_start = not first_year or start <= first_open
        covers_end = not last_year or end >= last_close
    return covers_start and covers_end


def calculate_actual_yearly_return(values, dates=None, calendar="XNYS"):
    """Return the return earned in each year of a value series.

    Parameters
    ----------
    values : pandas.Series or BacktestResult
        Portfolio values, oldest first.
    dates : array-like of datetimes, optional
        Labels for `values`, one per value, if its own index isn't them.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code; None is continuous mode.

    Returns
    -------
    pandas.DataFrame
        Indexed by year, with columns:

        ``return``
            R_y = V(last mark of y) / V(last mark of y-1) - 1; the first
            year is measured from the first value (br0), so the move across
            each year boundary is counted once.
        ``days``
            365 or 366 for a full year; for a partial year its actual span,
            the calendar days from its first to its last mark, inclusive.
        ``is_full``
            True iff the series covers the calendar's first session of the
            year from its open and its last session to its close. Continuous
            mode: 00:00 UTC on 1 January to 00:00 UTC on the next.

    Notes
    -----
    A mark's year is its session's year on the exchange clock, so a
    close on 31 December belongs to that year whatever its UTC date.
    Dated series (midnight labels) are closes: their first year is partial,
    since the first session's own move is not in the series.
    """
    values = _values_of(values)
    if dates is not None:
        values = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates))
    # Each mark's year; each year's return runs from the previous year's last mark (or br0)
    index = _labels_of(values)
    years = _mark_years(index, calendar)
    yearly = {}
    base = values.iloc[0]
    unique = list(dict.fromkeys(years))
    for year in unique:
        in_year = values[years == year]
        full = _year_covered(index, year, calendar, year == unique[0], year == unique[-1])
        is_leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
        start = index[0] if year == unique[0] else in_year.index[0]
        yearly[year] = {
            "return": float(in_year.iloc[-1] / base - 1),
            "days": (366 if is_leap else 365) if full else (in_year.index[-1] - start).days + 1,
            "is_full": bool(full),
        }
        # The next year runs from this year's last mark, so each move
        # across a year boundary is counted once
        base = in_year.iloc[-1]
    return pd.DataFrame.from_dict(yearly, orient="index")
