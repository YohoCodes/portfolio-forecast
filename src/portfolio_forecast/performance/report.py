import warnings

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from dateutil.tz import tzlocal

from ..utils.bar_times import _session_dates
from ..utils.periods import TRADING_DAYS, _parse_interval, _periods_per_year


def _session_marks(dates, calendar, tz):
    """Positions the daily returns run between: the first value, then each
    session's last value. A first value that is also its session's last
    counts once, so no zero return is invented."""
    sessions = _session_dates(dates, calendar, tz)
    is_mark = np.append(sessions[1:] != sessions[:-1], True)
    is_mark[0] = True
    return np.flatnonzero(is_mark)


def calculate_period_returns(values):
    """One-period simple returns of a value series.

    The period is whatever bar `values` is sampled on: daily bars give daily
    returns, hourly bars give hourly returns.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values or prices, one per bar, oldest first.

    Returns
    -------
    pandas.Series
        ``values[t] / values[t-1] - 1``, one shorter than `values`, with the
        first (undefined) return dropped.
    """
    # A return across a gap is measured from the last known value (explicit, as
    # pandas 3 no longer fills by default)
    period_returns = values.ffill().pct_change(fill_method=None).dropna()
    return period_returns


def calculate_daily_return(values):
    """Deprecated alias for `calculate_period_returns`.

    .. deprecated:: 0.4.0
        `calculate_daily_return` will be removed in 1.0.0; use
        `calculate_period_returns` instead. The returns are per bar, not per
        day, which the old name hid.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values or prices, one per bar, oldest first.

    Returns
    -------
    pandas.Series
        The same result as ``calculate_period_returns(values)``.

    Warns
    -----
    DeprecationWarning
        Always.
    """
    warnings.warn("calculate_daily_return is deprecated and will be removed in 1.0.0; "
                  "use calculate_period_returns instead", DeprecationWarning, stacklevel=2)
    return calculate_period_returns(values)


def calculate_cagr(values, periods_per_year):
    """Compound annual growth rate of a value series.

    The constant annual rate that turns the first value into the last one.
    Geometric, not a simple scaling of the total return, because each
    period's gain compounds into the next.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values, one per bar, oldest first.
    periods_per_year : float
        Bars in one year at the series' interval, e.g. 252 for daily bars
        (see ``portfolio_forecast.utils.PERIODS_PER_YEAR``).

    Returns
    -------
    float
        ``(values[-1] / values[0]) ** (1 / years) - 1`` with
        ``years = (len(values) - 1) / periods_per_year``. -1.0 (-100%) if
        the last value is 0, a total loss, for any horizon. NaN if the last
        value is below zero (relative to the first), where no real annual
        rate exists.

    Raises
    ------
    ValueError
        If `values` has fewer than two entries (zero elapsed time).

    Notes
    -----
    Time is counted in bars, not calendar days, the same clock the Sharpe and
    volatility annualization use: nights, weekends and holidays add no time.
    For intraday values, `performance_report` passes one value per session
    with ``periods_per_year=252``, so time is counted in sessions.
    Over short horizons the annualization magnifies the total return
    enormously (a 1% gain over one day is a CAGR of about 1,100%).
    """
    if len(values) < 2:
        raise ValueError("CAGR needs at least two values: one value spans no time")

    # Converting the bar count to years
    growth = values.iloc[-1] / values.iloc[0]
    num_years = (len(values) - 1) / periods_per_year

    # A total loss is -100% at any horizon, the limit of growth ** (1 / years) - 1
    # as growth -> 0. No real rate compounds to a negative ratio: numpy's
    # growth ** (1 / num_years) is NaN unless 1 / num_years is a whole number,
    # when it is a meaningless real (num_years = 1 gives growth - 1), so the
    # guard keeps the answer from depending on the horizon
    if growth == 0:
        return -1.0
    if not growth > 0:
        return np.nan

    return float(growth ** (1 / num_years) - 1)


def calculate_sharpe_ratio(period_returns, periods_per_year, risk_free_rate=0.0):
    """Annualized Sharpe ratio of a return series.

    Parameters
    ----------
    period_returns : pandas.Series
        One-period simple returns, e.g. from `calculate_period_returns`.
    periods_per_year : float
        Bars in one year at the returns' interval, used to annualize.
    risk_free_rate : float, default 0.0
        Annual risk-free rate as a fraction (0.05 = 5%), spread evenly
        across the year's bars.

    Returns
    -------
    float
        ``mean(r - rf / periods_per_year) / std(r) * sqrt(periods_per_year)``,
        with the sample standard deviation (``ddof=1``). Infinite or NaN if
        the returns have zero variance.
    """
    # Sample std (ddof=1) of the raw returns: subtracting a constant rf per
    # period doesn't change it
    excess_returns = period_returns - (risk_free_rate / periods_per_year)
    mean_excess_return = excess_returns.mean()
    std_dev = period_returns.std()
    sharpe_ratio = (mean_excess_return / std_dev) * np.sqrt(periods_per_year)
    return sharpe_ratio


def calculate_sortino_ratio(period_returns, periods_per_year, risk_free_rate=0.0):
    """Annualized Sortino ratio of a return series.

    Like the Sharpe ratio, but divides by downside deviation, so only returns
    below the risk-free rate count as risk.

    Parameters
    ----------
    period_returns : pandas.Series
        One-period simple returns, e.g. from `calculate_period_returns`.
    periods_per_year : float
        Bars in one year at the returns' interval, used to annualize.
    risk_free_rate : float, default 0.0
        Annual risk-free rate as a fraction (0.05 = 5%), spread evenly
        across the year's bars. It is also the target below which a return
        counts as downside.

    Returns
    -------
    float
        ``mean(excess) / downside_dev * sqrt(periods_per_year)``, or NaN if no
        return falls below the risk-free rate.

    Notes
    -----
    Downside deviation is ``sqrt(sum(min(excess, 0) ** 2) / n)`` over all n
    periods, so returns above the target contribute zero rather than being
    dropped from the count.
    """
    # Calculating downside deviation. Every period enters the average — the
    # non-negative ones contribute zero — so the sum of squared shortfalls is
    # divided by the total count, not by the number of losing periods.
    excess_returns = period_returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt((downside_returns**2).sum() / len(excess_returns))
    if downside_std == 0:
        return np.nan

    sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(
        periods_per_year
    )
    return sortino_ratio


def calculate_max_drawdown(values):
    """Largest peak-to-trough decline of a value series.

    The running maximum is the high-water mark; the max drawdown is the
    deepest drop below it, relative to the mark.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values, one per bar, oldest first.

    Returns
    -------
    float
        ``min(values / cummax(values) - 1)``, a fraction in [-1, 0]; e.g.
        -0.25 for a 25% drawdown, 0.0 if the series never falls.
    """
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return float(drawdown.min())


def _exchange_dates(dates, calendar, tz):
    """Each date's calendar date on the exchange's clock, as naive midnights.
    Dates that are all at midnight are daily (or longer) bar dates and are
    kept as they are; other times are read in `tz` if naive (default: this
    machine's local timezone) and moved to the exchange's timezone. With no
    calendar, each date's own date."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    wall = index if index.tz is None else index.tz_localize(None)
    if calendar is None or (wall == wall.normalize()).all():
        return wall.normalize()
    if index.tz is None:
        index = index.tz_localize(tz or tzlocal(), ambiguous="infer", nonexistent="shift_forward")
    exchange_tz = xcals.get_calendar(calendar).tz
    return index.tz_convert(exchange_tz).tz_localize(None).normalize()


def _year_bounds(first_year, last_year, calendar):
    """First and last trading day of each year: the calendar's sessions, or
    weekdays with no calendar."""
    start, end = pd.Timestamp(first_year, 1, 1), pd.Timestamp(last_year, 12, 31)
    if calendar is None:
        days = pd.bdate_range(start, end)
    else:
        days = xcals.get_calendar(calendar, start=start, end=end).sessions
    days = pd.Series(days, index=days.year)
    return days.groupby(level=0).min(), days.groupby(level=0).max()


def calculate_actual_yearly_return(values, dates, calendar="XNYS", tz=None):
    """Return earned in each calendar year of a value series.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values, one per bar, oldest first.
    dates : array-like of datetimes
        The date of each value, same length as `values`.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code. Intraday dates are placed in the
        year of their date on this exchange's clock, and its first and last
        sessions of a year decide whether that year is full. None uses each
        date's own date and the year's first and last weekdays.
    tz : str or tzinfo, optional
        Timezone that naive intraday `dates` are in. Defaults to this
        machine's local timezone, as in `simulate_buy_and_hold`. Unused for
        dates at midnight (daily and longer bars).

    Returns
    -------
    pandas.DataFrame
        Indexed by calendar year, with columns:

        ``return``
            Simple return over the year. The first year is measured from its
            first value; later years from the previous year's last value, so
            the move across the year boundary is counted.
        ``days``
            365 or 366 for a full year; for a partial year, the calendar days
            from its first to its last bar, inclusive.
        ``is_full``
            False for the first year if the data starts after that year's
            first session, and for the last year if it ends before that
            year's last session. Years in between are always full.

    Raises
    ------
    ValueError
        If `calendar` is not an ``exchange_calendars`` code or does not
        cover the years spanned.

    Notes
    -----
    Full is decided by session date, not time of day: a series whose first
    value is the close of the year's first session counts as full, although
    that session's own move is not in the return.
    """
    # Grouping portfolio values by calendar year on the exchange's clock, so a
    # New York close labelled 01:00 on Jan 1 in Dubai stays in the old year
    session_dates = _exchange_dates(dates, calendar, tz)
    values_series = pd.Series(np.asarray(values), index=session_dates)
    yearly_groups = values_series.groupby(values_series.index.year)

    # Global min and max dates across the entire dataset, and the first and
    # last trading day of the years they fall in
    global_start = values_series.index[0]
    global_end = values_series.index[-1]
    first_day, last_day = _year_bounds(global_start.year, global_end.year, calendar)

    yearly_returns = {}
    prior_close = None
    for year, group in yearly_groups:
        # Calculating total percentage change per calendar year. After the
        # first year, the base is the previous year's last value, so the move
        # from the last bar of one year to the first bar of the next is
        # counted (a whole month on monthly bars) rather than dropped.
        base = group.iloc[0] if prior_close is None else prior_close
        return_val = (group.iloc[-1] / base) - 1
        prior_close = group.iloc[-1]

        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        full_year_days = 366 if is_leap_year else 365

        # A year is partial ONLY if it contains the absolute start or end of the
        # entire dataset and misses that year's first or last trading day
        is_first_year = (year == global_start.year) and global_start > first_day[year]
        is_last_year = (year == global_end.year) and global_end < last_day[year]

        is_full_year = not (is_first_year or is_last_year)

        if is_full_year:
            display_days = full_year_days
        else:
            # For partial years, compute exact elapsed calendar days
            display_days = (group.index[-1] - group.index[0]).days + 1

        yearly_returns[year] = {
            "return": return_val,
            "days": display_days,
            "is_full": is_full_year,
        }

    return pd.DataFrame(yearly_returns).T


def performance_report(
    values, dates, interval, risk_free_rate=0.0, display=False, calendar="XNYS", tz=None
):
    """Performance and risk metrics for one portfolio value series.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values, oldest first, e.g. from `simulate_buy_and_hold`.
    dates : pandas.DatetimeIndex or array-like of datetimes
        The date of each value, same length as `values`. Used for the
        calendar-year returns, the displayed calendar span and, on intraday
        bars, to find each trading session.
    interval : str or pandas.Timedelta
        Bar size of `values`: a ``PERIODS_PER_YEAR`` key such as
        ``'1 day'`` or ``'5 mins'``, or any spelling
        `portfolio_forecast.utils.next_trading_dates` accepts (``'1D'``,
        ``'5min'``, yfinance's ``'5m'``, a Timedelta). On daily and longer
        bars it sets the annualization; on intraday bars it marks the values
        as intraday, and the risk figures and CAGR count sessions, 252 a year.
    risk_free_rate : float, default 0.0
        Annual risk-free rate as a fraction (0.05 = 5%) for the Sharpe and
        Sortino ratios and the downside deviation, spread evenly across the
        year's returns: ``risk_free_rate / 252`` per session on intraday
        and daily bars.
    display : bool, default False
        If True, also print a formatted report.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code whose sessions intraday values
        are grouped into for the daily returns. None groups by each date's
        own calendar day. Unused for daily and longer bars.
    tz : str or tzinfo, optional
        Timezone that naive intraday `dates` are in. Defaults to this
        machine's local timezone, as in `simulate_buy_and_hold`.

    Returns
    -------
    dict
        ``"Period Returns"``
            pandas.Series of one-period returns.
        ``"Daily Returns"``
            pandas.Series of the daily returns the risk figures use: on
            intraday bars, each session's last value over the previous
            session's; on daily bars, the period returns. None for weekly and
            longer bars.
        ``"Periods"``
            Number of bars, ``len(values) - 1``.
        ``"Total Days"``
            Calendar days from the first to the last date.
        ``"Total Years"``
            The time CAGR uses: on intraday bars, the number of daily
            returns / 252; otherwise ``Periods / periods_per_year``.
        ``"Returns"``
            dict with ``"Initial Bankroll"``, ``"Final Bankroll"``,
            ``"Total Return"``, ``"CAGR"`` and ``"Actual Yearly Returns"``
            (the DataFrame from `calculate_actual_yearly_return`).
        ``"Risk"``
            dict with ``"Period Volatility"`` (per bar, not annualized),
            ``"Annualized Volatility"``, ``"Downside Volatility"`` (annualized),
            ``"Max Drawdown"``, ``"Sharpe Ratio"`` and ``"Sortino Ratio"``.

    Raises
    ------
    ValueError
        If `interval` cannot be parsed or is not a bar size in
        ``PERIODS_PER_YEAR``; the message says which. Also, on intraday bars,
        if a date falls outside the calendar's sessions.

    Notes
    -----
    On intraday bars, the annualized volatility, downside volatility, Sharpe
    and Sortino ratios use daily returns, annualized with 252 sessions a
    year, with the risk-free rate spread as ``risk_free_rate / 252`` per
    session. A bar return that spans the night holds the whole overnight gap
    but is only one bar, so per-bar figures would treat the night as one
    more bar of trading. A daily return runs from one session's last value
    to the next and includes the gap, like every other day's. The first
    session is measured from the first value, so it is a partial session
    when the series starts after the open (e.g. a fill at the open or a
    first close mid-session).

    A value belongs to the session whose date it falls on, on the exchange's
    clock. Extended-hours values join their day's session, so its last value
    is the post-market one rather than the 16:00 close. A single naive time
    that is ambiguous in `tz` (the repeated hour when clocks go back) raises;
    pass timezone-aware dates in that case.

    Intraday prices from yfinance are not dividend-adjusted, so a series
    spanning an ex-date shows the dividend's price drop as a loss in that
    session's return. Adjusted daily closes don't have this problem.

    After a total loss (a value of 0), each later return is 0 / 0 and is left
    out of the period and daily returns, so the risk figures use the returns
    up to and including the -100% one. CAGR is then -100%.

    Total return and max drawdown use every value. CAGR runs from the first
    value to the last and counts time on the clock the risk figures use:
    sessions (252 a year) on intraday bars, bars on daily and longer ones.
    Nights, weekends and holidays add no time, and extended hours, half-days
    and missing bars don't change the count on intraday bars.

    The yearly returns place intraday values in the year of their session
    date, and call a year full when the series covers its first and last
    `calendar` sessions.

    See Also
    --------
    statistical_report : The same metrics across many simulated paths.
    """
    periods_per_year = _periods_per_year(interval)
    kind = _parse_interval(interval)[0]

    # Calculating return series at the chosen interval
    period_returns = calculate_period_returns(values)

    # The risk figures use daily returns. An intraday bar across the night
    # holds the whole overnight gap, so per-bar figures would count the night
    # as one more bar; a session-to-session return includes it like any day.
    if kind == "intraday":
        marks = values.ffill().iloc[_session_marks(dates, calendar, tz)]
        # dropna: after a total loss each return is 0 / 0, left out as in
        # calculate_period_returns, so the risk figures stop at the loss
        daily_returns = marks.pct_change(fill_method=None).iloc[1:].dropna()
        risk_returns, risk_periods = daily_returns, TRADING_DAYS
        # CAGR's clock is the sessions too: one value per session, 252 a year
        clock_values, clock_periods = marks, TRADING_DAYS
    else:
        daily_returns = period_returns if kind == "session" else None
        risk_returns, risk_periods = period_returns, periods_per_year
        clock_values, clock_periods = values, periods_per_year

    # Calculating overall summary metrics. Time is counted in sessions on
    # intraday bars and in bars otherwise; calendar days are for display only.
    initial_bankroll = float(values.iloc[0])
    final_bankroll = float(values.iloc[-1])
    total_return = (final_bankroll / initial_bankroll) - 1
    n_periods = len(values) - 1
    total_days = (dates[-1] - dates[0]).days
    total_years = (len(clock_values) - 1) / clock_periods
    cagr = calculate_cagr(clock_values, clock_periods)
    yearly_returns = calculate_actual_yearly_return(values, dates, calendar=calendar, tz=tz)
    period_vol = period_returns.std()
    ann_vol = risk_returns.std() * np.sqrt(risk_periods)
    # Same downside definition as calculate_sortino_ratio: every period enters
    # the average, so non-negative bars contribute zero rather than being dropped.
    excess_returns = risk_returns - (risk_free_rate / risk_periods)
    downside_dev = np.sqrt(
        (excess_returns.clip(upper=0) ** 2).sum() / len(excess_returns)
    )
    ann_downside_vol = downside_dev * np.sqrt(risk_periods)
    sharpe = calculate_sharpe_ratio(
        risk_returns,
        periods_per_year=risk_periods,
        risk_free_rate=risk_free_rate,
    )
    sortino = calculate_sortino_ratio(
        risk_returns,
        periods_per_year=risk_periods,
        risk_free_rate=risk_free_rate,
    )
    max_drawdown = calculate_max_drawdown(values)

    # Compiling report dictionary, grouped the same way the printed
    # sections are: period context, then returns, then risk.
    report = {
        "Period Returns": period_returns,
        "Daily Returns": daily_returns,
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
            "Period Volatility": period_vol,
            "Annualized Volatility": ann_vol,
            "Downside Volatility": ann_downside_vol,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
        },
    }

    if display:
        separator = "=" * 52
        sub_separator = "-" * 52

        print("\n" + separator)
        print(f"{'PORTFOLIO PERFORMANCE REPORT':^52}")
        print(sub_separator)
        print(f"  Total Time         : {n_periods:>5} periods  ({total_years:.2f} years)")
        print(f"  Calendar Span      : {total_days:>5} days")
        print(separator)
        print(f"{'RETURNS':^52}")
        print(sub_separator)
        print(f"  Initial Bankroll   : {initial_bankroll:>12.2f}")
        print(f"  Final Bankroll     : {final_bankroll:>12.2f}")
        print(f"  Total Return       : {total_return:>12.2%}")
        print(f"  CAGR               : {cagr:>12.2%}")
        print(sub_separator)
        print("  Actual Yearly Returns:")

        for year, row in yearly_returns.iterrows():
            ret_str = f"{row['return']:>12.2%}  ({int(row['days'])} days)"
            print(f"    - {year}            : {ret_str}")

        print(separator)
        print(f"{'RISK (FROM DAILY RETURNS)' if kind == 'intraday' else 'RISK':^52}")
        print(sub_separator)
        print(f"  Volatility         : {ann_vol:>12.2%}")
        print(f"  Downside Vol       : {ann_downside_vol:>12.2%}")
        print(f"  Max Drawdown       : {max_drawdown:>12.2%}")
        print(f"  Sharpe Ratio       : {sharpe:>12.2f}")
        print(f"  Sortino Ratio      : {sortino:>12.2f}")
        print(separator + "\n")

    return report


def _nan_std(returns):
    """Sample standard deviation (ddof=1) of each row, ignoring NaN; NaN for a
    row with fewer than two defined returns."""
    n = (~np.isnan(returns)).sum(axis=1)
    out = np.full(returns.shape[0], np.nan)
    ok = n > 1
    out[ok] = np.nanstd(returns[ok], axis=1, ddof=1)
    return out


def _warn_total_loss_and_exclusions(per_path, n_sims):
    """Warn how many paths end at a total loss, and which summaries leave out
    paths where their metric is undefined (NaN)."""
    parts = []
    n_loss = int((per_path["Final Bankroll"] == 0).sum())
    if n_loss:
        parts.append(
            f"{n_loss} of {n_sims} paths end at a total loss (value 0): their CAGR is -100% and "
            "their risk figures use the returns up to the loss"
        )
    left_out = per_path.isna().sum()
    left_out = left_out[left_out > 0]
    if len(left_out):
        listed = ", ".join(f"{metric} ({n} path{'s' if n != 1 else ''})" for metric, n in left_out.items())
        parts.append(f"left out of the summary where the metric is undefined: {listed}")
    if parts:
        warnings.warn("statistical_report: " + "; ".join(parts) + ".", UserWarning, stacklevel=3)


def _interval_summary(values, confidence):
    """Mean, median and the central `confidence` percentile range of a metric
    across paths. The range is where that share of simulated outcomes landed,
    e.g. confidence=0.95 spans the 2.5th to 97.5th percentiles: a spread of
    outcomes under the fitted model, not a confidence interval for the true
    value. Paths where the metric is undefined (NaN) are left out."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"Mean": np.nan, "Median": np.nan, "Lower": np.nan, "Upper": np.nan}
    tail = (1 - confidence) / 2
    lower, median, upper = np.quantile(values, [tail, 0.5, 1 - tail])
    return {"Mean": values.mean(), "Median": median, "Lower": lower, "Upper": upper}


def statistical_report(
    sims, interval, dates=None, risk_free_rate=0.0, confidence=0.95, display=False,
    calendar="XNYS", tz=None,
):
    """Performance and risk metrics summarized across simulated paths.

    `performance_report` for a matrix of paths: every metric is computed on
    each path, then summarized by its mean, median and the central
    `confidence` range of its values across paths. Tail risk of the total
    return is added.

    Parameters
    ----------
    sims : array-like of shape (n_sims, n_periods + 1)
        One portfolio-value path per row, starting value in column 0, as
        returned by the `portfolio_forecast.forecast` simulators.
    interval : str or pandas.Timedelta
        Bar size of one simulated period: a ``PERIODS_PER_YEAR`` key such
        as ``'1 day'``, or any spelling
        `portfolio_forecast.utils.next_trading_dates` accepts. On daily and
        longer bars it sets the annualization; on intraday bars the risk
        figures and CAGR count sessions, 252 a year.
    dates : array-like of datetimes, optional
        ``n_periods + 1`` dates, one per column of `sims`, e.g. the last
        historical date followed by `next_trading_dates`. Used for the
        calendar span and, on intraday bars, to find each trading session for
        the daily returns, so intraday `sims` require it. Time is counted in
        sessions on intraday bars (daily returns / 252) and in bars
        otherwise.
    risk_free_rate : float, default 0.0
        Annual risk-free rate as a fraction (0.05 = 5%) for the Sharpe and
        Sortino ratios and the downside deviation, spread evenly across the
        year's returns: ``risk_free_rate / 252`` per session on intraday
        and daily bars.
    confidence : float, default 0.95
        Share of paths inside the Lower-Upper range, strictly between 0 and
        1. 0.95 spans the 2.5th to 97.5th percentiles, and sets VaR and CVaR
        at the 5% tail. See Notes: this is a range of outcomes, not a
        confidence interval.
    display : bool, default False
        If True, also print a formatted report.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code whose sessions intraday `dates`
        are grouped into, as in `performance_report`.
    tz : str or tzinfo, optional
        Timezone that naive intraday `dates` are in. Defaults to this
        machine's local timezone.

    Returns
    -------
    dict
        ``"Per Path Metrics"``
            pandas.DataFrame, one row per path, columns ``"Final Bankroll"``,
            ``"Total Return"``, ``"CAGR"``, ``"Period Volatility"`` (per
            bar, not annualized), ``"Annualized Volatility"``, ``"Downside Volatility"``,
            ``"Max Drawdown"``, ``"Sharpe Ratio"``, ``"Sortino Ratio"``.
        ``"Period Returns"``
            numpy.ndarray of shape ``(n_sims, n_periods)``.
        ``"Daily Returns"``
            numpy.ndarray of the daily returns the risk figures use, one row
            per path: session to session on intraday bars, the period returns
            on daily bars, None for weekly and longer bars.
        ``"Paths"``, ``"Periods"``
            n_sims and n_periods.
        ``"Total Days"``
            Calendar days spanned by `dates`, or None without `dates`.
        ``"Total Years"``
            The time CAGR uses: on intraday bars, the number of daily
            returns per path / 252; otherwise ``n_periods / periods_per_year``.
        ``"Confidence"``
            The `confidence` used.
        ``"Returns"``, ``"Risk"``
            pandas.DataFrames with one row per metric (Returns: Final
            Bankroll, Total Return, CAGR; Risk: the five risk metrics above)
            and columns ``Mean``, ``Median``, ``Lower``, ``Upper``. Lower
            and Upper are the ``(1 - confidence) / 2`` and
            ``(1 + confidence) / 2`` quantiles across paths.
        ``"Tail Risk"``
            dict with ``"Probability of Loss"`` (share of paths with a
            negative total return), ``"Value at Risk"`` (the total return at
            the ``1 - confidence`` quantile) and ``"Conditional VaR"`` (the
            mean total return at or below it).

    Raises
    ------
    ValueError
        If `interval` is not a supported bar size, `confidence` is not in
        (0, 1), `sims` has fewer than two periods, or `dates` does not have
        one entry per column of `sims`. On intraday bars, also if `dates` is
        missing or a date falls outside the calendar's sessions.

    Warns
    -----
    UserWarning
        If any path ends at a total loss (a final value of 0), giving how
        many; or if a summary leaves out paths where its metric is
        undefined, naming each metric and how many paths.

    Notes
    -----
    Metrics use the same definitions as `performance_report`, including its
    daily returns for the risk figures on intraday bars. A path that ends at
    0 has a CAGR of -100%, and its risk figures use the returns up to and
    including the -100% one; the 0 / 0 returns after it are NaN in
    ``"Period Returns"`` and ``"Daily Returns"`` and left out. A metric that
    is undefined on a path (CAGR for a path ending below zero, Sharpe or
    Sortino for a path with no variance or no downside) is NaN there and left
    out of that metric's summary, with a warning.

    Lower and Upper are a range of outcomes, not a confidence interval. They
    say where a metric lands over this horizon across paths drawn from the
    fitted model, as if the model were true. They are not an interval for the
    true Sharpe or CAGR, and they leave out the uncertainty in the fitted
    parameters. More paths make the quantiles more precise; they don't narrow
    the range. To test whether a real strategy's Sharpe is above zero or a
    benchmark, use its historical returns (e.g. the probabilistic Sharpe
    ratio, a HAC standard error or a block bootstrap), not simulated paths.

    See Also
    --------
    performance_report : The same metrics for one historical series.
    portfolio_forecast.plotting.plot_simulated_paths : Fan chart of `sims`.

    Examples
    --------
    >>> report = statistical_report(sims, interval="1 day", display=True)
    >>> report["Tail Risk"]["Probability of Loss"]
    """
    periods_per_year = _periods_per_year(interval)
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}")

    values = np.asarray(sims, dtype=float)
    n_sims, n_points = values.shape
    n_periods = n_points - 1
    if n_periods < 2:
        raise ValueError("sims needs at least two simulated periods per path")

    # Time span of the simulation, in bars (in sessions on intraday bars, set
    # below); the calendar span is display only
    total_years = n_periods / periods_per_year
    total_days = None
    if dates is not None:
        dates = pd.to_datetime(dates)
        if len(dates) != n_points:
            raise ValueError(
                f"dates has {len(dates)} entries; expected {n_points} "
                "(one per column of sims, starting value included)"
            )
        total_days = (dates[-1] - dates[0]).days

    # Calculating each path's return series (one row per path). After a total
    # loss each return is 0 / 0 = NaN; the nan-aware statistics below leave
    # those out, as performance_report's dropna does
    with np.errstate(divide="ignore", invalid="ignore"):
        period_returns = values[:, 1:] / values[:, :-1] - 1

    # The risk figures use daily returns, as in performance_report. Intraday
    # paths need dates to find the sessions: per-bar figures would count each
    # overnight gap as one bar, a known-wrong answer, so there is no fallback.
    kind = _parse_interval(interval)[0]
    if kind == "intraday":
        if dates is None:
            raise ValueError(
                "intraday sims need dates to compute the risk figures from daily returns; pass "
                "one date per column, e.g. the last historical date followed by next_trading_dates"
            )
        marks = values[:, _session_marks(dates, calendar, tz)]
        with np.errstate(divide="ignore", invalid="ignore"):
            daily_returns = marks[:, 1:] / marks[:, :-1] - 1
        risk_returns, risk_periods = daily_returns, TRADING_DAYS
        # CAGR's clock is the sessions too, as in performance_report
        total_years = daily_returns.shape[1] / TRADING_DAYS
    else:
        daily_returns = period_returns if kind == "session" else None
        risk_returns, risk_periods = period_returns, periods_per_year

    # Calculating each path's summary metrics, with the same definitions as
    # performance_report's helpers, vectorized across paths
    initial_bankroll = values[:, 0]
    final_bankroll = values[:, -1]
    total_return = final_bankroll / initial_bankroll - 1
    # CAGR is -100% for a path ending at 0 (as calculate_cagr) and undefined for
    # one ending below zero
    growth = final_bankroll / initial_bankroll
    cagr = np.full(n_sims, np.nan)
    positive = growth > 0
    cagr[positive] = growth[positive] ** (1 / total_years) - 1
    cagr[growth == 0] = -1.0

    period_vol = _nan_std(period_returns)
    risk_vol = _nan_std(risk_returns)
    ann_vol = risk_vol * np.sqrt(risk_periods)
    # Every defined period enters the downside average; non-negative bars
    # contribute zero. NaN returns after a total loss are left out of the count
    excess_returns = risk_returns - (risk_free_rate / risk_periods)
    n_defined = (~np.isnan(excess_returns)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        downside_dev = np.sqrt(np.nansum(np.minimum(excess_returns, 0) ** 2, axis=1) / n_defined)
        mean_excess = np.nansum(excess_returns, axis=1) / n_defined
    ann_downside_vol = downside_dev * np.sqrt(risk_periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(risk_vol > 0, mean_excess / risk_vol, np.nan) * np.sqrt(risk_periods)
        sortino = np.where(downside_dev > 0, mean_excess / downside_dev, np.nan) * np.sqrt(risk_periods)
    running_max = np.maximum.accumulate(values, axis=1)
    max_drawdown = (values / running_max - 1).min(axis=1)

    per_path = pd.DataFrame({
        "Final Bankroll": final_bankroll,
        "Total Return": total_return,
        "CAGR": cagr,
        "Period Volatility": period_vol,
        "Annualized Volatility": ann_vol,
        "Downside Volatility": ann_downside_vol,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
    })

    def summarize(columns):
        return pd.DataFrame(
            {col: _interval_summary(per_path[col].to_numpy(), confidence) for col in columns}
        ).T

    returns_summary = summarize(["Final Bankroll", "Total Return", "CAGR"])
    risk_summary = summarize([
        "Period Volatility", "Annualized Volatility", "Downside Volatility",
        "Max Drawdown", "Sharpe Ratio", "Sortino Ratio",
    ])
    _warn_total_loss_and_exclusions(per_path, n_sims)

    # Calculating tail risk of the total return across paths: the chance of
    # ending below the start, Value at Risk (the return at the 1 - confidence
    # quantile) and Conditional VaR (the average return at or below it)
    prob_loss = float((total_return < 0).mean())
    var = float(np.quantile(total_return, 1 - confidence))
    cvar = float(total_return[total_return <= var].mean())

    # Compiling report dictionary, grouped like performance_report's: period
    # context, then returns, then risk. Each summary table has one row per
    # metric and columns Mean, Median, Lower, Upper.
    report = {
        "Per Path Metrics": per_path,
        "Period Returns": period_returns,
        "Daily Returns": daily_returns,
        "Paths": n_sims,
        "Periods": n_periods,
        "Total Days": total_days,
        "Total Years": total_years,
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
        width = 80
        separator = "=" * width
        sub_separator = "-" * width
        level = f"{confidence:.0%}"

        def row(label, stats, fmt):
            mean = format(stats["Mean"], fmt)
            median = format(stats["Median"], fmt)
            band = f"[{format(stats['Lower'], fmt)}, {format(stats['Upper'], fmt)}]"
            return f"  {label:<21}: {mean:>10}   {median:>10}   {band:>28}"

        # Column labels, repeated under each metric section's header
        column_header = f"  {'':<21}  {'Mean':>10}   {'Median':>10}   {level + ' range across paths':>28}"

        print("\n" + separator)
        print(f"{'SIMULATED PORTFOLIO STATISTICAL REPORT':^{width}}")
        print(sub_separator)
        print(f"  {'Simulated Paths':<21}: {n_sims:>5,}")
        print(f"  {'Horizon':<21}: {n_periods:>5} periods  ({total_years:.2f} years)")
        if total_days is not None:
            print(f"  {'Calendar Span':<21}: {total_days:>5} days")
        print(separator)
        print(f"{'RETURNS':^{width}}")
        print(sub_separator)
        print(column_header)
        print(sub_separator)
        print(row("Final Bankroll", returns_summary.loc["Final Bankroll"], ".2f"))
        print(row("Total Return", returns_summary.loc["Total Return"], ".2%"))
        print(row("CAGR", returns_summary.loc["CAGR"], ".2%"))
        print(separator)
        print(f"{'RISK (FROM DAILY RETURNS)' if kind == 'intraday' else 'RISK':^{width}}")
        print(sub_separator)
        print(column_header)
        print(sub_separator)
        print(row("Volatility", risk_summary.loc["Annualized Volatility"], ".2%"))
        print(row("Downside Vol", risk_summary.loc["Downside Volatility"], ".2%"))
        print(row("Max Drawdown", risk_summary.loc["Max Drawdown"], ".2%"))
        print(row("Sharpe Ratio", risk_summary.loc["Sharpe Ratio"], ".2f"))
        print(row("Sortino Ratio", risk_summary.loc["Sortino Ratio"], ".2f"))
        print(separator)
        print(f"{'TAIL RISK (TOTAL RETURN)':^{width}}")
        print(sub_separator)
        print(f"  {'Probability of Loss':<21}: {prob_loss:>10.2%}")
        print(f"  {f'VaR ({level})':<21}: {var:>10.2%}")
        print(f"  {f'CVaR ({level})':<21}: {cvar:>10.2%}")
        print(separator + "\n")

    return report
