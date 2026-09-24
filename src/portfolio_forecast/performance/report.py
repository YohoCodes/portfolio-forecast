import numpy as np
import pandas as pd

from ..utils.periods import _periods_per_year


# Helper function for Daily Returns
def calculate_daily_return(values):
    """One-period simple returns of a value series.

    Despite the name, the returns are per bar at whatever interval `values`
    is sampled on (hourly bars give hourly returns).

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
    # Calculating percentage change in portfolio value, measuring a return across a
    # gap from the last known value (explicit, as pandas 3 no longer fills by default)
    daily_returns = values.ffill().pct_change(fill_method=None).dropna()
    return daily_returns


# Helper function for CAGR (compound annual growth rate)
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
        ``years = (len(values) - 1) / periods_per_year``.

    Raises
    ------
    ZeroDivisionError
        If `values` has only one entry (zero elapsed time).

    Notes
    -----
    Time is counted in bars, not calendar days, the same clock the Sharpe and
    volatility annualization use: nights, weekends and holidays add no time.
    Over short horizons the annualization magnifies the total return
    enormously (a 1% gain over one day is a CAGR of about 1,100%).
    """
    # Calculating total return and converting the bar count to years
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    num_years = (len(values) - 1) / periods_per_year

    # Calculating CAGR via compound expansion
    cagr = (1 + total_return) ** (1 / num_years) - 1
    return cagr


# Helper function for Sharpe Ratio
def calculate_sharpe_ratio(period_returns, periods_per_year, risk_free_rate=0.0):
    """Annualized Sharpe ratio of a return series.

    Parameters
    ----------
    period_returns : pandas.Series
        One-period simple returns, e.g. from `calculate_daily_return`.
    periods_per_year : float
        Bars in one year at the returns' interval, used to annualize.
    risk_free_rate : float, default 0.0
        Annual risk-free rate, spread evenly across the year's bars.

    Returns
    -------
    float
        ``mean(r - rf / periods_per_year) / std(r) * sqrt(periods_per_year)``,
        with the sample standard deviation (``ddof=1``). Infinite or NaN if
        the returns have zero variance.
    """
    # Calculating mean and standard deviation of excess returns
    excess_returns = period_returns - (risk_free_rate / periods_per_year)
    mean_excess_return = excess_returns.mean()
    std_dev = period_returns.std()

    # Calculating annualized Sharpe Ratio
    sharpe_ratio = (mean_excess_return / std_dev) * np.sqrt(periods_per_year)
    return sharpe_ratio


# Helper function for Sortino Ratio
def calculate_sortino_ratio(period_returns, periods_per_year, risk_free_rate=0.0):
    """Annualized Sortino ratio of a return series.

    Like the Sharpe ratio, but divides by downside deviation, so only returns
    below the risk-free rate count as risk.

    Parameters
    ----------
    period_returns : pandas.Series
        One-period simple returns, e.g. from `calculate_daily_return`.
    periods_per_year : float
        Bars in one year at the returns' interval, used to annualize.
    risk_free_rate : float, default 0.0
        Annual risk-free rate, spread evenly across the year's bars. It is
        also the target below which a return counts as downside.

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

    # Calculating annualized Sortino Ratio
    sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(
        periods_per_year
    )
    return sortino_ratio


# Helper function for Max Drawdown
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


# Helper function for Actual Yearly Return
def calculate_actual_yearly_return(values, dates):
    """Return earned in each calendar year of a value series.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values, one per bar, oldest first.
    dates : array-like of datetimes
        The date of each value, same length as `values`.

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
            False for the first year if the data starts after January 5, and
            for the last year if it ends before December 25.
    """
    # Grouping portfolio values by calendar year to compute annual returns
    values_series = pd.Series(values.values, index=pd.to_datetime(dates))
    yearly_groups = values_series.groupby(values_series.index.year)

    # Global min and max dates across the entire dataset
    global_start = values_series.index[0]
    global_end = values_series.index[-1]

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

        # Check leap year status
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        full_year_days = 366 if is_leap_year else 365

        # A year is partial ONLY if it contains the absolute start or end of the entire dataset
        is_first_year = (year == global_start.year) and (
            global_start.month > 1 or global_start.day > 5
        )
        is_last_year = (year == global_end.year) and (
            global_end.month < 12 or global_end.day < 25
        )

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


# Main reporting function
def performance_report(
    values, dates, interval, risk_free_rate=0.0, display=False
):
    """Performance and risk metrics for one portfolio value series.

    Parameters
    ----------
    values : pandas.Series
        Portfolio values, oldest first, e.g. from `simulate_buy_and_hold`.
    dates : pandas.DatetimeIndex or array-like of datetimes
        The date of each value, same length as `values`. Used for the
        calendar-year returns and the displayed calendar span.
    interval : str or pandas.Timedelta
        Bar size of `values`, used to annualize: a ``PERIODS_PER_YEAR``
        key such as ``'1 day'`` or ``'5 mins'``, or any spelling
        `portfolio_forecast.utils.next_trading_dates` accepts (``'1D'``,
        ``'5min'``, yfinance's ``'5m'``, a Timedelta).
    risk_free_rate : float, default 0.0
        Annual risk-free rate for the Sharpe and Sortino ratios and the
        downside deviation.
    display : bool, default False
        If True, also print a formatted report.

    Returns
    -------
    dict
        ``"Period Returns"``
            pandas.Series of one-period returns.
        ``"Periods"``
            Number of bars, ``len(values) - 1``.
        ``"Total Days"``
            Calendar days from the first to the last date.
        ``"Total Years"``
            ``Periods / periods_per_year``, the time every annualized figure
            uses.
        ``"Returns"``
            dict with ``"Initial Bankroll"``, ``"Final Bankroll"``,
            ``"Total Return"``, ``"CAGR"`` and ``"Actual Yearly Returns"``
            (the DataFrame from `calculate_actual_yearly_return`).
        ``"Risk"``
            dict with ``"Period Volatility"`` (per bar), ``"Annualized
            Volatility"``, ``"Downside Volatility"`` (annualized),
            ``"Max Drawdown"``, ``"Sharpe Ratio"`` and ``"Sortino Ratio"``.

    Raises
    ------
    ValueError
        If `interval` cannot be parsed or is not a bar size in
        ``PERIODS_PER_YEAR``; the message says which.

    Notes
    -----
    Every annualized figure (CAGR, volatility, Sharpe, Sortino) counts time
    in bars, so nights, weekends and holidays add no time. ``PERIODS_PER_YEAR``
    assumes regular trading hours; intraday bars that include extended hours
    are annualized with too few bars per year.

    See Also
    --------
    statistical_report : The same metrics across many simulated paths.
    """
    periods_per_year = _periods_per_year(interval)

    # Calculating return series at the chosen interval
    period_returns = calculate_daily_return(values)

    # Calculating overall summary metrics. Time is counted in bars, the clock
    # every annualized figure here uses; calendar days are for display only.
    initial_bankroll = float(values.iloc[0])
    final_bankroll = float(values.iloc[-1])
    total_return = (final_bankroll / initial_bankroll) - 1
    n_periods = len(values) - 1
    total_days = (dates[-1] - dates[0]).days
    total_years = n_periods / periods_per_year
    cagr = calculate_cagr(values, periods_per_year)
    yearly_returns = calculate_actual_yearly_return(values, dates)
    period_vol = period_returns.std()
    ann_vol = period_vol * np.sqrt(periods_per_year)
    # Same downside definition as calculate_sortino_ratio: every period enters
    # the average, so non-negative bars contribute zero rather than being dropped.
    excess_returns = period_returns - (risk_free_rate / periods_per_year)
    downside_dev = np.sqrt(
        (excess_returns.clip(upper=0) ** 2).sum() / len(excess_returns)
    )
    ann_downside_vol = downside_dev * np.sqrt(periods_per_year)
    sharpe = calculate_sharpe_ratio(
        period_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    sortino = calculate_sortino_ratio(
        period_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    max_drawdown = calculate_max_drawdown(values)

    # Compiling report dictionary, grouped the same way the printed
    # sections are: period context, then returns, then risk.
    report = {
        "Period Returns": period_returns,
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

    # Displaying formatted performance report if enabled
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
        print(f"{'RISK':^52}")
        print(sub_separator)
        print(f"  Volatility         : {ann_vol:>12.2%}")
        print(f"  Downside Vol       : {ann_downside_vol:>12.2%}")
        print(f"  Max Drawdown       : {max_drawdown:>12.2%}")
        print(f"  Sharpe Ratio       : {sharpe:>12.2f}")
        print(f"  Sortino Ratio      : {sortino:>12.2f}")
        print(separator + "\n")

    return report


# Helper function for summarizing a metric across simulated paths
def _interval_summary(values, confidence):
    """Mean, median and the central `confidence` percentile band of a metric
    across paths. The band is where that share of simulated outcomes landed,
    e.g. confidence=0.95 spans the 2.5th to 97.5th percentiles. Paths where
    the metric is undefined (NaN) are left out."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"Mean": np.nan, "Median": np.nan, "Lower": np.nan, "Upper": np.nan}
    tail = (1 - confidence) / 2
    lower, median, upper = np.quantile(values, [tail, 0.5, 1 - tail])
    return {"Mean": values.mean(), "Median": median, "Lower": lower, "Upper": upper}


# Main statistical reporting function for simulated paths
def statistical_report(
    sims, interval, dates=None, risk_free_rate=0.0, confidence=0.95, display=False
):
    """Performance and risk metrics summarized across simulated paths.

    `performance_report` for a matrix of paths: every metric is computed on
    each path, then summarized by its mean, median and a central `confidence`
    interval across paths. Tail risk of the total return is added.

    Parameters
    ----------
    sims : array-like of shape (n_sims, n_periods + 1)
        One portfolio-value path per row, starting value in column 0, as
        returned by the `portfolio_forecast.forecast` simulators.
    interval : str or pandas.Timedelta
        Bar size of one simulated period, used to annualize: a
        ``PERIODS_PER_YEAR`` key such as ``'1 day'``, or any spelling
        `portfolio_forecast.utils.next_trading_dates` accepts.
    dates : array-like of datetimes, optional
        ``n_periods + 1`` dates, one per column of `sims`, e.g. the last
        historical date followed by `next_trading_dates`. Only used for the
        calendar span; time is always counted in bars.
    risk_free_rate : float, default 0.0
        Annual risk-free rate for the Sharpe and Sortino ratios and the
        downside deviation.
    confidence : float, default 0.95
        Coverage of the intervals, strictly between 0 and 1. 0.95 spans the
        2.5th to 97.5th percentiles, and sets VaR and CVaR at the 5% tail.
    display : bool, default False
        If True, also print a formatted report.

    Returns
    -------
    dict
        ``"Per Path Metrics"``
            pandas.DataFrame, one row per path, columns ``"Final Bankroll"``,
            ``"Total Return"``, ``"CAGR"``, ``"Period Volatility"``,
            ``"Annualized Volatility"``, ``"Downside Volatility"``,
            ``"Max Drawdown"``, ``"Sharpe Ratio"``, ``"Sortino Ratio"``.
        ``"Period Returns"``
            numpy.ndarray of shape ``(n_sims, n_periods)``.
        ``"Paths"``, ``"Periods"``
            n_sims and n_periods.
        ``"Total Days"``
            Calendar days spanned by `dates`, or None without `dates`.
        ``"Total Years"``
            ``n_periods / periods_per_year``.
        ``"Confidence"``
            The `confidence` used.
        ``"Returns"``, ``"Risk"``
            pandas.DataFrames with one row per metric (Returns: Final
            Bankroll, Total Return, CAGR; Risk: the five risk metrics above)
            and columns ``Mean``, ``Median``, ``Lower``, ``Upper``.
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
        one entry per column of `sims`.

    Notes
    -----
    Metrics use the same definitions as `performance_report`. A metric that
    is undefined on a path (CAGR for a path ending at or below zero, Sharpe
    or Sortino for a path with no variance or no downside) is NaN there and
    left out of that metric's summary.

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

    # Time span of the simulation, in bars; the calendar span is display only
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

    # Calculating each path's return series (one row per path)
    period_returns = values[:, 1:] / values[:, :-1] - 1

    # Calculating each path's summary metrics, with the same definitions as
    # performance_report's helpers, vectorized across paths
    initial_bankroll = values[:, 0]
    final_bankroll = values[:, -1]
    total_return = final_bankroll / initial_bankroll - 1
    # CAGR is undefined for a path that ends at or below zero
    growth = final_bankroll / initial_bankroll
    cagr = np.full(n_sims, np.nan)
    positive = growth > 0
    cagr[positive] = growth[positive] ** (1 / total_years) - 1

    period_vol = period_returns.std(axis=1, ddof=1)
    ann_vol = period_vol * np.sqrt(periods_per_year)
    # Every period enters the downside average; non-negative bars contribute zero
    excess_returns = period_returns - (risk_free_rate / periods_per_year)
    downside_dev = np.sqrt((np.minimum(excess_returns, 0) ** 2).mean(axis=1))
    ann_downside_vol = downside_dev * np.sqrt(periods_per_year)
    mean_excess = excess_returns.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(period_vol > 0, mean_excess / period_vol, np.nan) * np.sqrt(periods_per_year)
        sortino = np.where(downside_dev > 0, mean_excess / downside_dev, np.nan) * np.sqrt(periods_per_year)
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

    # Displaying formatted statistical report if enabled
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
        column_header = f"  {'':<21}  {'Mean':>10}   {'Median':>10}   {level + ' interval':>28}"

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
        print(f"{'RISK':^{width}}")
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
