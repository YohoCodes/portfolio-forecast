# Changelog

Notable changes to portfolio-forecast. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-10-11

Migrating from 0.3.0, one line per breaking change (old → new):

- `values, dates = simulate_buy_and_hold(...)` → `bt = simulate_buy_and_hold(...)`, then `bt.values`, `bt.dates`, `bt.classes`.
- `simulate_buy_and_hold(data, w)` buying at the close → `simulate_buy_and_hold(data, w, fill="close")`; the default is now `"open"`.
- Intraday data without opens → pass data with an `Open` field (intraday backtests need opens).
- Daily backtest labels as dates → timezone-aware session closes (16:00 New York on XNYS).
- `calendar=None` to keep the input labels → `calendar="24/5"` for weekday-only data off an exchange; `None` is now 24/7 continuous mode.
- `next_trading_dates(dates, n_periods)` → `next_trading_dates(values, n).dates`.
- Intraday horizons in bars (`n`, `sim_length`) → sessions.
- `sims = nonparametric_monte_carlo(...)` → `sims.paths` for the array; `sims, fit = ...(return_fit=True)` → `sims.fit`.
- `fit["AIC"]` of `parametric_monte_carlo` → `fit["regular"]["BIC"]`; pass `criterion="AIC"` for the 0.3.0 selection and `fit["regular"]["AIC"]`.
- `parametric_monte_carlo(values, distribution=stats.t)` fitted to simple returns → fitted to log returns $\ell = \ln(1 + r)$, per class.
- Seeded `regime_switching_monte_carlo` paths from hmmlearn's fit → the package's own EM; the same seed gives different paths.
- `import hmmlearn` through this package's dependencies → `pip install hmmlearn` yourself.
- NaN in the simulators' `values` (forward-filled) → `ValueError`; call `.dropna()` first.
- `performance_report(values, dates, interval)` → `performance_report(values)` (or a `BacktestResult`).
- `statistical_report(sims, interval, dates=dates)` → `statistical_report(sims)` with a `SimulationResult`.
- `"Period Volatility"` → `"Gap Volatility"` and `"Bar Volatility"` (per unit) and `"Annualized Volatility"`.
- Risk figures on per-bar returns annualized per bar → session returns annualized with 252.
- CAGR and `"Total Years"` in bars over bars per year → sessions over 252 (`cagr_clock="session"`).
- `"Max Drawdown"` of `-0.25` → `0.25`.
- Sortino `NaN` with no down session → `inf`, with a warning.
- `"Actual Yearly Returns"` `is_full` by 5 January / 24 December → by the first session's open and the last session's close.
- `PERIODS_PER_YEAR["5 mins"]` (and every intraday key) → removed; only `'1 day'`, `'1 week'`, `'1 month'` remain.
- `statistical_report` CAGR `NaN` for a path ending at 0 → `-1.0` (-100%), kept in the summaries.
- `calculate_period_returns(values)` / `calculate_daily_return(values)` dropping the returns after a total loss → kept as NaN; call `.dropna()` for the old series.

### Breaking

- `next_trading_dates(values, n, ...)` returns a `FutureLayout` (`dates`,
  `classes`, `tau`) instead of a `DatetimeIndex`; use `.dates` for the
  labels. It reads the history as marks (a value
  series or its labels) and continues their grid: each future session is an
  open mark (class `"gap"`) and its bar closes (`"regular"`), with half-days
  and short bars from the calendar, and `tau` the length of each return over
  the bar size. The first two parameters are renamed `values` and `n`
  (were `dates` and `n_periods`).
- `next_trading_dates`: on intraday history `n` counts sessions, not bars.
  A history that ends mid-session finishes that session first, and it counts as the
  first of the `n` sessions, so the horizon always ends at a session close.
  This also fixes a spurious extra 9:30 bar per session after
  `fill="open"` history. Daily and longer history: `n` still counts bars.
- `simulate_buy_and_hold` returns a `BacktestResult` (`values`, `dates`,
  `classes`) instead of `(values, dates)`; the old two-name unpacking now
  raises. `classes` gives each return's class
  (`"gap"`, `"regular"`, `"irregular"`) for the simulators and reports.
- `simulate_buy_and_hold` defaults to `fill="open"` (was `"close"`). Data
  without opens raises `ValueError` naming `fill="close"`; pass
  `fill="close"` for closes-only data.
- Intraday data without opens raises `ValueError` in
  `simulate_buy_and_hold`, which now marks each session's open (the
  overnight gap) as its own value, followed by each bar's close.
- Every `simulate_buy_and_hold` label is a timezone-aware moment on the
  exchange's clock: daily values are labelled at each
  session's close (were dates), and naive intraday input gives aware labels
  (was naive). Weekly and monthly bars support `fill="open"`.
- `simulate_buy_and_hold(..., calendar=None)` is continuous mode for 24/7
  data (a constant grid, labels at each bar's start plus the
  bar size), not "keep the input labels". Weekday-only data off an exchange
  (NAVs, indices, FX) should pass `calendar="24/5"`; continuous mode warns
  when more than 10% of returns are irregular.
- `performance_report(values, classes=None, *, interval=None,
  calendar="XNYS", risk_free_rate=0.0, cagr_clock="session", display=False)`:
  `interval` is optional and keyword-only, and the second positional
  argument is now `classes`, so 0.3.0 calls `performance_report(values,
  dates, interval)` raise `TypeError`; call `performance_report(values)`.
- `performance_report`'s `"Period Volatility"` is removed (see `"Gap
  Volatility"` and `"Bar Volatility"`).
- In `performance_report`, on intraday data, volatility, Sharpe and
  Sortino use session returns (last close to last close) annualized with 252, not
  per-bar returns annualized per bar. A partial first session is left out
  (daily data bought at the close keeps its first return), so the figures
  are the same under both fills. Weekly and monthly data annualize their
  own returns with 52 and 12.
- In `performance_report`, CAGR and `"Total Years"` count the sessions
  after the first value, over 252 (`cagr_clock="session"`), not bars; a
  partial first or last session counts whole; `cagr_clock="calendar"` uses
  calendar days over 365.25.
- `"Max Drawdown"` and `calculate_max_drawdown` are a positive fraction (0.25 for a 25% fall; was -0.25), taken over every mark,
  overnight gaps included.
- The Sortino ratio (`calculate_sortino_ratio` and `performance_report`)
  is `inf` (was NaN) when no session is below the risk-free rate, warning
  "Sortino ratio is infinite: none of the n sessions is below the risk-free
  rate"; NaN if the mean excess return is also zero, warning "Sortino ratio
  is NaN: all n sessions return exactly the risk-free rate".
- `"Actual Yearly Returns"` (and `calculate_actual_yearly_return`) marks a
  year full only when the series covers its first session's open and its last session's close (was:
  starts by 5 January and ends after 24 December); a mark's year is its
  session's year on the exchange's clock.
- `PERIODS_PER_YEAR` lists only `'1 day'`, `'1 week'` and `'1 month'`;
  its intraday entries are removed, since risk figures no longer annualize
  per bar.
- The three simulators return a `SimulationResult` (`paths`, `dates`,
  `classes`, `fit`) instead of an array (or `(sims, fit)` with
  `return_fit=True`): use `.paths` for the old array and `.fit` for the
  fit, always filled for the parametric and regime simulators. `dates` and
  `classes` are the future marks from `next_trading_dates`; `None` for a
  plain array.
- The simulators model two return classes: each future gap mark draws from the gaps (last close to next open), each bar from the
  bars, and irregular returns are left out. A short bar
  ($\rho = \tau/\Delta < 1$, e.g. a half-day's last bar) is rescaled by location and scale:
  history $\ell' = \hat m + (\ell - \rho\hat m)/\sqrt\rho$, future
  $\ell = \rho\hat m + \sqrt\rho\thinspace(\ell' - \hat m)$, with
  $\hat m = \sum_k \ell_k / \sum_k \rho_k$ over the bars (the regime's own
  mean in the HMM). Daily histories are one class, as before.
- On intraday history the simulators' `sim_length` counts sessions, not
  bars; if the history ends mid-session, the rest of that
  session counts as the first. Intraday input warns about the change.
  Daily and longer history, and continuous mode, still count bars.
- `parametric_monte_carlo` fits log returns $\ell = \ln(1 + r)$ instead of simple returns, so every path stays
  positive however heavy the fitted left tail; it fits each class on its
  own, selects by BIC by default (was AIC), and its fit is a dict per class
  (`fit["regular"]`, `fit["gap"]`) with `"Distribution"`, `"Parameters"`,
  `"Criterion"`, the score under the criterion's name (`"BIC"`; the old
  top-level `"AIC"` key is gone) and `"Candidates"`. `distribution` may be
  a dict `{"gap": ..., "regular": ...}`. Student's $t$ and Johnson SU are
  fitted from several starting points and polished to the maximum
  likelihood, since scipy's default $t$ fit can stop well short of it on
  fat-tailed returns and so pick the wrong family.
- `regime_switching_monte_carlo` fits its hidden Markov model with the
  package's own EM instead of `hmmlearn`, so seeded paths differ
  from 0.3.0. Its fit adds `"Gap Model"`, `"Kappa"` and `"Criterion"`, the
  scores are under the criterion's name, `"Regimes"` gains `"Gap Mean"` and
  `"Gap Volatility"` under `gap_model="regime"`, `"Gap"` holds the gap
  normal under `"independent"` with `bootstrap=False` (with
  `bootstrap=True` gaps are resampled and no gap normal is fitted), and
  `"Expected Duration"` is in marks.
- The simulators raise `ValueError` for non-finite or non-positive values
  (NaN was forward-filled; drop missing values first), for fewer than
  `min_sample` returns in a class (default 30), for a series with one mark
  per session away from the close, for intraday labels with neither
  `classes` nor open marks, for an `n_sims` that is not a positive integer,
  and for a negative `sim_length`.
- `hmmlearn` is no longer installed with the package; it moves to the
  `test` extra (`pip install portfolio-forecast[test]`), where
  it serves as the reference the package's own HMM fit is checked against.
- `statistical_report` computes every path's figures under
  `performance_report`'s rules: volatility, Sharpe and Sortino on session
  returns annualized with 252 (52 or 12 on weekly or monthly bars), not
  per-bar returns annualized per bar, leaving out a partial first session;
  CAGR and `"Total Years"` on `cagr_clock` (sessions over 252 by default;
  was bars over bars per year). Intraday paths need their marks (a
  `SimulationResult`, or timezone-aware `dates`): an intraday `interval`
  without them raises `ValueError`. Plain arrays still need `interval`.
- `statistical_report`'s `"Period Volatility"` is removed (see `"Gap Volatility"` and `"Bar Volatility"`), and `"Max Drawdown"` is
  a positive fraction as in `performance_report` (0.25 for a 25% fall; was
  -0.25).
- In `statistical_report`, a path ending at 0 has CAGR -100% (was NaN)
  and stays in every summary, with its returns after the total loss NaN in
  `"Period Returns"`; a path with no down session has Sortino `inf` (was
  NaN), kept in the summary with inf-safe percentiles; the Sortino mean
  across paths is then NaN.
- `calculate_period_returns` and `calculate_session_returns` (and so
  `performance_report`'s `"Period Returns"` and `"Daily Returns"`) keep the
  returns after a total loss as NaN (0/0) instead of dropping them, so
  their length follows the input: `len(values) - 1` returns between marks,
  one return per full session. The -100% return itself is unchanged, and
  the risk figures drop the NaN rows, so their values are unchanged.

### Added

- `BacktestResult`, `SimulationResult` and `FutureLayout` in
  `portfolio_forecast.types`, re-exported from `portfolio_forecast`: the
  NamedTuples the backtest, the simulators and `next_trading_dates` return.
- `classify_returns(values, calendar)` in `portfolio_forecast.utils`: the
  class of each return (`"gap"`, `"regular"`, `"irregular"`) of a value
  series, inferred from its labels on the exchange calendar, for Monte Carlo
  input from an external backtest. It sees labels only, not stale prices.
- `to_session_values(values, calendar)` in `portfolio_forecast.utils`:
  collapses an intraday value series (or `BacktestResult`) to its first value
  and each session's last value, for a daily forecast from intraday history.
  A session whose last mark is before its close (its final bars missing) has
  its last value labelled at the close, with one warning giving the count
  and the first such session. `calculate_session_returns` and the reports'
  session returns carry the same labels, without the warning. A last session
  still in progress at `now` (a new keyword, default the current time) is
  left out of the session values and of a history's session returns.
- A series with one mark per session away from the close (e.g. 12:00 every
  session) is neither daily nor intraday: `classify_returns` and
  `next_trading_dates` raise `ValueError` naming `to_session_values`, which
  accepts it and labels each session's value at that session's close.
- Continuous mode, `calendar=None`, for 24/7 data such as crypto, in
  `is_daily`, `classify_returns`, `to_session_values`, `next_trading_dates`,
  `close_times`, `session_spans` and `completed_mask`: a constant grid of
  bar size Δ (the most common spacing), no sessions or gaps, every return
  `"regular"` unless it spans a missing grid point. `is_daily` is true when
  Δ is one day; `to_session_values` gives the value at each 00:00 UTC;
  `next_trading_dates` continues $t_T + k\Delta$ for `n` bars. More than 10%
  irregular returns warns that the data is not 24/7 and names
  `calendar="24/5"` or an exchange calendar.
- `calculate_period_returns(values)` (returns between consecutive marks) and
  `calculate_session_returns(values, calendar)` (the session returns behind
  volatility, Sharpe and Sortino) in `portfolio_forecast.performance`.
- `performance_report` keys `"Gap Volatility"` and `"Bar Volatility"` (per
  gap and per full bar, unannualized, NaN on daily data; short bars scaled
  to a full bar in logs) and `"Daily Returns"` (the session returns). It
  takes `cagr_clock` (`"session"`, the default, or `"calendar"`), accepts a
  `BacktestResult` (using its classes) or explicit `classes`, and supports
  continuous mode (`calendar=None`): risk figures on calendar-day returns at
  00:00 UTC, annualized with 365, leaving out the first UTC day; bars of a
  day or longer use their own returns, annualized with 365 days over the
  bar size; `cagr_clock="calendar"` only.
- `nonparametric_monte_carlo`, `parametric_monte_carlo` and
  `regime_switching_monte_carlo` take keyword-only `classes` (each return's
  class; default: a `BacktestResult`'s, or `classify_returns`), `calendar`
  (default `"XNYS"`; `None` is continuous mode for 24/7 data: one class, no
  rescaling, future marks $t_T + k\Delta$, `sim_length` in bars) and
  `min_sample` (default 30: a class with fewer returns raises, naming the
  class, its count and the history needed; 0 disables the check). `values`
  may be a `BacktestResult`.
- `parametric_monte_carlo` and `regime_switching_monte_carlo` take
  `criterion` (`"BIC"`, the default, or `"AIC"`, any case) for the family
  and the regime count.
- `regime_switching_monte_carlo` takes `gap_model` (`"regime"`, the
  default: each regime has its own gap emission and the chain steps at
  gaps too; `"independent"`: the chain runs on bars only and gaps are one
  normal or one pool) and `kappa` (default 0.0: a sticky prior on staying
  in a regime; larger values give longer-lived regimes).
- `parametric_monte_carlo` and `regime_switching_monte_carlo` take
  `verbose` (keyword only, default True): True prints the selection and
  regime tables, False prints nothing to stdout; warnings are
  unaffected.
- The verbose printouts show the fitted model. `parametric_monte_carlo`
  adds the fitted parameters by name to each `Selected:` line, e.g.
  `Selected: t  (df 4.12, loc 0.000912, scale 0.00981)`, and with a family
  given prints one `Fitted:` line per class in the same format (log return
  per gap or per full bar). `regime_switching_monte_carlo` labels each
  regime's moments `regular` on intraday history and, under
  `gap_model="regime"` with gaps, adds the regime's gap mean and volatility.
- `statistical_report(sims, interval=None, dates=None, ..., *,
  calendar="XNYS", cagr_clock="session")` accepts a `SimulationResult`: the
  future marks and classes come from it, and column 0 stands at the last
  historical mark. It adds `"Gap Volatility"` and `"Bar Volatility"` (per
  path and summarized) and `"Daily Returns"` (each path's session returns),
  and supports continuous mode. It warns with the count and share of paths
  whose Sortino ratio is infinite ("Sortino ratio is infinite on k of N
  paths (p%); its mean is NaN"), the count of paths ending at a total
  loss, and the paths left out of a summary because a figure is undefined.
- `next_trading_dates` accepts a `DataFrame` of any number of columns, such
  as `yfinance.download(...)["Close"]`, and reads its index, as it does for
  a Series; the result is the same as passing the index.
- `calculate_cagr(values, periods_per_year=None, *, years=None)` takes the
  span in years (e.g. from `calculate_total_years`) instead of a bar count;
  it is -100% when the final value is 0 and NaN below 0.
- `calculate_actual_yearly_return(values, dates=None, calendar="XNYS")`:
  `dates` is optional (the labels come from `values`), and `calendar`
  places each mark in its session's year (`None`: continuous mode, years at
  00:00 UTC on 1 January).
- `calculate_sharpe_ratio` and `calculate_sortino_ratio` default to
  `periods_per_year=252`.
- `calculate_cagr`, `calculate_max_drawdown`,
  `calculate_actual_yearly_return`, the return functions,
  `classify_returns` and `to_session_values` accept a `BacktestResult` as
  `values`.
- The simulators accept a NumPy array, a list or a Series on a
  `RangeIndex` as a plain array: one class, no labels, `dates` and
  `classes` of the result `None`.
- Module-level functions, importable from their modules and not
  re-exported: `session_spans` and `completed_mask` in
  `portfolio_forecast.utils.calendar`; `is_daily` and `history_marks` in
  `portfolio_forecast.utils.marks`; `count_sessions`,
  `calculate_total_years`, `calculate_excess_returns`,
  `calculate_volatility`, `calculate_downside_deviation` and
  `calculate_class_volatility` in `portfolio_forecast.performance.metrics`.
- `plot_simulated_paths` and `plot_path_comparison` accept a
  `SimulationResult`: its future marks label the x-axis, one step per mark
  (each session's open mark is its own step), and column 0 reads "Start".
  Passing `dates` still labels every column, as before.

### Changed

- The demo notebooks are replaced by three new ones in `demos/`:
  `01_quick_start_daily.ipynb` (a daily quick start through to a PDF
  report), `02_intraday_sessions_and_gaps.ipynb` (hourly sessions, gaps,
  return classes and continuous mode) and `03_choosing_a_simulator.ipynb`
  (the three simulators on a synthetic history with a known model).
  `demo_full_report.ipynb` and `demo_simulate_portfolio.ipynb` are removed.
- The PyPI project page shows `PYPI.md`, a short math-free description
  with installation and the quick start, linking to the full README on
  GitHub (PyPI doesn't render the README's math).
- The source distribution now ships the full test suite (with the golden
  cases), `CHANGELOG.md` and `CONTRIBUTING.md`, so `pytest` runs from an
  unpacked sdist; the demo notebooks stay out.
- `simulate_buy_and_hold` takes `now` (default: the current time) and drops
  bars still in progress at `now`, without a warning.
- `simulate_buy_and_hold` warns, each with a count and the symbols or dates
  involved: intraday bars outside every session, which are dropped
  (extended hours, weekends of 24/7 assets); held symbols with no price at
  the fill, which are dropped with the weights renormalized (message
  reworded, now judged on the fill bar for both fills); held symbols whose
  prices stop early, which are held as cash at their last price; sessions
  missing their first bar; and returns made irregular by a stale price or
  a missing scheduled bar.
- Daily bars dated on a day that is not a session of `calendar` raise
  `ValueError` in `simulate_buy_and_hold` for both fills.
- `performance_report` raises `ValueError` for fewer than two values, and
  CAGR is -100% when the final value is 0; returns after a total loss are
  left out of the risk figures.
- `classify_returns` classes a return `"irregular"` when its span holds a
  scheduled mark missing from the series: a mid-session bar, or a whole
  session that is not a holiday. Holidays keep their gap.
- A session that opens the instant the previous one closed (weeknights on
  `calendar="24/5"`) has no gap: `classify_returns` and `next_trading_dates`
  class its first bar `"regular"`, and future layouts give it no open mark.
- `calculate_sharpe_ratio` is NaN for zero variance or fewer than two
  returns (was infinite or NaN).
- `close_times` moved to `portfolio_forecast.utils.calendar`;
  `portfolio_forecast.utils.bar_times` still imports it, and
  `from portfolio_forecast.utils import close_times` is unchanged.
- `statistical_report` raises `ValueError` only for fewer than two marks
  per path (was fewer than two periods).
- `statistical_report` describes Lower/Upper as the "95% range across
  paths" (for the default `confidence`), in its docstring and printed
  headers (was "95% interval"): they are percentiles across paths, not a
  confidence interval. The keys are unchanged.
- `compile_statistical_reports` tables follow the new keys (Gap and Bar
  Volatility rows, no Period Volatility) and the same wording: the Lower
  and Upper columns are headed "95% range across paths" (was "95%
  interval") and the setup row "Confidence level" is "Range across paths".
  An infinite Sortino ratio prints as $\infty$.
- The printed `performance_report` and `statistical_report` headers read
  `Total Time` in calendar years (first to last mark over 365.25 days, on
  every series kind and either `cagr_clock`) with the whole calendar days
  in brackets, `Total Time : 3.00 years (1095 days)`, then the number of
  returns (periods), with no clock label. The Gap Vol row
  prints only if the series has a gap return, and the Bar Vol row only on
  intraday series; the returned keys are unchanged (NaN when absent), and
  `"Total Years"` and CAGR still follow `cagr_clock`. `statistical_report`
  adds `"Total Time (Years)"` (None for paths without dates), which the
  printout and the `compile_statistical_reports` setup table show: its
  Horizon row no longer carries years on `cagr_clock`, and a "Total time"
  row in years and days precedes it, replacing the calendar span row. Its risk table follows the same row rule as the
  printout, read from the new `"Has Gaps"` and `"Intraday"` keys of
  `statistical_report`.

### Deprecated

- `calculate_daily_return`: use `calculate_period_returns` (same result) or
  `calculate_session_returns`. It raises a `DeprecationWarning` and is
  removed in 1.0.0.
- `performance_report(..., dates=...)`: the labels come from `values` (or a
  `BacktestResult`). It raises a `DeprecationWarning` and is removed in
  1.0.0.
- `return_fit` of `parametric_monte_carlo` and
  `regime_switching_monte_carlo`: ignored, since the fit is always in
  `SimulationResult.fit`. It raises a `DeprecationWarning` and is removed in
  1.0.0.

### Fixed

- `plot_simulated_paths` and `plot_path_comparison` labelled the y-axis
  "Portfolio Value (start = 1)" and drew the starting-value line at the
  first path's start whatever the paths started at. The label now names
  the shared start (`start = 100` for paths starting at 100), and paths
  with mixed starts read "Portfolio Value" with no starting-value line.

## [0.3.0] - 2026-09-24

### Added

- `interval` accepts yfinance spellings (`'1m'`, `'5m'`, `'60m'`, `'1h'`,
  `'1d'`, `'1wk'`, `'1mo'`) wherever it accepts a bar size:
  `statistical_report`, `performance_report` and `next_trading_dates`. A bare
  `m` is minutes and a bare `M` is still months.
- `simulate_buy_and_hold(..., fill="open")` buys at the first bar's open
  instead of its close, so the first bar's own move is counted. The first
  value is `br0` at the first bar's start (daily bars: the session's open),
  then one value per bar close. Opens are scaled onto an adjusted close's
  basis when there is one.
- `close_times(index)` in `portfolio_forecast.utils`: the moment each
  start-labelled bar closes, from the exchange calendar (16:00, or 13:00 on
  half-days).
- `simulate_buy_and_hold` takes `calendar` and `tz`, as `next_trading_dates`
  does.

### Changed

- An unsupported `interval` now says why it was rejected: either the unit
  was not recognized, or the bar size it was read as (e.g. `'90m'` as
  90-minute bars) has no `PERIODS_PER_YEAR` entry.

### Fixed

- `simulate_buy_and_hold` labelled intraday values with their bar's start
  time, one bar before the close they were computed from. Intraday values are
  now labelled with their bar's close time (e.g. an hourly 9:30 bar's value at
  10:00, the 15:00 bar's at 16:00); the values themselves are unchanged. Daily
  and longer bars keep their dates. Pass `calendar=None` for the old labels.

### Removed

- `simulate_rebalancing_MVO`. It backtested weights from a mean-variance
  optimizer that was never part of this package, and treated weights summing
  to less than 1 as fully invested instead of holding the rest as cash. To
  keep using it, pin `portfolio-forecast<0.3` or copy it from the 0.2.0
  source. `simulate_buy_and_hold` is unaffected.

## [0.2.0] - 2026-09-21

### Added

- `simulate_buy_and_hold` accepts weights as a dict, aligned by symbol like a
  Series.
- `simulate_buy_and_hold` warns when it drops a symbol for missing prices,
  naming it, how many returns it is missing and the date of its first price.
  Previously the symbol was dropped silently and the other weights scaled up.

### Changed

- `get_closing_prices`, and so `simulate_buy_and_hold` and
  `simulate_rebalancing_MVO`, prefers an adjusted close (`Adj Close`) over
  `Close` when both are present. Backtests of
  `yfinance.download(..., auto_adjust=False)` data now include dividends, so
  their results will be higher than in 0.1.1.
- The first parameter of `nonparametric_monte_carlo`, `parametric_monte_carlo`
  and `regime_switching_monte_carlo` is now `values` (was `prices`), since it
  can be prices or portfolio values. Positional calls are unaffected.

### Deprecated

- `prices=` in the three Monte Carlo simulators. It still works, with a
  `DeprecationWarning`; use `values=` instead. It will be removed in 1.0.0.

### Fixed

- Return calculations no longer rely on pandas' deprecated `pct_change` fill
  default, so results are the same under pandas 3, and pandas 2 no longer
  emits a `FutureWarning` for data with gaps.
