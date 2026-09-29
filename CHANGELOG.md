# Changelog

Notable changes to portfolio-forecast. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `calculate_period_returns(values)` in `portfolio_forecast.performance`: the
  one-period returns of a value series, per bar at the series' interval.
- `performance_report` and `statistical_report` take `calendar` and `tz`, as
  `simulate_buy_and_hold` does, to find the trading sessions of intraday
  values. Both reports also return `"Daily Returns"`.
- `simulate_buy_and_hold` warns when a held symbol's prices stop before the
  last bar (a delisting, an acquisition or a data gap at the end), naming it
  and the date of its last price. On intraday bars the times are the close
  labels the values carry, not the bar starts. The symbol is still held at that price,
  which is the same as selling it there and keeping the cash at 0%, so a
  delisting loss is not booked. Values are unchanged.
- The Monte Carlo simulators warn when `values` looks intraday (times of day
  across several dates). They simulate each session's first return, which
  holds the overnight gap, as an ordinary bar: this distorts the intraday
  path shape and max drawdown, and can make the regime-switching model's
  regimes follow the time of day. Use daily values for the regime-switching
  model. Simulated paths are unchanged.
- `statistical_report` warns how many paths end at a total loss, and which
  summaries leave out paths where their metric is undefined (e.g. Sortino
  on a path with no losing bar), with the number of paths.

### Changed

- On intraday bars, `performance_report` and `statistical_report` compute
  annualized volatility, downside volatility, Sharpe and Sortino from daily
  returns (session close to session close, annualized with 252) instead of
  per-bar returns. A bar return across the night holds the whole overnight
  gap but counted as one bar of trading, which skewed these figures. Daily
  data gives the same numbers as before, and total return and drawdown are
  unchanged. (For CAGR on intraday bars, see below.)
- **Breaking:** `statistical_report` raises `ValueError` for intraday `sims`
  without `dates`, since it needs them to find the sessions. Pass one date
  per column, e.g. the last historical date followed by
  `next_trading_dates`.
- On intraday bars, CAGR and Total Years in `performance_report` and
  `statistical_report` count time in sessions (years = number of daily
  returns / 252) instead of bars. Counting bars assumed a full regular-hours
  session every day, so extended hours, half-days and missing bars misstated
  the time. Hourly or 5-minute CAGR figures change; daily data is unchanged.
- `calculate_actual_yearly_return` takes `calendar` (default `"XNYS"`) and
  `tz`, which `performance_report` passes through. A year now counts as full
  only if the series covers its first and last exchange sessions. Before, a
  start up to January 5 or an end from December 25 counted as full, and
  showed 365 or 366 days. Intraday values go into the year of their session
  date on the exchange's clock, so a New York close labelled 01:00 on
  January 1 in another timezone stays in the old year.
- **Breaking:** `calculate_cagr` raises `ValueError` instead of
  `ZeroDivisionError` for a series with fewer than two values. Catch
  `ValueError` instead.
- `statistical_report`'s Lower and Upper columns are described as the range
  of outcomes across simulated paths, not a confidence interval. The printed
  report and the PDF tables now read "95% range across paths". The range
  shows where a metric lands if the fitted model is true. It leaves out
  uncertainty in the fitted parameters and is not an interval for the true
  Sharpe or CAGR. Keys and numbers are unchanged.
- Documented a limitation: intraday prices from yfinance aren't
  dividend-adjusted, so an ex-date's price drop shows as a loss in that
  session's return in `performance_report`'s risk figures.
- `parametric_monte_carlo` treats a simulated return below −100% as a total
  loss: the path drops to 0 and stays there. Before, such a draw made the
  path's value negative. Paths without such a draw are unchanged.
  `statistical_report` counts these paths with a Total Return, Max Drawdown
  and CAGR of −100%.

### Deprecated

- `calculate_daily_return` is renamed `calculate_period_returns`, since it
  returns per-bar returns (hourly on hourly bars), not daily ones. The old
  name still works but raises a `DeprecationWarning`, and is removed in
  1.0.0. Results are unchanged; switch to the new name.

### Fixed

- `next_trading_dates` no longer adds a bar at an off-grid open (e.g. 09:30
  on hourly bars) to every future session when continuing a
  `simulate_buy_and_hold(..., fill="open")` series. Its first value, at the
  fill, was read as a bar. `statistical_report` with those dates compounded
  one extra bar into each simulated session, so daily volatility and Sharpe
  came out about 7% too high (√(8/7) on regular-hours hourly bars).
- CAGR is −100% when a series ends at 0 and NaN when it ends below 0, in
  `calculate_cagr`, `performance_report` and `statistical_report`. Before,
  `calculate_cagr` gave −100% or NaN depending on the number of bars, and
  `statistical_report` gave NaN, which left total losses out of its CAGR
  summary.
- After a total loss (a value of 0), the returns that follow are 0 / 0 and
  are left out, so the risk figures use the returns up to and including the
  −100% one. Before, `statistical_report` gave such a path NaN volatility,
  Sharpe and Sortino and left it out of those summaries, and
  `performance_report` on intraday bars counted the undefined returns in
  its downside deviation. The two reports now agree.

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
