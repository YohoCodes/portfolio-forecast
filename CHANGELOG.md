# Changelog

Notable changes to portfolio-forecast. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
