# Golden tests

Hand-computed cases on tiny synthetic price tables. Each YAML file holds the
inputs, the expected value at **every** point with its label, and the
derivation in its comments. The expected numbers come from the derivation,
never from running the package. `test_golden.py` runs every case in
`simulate/` and `metrics/`.

**A case is never edited to make the code pass.** A failing case means the
code is wrong until shown otherwise. If a derivation really is wrong, fix it
in its own commit that says why, and have it reviewed on its own. A rule
changes only with Isaac's approval; the cases for it are rewritten first,
and fail until the code follows.

The rules follow BacktestingPipeline's (`CLIENT_SPECS.md` §4 and §8) where
this package does the same job.

## Why the derivations use shares

The package computes a buy-and-hold book from returns: it compounds each
symbol's returns and weights them. The cases work in shares instead: buy
q = w · br0 / P_fill of each symbol at the fill, then the book is worth
V = Σ q · P at every close. The two methods agree only if the package is
right, so a case checks the code rather than repeating its formula.

## How to read a case

1. Read the comment block at the top. It states the setup, the one rule
   under test, and the derivation, one value per line.
2. Check each derived value against `expected.values`, then any
   `expected.report` figures against their derivation.

## Simulation cases (`simulate/`)

| Key | Meaning |
|---|---|
| `fields` | The price fields each bar lists, e.g. `[Open, Close]`. The runner lays them out like `yfinance.download`: `(Price, Ticker)` columns. |
| `bars_tz` | Timezone the bar times are written in, or `null` for naive times. |
| `bars` | `SYMBOL: [[time, value per field], ...]`. `null` is a missing price. |
| `weights`, `br0`, `fill`, `tz` | Passed to `simulate_buy_and_hold`. `calendar` defaults to `XNYS`. |
| `interval` | Bar size, for the report checks. |
| `expected.label_tz` | Timezone the output labels must carry, or `null` for naive. Label strings are on that clock. |
| `expected.values` | `[label, V]` for every value, in order. |
| `expected.fill_prices`, `expected.shares` | Each held symbol's fill price and share count. |
| `expected.warnings` | Optional: text each `UserWarning` must contain, in order. No key means no warning. |
| `expected.report` | Optional: `performance_report` figures on what the package simulated. |

## Metrics cases (`metrics/`)

| Key | Meaning |
|---|---|
| `interval`, `risk_free_rate`, `tz`, `calendar` | Passed to `performance_report` (and to `statistical_report` for the cross-check). `calendar` defaults to `XNYS`. |
| `values` | `[date, V]`, oldest first. |
| `expected` | `performance_report` figures. |

A report block names figures by their `performance_report` keys (`Periods`,
`Total Return`, `Sharpe Ratio`, ...). `Period Returns` and `Daily Returns`
are lists, and
`Yearly Returns` lists `[year, return, days, is_full]`. Only the figures a
case names are checked.

## Checks the runner adds to every case

- **The case's arithmetic** (`test_case_arithmetic`). Without running the
  package, the shares must cost br0 in the target weights, and every expected
  value must equal Σ q · P at its close. So an arithmetic slip in a case
  fails on its own.
- **Compounding.** Compounding the period returns, and the daily returns,
  gives back final / initial value, so no return is dropped or counted
  twice.
- **The two reports agree.** `statistical_report` computes every metric
  again, vectorized across paths. On the case's values as one path, it must
  match `performance_report`.

## Conventions

- **Fill.** `close`: the book is bought at the first bar's close, where it is
  worth br0. `open`: bought at the first bar's open, so the first bar's own
  move counts.
- **Labels.** A value is labelled with the moment it is true. Intraday
  values sit at their bar's close, and daily close-fill values keep their
  dates. Open-fill daily values are timezone-aware session times: the
  09:30 open, then each 16:00 close.
- **Adjusted closes.** When a case has `Adj Close`, it is the close used, and
  an open is scaled by its bar's `Adj Close / Close` ratio.
- **Gaps.** A missing price holds the last one, and the next price carries
  the whole move.
- **Daily returns for risk.** Volatility, downside volatility, Sharpe and
  Sortino use daily returns, annualized with 252: on intraday bars, each
  session's last value over the previous session's, the overnight gap
  included (the first session from the first value). On daily bars these are
  the period returns. A value's session is its date on the exchange's clock.
- **Metrics.** Total return and drawdown use every value. CAGR runs from the
  first value to the last and counts time in sessions on intraday bars
  (daily returns / 252) and in bars otherwise, so nights and weekends add
  none. It is −100% when the last value is 0 and NaN when it is below 0.
- **Total loss.** Once a value is 0, each later return is 0 / 0 and is left
  out: the risk figures use the returns up to and including the −100% one.
- **Years.** A value's year is its session date's year on the exchange's
  clock. The first year is full only if the series starts on or before its
  first session, the last only if it ends on or after its last session. Volatility is the sample
  standard deviation (n − 1). The downside deviation averages squared
  shortfalls below the per-period risk-free rate over **all** periods, and
  the risk-free rate is spread evenly, rf / 252 a day.
- **Tolerance.** Values match to $0.000001 and are written to 9 decimals when
  not round. Metrics match to a relative 1e-9 and are written to 12
  significant digits.

## Cases

| File | What it proves |
|---|---|
| `S01_close_fill` | br0 sits at the first close, the first period's return is kept, and opens are ignored. |
| `S02_drift` | Weights drift with prices and the book is never rebalanced. |
| `S03_open_fill_daily` | An open fill starts at the 09:30 open and counts the first day's move: one period per bar. |
| `S04_open_fill_intraday` | Values sit at bar closes, and the overnight gap lands in the next day's first bar; the risk figures use the two session returns. |
| `S05_adjusted_open` | The open is scaled onto the adjusted close's basis, so day 1 earns the traded move. |
| `S06_gap` | A missing price holds its mark, and the next price carries the move. |
| `S07a_late_listing_close` | A symbol with no first price is dropped, and the others are renormalized and named in a warning. |
| `S07b_late_listing_open` | The same with an open fill: dropped for having no first open. |
| `S08_half_day` | On a 13:00 half-day, the last bar's value is labelled 13:00. |
| `S09_dst` | Across spring-forward, labels stay on the New York clock, an hour earlier in UTC. |
| `S10_prices_stop_early` | A symbol whose prices stop early is held at its last price (cash at 0%), with a warning naming it. |
| `M01_core_metrics` | Returns, volatility, Sharpe, Sortino, drawdown and CAGR on one small series. |
| `M02_risk_free_rate` | The risk-free rate is spread per bar, and moves Sharpe and the downside target but not volatility. |
| `M03_year_boundary` | A year's return is measured from the previous year's last value. |
| `M04_intraday_daily_statistics` | On hourly bars, the risk figures come from daily returns, the overnight gap included, and CAGR counts time in sessions. |
| `M05_no_downside` | With no losing bar, the downside is 0, Sortino is NaN and the drawdown is 0. |
| `M06_sessions_on_exchange_clock` | Values labelled in Dubai time still group into New York sessions, not Dubai dates. |
| `M07_sessions_across_dst` | The same across the US clock change, with a Fri → Mon daily return over the weekend. |
| `M08_partial_years_on_sessions` | A year is partial unless the series covers its first and last XNYS sessions. |
| `M09_years_on_exchange_clock` | Values labelled in Dubai time go into the year of their New York session. |
| `M10_cagr_at_total_loss` | CAGR is −100% for a series that ends at zero. |
| `M11_returns_stop_at_total_loss` | After a total loss, the risk figures use the returns up to the −100% one; the 0 / 0 returns after it are left out. |
| `M12_cagr_below_zero` | CAGR is NaN for a series that ends below zero. |
