# portfolio-forecast

Monte Carlo simulation of the future value of a portfolio or trading
strategy. Give it a price or portfolio-value history and it simulates
thousands of possible future paths by three methods: resampling past returns,
drawing from a fitted distribution, or switching between fitted market
regimes. It then summarizes the spread of outcomes (returns, risk and tail
risk, with confidence intervals), draws them as fan charts, and typesets the
results as a PDF report. It also includes backtests for buy-and-hold and
scheduled-rebalancing portfolios, and a performance report for a single
historical series.

## Table of contents

- [Concepts](#concepts)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Standardized objects](#standardized-objects)
- [API reference](#api-reference)
  - [Simulation](#simulation)
  - [Reporting](#reporting)
  - [Metrics](#metrics)
  - [Plotting](#plotting)
  - [PDF reports](#pdf-reports)
  - [Trading dates and bar sizes](#trading-dates-and-bar-sizes)
  - [Backtesting](#backtesting)
  - [Internal helpers](#internal-helpers)
- [License](#license)

---

## Concepts

**Simulation methods.** Each simulator takes a history and returns an
[`sims` array](#sims-array) of future paths. One simulated period is one bar
of the input, so daily prices simulate days and hourly prices simulate hours.

| Method | Draws each period's return from | Keeps fat tails | Keeps volatility clustering | Selects the model by |
| --- | --- | --- | --- | --- |
| [`nonparametric_monte_carlo`](#nonparametric_monte_carlo) | past returns, resampled with replacement | yes | no | — |
| [`parametric_monte_carlo`](#parametric_monte_carlo) | a distribution fitted to past returns | if the fit has them (t, Johnson SU) | no | AIC over normal, t, Johnson SU |
| [`regime_switching_monte_carlo`](#regime_switching_monte_carlo) | the current regime of a hidden Markov model | with `bootstrap=True` | yes | BIC over 1–3 regimes |

**Annualization counts bars, not calendar days.** Every annualized figure
(CAGR, volatility, Sharpe, Sortino) converts bars to years with the
bars-per-year table [`PERIODS_PER_YEAR`](#periods_per_year). For example, a day is 1/252 of
a year and a 5-minute bar 1/19,656. Nights, weekends and holidays add no time.
The table assumes regular trading hours (6.5 hours a day), so intraday bars
that include extended hours are annualized with too few bars per year.

Over short horizons, annualization magnifies returns enormously: a 1% gain
in one day is a CAGR of about 1,100%. For horizons of days or hours, total
return is the more meaningful figure.

---

## Installation

Requires Python 3.10 or later.

```bash
pip install portfolio-forecast
```

Dependencies: numpy, pandas, scipy, matplotlib, hmmlearn, exchange_calendars
and python-dateutil. The package does not fetch data itself; the
[Quick start](#quick-start) downloads prices with yfinance
(`pip install yfinance`).

[`compile_statistical_reports`](#compile_statistical_reports) also needs a
LaTeX distribution with `pdflatex` on the PATH: TeX Live, MacTeX on macOS or
MiKTeX on Windows. Nothing else in the package uses LaTeX.

For local development, from the repo root:

```bash
pip install -e ".[notebook]"
```

The `notebook` extra adds `ipykernel` and `yfinance` for `demo.ipynb`, a worked
example on ten years of daily AAPL prices from Yahoo Finance: it runs all
three simulators, plots and compares their paths, prints their statistical
reports, and compiles a PDF report of the parametric and regime-switching
results (this last step needs LaTeX).

To run the tests (no network access needed; the PDF tests are skipped
without `pdflatex`):

```bash
pip install -e ".[test]"
pytest
```

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the code
and docstring conventions, and what a pull request needs to pass.

---

## Quick start

```python
import yfinance as yf

from portfolio_forecast.forecast import nonparametric_monte_carlo
from portfolio_forecast.performance import statistical_report
from portfolio_forecast.plotting import plot_simulated_paths
from portfolio_forecast.reporting import compile_statistical_reports
from portfolio_forecast.utils import next_trading_dates

# Ten years of daily closes: a date-indexed frame with one column. dropna
# removes a trailing row with no close, which Yahoo can return mid-session.
prices = yf.download("AAPL", period="10y", auto_adjust=True, progress=False)["Close"].dropna()

# 1,000 paths of the next 100 trading days, seeded for reproducibility
sims = nonparametric_monte_carlo(prices, sim_length=100, n_sims=1000, random_state=42)

# Dates for the plot: the last close followed by the next 100 trading days
dates = prices.index[-1:].append(next_trading_dates(prices.index, 100))

report = statistical_report(sims, interval="1 day", dates=dates, display=True)
fig, ax = plot_simulated_paths(sims, method="Non-parametric Monte Carlo", dates=dates)

# Typeset the results and chart as Reporting/AAPL_100_Day_Outlook.pdf (needs LaTeX)
compile_statistical_reports({
    "Title": "AAPL 100-Day Outlook",
    "Introduction": "1,000 paths resampled from ten years of daily AAPL returns.",
    "Reports": {
        "Non-parametric Monte Carlo": {
            "Description": "Bootstrap of historical daily returns.",
            "Results": report,
            "Figures": {"Paths": {"Image": fig, "Caption": "Simulated paths and their median."}},
        },
    },
})
```

---

## Project layout

The distribution is `portfolio-forecast`; it installs the `portfolio_forecast`
package.

| Path | Role |
| --- | --- |
| `src/portfolio_forecast/forecast/monte_carlo.py` | Simulators: [`nonparametric_monte_carlo`](#nonparametric_monte_carlo), [`parametric_monte_carlo`](#parametric_monte_carlo), [`regime_switching_monte_carlo`](#regime_switching_monte_carlo) |
| `src/portfolio_forecast/performance/report.py` | Reports: [`statistical_report`](#statistical_report), [`performance_report`](#performance_report); [metrics](#metrics) |
| `src/portfolio_forecast/performance/simulate.py` | Backtests: [`simulate_buy_and_hold`](#simulate_buy_and_hold), [`simulate_rebalancing_MVO`](#simulate_rebalancing_mvo), [`get_closing_prices`](#get_closing_prices) |
| `src/portfolio_forecast/plotting/paths.py` | Fan charts: [`plot_simulated_paths`](#plot_simulated_paths), [`plot_path_comparison`](#plot_path_comparison) |
| `src/portfolio_forecast/reporting/pdf.py` | PDF reports: [`compile_statistical_reports`](#compile_statistical_reports) |
| `src/portfolio_forecast/utils/trading_dates.py` | [`next_trading_dates`](#next_trading_dates) |
| `src/portfolio_forecast/utils/periods.py` | [`PERIODS_PER_YEAR`](#periods_per_year), the bars-per-year table |
| `tests/` | `pytest` suite covering every subpackage |
| `demo.ipynb` | Worked example: all three simulators, their plots and statistical reports, and a PDF report |
| `CONTRIBUTING.md` | How to set up, the code and docstring conventions, and what a pull request needs |

Each subpackage re-exports its public functions, e.g.
`from portfolio_forecast.performance import statistical_report`.
`get_closing_prices` is not re-exported; import it from
`portfolio_forecast.performance.simulate`.

---

## Standardized objects

### `values` input

What the simulators take as history.

| Property | Requirement |
| --- | --- |
| Type | `pandas.Series` or single-column `pandas.DataFrame` |
| Rows | One per bar, oldest first; the bar size is the simulated period |
| Values | Prices or portfolio values; positive for [`regime_switching_monte_carlo`](#regime_switching_monte_carlo) (log returns) |
| Missing values | None; drop them first (e.g. `.dropna()`), or a trailing row with no price starts the simulation from a date with no close |

### `sims` array

What the simulators return and the reports and plots take.

| Property | Value |
| --- | --- |
| Type | `numpy.ndarray` of shape `(n_sims, n_periods + 1)` |
| Rows | One simulated path each |
| Column 0 | Starting value, `1.0` |
| Values | Growth multiples of the last historical price (1.10 = up 10%) |

### `interval`

The bar size of one period, which the reports use to look up bars per year.

| Accepted form | Examples |
| --- | --- |
| [`PERIODS_PER_YEAR`](#periods_per_year) key | `'1 secs'` … `'30 secs'`, `'1 min'` … `'30 mins'`, `'1 hour'` … `'8 hours'`, `'1 day'`, `'1 week'`, `'1 month'` |
| pandas-style string | `'1D'`, `'1d'`, `'5min'`, `'1h'` |
| `pandas.Timedelta` | `pd.Timedelta(minutes=5)` |

The bar size must be one in `PERIODS_PER_YEAR`; others, such as `'7 mins'`,
raise `ValueError`.

### `dates` for simulated paths

Optional dates for a `sims` array, one per column: the last historical date
(the starting value) followed by the simulated periods. Build it with
[`next_trading_dates`](#next_trading_dates):

```python
dates = prices.index[-1:].append(next_trading_dates(prices.index, n_periods))
```

The plots use it to label the x-axis, and
[`statistical_report`](#statistical_report) to show the calendar span.

### Report dictionary

What [`compile_statistical_reports`](#compile_statistical_reports) takes.
Reports and figures appear in the PDF in dictionary order.

```python
{
    "Title": "Generic Title",
    "Introduction": "",
    "Reports": {
        "Report 1 Name": {
            "Description": "",
            "Results": results1,
            "Figures": {
                "Fig1": {"Image": fig1, "Caption": "", "Width": 1.0},
            },
        },
    },
}
```

| Key | Required | Type | Becomes |
| --- | --- | --- | --- |
| `"Title"` | yes | `str` | Document title, page header and default file name |
| `"Introduction"` | no | `str` | Text under the title |
| `"Reports"` | yes | `dict` | One numbered section per entry, titled by its key |
| `"Description"` | no | `str` | Text at the start of the section |
| `"Results"` | yes | `dict` | Tables: the dict returned by [`statistical_report`](#statistical_report) |
| `"Figures"` | no | `dict` | Numbered figures; the keys only name them in error messages |
| `"Image"` | yes, per figure | `matplotlib.figure.Figure` | The figure, embedded as vector graphics |
| `"Caption"` | no | `str` | Caption under the figure |
| `"Width"` | no | `float` | Figure width as a share of the text width, greater than 0 and at most 1; default `1.0` (full width). The figure is centered and keeps its aspect ratio |

All text is printed literally: LaTeX special characters (`% & $ # _ { } ~ ^ \`)
are escaped, and a blank line starts a new paragraph.

---

## API reference

### Simulation

`from portfolio_forecast.forecast import ...`

#### `nonparametric_monte_carlo`

```python
nonparametric_monte_carlo(values, sim_length=100, n_sims=1000, random_state=None, *, prices=None)
```

Simulate future paths by resampling historical returns with replacement (a
bootstrap) and compounding them. No distribution is assumed, so fat tails and
skew carry over; every draw is independent, so volatility clustering does not.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | Series or DataFrame | — | History; see [`values` input](#values-input). |
| `sim_length` | `int` | `100` | Future periods per path. |
| `n_sims` | `int` | `1000` | Number of paths. |
| `random_state` | `int`, `Generator` or `None` | `None` | Seed or Generator; an int or Generator gives reproducible paths, `None` a fresh unseeded Generator. |
| `prices` | Series or DataFrame | `None` | Deprecated alias for `values`, keyword only; warns, and is removed in 1.0.0. |

**Returns** — A [`sims` array](#sims-array) of shape `(n_sims, sim_length + 1)`.

**Raises** — `ValueError` if `values` has more than one column; `TypeError` if neither or both of `values` and `prices` are given.

**See also** — [`parametric_monte_carlo`](#parametric_monte_carlo),
[`regime_switching_monte_carlo`](#regime_switching_monte_carlo).

#### `parametric_monte_carlo`

```python
parametric_monte_carlo(values, distribution=None, sim_length=100, n_sims=1000, random_state=None,
                       return_fit=False, *, prices=None)
```

Simulate future paths from a `scipy.stats` distribution fitted to historical
returns by maximum likelihood. Each path compounds independent draws.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | Series or DataFrame | — | History; see [`values` input](#values-input). |
| `distribution` | `scipy.stats` distribution or `None` | `None` | Distribution to fit, e.g. `stats.t`, `stats.norm`, `stats.laplace`. `None` fits normal, t and Johnson SU and uses the lowest AIC. |
| `sim_length` | `int` | `100` | Future periods per path. |
| `n_sims` | `int` | `1000` | Number of paths. |
| `random_state` | `int`, `Generator` or `None` | `None` | As in [`nonparametric_monte_carlo`](#nonparametric_monte_carlo). |
| `return_fit` | `bool` | `False` | Also return the fitted distribution. |
| `prices` | Series or DataFrame | `None` | Deprecated alias for `values`, keyword only; warns, and is removed in 1.0.0. |

**Returns** — A [`sims` array](#sims-array) of shape `(n_sims, sim_length + 1)`.
With `return_fit=True`, a tuple `(sims, fit)`, where `fit` is a `dict`:

| Key | Contents |
| --- | --- |
| `"Distribution"` | scipy.stats name of the distribution used, e.g. `'johnsonsu'` |
| `"Parameters"` | Its fitted parameters by name: shape parameters, then `loc` and `scale` |
| `"AIC"` | Its AIC, `2k - 2 log L` |
| `"Candidates"` | Every candidate's AIC by name, lowest first, when `distribution=None`; otherwise `None` |

**Notes** — With `distribution=None`, prints each candidate's AIC
(`2k - 2 log L`) and the one selected. The fitted distribution is unbounded,
so a draw below −100% can send a path to zero or below;
[`statistical_report`](#statistical_report) leaves CAGR as NaN for such paths.

**See also** — [`nonparametric_monte_carlo`](#nonparametric_monte_carlo),
[`regime_switching_monte_carlo`](#regime_switching_monte_carlo).

#### `regime_switching_monte_carlo`

```python
regime_switching_monte_carlo(values, n_regimes=None, bootstrap=False, n_starts=10,
                             sim_length=100, n_sims=1000, random_state=None, return_fit=False,
                             *, prices=None)
```

Fit a Gaussian hidden Markov model to log returns, each hidden regime having
its own mean and volatility, and simulate paths that move between regimes by
the fitted transition matrix. Each path starts in a regime drawn from the
model's probabilities for the last historical bar. Keeps volatility
clustering.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | Series or DataFrame | — | Positive history; see [`values` input](#values-input). |
| `n_regimes` | `int` or `None` | `None` | Number of regimes. `None` fits 1, 2 and 3 and uses the lowest BIC. |
| `bootstrap` | `bool` | `False` | `False` draws from the regime's fitted normal; `True` resamples the historical returns assigned to that regime, keeping their fat tails. |
| `n_starts` | `int` | `10` | Random restarts per fit; EM finds only a local optimum, so the best of these is kept. |
| `sim_length` | `int` | `100` | Future periods per path. |
| `n_sims` | `int` | `1000` | Number of paths. |
| `random_state` | `int`, `Generator` or `None` | `None` | Seed or Generator for the restarts and the draws. |
| `return_fit` | `bool` | `False` | Also return the fitted model. |
| `prices` | Series or DataFrame | `None` | Deprecated alias for `values`, keyword only; warns, and is removed in 1.0.0. |

**Returns** — A [`sims` array](#sims-array) of shape `(n_sims, sim_length + 1)`.
With `return_fit=True`, a tuple `(sims, fit)`, where `fit` is a `dict`:

| Key | Contents |
| --- | --- |
| `"Regime Count"` | Number of regimes used |
| `"BIC"` | BIC by regime count: every count fitted when `n_regimes=None`, otherwise just `n_regimes` |
| `"Log Likelihood"` | Log-likelihood of the model used |
| `"Regimes"` | `DataFrame`, one row per regime from lowest to highest volatility: `Mean` and `Volatility` of the one-period log return, `Expected Duration` in periods (`1 / (1 - p_kk)`), and `Current Probability` of being in it at the last historical bar |
| `"Transition Matrix"` | `DataFrame` of one-period transition probabilities, row regime to column regime, same order |

**Raises** — `ValueError` if no restart gives a usable fit for `n_regimes`,
or, with `n_regimes=None`, if none of 1, 2 or 3 regimes can be fitted.

**Notes** — Each regime is fitted by maximum likelihood (EM with no prior on
the variances). A restart is rejected if any parameter is non-finite or a regime
holds less than `max(2, 1% of the history)` of the expected occupancy. BIC is
`-2 log L + p log n` with `p = (K-1) + K(K-1) + 2K` for K regimes; regime
counts that cannot be fitted are skipped. With `bootstrap=True`, a regime with
no historical returns falls back to its normal. Prints the BIC per regime count
(when selecting), then each regime's mean, volatility and expected duration
`1 / (1 - p_kk)`, from lowest to highest volatility. The model numbers regimes
arbitrarily; ordering them by volatility changes only how they are reported.

**See also** — [`nonparametric_monte_carlo`](#nonparametric_monte_carlo),
[`parametric_monte_carlo`](#parametric_monte_carlo).

### Reporting

`from portfolio_forecast.performance import ...`

#### `statistical_report`

```python
statistical_report(sims, interval, dates=None, risk_free_rate=0.0, confidence=0.95, display=False)
```

Compute every [`performance_report`](#performance_report) metric on each
simulated path, then summarize each by its mean, median and central
`confidence` interval across paths, and add tail risk of the total return.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `sims` | array-like | — | A [`sims` array](#sims-array). |
| `interval` | `str` or `Timedelta` | — | Bar size of one period; see [`interval`](#interval). |
| `dates` | array-like of datetimes | `None` | One per column of `sims`; see [`dates`](#dates-for-simulated-paths). Only used for the calendar span. |
| `risk_free_rate` | `float` | `0.0` | Annual risk-free rate for Sharpe, Sortino and downside deviation. |
| `confidence` | `float` | `0.95` | Interval coverage in (0, 1); 0.95 spans the 2.5th–97.5th percentiles and puts VaR and CVaR at the 5% tail. |
| `display` | `bool` | `False` | Also print a formatted report. |

**Returns** — `dict`:

| Key | Type | Contents |
| --- | --- | --- |
| `"Per Path Metrics"` | `DataFrame` | One row per path; columns Final Bankroll, Total Return, CAGR, Period Volatility, Annualized Volatility, Downside Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio |
| `"Period Returns"` | `ndarray` | Shape `(n_sims, n_periods)` |
| `"Paths"`, `"Periods"` | `int` | `n_sims`, `n_periods` |
| `"Total Days"` | `int` or `None` | Calendar days spanned by `dates`; `None` without `dates` |
| `"Total Years"` | `float` | `n_periods / periods_per_year` |
| `"Confidence"` | `float` | The `confidence` used |
| `"Returns"` | `DataFrame` | Rows Final Bankroll, Total Return, CAGR; columns Mean, Median, Lower, Upper |
| `"Risk"` | `DataFrame` | Rows Period Volatility, Annualized Volatility, Downside Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio; same columns |
| `"Tail Risk"` | `dict` | `"Probability of Loss"` (share of paths with a negative total return), `"Value at Risk"` (total return at the `1 - confidence` quantile), `"Conditional VaR"` (mean total return at or below it) |

**Raises** — `ValueError` if `interval` is unsupported, `confidence` is not in
(0, 1), `sims` has fewer than two periods, or `dates` has the wrong length.

**Notes** — A metric undefined on a path (CAGR for a path ending at or below
zero; Sharpe or Sortino with no variance or no downside) is NaN there and left
out of that metric's summary.

**See also** — [`performance_report`](#performance_report),
[`plot_simulated_paths`](#plot_simulated_paths).

#### `performance_report`

```python
performance_report(values, dates, interval, risk_free_rate=0.0, display=False)
```

Compute performance and risk metrics for one portfolio value series, such as a
backtest from [`simulate_buy_and_hold`](#simulate_buy_and_hold).

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | `Series` | — | Portfolio values, one per bar, oldest first. |
| `dates` | `DatetimeIndex` or array-like | — | The date of each value; used for calendar-year returns and the displayed calendar span. |
| `interval` | `str` or `Timedelta` | — | Bar size of `values`; see [`interval`](#interval). |
| `risk_free_rate` | `float` | `0.0` | Annual risk-free rate for Sharpe, Sortino and downside deviation. |
| `display` | `bool` | `False` | Also print a formatted report. |

**Returns** — `dict`:

| Key | Contents |
| --- | --- |
| `"Period Returns"` | `Series` of one-period returns |
| `"Periods"` | Number of bars, `len(values) - 1` |
| `"Total Days"` | Calendar days from first to last date |
| `"Total Years"` | `Periods / periods_per_year` |
| `"Returns"` | `dict`: Initial Bankroll, Final Bankroll, Total Return, CAGR, Actual Yearly Returns (from [`calculate_actual_yearly_return`](#calculate_actual_yearly_return)) |
| `"Risk"` | `dict`: Period Volatility (per bar), Annualized Volatility, Downside Volatility (annualized), Max Drawdown, Sharpe Ratio, Sortino Ratio |

**Raises** — `ValueError` if `interval` is unsupported.

**See also** — [`statistical_report`](#statistical_report).

### Metrics

`from portfolio_forecast.performance import ...` — the building blocks of
[`performance_report`](#performance_report). `periods_per_year` is bars in a
year at the series' interval, e.g. `PERIODS_PER_YEAR['1 day']` (252).

#### `calculate_daily_return`

```python
calculate_daily_return(values)
```

One-period simple returns, `values[t] / values[t-1] - 1`, with the first
(undefined) return dropped. Despite the name, the period is whatever bar
`values` is sampled on.

**Parameters** — `values` (`Series`) portfolio values.

**Returns** — `Series`, one shorter than `values`.

#### `calculate_cagr`

```python
calculate_cagr(values, periods_per_year)
```

Compound annual growth rate: `(values[-1] / values[0]) ** (1 / years) - 1`
with `years = (len(values) - 1) / periods_per_year`, counting time in bars.

**Parameters** — `values` (`Series`) portfolio values; `periods_per_year`
(`float`) bars per year.

**Returns** — `float`.

**Raises** — `ZeroDivisionError` if `values` has one entry.

#### `calculate_sharpe_ratio`

```python
calculate_sharpe_ratio(period_returns, periods_per_year, risk_free_rate=0.0)
```

Annualized Sharpe ratio: `mean(r - rf / periods_per_year) / std(r) *
sqrt(periods_per_year)`, with the sample standard deviation.

**Parameters** — `period_returns` (`Series`) one-period returns;
`periods_per_year` (`float`) bars per year; `risk_free_rate` (`float`,
default `0.0`) annual rate, spread evenly across the year's bars.

**Returns** — `float`; infinite or NaN if the returns have zero variance.

#### `calculate_sortino_ratio`

```python
calculate_sortino_ratio(period_returns, periods_per_year, risk_free_rate=0.0)
```

Annualized Sortino ratio: the Sharpe ratio with downside deviation,
`sqrt(sum(min(excess, 0) ** 2) / n)` over all n periods, in place of the
standard deviation.

**Parameters** — as [`calculate_sharpe_ratio`](#calculate_sharpe_ratio); the
risk-free rate is also the downside target.

**Returns** — `float`; NaN if no return falls below the risk-free rate.

#### `calculate_max_drawdown`

```python
calculate_max_drawdown(values)
```

Largest peak-to-trough decline: `min(values / cummax(values) - 1)`.

**Parameters** — `values` (`Series`) portfolio values.

**Returns** — `float` in [−1, 0]; −0.25 is a 25% drawdown.

#### `calculate_actual_yearly_return`

```python
calculate_actual_yearly_return(values, dates)
```

Return earned in each calendar year. The first year is measured from its
first value; later years from the previous year's last value, so the move
across each year boundary is counted.

**Parameters** — `values` (`Series`) portfolio values; `dates` (array-like)
the date of each value.

**Returns** — `DataFrame` indexed by year, with columns `return`; `days`
(365 or 366 for a full year, else the calendar days from its first to last
bar, inclusive); and `is_full` (False for a first year starting after
January 5 or a last year ending before December 25).

### Plotting

`from portfolio_forecast.plotting import ...`

#### `plot_simulated_paths`

```python
plot_simulated_paths(sims, method='Monte Carlo', dates=None, figsize=(12, 6.5), dpi=120)
```

Draw a fan chart: every path, colored by its terminal value (viridis), with
the median path and the starting value marked.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `sims` | array-like | — | A [`sims` array](#sims-array). |
| `method` | `str` | `'Monte Carlo'` | Simulation method, shown in the title. |
| `dates` | array-like of datetimes | `None` | X-axis labels; see [`dates`](#dates-for-simulated-paths). `None` labels periods 0 to n. |
| `figsize` | `tuple` | `(12, 6.5)` | Figure size in inches. |
| `dpi` | `int` | `120` | Figure resolution. |

**Returns** — `(fig, ax)`: the matplotlib Figure and Axes.

**Raises** — `ValueError` if `dates` does not have one entry per column of
`sims`.

**Notes** — Periods stay evenly spaced even with `dates`, so weekends,
holidays and overnight gaps do not stretch the paths.

**See also** — [`plot_path_comparison`](#plot_path_comparison).

#### `plot_path_comparison`

```python
plot_path_comparison(sims_by_method, dates=None, figsize=None, dpi=120)
```

Draw several methods' fan charts side by side, on a shared y-axis and one
color scale, so heights and colors compare directly across panels.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `sims_by_method` | `dict` | — | Method name → [`sims` array](#sims-array), in panel order. |
| `dates` | array-like of datetimes | `None` | X-axis labels shared by every panel. |
| `figsize` | `tuple` | `None` | Figure size; defaults to `(6.5 * n_methods, 6)`. |
| `dpi` | `int` | `120` | Figure resolution. |

**Returns** — `(fig, axes)`: the Figure and an array of Axes, one per method.

**See also** — [`plot_simulated_paths`](#plot_simulated_paths).

### PDF reports

`from portfolio_forecast.reporting import compile_statistical_reports`

#### `compile_statistical_reports`

```python
compile_statistical_reports(spec, output_dir='Reporting', filename=None, engine='pdflatex',
                            table_of_contents=False)
```

Typeset a [report dictionary](#report-dictionary) as a PDF with LaTeX: the
title, date and introduction, an optional table of contents, then a section
per report with its description, its results as tables
(simulation setup, returns, risk, tail risk) and its figures. LaTeX runs in a
temporary directory, so the PDF is the only file written.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `spec` | `dict` | — | A [report dictionary](#report-dictionary). |
| `output_dir` | `str` or path | `'Reporting'` | Directory for the PDF, created if missing; relative to the working directory, normally the project folder. |
| `filename` | `str` | `None` | PDF name; defaults to the title with non-alphanumeric runs replaced by `_` (`Generic_Title.pdf`). `.pdf` is added if missing. An existing file of that name is replaced. |
| `engine` | `str` | `'pdflatex'` | LaTeX engine; `xelatex` and `lualatex` also work. |
| `table_of_contents` | `bool` | `False` | List the reports with page numbers after the introduction, each linked to its section. |

**Returns** — `pathlib.Path`, the absolute path of the PDF.

**Raises** — `TypeError` if `spec` is not a dict, an `"Image"` is not a
matplotlib Figure, or a `"Width"` is not a number. `ValueError` if `"Title"`
or `"Reports"` is missing or empty, a `"Results"` is not from
`statistical_report`, or a `"Width"` is not in (0, 1]. `RuntimeError` if
the engine is not installed, or LaTeX fails (the end of its log is included);
nothing is written in either case.

**Notes** — Needs the booktabs, caption, fancyhdr, float, geometry, hyperref,
lmodern and microtype LaTeX packages, all part of TeX Live, MacTeX and MiKTeX.
Figures are embedded as vector PDF and are not modified or closed; a figure
is scaled to its `"Width"` share of the text width, so a wide one (such as a
three-panel [`plot_path_comparison`](#plot_path_comparison)) gets small labels,
and a narrower `"Width"` makes them smaller still. With
`pdflatex`, text must use characters it can typeset (Latin scripts, common
symbols); an emoji, for example, makes LaTeX fail.

**See also** — [`statistical_report`](#statistical_report),
[`plot_simulated_paths`](#plot_simulated_paths).

### Trading dates and bar sizes

`from portfolio_forecast.utils import next_trading_dates, PERIODS_PER_YEAR`

#### `next_trading_dates`

```python
next_trading_dates(dates, n_periods, interval=None, calendar='XNYS', tz=None)
```

Return the future trading dates or bar times that continue a series of bars,
skipping weekends, exchange holidays and closed hours. Used to build
[`dates` for simulated paths](#dates-for-simulated-paths).

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dates` | array-like of datetimes | — | Historical bar dates or times, e.g. `prices.index`. |
| `n_periods` | `int` | — | Number of future periods; zero or less returns an empty index. |
| `interval` | `str` or `Timedelta` | `None` | Bar size as a bar-size string (`'5 mins'`), pandas string (`'5min'`) or `Timedelta`; `None` infers it from `dates`. |
| `calendar` | `str` | `'XNYS'` | `exchange_calendars` code; XNYS is the NYSE. |
| `tz` | `str` or `tzinfo` | `None` | Timezone of naive intraday timestamps; defaults to the machine's local timezone. Ignored for daily and longer bars. |

**Returns** — `DatetimeIndex` of `n_periods` dates (daily and longer) or bar
start times (intraday), naive or timezone-aware like the input.

**Raises** — `ValueError` if `dates` is empty, `interval` cannot be parsed, or
`interval` is `None` with fewer than two dates.

**Notes** — n-day bars count trading sessions. Weekly and monthly bars keep
the input's convention of stamping each bar with the first or last session of
its period. Intraday bars follow the session layout learned from the input:
grid phase, a bar at an off-grid open such as 9:30, and any pre- or
post-market bars. Early closes are respected.

#### `PERIODS_PER_YEAR`

```python
PERIODS_PER_YEAR = {'1 secs': 5896800, ..., '1 day': 252, '1 week': 52, '1 month': 12}
```

Bars per year for each supported bar size, used by the reports to annualize.
Intraday entries assume 252 regular-hours sessions of 6.5 hours, counting a
bar for a leftover stub: `'1 hour'` is 7 bars a day (a 30-minute stub at the
9:30 open plus six full hours), so 1,764 a year.

**Keys** — `'1 secs'`, `'5 secs'`, `'10 secs'`, `'15 secs'`, `'30 secs'`,
`'1 min'`, `'2 mins'`, `'3 mins'`, `'5 mins'`, `'10 mins'`, `'15 mins'`,
`'20 mins'`, `'30 mins'`, `'1 hour'`, `'2 hours'`, `'3 hours'`, `'4 hours'`,
`'8 hours'`, `'1 day'`, `'1 week'`, `'1 month'`. The reports also accept other
spellings of these sizes; see [`interval`](#interval).

### Backtesting

`from portfolio_forecast.performance import ...`

#### `simulate_buy_and_hold`

```python
simulate_buy_and_hold(data, w, br0=1)
```

Backtest a fixed-weight portfolio bought at the first close and never
rebalanced, so its weights drift with prices.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `DataFrame` | — | Historical prices in any layout [`get_closing_prices`](#get_closing_prices) accepts. |
| `w` | `Series` or array-like | — | Weights: a Series aligned by symbol (missing symbols get zero), or an array matched by column position. Normalized to sum to 1. |
| `br0` | `float` | `1` | Starting portfolio value. |

**Returns** — `(values, dates)`: a `Series` of portfolio values starting at
`br0` on the purchase date, and its index.

**Raises** — `ValueError` if the weights sum to zero or less, or an array of
weights does not have one entry per symbol.

**Notes** — Symbols with any missing return are dropped, then bars with any
missing return.

**See also** — [`performance_report`](#performance_report).

#### `simulate_rebalancing_MVO`

```python
simulate_rebalancing_MVO(data, w, beta=0, br0=1)
```

Backtest a book re-weighted on a schedule, trading only when the suggested
book differs enough from the drifted book actually held.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `DataFrame` | — | Historical prices in any layout [`get_closing_prices`](#get_closing_prices) accepts. |
| `w` | `DataFrame` | — | Suggested weights, one row per symbol and one column per rebalance timestamp (e.g. from a mean-variance optimizer, not included). A column must be all weights or all NaN (no suggestion); every label must be in the price index. |
| `beta` | `float` | `0` | No-trade band in [0, 1]: trade only if one-way turnover exceeds it. `0` trades at every suggestion; `1` never trades after the first. |
| `br0` | `float` | `1` | Starting portfolio value. |

**Returns** — `(values, dates, decisions)`: portfolio value at every bar
(flat at `br0` until the first trade), its index, and a `Series` over the
rebalance timestamps reading `'traded'`, `'held'` (inside the band) or
`'skipped'` (no suggestion).

**Raises** — `TypeError` if `w` is not a DataFrame. `ValueError` if `beta` is
outside [0, 1], a column is partly NaN, a timestamp is not in the price index,
or timestamps are not strictly increasing.

**Notes** — One-way turnover is `0.5 * (sum(|w_new - w_drifted|) +
|cash_new - cash_drifted|)`. A symbol with no quote on the fill bar is dropped
and the book renormalized. Missing quotes while held carry the last price
forward. No transaction costs.

**See also** — [`simulate_buy_and_hold`](#simulate_buy_and_hold).

#### `get_closing_prices`

```python
get_closing_prices(data)
```

Select closing prices as a (date × symbol) frame. Accepts MultiIndex columns
with a `close` field on either level (e.g. from `yfinance.download`), flat
OHLCV columns for one symbol, or a frame that is already closes. Import from
`portfolio_forecast.performance.simulate`.

**Parameters** — `data` (`DataFrame`) prices.

**Returns** — `DataFrame`, one column per symbol; returned unchanged if it has
no `close` column.

**Raises** — `KeyError` if MultiIndex columns have no `close` field.

### Internal helpers

Private functions, listed for completeness.

| Function | Module | Role |
| --- | --- | --- |
| `_resolve_values(values, prices, func_name)` | `forecast/monte_carlo.py` | Accept the deprecated `prices` keyword in place of `values`, with a warning |
| `_periods_per_year(interval)` | `performance/report.py` | Look up bars per year for any accepted [`interval`](#interval) spelling |
| `_interval_summary(values, confidence)` | `performance/report.py` | Mean, median and interval of one metric across paths, ignoring NaN |
| `_hold(closing, start_i, end_i, held, w_held)` | `performance/simulate.py` | Growth and drifted weights of a book held between two bars |
| `_buyable(w_new, quotes)` | `performance/simulate.py` | The book a suggestion can actually buy, renormalized |
| `_turnover(a, b)` | `performance/simulate.py` | One-way turnover between two books, counting cash |
| `_label_dates`, `_draw_paths`, `_add_colorbar` | `plotting/paths.py` | Axis labels, one fan of paths, and the colorbar |
| `_escape`, `_number`, `_slug`, `_interval_table`, `_results_section`, `_validate` | `reporting/pdf.py` | Escape text for LaTeX, format table values, name the file, build the tables, and check the report dictionary |
| `_parse_interval`, `_infer_interval`, `_next_sessions`, `_next_periodic`, `_next_intraday` | `utils/trading_dates.py` | Parse or infer a bar size, then step forward by sessions, weeks/months or intraday bars |

---

## License

MIT; see [LICENSE](LICENSE).
