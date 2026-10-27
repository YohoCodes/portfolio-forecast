# portfolio-forecast

Monte Carlo simulation of the future value of a portfolio or trading
strategy. Give it a price or portfolio-value history and it simulates
thousands of possible future paths by three methods: resampling past returns,
drawing from a fitted distribution, or switching between fitted market
regimes. It then summarizes the spread of outcomes (returns, risk and tail
risk, each with its 95% range across paths), draws them as fan charts, and
typesets the results as a PDF report. It also includes a buy-and-hold
backtest and a performance report for a single historical series, on daily
or intraday bars, labelled on the exchange calendar.

## Table of contents

- [Concepts](#concepts)
- [Installation](#installation)
  - [Development](#development)
- [Quick start](#quick-start)
  - [Intraday history](#intraday-history)
- [Demos](#demos)
  - [Installing the demos](#installing-the-demos)
- [Project layout](#project-layout)
- [Standardized objects](#standardized-objects)
  - [values input](#values-input)
  - [BacktestResult](#backtestresult)
  - [SimulationResult](#simulationresult)
  - [FutureLayout](#futurelayout)
  - [Return classes](#return-classes)
  - [interval](#interval)
  - [dates for simulated paths](#dates-for-simulated-paths)
  - [Report dictionary](#report-dictionary)
- [API reference](#api-reference)
  - [Simulation](#simulation): [nonparametric_monte_carlo](#nonparametric_monte_carlo) · [parametric_monte_carlo](#parametric_monte_carlo) · [regime_switching_monte_carlo](#regime_switching_monte_carlo)
  - [Reporting](#reporting): [statistical_report](#statistical_report) · [performance_report](#performance_report)
  - [Returns and metrics](#returns-and-metrics): [calculate_period_returns](#calculate_period_returns) · [calculate_session_returns](#calculate_session_returns) · [calculate_daily_return](#calculate_daily_return) · [calculate_cagr](#calculate_cagr) · [calculate_sharpe_ratio](#calculate_sharpe_ratio) · [calculate_sortino_ratio](#calculate_sortino_ratio) · [calculate_max_drawdown](#calculate_max_drawdown) · [calculate_actual_yearly_return](#calculate_actual_yearly_return)
  - [Plotting](#plotting): [plot_simulated_paths](#plot_simulated_paths) · [plot_path_comparison](#plot_path_comparison)
  - [PDF reports](#pdf-reports): [compile_statistical_reports](#compile_statistical_reports)
  - [Trading dates, marks and bar sizes](#trading-dates-marks-and-bar-sizes): [next_trading_dates](#next_trading_dates) · [classify_returns](#classify_returns) · [to_session_values](#to_session_values) · [close_times](#close_times) · [PERIODS_PER_YEAR](#periods_per_year)
  - [Backtesting](#backtesting): [simulate_buy_and_hold](#simulate_buy_and_hold) · [get_closing_prices](#get_closing_prices)
  - [Module-level functions](#module-level-functions)
  - [Internal helpers](#internal-helpers)
- [How the simulators work](#how-the-simulators-work)
  - [Notation](#notation)
  - [What all three share](#what-all-three-share)
  - [Bootstrap (nonparametric_monte_carlo)](#bootstrap-nonparametric_monte_carlo)
  - [Parametric (parametric_monte_carlo)](#parametric-parametric_monte_carlo)
  - [Regime-switching (regime_switching_monte_carlo)](#regime-switching-regime_switching_monte_carlo)
- [Tuning the simulators](#tuning-the-simulators)
  - [gap_model (regime-switching)](#gap_model-regime-switching)
  - [kappa (regime-switching)](#kappa-regime-switching)
  - [criterion (both simulators)](#criterion-both-simulators)
  - [min_sample (all three simulators)](#min_sample-all-three-simulators)
  - [distribution (parametric)](#distribution-parametric)
- [Logic](#logic)
  - [Backtest](#backtest): [B1.1](#b11-point-in-time) · [B1.2](#b12-first-bar) · [B1.3](#b13-later-bars) · [B1.4](#b14-gap) · [B1.5](#b15-no-opens) · [B1.6](#b16-missing-first-bar) · [B2](#b2-purchase) · [B3](#b3-labels) · [B4.1](#b41-stale-prices) · [B4.2](#b42-no-price-at-the-fill) · [B4.3](#b43-prices-stop-early) · [B4.4](#b44-bars-outside-the-session) · [B5](#b5-adjusted-prices) · [B6](#b6-completed-bars-only)
  - [Return classes and units](#return-classes-and-units): [U1](#u1-classes) · [U2](#u2-units) · [U3](#u3-continuous-mode)
  - [Metrics](#metrics): [M1](#m1-return-series-for-risk-figures) · [M2](#m2-first-session) · [M3](#m3-risk-free-rate) · [M4](#m4-time-and-cagr) · [M5](#m5-per-class-volatility) · [M6](#m6-max-drawdown) · [M7](#m7-sortino) · [M8](#m8-yearly-returns) · [M9](#m9-total-loss) · [M10](#m10-return-helpers) · [M11](#m11-lowerupper-wording) · [M12](#m12-degenerate-series)
  - [Monte Carlo](#monte-carlo): [MC1](#mc1-input) · [MC2](#mc2-horizon-and-future-layout) · [MC3](#mc3-bootstrap) · [MC4](#mc4-fitting-scale) · [MC5](#mc5-parametric-family-selection) · [MC6](#mc6-minimum-sample) · [MC7](#mc7-regime-switching-hmm) · [MC8](#mc8-future-marks)
- [License](#license)

---

## Concepts

**Marks.** A value series is a list of *marks*: a moment and the portfolio's
value then. Daily marks are the purchase, then each session's close.
Intraday marks are each session's open, then each bar's close, so the
overnight move from one close to the next open is a return of its own.
[`simulate_buy_and_hold`](#simulate_buy_and_hold) builds these marks and
labels each with the moment it is true, on the exchange's clock.

**Return classes.** Every return between two marks has a
[class](#return-classes): a *gap* (last close to the next open), a *regular*
bar, or *irregular* (a stale price or a missing bar). Gaps and bars are not
alike (a night is not an hour), so the simulators draw each future mark from
its own class, and leave irregular returns out.

**Simulation methods.** Each simulator takes a history and returns a
[`SimulationResult`](#simulationresult): the simulated paths and the dates of
the future marks they run over.

| Method | Draws each mark's log return from | Keeps fat tails | Keeps volatility clustering | Selects the model by |
| --- | --- | --- | --- | --- |
| [`nonparametric_monte_carlo`](#nonparametric_monte_carlo) | past returns of its class, resampled with replacement | as heavy as the history's own, never beyond its largest move | no | — |
| [`parametric_monte_carlo`](#parametric_monte_carlo) | a distribution fitted to its class | if the fit has them (t, Johnson SU) | no | BIC (or AIC) over normal, t, Johnson SU |
| [`regime_switching_monte_carlo`](#regime_switching_monte_carlo) | the current regime of a hidden Markov model | with `bootstrap=True`, as heavy as the history's own | yes | BIC (or AIC) over 1–3 regimes |

**Daily or intraday is decided on the calendar.** A history with at most one
mark per session, at its close, is daily; more marks per session make it
intraday. On daily history the horizon `sim_length` counts sessions, which
are its bars. On intraday history it also counts sessions, each simulated as
its open mark and its bar closes, so a 5-session hourly forecast has 40
marks. To forecast days from intraday history, collapse it first with
[`to_session_values`](#to_session_values).

**Annualization counts sessions.** Volatility, Sharpe and Sortino are
computed on session returns (one close to the next), annualized with 252,
whatever the bar size; weekly and monthly bars use their own returns with 52
and 12. CAGR counts the sessions after the first value over 252 by default,
or calendar days over 365.25 with `cagr_clock="calendar"`. Over short
horizons, annualization magnifies returns enormously: a 1% gain in one
session is a CAGR of about 1,100%. For horizons of days or hours, total
return is the more meaningful figure.

**Calendars.** Every function that reads dates takes `calendar`, an
[`exchange_calendars`](https://github.com/gerrymanoim/exchange_calendars)
code; the default `"XNYS"` is the NYSE, with its holidays and half-days.
Weekday-only data off an exchange (fund NAVs, indices, FX) takes `"24/5"`.
`calendar=None` is *continuous mode* for 24/7 assets such as crypto: a
constant bar grid, one class, no sessions or gaps, risk figures on
calendar-day returns at 00:00 UTC annualized with 365.

**Which bars a backtest values.** Equities on an exchange calendar (the
default `"XNYS"`) are valued at the calendar's session bars. An intraday bar
outside the session, pre- or post-market, is dropped with a warning, and its
move appears in the next gap ([bars outside the session](#b44-bars-outside-the-session)).
Calendars cover regular hours only; extended-hours sessions are not a
calendar option. A 24/7 asset such as crypto, backtested on its own, takes
`calendar=None` (continuous mode), and every bar is kept. A portfolio mixing
stocks and a 24/7 asset on an exchange calendar is valued at session bars, so
the 24/7 asset's weekend move appears in Monday's gap. A daily bar dated on a
non-session day raises `ValueError`, so drop a mixed portfolio's weekend rows
before a daily backtest.

---

## Installation

Requires Python 3.10 or later.

```bash
pip install portfolio-forecast
```

Dependencies: numpy, pandas, scipy, matplotlib, exchange_calendars and
python-dateutil. The package does not fetch data itself; the
[Quick start](#quick-start) downloads prices with yfinance
(`pip install yfinance`).

[`compile_statistical_reports`](#compile_statistical_reports) also needs a
LaTeX distribution with `pdflatex` on the PATH: TeX Live, MacTeX on macOS or
MiKTeX on Windows. Nothing else in the package uses LaTeX.

To run the demo notebooks, see [Demos](#demos).

### Development

For contributors working from a clone of the repository.
`pip install -e ".[test]"` installs the package in editable mode with pytest
and `hmmlearn`, which one test uses as a reference for the package's
regime-switching fit. From the repo root, run the tests (no network access
needed; the PDF tests are skipped without `pdflatex`):

```bash
pip install -e ".[test]"
pytest
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers the full setup, the code and
docstring conventions, and what a pull request needs to pass.

---

## Quick start

Daily closes in, a forecast, its report, its chart and a PDF out:

```python
import yfinance as yf

from portfolio_forecast.forecast import nonparametric_monte_carlo
from portfolio_forecast.performance import statistical_report
from portfolio_forecast.plotting import plot_simulated_paths
from portfolio_forecast.reporting import compile_statistical_reports
from portfolio_forecast.utils.calendar import completed_mask

# Ten years of daily closes: a date-indexed frame with one column. dropna
# removes rows with no close (the simulators raise on missing values rather
# than guess), and completed_mask drops today's row while its session is
# still trading, since its "close" is only the latest price.
prices = yf.download("AAPL", period="10y", auto_adjust=True, progress=False)["Close"].dropna()
prices = prices[completed_mask(prices.index)]

# 1,000 paths of the next 100 trading days, seeded for reproducibility.
# sims.paths holds the paths, sims.dates the 100 future sessions.
sims = nonparametric_monte_carlo(prices, sim_length=100, n_sims=1000, random_state=42)

# Both read the dates from sims: no interval or dates to pass
report = statistical_report(sims, display=True)
fig, ax = plot_simulated_paths(sims, method="Non-parametric Monte Carlo")

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

`sims.paths` has shape `(1000, 101)`: column 0 is the last close, as 1.0, so
every value is a multiple of it (1.10 = up 10%).

### Intraday history

Intraday bars need opens, since each session's open is a mark of its own.
Backtest the bars, then forecast from the backtest, which carries each
return's class:

```python
import yfinance as yf

from portfolio_forecast.forecast import nonparametric_monte_carlo
from portfolio_forecast.performance import performance_report, simulate_buy_and_hold, statistical_report
from portfolio_forecast.utils import to_session_values

# 60 days of hourly bars with opens, each labelled with its start time
data = yf.download("AAPL", period="60d", interval="1h", auto_adjust=True, progress=False)

# A value at each session's open and each bar's close, starting at 1.0
backtest = simulate_buy_and_hold(data, {"AAPL": 1.0})
history = performance_report(backtest, display=True)

# On intraday history sim_length counts sessions (with a warning saying so):
# the next 5 sessions, each an open mark and its hourly closes
sims = nonparametric_monte_carlo(backtest, sim_length=5, n_sims=1000, random_state=42)
report = statistical_report(sims)

# For a daily forecast from the same history, collapse it to session closes
daily_sims = nonparametric_monte_carlo(to_session_values(backtest), sim_length=5, n_sims=1000,
                                       random_state=42)
```

---

## Demos

Three Jupyter notebooks work through complete examples. They live in their
own repository,
[portfolio-forecast-demos](https://github.com/YohoCodes/portfolio-forecast-demos),
and run against the release on PyPI. The first two download recent prices
from Yahoo Finance, over a window that ends today, so their numbers change
from run to run; the third generates its own synthetic history.

| Notebook | What it does |
| --- | --- |
| `01_quick_start_daily.ipynb` | Quick start on daily prices: backtests a four-stock portfolio with [`simulate_buy_and_hold`](#simulate_buy_and_hold), forecasts it with all three simulators, compares their fan charts and statistical reports, and compiles a PDF report with [`compile_statistical_reports`](#compile_statistical_reports) |
| `02_intraday_sessions_and_gaps.ipynb` | Intraday sessions and gaps on hourly bars: the marks of a session, [return classes](#return-classes) and their warnings, gap and bar volatility, future marks across a weekend and a half-day, a forecast counted in sessions, [`to_session_values`](#to_session_values), and continuous mode for a 24/7 asset |
| `03_choosing_a_simulator.ipynb` | Choosing a simulator, on a synthetic two-regime history whose true model is known, so each method can be checked against the truth |

To read them without installing anything, open them on
[GitHub](https://github.com/YohoCodes/portfolio-forecast-demos), which shows
the saved outputs.

### Installing the demos

1. Clone the demos repository and create a virtual environment:

   ```bash
   git clone https://github.com/YohoCodes/portfolio-forecast-demos.git
   cd portfolio-forecast-demos
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. Install the requirements: this package from PyPI with the `notebook` extra,
   which adds `yfinance` to download prices and `ipykernel` to run the
   notebooks, plus JupyterLab:

   ```bash
   pip install -r requirements.txt
   ```

3. Open a demo:

   ```bash
   jupyter lab 01_quick_start_daily.ipynb
   ```

   In VS Code, open the notebook and select the `venv` Python as its kernel.

4. Run all cells. The first two demos need internet access to download
   prices, and the last step of `01_quick_start_daily.ipynb` needs LaTeX (see
   [Installation](#installation)); without it, that cell says so and skips the
   PDF. The PDF is saved to `Reporting/`.

---

## Project layout

The distribution is `portfolio-forecast`; it installs the `portfolio_forecast`
package.

| Path | Role |
| --- | --- |
| `src/portfolio_forecast/types.py` | Result types: [`BacktestResult`](#backtestresult), [`SimulationResult`](#simulationresult), [`FutureLayout`](#futurelayout) |
| `src/portfolio_forecast/forecast/bootstrap.py` | [`nonparametric_monte_carlo`](#nonparametric_monte_carlo) |
| `src/portfolio_forecast/forecast/parametric.py` | [`parametric_monte_carlo`](#parametric_monte_carlo) |
| `src/portfolio_forecast/forecast/regime.py` | [`regime_switching_monte_carlo`](#regime_switching_monte_carlo) |
| `src/portfolio_forecast/forecast/_common.py`, `_hmm.py` | The simulators' shared path, and the hidden Markov model's EM fit (private) |
| `src/portfolio_forecast/performance/simulate.py` | Backtests: [`simulate_buy_and_hold`](#simulate_buy_and_hold), [`get_closing_prices`](#get_closing_prices) |
| `src/portfolio_forecast/performance/returns.py` | [Return series](#returns-and-metrics): per mark and per session |
| `src/portfolio_forecast/performance/metrics.py` | [Metrics](#returns-and-metrics): CAGR, volatility, Sharpe, Sortino, drawdown, yearly returns |
| `src/portfolio_forecast/performance/report.py` | Reports: [`statistical_report`](#statistical_report), [`performance_report`](#performance_report) |
| `src/portfolio_forecast/plotting/paths.py` | Fan charts: [`plot_simulated_paths`](#plot_simulated_paths), [`plot_path_comparison`](#plot_path_comparison) |
| `src/portfolio_forecast/reporting/pdf.py` | PDF reports: [`compile_statistical_reports`](#compile_statistical_reports) |
| `src/portfolio_forecast/utils/calendar.py` | Sessions on the exchange calendar: [`close_times`](#close_times), [`session_spans`](#module-level-functions), [`completed_mask`](#module-level-functions) |
| `src/portfolio_forecast/utils/marks.py` | Marks and classes: [`classify_returns`](#classify_returns), [`to_session_values`](#to_session_values), the future layout |
| `src/portfolio_forecast/utils/trading_dates.py` | [`next_trading_dates`](#next_trading_dates) |
| `src/portfolio_forecast/utils/periods.py` | [`PERIODS_PER_YEAR`](#periods_per_year) and parsing of every [`interval`](#interval) spelling |
| `src/portfolio_forecast/utils/bar_times.py` | Old location of `close_times`, kept so 0.3.0 imports work |
| `tests/` | `pytest` suite covering every subpackage; `tests/golden/` holds hand-derived cases for each [Logic](#logic) rule |
| `CONTRIBUTING.md` | How to set up, the code and docstring conventions, and what a pull request needs |
| `CHANGELOG.md` | Notable changes in each release, with a migration list for 0.4.0 |

Each subpackage re-exports its public functions, e.g.
`from portfolio_forecast.performance import statistical_report`, and the
result types are importable from the top level,
`from portfolio_forecast import SimulationResult`. `get_closing_prices` and
the [module-level functions](#module-level-functions) are not re-exported;
import them from their modules.

---

## Standardized objects

### `values` input

What the simulators, the reports and the date functions take as history.

| Property | Requirement |
| --- | --- |
| Type | `pandas.Series` on a `DatetimeIndex`, a single-column `DataFrame`, or a [`BacktestResult`](#backtestresult). For the simulators, a NumPy array, a list or a Series on a `RangeIndex` is a *plain array*: one class and no dates. |
| Rows | One per [mark](#concepts), oldest first |
| Values | Prices or portfolio values, finite and positive. The simulators raise `ValueError` on NaN, infinite, zero or negative values (0.3.0 forward-filled NaN); drop missing rows first, e.g. `.dropna()`. |
| Daily labels | Dates (naive midnight), or each session's close as a timezone-aware moment, as [`simulate_buy_and_hold`](#simulate_buy_and_hold) gives. One optional first mark before the first close (the purchase at the open) is allowed. [`performance_report`](#performance_report) also accepts daily dates that are not sessions of `calendar` (a weekend, a holiday), counting time by the sessions they span. |
| Intraday labels | Timezone-aware: each session's open mark, then each bar's close, as `simulate_buy_and_hold` gives. A series without open marks needs `classes`; naive times of day raise, since their clock is unknown (localize them first). |
| Neither | A series with one mark per session away from the close (12:00 every day) is neither daily nor intraday and raises `ValueError`; [`to_session_values`](#to_session_values) turns it into a daily series. |

### `BacktestResult`

What [`simulate_buy_and_hold`](#simulate_buy_and_hold) returns. A
`NamedTuple`, so it unpacks into three names
(`values, dates, classes = simulate_buy_and_hold(...)`) and its fields read
by name. The simulators, the reports and the return functions all accept it
in place of a value series, and use its classes.

| Field | Type | Contents |
| --- | --- | --- |
| `values` | `pandas.Series` | Portfolio value at each mark, starting at `br0`, labelled with the moment it is true |
| `dates` | `pandas.DatetimeIndex` | The marks, the same as `values.index`; timezone-aware on the exchange's clock |
| `classes` | `pandas.Series` | The [class](#return-classes) of each return, indexed by the mark where it ends (one shorter than `values`) |

### `SimulationResult`

What the simulators return and the reports and plots take. A `NamedTuple`.

| Field | Type | Contents |
| --- | --- | --- |
| `paths` | `numpy.ndarray` | Shape `(n_sims, n_marks + 1)`, one path per row. Column 0 is `1.0`, at the last historical mark; every value is a growth multiple of the last historical value (1.10 = up 10%). |
| `dates` | `pandas.DatetimeIndex` or `None` | The `n_marks` future marks from [`next_trading_dates`](#next_trading_dates), labelled like the history; the last historical mark is not included. `None` for a plain array. |
| `classes` | `numpy.ndarray` or `None` | The class of the return ending at each future mark, `"gap"` or `"regular"`; `None` for a plain array |
| `fit` | `dict` or `None` | The fitted model: `None` for [`nonparametric_monte_carlo`](#nonparametric_monte_carlo), a dict for the other two |

[`statistical_report`](#statistical_report),
[`plot_simulated_paths`](#plot_simulated_paths) and
[`plot_path_comparison`](#plot_path_comparison) read `dates` and `classes`
from it, so nothing else is needed. They also take a bare array of paths, as
in 0.3.0 (`sims.paths`, or any array of that shape), with
[`dates`](#dates-for-simulated-paths) and `interval` as needed.

### `FutureLayout`

What [`next_trading_dates`](#next_trading_dates) returns: the marks that
continue a history. A `NamedTuple`.

| Field | Type | Contents |
| --- | --- | --- |
| `dates` | `pandas.DatetimeIndex` | The future marks, oldest first, not including the last historical mark |
| `classes` | `numpy.ndarray` | Class of the return ending at each mark: `"gap"` at a session's open mark, `"regular"` otherwise |
| `tau` | `numpy.ndarray` | Each return's length as a fraction of the bar size, $\tau/\Delta$: 1.0 for full bars and for gaps (a gap is one unit whatever its clock length), less than 1.0 for a short bar such as a half-day's last |

### Return classes

The class of the return ending at each mark, as
[`simulate_buy_and_hold`](#simulate_buy_and_hold) and
[`classify_returns`](#classify_returns) give it.

| Class | Return | Used by the simulators |
| --- | --- | --- |
| `"gap"` | From a session's last close to the next session's open (intraday only) | Drawn for future open marks |
| `"regular"` | One bar: open to close, close to close, or a daily or longer bar | Drawn for future bar closes |
| `"irregular"` | A held symbol is stale at either end, the session's first bar is missing, or the return spans a missing scheduled bar or session | Never drawn; kept in the values and the reports |

Daily and longer series, and continuous mode, have no gaps. The simulators
take the classes from a `BacktestResult`, from their `classes` argument, or
infer them from the labels with `classify_returns`, which sees labels only:
a stale price inside an external backtest can't be seen there.

### `interval`

The bar size, where a function takes one. The reports infer it from the
labels when it is left out, and need it only for paths given without dates.

| Accepted form | Examples |
| --- | --- |
| Bar-size string | `'30 secs'`, `'5 mins'`, `'1 hour'`, `'1 day'`, `'1 week'`, `'1 month'` |
| pandas-style string | `'1D'`, `'1d'`, `'5min'`, `'1h'`, `'1M'` |
| yfinance-style string | `'1m'`, `'5m'`, `'60m'`, `'1h'`, `'1d'`, `'1wk'`, `'1mo'` |
| `pandas.Timedelta` | `pd.Timedelta(minutes=5)`; a whole number of days counts sessions |

A bare `m` means minutes (yfinance) and a bare `M` months (pandas), so `'1m'`
is a 1-minute bar and `'1M'` a monthly one.

Where an annualization factor is needed (the interval of a daily or longer
series, and paths without dates in `statistical_report`) the bar size must be
`'1 day'`, `'1 week'` or `'1 month'`, the keys of
[`PERIODS_PER_YEAR`](#periods_per_year); others, such as `'3d'` or `'5m'`,
raise `ValueError` naming the bar size they were read as. On intraday series
the interval is the bar size $\Delta$ that short bars are measured against.

### `dates` for simulated paths

Optional labels for a bare array of paths, one per column: the last
historical mark (the starting value) followed by the future marks. With a
`SimulationResult` they are not needed; passed anyway, they replace its
labels. Build them from [`next_trading_dates`](#next_trading_dates):

```python
from portfolio_forecast.utils import next_trading_dates

# The last close, then the next 100 sessions: 101 labels for 100-mark paths
dates = prices.index[-1:].append(next_trading_dates(prices, 100).dates)
```

The plots use them to label the x-axis, and
[`statistical_report`](#statistical_report) to place each path's marks on the
calendar. Intraday labels must be timezone-aware.

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

The three simulators share their input, horizon and output. Each takes a
[`values`](#values-input) history, splits its returns into
[classes](#return-classes), lays out the future marks with
[`next_trading_dates`](#next_trading_dates), draws one log return
$\ell = \ln(1 + r)$ per future mark from its class's model, and compounds
them, $V_t = \exp(\sum \ell)$, so every path stays positive. A short future
bar (a half-day's last) is drawn per full bar and scaled back, its drift by
$\rho = \tau/\Delta$ and its volatility by $\sqrt\rho$. Parameters shared by
all three:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | Series, `BacktestResult` or array-like | — | History; see [`values` input](#values-input). |
| `sim_length` | `int` | `100` | Horizon: sessions on intraday history (if it ends mid-session, the rest of that session counts as the first, and the horizon ends at a session close); bars on daily and longer history and in continuous mode. Intraday history warns that it counts sessions. |
| `n_sims` | `int` | `1000` | Number of paths. |
| `random_state` | `int`, `Generator` or `None` | `None` | Seed or Generator; an int or Generator gives reproducible paths, `None` a fresh unseeded Generator. |
| `classes` | Series, array-like or `None` | `None` | Keyword only. The class of each return, indexed by the mark where it ends (or one per return, in order). `None` takes a `BacktestResult`'s classes, or infers them with [`classify_returns`](#classify_returns). Not allowed with a plain array. |
| `calendar` | `str` or `None` | `"XNYS"` | Keyword only. Calendar for the classes and the future marks; `None` is continuous mode: one class, future marks every bar size $\Delta$. |
| `min_sample` | `int` | `30` | Keyword only. Fewest returns a class may have; fewer raises `ValueError` naming the class, its count and the history needed. `0` disables the check, but a class the forecast needs with no returns always raises. See [Tuning the simulators](#tuning-the-simulators). |
| `prices` | Series | `None` | Keyword only. Deprecated alias for `values`; warns, and is removed in 1.0.0. |

**Returns** — A [`SimulationResult`](#simulationresult). For a plain array,
`paths` has `sim_length + 1` columns and `dates` and `classes` are `None`.

**Raises** — `ValueError` if `values` has fewer than two values, a NaN,
infinite, zero or negative value, or more than one column; if `classes`
doesn't match it; if `n_sims` is not a positive integer, `sim_length` is
negative or `min_sample` is not an integer of at least 0; if a class has
fewer than `min_sample` returns; if the series has one mark per session away
from the close (see [`to_session_values`](#to_session_values)); or if
intraday labels have neither `classes` nor open marks. `TypeError` if neither
or both of `values` and `prices` are given, or `classes` is given with a
plain array.

**Warns** — `UserWarning` on intraday history, that `sim_length` counts
sessions. `DeprecationWarning` for `prices` (and `return_fit`).

#### `nonparametric_monte_carlo`

```python
nonparametric_monte_carlo(values, sim_length=100, n_sims=1000, random_state=None, *,
                          classes=None, calendar="XNYS", min_sample=30, prices=None)
```

Simulate future paths by resampling historical log returns with replacement
(a bootstrap), each future mark from the pool of its class: a gap mark from
the gaps, a bar from the regular bars. No distribution is assumed, so fat
tails and skew carry over; every draw is independent, so volatility
clustering does not.

**Parameters** — the [shared parameters](#simulation).

**Returns** — A [`SimulationResult`](#simulationresult) with `fit=None`.

**See also** — [`parametric_monte_carlo`](#parametric_monte_carlo),
[`regime_switching_monte_carlo`](#regime_switching_monte_carlo).

#### `parametric_monte_carlo`

```python
parametric_monte_carlo(values, distribution=None, sim_length=100, n_sims=1000, random_state=None,
                       return_fit=False, *, classes=None, calendar="XNYS", criterion="BIC",
                       min_sample=30, prices=None, verbose=True)
```

Simulate future paths from `scipy.stats` distributions fitted by maximum
likelihood to historical log returns, one per class. Each path compounds
independent draws.

**Parameters** — the [shared parameters](#simulation), and:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `distribution` | `scipy.stats` distribution, `dict` or `None` | `None` | Family to fit to every class's log returns, e.g. `stats.t` or `stats.laplace`; or a dict `{"gap": ..., "regular": ...}` with one family (or `None`) per class the history has. `None` fits normal, t and Johnson SU and uses the lowest `criterion`. |
| `return_fit` | `bool` | `False` | Deprecated and ignored: the fit is always in `.fit`. Warns; removed in 1.0.0. |
| `criterion` | `"BIC"` or `"AIC"` | `"BIC"` | Keyword only, any case. Score that selects the family when `distribution=None`: $\mathrm{BIC} = -2\ln L + k\ln n$, $\mathrm{AIC} = -2\ln L + 2k$. |
| `verbose` | `bool` | `True` | Keyword only, last. `True` prints each class's candidate scores and the family selected with its fitted parameters by name (log return per gap or per full bar), e.g. `Selected: t  (df 4.12, loc 0.000912, scale 0.00981)`; with a family given, one `Fitted:` line per class in that format. `False` prints nothing. Warnings are unaffected. |

**Returns** — A [`SimulationResult`](#simulationresult) whose `fit` is a
dict with one entry per modelled class, `"regular"` and (intraday) `"gap"`,
each a dict:

| Key | Contents |
| --- | --- |
| `"Distribution"` | scipy.stats name of the family used, e.g. `'johnsonsu'` |
| `"Parameters"` | Its fitted parameters by name: shape parameters, then `loc` and `scale`, of the log return per unit (per gap, or per full bar) |
| `"Criterion"` | `"BIC"` or `"AIC"` |
| `"BIC"` or `"AIC"` | The family's score, under the criterion's name |
| `"Candidates"` | Every candidate's score by name, lowest first, when the family was selected; otherwise `None` |

**Raises** — as the [shared parameters](#simulation), and `ValueError` if a
class has fewer returns than the family has parameters, `distribution` is a
dict missing a modelled class or with an unknown key, or `criterion` is not
AIC or BIC.

**Notes** — Fitting log returns keeps every path positive however heavy the
fitted left tail. Student's t and Johnson SU are fitted from several
starting points and polished to the maximum likelihood.

**See also** — [`nonparametric_monte_carlo`](#nonparametric_monte_carlo),
[`regime_switching_monte_carlo`](#regime_switching_monte_carlo),
[Tuning the simulators](#tuning-the-simulators).

#### `regime_switching_monte_carlo`

```python
regime_switching_monte_carlo(values, n_regimes=None, bootstrap=False, n_starts=10,
                             sim_length=100, n_sims=1000, random_state=None, return_fit=False,
                             *, classes=None, calendar="XNYS", gap_model="regime", kappa=0.0,
                             criterion="BIC", min_sample=30, prices=None, verbose=True)
```

Fit a Gaussian hidden Markov model to the history's log returns, one chain
of hidden regimes with a normal emission per regime and class, and simulate
paths that move between regimes by the fitted transition matrix, one step
per mark. Each path starts from the model's regime probabilities at the last
historical mark. Keeps volatility clustering. The fit is the package's own
EM.

**Parameters** — the [shared parameters](#simulation), and:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `n_regimes` | `int` or `None` | `None` | Number of regimes. `None` fits 1, 2 and 3 and uses the eligible count with the lowest `criterion`. |
| `bootstrap` | `bool` | `False` | `False` draws from the fitted normal of the mark's regime and class; `True` resamples the historical returns of that class whose most likely regime is that one, so the tails are the history's own, as heavy as they were and never beyond its largest move. |
| `n_starts` | `int` | `10` | EM starts per regime count; EM finds a local optimum, so the best of these is kept. |
| `return_fit` | `bool` | `False` | Deprecated and ignored: the fit is always in `.fit`. Warns; removed in 1.0.0. |
| `gap_model` | `"regime"` or `"independent"` | `"regime"` | Keyword only. `"regime"`: the chain steps at every mark, gaps included, and each regime has its own gap emission. `"independent"`: the chain runs on bars only, and gaps are one normal (or one pool) whatever the regime. Ignored without gaps. |
| `kappa` | `float` | `0.0` | Keyword only. Sticky prior weight on staying in a regime; larger values give longer-lived regimes, `0` is maximum likelihood. |
| `criterion` | `"BIC"` or `"AIC"` | `"BIC"` | Keyword only, any case. Score that picks the regime count when `n_regimes=None`. |
| `verbose` | `bool` | `True` | Keyword only, last. `True` prints the score of each regime count (when selecting) and each regime's mean, volatility and expected duration; on intraday history the moments are labelled `regular` (per full bar), and under `gap_model="regime"` with gaps each regime's gap mean and volatility follow. `False` prints nothing. Warnings are unaffected. |

**Returns** — A [`SimulationResult`](#simulationresult) whose `fit` is a
dict:

| Key | Contents |
| --- | --- |
| `"Regime Count"` | Number of regimes used |
| `"Gap Model"`, `"Kappa"`, `"Criterion"` | The settings used |
| `"BIC"` or `"AIC"` | Score by regime count, under the criterion's name: every count fitted and eligible when `n_regimes=None`, otherwise just `n_regimes` |
| `"Log Likelihood"` | Log-likelihood of the model used |
| `"Regimes"` | `DataFrame`, one row per regime from lowest to highest volatility: `Mean` and `Volatility` of the log return per full bar, `Gap Mean` and `Gap Volatility` (under `"regime"` with gaps), `Expected Duration` in marks ($1/(1 - p_{kk})$), and `Current Probability` at the last historical mark |
| `"Transition Matrix"` | `DataFrame` of one-step transition probabilities, row regime to column regime, same order |
| `"Gap"` | Under `"independent"` with gaps and `bootstrap=False` only: the gap normal's `Mean` and `Volatility` |

**Raises** — as the [shared parameters](#simulation), and `ValueError` if
`n_regimes` is not a positive integer, a given `n_regimes` can't be fitted or
fails `min_sample`, no count of 1–3 is eligible, the gaps are constant under
`"independent"` with `bootstrap=False`, or `gap_model`, `criterion` or a
negative `kappa` is invalid.

**Notes** — The model numbers regimes arbitrarily; ordering them by
volatility changes only how they are reported. Seeded paths differ from
0.3.0, which fitted with `hmmlearn`.

**See also** — [`nonparametric_monte_carlo`](#nonparametric_monte_carlo),
[`parametric_monte_carlo`](#parametric_monte_carlo),
[Tuning the simulators](#tuning-the-simulators).

### Reporting

`from portfolio_forecast.performance import ...`

#### `statistical_report`

```python
statistical_report(sims, interval=None, dates=None, risk_free_rate=0.0, confidence=0.95, display=False,
                   *, calendar="XNYS", cagr_clock="session")
```

Compute every [`performance_report`](#performance_report) figure on each
simulated path, under the same rules, then summarize each by its mean,
median and range across paths, and add tail risk of the total return.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `sims` | `SimulationResult` or array-like | — | A [`SimulationResult`](#simulationresult), whose future marks and classes place the paths on the calendar (column 0 at the last historical mark); or an array of shape `(n_sims, n_marks + 1)`. |
| `interval` | `str` or `Timedelta` | `None` | Bar size; see [`interval`](#interval). Needed only for paths without dates, which are read as daily, weekly or monthly bars. |
| `dates` | array-like of datetimes | `None` | `n_marks + 1` labels; see [`dates`](#dates-for-simulated-paths). Replace a `SimulationResult`'s labels if given. |
| `risk_free_rate` | `float` | `0.0` | Annual risk-free rate, spread evenly over the year's sessions ($r_f/252$). |
| `confidence` | `float` | `0.95` | Share of paths inside each range, in (0, 1); 0.95 spans the 2.5th–97.5th percentiles and puts VaR and CVaR at the 5% tail. |
| `display` | `bool` | `False` | Also print a formatted report. Its header gives the total time in calendar years (first to last mark over 365.25 days, whatever `cagr_clock` is) with the whole calendar days in brackets, e.g. `Total Time : 3.00 years (1095 days)`, then the number of periods; paths without dates show only the number of periods. The Gap Vol row appears only if the marks include a gap, the Bar Vol row only on intraday marks; the returned tables keep both rows. |
| `calendar` | `str` or `None` | `"XNYS"` | Keyword only. Calendar code; `None` is continuous mode. |
| `cagr_clock` | `"session"` or `"calendar"` | `"session"` | Keyword only. Clock of CAGR and Total Years: sessions after column 0 over 252, or calendar days over 365.25 (needs dates; required in continuous mode). Paths without dates count bars. |

**Returns** — `dict`:

| Key | Type | Contents |
| --- | --- | --- |
| `"Per Path Metrics"` | `DataFrame` | One row per path; columns Final Bankroll, Total Return, CAGR, Gap Volatility, Bar Volatility, Annualized Volatility, Downside Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio |
| `"Period Returns"` | `ndarray` | Shape `(n_sims, n_marks)`: returns between consecutive marks, NaN after a total loss |
| `"Daily Returns"` | `ndarray` | One row per path: the session returns behind the risk figures, a partial first session left out |
| `"Paths"`, `"Periods"` | `int` | `n_sims`, `n_marks` |
| `"Total Days"` | `int` or `None` | Whole calendar days from column 0 to the last mark; `None` without dates |
| `"Total Years"` | `float` | Years on `cagr_clock` |
| `"Total Time (Years)"` | `float` or `None` | Calendar years from column 0 to the last mark (days over 365.25), whatever `cagr_clock` is; `None` without dates. The printed and PDF headers show it |
| `"Has Gaps"`, `"Intraday"` | `bool` | Whether the marks include at least one gap return, and whether they are intraday; both `False` without dates. The printout and the PDF show the Gap and Bar Volatility rows only when these are `True` |
| `"Confidence"` | `float` | The `confidence` used |
| `"Returns"` | `DataFrame` | Rows Final Bankroll, Total Return, CAGR; columns Mean, Median, Lower, Upper |
| `"Risk"` | `DataFrame` | Rows Gap Volatility, Bar Volatility, Annualized Volatility, Downside Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio; same columns |
| `"Tail Risk"` | `dict` | `"Probability of Loss"` (share of paths with a negative total return), `"Value at Risk"` (total return at the `1 - confidence` quantile), `"Conditional VaR"` (mean total return at or below it) |

Lower and Upper bound the 95% range across paths (for the default
`confidence`): percentiles across paths under the fitted model, not a
confidence interval.

**Raises** — `ValueError` if a path has fewer than two marks, `confidence` is
not in (0, 1), `cagr_clock` is unknown or doesn't suit the input
(`"calendar"` without dates, `"session"` in continuous mode), `interval` is
missing, intraday or unsupported for paths without dates, the dates or
classes don't match the paths, or the marks can't be placed on the calendar.

**Warns** — `UserWarning` with the count and share of paths whose Sortino
ratio is infinite ("Sortino ratio is infinite on {k} of {N} paths
({p:.1%}); its mean is NaN"), the count of paths ending at a total loss, and
the count of paths left out of a summary because a figure is undefined there.

**Notes** — A path ending at 0 has CAGR −100% and stays in every summary. A
Sortino ratio of $+\infty$ (no down session) stays in its summary: the
Sortino mean is then NaN, while the median and range use inf-safe
percentiles. Gap and Bar Volatility are NaN on daily paths.

**See also** — [`performance_report`](#performance_report),
[`plot_simulated_paths`](#plot_simulated_paths).

#### `performance_report`

```python
performance_report(values, classes=None, *, interval=None, calendar="XNYS", risk_free_rate=0.0,
                   cagr_clock="session", display=False, dates=None)
```

Compute performance and risk figures for one portfolio value series, such as
a [`BacktestResult`](#backtestresult) from
[`simulate_buy_and_hold`](#simulate_buy_and_hold).

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | `Series` or `BacktestResult` | — | Portfolio values at every mark, oldest first, on a `DatetimeIndex`: dates are fine for daily and longer series (sessions or not), intraday labels must be timezone-aware. A `BacktestResult` also supplies the classes. |
| `classes` | `Series` | `None` | Class of each return, indexed by the mark where it ends; `None` takes a `BacktestResult`'s or infers them. Used for Gap and Bar Volatility only. |
| `interval` | `str` or `Timedelta` | `None` | Keyword only. Bar size; `None` infers it. Picks the annualization of daily and longer series (252, 52 or 12); on intraday series, the bar size short bars are scaled to. |
| `calendar` | `str` or `None` | `"XNYS"` | Keyword only. Calendar code; `None` is continuous mode. |
| `risk_free_rate` | `float` | `0.0` | Keyword only. Annual risk-free rate, spread evenly over the year's sessions. |
| `cagr_clock` | `"session"` or `"calendar"` | `"session"` | Keyword only. Clock of CAGR and Total Years, as in [`statistical_report`](#statistical_report). |
| `display` | `bool` | `False` | Keyword only. Also print a formatted report. Its header gives the total time in calendar years (first to last mark over 365.25 days, whatever `cagr_clock` is) with the whole calendar days in brackets, e.g. `Total Time : 3.00 years (1095 days)`, then the number of returns. The Gap Vol row appears only if the series has a gap return, the Bar Vol row only on intraday series; the dict keeps both keys, NaN when absent. |
| `dates` | array-like | `None` | Keyword only. Deprecated: the labels come from `values`. Warns, and is removed in 1.0.0. |

**Returns** — `dict`:

| Key | Contents |
| --- | --- |
| `"Period Returns"` | `Series` of returns between consecutive marks, NaN after a total loss |
| `"Daily Returns"` | `Series` of session returns, the series behind the risk figures; UTC-day returns in continuous mode; NaN after a total loss, rows the risk figures drop |
| `"Periods"` | Number of returns between marks, `len(values) - 1` |
| `"Total Days"` | Whole calendar days from the first mark to the last |
| `"Total Years"` | Years on `cagr_clock` |
| `"Returns"` | `dict`: Initial Bankroll, Final Bankroll, Total Return, CAGR (−100% at a total loss), Actual Yearly Returns (from [`calculate_actual_yearly_return`](#calculate_actual_yearly_return)) |
| `"Risk"` | `dict`: Gap Volatility and Bar Volatility (per unit, unannualized, NaN on daily data), Annualized Volatility, Downside Volatility (annualized), Max Drawdown (a fraction in [0, 1]), Sharpe Ratio, Sortino Ratio |

**Raises** — `ValueError` if `values` has fewer than two marks, `cagr_clock`
is unknown or `"session"` in continuous mode, `interval` can't be parsed, has
no annualization factor or is intraday on a daily series, or the labels can't
be placed on the calendar. `TypeError` if `values` has no `DatetimeIndex`.

**Warns** — `UserWarning` when no session return is below the risk-free
rate (Sortino is then $+\infty$, or NaN if the mean excess return is 0), and
in continuous mode when the series spans fewer than two 00:00 UTC boundaries
(volatility, Sharpe and Sortino are then NaN).

**Notes**

- Volatility, Sharpe and Sortino use session returns
  $R_d = V(C_d)/V(C_{d-1}) - 1$, never per-mark returns, since gap and bar
  returns are not alike. A partial first session $R_1$ is left out, so the
  figures are the same under both fills; daily data bought at the close (or
  dated) keeps it. Total return, CAGR and drawdown use every mark.
- In continuous mode the fill can't be read from the labels: a series of
  daily or longer bars is read as bought at the close and keeps its first
  return $R_1$; an intraday series always leaves out its first UTC day.
- Daily dates need not be sessions of `calendar`: the session clock counts
  the sessions they span.

**See also** — [`statistical_report`](#statistical_report).

### Returns and metrics

`from portfolio_forecast.performance import ...` — the building blocks of
[`performance_report`](#performance_report). Each value-series function also
accepts a [`BacktestResult`](#backtestresult). Returns are simple,
$r = V_t/V_{t-1} - 1$, as fractions.

#### `calculate_period_returns`

```python
calculate_period_returns(values)
```

Returns between consecutive marks, labelled with the mark where each ends. A
return across a missing value is measured from the last known value; returns
after a total loss are $0/0$, undefined, and kept as NaN (the $-100\%$ return
itself is kept as $-1$).

**Returns** — `Series`, always one shorter than `values`.

#### `calculate_session_returns`

```python
calculate_session_returns(values, calendar="XNYS")
```

The session returns behind volatility, Sharpe and Sortino: one return per
full session, $R_d = V(C_d)/V(C_{d-1}) - 1$, labelled with the session's
last mark. On intraday data this is the gap and the bars compounded. Daily,
weekly and monthly series give their own bar returns; continuous mode gives
calendar-day returns at 00:00 UTC. A partial first session is left out.
Returns after a total loss are $0/0$ and kept as NaN, so the length depends
only on the labels and the calendar; the risk figures drop those rows.

**Returns** — `Series`, one return per full session.

**Raises** — `ValueError` if the labels can't be placed on the calendar.

#### `calculate_daily_return`

```python
calculate_daily_return(values)
```

Deprecated name of [`calculate_period_returns`](#calculate_period_returns)
(the returns are per mark, not per day). Same result, with a
`DeprecationWarning`; removed in 1.0.0.

#### `calculate_cagr`

```python
calculate_cagr(values, periods_per_year=None, *, years=None)
```

Compound annual growth rate, $(V_T/V_0)^{1/Y} - 1$.

**Parameters** — `values` (`Series`) portfolio values; `periods_per_year`
(`float`, optional) marks per year, for the 0.3.0 bar clock
$Y = (n - 1)/\text{periods\_per\_year}$, ignored when `years` is given;
`years` (`float`, keyword only) the span $Y$, e.g. from
[`calculate_total_years`](#module-level-functions).

**Returns** — `float`; exactly −1.0 when $V_T = 0$, NaN when $V_T < 0$.

**Raises** — `ZeroDivisionError` if $Y$ is zero; `ValueError` if neither
`periods_per_year` nor `years` is given.

#### `calculate_sharpe_ratio`

```python
calculate_sharpe_ratio(period_returns, periods_per_year=252, risk_free_rate=0.0)
```

Annualized Sharpe ratio of session returns:
$\bar e / \hat\sigma(R) \cdot \sqrt{\text{periods\_per\_year}}$ with
$e = R - r_f/\text{periods\_per\_year}$ and the sample standard deviation.

**Parameters** — `period_returns` (`Series`) session returns, e.g. from
[`calculate_session_returns`](#calculate_session_returns);
`periods_per_year` (`float`, default 252) returns per year;
`risk_free_rate` (`float`, default 0.0) annual rate.

**Returns** — `float`; NaN for fewer than two returns or zero variance.

#### `calculate_sortino_ratio`

```python
calculate_sortino_ratio(period_returns, periods_per_year=252, risk_free_rate=0.0)
```

Annualized Sortino ratio: the Sharpe ratio with the downside deviation
$\sqrt{\sum \min(e, 0)^2 / n}$, over all $n$ sessions, in place of the
standard deviation. The risk-free rate is also the downside target.

**Parameters** — as [`calculate_sharpe_ratio`](#calculate_sharpe_ratio).

**Returns** — `float`; $+\infty$ when no session is below the target (NaN if
the mean excess return is also 0); NaN for no returns.

**Warns** — `UserWarning` when no session is below the target: "Sortino
ratio is infinite: none of the {n} sessions is below the risk-free rate", or,
when the mean excess return is also 0, "Sortino ratio is NaN: all {n}
sessions return exactly the risk-free rate".

#### `calculate_max_drawdown`

```python
calculate_max_drawdown(values)
```

Largest fall from a running high, $\max_t \left(1 - V(t)/\max_{s \le t} V(s)\right)$,
over every mark, overnight gaps included.

**Returns** — `float` in [0, 1]; 0.25 is a 25% fall (0.3.0 returned −0.25).

#### `calculate_actual_yearly_return`

```python
calculate_actual_yearly_return(values, dates=None, calendar="XNYS")
```

Return earned in each calendar year. The first year is measured from the
first value; later years from the previous year's last value, so the move
across each year boundary is counted once. A mark's year is its session's
year on the exchange's clock.

**Parameters** — `values` (`Series`) portfolio values; `dates` (array-like,
optional) labels, if `values` has none; `calendar` (`str` or `None`, default
`"XNYS"`).

**Returns** — `DataFrame` indexed by year, with columns `return`; `days`
(365 or 366 for a full year, else the calendar days from its first to last
mark, inclusive); and `is_full` (True when the series covers the year's first
session from its open and its last session to its close; in continuous mode,
00:00 UTC on 1 January to the next).

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
| `sims` | `SimulationResult` or array-like | — | A [`SimulationResult`](#simulationresult), whose future marks label the x-axis (column 0 reads "Start"); or an array of paths. |
| `method` | `str` | `'Monte Carlo'` | Simulation method, shown in the title. |
| `dates` | array-like of datetimes | `None` | X-axis labels for every column; see [`dates`](#dates-for-simulated-paths). Replace a `SimulationResult`'s labels if given. With neither, the marks are numbered 0 to n. |
| `figsize` | `tuple` | `(12, 6.5)` | Figure size in inches. |
| `dpi` | `int` | `120` | Figure resolution. |

**Returns** — `(fig, ax)`: the matplotlib Figure and Axes.

**Raises** — `ValueError` if `dates` does not have one entry per column, or a
`SimulationResult`'s dates one per column after column 0.

**Notes** — Marks stay evenly spaced even with dates, so weekends, holidays
and nights do not stretch the paths; on intraday paths each session's open
mark is its own step. When every path starts at the same value $s$ (1.0 for
the simulators), the y-axis reads "Portfolio Value (start = $s$)" and a
dashed line marks $s$; paths with mixed starts read "Portfolio Value", with
no line.

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
| `sims_by_method` | `dict` | — | Method name → `SimulationResult` or array of paths, in panel order. |
| `dates` | array-like of datetimes | `None` | X-axis labels shared by every panel; `None` labels each panel from its `SimulationResult`'s dates, if any. |
| `figsize` | `tuple` | `None` | Figure size; defaults to `(6.5 * n_methods, 6)`. |
| `dpi` | `int` | `120` | Figure resolution. |

**Returns** — `(fig, axes)`: the Figure and an array of Axes, one per method.

**Raises** — as [`plot_simulated_paths`](#plot_simulated_paths), for every
method's paths.

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
(simulation setup, returns, risk, tail risk) and its figures. As in the
printout, the risk table has a Gap Volatility row only if the marks include a
gap, and a Bar Volatility row only on intraday marks. LaTeX runs in a
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

**Notes** — The Lower and Upper columns are headed "95% range across paths"
(for the default `confidence`), and an infinite Sortino ratio prints as
$\infty$. Needs the booktabs, caption, fancyhdr, float, geometry, graphicx,
hyperref, lmodern, microtype and xcolor LaTeX packages, all part of TeX Live,
MacTeX and MiKTeX. Figures are embedded as vector PDF and are not modified or
closed; a figure is scaled to its `"Width"` share of the text width, so a
wide one (such as a three-panel
[`plot_path_comparison`](#plot_path_comparison)) gets small labels, and a
narrower `"Width"` makes them smaller still. With `pdflatex`, text must use
characters it can typeset (Latin scripts, common symbols); an emoji, for
example, makes LaTeX fail.

**See also** — [`statistical_report`](#statistical_report),
[`plot_simulated_paths`](#plot_simulated_paths).

### Trading dates, marks and bar sizes

`from portfolio_forecast.utils import next_trading_dates, classify_returns, to_session_values, close_times, PERIODS_PER_YEAR`

#### `next_trading_dates`

```python
next_trading_dates(values, n, interval=None, calendar='XNYS', tz=None)
```

Return the future marks that continue a history on the exchange calendar,
with the class and length of each return. The simulators use it for their
future layout.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | Series, DataFrame, `BacktestResult` or array-like of datetimes | — | The history: a value series or a frame with any number of columns (only its index is used, so the result is the same as passing `values.index`), or its labels. Duplicates and NaT are dropped and the rest sorted. |
| `n` | `int` | — | Horizon: sessions on intraday history, bars on daily and longer history and in continuous mode. Zero or less gives an empty layout. |
| `interval` | `str` or `Timedelta` | `None` | Bar size; see [`interval`](#interval). `None` infers it from the history. |
| `calendar` | `str` or `None` | `'XNYS'` | Calendar code; `None` is continuous mode. |
| `tz` | `str` or `tzinfo` | `None` | Timezone of naive intraday labels; defaults to the machine's local timezone. Ignored for dates. |

**Returns** — A [`FutureLayout`](#futurelayout); its `dates` are labelled
like the history (naive or timezone-aware; dates stay dates).

**Raises** — `ValueError` if `values` is empty, `interval` can't be parsed
or inferred, a mark falls outside the calendar's sessions, `interval` is
intraday but the history is dates, or the history has one mark per session
away from the close (see [`to_session_values`](#to_session_values)); in
continuous mode, if `interval` is in months.

**Notes** — Intraday: the grid (bar size and phase) is learned from the
history. If it ends mid-session, that session's remaining bars come first and
count as the first of the `n` sessions, so the horizon ends at a session
close. Every other session has one open mark (class `"gap"`) and its bar
closes, with a short last bar at the session close (half-days included). A
session that opens the instant the previous one closed (weeknights on
`"24/5"`) has no open mark. Daily and longer: n-day bars step in sessions;
weekly and monthly bars keep the history's labelling. Continuous mode: the
marks are $t_T + k\Delta$ for $k = 1, \dots, n$.

**Example**

```python
from portfolio_forecast.utils import next_trading_dates

layout = next_trading_dates(prices, 20)
layout.dates    # the next 20 sessions
```

#### `classify_returns`

```python
classify_returns(values, calendar="XNYS")
```

The [class](#return-classes) of every return of a value series, inferred
from its labels on the calendar, for Monte Carlo input from an external
backtest.

**Parameters** — `values` (`Series`, `BacktestResult` or `DatetimeIndex`)
the value series or its labels, oldest first; intraday labels
timezone-aware. `calendar` (`str` or `None`, default `"XNYS"`).

**Returns** — `Series` of `"gap"`, `"regular"` or `"irregular"`, indexed by
the mark where each return ends (`values.index[1:]`).

**Raises** — `ValueError` if an intraday series spanning several sessions
has no open marks to infer the gaps from (pass `classes` to the simulator,
or collapse the series with `to_session_values`), the series has one mark
per session away from the close, labels are naive times of day, or a mark
falls outside every session.

**Warns** — `UserWarning` in continuous mode when more than 10% of returns
are irregular, naming `calendar="24/5"` or an exchange calendar as the fix.

**Notes** — Classes come from the labels only: a symbol priced stale inside
an external backtest can't be seen here, whereas
[`simulate_buy_and_hold`](#simulate_buy_and_hold) supplies classes that
account for it.

#### `to_session_values`

```python
to_session_values(values, calendar="XNYS", *, now=None)
```

Collapse a value series to its first value and each session's last value,
for a daily forecast or daily figures from intraday history. Its returns are
the session returns behind the risk figures.

**Parameters** — `values` (`Series` or `BacktestResult`), oldest first;
intraday labels timezone-aware. `calendar` (`str` or `None`, default
`"XNYS"`). `now` (datetime-like, keyword only, default `None`): the current
time, for leaving out a session still in progress; `None` uses the clock, and
a naive `now` is read in this machine's local timezone, as in
[`simulate_buy_and_hold`](#simulate_buy_and_hold).

**Returns** — `Series`: the first value (the purchase), then each session's
last value. A daily or longer series comes back unchanged. A series with one
mark per session away from the close (12:00 every session) is accepted, and
each value is relabelled at its session's close, which makes it daily. On
intraday data, a session whose last mark is before its close (its final bars
are missing, as Yahoo Finance sometimes serves a half-day without its last
bar) has its last value labelled at the close too, if it has closed by `now`.
An intraday series' last session whose close is after `now` is still in
progress and is left out, without a warning. In continuous mode: the
first value, then the value at the last mark at or before each 00:00 UTC,
labelled at that midnight.

**Raises** — `ValueError` if labels are naive times of day, not strictly
increasing, or outside every session; in continuous mode, if a non-daily
series has naive labels.

**Warns** — `UserWarning` when intraday sessions end before their close:
"{k} session(s) end before their close, first {date}: each one's last value
is labelled at its close". A series with one mark per session is relabelled
without a warning.

#### `close_times`

```python
close_times(index, calendar='XNYS', tz=None)
```

The moment each bar closes, for an index labelled with bar start times (as
yfinance and Interactive Brokers label bars). A close, or a value computed
from it, is only known when its bar ends, so this is the label that says when
it is true. [`simulate_buy_and_hold`](#simulate_buy_and_hold) uses it to
label bar closes.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `index` | array-like of datetimes | — | Bar start times, oldest first. |
| `calendar` | `str` or `None` | `'XNYS'` | Calendar code for the session closes; `None` is continuous mode, where each bar closes one grid step after its start. |
| `tz` | `str` or `tzinfo` | `None` | Timezone of naive intraday times; defaults to the machine's local timezone. |

**Returns** — `DatetimeIndex`, one close per bar. Intraday closes keep the
input's timezone (or naive wall clock); daily closes are timezone-aware, on
the exchange's clock; continuous mode keeps the input's labelling.

**Raises** — `ValueError` if a bar falls outside the calendar's sessions.

**Notes** — An intraday bar closes at the next point of its bar grid and no
later than its session's close: hourly bars starting at 9:30, 10:30, … close at
10:30, 11:30, …, 15:30 and 16:00 (13:00 on a half-day). The grid is learned from
the index, so a missing bar doesn't stretch the one before it. Dates spaced
like daily bars close at their session's close; weekly and monthly dates are
returned unchanged. An unfinished bar gets its scheduled close.

#### `PERIODS_PER_YEAR`

```python
PERIODS_PER_YEAR = {'1 day': 252, '1 week': 52, '1 month': 12}
```

Periods per year of the series risk figures are annualized on: 252
sessions, 52 weeks, 12 months. These are conventions, not counts of a real
year. Intraday bar sizes have no entry: their risk figures use session
returns, annualized with 252 (0.3.0 annualized each bar). Continuous mode
annualizes calendar days with 365. The reports also accept other spellings
of these sizes; see [`interval`](#interval).

### Backtesting

`from portfolio_forecast.performance import ...`

#### `simulate_buy_and_hold`

```python
simulate_buy_and_hold(data, w, br0=1, fill="open", calendar="XNYS", tz=None, now=None)
```

Backtest a fixed-weight portfolio bought once, at the first bar's open or
close, and never rebalanced, so its weights drift with prices. Every value is
the portfolio's worth at prices known at its label, labelled with the moment
it is true on the exchange's clock.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `DataFrame` | — | Historical prices, indexed by bar label (start times for intraday bars, dates for daily and longer), in any layout [`get_closing_prices`](#get_closing_prices) accepts. Opens are read from an `Open` field. |
| `w` | `Series`, `dict` or array-like | — | Weights: a Series or dict aligned by symbol (missing symbols get zero), or an array matched by column position. Normalized to sum to 1. |
| `br0` | `float` | `1` | Starting portfolio value. |
| `fill` | `"open"` or `"close"` | `"open"` | Buy at the first bar's open, which is then the first mark, so the first bar's own move counts (needs opens); or at its close. |
| `calendar` | `str` or `None` | `'XNYS'` | Calendar code for sessions, opens and closes. `None` is continuous mode for 24/7 data; weekday-only data off an exchange takes `"24/5"`. |
| `tz` | `str` or `tzinfo` | `None` | Timezone of naive intraday times (in continuous mode, of naive dates too) and of a naive `now`; defaults to the machine's local timezone. |
| `now` | datetime-like | `None` | The current time; bars still in progress at `now` are dropped, without a warning. `None` uses the clock. |

**Returns** — A [`BacktestResult`](#backtestresult). Its marks:

| Data | First mark (`fill="open"`) | Later marks |
| --- | --- | --- |
| Intraday | The first bar's start, at `br0` | Each session's open mark (class `"gap"`), then each bar's close |
| Daily and longer | The first session's open, at `br0` | Each bar's last session close |
| Continuous mode | The first bar's start, at `br0` | Each bar's start plus the bar size |

With `fill="close"` the first mark is the first bar's close, at `br0`, and
session 1 has no open mark.

**Raises** — `ValueError` if `fill` is unknown; intraday data has no opens;
`fill="open"` is given data without opens (the message names
`fill="close"`); the weights sum to zero or less, an array of weights does
not have one entry per symbol, or no held symbol has a price at the fill; a
daily bar is dated on a day that is not a session; no bar falls inside the
calendar's sessions; or no bar has closed by `now`.

**Warns** — `UserWarning`, one per case, each with its count and the symbols
or dates involved: intraday bars dropped outside every session; held symbols
dropped for having no price at the fill (the remaining weights are
renormalized); held symbols whose prices stop early (held as cash at their
last price); sessions whose first bar is missing; returns made irregular by
a stale price or a missing scheduled bar; in continuous mode, more than 10%
irregular returns.

**Notes**

- A gap inside a symbol's history holds its last price, and the returns
  touching it are irregular. A symbol with no price at the fill (e.g. listed
  later) can't be bought, so it is dropped; the drop never depends on prices
  after the purchase.
- A bar with an open but no close (Yahoo can return one for a session in
  progress) is still a bar: its close mark is valued at the last known
  prices, as in 0.3.0, and the symbol is reported as stopping early. A bar
  still in progress at `now` is dropped first, so with the default `now`
  this only happens to data that stops mid-session in the past.
- yfinance and Interactive Brokers label bars with their start time, but a
  bar's close is only known at its end, so bar values move to
  [`close_times`](#close_times).
- With an adjusted close, each open is put on the adjusted basis with its
  own bar's factor, so a dividend lands in the gap. yfinance intraday data
  has no adjusted close, so intraday backtests omit dividends.

**See also** — [`performance_report`](#performance_report),
[`classify_returns`](#classify_returns).

#### `get_closing_prices`

```python
get_closing_prices(data)
```

Select closing prices as a (date × symbol) frame. Accepts MultiIndex columns
with a close field on either level (e.g. from `yfinance.download`), flat
OHLCV columns for one symbol, or a frame that is already closes. Field names
match case-insensitively, and an adjusted close (`Adj Close`, `adj_close` or
`adjclose`) is preferred over `Close` because it includes dividends. Import
from `portfolio_forecast.performance.simulate`.

**Parameters** — `data` (`DataFrame`) prices.

**Returns** — `DataFrame`, one column per symbol; returned unchanged if it has
no `close` column.

**Raises** — `KeyError` if MultiIndex columns have no `close` field.

### Module-level functions

Public functions the package uses internally, importable from their modules
but not re-exported. Their docstrings give the full detail.

| Function | Module | Returns |
| --- | --- | --- |
| `session_spans(index, calendar="XNYS", tz=None)` | `utils.calendar` | `DataFrame` with each bar's `first_open`, `last_close` and `n_sessions`. Raises `ValueError` for a bar outside the calendar's sessions or an intraday bar starting outside session hours (the backtest drops those with a warning before calling it). |
| `completed_mask(index, calendar="XNYS", tz=None, now=None)` | `utils.calendar` | Boolean array: which bars have closed by `now` |
| `is_daily(values, calendar="XNYS")` | `utils.marks` | Whether a series is daily or longer (at most one mark per session, at its close, plus an optional first mark), decided on the calendar |
| `history_marks(opens, closes, held, calendar="XNYS", tz=None, fill="open")` | `utils.marks` | The backtest's marks, each symbol's freshness at them, and each return's class |
| `count_sessions(values, calendar="XNYS")` | `performance.metrics` | The sessions the session clock counts: closes after the first mark, a partial first or last session counted whole. Daily dates need not be sessions. |
| `calculate_total_years(values, calendar="XNYS", cagr_clock="session")` | `performance.metrics` | Years spanned, $N/252$ or calendar days over 365.25 |
| `calculate_excess_returns(returns, risk_free_rate=0.0, periods_per_year=252)` | `performance.metrics` | $R - r_f/\text{periods\_per\_year}$ |
| `calculate_volatility(returns, periods_per_year=252)` | `performance.metrics` | Annualized volatility, $\hat\sigma(R)\sqrt{\text{periods\_per\_year}}$ |
| `calculate_downside_deviation(period_returns, periods_per_year=252, risk_free_rate=0.0)` | `performance.metrics` | Per-session downside deviation, $\sqrt{\sum \min(e, 0)^2/n}$ |
| `calculate_class_volatility(values, classes=None, calendar="XNYS", interval=None)` | `performance.metrics` | `"Gap Volatility"` and `"Bar Volatility"`, per unit, short bars scaled to a full bar |

### Internal helpers

Private functions, listed for completeness.

| Function | Module | Role |
| --- | --- | --- |
| `resolve_values`, `warn_return_fit` | `forecast/_common.py` | Accept the deprecated `prices` keyword in place of `values`, and warn that `return_fit` is ignored |
| `prepare_history`, `check_min_sample`, `to_unit`, `from_unit`, `assemble` | `forecast/_common.py` | Resolve the history into log returns per class and the future layout, apply `min_sample`, rescale short bars, and compound the draws into a `SimulationResult` |
| `fit_hmm`, `score`, `log_likelihood`, `posteriors` | `forecast/_hmm.py` | The hidden Markov model's EM fit, its BIC or AIC, and evaluation at fixed parameters |
| `_resolve_report_input`, `_resolve_paths`, `_layout_origin` | `performance/report.py` | Read the reports' inputs, and recover the last historical mark of a `SimulationResult` |
| `_range_summary`, `_quantile` | `performance/report.py` | Mean, median and inf-safe range of one figure across paths |
| `_find_field`, `_get_field` | `performance/simulate.py` | Find a price field among column labels (case-insensitive, in order of preference) and select it from any accepted layout |
| `_get_opening_prices`, `_normalize_weights` | `performance/simulate.py` | Opens on the closes' price basis, and weights aligned and normalized |
| `_build_marks`, `_series_kind`, `_future_layout` | `utils/marks.py` | Build a backtest's marks and classes, decide daily against intraday, and lay out the future marks |
| `_bar_spans`, `_daily_sessions` | `utils/calendar.py` | Map bars to sessions, and session opens and closes for daily dates |
| `_resolve_sims`, `_label_dates`, `_draw_paths`, `_add_colorbar` | `plotting/paths.py` | Read the plot input, axis labels, one fan of paths, and the colorbar |
| `_escape`, `_number`, `_slug`, `_interval_table`, `_results_section`, `_validate` | `reporting/pdf.py` | Escape text for LaTeX, format table values, name the file, build the tables, and check the report dictionary |
| `_parse_interval`, `_periods_per_year`, `_describe_bar` | `utils/periods.py` | Parse any accepted [`interval`](#interval) spelling, look up its annualization factor, and name the bar size in errors |

## How the simulators work

The three simulators all work the same way. They turn the history into log
returns, draw one log return for each future mark, and add the draws up into
paths. The only difference between them is how a draw is made: the bootstrap
resamples past returns, the parametric simulator draws from a fitted
distribution, and the regime-switching simulator draws from whichever market
regime a hidden chain is in at that moment. This section explains each model
in plain words first, then gives its equations, then links to the rules in
[Logic](#logic) that pin down the details.

### Notation

Every symbol in this section's math, in order of first use.

| Symbol | Meaning | Values or unit | Parameter |
| --- | --- | --- | --- |
| $r$ | A simple return. | Fraction | data (from `values`) |
| $\ell$ | A log return, $\ln(1+r)$; $\ell_t$ and $\ell'_t$ are the return at mark $t$. In the worked example, the vector of the five returns. | Log units per bar | data (from `values`) |
| $\ell'$ | A log return put on a full bar's footing by the drift-adjusted map under "Short bars"; $\ell' = \ell$ when $\rho = 1$. | Log units per full bar | derived |
| $V_t$ | The value of a path after mark $t$. | Growth multiple | derived; history from `values` |
| $V_0$ | The starting value of a path. | 1 for simulated paths | derived |
| $t$ | The index of marks and returns, along a path, through the history or through one class's returns. | 1, 2, … | data |
| $\tau$ | The clock length of a return. | Clock time, same unit as the bar size | data (labels of `values`) |
| $\Delta$ | The bar size, the length of a full bar. The simulators take no `interval` argument; they infer it. | Clock time | derived (labels of `values` on `calendar`) |
| $\rho$ | The length ratio $\tau/\Delta$ of a return; $\rho_t$ is that of return $t$. | Above 0, at most 1; 1 for full bars, gaps, daily and continuous bars | derived |
| $\hat\sigma$ | An estimated volatility, the standard deviation of log returns. | Log units per bar | fitted |
| $\hat m$ | The regular drift per full bar, the total regular log return over the total length in full bars. | Log units per full bar | fitted |
| $c$, $c_t$ | A return class; $c_t$ is the class of mark $t$. | Gap or regular | `classes` |
| $n_c$ | The number of kept returns in class $c$. | Count | data |
| $f$ | The density of a parametric family, $f(\ell;\theta)$. | Per log unit | `distribution` (`parametric_monte_carlo`) |
| $\theta$, $\hat\theta$, $\hat\theta_c$ | A family's parameters; their maximum-likelihood fit; the fit for class $c$. | Real numbers, 2 to 4 per family | fitted |
| $k$ | The number of free parameters of a model. | Normal 2, Student's t 3, Johnson SU 4; regime models as in [the regime-switching rule](#mc7-regime-switching-hmm) | derived |
| $n$ | The number of observations the criterion counts: one class's returns in Parametric; in Regime-switching, the chain's marks plus the gaps when the gap normal is fitted. | Count | data |
| $T$ | The number of marks the chain runs on, as in $\ell_{1:T}$ and $t = 1, \dots, T-1$. | Count | data |
| $\log L$ | The maximized log-likelihood. | Natural log of a density | fitted |
| BIC, AIC | The information criteria; the lowest score wins. | Score | `criterion` (parametric and regime-switching) |
| $S$ | The number of regimes. | 1, 2 or 3 | `n_regimes` (`regime_switching_monte_carlo`) |
| $s$, $s_t$ | A regime; $s_t$ is the hidden regime at mark $t$. | 1 to S | fitted |
| $i$, $j$, $j'$ | Regimes: $i$ the regime moved from and $j$ the regime moved to in $\pi_i$, $P_{ij}$, $\xi_t(i,j)$ and $\Xi_{ij}$; $j'$ runs over every regime in the row sum of $\hat P_{ij}$. | 1 to S | `n_regimes` |
| $\pi$, $\pi_i$, $\hat\pi$ | The initial regime distribution, its entry for regime $i$, and its fit. | Probabilities summing to 1 | fitted |
| $P$, $P_{ij}$, $\hat P$ | The transition matrix, its entry from regime $i$ to $j$, and its fit; $P_{1\cdot}$ is row 1. | S by S, each row summing to 1 | fitted |
| $\mu_{s,c}$, $\hat\mu_{s,c}$ | The mean log return of regime $s$ for class $c$, per full unit, and its fit. With one class (the worked example) $\mu_s$, and $\mu$ is the vector over regimes. | Log units per full bar | fitted |
| $\sigma^2_{s,c}$, $\hat\sigma^2_{s,c}$ | The variance of regime $s$ for class $c$, per full unit, and its fit; $\sigma^2_s$ and $\sigma^2$ as for the mean. | Squared log units per full bar | fitted |
| $\gamma_t(s)$, $\gamma$, $\gamma_1$ | The probability of regime $s$ at mark $t$ given all the data; $\gamma_1$ is the vector at the first mark. | 0 to 1 | fitted |
| $\xi_t(i,j)$ | The probability of regime $i$ at mark $t$ and $j$ at mark $t+1$ given all the data. | 0 to 1 | fitted |
| $\Xi_{ij}$ | The expected number of transitions from regime $i$ to $j$, $\sum_t \xi_t(i,j)$. | Expected count; all of them sum to T − 1 | fitted |
| $\kappa$ | The sticky weight: pretend stays added to each regime's row of $\hat P$. | 0 or more | `kappa` (`regime_switching_monte_carlo`) |
| $b_t(s)$ | The normal density of mark $t$ under regime $s$. | Per log unit | derived |
| $N_s$ | The expected number of marks in regime $s$, $\sum_t \gamma_t(s)$, in the one-class worked example. | Expected count | fitted |

Conventions: $\sum$ is a sum; $\exp$ and $e$ are the exponential, $e$ being
Euler's number; $\ln$ and $\log$ are both the natural log;
$\mathcal N(\mu, \sigma^2)$ is the normal with mean $\mu$ and variance
$\sigma^2$; $\Pr$ is a probability and $\sim$ means "is drawn from";
$\arg\max$ is the argument that maximizes; $\mathbb 1[\cdot]$ is 1 when its
condition holds and 0 otherwise; $\ell_{1:T}$ is all of
$\ell_1, \dots, \ell_T$. A hat marks an estimate and a prime on a return a
return per full bar. Two letters also keep their usual meaning: in
"Student's $t$" the $t$ names the distribution, not the index, and the $\pi$
in $\sqrt{2\pi\sigma^2_s}$ is the circle constant, not the initial
distribution.

### What all three share

**Log returns and paths.** A simple return $r$ becomes a log return
$\ell = \ln(1+r)$. Log returns add up over time where simple returns multiply,
so a path is the starting value times the exponential of a running sum:

$$
V_t = V_0 \exp\Big(\ell_1 + \dots + \ell_t\Big).
$$

The exponential is always positive, so no simulated path can fall below zero,
however bad the draws. Paths are reported as growth multiples, with
$V_0 = 1$. See [how paths are compounded](#mc4-fitting-scale).

**Gap and regular returns.** On intraday history the move from one session's
last close to the next session's open (the overnight gap) behaves differently
from the move within a bar: it collects the news of a whole night or weekend.
So returns are split into two classes, and each class gets its own model.
Future open marks draw from the gap model, future bar closes from the regular
model. A gap counts as one unit whatever its clock length, so a weekend gap is
not scaled up against a weeknight gap. Daily, longer and continuous histories
have no gaps: every return is regular and there is one model. Returns marked
irregular (a stale price, a missing bar) are left out of both. See
[return classes](#u1-classes), [units](#u2-units), [simulator input](#mc1-input) and
[horizon and future layout](#mc2-horizon-and-future-layout).

**Short bars.** Most regular bars last exactly one bar size, but a few are
shorter, and their returns have to be put on the same footing as the rest
before they are pooled or fitted.

*Why returns scale with time.* Picture the price as a random walk: every
minute adds a small random step, each with the same average and the same
spread, independent of the others. Over a stretch of $\tau$ minutes the steps
add up. The expected move (the drift) is the sum of the averages, so it grows
in proportion to $\tau$. Independent variances add too, so the variance also
grows in proportion to $\tau$, and the volatility, its square root, grows with
$\sqrt\tau$. A bar half as long therefore has half the drift but
$1/\sqrt2 \approx 0.71$ of the volatility, not half of it.

*Where short bars come from.* The New York Stock Exchange trades from 9:30 to
16:00, six and a half hours. Hourly bars start at 9:30, 10:30, …, 15:30, so
the last bar of every session runs from 15:30 to the close: 30 minutes, half a
bar. Half-day sessions, such as the day after Thanksgiving, close at 13:00,
and their last hourly bar (12:30 to 13:00) is 30 minutes as well.

*What goes wrong without a correction.* Treated as a full hour, a 30-minute
bar looks calm: its moves are only about 0.71 times as large. With one such
bar in every session, about one hourly return in seven is too small, which
pulls the estimated volatility $\hat\sigma$ down. In the regime-switching model
it can do worse. A batch of small returns that arrives like clockwork once a
session is just what an extra regime can explain, so the fit can invent a
"calm regime" that the chain enters at the last bar of every session and
leaves at the next open: a daily switch that is an artefact of the clock, not
a change in the market.

*The formulas.* Each return's length ratio is

$$
\rho = \frac{\tau}{\Delta},
$$

where $\tau$ is the return's length and $\Delta$ the bar size. So $\rho = 1$ for
full bars, every gap, and every daily or continuous bar, and $\rho < 1$ only
for a short intraday bar ($\rho = 0.5$ for the 30-minute bars above). The
regular drift per full bar is the total regular log return over the total
length, counted in full bars:

$$
\hat m = \frac{\sum_t \ell_t}{\sum_t \rho_t}
$$

over the regular returns. Every short historical bar is stretched to a full
bar before it is used: take away the drift it should have had, $\rho\,\hat m$,
scale what is left by $1/\sqrt\rho$ (the volatility ratio from the random
walk), and add a full bar's drift. Every draw for a short future bar is shrunk
back the same way in reverse:

$$
\ell' = \hat m + \frac{\ell - \rho\,\hat m}{\sqrt\rho}
\qquad\text{(history, to a full bar)},
$$

$$
\ell = \rho\,\hat m + \sqrt\rho\,\big(\ell' - \hat m\big)
\qquad\text{(future, back to the bar's length)}.
$$

The two maps are inverses, and with $\rho = 1$ both leave the return untouched.

*A worked number.* Take hourly history whose regular drift is
$\hat m = 0.0002$ (0.02% per full hour), and a 15:30 to 16:00 bar with
$\ell = -0.006$, so $\rho = 0.5$. Its expected drift is
$\rho\,\hat m = 0.0001$, so it moved $-0.0061$ away from it. Stretched to a
full hour,

$$
\ell' = 0.0002 + \frac{-0.006 - 0.0001}{\sqrt{0.5}} = 0.0002 - 0.008627 = -0.008427.
$$

In the pool it counts as a −0.84% hourly return: larger than the raw −0.60%,
which would look too calm, and smaller than the −1.20% of simply doubling it,
which would treat volatility as growing with time rather than with its square
root. A future 30-minute bar that draws $\ell' = -0.008427$ is mapped back to
$\ell = -0.006$.

*Why gaps are not scaled.* A gap is never compared with a bar: each gap
belongs to the gap class and is pooled or fitted only with other gaps. Within
that class every gap counts as one unit whatever its clock length, because
overnight news does not arrive in proportion to clock hours: a weekend gap of
65.5 hours is not three or four times as eventful as a weeknight gap of 17.5
hours. So every gap has $\rho = 1$.

See [units](#u2-units) and [the bootstrap rule](#mc3-bootstrap).

### Bootstrap (`nonparametric_monte_carlo`)

**Intuition.** The future looks like the past, shuffled. Each future return is
one of the historical returns of its class, picked at random with
replacement, independently of every other pick. Nothing is assumed about the
shape of the distribution: the history's own tails, skew and crashes carry
over exactly as heavy as they were. That cuts both ways. The tails are only
as good as the history, so it has to be long enough to contain tail events,
and no draw can ever be larger or smaller than the largest or smallest
return already seen, whereas a fitted Student's $t$ or Johnson SU can
extrapolate past them. And since the picks are independent, a calm day is as
likely to follow a crash as any other day, so the volatility clustering of
real markets is lost.

**Equations.** With $n_c$ kept returns in class $c$, each stretched to a full
bar as above, a future mark of class $c$ draws

$$
\Pr\big(\ell' = \ell'_t\big) = \frac{1}{n_c}, \qquad t = 1, \dots, n_c,
$$

and a short future bar is shrunk back by the map above.

**Rules.** [The bootstrap rule](#mc3-bootstrap) (the pools and the short-bar maps) and
[minimum sample](#mc6-minimum-sample) (how many returns each class needs).

### Parametric (`parametric_monte_carlo`)

**Intuition.** Each class's returns are independent draws from one smooth
distribution, and the job is to find which distribution. Three families are
fitted by maximum likelihood: the normal (thin tails), Student's $t$ (fat
tails) and Johnson SU (fat tails and skew). An information criterion picks
the one that fits best for its number of parameters. Unlike the bootstrap, a
fitted distribution can produce returns beyond anything in the history, which
matters for tail risk. Like the bootstrap, draws are independent, so there is
no volatility clustering.

**Equations.** For a family with density $f(\ell;\theta)$ and $k$ parameters
(normal 2, $t$ 3, Johnson SU 4), the fit on a class's $n$ returns is

$$
\hat\theta = \arg\max_\theta \sum_{t=1}^{n} \ln f\big(\ell'_t;\theta\big),
\qquad
\log L = \sum_{t=1}^{n} \ln f\big(\ell'_t;\hat\theta\big),
$$

and the family with the lowest score wins:

$$
\text{BIC} = -2\log L + k\ln n, \qquad \text{AIC} = -2\log L + 2k.
$$

The $k\ln n$ term charges each extra parameter, so a fatter-tailed family has
to earn its place with a clearly higher likelihood. A future mark of class $c$
draws $\ell' \sim f(\cdot\,;\hat\theta_c)$, and a short future bar is shrunk back
by the map above.

**Rules.** [Fitting scale](#mc4-fitting-scale) (fitting on log returns),
[family selection](#mc5-parametric-family-selection) (the families and the criterion) and
[minimum sample](#mc6-minimum-sample) (minimum sample, and at least $k$ returns per family).

### Regime-switching (`regime_switching_monte_carlo`)

**Intuition.** Markets take turns being calm and turbulent, and each mood
lasts a while. The model has $S$ hidden regimes (one to three). In each regime
returns are normal with that regime's own mean and volatility, separately for
gaps and regular bars. Which regime the market is in is never observed; it
follows a Markov chain that, at every mark, stays put or switches with fixed
probabilities. Because regimes persist, a turbulent stretch tends to be
followed by more turbulence, which gives the volatility clustering the other
two simulators lack. The regimes are found from the data by the EM algorithm:
guess the parameters, work out how likely each mark is to belong to each
regime (E-step), refit each regime's parameters from the marks weighted by
those probabilities (M-step), and repeat until the fit stops improving.

**Equations.** The chain starts in regime $i$ with probability $\pi_i$ and
steps by the transition matrix $P$:

$$
\Pr\big(s_{t+1} = j \mid s_t = i\big) = P_{ij}.
$$

Given regime $s$, a mark of class $c$ and length ratio $\rho$ is normal, with
mean and variance per full unit scaled by $\rho$:

$$
\ell_t \mid s_t = s \;\sim\; \mathcal N\big(\rho_t\,\mu_{s,c},\ \rho_t\,\sigma^2_{s,c}\big).
$$

So $\rho_t$ enters by scaling the regime's mean and variance down to the
bar's length, which tells the model to expect a 30-minute bar to be calmer
by exactly the random-walk amount, so it needs no extra regime to explain it
(see "Short bars" under [What all three share](#what-all-three-share)).

The E-step (forward–backward) gives, for every mark, the probability of each
regime and of each pair of consecutive regimes given all the data:

$$
\gamma_t(s) = \Pr\big(s_t = s \mid \ell_{1:T}\big),
\qquad
\xi_t(i,j) = \Pr\big(s_t = i,\ s_{t+1} = j \mid \ell_{1:T}\big).
$$

The M-step refits every regime and class from the marks of that class,
weighted by $\gamma$, and the transition matrix from the expected transition
counts $\Xi_{ij} = \sum_{t=1}^{T-1} \xi_t(i,j)$:

$$
\hat\mu_{s,c} = \frac{\sum_{t:\,c_t=c} \gamma_t(s)\,\ell_t}{\sum_{t:\,c_t=c} \gamma_t(s)\,\rho_t},
\qquad
\hat\sigma^2_{s,c} = \frac{\sum_{t:\,c_t=c} \gamma_t(s)\,\big(\ell_t - \rho_t\hat\mu_{s,c}\big)^2/\rho_t}{\sum_{t:\,c_t=c} \gamma_t(s)},
$$

$$
\hat P_{ij} = \frac{\Xi_{ij} + \kappa\,\mathbb 1[i=j]}{\sum_{j'} \Xi_{ij'} + \kappa},
\qquad
\hat\pi = \gamma_1.
$$

With every $\rho_t = 1$ these are a weighted mean and a weighted variance. The
sticky weight $\kappa \ge 0$ adds $\kappa$ pretend stays to each regime, which
lengthens regimes; $\kappa = 0$ is plain maximum likelihood (see
[`kappa`](#kappa-regime-switching)). The number of regimes is chosen by BIC or
AIC, as in the parametric simulator, with the parameter count $k$ given in
[the regime-switching rule](#mc7-regime-switching-hmm). A simulated path starts from the regime probabilities at the last
historical mark, steps the chain once per future mark, and draws each mark
from its current regime's normal for that class (or, with `bootstrap=True`,
from the historical returns of that class most likely to belong to that
regime).

**Worked example: one EM iteration.** Five daily log returns, one class, so
every $\rho_t = 1$, and $S = 2$:

$$
\ell = (0.012,\ 0.008,\ -0.035,\ -0.010,\ 0.015).
$$

Start from regime 1 calm and regime 2 turbulent, equally likely at the start,
each staying put with probability 0.8, and $\kappa = 0$:

$$
\pi = (0.5,\ 0.5), \qquad
P_{1\cdot} = (0.8,\ 0.2), \qquad
P_{2\cdot} = (0.2,\ 0.8), \qquad
\mu = (0.01,\ -0.02), \qquad
\sigma^2 = (0.0001,\ 0.0009).
$$

That is volatilities of 1% and 3%. Each mark's density under each regime is
$b_t(s) = \exp\big(-(\ell_t-\mu_s)^2/2\sigma^2_s\big)/\sqrt{2\pi\sigma^2_s}$. For the
first mark,

$$
b_1(1) = \frac{e^{-0.002^2/0.0002}}{\sqrt{2\pi \cdot 0.0001}} = \frac{0.9802}{0.02507} = 39.10,
\qquad
b_1(2) = \frac{e^{-0.032^2/0.0018}}{\sqrt{2\pi \cdot 0.0009}} = \frac{0.5662}{0.07520} = 7.529.
$$

E-step. Forward–backward combines these densities with $\pi$ and $P$ (here
the same as summing the probability of all $2^5 = 32$ regime paths through
each mark):

| $t$ | $\ell_t$ | $b_t(1)$ | $b_t(2)$ | $\gamma_t(1)$ | $\gamma_t(2)$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.012 | 39.10 | 7.529 | 0.8487 | 0.1513 |
| 2 | 0.008 | 39.10 | 8.603 | 0.7292 | 0.2708 |
| 3 | −0.035 | 0.001598 | 11.74 | 0.0004109 | 0.9996 |
| 4 | −0.010 | 5.399 | 12.58 | 0.2033 | 0.7967 |
| 5 | 0.015 | 35.21 | 6.733 | 0.6454 | 0.3546 |
| Sum | | | | 2.427 | 2.573 |

The −3.5% mark is almost certainly turbulent. On its density alone the
−1.0% mark would be turbulent with probability $12.58/(5.399+12.58) = 0.70$;
it sits next to the crash and the chain is sticky, so its neighbours push
that up to 0.80. The 1.5% mark goes the other way: 0.84 calm on its own
density, 0.65 after its turbulent neighbour pulls it. Summing $\xi_t(i,j)$ over the four transitions gives the
expected transition counts, which add up to 4:

$$
\Xi_{11} = 0.8905, \qquad \Xi_{12} = 0.8911, \qquad \Xi_{21} = 0.6879, \qquad \Xi_{22} = 1.531.
$$

M-step. With $N_s = \sum_t \gamma_t(s)$, so $N_1 = 2.427$ and $N_2 = 2.573$:

$$
\hat\mu_1 = \frac{\sum_t \gamma_t(1)\,\ell_t}{N_1} = \frac{0.02365}{2.427} = 0.009745,
\qquad
\hat\mu_2 = \frac{-0.03365}{2.573} = -0.01308,
$$

$$
\hat\sigma^2_1 = \frac{\sum_t \gamma_t(1)\,(\ell_t - \hat\mu_1)^2}{N_1} = \frac{0.0001044}{2.427} = 4.303 \times 10^{-5},
\qquad
\hat\sigma^2_2 = \frac{0.0009829}{2.573} = 3.820 \times 10^{-4},
$$

$$
\hat P_{1\cdot} = \Big(\frac{0.8905}{1.782},\ \frac{0.8911}{1.782}\Big) = (0.4998,\ 0.5002),
\qquad
\hat P_{2\cdot} = \Big(\frac{0.6879}{2.218},\ \frac{1.531}{2.218}\Big) = (0.3101,\ 0.6899),
$$

$$
\hat\pi = \gamma_1 = (0.8487,\ 0.1513).
$$

After one iteration the calm regime has a volatility of 0.6560% and the
turbulent one 1.955%, and the log-likelihood has risen from 12.69 to 14.80.
EM never lowers it. Five marks are far too few to estimate a regime model, so
these numbers only show the mechanics. The package refuses so short a
history: a 2-regime model has 7 parameters, more than the 5 returns, and the
default minimum sample is 30 (see [minimum sample](#mc6-minimum-sample) and
[the regime-switching rule](#mc7-regime-switching-hmm)).

**Rules.** [The regime-switching rule](#mc7-regime-switching-hmm) (the chain, the emissions, EM,
starts, the criterion and simulation) and [minimum sample](#mc6-minimum-sample) (how
many gaps each regime needs).

## Tuning the simulators

What each setting of `regime_switching_monte_carlo` and `parametric_monte_carlo`
changes, when to move it, and how it fails. The numbers below come
from seeded runs on synthetic histories, so take them as examples, not
thresholds. Pass `verbose=True` (the default) to see the score of every
candidate and which regime counts were skipped and why.

**Notation.** Symbols as in [Notation](#notation), plus:

| Symbol | Meaning | Values or unit | Parameter |
| --- | --- | --- | --- |
| $\text{Dir}(1,\dots,1+\kappa,\dots,1)$ | The Dirichlet prior on one row of $P$: weight 1 on each regime and $1+\kappa$ on staying. | A distribution over rows that sum to 1 | `kappa` (`regime_switching_monte_carlo`) |
| $\hat d_i$ | The expected duration of regime $i$, $1/(1-\hat P_{ii})$. | Marks the chain steps on | fitted |
| $\hat N_i$ | The expected steps out of regime $i$, stays included. | Expected count | fitted |
| $\hat L_i$ | The expected switches away from regime $i$. | Expected count | fitted |
| $t_3$, $t_4$, $t_8$ | Student's t with 3, 4 or 8 degrees of freedom, the distribution of the synthetic returns in a seeded run. | A distribution | data |

Also $\propto$ means "proportional to" and $i \to j$ is a move from regime
$i$ to regime $j$.

### `gap_model` (regime-switching)

Applies only when the history has gaps (intraday). On daily or
continuous history there is one class and the two values give the same model.

- **`"regime"`** (default). The regime chain steps at every mark, gaps
  included, and each regime $s$ has its own gap emission
  $\mathcal N(\mu_{s,\text{gap}}, \sigma^2_{s,\text{gap}})$ as well as its regular one. A
  turbulent regime gets large overnight gaps as well as volatile bars. This
  costs parameters, $k = (S-1) + S(S-1) + 4S$, and needs gap data in every
  regime (see [`min_sample`](#min_sample-all-three-simulators)).
- **`"independent"`**. The chain runs on regular marks only, and every gap is
  drawn from one distribution whatever the regime:
  $k = (S-1) + S(S-1) + 2S$, plus 2 for the gap normal. Gap size no longer
  depends on the regime.

| $S$ | `"regime"` with gaps | `"independent"`, `bootstrap=False` | One class, or `"independent"` with `bootstrap=True` |
| --- | --- | --- | --- |
| 1 | 4 | 4 | 2 |
| 2 | 11 | 9 | 7 |
| 3 | 20 | 16 | 14 |

`bootstrap` decides how gaps are drawn. `False` draws from the
fitted normal: the regime's gap normal under `"regime"`, the single gap normal
under `"independent"`. `True` resamples historical gaps: under `"regime"`, the gaps
whose most likely regime is the current one; under `"independent"`, the whole gap
pool. Under `"independent"` with `bootstrap=True` no gap normal is fitted, so
$k$ drops by 2 and the gaps are left out of $n$ and $\log L$. The scores are then
on different data from the `bootstrap=False` ones and can't be compared. On
320 hourly sessions under `"independent"`, the 1-regime BIC was $-18{,}902.4$ with `bootstrap=False`
and $-17{,}060.6$ with `True`.

**When to move it.** Switch to `"independent"` when the history is too short
to give every regime its gaps, or when you have reason to think overnight gaps
don't follow the intraday regime. On a 2-regime hourly history (seeded run),
`"regime"` could only use one regime at 40 and 80 sessions; `"independent"` found
two at both lengths. At 160 and 320 sessions both found two.

**Failure mode.** `"regime"` on a short history quietly falls back to fewer
regimes (the counts that fail `min_sample` are skipped). `"independent"`
draws gaps the same way in calm and turbulent regimes, which understates
how large gaps get in a turbulent regime. Expected durations count the
marks the chain steps on: 8 per full hourly session under `"regime"` (the
gap, then 7 bars), 7 under `"independent"`.

### `kappa` (regime-switching)

A sticky prior on staying in a regime: each row of $P$ has prior
$\text{Dir}(1,\dots,1+\kappa,\dots,1)$, and the fit is the MAP estimate,
$\hat P_{ij} \propto \Xi_{ij} + \kappa\,\mathbb 1[i=j]$, where $\Xi_{ij}$ is the
expected number of $i \to j$ transitions. It is a point estimate, not a
posterior. $\kappa = 0$ is exactly the maximum-likelihood fit. Expected
duration becomes

$$
\hat d_i = \frac{1}{1-\hat P_{ii}} = \frac{\hat N_i + \kappa}{\hat L_i},
$$

where $\hat N_i = \sum_j \Xi_{ij}$ counts the expected steps out of regime $i$
(stays included) and $\hat L_i = \sum_{j\ne i}\Xi_{ij}$ the expected switches
away. Each unit of $\kappa$ acts as one extra observed stay, so it only
matters relative to the marks each regime holds. On 750 daily bars of a
2-regime chain with about 375 bars per regime and a realized mean run of 35.7
bars, the two regimes' expected durations were:

| $\kappa$ | Expected durations (bars) | $\log L$ |
| --- | --- | --- |
| 0 | 49.6, 52.6 | 2242.34 |
| 10 | 51.6, 54.7 | 2242.33 |
| 100 | 69.0, 72.6 | 2241.85 |
| 1000 | 234.8, 241.0 | 2234.28 |

The regime volatilities barely moved (0.643% and 1.991% at $\kappa=0$, 0.644%
and 1.975% at 1000). On 320 hourly sessions under `"regime"`, $\kappa$ of 0, 100
and 1000 gave durations of 110.9 and 94.1, 119.9 and 103.7, and 198.0 and 188.2
marks.

**When to move it.** Raise it when the fitted regimes flicker: durations of a
few marks, with paths switching regime almost every step. Choose $\kappa$ by the
durations it gives, not by $\log L$, which is highest at $\kappa = 0$ and fell
as $\kappa$ grew in the run above.

**Failure mode.** A large $\kappa$ forces persistence onto the data rather than
finding it there. On i.i.d. Student's $t_3$ returns (no regimes at all) the fit
picked 2 regimes: one of 3.76% volatility with an expected duration of 1.08
bars, which is a fat tail rather than a regime. With $\kappa = 100$ that regime's
expected duration rose to 6.0 bars, so it looked like a regime. The selection
score uses the $\log L$ at the MAP estimate, an approximation when $\kappa>0$.

### `criterion` (both simulators)

Chooses the regime count (from 1 to 3, when `n_regimes=None`) and the
parametric family (when `distribution=None`), case-insensitive:
$\text{BIC} = -2\log L + k\ln n$ (default) or $\text{AIC} = -2\log L + 2k$. For the
HMM, $k$ is counted as in the [`gap_model`](#gap_model-regime-switching) table
and $n$ is the number of marks the likelihood covers. For the parametric
simulator, $k$ is the family's parameter count (normal 2, $t$ 3, Johnson SU 4)
and $n$ the class's returns. Going from 1 to 2 regimes on one class adds 5
parameters. On 750 bars, BIC accepts the second regime only if $\log L$ rises by
more than $5\ln(750)/2 \approx 16.5$; AIC needs only 5.

**When to move it.** Keep BIC for forecasting: it adds regimes and tail
parameters only when the data clearly call for them. Try AIC when you suspect BIC is
missing real structure on a short history.

**Failure mode.** AIC overfits: on weakly separated 2-regime daily data
(volatilities 0.8% and 1.2%), it picked 3 regimes in one of three seeds at 1,000
bars. BIC underfits short histories: on the same data it picked 1 regime in
all three seeds at 250 bars and in two of three at 500 and 1,000; at 2,000 bars
both criteria picked 2 in all three. In the parametric simulator on i.i.d. $t_8$
returns, BIC picked the normal in 3 of 15 seeded runs (100–2,000 returns) and AIC
in 2. AIC picked Johnson SU in 5 runs, BIC in 1. Neither criterion can tell
fat tails from regimes (see `kappa` above): check the regimes' expected
durations.

### `min_sample` (all three simulators)

The fewest returns each part of the model must have. Default 30; `0`
disables the count check.

- **Bootstrap and parametric:** per class. A class with $n <$ `min_sample`
  raises `ValueError` naming the class, its $n$ and the history needed. A class
  the forecast needs with $n = 0$, or a parametric class with fewer returns
  than the family's $k$, always raises, even with `min_sample=0`. Example, 10
  hourly sessions: *"The gap class has 9 returns, fewer than
  min_sample=30; about 21 more sessions of history are needed (the history
  spans 10 sessions). Pass a longer history, or a lower min_sample (0
  disables the check)"*
- **Regime-switching, `gap_model="regime"` with gaps:** per regime, the
  expected number of gap marks in it,
  $\sum_{t:\,c_t=\text{gap}} \gamma_t(s) \ge$ `min_sample`. With `bootstrap=True`
  it counts the gaps whose most likely regime is $s$, and a regime with no
  gaps to resample always fails.
- **Regime-switching otherwise** (`"independent"`, or one class): the per-class
  rule above, $n \ge$ `min_sample` for each class.

**Regime count under `n_regimes=None`.** A regime count that fails the rule is
skipped, not raised, so `min_sample` caps the number of regimes the history
can support. Under `"regime"`, $S$ regimes need at least $S \times$ `min_sample`
gaps, 60 for two regimes at the default, and more in practice because
regimes are rarely equal in size. In the seeded hourly run, 79 gaps (80
sessions) were not enough for two regimes: a 2-regime fit had expected gap
counts of 57.7 and 21.3. With `n_regimes` given, a failing count raises with the counts.

**When to move it.** Lower it only when you accept fitting a gap volatility
from fewer returns. At 40 sessions, `min_sample=0` let `"regime"` select 2 regimes,
where a 2-regime fit holds about 25 and 14 expected gaps. Raise it when you want every regime's
gap emission estimated from more data. On short intraday histories,
`gap_model="independent"` is usually the better fix: it needs `min_sample`
gaps in total, not per regime.

**Failure mode.** Set too high, it silently reduces the forecast to one regime
(look for "not eligible" lines in the verbose output). Set too low, it lets
volatility estimates rest on a handful of returns, and with `bootstrap=True`
a regime's pool can hold just a few gaps, each drawn over and over.

### `distribution` (parametric)

The family fitted to each class's log returns (gaps as they are, regular bars
per full bar), by maximum likelihood:

- `None` (default): normal, Student's $t$ and Johnson SU are fitted per class
  and the lowest `criterion` wins; `fit[cls]["Candidates"]` lists every score.
- One `scipy.stats` continuous distribution, e.g. `stats.t`: used for every
  class, with no selection and no printed scores; `verbose=True` prints one
  `Fitted:` line per class with its parameters.
- A dict `{"gap": ..., "regular": ...}`: one family (or `None`, to select) per
  class. It must name every class the history has.

**When to move it.** Fix the family when you need the same model across runs
or portfolios: with `None`, the choice can change between histories of the same
process. On i.i.d. $t_8$ returns, three seeds per length, the pick changed from
seed to seed at every length tried under AIC (100 to 2,000 returns) and up to
500 returns under BIC. Pass a dict to model fat-tailed gaps
and near-normal bars differently.

**Failure mode.** On a short class, selection falls back to the normal and
understates the tails. On 60 i.i.d. $t_4$ returns both criteria picked the
normal; from 250 returns up, both picked $t$. A fitted family with very heavy
tails gives extreme draws, though paths stay positive because they compound
$\ell$. A family with $k$ parameters can't be fitted to fewer than $k$
returns, whatever `min_sample`.

**Reading the printout.** With `verbose=True` each class's line names the
family and its fitted parameters, shapes first, then `loc` and `scale`, e.g.
`regular Selected: t  (df 4.12, loc 0.000912, scale 0.00981)`. They are the
parameters of the log return per unit: per gap, or per full bar (a short
bar's draw is shrunk from it), as in `fit[cls]["Parameters"]`.

## Logic

The rules the package implements, by rule ID. Each rule is enforced by golden
cases in `tests/golden/` that cite its ID, and its Cases line names them: the
cases named after the rule, or every case citing it when none is.
`tests/golden/test_traceability.py` fails if a rule has no case or names a case
that does not cite it. Session $d$ has regular bars $k = 1,\dots,K_d$; $O_{d,1}$ is its first open, $C_{d,k}$ the
close of bar $k$, $V(\cdot)$ the portfolio value at those prices and
$\ell = \ln(1+r)$.

### Backtest

**Notation.** Every symbol in this subsection's math, in order of first use.

| Symbol | Meaning | Values or unit | Parameter |
| --- | --- | --- | --- |
| $t$ | A point in time, the label of a value. | Timestamp on the exchange clock | data (index of `data`) |
| $V(t)$, $V(\cdot)$ | The portfolio value at time $t$, or at the prices of the open or close in the brackets. | Currency, as `br0` | derived |
| $i$ | A symbol (ticker) of the portfolio. | Each symbol in `data` | `w` |
| $n_i$ | The number of shares of symbol $i$ held, fixed at the fill. | Shares | derived (`w`, `br0`) |
| $P_i(t)$ | The last price of symbol $i$ known at $t$. | Price | data (`data`) |
| $d$ | A session, numbered from the fill's. | 1, 2, … | data (`calendar`) |
| $r_{d,k}$ | The simple return of bar $k$ of session $d$. | Fraction | derived |
| $C_{d,k}$ | The close of bar $k$ of session $d$: its price, valued at the bar's close time. | Price | data (`Close` in `data`) |
| $O_{d,1}$ | The open of session $d$'s first bar: its price, valued at the bar's start. | Price | data (`Open` in `data`) |
| $k$ | A regular bar of session $d$. | 1 to $K_d$ | data |
| $g_d$ | The gap return of session $d$, from the last close to the next open. | Fraction | derived |
| $K_d$ | The number of regular bars of session $d$. | Count | data (`calendar`) |
| $O_d$, $C_d$ | In B1.4 and B4.4, the open and close times of session $d$ on the calendar. In B5, $C_d$ is the close price of the bar that opens at $O_{d,1}$. | Timestamp (B1.4, B4.4); price (B5) | `calendar`; data (`Close` in `data`) |
| $V_0$ | The starting value, invested at the fill. | Currency | `br0` |
| $\text{br0}$ | The amount invested. | Currency | `br0` |
| $j$ | A bar of the input, in order. | 1, 2, … | data (index of `data`) |
| $s_\text{first}$, $s_\text{last}$ | The first and last sessions a bar spans. | Sessions | derived (`calendar`) |
| $w_i$ | The weight of symbol $i$; the symbol is held when $w_i > 0$. | Fraction | `w` |
| $\tau$ | The clock length of a return. | Clock time | data (labels) |
| $\Delta$ | The bar size, the length of a full bar. | Clock time | derived (index of `data` on `calendar`) |
| $P_i^\text{last}$ | The last price of symbol $i$ in the data. | Price | data (`data`) |
| $O'_{d,1}$ | The open of session $d$'s first bar on the adjusted basis. | Adjusted price | derived |
| $f_d$ | The adjustment factor of session $d$'s bar, $\text{AdjC}_d / C_d$. | Ratio | derived |
| $\text{AdjC}_d$ | The adjusted close of the bar that opens at $O_{d,1}$. | Adjusted price | data (`Adj Close` in `data`) |
| $c_j$ | The close time of bar $j$. | Timestamp | derived (index of `data` on `calendar`) |
| $\text{now}$ | The current time; a bar that closes after it is in progress. | Timestamp | `now` |

Conventions: $\sum$ is a sum and $\{\dots\}$ a set; square brackets around two
times are the closed interval between them. A prime on a price marks it
adjusted.

#### B1.1 Point in time

The value labelled $t$ is $V(t) = \sum_i n_i P_i(t)$, with $P_i(t)$ the last
price known at $t$. A bar's value is labelled at its close. Intraday bars give
the marks of B1.2–B1.4; daily and longer bars give one mark per bar, at its
close, and close-to-close returns.

Cases: `b1.4_backtest_identity_two_symbols.yaml`, `b1.6_backtest_missing_first_bars.yaml`, `b1.6_missing_first_bar.yaml`, `b4.1_backtest_stale_symbol.yaml`, `u3_backtest_continuous_fill_close.yaml`.

#### B1.2 First bar

The first bar of a session runs open to close: $r_{d,1} = V(C_{d,1})/V(O_{d,1}) - 1$.

Cases: `b1.4_backtest_identity_two_symbols.yaml`, `b1.4_24_5_hourly_week.yaml`, `b2_fill_open_hourly.yaml`.

#### B1.3 Later bars

Later bars run close to close: $r_{d,k} = V(C_{d,k})/V(C_{d,k-1}) - 1$ for $k \ge 2$.

Cases: `b1.4_backtest_identity_two_symbols.yaml`, `b1.4_24_5_hourly_week.yaml`, `b4.1_mid_session_bar_missing_all.yaml`.

#### B1.4 Gap

The gap runs from the last close to the next open:
$g_d = V(O_{d,1})/V(C_{d-1,K_{d-1}}) - 1$, so each session carries $K_d + 1$ marks.
Only where the market was closed: if $O_d = C_{d-1}$ (weeknights on `"24/5"`),
session $d$ has no open mark and no gap, and its first bar is close to close, regular.

Cases: `b1.4_backtest_identity_two_symbols.yaml`, `b1.4_24_5_classify_week.yaml`, `b1.4_24_5_future_layout_weekend.yaml`, `b1.4_24_5_hourly_week.yaml`.

#### B1.5 No opens

Intraday input without opens raises `ValueError`.

Cases: `b1.5_intraday_without_opens_raises.yaml`.

#### B1.6 Missing first bar

If session $d$'s first available bar starts after the session open, the open
mark is labelled at that bar's start and valued at its open; the return from
$C_{d-1,K_{d-1}}$ to it is irregular (kept in values and metrics, excluded from
Monte Carlo fits), and a warning counts such sessions.

Cases: `b1.6_backtest_missing_first_bars.yaml`, `b1.6_missing_first_bar.yaml`.

#### B2 Purchase

$V_0 = \text{br0}$ at $V(O_{1,1})$ for `fill="open"` (default) or at
$V(C_{1,1})$ for `fill="close"`, which has no open mark; session 1 has no gap.
`fill="open"` on data without opens raises `ValueError` naming `fill="close"`.

Cases: `b2_fill_close_daily.yaml`, `b2_fill_open_daily.yaml`, `b2_fill_open_hourly.yaml`, `b2_fill_open_intraday_without_opens_raises.yaml`, `b2_fill_open_without_opens_raises.yaml`.

#### B3 Labels

Every label is a tz-aware timestamp on the exchange clock: the fill at the open
or close it fills, each open mark at its bar's start, each other value at its
bar's close (the session close on a half-day or for a short last bar). Bar $j$
maps to its session span $\{s_\text{first},\dots,s_\text{last}\}$: an open fill
at $s_\text{first}$'s open, each value at $s_\text{last}$'s close, so every bar size
is supported and `fill="open"` works on weekly and longer bars.

Cases: `b3_backtest_weekly_fill_open.yaml`, `b3_daily_fill_open_labels.yaml`, `b3_daily_labels_at_session_close.yaml`, `b3_fill_close_naive_hourly_input.yaml`, `b3_half_day_hourly_labels.yaml`, `b3_hourly_normal_day_labels.yaml`, `b3_short_first_bar_on_the_hour_grid.yaml`, `b3_short_last_bar_label.yaml`, `b3_spring_forward_labels.yaml`, `b3_weekly_bar_session_span.yaml`, `b3_weekly_fill_close_labels.yaml`, `b3_weekly_fill_open_labels.yaml`.

#### B4.1 Stale prices

A return keeps its class only if every held symbol ($w_i > 0$) has its own price
at both marks; otherwise it is irregular (kept in values and metrics, excluded
from Monte Carlo fits, counted in a warning). A return whose span contains a
scheduled mark missing for every held symbol (a mid-session bar, $\tau = 2\Delta$,
or a whole non-holiday session) is irregular too.

Cases: `b4.1_backtest_stale_symbol.yaml`, `b4.1_backtest_bar_missing_all.yaml`, `b4.1_classify_holiday_stays_gap.yaml`, `b4.1_classify_mid_session_mark_missing.yaml`, `b4.1_classify_missing_session.yaml`, `b4.1_holiday_stays_gap.yaml`, `b4.1_mid_session_bar_missing_all.yaml`, `b4.1_missing_session_daily.yaml`, `b4.1_missing_session_hourly.yaml`, `b4.1_mid_session_missing_bar.yaml`, `b4.1_single_symbol_missing_open.yaml`, `b4.1_unheld_stale_symbol_ignored.yaml`.

#### B4.2 No price at the fill

A symbol with no price at the fill is dropped, the other weights are
renormalized, and a warning names it with its first price date.

Cases: `b4.2_late_listing_dropped.yaml`.

#### B4.3 Prices stop early

After its last price $P_i^\text{last}$ a position is cash at 0%, worth
$n_i P_i^\text{last}$ and always fresh; a warning names the symbol and its last
price date.

Cases: `b4.3_prices_stop_early.yaml`.

#### B4.4 Bars outside the session

With a calendar set, intraday bars outside every session (outside $[O_d, C_d]$ on
a session day, or on a non-session day) are dropped before marks are built, with
one warning giving the count; their moves fall into $g_d$. A daily bar dated on a
non-session day raises `ValueError`.

Cases: `b4.4_backtest_daily_bar_on_holiday_raises.yaml`, `b4.4_backtest_extended_hours_in_gap.yaml`, `b4.4_daily_bar_on_holiday_raises.yaml`, `b4.4_post_market_bars_dropped.yaml`, `b4.4_pre_market_bars_dropped.yaml`, `b4.4_saturday_bars_dropped.yaml`.

#### B5 Adjusted prices

Each open is put on the adjusted basis by its own bar's factor:
$O'_{d,1} = O_{d,1} f_d$ with $f_d = \text{AdjC}_d / C_d$, so a dividend lands in the gap.

Cases: `b5_dividend_lands_in_gap.yaml`.

#### B6 Completed bars only

Bar $j$ with close $c_j$ is kept only if $c_j \le \text{now}$; an in-progress bar
is dropped without a warning.

Cases: `b6_backtest_in_progress_bar_dropped.yaml`, `b6_bar_closing_at_now_kept.yaml`, `b6_daily_half_day_complete.yaml`, `b6_daily_session_in_progress_dropped.yaml`, `b6_in_progress_bar_dropped.yaml`.

### Return classes and units

**Notation.** Every symbol in this subsection's math, in order of first use.

| Symbol | Meaning | Values or unit | Parameter |
| --- | --- | --- | --- |
| $d$ | A session. | 1, 2, … | data (`calendar`) |
| $g_d$ | The gap return of session $d$, from the last close to the next open. | Fraction | derived |
| $\tau$ | The length of a return: a clock time in U2; in U3, $\tau = 1$ counts it in grid steps. | Clock time (U2); grid steps (U3) | data (labels of `values`) |
| $\Delta$ | The bar size, the length of a full bar; in continuous mode the grid step. | Clock time | derived (labels of `values`) |
| $\ell$ | A log return, $\ln(1+r)$ of a simple return $r$. | Log units per bar | derived |
| $\ell'$ | A log return scaled to a full bar. | Log units per full bar | derived |
| $t_k$, $t_0$, $t_T$ | Grid points: $t_0$ the first mark, $t_k = t_0 + k\Delta$, $t_T$ the last mark. | UTC timestamps | data (labels of `values`) |
| $k$ | The index of a grid point. | 0, 1, … | data |
| $r_k$ | The simple return from grid point $t_{k-1}$ to $t_k$. | Fraction | derived |
| $V(t_k)$ | The value at grid point $t_k$. | As `values` | data (`values`) |
| $T$ | The index of the last grid point. | Count | data |
| $r_f$ | The annual risk-free rate. | Fraction per year | `risk_free_rate` |
| $\rho$ | The length ratio $\tau/\Delta$ of a return. | 1 in continuous mode | derived |
| $C$ | The number of return classes of the regime model. | 1 in continuous mode | derived |

Conventions: $\sqrt{\cdot}$ is the square root and $\text{d}$ a day. A prime
on a log return marks it per full bar.

#### U1 Classes

Intraday returns are **regular** (bars, open to close or close to close) or
**gap** ($g_d$); daily returns are their own series, one per session.

Cases: `b1.4_backtest_identity_two_symbols.yaml`.

#### U2 Units

One session is one unit (daily) and one gap is one unit, never scaled by clock
length; a regular bar shorter than the grid ($\tau < \Delta$) is scaled as
$\ell' = \ell\sqrt{\Delta/\tau}$.

Cases: `m1_daily_across_holiday_and_weekend.yaml`, `m1_intraday_three_session_report.yaml`.

#### U3 Continuous mode

With `calendar=None` (24/7 assets): grid $t_k = t_0 + k\Delta$, every return
$r_k = V(t_k)/V(t_{k-1}) - 1$ regular with $\tau = 1$, no gaps, open marks or
sessions; `sim_length` counts bars; a grid point missing from the data makes the
return spanning it irregular, with a warning naming `calendar="24/5"` when more
than 10% of returns are irregular. Daily iff $\Delta = 1$ day; future marks
$t_T + k\Delta$; risk figures on returns between the last marks at or before
each 00:00 UTC, annualized with 365 (bars longer than a day: their own returns
with $365\,\text{d}/\Delta$); $\Delta \ge 1$ d keeps every bar return under both
fills, $\Delta < 1$ d always excludes the first UTC day; $r_f/365$, calendar
clock only, years bounded at 00:00 UTC on 1 January; fewer than two 00:00 UTC
boundaries give NaN volatility, Sharpe and Sortino with a warning (M12); the
simulators use one pool with no rescaling ($\rho = 1$) and an HMM with $C = 1$.

Cases: `u3_backtest_continuous_fill_close.yaml`, `u3_backtest_continuous_fill_open.yaml`, `u3_bootstrap_continuous_daily_weekend.yaml`, `u3_bootstrap_continuous_hourly_missing_point.yaml`, `u3_classify_returns_missing_grid_point.yaml`, `u3_is_daily_hourly_false.yaml`, `u3_is_daily_one_day_step.yaml`, `u3_missing_grid_point_at_ten_percent.yaml`, `u3_next_trading_dates_daily.yaml`, `u3_next_trading_dates_hourly.yaml`, `u3_parametric_continuous_layout.yaml`, `u3_regime_continuous_layout.yaml`, `u3_statistical_report_continuous_daily.yaml`, `u3_statistical_report_session_clock_raises.yaml`, `u3_to_session_values_utc_midnight.yaml`, `u3_weekday_only_daily_warns.yaml`.

### Metrics

**Notation.** Every symbol in this subsection's math, in order of first use.

| Symbol | Meaning | Values or unit | Parameter |
| --- | --- | --- | --- |
| $R_d$ | The return of session $d$, close to close; $R_1$ is the first session's, partial when the series starts mid-session. | Fraction | derived |
| $d$ | A session. | 1, 2, … | data (`calendar`) |
| $g_d$ | The gap return of session $d$. | Fraction | derived |
| $r_{d,k}$ | The simple return of regular bar $k$ of session $d$. | Fraction | derived |
| $k$ | A regular bar of session $d$. | 1, 2, … | data |
| $R_t$ | In continuous mode, the return of UTC day $t$. | Fraction | derived |
| $t$ | In M1, a UTC day ($R_t$, $D_t$). In M4, $t_0$ and $t_T$ are the times of the first and last marks. In M6, a mark ($V(t)$). | Day (M1); timestamp (M4, M6) | data (labels of `values`) |
| $V$ | The portfolio value, at a mark or time; $V_0$ at the first mark and $V_T$ at the last. | As `values` | data (`values`) |
| $D_t$ | The last mark at or before the 00:00 UTC that starts day $t$. | Timestamp | derived |
| $r_f$ | The annual risk-free rate. | Fraction per year | `risk_free_rate` |
| CAGR | The compound annual growth rate. | Fraction per year | derived (`cagr_clock`) |
| $T$ | The index of the last mark. | Count | data |
| $Y$ | The length of the series in years. | Years | derived (`cagr_clock`) |
| $N$ | The number of sessions after the fill. | Count | data (`calendar`) |
| $\hat\sigma(\cdot)$ | The sample standard deviation of a series. | As the series | derived |
| $r'$ | A bar return scaled to a full bar, $\exp(\ell/\sqrt\rho) - 1$. | Fraction per full bar | derived |
| $\ell$ | A log return, $\ln(1+r)$ of a bar return. | Log units per bar | derived |
| $\Delta$ | The bar size, the length of a full bar. | Clock time | `interval` (`performance_report`) |
| $\tau$ | The clock length of a bar. | Clock time | data (labels of `values`) |
| $\rho$ | The length ratio $\tau/\Delta$ of a bar. | Above 0, at most 1 | derived |
| $\ell'$ | A log return put on a full bar by the location–scale map with drift $\hat m$. | Log units per full bar | derived |
| $\hat m$ | The regular drift per full bar, the total regular log return over the total length in full bars. | Log units per full bar | fitted |
| MDD | The max drawdown. | Fraction, 0 to 1 | derived |
| $s$ | A mark at or before $t$, in the running high. | Timestamp | data (labels of `values`) |
| $e_d$ | The excess return of session $d$. | Fraction per session | derived (`risk_free_rate`) |
| DD | The downside deviation. | Fraction per session | derived |
| $n$ | The number of sessions in the risk series. | Count | data |
| $\bar e$ | The mean excess return. | Fraction per session | derived |
| $R_y$ | The return of calendar year $y$. | Fraction | derived |
| $y$ | A calendar year. | Year | data (labels of `values`) |

Conventions: $\prod$ is a product and $\sum$ a sum; $\exp$ is the
exponential; $\max$ and $\min$ are the largest and smallest; $+\infty$ is
positive infinity; $\text{days}$ counts calendar days. A bar marks a mean, a
hat an estimate, and a prime a return per full bar.

#### M1 Return series for risk figures

Intraday volatility, Sharpe and Sortino use session returns
$R_d = (1+g_d)\prod_k (1+r_{d,k}) - 1$, annualized with 252; daily data its own
returns with 252; weekly and monthly bars their own returns with 52 and 12, and
the session clock of CAGR counts the sessions they span. In continuous mode
(U3): $R_t = V(D_t)/V(D_{t-1}) - 1$ with $D_t$ the last mark at or before 00:00 UTC,
annualized with 365; daily input uses its own returns with 365.

Cases: `m1_continuous_daily_input.yaml`, `m1_continuous_daily_returns_labels.yaml`, `m1_continuous_utc_day_returns.yaml`, `m1_continuous_weekly_bars.yaml`, `m1_daily_across_holiday_and_weekend.yaml`, `m1_daily_fill_open_report.yaml`, `m1_intraday_three_session_report.yaml`, `m1_weekly_fill_open_report.yaml`.

#### M2 First session

Risk figures use full close-to-close sessions only: a partial $R_1$ is excluded
(always, except `fill="close"` on daily and longer bars); total return, CAGR and drawdown start at br0.

Cases: `m2_continuous_hourly_fill_close.yaml`, `m2_continuous_hourly_fill_open.yaml`, `m2_daily_fill_close_keeps_first_return.yaml`, `m2_intraday_fill_close_same_risk.yaml`, `m2_statistical_report_mid_session_start.yaml`, `m2_weekly_fill_close_keeps_first_return.yaml`.

#### M3 Risk-free rate

Excess session return: $R_d - r_f/252$.

Cases: `m3_risk_free_rate_daily.yaml`.

#### M4 Time and CAGR

$\text{CAGR} = (V_T/V_0)^{1/Y} - 1$ with $Y = N/252$, $N$ the sessions after the
fill, a partial first or last session counting whole (`cagr_clock="session"`,
default), or $Y = (t_T - t_0)/365.25\,\text{days}$
(`"calendar"`); Total Years follows the same clock, and `statistical_report`
needs `dates` for `"calendar"`.

Cases: `m4_calendar_clock_daily.yaml`, `m4_session_clock_partial_last_session.yaml`, `m4_session_clock_raises_continuous.yaml`, `m4_simulated_calendar_clock_from_last_mark.yaml`, `m4_statistical_report_calendar_clock_with_dates.yaml`, `m4_statistical_report_calendar_without_dates_raises.yaml`.

#### M5 Per-class volatility

"Gap Volatility" $= \hat\sigma(g_d)$ and "Bar Volatility" $= \hat\sigma(r')$ with
$r' = \exp\big(\ell\sqrt{\Delta/\tau}\big) - 1$, per unit and unannualized,
irregular returns excluded, NaN on daily data. Two conventions for short bars
($\rho = \tau/\Delta$): Bar Volatility keeps this pure $\sqrt\rho$ normalization,
$r' = \exp(\ell/\sqrt\rho) - 1$; the simulators use the location–scale map of
MC3, $\ell' = \hat m + (\ell - \rho\hat m)/\sqrt\rho$. They differ in $\hat\sigma$ by
about $2\cdot10^{-6}$ relative.

Cases: `m5_continuous_daily_bar_volatility_nan.yaml`.

#### M6 Max drawdown

$\text{MDD} = \max_t \big(1 - V(t)/\max_{s \le t} V(s)\big)$ over every mark, open marks included.

Cases: `m1_continuous_utc_day_returns.yaml`, `m1_daily_fill_open_report.yaml`, `m1_intraday_three_session_report.yaml`, `m2_statistical_report_mid_session_start.yaml`, `m7_statistical_report_infinite_sortino_path.yaml`.

#### M7 Sortino

$e_d = R_d - r_f/252$, $\text{DD} = \sqrt{\sum_d \min(e_d,0)^2/n}$, Sortino
$= \bar e\sqrt{252}/\text{DD}$; with $\text{DD} = 0$ it is $+\infty$ if
$\bar e > 0$ and NaN if $\bar e = 0$. Warnings, one line each: "Sortino ratio
is infinite: none of the {n} sessions is below the risk-free rate"; "Sortino
ratio is NaN: all {n} sessions return exactly the risk-free rate".
`statistical_report`: the Sortino mean across paths is NaN when any path is
$+\infty$; the median and range use an inf-safe percentile; its warning:
"Sortino ratio is infinite on {k} of {N} paths ({p:.1%}); its mean is NaN".

Cases: `m7_flat_series_sortino_nan.yaml`, `m7_no_down_session_infinite.yaml`, `m7_statistical_report_infinite_sortino_path.yaml`.

#### M8 Yearly returns

$R_y = V(\text{last mark of } y)/V(\text{last mark of } y-1) - 1$, the first year
from br0; a year is full iff the series covers its first and last sessions.

Cases: `m8_continuous_year_bounded_at_utc_new_year.yaml`, `m8_partial_first_year.yaml`.

#### M9 Total loss

At $V_T = 0$, $\text{CAGR} = -100\%$; returns after ruin ($0/0$) are excluded from
every risk figure (the return helpers keep them as NaN rows, M10), and
ruined paths stay in `statistical_report` summaries with a warning.

Cases: `m9_statistical_report_ruined_path.yaml`, `m9_total_loss.yaml`, `m9_total_loss_mid_history_helpers.yaml`.

#### M10 Return helpers

`calculate_period_returns` gives returns between consecutive marks (deprecated
alias `calculate_daily_return`); `calculate_session_returns` gives $R_d$ (M1).
Rows after a total loss ($0/0$) stay NaN, so the length depends only on the
input and the calendar: `len(values) - 1` for `calculate_period_returns`, one
per full session (M2) for `calculate_session_returns`.

Cases: `m10_daily_return_alias_warns.yaml`, `m10_period_returns.yaml`, `m10_session_returns_continuous.yaml`, `m10_session_returns_continuous_labels.yaml`, `m10_session_returns_intraday.yaml`.

#### M11 Lower/Upper wording

Lower/Upper are percentiles across paths under the fitted model, worded
"95% range across paths"; keys unchanged.

Cases: `m11_statistical_report_header_wording.yaml`.

#### M12 Degenerate series

Fewer than two marks: `performance_report` and `statistical_report` raise `ValueError`.
In continuous mode (U3), fewer than two 00:00 UTC boundaries give NaN volatility,
Sharpe and Sortino, with a warning.

Cases: `m12_continuous_one_day_boundary.yaml`, `m12_single_mark_raises.yaml`, `m12_statistical_report_continuous_one_boundary.yaml`, `m12_statistical_report_single_mark_raises.yaml`.

### Monte Carlo

**Notation.** Symbols as in [Notation](#notation), plus:

| Symbol | Meaning | Values or unit | Parameter |
| --- | --- | --- | --- |
| $R_d$ | The return of session $d$, close to close. | Fraction | derived |
| $d$ | A session; in MC2, the session the history ends in. | Sessions | data (`calendar`) |
| $H$ | The horizon: sessions on intraday history, bars on daily and longer. | Count | `sim_length` |
| $C_{d+H-1}$ | The close of the horizon's last session or bar. | Timestamp | derived (`calendar`) |
| $s'$ | Any regime, in the arg max over regimes. | 1 to S | `n_regimes` |
| $\mu_\text{gap}$, $\sigma^2_\text{gap}$ | The mean and variance of the one gap normal under `gap_model="independent"` with `bootstrap=False`, whatever the regime. | Log units; squared log units | fitted |
| $\text{Dir}$ | The Dirichlet distribution, the prior on one row of $P$. | A distribution over rows that sum to 1 | `kappa` |
| $J$ | The objective EM maximizes, the log-likelihood plus the prior's log weight on staying. | Natural log | fitted |
| $N_{s,c}$ | The expected number of marks of class $c$ in regime $s$. | Expected count | fitted |
| $T$ | The number of marks the chain runs on, as in $\ell_{1:T}$ and $t = 1, \dots, T-1$; $s_T$ is the regime at the last historical mark. | Count | data |

Conventions: as in [Notation](#notation), plus: $\#\{\dots\}$ is the number of
elements of a set and $\max$ the largest; $\text{gap}$ and $\text{reg}$ name
the gap and regular classes.

#### MC1 Input

Simulators take `values` and optional `classes` in {"gap", "regular", "irregular"};
`classify_returns` infers them from labels. A series is daily if it has at most
one mark per session, at its close, plus an optional fill mark; one mark per session
not at the close raises `ValueError` naming `to_session_values`;
`to_session_values` collapses intraday values to session closes, whose returns are $R_d$.
A session whose last mark is before its close (missing final bar(s)) has its last
value labelled at the close, with one warning: "{k} session(s) end before their
close, first {date}: each one's last value is labelled at its close" ({k} such
sessions, {date} the first one's date as YYYY-MM-DD). An intraday series' last
session whose close is after `now` (keyword, default the current time) is still in
progress and is left out entirely, with no relabel and no warning; the relabel and
its warning apply only to sessions whose close is at or before `now`.

Cases: `mc1_bootstrap_midday_marks_raises.yaml`, `mc1_bootstrap_explicit_classes.yaml`, `mc1_bootstrap_intraday_without_opens_raises.yaml`, `mc1_bootstrap_plain_array.yaml`, `mc1_classify_returns_fill_close.yaml`, `mc1_classify_returns_late_open_mark.yaml`, `mc1_classify_returns_midday_marks_raises.yaml`, `mc1_classify_returns_no_open_marks_raises.yaml`, `mc1_classify_returns_open_marks.yaml`, `mc1_is_daily_fill_close_daily_output.yaml`, `mc1_is_daily_fill_open_daily_output.yaml`, `mc1_is_daily_hourly_intraday.yaml`, `mc1_is_daily_midday_marks.yaml`, `mc1_is_daily_midnight_dates.yaml`, `mc1_is_daily_open_and_close_each_session.yaml`, `mc1_parametric_midday_marks_raises.yaml`, `mc1_parametric_intraday_without_opens_raises.yaml`, `mc1_regime_midday_marks_raises.yaml`, `mc1_regime_intraday_without_opens_raises.yaml`, `mc1_to_session_values_backtest_result.yaml`, `mc1_to_session_values_ended_session_relabelled.yaml`, `mc1_to_session_values_full_sessions_no_warning.yaml`, `mc1_to_session_values_hourly.yaml`, `mc1_to_session_values_live_session_dropped.yaml`, `mc1_to_session_values_midday_relabel.yaml`, `mc1_to_session_values_session_closing_at_now_kept.yaml`, `mc1_to_session_values_short_sessions.yaml`.

#### MC2 Horizon and future layout

On intraday history `sim_length` counts sessions, on daily and longer bars; future
marks continue the history's grid on the calendar (gap, then bars), each with its class;
a history ending mid-session $d$ counts it as the first session, so the horizon ends at $C_{d+H-1}$.

Cases: `mc2_bootstrap_constant_mid_session.yaml`, `mc2_daily_history.yaml`, `mc2_hourly_across_holiday_half_day_weekend.yaml`, `mc2_hourly_history_ending_at_close.yaml`, `mc2_mid_session_history_one_session.yaml`, `mc2_mid_session_history_two_sessions.yaml`.

#### MC3 Bootstrap

Gap marks draw $\ell$ from the gap pool and regular marks $\ell'$ from the regular
pool, i.i.d. within class; irregular returns are in neither pool. Short bars use a
location–scale map with $\rho = \frac{\tau}{\Delta}$ and $\hat m = \frac{\sum_t \ell_t}{\sum_t \rho_t}$ over
regular returns: history $\ell' = \hat m + \frac{\ell - \rho\,\hat m}{\sqrt\rho}$, future
$\ell = \rho\,\hat m + \sqrt\rho\,\big(\ell' - \hat m\big)$, so drift scales with $\rho$ and
volatility with $\sqrt\rho$.

Cases: `mc3_bootstrap_class_moments.yaml`, `mc3_bootstrap_constant_daily.yaml`, `mc3_bootstrap_constant_hourly_half_day.yaml`, `mc3_bootstrap_irregular_excluded.yaml`.

#### MC4 Fitting scale

Simulators work on $\ell = \ln(1+r)$; paths are $V_t = V_0 \exp\Big(\ell_1 + \dots + \ell_t\Big)$,
positive by construction. Parametric fits each class on $\ell$ (the regular class on
$\ell'$) and maps short future bars with the regular pool's $\hat m$, as in MC3.

Cases: `mc4_parametric_class_moments.yaml`.

#### MC5 Parametric family selection

Per class, normal, Student's $t$ and Johnson SU are fitted and the lowest
`criterion` wins: $\text{BIC} = -2\log L + k\ln n$ (default) or $\text{AIC} = -2\log L + 2k$;
`fit` reports per class the family, its parameters, the criterion and every candidate's score.

Cases: `mc5_parametric_criterion_aic_lowercase.yaml`, `mc5_parametric_distribution_dict.yaml`, `mc5_parametric_return_fit_deprecated.yaml`, `mc5_parametric_selects_t_and_normal.yaml`.

#### MC6 Minimum sample

All three simulators take `min_sample: int = 30`; `min_sample=0` disables the
count check below, but never the two that always raise.

- **Per-class rule** (bootstrap, parametric). Each modelled class $c$ (one the
  history has returns of or the forecast has marks of) needs
  $n_c \ge$ `min_sample` kept returns, irregular ones excluded; otherwise
  `ValueError` naming the class, its $n_c$ and about how much more history is
  needed.
- **Always raises**, whatever `min_sample`: a class the forecast has marks of
  and the history has no returns of ($n_c = 0$); for parametric, $n_c < k$ for
  the class's $k$-parameter family.
- **HMM, `gap_model="regime"` with gaps.** Only each regime's gap occupancy is
  checked: for every regime $s$, $\sum_{t:\,c_t=\text{gap}} \gamma_t(s) \ge$
  `min_sample` ($\gamma$ of MC7), or with `bootstrap=True` the hard count
  $\#\{t: c_t = \text{gap},\ \arg\max_{s'} \gamma_t(s') = s\} \ge$ `min_sample`.
  Regular cells get no `min_sample` check, only the start rule of MC7
  (expected occupancy at least $\max(2,\ 0.01\,n_c)$ of the class's marks).
- **HMM, `gap_model="independent"`, or one class** (daily, continuous, or
  intraday without gaps) under either `gap_model`: the per-class rule,
  $n_c \ge$ `min_sample`, checked once before any regime count is fitted and
  raising if it fails.
- **HMM, `bootstrap=True`.** A regime with no gap marks (under `"regime"`) or no
  regular marks by most likely regime has an empty pool, so that count fails
  even with `min_sample=0`.
- A regime count that fails the gap-occupancy or empty-pool check is
  ineligible when `n_regimes=None`, and raises `ValueError` with the counts when
  `n_regimes` is given (MC7).

Cases: `mc6_bootstrap_empty_gap_pool_raises.yaml`, `mc6_bootstrap_gap_below_min_sample_raises.yaml`, `mc6_parametric_fewer_than_k_raises.yaml`, `mc6_parametric_regular_below_min_sample_raises.yaml`, `mc6_regime_gap_cell_below_min_sample_raises.yaml`.

#### MC7 Regime-switching (HMM)

`regime_switching_monte_carlo` fits a Gaussian hidden Markov model to the kept
log returns $\ell_t$ (MC4), class $c_t$ known.

- **Chain.** Regimes $s_t \in \{1,\dots,S\}$ with initial distribution $\pi$ and
  one transition matrix $P$, $\Pr\big(s_{t+1} = j \mid s_t = i\big) = P_{ij}$, stepping
  once per mark it runs on: every mark, gaps included, under
  `gap_model="regime"` (default); regular marks only under `"independent"` or
  when the history has no gaps (one class). There is no separate gap transition
  matrix.
- **Emissions.** With $\rho_t = \frac{\tau_t}{\Delta}$ for a regular mark and
  $\rho_t = 1$ for a gap (and in continuous mode),
  $\ell_t \mid s_t = s \;\sim\; \mathcal N\big(\rho_t\,\mu_{s,c},\ \rho_t\,\sigma^2_{s,c}\big)$ for a mark of class $c$;
  $\mu_{s,c}$, $\sigma^2_{s,c}$ are per full unit and the history is not
  pre-normalized. Under `"independent"` gaps ignore the regime: with
  `bootstrap=False` one normal $\mathcal N(\mu_\text{gap}, \sigma^2_\text{gap})$,
  fitted by maximum likelihood to the gap pool (constant gaps raise); with
  `bootstrap=True` the whole gap pool, no normal fitted.
- **Prior.** `kappa: float = 0.0`, finite and $\ge 0$:
  $P_{i\cdot} \sim \text{Dir}(1,\dots,1+\kappa,\dots,1)$ with $1+\kappa$ at
  position $i$. EM maximizes $J = \log L + \kappa \sum_i \log P_{ii}$, a MAP point
  estimate, not a posterior; $\kappa = 0$ is exactly the MLE.
- **EM.** The E-step (scaled forward–backward) gives $\gamma_t(s) = \Pr\big(s_t = s \mid \ell_{1:T}\big)$
  and $\xi_t(i,j) = \Pr\big(s_t = i,\ s_{t+1} = j \mid \ell_{1:T}\big)$. With
  $N_{s,c} = \sum_{t:\,c_t=c} \gamma_t(s)$ and $\Xi_{ij} = \sum_{t=1}^{T-1} \xi_t(i,j)$
  the M-step is
  $\hat\mu_{s,c} = \frac{\sum_{t:\,c_t=c} \gamma_t(s)\,\ell_t}{\sum_{t:\,c_t=c} \gamma_t(s)\,\rho_t}$,
  $\hat\sigma^2_{s,c} = \max\big(\sum_{t:\,c_t=c} \gamma_t(s)(\ell_t - \rho_t\hat\mu_{s,c})^2/\rho_t \big/ N_{s,c},\ 10^{-12}\big)$,
  $\hat P_{ij} = \frac{\Xi_{ij} + \kappa\,\mathbb 1[i=j]}{\sum_{j'} \Xi_{ij'} + \kappa}$,
  $\hat\pi = \gamma_1$; with every $\rho_t = 1$ these are the plain Gaussian HMM
  updates. EM stops when $J$ rises by less than $10^{-6}$ or after 1000 iterations.
- **Starts.** `n_starts` (default 10) random starts: means seeded k-means++
  style on $\ell$ per class, variances at the class variance, $\pi$ and $P$
  uniform; the highest $\log L$ is kept. A start is rejected if anything is
  non-finite, if any cell has $N_{s,c} < \max(2,\ 0.01\,n_c)$, or if any
  $\hat\sigma^2_{s,c}$ is at the floor $10^{-12}$. A count $S$ cannot be fitted if
  a class on the chain has $n_c < 2S$, if $n < k$, or if every start is rejected.
- **Parameters and criterion.** $k = (S-1) + S(S-1)$ plus $2S$ emission
  terms for each class on the chain: $4S$ under `"regime"` with gaps; $2S$, plus 2 for the gap normal under
  `"independent"` with `bootstrap=False` (the same for every $S$, so it cancels
  in regime-count selection), otherwise. $n$ counts the marks the likelihood
  covers (the chain's, plus the gaps when the gap normal is fitted).
  $\text{BIC} = -2\log L + k\ln n$ or $\text{AIC} = -2\log L + 2k$ (MC5), with
  $\log L$ the data log-likelihood at the estimate (at the MAP when
  $\kappa > 0$, an approximation).
- **Eligibility.** With `n_regimes=None`, $S = 1, 2, 3$ are fitted; a count is
  eligible if it can be fitted and passes MC6, and the eligible count with the
  lowest `criterion` is used; if none is eligible, `ValueError` gives each
  count's reason. A given `n_regimes` that cannot be fitted or fails MC6 raises.
- **Simulation.** Each path starts from the filtered $\Pr\big(s_T \mid \ell_{1:T}\big)$ at
  the last historical mark and steps by $P$ once per future mark the chain runs
  on. A mark of class $c$ in regime $s$ draws from
  $\mathcal N(\mu_{s,c}, \sigma^2_{s,c})$ (`bootstrap=False`) or from the history's
  class-$c$ returns whose most likely regime $\arg\max_s \gamma_t(s)$ is $s$
  (`bootstrap=True`, regular returns per full unit,
  $\ell' = \mu_{s,\text{reg}} + \frac{\ell - \rho\,\mu_{s,\text{reg}}}{\sqrt\rho}$); a short future
  bar maps by $\ell = \rho\,\mu_{s,\text{reg}} + \sqrt\rho\,\big(\ell' - \mu_{s,\text{reg}}\big)$.
- **Tuning.** More regimes and `"regime"` gap emissions need more data per cell;
  `min_sample` refuses counts whose regimes see too few gaps; BIC prefers
  fewer regimes than AIC on long histories; larger `kappa` gives longer-lived
  regimes that switch less, at some cost in fit.
- **Limitation.** Irregular returns are dropped before the fit, so the chain
  treats the returns on either side of one as adjacent: one transition where
  the calendar has two. This shortens fitted regime durations by about the
  share of irregular marks, and leaves the means and variances unbiased.

Cases: `mc7_regime_bootstrap_gap_pool_moments.yaml`, `mc7_regime_class_moments_short_bars.yaml`, `mc7_regime_count_by_criterion.yaml`, `mc7_regime_independent_bootstrap_gaps.yaml`, `mc7_regime_independent_normal_gaps.yaml`, `mc7_regime_kappa_sticky.yaml`, `mc7_regime_kappa_zero_switch_rate.yaml`, `mc7_regime_return_fit_deprecated.yaml`.

#### MC8 Future marks

`next_trading_dates` is the single source of future marks: the MC2 layout, open
marks included, with each mark's class.

Cases: `b1.4_24_5_future_layout_weekend.yaml`, `u3_bootstrap_continuous_hourly_missing_point.yaml`, `u3_next_trading_dates_daily.yaml`, `u3_next_trading_dates_hourly.yaml`, `u3_parametric_continuous_layout.yaml`, `u3_regime_continuous_layout.yaml`, `mc2_bootstrap_constant_mid_session.yaml`, `mc2_daily_history.yaml`, `mc2_hourly_across_holiday_half_day_weekend.yaml`, `mc2_hourly_history_ending_at_close.yaml`, `mc2_mid_session_history_one_session.yaml`, `mc2_mid_session_history_two_sessions.yaml`, `mc3_bootstrap_constant_hourly_half_day.yaml`, `m4_simulated_calendar_clock_from_last_mark.yaml`.

---

## License

MIT; see [LICENSE](LICENSE).
