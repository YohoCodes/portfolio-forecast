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

The full documentation (concepts, intraday history, the API reference and
the logic behind every figure) is in the
[README on GitHub](https://github.com/YohoCodes/portfolio-forecast#readme).

## Installation

Requires Python 3.10 or later.

```bash
pip install portfolio-forecast
```

Dependencies: numpy, pandas, scipy, matplotlib, exchange_calendars and
python-dateutil. The package does not fetch data itself; the quick start
below downloads prices with yfinance (`pip install yfinance`).
`compile_statistical_reports` also needs a LaTeX distribution with
`pdflatex` on the PATH (TeX Live, MacTeX or MiKTeX); nothing else in the
package uses LaTeX.

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
