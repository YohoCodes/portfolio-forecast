import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast.plotting import plot_path_comparison, plot_simulated_paths
from portfolio_forecast.utils import PERIODS_PER_YEAR, next_trading_dates
from portfolio_forecast.utils.periods import _periods_per_year


class TestIntervals:
    @pytest.mark.parametrize("interval, expected", [
        ("1 day", 252), ("1D", 252), ("1d", 252), (pd.Timedelta(days=1), 252),
        ("5 mins", 19656), ("5min", 19656), (pd.Timedelta(minutes=5), 19656),
        ("1 hour", 1764), ("1h", 1764), ("60min", 1764),
        ("1 week", 52), ("1W", 52), ("1 month", 12),
    ])
    def test_spellings_resolve_to_the_table(self, interval, expected):
        assert _periods_per_year(interval) == expected

    @pytest.mark.parametrize("ib, yf", [
        ("1 min", "1m"), ("2 mins", "2m"), ("5 mins", "5m"), ("15 mins", "15m"),
        ("30 mins", "30m"), ("1 hour", "60m"), ("1 hour", "1h"),
        ("1 day", "1d"), ("1 week", "1wk"), ("1 month", "1mo"),
    ])
    def test_yfinance_spellings_match_ib(self, ib, yf):
        assert _periods_per_year(yf) == _periods_per_year(ib)

    def test_bare_m_is_minutes_and_capital_m_is_months(self):
        assert _periods_per_year("1m") == PERIODS_PER_YEAR["1 min"]
        assert _periods_per_year("1M") == PERIODS_PER_YEAR["1 month"]

    @pytest.mark.parametrize("interval, reason", [
        ("7 mins", "7-minute bars"), ("90m", "90-minute bars"),
        ("3 days", "3-day bars"), ("5d", "5-day bars"), ("3mo", "3-month bars"),
        ("banana", "Unrecognized interval unit"),
    ])
    def test_unsupported_bar_size_raises_with_reason(self, interval, reason):
        with pytest.raises(ValueError, match=f"Unsupported interval.*{reason}"):
            _periods_per_year(interval)

    def test_every_table_key_resolves_to_itself(self):
        for key, n in PERIODS_PER_YEAR.items():
            assert _periods_per_year(key) == n


class TestNextTradingDates:
    def test_daily_skips_weekends_and_holidays(self):
        # Thursday 3 July 2025; Friday 4 July is an NYSE holiday
        history = pd.bdate_range("2025-06-23", "2025-07-03")
        out = next_trading_dates(history, 3)
        assert out.tolist() == [pd.Timestamp(d) for d in ("2025-07-07", "2025-07-08", "2025-07-09")]

    def test_interval_can_be_given(self):
        out = next_trading_dates([pd.Timestamp("2025-07-03")], 2, interval="1 day")
        assert out.tolist() == [pd.Timestamp("2025-07-07"), pd.Timestamp("2025-07-08")]

    def test_intraday_continues_into_the_next_session(self):
        # Five-minute bars to Friday's last bar; the next bars open Monday at 9:30
        history = pd.date_range("2025-03-07 09:30", "2025-03-07 15:55", freq="5min", tz="America/New_York")
        out = next_trading_dates(history, 2)
        assert out.tolist() == [pd.Timestamp("2025-03-10 09:30", tz="America/New_York"),
                                pd.Timestamp("2025-03-10 09:35", tz="America/New_York")]

    def test_keeps_the_input_timezone_form(self):
        naive = next_trading_dates(pd.bdate_range("2025-06-02", periods=5), 1)
        aware = next_trading_dates(pd.bdate_range("2025-06-02", periods=5, tz="UTC"), 1)
        assert naive.tz is None and str(aware.tz) == "UTC"

    def test_zero_periods_is_empty(self):
        assert len(next_trading_dates(pd.bdate_range("2025-06-02", periods=5), 0)) == 0

    def test_empty_dates_raise(self):
        with pytest.raises(ValueError, match="empty"):
            next_trading_dates([], 3)

    def test_one_date_without_interval_raises(self):
        with pytest.raises(ValueError, match="at least two dates"):
            next_trading_dates([pd.Timestamp("2025-07-03")], 3)


class TestPlots:
    @pytest.fixture(autouse=True)
    def close_figures(self):
        yield
        plt.close("all")

    @pytest.fixture
    def sims(self):
        rng = np.random.default_rng(0)
        return np.hstack([np.ones((20, 1)), np.cumprod(1 + rng.normal(0, 0.01, (20, 10)), axis=1)])

    def test_fan_chart(self, sims):
        fig, ax = plot_simulated_paths(sims, method="Test")
        assert ax.get_title(loc="left").startswith("Test: 20 Simulated")
        assert ax.get_xlim() == (0, 10)

    def test_date_labels(self, sims):
        dates = pd.bdate_range("2025-06-02", periods=11)
        fig, ax = plot_simulated_paths(sims, dates=dates)
        assert ax.get_xlabel() == "Date"

    def test_wrong_number_of_dates_raises(self, sims):
        with pytest.raises(ValueError, match="dates has 5 entries"):
            plot_simulated_paths(sims, dates=pd.bdate_range("2025-06-02", periods=5))

    def test_comparison_shares_the_y_axis(self, sims):
        fig, axes = plot_path_comparison({"One": sims, "Two": sims * 1.5})
        assert len(axes) == 2
        assert axes[0].get_ylim() == axes[1].get_ylim()


def test_package_does_not_import_ib_or_market():
    # The public package must not depend on the private IB-Wrapper packages
    code = ("import sys, portfolio_forecast.forecast, portfolio_forecast.performance, "
            "portfolio_forecast.plotting, portfolio_forecast.utils; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('IB', 'market')))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
