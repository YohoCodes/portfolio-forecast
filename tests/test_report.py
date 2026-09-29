import time
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import compounding

from portfolio_forecast.performance import (
    calculate_actual_yearly_return,
    calculate_cagr,
    calculate_daily_return,
    calculate_max_drawdown,
    calculate_period_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    performance_report,
    statistical_report,
)


class TestMetrics:
    def test_period_returns(self):
        values = pd.Series([100.0, 110.0, 99.0])
        assert calculate_period_returns(values).tolist() == pytest.approx([0.10, -0.10])

    def test_cagr_counts_bars(self):
        # 252 daily bars of +0.1% is exactly one year
        values = compounding(pd.bdate_range("2025-01-02", periods=253), 0.001)
        assert calculate_cagr(values, 252) == pytest.approx(1.001**252 - 1)

    def test_cagr_of_one_value_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            calculate_cagr(pd.Series([100.0]), 252)

    # 252 / 2 = 126 and 252 / 3 = 84 are integer exponents, where numpy gives a real
    # number for a negative ratio instead of NaN; 250 / 2 = 125 is odd
    @pytest.mark.parametrize("n_values, periods_per_year", [(3, 252), (4, 252), (3, 250)])
    def test_cagr_is_minus_100_percent_at_zero(self, n_values, periods_per_year):
        # A total loss is -100% at any horizon: 0 ** (1 / years) - 1 = -1
        values = pd.Series([100.0] * (n_values - 1) + [0.0])
        assert calculate_cagr(values, periods_per_year) == -1.0

    @pytest.mark.parametrize("n_values, periods_per_year", [(3, 252), (4, 252), (3, 250)])
    def test_cagr_is_nan_below_zero(self, n_values, periods_per_year):
        values = pd.Series([100.0] * (n_values - 1) + [-10.0])
        assert np.isnan(calculate_cagr(values, periods_per_year))

    def test_sharpe(self):
        r = pd.Series([0.01, -0.02, 0.03, 0.00])
        expected = r.mean() / r.std() * np.sqrt(252)
        assert calculate_sharpe_ratio(r, 252) == pytest.approx(expected)

    def test_sharpe_subtracts_the_per_bar_risk_free_rate(self):
        r = pd.Series([0.01, -0.02, 0.03, 0.00])
        expected = (r.mean() - 0.0252 / 252) / r.std() * np.sqrt(252)
        assert calculate_sharpe_ratio(r, 252, risk_free_rate=0.0252) == pytest.approx(expected)

    def test_sortino_divides_shortfalls_by_every_period(self):
        r = pd.Series([0.02, -0.01, 0.03, -0.03])
        downside = np.sqrt((0.01**2 + 0.03**2) / 4)
        assert calculate_sortino_ratio(r, 252) == pytest.approx(r.mean() / downside * np.sqrt(252))

    def test_sortino_without_downside_is_nan(self):
        assert np.isnan(calculate_sortino_ratio(pd.Series([0.01, 0.02]), 252))

    def test_max_drawdown(self):
        values = pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])
        assert calculate_max_drawdown(values) == pytest.approx(90 / 120 - 1)

    def test_max_drawdown_of_rising_series_is_zero(self):
        assert calculate_max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0


class TestDeprecatedDailyReturn:
    """`calculate_daily_return` still works as an alias until 1.0.0, with a warning."""

    def test_warns_and_matches_period_returns(self, prices):
        values = prices["X"]
        with pytest.warns(DeprecationWarning, match="removed in 1.0.0"):
            got = calculate_daily_return(values)
        pd.testing.assert_series_equal(got, calculate_period_returns(values))

    def test_warning_points_at_the_caller(self, prices):
        with pytest.warns(DeprecationWarning) as record:
            calculate_daily_return(prices["X"])
        assert record[0].filename == __file__

    def test_performance_report_does_not_warn(self, prices, recwarn):
        performance_report(prices["X"], prices.index, "1 day")
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


class TestYearlyReturns:
    # A 1% gain every bar: each full calendar year returns 1.01**bars_in_year - 1,
    # which is only right if the move across the year boundary is counted
    @pytest.mark.parametrize("freq, start, end", [
        ("B", "2019-12-31", "2023-01-03"),
        ("W-FRI", "2019-12-27", "2023-01-06"),
        ("ME", "2019-12-31", "2023-01-31"),
    ], ids=["daily", "weekly", "monthly"])
    def test_counts_the_move_across_each_year_boundary(self, freq, start, end):
        values = compounding(pd.date_range(start, end, freq=freq), 0.01)
        yearly = calculate_actual_yearly_return(values, values.index)
        for year in (2020, 2021, 2022):
            true = values[values.index.year == year].iloc[-1] / values[values.index.year == year - 1].iloc[-1] - 1
            assert yearly.loc[year, "return"] == pytest.approx(true)

    def test_first_year_is_measured_from_its_first_value(self):
        values = compounding(pd.bdate_range("2021-06-01", "2022-03-01"), 0.001)
        yearly = calculate_actual_yearly_return(values, values.index)
        first = values[values.index.year == 2021]
        assert yearly.loc[2021, "return"] == pytest.approx(first.iloc[-1] / first.iloc[0] - 1)

    def test_partial_and_full_years(self):
        values = compounding(pd.bdate_range("2021-06-01", "2023-06-30"), 0.0)
        yearly = calculate_actual_yearly_return(values, values.index)
        assert yearly["is_full"].tolist() == [False, True, False]
        assert yearly.loc[2022, "days"] == 365

    def test_a_year_from_its_first_to_its_last_session_is_full(self):
        # XNYS 2024: Tue 01-02 to Tue 12-31
        values = compounding(pd.bdate_range("2024-01-02", "2024-12-31"), 0.0)
        yearly = calculate_actual_yearly_return(values, values.index)
        assert yearly.loc[2024, "is_full"] and yearly.loc[2024, "days"] == 366

    def test_without_a_calendar_a_year_runs_between_weekdays(self):
        # 2024-01-01 is a Monday: an XNYS holiday, but a weekday
        values = compounding(pd.bdate_range("2024-01-02", "2024-12-31"), 0.0)
        assert not calculate_actual_yearly_return(values, values.index, calendar=None).loc[2024, "is_full"]
        values = compounding(pd.bdate_range("2024-01-01", "2024-12-31"), 0.0)
        assert calculate_actual_yearly_return(values, values.index, calendar=None).loc[2024, "is_full"]


class TestPerformanceReport:
    def test_time_is_counted_in_bars(self):
        values = compounding(pd.bdate_range("2025-01-02", periods=253), 0.001)
        report = performance_report(values, values.index, "1 day")
        assert report["Periods"] == 252
        assert report["Total Years"] == pytest.approx(1.0)
        assert report["Returns"]["CAGR"] == pytest.approx(1.001**252 - 1)
        # Calendar days are informational only
        assert report["Total Days"] == (values.index[-1] - values.index[0]).days

    def test_single_intraday_session_does_not_crash(self):
        # Once divided by a zero-day calendar span
        idx = pd.date_range("2025-03-03 09:30", "2025-03-03 15:55", freq="5min")
        values = compounding(idx, 0.0001)
        report = performance_report(values, values.index, "5 mins", tz="America/New_York")
        # Time is counted in sessions on intraday bars: one daily return, 1 / 252 years
        assert report["Total Years"] == pytest.approx(1 / 252)
        assert report["Returns"]["CAGR"] == pytest.approx(1.0001 ** (77 * 252) - 1)
        # One session gives one daily return, too few for a volatility
        assert len(report["Daily Returns"]) == 1 and np.isnan(report["Risk"]["Annualized Volatility"])

    def test_matches_the_metric_helpers(self, prices):
        values = prices["X"]
        report = performance_report(values, values.index, "1D", risk_free_rate=0.02)
        r = calculate_period_returns(values)
        assert report["Risk"]["Sharpe Ratio"] == pytest.approx(calculate_sharpe_ratio(r, 252, 0.02))
        assert report["Risk"]["Sortino Ratio"] == pytest.approx(calculate_sortino_ratio(r, 252, 0.02))
        assert report["Risk"]["Max Drawdown"] == pytest.approx(calculate_max_drawdown(values))
        assert report["Risk"]["Annualized Volatility"] == pytest.approx(r.std() * np.sqrt(252))

    def test_display_prints_the_report(self, prices, capsys):
        performance_report(prices["X"], prices.index, "1 day", display=True)
        out = capsys.readouterr().out
        assert "PORTFOLIO PERFORMANCE REPORT" in out and "Sharpe Ratio" in out


class TestStatisticalReport:
    def test_display_prints_every_section(self, capsys):
        rng = np.random.default_rng(1)
        sims = 100 * np.cumprod(1 + rng.normal(5e-4, 0.01, (50, 253)), axis=1)
        dates = pd.bdate_range("2025-01-02", periods=253)
        statistical_report(sims, "1 day", dates=dates, confidence=0.9, display=True)
        out = capsys.readouterr().out
        for text in ("SIMULATED PORTFOLIO STATISTICAL REPORT", "Simulated Paths", "Calendar Span",
                     "90% range across paths", "Final Bankroll", "Sortino Ratio", "VaR (90%)", "CVaR (90%)"):
            assert text in out

    def test_intraday_cagr_counts_sessions_not_calendar_days(self):
        # 100 five-minute bars over two sessions spanning a weekend: two daily returns, 2 / 252
        # years. Calendar days would give 3 / 365, bars 100 / 19,656
        sims = np.tile(np.cumprod(np.r_[1, np.full(100, 1.0001)]), (3, 1))
        dates = pd.date_range("2025-03-07 09:30", periods=78, freq="5min").append(
            pd.date_range("2025-03-10 09:30", periods=23, freq="5min"))
        # Both session returns are gains, so no path has a downside and Sortino is undefined
        with pytest.warns(UserWarning, match=r"Sortino Ratio \(3 paths\)"):
            report = statistical_report(sims, "5 mins", dates=dates, tz="America/New_York")
        assert report["Total Years"] == pytest.approx(2 / 252)
        assert report["Returns"].loc["CAGR", "Median"] == pytest.approx(1.0001 ** (100 * 126) - 1)
        assert report["Total Days"] == 3

    def test_intraday_sims_without_dates_raise(self):
        # Per-bar risk figures would count each overnight gap as one bar
        with pytest.raises(ValueError, match="intraday sims need dates"):
            statistical_report(np.ones((2, 5)), "5 mins")

    def test_tail_risk(self):
        # 101 two-period paths whose total returns are -50%, -49%, ..., +50%
        total = np.linspace(-0.5, 0.5, 101)
        sims = np.column_stack([np.ones(101), np.ones(101), 1 + total])
        with pytest.warns(UserWarning, match=r"Sharpe Ratio \(1 path\), Sortino Ratio \(51 paths\)"):
            tail = statistical_report(sims, "1 day", confidence=0.9)["Tail Risk"]
        assert tail["Probability of Loss"] == pytest.approx(50 / 101)
        assert tail["Value at Risk"] == pytest.approx(np.quantile(total, 0.1))
        assert tail["Conditional VaR"] == pytest.approx(total[total <= tail["Value at Risk"]].mean())

    def test_interval_bounds(self):
        total = np.linspace(-0.5, 0.5, 101)
        sims = np.column_stack([np.ones(101), np.ones(101), 1 + total])
        with pytest.warns(UserWarning, match=r"Sharpe Ratio \(1 path\), Sortino Ratio \(51 paths\)"):
            row = statistical_report(sims, "1 day", confidence=0.9)["Returns"].loc["Total Return"]
        assert row["Lower"] == pytest.approx(np.quantile(total, 0.05))
        assert row["Upper"] == pytest.approx(np.quantile(total, 0.95))
        assert row["Median"] == pytest.approx(0.0)

    def test_path_ending_at_zero_has_cagr_minus_100_percent_and_stays_in(self):
        # Path 0: 1.21 over 2 bars -> 1.21 ** (252 / 2) - 1 = 1.1 ** 252 - 1. Path 1: -1.
        sims = np.array([[1.0, 1.1, 1.21], [1.0, 0.5, 0.0]])
        with pytest.warns(UserWarning, match="1 of 2 paths end at a total loss"):
            report = statistical_report(sims, "1 day")
        assert report["Per Path Metrics"]["CAGR"].iloc[1] == -1.0
        assert report["Returns"].loc["CAGR", "Mean"] == pytest.approx((1.1**252 - 1 - 1) / 2)

    @pytest.mark.parametrize("kwargs, message", [
        ({"interval": "7 mins"}, "Unsupported interval"),
        ({"confidence": 1.0}, "confidence"),
        ({"dates": pd.bdate_range("2025-01-02", periods=3)}, "dates has 3 entries"),
    ])
    def test_invalid_arguments_raise(self, kwargs, message):
        sims = np.ones((2, 5))
        with pytest.raises(ValueError, match=message):
            statistical_report(sims, **{"interval": "1 day", **kwargs})

    def test_fewer_than_two_periods_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            statistical_report(np.ones((2, 2)), "1 day")



class TestTotalLoss:
    # Intraday values, two per XNYS session on the New York clock (tz passed):
    #   Jun 2: 100, 110   Jun 3: 50, 0   Jun 4: 0, 0
    # Daily returns run from the first value to each session's last:
    #   110 / 100 - 1 = 0.1, 0 / 110 - 1 = -1, then 0 / 0, left out. n = 2.
    # Downside: (0 + 1) / 2 = 0.5 -> sqrt(0.5 x 252) = sqrt(126) = 11.224972160.
    # Counting the 0 / 0 return in n would give sqrt(252 / 3) = sqrt(84) = 9.165.
    TIMES = ["2025-06-02 10:30", "2025-06-02 16:00", "2025-06-03 10:30",
             "2025-06-03 16:00", "2025-06-04 10:30", "2025-06-04 16:00"]
    VALUES = [100, 110, 50, 0, 0, 0]
    TZ = "America/New_York"

    def _values(self):
        return pd.Series(self.VALUES, index=pd.DatetimeIndex(self.TIMES), dtype=float)

    def test_intraday_daily_returns_stop_at_the_loss(self):
        v = self._values()
        report = performance_report(v, v.index, "1 hour", tz=self.TZ)
        np.testing.assert_allclose(report["Daily Returns"].to_numpy(), [0.1, -1.0])
        assert report["Risk"]["Downside Volatility"] == pytest.approx(np.sqrt(126), rel=1e-12)
        assert report["Returns"]["CAGR"] == -1.0

    def test_intraday_reports_agree_after_a_loss(self):
        v = self._values()
        single = performance_report(v, v.index, "1 hour", tz=self.TZ)
        with pytest.warns(UserWarning, match="1 of 1 paths end at a total loss"):
            paths = statistical_report(v.to_numpy()[None, :], "1 hour", dates=v.index, tz=self.TZ)
        per_path = paths["Per Path Metrics"].iloc[0]
        for key in ("CAGR", "Annualized Volatility", "Downside Volatility", "Sharpe Ratio", "Sortino Ratio"):
            section = single["Returns"] if key == "CAGR" else single["Risk"]
            assert per_path[key] == pytest.approx(section[key], rel=1e-12), key
        # Session to session, 0 / 0 after the loss kept as NaN: 110/100 - 1, 0/110 - 1, 0/0
        np.testing.assert_allclose(paths["Daily Returns"][0], [0.1, -1.0, np.nan], rtol=1e-12)

    def test_warning_counts_losses_and_left_out_paths(self):
        sims = np.array([
            [1.0, 0.5, 0.0, 0.0],    # total loss: CAGR -100%, stays in every summary
            [1.0, 1.1, 1.2, 1.3],    # no losing bar: Sortino undefined
            [1.0, 1.1, 0.9, 1.05],
        ])
        with pytest.warns(UserWarning) as record:
            report = statistical_report(sims, "1 day")
        message = str(record[0].message)
        assert "1 of 3 paths end at a total loss" in message
        assert "Sortino Ratio (1 path)" in message
        assert "CAGR" not in message.split("undefined:")[1]
        assert report["Per Path Metrics"]["CAGR"].iloc[0] == -1.0

    def test_no_warning_when_nothing_to_report(self):
        sims = np.array([[1.0, 1.1, 0.9, 1.05], [1.0, 0.9, 1.1, 0.95]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            statistical_report(sims, "1 day")


@pytest.fixture
def machine_in_new_york(monkeypatch):
    """Set this machine's zone to New York for one test. Tbilisi (UTC+4) matches
    Dubai's offset and UTC shifts no NY label across midnight, so on those machines
    a report that ignored `tz` would still pass."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


class TestTimezone:
    # Golden M06's values: XNYS sessions Mar 2-4 2026, labelled at their closes on
    # Dubai's clock (UTC+4; New York is UTC-5), so each session's last close is
    # 01:00 the next Dubai day. Session last / previous last:
    #   101 / 100 - 1 = 0.01,  99.99 / 101 - 1 = -0.01,  102.9897 / 99.99 - 1 = 0.03
    # Mean 0.01, sample sd sqrt((0 + 0.02^2 + 0.02^2) / 2) = 0.02, x sqrt(252) = 0.317490157328
    TIMES = ["2026-03-02 19:30", "2026-03-02 22:30", "2026-03-03 01:00", "2026-03-03 19:30", "2026-03-03 22:30",
             "2026-03-04 01:00", "2026-03-04 19:30", "2026-03-04 22:30", "2026-03-05 01:00"]
    VALUES = [100, 103, 101, 96, 98, 99.99, 101, 104, 102.9897]

    def test_naive_intraday_times_are_read_in_tz_not_machine_zone(self, machine_in_new_york):
        v = pd.Series(self.VALUES, index=pd.DatetimeIndex(self.TIMES), dtype=float)
        single = performance_report(v, v.index, "1h", tz="Asia/Dubai")
        paths = statistical_report(v.to_numpy()[None, :], "1h", dates=v.index, tz="Asia/Dubai")
        np.testing.assert_allclose(single["Daily Returns"].to_numpy(), [0.01, -0.01, 0.03], rtol=1e-9)
        assert paths["Per Path Metrics"]["Annualized Volatility"].iloc[0] == pytest.approx(0.317490157328, rel=1e-9)
