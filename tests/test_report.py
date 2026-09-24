import numpy as np
import pandas as pd
import pytest
from conftest import compounding

from portfolio_forecast.performance import (
    calculate_actual_yearly_return,
    calculate_cagr,
    calculate_daily_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    performance_report,
    statistical_report,
)


class TestMetrics:
    def test_daily_return(self):
        values = pd.Series([100.0, 110.0, 99.0])
        assert calculate_daily_return(values).tolist() == pytest.approx([0.10, -0.10])

    def test_cagr_counts_bars(self):
        # 252 daily bars of +0.1% is exactly one year
        values = compounding(pd.bdate_range("2025-01-02", periods=253), 0.001)
        assert calculate_cagr(values, 252) == pytest.approx(1.001**252 - 1)

    def test_cagr_of_one_value_raises(self):
        with pytest.raises(ZeroDivisionError):
            calculate_cagr(pd.Series([100.0]), 252)

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
        report = performance_report(values, values.index, "5 mins")
        assert report["Total Years"] == pytest.approx((len(idx) - 1) / 19656)

    def test_matches_the_metric_helpers(self, prices):
        values = prices["X"]
        report = performance_report(values, values.index, "1D", risk_free_rate=0.02)
        r = calculate_daily_return(values)
        assert report["Risk"]["Sharpe Ratio"] == pytest.approx(calculate_sharpe_ratio(r, 252, 0.02))
        assert report["Risk"]["Sortino Ratio"] == pytest.approx(calculate_sortino_ratio(r, 252, 0.02))
        assert report["Risk"]["Max Drawdown"] == pytest.approx(calculate_max_drawdown(values))
        assert report["Risk"]["Annualized Volatility"] == pytest.approx(r.std() * np.sqrt(252))

    def test_display_prints_the_report(self, prices, capsys):
        performance_report(prices["X"], prices.index, "1 day", display=True)
        out = capsys.readouterr().out
        assert "PORTFOLIO PERFORMANCE REPORT" in out and "Sharpe Ratio" in out


class TestStatisticalReport:
    def test_one_path_matches_performance_report(self, prices):
        # The vectorized per-path metrics use the same definitions
        values = prices["X"] / prices["X"].iloc[0]
        single = performance_report(values, values.index, "1 day", risk_free_rate=0.01)
        stat = statistical_report(values.to_numpy()[None, :], "1 day", risk_free_rate=0.01)
        per_path = stat["Per Path Metrics"].iloc[0]
        assert per_path["CAGR"] == pytest.approx(single["Returns"]["CAGR"])
        for metric in ("Annualized Volatility", "Downside Volatility", "Max Drawdown",
                       "Sharpe Ratio", "Sortino Ratio"):
            assert per_path[metric] == pytest.approx(single["Risk"][metric]), metric

    def test_cagr_does_not_depend_on_dates(self):
        # 100 five-minute bars spanning a weekend: once gave 613.8% vs 237.9%
        sims = np.tile(np.cumprod(np.r_[1, np.full(100, 1.0001)]), (3, 1))
        dates = pd.date_range("2025-03-07 09:30", periods=78, freq="5min").append(
            pd.date_range("2025-03-10 09:30", periods=23, freq="5min"))
        without = statistical_report(sims, "5 mins")
        with_dates = statistical_report(sims, "5 mins", dates=dates)
        assert with_dates["Returns"].loc["CAGR", "Median"] == pytest.approx(
            without["Returns"].loc["CAGR", "Median"])
        assert without["Total Days"] is None
        assert with_dates["Total Days"] == 3

    def test_tail_risk(self):
        # 101 two-period paths whose total returns are -50%, -49%, ..., +50%
        total = np.linspace(-0.5, 0.5, 101)
        sims = np.column_stack([np.ones(101), np.ones(101), 1 + total])
        tail = statistical_report(sims, "1 day", confidence=0.9)["Tail Risk"]
        assert tail["Probability of Loss"] == pytest.approx(50 / 101)
        assert tail["Value at Risk"] == pytest.approx(np.quantile(total, 0.1))
        assert tail["Conditional VaR"] == pytest.approx(total[total <= tail["Value at Risk"]].mean())

    def test_interval_bounds(self):
        total = np.linspace(-0.5, 0.5, 101)
        sims = np.column_stack([np.ones(101), np.ones(101), 1 + total])
        row = statistical_report(sims, "1 day", confidence=0.9)["Returns"].loc["Total Return"]
        assert row["Lower"] == pytest.approx(np.quantile(total, 0.05))
        assert row["Upper"] == pytest.approx(np.quantile(total, 0.95))
        assert row["Median"] == pytest.approx(0.0)

    def test_path_ending_at_zero_has_nan_cagr_and_is_left_out(self):
        sims = np.array([[1.0, 1.1, 1.21], [1.0, 0.5, 0.0]])
        report = statistical_report(sims, "1 day")
        assert np.isnan(report["Per Path Metrics"]["CAGR"].iloc[1])
        assert report["Returns"].loc["CAGR", "Mean"] == pytest.approx(1.1**252 - 1)

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


def test_statistical_report_display_prints_every_section(capsys):
    rng = np.random.default_rng(1)
    sims = 100 * np.cumprod(1 + rng.normal(5e-4, 0.01, (50, 253)), axis=1)
    dates = pd.bdate_range("2025-01-02", periods=253)
    statistical_report(sims, "1 day", dates=dates, confidence=0.9, display=True)
    out = capsys.readouterr().out
    for text in ("SIMULATED PORTFOLIO STATISTICAL REPORT", "Simulated Paths", "Calendar Span",
                 "90% interval", "Final Bankroll", "Sortino Ratio", "VaR (90%)", "CVaR (90%)"):
        assert text in out
