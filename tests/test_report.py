import contextlib
import io
import warnings

import exchange_calendars as xcals
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
    calculate_session_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    performance_report,
    simulate_buy_and_hold,
    statistical_report,
)

NY = "America/New_York"
SESSIONS = xcals.get_calendar("XNYS", start="2024-01-01", end="2026-12-31").sessions


class TestMetrics:
    def test_daily_return(self):
        # M10: the old name still works, with a DeprecationWarning
        values = pd.Series([100.0, 110.0, 99.0])
        with pytest.warns(DeprecationWarning, match="calculate_period_returns"):
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

    def test_sortino_without_downside_is_infinite_with_a_warning(self):
        # No down session gives +inf (was NaN), with the session count
        with pytest.warns(UserWarning, match="^Sortino ratio is infinite: none of the 2 sessions is below"):
            assert calculate_sortino_ratio(pd.Series([0.01, 0.02]), 252) == np.inf

    def test_max_drawdown(self):
        # M6: MDD = max(1 - V / running max), a positive fraction (was negative)
        values = pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])
        assert calculate_max_drawdown(values) == pytest.approx(1 - 90 / 120)

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
    def test_time_is_counted_in_sessions(self):
        # M4: dated closes on 253 XNYS sessions span 252 session closes
        values = compounding(SESSIONS[SESSIONS >= "2025-01-02"][:253], 0.001)
        with pytest.warns(UserWarning, match="Sortino ratio is infinite"):  # it never falls
            report = performance_report(values, interval="1 day")
        assert report["Periods"] == 252
        assert report["Total Years"] == pytest.approx(1.0)
        assert report["Returns"]["CAGR"] == pytest.approx(1.001**252 - 1)
        # Calendar days are informational only
        assert report["Total Days"] == (values.index[-1] - values.index[0]).days

    def test_single_intraday_session_does_not_crash(self):
        # Once divided by a zero-day calendar span. Labels are tz-aware (B3);
        # the partial session counts whole (M4) and there is no full
        # session for the risk figures (M2)
        idx = pd.date_range("2025-03-03 09:30", "2025-03-03 15:55", freq="5min", tz=NY)
        values = compounding(idx, 0.0001)
        report = performance_report(values, interval="5 mins")
        assert report["Total Years"] == pytest.approx(1 / 252)
        assert report["Returns"]["CAGR"] == pytest.approx(1.0001 ** ((len(idx) - 1) * 252) - 1)
        assert len(report["Daily Returns"]) == 0 and np.isnan(report["Risk"]["Annualized Volatility"])

    def test_matches_the_metric_helpers(self, prices):
        values = prices["X"]
        report = performance_report(values, interval="1D", risk_free_rate=0.02)
        r = values.pct_change().dropna()
        assert report["Risk"]["Sharpe Ratio"] == pytest.approx(calculate_sharpe_ratio(r, 252, 0.02))
        assert report["Risk"]["Sortino Ratio"] == pytest.approx(calculate_sortino_ratio(r, 252, 0.02))
        assert report["Risk"]["Max Drawdown"] == pytest.approx(calculate_max_drawdown(values))
        assert report["Risk"]["Annualized Volatility"] == pytest.approx(r.std() * np.sqrt(252))

    def test_display_prints_the_report(self, prices, capsys):
        performance_report(prices["X"], interval="1 day", display=True)
        out = capsys.readouterr().out
        assert "PORTFOLIO PERFORMANCE REPORT" in out and "Sharpe Ratio" in out


class TestStatisticalReport:
    def test_display_prints_every_section(self, capsys):
        rng = np.random.default_rng(1)
        sims = 100 * np.cumprod(1 + rng.normal(5e-4, 0.01, (50, 253)), axis=1)
        dates = pd.bdate_range("2025-01-02", periods=253)
        statistical_report(sims, "1 day", dates=dates, confidence=0.9, display=True)
        out = capsys.readouterr().out
        for text in ("SIMULATED PORTFOLIO STATISTICAL REPORT", "Simulated Paths", "years (",
                     "90% range across paths", "Final Bankroll", "Sortino Ratio", "VaR (90%)", "CVaR (90%)"):
            assert text in out

    def test_one_path_matches_performance_report(self, prices):
        # The vectorized per-path metrics use the same definitions. On XNYS
        # sessions, the session clock (M4) counts one session per bar
        values = pd.Series((prices["X"] / prices["X"].iloc[0]).to_numpy(),
                           index=SESSIONS[SESSIONS >= "2024-01-02"][:400])
        single = performance_report(values, interval="1 day", risk_free_rate=0.01)
        stat = statistical_report(values.to_numpy()[None, :], "1 day", risk_free_rate=0.01)
        per_path = stat["Per Path Metrics"].iloc[0]
        assert per_path["CAGR"] == pytest.approx(single["Returns"]["CAGR"])
        for metric in ("Annualized Volatility", "Downside Volatility",
                       "Sharpe Ratio", "Sortino Ratio"):
            assert per_path[metric] == pytest.approx(single["Risk"][metric]), metric
        # M6: the drawdown is a positive fraction in both reports
        assert per_path["Max Drawdown"] == pytest.approx(single["Risk"]["Max Drawdown"])

    def test_intraday_cagr_counts_sessions(self):
        # 100 five-minute marks spanning a weekend. M1: intraday paths
        # without their marks raise; M4: with them, Y counts the two
        # sessions after the first mark, not bars
        sims = np.tile(np.cumprod(np.r_[1, np.full(100, 1.0001)]), (3, 1))
        dates = pd.date_range("2025-03-07 09:30", periods=78, freq="5min", tz=NY).append(
            pd.date_range("2025-03-10 09:30", periods=23, freq="5min", tz=NY))
        with pytest.raises(ValueError, match="intraday risk figures use session returns"):
            statistical_report(sims, "5 mins")
        # One full session return (Monday's): M7 no down session, and no variance
        with pytest.warns(UserWarning, match="Sortino ratio is infinite|undefined on some paths"):
            with_dates = statistical_report(sims, "5 mins", dates=dates)
        assert with_dates["Total Years"] == pytest.approx(2 / 252)
        assert with_dates["Returns"].loc["CAGR", "Median"] == pytest.approx(1.0001 ** (100 * 126) - 1)
        assert with_dates["Total Days"] == 3

    def test_tail_risk(self):
        # 101 two-period paths whose total returns are -50%, -49%, ..., +50%
        total = np.linspace(-0.5, 0.5, 101)
        sims = np.column_stack([np.ones(101), np.ones(101), 1 + total])
        # M7: the 50 rising paths have no down session
        # and the flat path no variance
        with pytest.warns(UserWarning) as record:
            tail = statistical_report(sims, "1 day", confidence=0.9)["Tail Risk"]
        assert any("infinite on 50 of 101" in str(w.message) for w in record)
        assert tail["Probability of Loss"] == pytest.approx(50 / 101)
        assert tail["Value at Risk"] == pytest.approx(np.quantile(total, 0.1))
        assert tail["Conditional VaR"] == pytest.approx(total[total <= tail["Value at Risk"]].mean())

    def test_interval_bounds(self):
        total = np.linspace(-0.5, 0.5, 101)
        sims = np.column_stack([np.ones(101), np.ones(101), 1 + total])
        with pytest.warns(UserWarning, match="infinite on 50 of 101|undefined on some paths"):
            row = statistical_report(sims, "1 day", confidence=0.9)["Returns"].loc["Total Return"]
        assert row["Lower"] == pytest.approx(np.quantile(total, 0.05))
        assert row["Upper"] == pytest.approx(np.quantile(total, 0.95))
        assert row["Median"] == pytest.approx(0.0)

    def test_path_ending_at_zero_is_kept_with_cagr_minus_100(self):
        # M9: CAGR -100% at a total loss; the ruined path stays in the summary
        sims = np.array([[1.0, 1.1, 1.21], [1.0, 0.5, 0.0]])
        with pytest.warns(UserWarning) as record:
            report = statistical_report(sims, "1 day")
        assert any("1 of 2 paths end at a total loss" in str(w.message) for w in record)
        assert report["Per Path Metrics"]["CAGR"].iloc[1] == -1.0
        assert report["Returns"].loc["CAGR", "Mean"] == pytest.approx((1.1**252 - 1 - 1) / 2)

    @pytest.mark.parametrize("kwargs, message", [
        ({"interval": "3 days"}, "Unsupported interval"),
        ({"confidence": 1.0}, "confidence"),
        ({"dates": pd.bdate_range("2025-01-02", periods=3)}, "dates has 3 entries"),
    ])
    def test_invalid_arguments_raise(self, kwargs, message):
        sims = np.ones((2, 5))
        with pytest.raises(ValueError, match=message):
            statistical_report(sims, **{"interval": "1 day", **kwargs})

    def test_fewer_than_two_marks_raises(self):
        # M12: one mark per path raises; two are enough
        with pytest.raises(ValueError, match="at least two marks"):
            statistical_report(np.ones((2, 1)), "1 day")



def ny(text):
    return pd.Timestamp(text, tz=NY)


def hourly_ohlc(days, seed=3):
    """Hourly OHLC bars (9:30, then 10:00 to 15:00) of one symbol on `days`."""
    index = pd.DatetimeIndex([ny(f"{d} {c}") for d in days for c in ["09:30"] + [f"{h}:00" for h in range(10, 16)]])
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    opens = closes * np.exp(rng.normal(0, 0.005, len(index)))
    return pd.concat({"Open": pd.DataFrame({"A": opens}, index=index),
                      "Close": pd.DataFrame({"A": closes}, index=index)}, axis=1, names=["Price", "Ticker"])


RISK = ("Annualized Volatility", "Downside Volatility", "Sharpe Ratio", "Sortino Ratio",
        "Gap Volatility", "Bar Volatility")


class TestPerformanceReportRules:
    DAYS = ["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-09"]

    def test_keys(self):
        result = simulate_buy_and_hold(hourly_ohlc(self.DAYS), [1.0])
        report = performance_report(result)
        assert "Period Volatility" not in report["Risk"]
        assert {"Gap Volatility", "Bar Volatility"} <= set(report["Risk"])
        assert report["Daily Returns"].equals(calculate_session_returns(result))
        assert report["Period Returns"].equals(calculate_period_returns(result))
        assert list(report["Returns"]["Actual Yearly Returns"].columns) == ["return", "days", "is_full"]

    def test_short_session_does_not_warn_inside_the_report(self):
        # A session missing its last bars: the risk figures use its last
        # value, and the session-values warning stays out of the report
        data = hourly_ohlc(self.DAYS)
        data = data.drop([ny(f"{self.DAYS[2]} 14:00"), ny(f"{self.DAYS[2]} 15:00")])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = simulate_buy_and_hold(data, [1.0])
            report = performance_report(result)
        assert not [w for w in record if "end before their close" in str(w.message)]
        # The short session's return is its last value (14:00) over the previous close
        day = report["Daily Returns"].loc[ny(f"{self.DAYS[2]} 16:00")]
        assert day == pytest.approx(result.values[ny(f"{self.DAYS[2]} 14:00")]
                                    / result.values[ny(f"{self.DAYS[1]} 16:00")] - 1)

    def test_session_in_progress_left_out_of_history_but_not_of_paths(self):
        # Marks far in the future ending mid-session: as a history the last
        # session is still trading and left out; as simulated paths it counts
        days = pd.bdate_range("2099-06-01", periods=3).strftime("%Y-%m-%d")
        hours = ["09:30"] + [f"{h}:00" for h in range(10, 17)]
        labels = pd.DatetimeIndex([ny(f"{d} {h}") for d in days for h in (hours if d != days[2] else hours[:4])])
        values = pd.Series(np.cumprod(np.r_[1.0, np.full(len(labels) - 1, 1.001)]), index=labels)
        # From the open, the partial first session goes: day 2 only, then days 2 and 3
        assert list(calculate_session_returns(values).index) == [ny(f"{days[1]} 16:00")]
        paths = np.vstack([values.to_numpy(), values.to_numpy() ** 2])
        with pytest.warns(UserWarning, match="infinite"):
            report = statistical_report(paths, dates=labels)
        assert report["Daily Returns"].shape == (2, 2)

    def test_intraday_risk_is_the_same_under_both_fills(self):
        # M2: the partial first session is left out under both fills
        data = hourly_ohlc(self.DAYS)
        by_open = performance_report(simulate_buy_and_hold(data, [1.0], fill="open"))
        by_close = performance_report(simulate_buy_and_hold(data, [1.0], fill="close"))
        for key in RISK[:4]:
            assert by_open["Risk"][key] == pytest.approx(by_close["Risk"][key]), key
        pd.testing.assert_series_equal(by_open["Daily Returns"], by_close["Daily Returns"], rtol=1e-12)
        assert len(by_open["Daily Returns"]) == len(self.DAYS) - 1

    def test_intraday_risk_uses_session_returns_annualized_with_252(self):
        result = simulate_buy_and_hold(hourly_ohlc(self.DAYS), [1.0])
        report = performance_report(result, risk_free_rate=0.0252)
        closes = result.values.groupby(result.values.index.normalize()).last()
        sessions = closes.pct_change().dropna()
        assert report["Risk"]["Annualized Volatility"] == pytest.approx(sessions.std(ddof=1) * np.sqrt(252))
        assert report["Risk"]["Sharpe Ratio"] == pytest.approx(
            (sessions - 0.0001).mean() / sessions.std(ddof=1) * np.sqrt(252))

    def test_gap_volatility_uses_the_backtest_classes(self):
        result = simulate_buy_and_hold(hourly_ohlc(self.DAYS), [1.0])
        r = result.values.pct_change().iloc[1:]
        report = performance_report(result)
        assert report["Risk"]["Gap Volatility"] == pytest.approx(r[(result.classes == "gap").to_numpy()].std(ddof=1))

    def test_daily_risk_is_the_same_under_both_fills(self):
        index = pd.DatetimeIndex(self.DAYS)
        rng = np.random.default_rng(5)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        data = pd.concat({"Open": pd.DataFrame({"A": closes * 0.99}, index=index),
                          "Close": pd.DataFrame({"A": closes}, index=index)}, axis=1)
        by_open = performance_report(simulate_buy_and_hold(data, [1.0], fill="open"))
        by_close = performance_report(simulate_buy_and_hold(data, [1.0], fill="close"))
        for key in RISK[:4]:
            assert by_open["Risk"][key] == pytest.approx(by_close["Risk"][key]), key
        assert np.isnan(by_open["Risk"]["Gap Volatility"]) and np.isnan(by_open["Risk"]["Bar Volatility"])
        # Total return counts the first session's open-to-close move only under fill="open"
        assert by_open["Returns"]["Total Return"] != pytest.approx(by_close["Returns"]["Total Return"])

    def test_continuous_mode_both_fills(self):
        # 24/7 hourly data starting off a UTC midnight: the partial first UTC
        # day is left out under both fills, so the risk figures agree
        index = pd.date_range("2025-06-01 05:00", periods=24 * 5, freq="h", tz="UTC")
        rng = np.random.default_rng(7)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(index))))
        data = pd.concat({"Open": pd.DataFrame({"BTC": closes * 0.999}, index=index),
                          "Close": pd.DataFrame({"BTC": closes}, index=index)}, axis=1)
        reports = [performance_report(simulate_buy_and_hold(data, [1.0], calendar=None, fill=fill),
                                      calendar=None, cagr_clock="calendar") for fill in ("open", "close")]
        for key in RISK[:4]:
            assert reports[0]["Risk"][key] == pytest.approx(reports[1]["Risk"][key]), key
        daily = reports[0]["Daily Returns"]
        assert list(daily.index) == list(pd.date_range("2025-06-03", periods=4, freq="D", tz="UTC"))
        assert np.isnan(reports[0]["Risk"]["Gap Volatility"])

    def test_continuous_mode_from_midnight_both_fills(self):
        # M2, U3: data starting at 00:00 UTC; the first UTC day is excluded
        # under both fills, so the risk figures agree
        index = pd.date_range("2025-06-01 00:00", periods=24 * 5, freq="h", tz="UTC")
        rng = np.random.default_rng(8)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(index))))
        data = pd.concat({"Open": pd.DataFrame({"BTC": closes * 0.999}, index=index),
                          "Close": pd.DataFrame({"BTC": closes}, index=index)}, axis=1)
        reports = [performance_report(simulate_buy_and_hold(data, [1.0], calendar=None, fill=fill),
                                      calendar=None, cagr_clock="calendar") for fill in ("open", "close")]
        for key in RISK[:4]:
            assert reports[0]["Risk"][key] == pytest.approx(reports[1]["Risk"][key]), key

    def test_continuous_session_clock_raises(self):
        values = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2025-06-01", periods=3, freq="D", tz="UTC"))
        with pytest.raises(ValueError, match="cagr_clock='calendar'"):
            performance_report(values, calendar=None)

    def test_continuous_fewer_than_two_utc_midnights_warns(self):
        # M12, U3: one boundary, no full UTC day
        values = pd.Series([100.0, 101.0, 102.0],
                           index=pd.date_range("2025-06-01 23:00", periods=3, freq="h", tz="UTC"))
        with pytest.warns(UserWarning, match="fewer than two"):
            report = performance_report(values, calendar=None, cagr_clock="calendar")
        assert all(np.isnan(report["Risk"][k]) for k in ("Annualized Volatility", "Sharpe Ratio", "Sortino Ratio"))
        assert report["Returns"]["Total Return"] == pytest.approx(0.02)

    def test_fewer_than_two_marks_raises(self):
        with pytest.raises(ValueError, match="at least two marks"):
            performance_report(pd.Series([1.0], index=pd.DatetimeIndex(["2025-06-02"])))

    def test_dates_keyword_is_deprecated(self):
        values = pd.Series([100.0, 101.0, 99.0])
        dates = pd.DatetimeIndex(["2025-06-02", "2025-06-03", "2025-06-04"])
        with pytest.warns(DeprecationWarning, match="dates is deprecated"):
            old = performance_report(values, dates=dates)
        new = performance_report(pd.Series(values.to_numpy(), index=dates))
        assert old["Risk"]["Sharpe Ratio"] == pytest.approx(new["Risk"]["Sharpe Ratio"])

    def test_total_loss(self):
        # M9: CAGR -100%; risk figures stop at the -100% session
        values = pd.Series([100.0, 50.0, 0.0, 0.0], index=pd.DatetimeIndex(
            ["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05"]))
        report = performance_report(values)
        assert report["Returns"]["CAGR"] == -1.0
        # The 0/0 session after ruin is kept as NaN and left out of the figures
        daily = report["Daily Returns"]
        assert daily.iloc[:2].tolist() == [-0.5, -1.0] and np.isnan(daily.iloc[2])
        assert len(report["Period Returns"]) == 3 and np.isnan(report["Period Returns"].iloc[2])
        r = pd.Series([-0.5, -1.0])
        assert report["Risk"]["Annualized Volatility"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
        assert report["Risk"]["Max Drawdown"] == 1.0


def printed(report_call):
    """Return the printout of `report_call` and its header lines (between the title and the first section)."""
    with warnings.catch_warnings():
        # The figures' own warnings are tested elsewhere; only the printout is checked here
        warnings.simplefilter("ignore", UserWarning)
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            report_call()
    out = buffer.getvalue()
    lines = out.strip("\n").splitlines()[3:]
    return out, lines[:next(i for i, line in enumerate(lines) if line.startswith("="))]


class TestPrintedPerformanceReport:
    """The header gives calendar years with their calendar days, then the
    returns, on every series kind; each class volatility row prints only if the series has
    that class."""

    def test_daily(self):
        # The 250 XNYS sessions of 2025: 2025-01-02 to 2025-12-31 is 363 days, 363 / 365.25 = 0.99 years
        days = SESSIONS[(SESSIONS >= "2025-01-02") & (SESSIONS <= "2025-12-31")]
        rng = np.random.default_rng(0)
        values = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
        out, header = printed(lambda: performance_report(values, display=True))
        assert header == ["  Total Time         :  0.99 years (363 days)",
                          "  Periods            :   249 returns"]
        assert "Gap Vol" not in out and "Bar Vol" not in out

    @pytest.mark.parametrize("freq", ["W-FRI", "BME"])
    def test_weekly_and_monthly_have_no_class_rows(self, freq):
        index = pd.date_range("2024-01-01", periods=30, freq=freq)
        values = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 30))), index=index)
        out, _ = printed(lambda: performance_report(values, display=True))
        assert "Gap Vol" not in out and "Bar Vol" not in out

    def test_intraday(self):
        # 2025-06-02 09:30 to 2025-06-09 16:00 New York is 7 days 6.5 hours,
        # 7.2708 / 365.25 = 0.02 years; eight marks a session over six sessions
        result = simulate_buy_and_hold(hourly_ohlc(TestPerformanceReportRules.DAYS), [1.0])
        out, header = printed(lambda: performance_report(result, display=True))
        assert header == ["  Total Time         :  0.02 years (7 days)",
                          "  Periods            :    47 returns"]
        assert "Gap Vol (per gap)" in out and "Bar Vol (per bar)" in out

    def test_one_gap_prints_its_row_even_when_its_volatility_is_undefined(self):
        # A close, the next open and one bar: one gap return, so Gap Volatility
        # is NaN (no sample deviation) but the series has a gap
        values = pd.Series([100.0, 101.0, 100.5], index=pd.DatetimeIndex(
            [ny("2025-06-02 16:00"), ny("2025-06-03 09:30"), ny("2025-06-03 10:30")]))
        out, _ = printed(lambda: performance_report(values, interval="1h", display=True))
        assert "Gap Vol (per gap)  :         nan%" in out and "Bar Vol (per bar)" in out

    def test_continuous_intraday(self):
        # 2025-06-01 05:00 to 2025-06-05 04:00 UTC is 4 days 23 hours, 4.9583 / 365.25 = 0.01 years
        index = pd.date_range("2025-06-01 05:00", periods=24 * 5, freq="h", tz="UTC")
        values = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.005, len(index)))),
                           index=index)
        out, header = printed(lambda: performance_report(values, calendar=None, cagr_clock="calendar",
                                                         display=True))
        assert header == ["  Total Time         :  0.01 years (4 days)",
                          "  Periods            :   119 returns"]
        # No gaps in continuous mode; hourly bars are regular
        assert "Gap Vol" not in out and "Bar Vol (per bar)" in out

    def test_continuous_daily(self):
        # 2025-06-01 to 2025-06-10 is 9 days, 9 / 365.25 = 0.02 years
        index = pd.date_range("2025-06-01", periods=10, freq="D", tz="UTC")
        values = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.02, 10))), index=index)
        out, header = printed(lambda: performance_report(values, calendar=None, cagr_clock="calendar",
                                                         display=True))
        assert header == ["  Total Time         :  0.02 years (9 days)",
                          "  Periods            :     9 returns"]
        assert "Gap Vol" not in out and "Bar Vol" not in out

    def test_header_is_the_same_on_both_clocks_and_keeps_the_dict(self):
        # The printed years are calendar years whatever cagr_clock is; the
        # returned Total Years stays on cagr_clock
        days = SESSIONS[(SESSIONS >= "2025-01-02") & (SESSIONS <= "2025-12-31")]
        values = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(days)))),
                           index=days)
        _, by_session = printed(lambda: performance_report(values, display=True))
        _, by_calendar = printed(lambda: performance_report(values, cagr_clock="calendar", display=True))
        assert by_session == by_calendar
        assert performance_report(values)["Total Years"] == pytest.approx(249 / 252)
        for label in ("sessions", "/252", "/365.25", "bars"):
            assert not any(label in line for line in by_session), label
