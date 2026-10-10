"""statistical_report on SimulationResults (M1-M12, U3): each path's
figures follow performance_report's rules, summarized across paths."""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_forecast import SimulationResult
from portfolio_forecast.performance import performance_report, statistical_report
from portfolio_forecast.performance.metrics import calculate_max_drawdown
from portfolio_forecast.performance.report import _layout_origin, _quantile, _range_summary
from portfolio_forecast.utils import next_trading_dates

NY = "America/New_York"
FIGURES = ("CAGR", "Gap Volatility", "Bar Volatility", "Annualized Volatility", "Downside Volatility",
           "Max Drawdown", "Sharpe Ratio", "Sortino Ratio")


def ny(text):
    return pd.Timestamp(text, tz=NY)


def hourly_marks(days):
    """B3 marks of hourly bars starting 9:30: each session's open mark, then
    its bar closes 10:00, 11:00, ..., 16:00."""
    return pd.DatetimeIndex([ny(f"{d} {t}") for d in days
                             for t in ["09:30"] + [f"{h}:00" for h in range(10, 17)]])


def random_paths(n_sims, n_points, seed=0, scale=0.01):
    """Growth-multiple paths with column 0 = 1.0, as the simulators return."""
    rng = np.random.default_rng(seed)
    steps = np.exp(rng.normal(2e-4, scale, (n_sims, n_points - 1)))
    return np.hstack([np.ones((n_sims, 1)), np.cumprod(steps, axis=1)])


def simulation(history, n, n_sims=4, seed=0, calendar="XNYS"):
    """A hand-built SimulationResult on the layout that continues `history`."""
    layout = next_trading_dates(history, n, calendar=calendar)
    paths = random_paths(n_sims, len(layout.dates) + 1, seed)
    return SimulationResult(paths, layout.dates, layout.classes, None)


def path_report(sim, row, origin, **kwargs):
    """performance_report on one path of `sim`, labelled with `origin` then
    the future marks, with the simulation's classes."""
    labels = pd.DatetimeIndex([origin]).append(sim.dates)
    values = pd.Series(sim.paths[row], index=labels)
    classes = pd.Series(sim.classes, index=labels[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return performance_report(values, classes, **kwargs)


def assert_matches_performance_report(sim, origin, **kwargs):
    """Every path's figures equal performance_report's on that path."""
    stat = statistical_report(sim, **kwargs)
    report_kwargs = {k: v for k, v in kwargs.items() if k != "confidence"}
    for row in range(len(sim.paths)):
        single = path_report(sim, row, origin, **report_kwargs)
        per_path = stat["Per Path Metrics"].iloc[row]
        expected = {**single["Returns"], **single["Risk"]}
        for figure in FIGURES:
            assert per_path[figure] == pytest.approx(expected[figure], nan_ok=True), (row, figure)
        np.testing.assert_allclose(stat["Daily Returns"][row], single["Daily Returns"].to_numpy())
        assert stat["Total Years"] == pytest.approx(single["Total Years"])
        assert stat["Total Days"] == single["Total Days"]
    return stat


class TestLayoutOrigin:
    """Column 0's label is the grid point before the first future mark (MC2, MC8)."""

    @pytest.mark.parametrize("history", [
        hourly_marks(["2025-06-02", "2025-06-03"]),                     # ends at a close
        hourly_marks(["2025-06-02", "2025-06-03"])[:-4],                # ends mid-session, 12:00
        hourly_marks(["2025-06-02", "2025-06-03"])[:-7],                # ends at the open mark
        hourly_marks(["2024-11-26", "2024-11-27"])[:-3].append(          # half-day, 13:00 close
            pd.DatetimeIndex([ny("2024-11-29 09:30"), ny("2024-11-29 10:00"), ny("2024-11-29 11:00"),
                              ny("2024-11-29 12:00"), ny("2024-11-29 13:00")])),
        hourly_marks(["2024-11-26", "2024-11-27"]).append(               # mid half-day
            pd.DatetimeIndex([ny("2024-11-29 09:30"), ny("2024-11-29 10:00"), ny("2024-11-29 11:00")])),
    ], ids=["close", "mid-session", "open-mark", "half-day-close", "mid-half-day"])
    def test_intraday(self, history):
        layout = next_trading_dates(history, 2)
        assert _layout_origin(layout.dates) == history[-1]

    def test_24_5_session_without_open_mark(self):
        # B1.4: on "24/5" Tuesday opens at Monday's close, so the first future
        # bar runs from that close
        history = pd.date_range("2025-06-02 00:00", "2025-06-03 00:00", freq="h", tz="UTC")
        layout = next_trading_dates(history, 2, calendar="24/5")
        assert layout.classes[0] == "regular"
        assert _layout_origin(layout.dates, calendar="24/5") == history[-1]

    @pytest.mark.parametrize("history", [
        pd.DatetimeIndex(["2025-06-26", "2025-06-27", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03"]),
        pd.DatetimeIndex([ny("2025-06-30 16:00"), ny("2025-07-01 16:00"), ny("2025-07-03 13:00")]),
        pd.DatetimeIndex(["2025-06-24", "2025-06-26", "2025-06-30", "2025-07-02"]),   # 2-session bars
        pd.DatetimeIndex(["2025-08-18", "2025-08-25", "2025-09-02", "2025-09-08"]),   # weeks, first session
        pd.DatetimeIndex(["2025-06-20", "2025-06-27", "2025-07-03", "2025-07-11"]),   # weeks, last session
        pd.DatetimeIndex(["2025-05-30", "2025-06-30", "2025-07-31"]),                 # months
    ], ids=["daily", "daily-closes", "2-session", "weekly-first", "weekly-last", "monthly"])
    def test_daily_and_longer(self, history):
        layout = next_trading_dates(history, 3)
        assert _layout_origin(layout.dates) == history[-1]

    def test_continuous(self):
        history = pd.date_range("2025-06-01", periods=30, freq="h", tz="UTC")
        layout = next_trading_dates(history, 5, calendar=None)
        assert _layout_origin(layout.dates, calendar=None) == history[-1]

    def test_one_future_mark_needs_the_interval(self):
        history = hourly_marks(["2025-06-02"])[:-1]
        layout = next_trading_dates(history, 1)
        with pytest.raises(ValueError, match="pass interval"):
            _layout_origin(layout.dates)
        assert _layout_origin(layout.dates, interval="1h") == history[-1]


class TestSamePerPathFigures:
    """M1-M9: per path, the figures performance_report gives on that path."""

    def test_daily(self):
        history = pd.bdate_range("2025-06-02", periods=10)
        sim = simulation(history, 30)
        for clock in ("session", "calendar"):
            assert_matches_performance_report(sim, history[-1], risk_free_rate=0.02, cagr_clock=clock)

    def test_daily_closes(self):
        history = pd.DatetimeIndex([ny(f"2025-06-{d:02d} 16:00") for d in (2, 3, 4, 5, 6)])
        sim = simulation(history, 25)
        stat = assert_matches_performance_report(sim, history[-1])
        assert stat["Daily Returns"].shape == (4, 25)  # M2: the first return is a full session
        assert stat["Per Path Metrics"]["Gap Volatility"].isna().all()  # M5: NaN on daily data

    def test_weekly(self):
        history = pd.DatetimeIndex(["2025-06-06", "2025-06-13", "2025-06-20"])
        sim = simulation(history, 12)
        assert_matches_performance_report(sim, history[-1])

    def test_intraday_from_a_close(self):
        history = hourly_marks(["2025-06-02", "2025-06-03"])
        sim = simulation(history, 6)
        stat = assert_matches_performance_report(sim, history[-1], risk_free_rate=0.01)
        # M1: one session return per future session, the first a full one
        assert stat["Daily Returns"].shape == (4, 6)
        assert stat["Total Years"] == pytest.approx(6 / 252)
        assert stat["Per Path Metrics"]["Gap Volatility"].notna().all()

    def test_intraday_mid_session_start(self):
        # History ends Tuesday 12:00; the layout's first session is the rest
        # of Tuesday (MC2). M2: that partial session is left out of the risk
        # figures; M4: it counts whole on the session clock
        history = hourly_marks(["2025-06-02", "2025-06-03"])[:-4]
        sim = simulation(history, 8)
        for clock in ("session", "calendar"):
            stat = assert_matches_performance_report(sim, history[-1], cagr_clock=clock)
        assert stat["Daily Returns"].shape == (4, 7)
        session = statistical_report(sim)
        assert session["Total Years"] == pytest.approx(8 / 252)
        # The full sessions' returns, by hand: Wednesday's close over Tuesday's
        labels = pd.DatetimeIndex([history[-1]]).append(sim.dates)
        tuesday, wednesday = labels.get_loc(ny("2025-06-03 16:00")), labels.get_loc(ny("2025-06-04 16:00"))
        np.testing.assert_allclose(session["Daily Returns"][:, 0],
                                   sim.paths[:, wednesday] / sim.paths[:, tuesday] - 1)

    def test_intraday_short_bars_are_scaled(self):
        # Bars on the clock hour from a 9:30 open: 9:30-10:00 is a half bar,
        # scaled to a full one in logs for Bar Volatility (M5, U2)
        history = hourly_marks(["2025-06-02", "2025-06-03"])
        sim = simulation(history, 5, seed=3)
        assert (next_trading_dates(history, 5).tau < 1).any()
        assert_matches_performance_report(sim, history[-1])

    def test_continuous(self):
        # U3: calendar-day returns ×365, calendar clock, no gaps
        history = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
        sim = simulation(history, 24 * 6, calendar=None)
        stat = assert_matches_performance_report(sim, history[-1], calendar=None, cagr_clock="calendar",
                                                 risk_free_rate=0.03)
        assert stat["Per Path Metrics"]["Gap Volatility"].isna().all()
        with pytest.raises(ValueError, match="cagr_clock='calendar'"):
            statistical_report(sim, calendar=None)

    def test_continuous_fewer_than_two_utc_boundaries_warns(self):
        # M12, U3
        history = pd.date_range("2025-06-01 20:00", periods=3, freq="h", tz="UTC")
        sim = simulation(history, 4, calendar=None)
        with pytest.warns(UserWarning, match="fewer than two"):
            stat = statistical_report(sim, calendar=None, cagr_clock="calendar")
        assert stat["Per Path Metrics"]["Annualized Volatility"].isna().all()

    def test_array_with_dates_classifies_from_them(self):
        history = hourly_marks(["2025-06-02", "2025-06-03"])
        sim = simulation(history, 4)
        dates = pd.DatetimeIndex([history[-1]]).append(sim.dates)
        from_array = statistical_report(sim.paths, "1h", dates=dates)
        from_result = statistical_report(sim)
        pd.testing.assert_frame_equal(from_array["Per Path Metrics"], from_result["Per Path Metrics"])


class TestSummaries:
    def test_drawdown_is_a_positive_fraction(self):
        # M6: same sign and value as performance_report
        sims = np.array([[1.0, 1.2, 0.9, 1.1], [1.0, 0.8, 0.6, 0.7]])
        stat = statistical_report(sims, "1 day")
        expected = [calculate_max_drawdown(pd.Series(row)) for row in sims]
        assert stat["Per Path Metrics"]["Max Drawdown"].tolist() == pytest.approx(expected)
        assert stat["Per Path Metrics"]["Max Drawdown"].tolist() == pytest.approx([0.25, 0.4])

    def test_keys(self):
        # M5: Period Volatility replaced by Gap and Bar Volatility
        stat = statistical_report(simulation(hourly_marks(["2025-06-02"]), 3))
        assert "Period Volatility" not in stat["Risk"].index
        assert "Period Volatility" not in stat["Per Path Metrics"].columns
        assert list(stat["Risk"].index) == ["Gap Volatility", "Bar Volatility", "Annualized Volatility",
                                            "Downside Volatility", "Max Drawdown", "Sharpe Ratio",
                                            "Sortino Ratio"]
        assert list(stat["Risk"].columns) == ["Mean", "Median", "Lower", "Upper"]  # M11: keys unchanged
        assert {"Daily Returns", "Period Returns", "Tail Risk"} <= set(stat)

    def test_infinite_sortino_is_kept_and_counted(self):
        # One of three paths never has a down session. Its +inf stays in the
        # summary: the mean is NaN, the upper percentile +inf, and the
        # median of [a, b, inf] is the finite b
        sims = np.array([[1.0, 1.01, 1.02, 1.03], [1.0, 0.99, 1.01, 1.0], [1.0, 1.02, 0.97, 1.01]])
        with pytest.warns(UserWarning) as record:
            stat = statistical_report(sims, "1 day")
        assert [str(w.message) for w in record] == [
            "Sortino ratio is infinite on 1 of 3 paths (33.3%); its mean is NaN"]
        sortino = stat["Per Path Metrics"]["Sortino Ratio"]
        assert np.isposinf(sortino.iloc[0])
        row = stat["Risk"].loc["Sortino Ratio"]
        assert np.isnan(row["Mean"]) and np.isposinf(row["Upper"])
        assert row["Median"] == pytest.approx(sortino.iloc[1:].max())
        assert np.isfinite(row["Lower"])

    def test_sortino_mean_is_nan_only_with_an_infinite_path(self, capsys):
        # Without an infinite path the mean is the plain mean of the ratios;
        # with one it is NaN and prints as nan, while the other means stay
        finite = np.array([[1.0, 0.99, 1.01, 1.0], [1.0, 1.02, 0.97, 1.01]])
        stat = statistical_report(finite, "1 day")
        sortino = stat["Per Path Metrics"]["Sortino Ratio"]
        assert stat["Risk"].loc["Sortino Ratio", "Mean"] == pytest.approx(sortino.mean())
        rising = np.vstack([[1.0, 1.01, 1.02, 1.03], finite])
        with pytest.warns(UserWarning, match="its mean is NaN"):
            stat = statistical_report(rising, "1 day", display=True)
        assert np.isnan(stat["Risk"].loc["Sortino Ratio", "Mean"])
        assert np.isfinite(stat["Risk"].loc["Sharpe Ratio", "Mean"])
        line = next(x for x in capsys.readouterr().out.splitlines() if x.lstrip().startswith("Sortino Ratio"))
        assert line.split(":")[1].split()[0] == "nan"

    def test_inf_safe_quantile_matches_numpy_on_finite_values(self):
        values = np.sort(np.random.default_rng(2).normal(size=37))
        for q in (0.0, 0.025, 0.3, 0.5, 0.975, 1.0):
            assert _quantile(values, q) == pytest.approx(np.quantile(values, q))
        with_inf = np.r_[values, np.inf]
        assert _quantile(with_inf, 0.99) == np.inf and np.isfinite(_quantile(with_inf, 0.5))
        assert _range_summary(np.array([np.nan, 1.0, 2.0]), 0.95)["Mean"] == pytest.approx(1.5)

    def test_ruined_paths_are_kept_and_counted(self):
        # M9: CAGR -100%; returns after the loss are undefined and left out,
        # so the risk figures stop at the -100% session
        sims = np.array([[1.0, 1.01, 1.0, 1.02, 1.01], [1.0, 0.5, 0.0, 0.0, 0.0]])
        with pytest.warns(UserWarning) as record:
            stat = statistical_report(sims, "1 day")
        assert any("1 of 2 paths end at a total loss" in str(w.message) for w in record)
        per_path = stat["Per Path Metrics"]
        assert per_path["CAGR"].iloc[1] == -1.0
        assert per_path["Max Drawdown"].iloc[1] == 1.0
        dates = pd.bdate_range("2025-06-02", periods=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            single = performance_report(pd.Series(sims[1], index=dates))
        assert per_path["Annualized Volatility"].iloc[1] == pytest.approx(single["Risk"]["Annualized Volatility"])
        assert stat["Returns"].loc["CAGR", "Mean"] == pytest.approx((per_path["CAGR"].iloc[0] - 1) / 2)

    def test_range_wording(self, capsys):
        # M11: Lower/Upper are a range across paths, not a confidence interval
        statistical_report(random_paths(20, 30), "1 day", display=True)
        out = capsys.readouterr().out
        assert "95% range across paths" in out and "interval" not in out
        assert "95% range across paths" in statistical_report.__doc__


class TestInputs:
    def test_simulation_without_dates_is_an_array(self):
        paths = random_paths(5, 12)
        result = statistical_report(SimulationResult(paths, None, None, None), "1 day")
        array = statistical_report(paths, "1 day")
        pd.testing.assert_frame_equal(result["Per Path Metrics"], array["Per Path Metrics"])
        assert array["Total Years"] == pytest.approx(11 / 252)

    def test_dates_replace_a_simulations_labels(self):
        history = hourly_marks(["2025-06-02", "2025-06-03"])
        sim = simulation(history, 3)
        dates = pd.DatetimeIndex([history[-1]]).append(sim.dates)
        pd.testing.assert_frame_equal(statistical_report(sim, dates=dates)["Per Path Metrics"],
                                      statistical_report(sim)["Per Path Metrics"])

    @pytest.mark.parametrize("call, message", [
        (lambda p: statistical_report(p), "interval is required"),
        (lambda p: statistical_report(p, "1 hour"), "intraday risk figures use session returns"),
        (lambda p: statistical_report(p, "1 day", cagr_clock="calendar"), "needs the marks"),
        (lambda p: statistical_report(p, "1 day", calendar=None), "needs the marks"),
        (lambda p: statistical_report(p, "1 day", cagr_clock="bars"), "cagr_clock"),
        (lambda p: statistical_report(p[0], "1 day"), "2-D"),
    ])
    def test_paths_without_marks(self, call, message):
        with pytest.raises(ValueError, match=message):
            call(random_paths(3, 6))

    def test_simulation_dates_must_match_the_paths(self):
        sim = simulation(pd.bdate_range("2025-06-02", periods=5), 4)
        bad = sim._replace(paths=sim.paths[:, :-1])
        with pytest.raises(ValueError, match="future marks"):
            statistical_report(bad)


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


class TestPrintedReport:
    """The header gives calendar years, the period count and calendar days;
    each class volatility row prints only if the marks have that class."""

    def test_daily(self):
        # 21 XNYS sessions from 2025-06-02 (June 19 is a holiday) end on
        # 2025-07-01: 29 days, 29 / 365.25 = 0.08 years
        dates = pd.DatetimeIndex(["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06",
                                  "2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13",
                                  "2025-06-16", "2025-06-17", "2025-06-18", "2025-06-20", "2025-06-23",
                                  "2025-06-24", "2025-06-25", "2025-06-26", "2025-06-27", "2025-06-30",
                                  "2025-07-01"])
        out, header = printed(lambda: statistical_report(random_paths(20, 21), "1 day", dates=dates,
                                                         display=True))
        assert header == ["  Simulated Paths      :    20",
                          "  Total Time           :  0.08 years (29 days)",
                          "  Horizon              :    20 periods"]
        assert "Gap Vol" not in out and "Bar Vol" not in out

    def test_daily_simulation(self):
        sim = simulation(pd.bdate_range("2025-06-02", periods=10), 30)
        out, header = printed(lambda: statistical_report(sim, display=True))
        assert [line.split(":")[0].strip() for line in header] == [
            "Simulated Paths", "Total Time", "Horizon"]
        assert "Gap Vol" not in out and "Bar Vol" not in out

    def test_intraday(self):
        # Hourly marks 2025-06-02 09:30 to 2025-06-06 16:00 New York: 4 days
        # 6.5 hours, 4.2708 / 365.25 = 0.01 years; eight marks a session
        dates = hourly_marks(["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06"])
        out, header = printed(lambda: statistical_report(random_paths(20, 40), "1h", dates=dates,
                                                         display=True))
        assert header == ["  Simulated Paths      :    20",
                          "  Total Time           :  0.01 years (4 days)",
                          "  Horizon              :    39 periods"]
        assert "Gap Vol (per gap)" in out and "Bar Vol (per bar)" in out

    def test_one_gap_prints_its_row_even_when_its_volatility_is_undefined(self):
        # From a close: the next open (one gap return) and one bar
        history = hourly_marks(["2025-06-02", "2025-06-03"])
        sim = simulation(history, 1)
        assert list(sim.classes).count("gap") == 1
        out, _ = printed(lambda: statistical_report(sim, display=True))
        assert "Gap Vol (per gap)" in out and "Bar Vol (per bar)" in out

    def test_continuous(self):
        # Hourly UTC marks over exactly 3 days: 3 / 365.25 = 0.01 years
        dates = pd.date_range("2025-06-01", periods=24 * 3 + 1, freq="h", tz="UTC")
        out, header = printed(lambda: statistical_report(random_paths(20, len(dates)), dates=dates, calendar=None,
                                                         cagr_clock="calendar", display=True))
        assert header == ["  Simulated Paths      :    20",
                          "  Total Time           :  0.01 years (3 days)",
                          "  Horizon              :    72 periods"]
        # No gaps in continuous mode; hourly bars are regular
        assert "Gap Vol" not in out and "Bar Vol (per bar)" in out

    def test_continuous_daily(self):
        dates = pd.date_range("2025-06-01", periods=11, freq="D", tz="UTC")
        out, _ = printed(lambda: statistical_report(random_paths(20, 11), "1 day", dates=dates, calendar=None,
                                                    cagr_clock="calendar", display=True))
        assert "Gap Vol" not in out and "Bar Vol" not in out

    def test_paths_without_marks_print_only_the_period_count(self):
        out, header = printed(lambda: statistical_report(random_paths(20, 30), "1 day", display=True))
        assert header == ["  Simulated Paths      :    20",
                          "  Horizon              :    29 periods"]
        assert "Gap Vol" not in out and "Bar Vol" not in out

    def test_header_is_the_same_on_both_clocks_and_keeps_the_dict(self):
        dates = hourly_marks(["2025-06-02", "2025-06-03", "2025-06-04"])
        paths = random_paths(20, len(dates))
        _, by_session = printed(lambda: statistical_report(paths, "1h", dates=dates, display=True))
        _, by_calendar = printed(lambda: statistical_report(paths, "1h", dates=dates, cagr_clock="calendar",
                                                            display=True))
        assert by_session == by_calendar
        # The returned Total Years stays on the session clock: the first mark is
        # the first day's open, so three session closes follow it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert statistical_report(paths, "1h", dates=dates)["Total Years"] == pytest.approx(3 / 252)
        for label in ("sessions", "/252", "/365.25", "bars"):
            assert not any(label in line for line in by_session), label
