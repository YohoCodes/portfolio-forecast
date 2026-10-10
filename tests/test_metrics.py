"""Metric functions (M3-M9). Expected figures are derived from the rules in
each test, not from the functions under test."""
import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast import BacktestResult
from portfolio_forecast.performance.metrics import (
    calculate_actual_yearly_return,
    calculate_cagr,
    calculate_class_volatility,
    calculate_downside_deviation,
    calculate_excess_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_years,
    calculate_volatility,
    count_sessions,
)
from portfolio_forecast.utils.calendar import session_spans

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def utc(text):
    return pd.Timestamp(text, tz="UTC")


def series(labels, values):
    return pd.Series([float(v) for v in values], index=pd.DatetimeIndex(labels))


def session_marks(day, opening=True):
    """An hourly session's marks: the 9:30 open mark, then 10:00 to 16:00."""
    return ([ny(f"{day} 09:30")] if opening else []) + [ny(f"{day} {h}:00") for h in range(10, 17)]


class TestExcessReturns:
    def test_risk_free_rate_per_session(self):
        # M3: e = R - r_f / 252
        r = pd.Series([0.01, -0.01])
        assert calculate_excess_returns(r, 0.0252).tolist() == pytest.approx([0.0099, -0.0101])

    def test_continuous_mode_per_day(self):
        # U3: r_f / 365
        assert calculate_excess_returns(pd.Series([0.0]), 0.0365, 365).tolist() == pytest.approx([-0.0001])


class TestTime:
    """M4: N session closes after the first mark; Y = N / 252 or calendar days / 365.25."""

    def test_partial_first_session_counts_whole(self):
        marks = pd.DatetimeIndex(session_marks("2025-06-04") + session_marks("2025-06-05"))
        assert count_sessions(marks) == 2
        assert count_sessions(marks[1:]) == 2  # fill="close": still after the fill

    def test_first_mark_at_a_close_does_not_count_its_session(self):
        assert count_sessions(pd.DatetimeIndex([ny("2025-06-04 16:00"), ny("2025-06-05 16:00")])) == 1

    def test_session_in_progress_at_the_end_counts_whole(self):
        # M4: the 5th is partial at 12:00 but counts
        assert count_sessions(pd.DatetimeIndex([ny("2025-06-04 09:30"), ny("2025-06-05 12:00")])) == 2
        # Ending at an open mark is a partial session too
        assert count_sessions(pd.DatetimeIndex([ny("2025-06-04 16:00"), ny("2025-06-05 09:30")])) == 1

    def test_24_5_close_that_is_the_next_open_is_not_partial(self):
        # Tuesday 00:00 UTC is Monday's close on "24/5": nothing of Tuesday has elapsed
        marks = pd.DatetimeIndex([utc("2025-06-02 12:00"), utc("2025-06-03 00:00")])
        assert count_sessions(marks, calendar="24/5") == 1

    def test_weekly_counts_the_sessions_spanned(self):
        # M1: 9-13 June is 5 sessions, 16-20 June 4 (Juneteenth on the 19th)
        closes = pd.DatetimeIndex([ny("2025-06-06 16:00"), ny("2025-06-13 16:00"), ny("2025-06-20 16:00")])
        assert count_sessions(closes) == 9
        dates = pd.DatetimeIndex(["2025-06-06", "2025-06-13", "2025-06-20"])
        assert count_sessions(dates) == 9

    def test_dated_two_session_bars_count_as_session_spans_reads_them(self):
        # M1, M4: N counts the sessions after the fill, i.e. after the first
        # mark. session_spans reads each date as the start of a 2-session
        # bar, valued at its last session's close: the first mark is
        # C(06-03), the last C(06-13). Sessions with closes in
        # (C(06-03), C(06-13)] are 06-04..06-06 and 06-09..06-13: 8, the
        # 2 sessions of each of the 4 bars after the first
        dates = pd.DatetimeIndex(["2025-06-02", "2025-06-04", "2025-06-06", "2025-06-10", "2025-06-12"])
        spans = session_spans(dates)
        assert spans["n_sessions"].tolist() == [2] * 5
        assert spans["last_close"].iloc[0] == ny("2025-06-03 16:00")
        assert spans["last_close"].iloc[-1] == ny("2025-06-13 16:00")
        assert count_sessions(dates) == 8 == spans["n_sessions"].iloc[1:].sum()

    def test_dated_weekly_bars_count_as_session_spans_reads_them(self):
        # M1, M4: weeks dated by their Monday. session_spans values each
        # bar at its week's last close, so the first mark is C(06-06) and the
        # sessions after it are the weeks of 9 June (5) and 16 June (4,
        # Juneteenth on the 19th): 9
        dates = pd.DatetimeIndex(["2025-06-02", "2025-06-09", "2025-06-16"])
        spans = session_spans(dates)
        assert spans["n_sessions"].tolist() == [5, 5, 4]
        assert spans["last_close"].iloc[0] == ny("2025-06-06 16:00")
        assert count_sessions(dates) == 9 == spans["n_sessions"].iloc[1:].sum()

    def test_daily_dates_off_the_calendar_still_count(self):
        # bdate_range includes 4 July 2025, not a session: 30 June to 7 July is 5 closes after the first
        assert count_sessions(pd.bdate_range("2025-06-30", "2025-07-07")) == 4

    def test_total_years_on_both_clocks(self):
        marks = pd.DatetimeIndex([ny("2025-06-04 09:30"), ny("2025-06-05 16:00")])
        assert calculate_total_years(marks) == pytest.approx(2 / 252)
        days = (ny("2025-06-05 16:00") - ny("2025-06-04 09:30")) / pd.Timedelta(days=1)
        assert calculate_total_years(marks, cagr_clock="calendar") == pytest.approx(days / 365.25)

    def test_session_clock_raises_in_continuous_mode(self):
        hours = pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC")
        with pytest.raises(ValueError, match="cagr_clock='calendar'"):
            calculate_total_years(hours, calendar=None)
        assert calculate_total_years(hours, calendar=None, cagr_clock="calendar") == pytest.approx(2 / 24 / 365.25)

    def test_unknown_clock_raises(self):
        with pytest.raises(ValueError, match="cagr_clock"):
            calculate_total_years(pd.DatetimeIndex(["2025-06-04", "2025-06-05"]), cagr_clock="bars")


class TestCagr:
    def test_years(self):
        values = pd.Series([100.0, 121.0])
        assert calculate_cagr(values, years=2.0) == pytest.approx(0.1)

    def test_total_loss_is_minus_100_percent(self):
        # M9
        assert calculate_cagr(pd.Series([100.0, 50.0, 0.0]), years=0.5) == -1.0

    def test_needs_a_clock(self):
        with pytest.raises(ValueError, match="years"):
            calculate_cagr(pd.Series([1.0, 2.0]))


class TestSessionRisk:
    def test_sharpe_uses_the_sample_deviation_of_r(self):
        r = pd.Series([0.01, -0.02, 0.03, 0.00])
        expected = (r.mean() - 0.0252 / 252) / r.std(ddof=1) * np.sqrt(252)
        assert calculate_sharpe_ratio(r, 252, 0.0252) == pytest.approx(expected)

    def test_sharpe_of_constant_returns_is_nan(self):
        assert np.isnan(calculate_sharpe_ratio(pd.Series([0.01, 0.01]), 252))

    def test_downside_deviation_over_all_sessions(self):
        # M7: DD = sqrt((0.01² + 0.03²) / 4)
        r = pd.Series([0.02, -0.01, 0.03, -0.03])
        assert calculate_downside_deviation(r) == pytest.approx(np.sqrt((0.01**2 + 0.03**2) / 4))

    def test_sortino(self):
        r = pd.Series([0.02, -0.01, 0.03, -0.03])
        dd = np.sqrt((0.01**2 + 0.03**2) / 4)
        assert calculate_sortino_ratio(r) == pytest.approx(0.0025 * np.sqrt(252) / dd)

    def test_no_down_session_is_infinite_with_a_warning(self):
        with pytest.warns(UserWarning) as record:
            assert calculate_sortino_ratio(pd.Series([0.01, 0.0, 0.02])) == np.inf
        assert [str(w.message) for w in record] == [
            "Sortino ratio is infinite: none of the 3 sessions is below the risk-free rate"]

    def test_no_down_session_and_zero_mean_is_nan(self):
        with pytest.warns(UserWarning) as record:
            assert np.isnan(calculate_sortino_ratio(pd.Series([0.0, 0.0])))
        assert [str(w.message) for w in record] == [
            "Sortino ratio is NaN: all 2 sessions return exactly the risk-free rate"]

    def test_nan_after_ruin_is_dropped(self):
        # M9: the 0/0 returns after a total loss leave every figure unchanged,
        # including the session count n of the downside deviation
        r = pd.Series([0.02, -0.01, 0.03, -1.0])
        with_nan = pd.concat([r, pd.Series([np.nan, np.nan])], ignore_index=True)
        assert calculate_volatility(with_nan) == pytest.approx(calculate_volatility(r))
        assert calculate_sharpe_ratio(with_nan, 252, 0.0252) == pytest.approx(calculate_sharpe_ratio(r, 252, 0.0252))
        assert calculate_downside_deviation(with_nan) == pytest.approx(calculate_downside_deviation(r))
        assert calculate_sortino_ratio(with_nan) == pytest.approx(calculate_sortino_ratio(r))

    def test_no_returns_is_nan_without_a_warning(self, recwarn):
        assert np.isnan(calculate_sortino_ratio(pd.Series([], dtype=float)))
        assert not recwarn


class TestMaxDrawdown:
    def test_open_marks_count(self):
        # M6: the 4th closes at 110, the 5th opens at 99, then closes at 111.
        # The open mark holds the trough: 1 - 99/110 = 0.1; closes alone never fall
        values = series([ny("2025-06-04 09:30"), ny("2025-06-04 16:00"), ny("2025-06-05 09:30"),
                         ny("2025-06-05 16:00")], [100, 110, 99, 111])
        assert calculate_max_drawdown(values) == pytest.approx(0.1)
        assert calculate_max_drawdown(values.iloc[[0, 1, 3]]) == 0.0

    def test_total_loss(self):
        assert calculate_max_drawdown(pd.Series([100.0, 120.0, 0.0])) == 1.0


class TestClassVolatility:
    """M5: Gap and Bar Volatility per unit; a short bar scaled in logs (U2)."""

    # Two hourly sessions with a :00 grid: the 9:30-10:00 first bar is a half bar
    LABELS = session_marks("2025-06-04") + session_marks("2025-06-05")
    VALUES = [100, 101, 100.5, 101.5, 102, 101, 102.5, 103, 101, 102, 103.5, 103, 104, 104.5, 104, 105]

    def test_gap_and_bar_volatility(self):
        values = series(self.LABELS, self.VALUES)
        r = values.pct_change().iloc[1:].to_numpy()
        # Returns ending at 10:00 are half bars (tau = 0.5); index 7 is the gap
        half = np.array([0, 8])
        scaled = r.copy()
        scaled[half] = np.exp(np.log1p(r[half]) * np.sqrt(2)) - 1
        bars = np.delete(scaled, 7)
        out = calculate_class_volatility(values)
        assert np.isnan(out["Gap Volatility"])  # one gap: no sample deviation
        assert out["Bar Volatility"] == pytest.approx(np.std(bars, ddof=1))

    def test_irregular_returns_are_left_out_and_gaps_pooled(self):
        labels = self.LABELS + session_marks("2025-06-06")
        values = series(labels, self.VALUES + [103, 104, 103, 104, 105, 104, 105, 106])
        classes = pd.Series("regular", index=values.index[1:])
        classes[[ny("2025-06-05 09:30"), ny("2025-06-06 09:30")]] = "gap"
        classes[ny("2025-06-06 12:00")] = "irregular"
        r = values.pct_change().iloc[1:]
        out = calculate_class_volatility(BacktestResult(values, values.index, classes))
        gaps = r[[ny("2025-06-05 09:30"), ny("2025-06-06 09:30")]]
        assert out["Gap Volatility"] == pytest.approx(gaps.std(ddof=1))
        regular = r[(classes == "regular").to_numpy()].copy()
        short = regular.index.hour == 10
        regular[short] = np.expm1(np.log1p(regular[short]) * np.sqrt(2))
        assert out["Bar Volatility"] == pytest.approx(regular.std(ddof=1))

    def test_nan_on_daily_data(self):
        values = series(pd.DatetimeIndex(["2025-06-04", "2025-06-05", "2025-06-06"]), [1, 2, 3])
        out = calculate_class_volatility(values)
        assert np.isnan(out["Gap Volatility"]) and np.isnan(out["Bar Volatility"])

    def test_nan_on_continuous_daily_data(self):
        # M5: continuous daily data is daily data
        values = series(pd.date_range("2025-06-01", periods=5, freq="D", tz="UTC"), [100, 101, 100, 102, 101])
        out = calculate_class_volatility(values, calendar=None)
        assert np.isnan(out["Gap Volatility"]) and np.isnan(out["Bar Volatility"])

    def test_continuous_mode_has_no_gaps_and_no_scaling(self):
        values = series(pd.date_range("2025-06-01", periods=5, freq="h", tz="UTC"), [100, 101, 100, 102, 101])
        out = calculate_class_volatility(values, calendar=None)
        assert np.isnan(out["Gap Volatility"])
        assert out["Bar Volatility"] == pytest.approx(values.pct_change().std(ddof=1))


class TestYearlyReturns:
    """M8: a year is full iff the series covers its first and last sessions."""

    SESSIONS = xcals.get_calendar("XNYS", start="2024-12-01", end="2026-01-31").sessions

    def daily(self, first, last, fill_open):
        days = self.SESSIONS[(self.SESSIONS >= first) & (self.SESSIONS <= last)]
        closes = [ny(f"{d:%Y-%m-%d} 16:00") for d in days]
        closes = [t.replace(hour=13) if t.date() in (pd.Timestamp("2024-12-24").date(),
                                                      pd.Timestamp("2025-07-03").date(),
                                                      pd.Timestamp("2025-11-28").date(),
                                                      pd.Timestamp("2025-12-24").date()) else t for t in closes]
        labels = ([ny(f"{days[0]:%Y-%m-%d} 09:30")] if fill_open else []) + closes
        return series(labels, 100 * 1.001 ** np.arange(len(labels)))

    def test_full_and_partial_years(self):
        values = self.daily("2024-12-02", "2026-01-05", fill_open=True)
        yearly = calculate_actual_yearly_return(values)
        assert list(yearly.index) == [2024, 2025, 2026]
        assert yearly["is_full"].tolist() == [False, True, False]
        assert yearly.loc[2025, "days"] == 365
        last = {y: values[values.index.year == y].iloc[-1] for y in (2024, 2025, 2026)}
        assert yearly.loc[2024, "return"] == pytest.approx(last[2024] / 100 - 1)
        assert yearly.loc[2025, "return"] == pytest.approx(last[2025] / last[2024] - 1)
        assert yearly.loc[2024, "days"] == 30  # 2 to 31 December

    def test_fill_open_on_the_first_session_covers_the_year(self):
        values = self.daily("2025-01-02", "2025-12-31", fill_open=True)
        assert calculate_actual_yearly_return(values)["is_full"].tolist() == [True]

    def test_fill_close_on_the_first_session_misses_its_move(self):
        values = self.daily("2025-01-02", "2025-12-31", fill_open=False)
        assert calculate_actual_yearly_return(values)["is_full"].tolist() == [False]

    def test_continuous_midnight_closes_the_day_before(self):
        # The 2025-01-01 00:00 UTC mark ends 31 December: one year, 2024
        values = series(pd.date_range("2024-12-31", "2025-01-01", freq="6h", tz="UTC"), [100, 101, 102, 103, 104])
        yearly = calculate_actual_yearly_return(values, calendar=None)
        assert list(yearly.index) == [2024]
        assert yearly.loc[2024, "return"] == pytest.approx(0.04)
