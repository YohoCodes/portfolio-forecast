"""next_trading_dates as the future layout (MC2, MC8): marks, classes and
tau continuing the history's grid. Expected marks are read off the NYSE
calendar by hand."""
import numpy as np
import pandas as pd

from portfolio_forecast import BacktestResult, FutureLayout
from portfolio_forecast.utils import classify_returns, next_trading_dates
from portfolio_forecast.utils.marks import history_marks

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def session(day, closes=range(10, 17), opening=True):
    """An hourly session's marks: the 9:30 open mark, then closes on :00."""
    return ([ny(f"{day} 09:30")] if opening else []) + [ny(f"{day} {h}:00") for h in closes]


HOURLY_TAU = [1.0, 0.5, 1, 1, 1, 1, 1, 1]  # gap, 9:30-10:00 half bar, six full bars


def hhmm(index):
    return [t.strftime("%H:%M") for t in index]


class TestIssuesRepro:
    """ISSUES.md: hourly fill="open" history gained a spurious 9:30 bar per
    future session. Each future session must have exactly one open mark,
    then its bar closes."""

    def test_0_3_0_fill_open_labels(self):
        # 0.3.0 simulate_buy_and_hold(fill="open") labels: the 9:30 fill,
        # then every bar close; no open mark in later sessions
        history = pd.DatetimeIndex(session("2025-06-05") + session("2025-06-06", opening=False))
        layout = next_trading_dates(history, 2, interval="1h", tz=NY)
        assert isinstance(layout, FutureLayout)
        assert hhmm(layout.dates) == ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"] * 2
        assert [d.day for d in layout.dates] == [9] * 8 + [10] * 8  # Monday, Tuesday
        assert list(layout.classes) == (["gap"] + ["regular"] * 7) * 2
        assert list(layout.tau) == HOURLY_TAU * 2

    def test_history_marks_output(self):
        bars = pd.DatetimeIndex([t for d in ("2025-06-05", "2025-06-06")
                                 for t in [ny(f"{d} 09:30")] + [ny(f"{d} {h}:00") for h in range(10, 16)]])
        px = pd.DataFrame(1.0, index=bars, columns=["A"])
        labels, _, _ = history_marks(px, px, ["A"])
        layout = next_trading_dates(labels, 2)
        assert (hhmm(layout.dates)).count("09:30") == 2 and len(layout.dates) == 16
        assert list(layout.classes) == (["gap"] + ["regular"] * 7) * 2


class TestIntraday:
    HISTORY = pd.DatetimeIndex(session("2025-06-05") + session("2025-06-06"))

    def test_weekend_then_sessions(self):
        layout = next_trading_dates(self.HISTORY, 1)
        assert layout.dates[0] == ny("2025-06-09 09:30") and layout.dates[-1] == ny("2025-06-09 16:00")
        assert list(layout.tau) == HOURLY_TAU

    def test_half_day(self):
        # After Wednesday 2024-11-27: Thanksgiving closed, Friday closes 13:00
        history = pd.DatetimeIndex(session("2024-11-26") + session("2024-11-27"))
        layout = next_trading_dates(history, 2)
        friday = layout.dates[:5]
        assert list(friday) == [ny("2024-11-29 09:30"), ny("2024-11-29 10:00"), ny("2024-11-29 11:00"),
                                ny("2024-11-29 12:00"), ny("2024-11-29 13:00")]
        assert list(layout.classes[:5]) == ["gap"] + ["regular"] * 4
        assert layout.dates[5] == ny("2024-12-02 09:30") and len(layout.dates) == 5 + 8

    def test_short_last_bar(self):
        # Hourly bars phased on :30 close at 10:30 ... 15:30, then a 30-minute
        # bar to the 16:00 close
        def day(d):
            return [ny(f"{d} 09:30")] + [ny(f"{d} {h}:30") for h in range(10, 16)] + [ny(f"{d} 16:00")]
        layout = next_trading_dates(pd.DatetimeIndex(day("2025-06-05") + day("2025-06-06")), 1)
        assert hhmm(layout.dates) == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
        assert list(layout.tau) == [1.0, 1, 1, 1, 1, 1, 1, 0.5]

    MID_SESSION = pd.DatetimeIndex(session("2025-06-05") + session("2025-06-06", closes=range(10, 13)))

    def test_history_ending_mid_session_finishes_that_session_first(self):
        # MC2: the history ends at Friday 12:00; 13:00 ... 16:00 come first
        layout = next_trading_dates(self.MID_SESSION, 2)
        assert list(layout.dates[:6]) == [ny("2025-06-06 13:00"), ny("2025-06-06 14:00"), ny("2025-06-06 15:00"),
                                          ny("2025-06-06 16:00"), ny("2025-06-09 09:30"), ny("2025-06-09 10:00")]
        assert list(layout.classes[:6]) == ["regular"] * 4 + ["gap", "regular"]
        assert list(layout.tau[:6]) == [1, 1, 1, 1, 1, 0.5]

    def test_partial_session_counts_as_the_first_of_n(self):
        # MC2: ending mid-session d, the horizon ends at the close of d + n - 1.
        # n=1 finishes Friday and stops; n=2 adds Monday and ends at its close
        one = next_trading_dates(self.MID_SESSION, 1)
        assert hhmm(one.dates) == ["13:00", "14:00", "15:00", "16:00"]
        assert one.dates[-1] == ny("2025-06-06 16:00")
        two = next_trading_dates(self.MID_SESSION, 2)
        assert len(two.dates) == 4 + 8 and two.dates[-1] == ny("2025-06-09 16:00")

    def test_history_ending_at_an_open_mark_is_mid_session(self):
        # Friday has only its 9:30 open mark: its bars are session 1 of n
        history = pd.DatetimeIndex(session("2025-06-05") + [ny("2025-06-06 09:30")])
        layout = next_trading_dates(history, 1)
        assert hhmm(layout.dates) == ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]
        assert layout.tau[0] == 0.5

    def test_history_ending_at_a_close_takes_n_full_sessions(self):
        layout = next_trading_dates(self.HISTORY, 2)
        assert len(layout.dates) == 16 and layout.dates[-1] == ny("2025-06-10 16:00")

    def test_off_grid_last_mark_gives_a_short_bar(self):
        # History ends at a 10:30 open mark (B1.6) under a :00 grid
        history = pd.DatetimeIndex(session("2025-06-05") + [ny("2025-06-06 10:30")])
        layout = next_trading_dates(history, 1)
        assert layout.dates[0] == ny("2025-06-06 11:00") and layout.tau[0] == 0.5
        assert layout.dates[-1] == ny("2025-06-06 16:00")  # MC2: the partial session is the one session

    def test_dst_spring_forward(self):
        # History before the 2024-03-10 change, in UTC: Monday's open is
        # 13:30 UTC, a week earlier it was 14:30 UTC
        history = pd.DatetimeIndex(session("2024-03-07") + session("2024-03-08")).tz_convert("UTC")
        layout = next_trading_dates(history, 1)
        assert str(layout.dates.tz) == "UTC"
        assert layout.dates[0] == pd.Timestamp("2024-03-11 13:30", tz="UTC")
        assert layout.dates[-1] == pd.Timestamp("2024-03-11 20:00", tz="UTC")

    def test_five_minute_grid(self):
        history = pd.date_range("2025-06-06 09:30", "2025-06-06 16:00", freq="5min", tz=NY)
        layout = next_trading_dates(history, 1)
        assert len(layout.dates) == 1 + 78 and (layout.tau == 1.0).all()

    def test_matches_classify_returns_of_the_extended_series(self):
        layout = next_trading_dates(self.HISTORY, 2)
        extended = pd.Series(1.0, index=self.HISTORY[-1:].append(layout.dates))
        assert list(classify_returns(extended)) == list(layout.classes)


class TestDailyAndLonger:
    def test_daily_closes_continue_as_closes(self):
        # Our fill="open" daily output: 9:30 fill, then 16:00 closes. After
        # Wednesday 2024-11-27: Friday's 13:00 close, then Monday's
        history = pd.DatetimeIndex([ny("2024-11-25 09:30"), ny("2024-11-25 16:00"),
                                    ny("2024-11-26 16:00"), ny("2024-11-27 16:00")])
        layout = next_trading_dates(history, 2)
        assert list(layout.dates) == [ny("2024-11-29 13:00"), ny("2024-12-02 16:00")]
        assert list(layout.classes) == ["regular", "regular"] and list(layout.tau) == [1.0, 1.0]

    def test_weekly_closes(self):
        history = pd.DatetimeIndex([ny("2024-11-15 16:00"), ny("2024-11-22 16:00")])
        layout = next_trading_dates(history, 2)
        assert list(layout.dates) == [ny("2024-11-29 13:00"), ny("2024-12-06 16:00")]

    def test_dates_stay_dates(self):
        layout = next_trading_dates(pd.bdate_range("2025-06-23", "2025-07-03"), 3)
        assert list(layout.dates) == [pd.Timestamp(d) for d in ("2025-07-07", "2025-07-08", "2025-07-09")]
        assert list(layout.classes) == ["regular"] * 3 and (layout.tau == 1.0).all()


class TestInputs:
    def test_series_and_backtest_result(self):
        history = pd.DatetimeIndex(session("2025-06-05") + session("2025-06-06"))
        values = pd.Series(np.linspace(100, 101, len(history)), index=history)
        expected = next_trading_dates(history, 1).dates
        assert next_trading_dates(values, 1).dates.equals(expected)
        result = BacktestResult(values, history, classify_returns(values))
        assert next_trading_dates(result, 1).dates.equals(expected)

    @staticmethod
    def assert_same_layout(layout, expected):
        assert layout.dates.equals(expected.dates)
        assert list(layout.classes) == list(expected.classes)
        assert np.array_equal(layout.tau, expected.tau)

    def test_series_and_frames_read_their_index(self):
        # Daily dates as yfinance gives them: a one-column frame from
        # download(...)["Close"], and the full two-symbol close frame
        dates = pd.DatetimeIndex(pd.bdate_range("2025-06-02", "2025-06-30").drop(pd.Timestamp("2025-06-19")),
                                 name="Date")
        expected = next_trading_dates(dates, 5)
        prices = np.linspace(100, 110, len(dates))
        series = pd.Series(prices, index=dates)
        one_column = pd.DataFrame({"AAPL": prices}, index=dates)
        two_columns = pd.DataFrame({"AAPL": prices, "MSFT": prices * 2}, index=dates)
        for values in (series, one_column, two_columns):
            self.assert_same_layout(next_trading_dates(values, 5), expected)

    def test_intraday_frame_reads_its_index(self):
        history = pd.DatetimeIndex(session("2025-06-05") + session("2025-06-06"))
        frame = pd.DataFrame({"A": np.linspace(100, 101, len(history)), "B": 1.0}, index=history)
        self.assert_same_layout(next_trading_dates(frame, 1), next_trading_dates(history, 1))

    def test_zero_is_an_empty_layout(self):
        layout = next_trading_dates(pd.bdate_range("2025-06-02", periods=5), 0)
        assert len(layout.dates) == len(layout.classes) == len(layout.tau) == 0


def utc(text):
    return pd.Timestamp(text, tz="UTC")


class TestZeroLengthGaps:
    """B1.4 on "24/5": weeknights have no gap and no open mark; the weekend does."""

    # Hourly marks ending at Wednesday 2025-06-04 00:00, Tuesday's close
    HISTORY = pd.date_range("2025-06-03 00:00", "2025-06-04 00:00", freq="h", tz="UTC")

    def test_weeknight_sessions_are_bar_closes_only(self):
        layout = next_trading_dates(self.HISTORY, 3, calendar="24/5")
        # Wednesday to Friday: 24 bar closes each, the first at 01:00
        assert len(layout.dates) == 72
        assert layout.dates[0] == utc("2025-06-04 01:00") and layout.dates[-1] == utc("2025-06-07 00:00")
        assert set(layout.classes) == {"regular"} and set(layout.tau) == {1.0}

    def test_monday_has_the_one_gap(self):
        layout = next_trading_dates(self.HISTORY, 4, calendar="24/5")
        gaps = layout.dates[layout.classes == "gap"]
        assert list(gaps) == [utc("2025-06-09 00:00")]
        assert len(layout.dates) == 72 + 25

    def test_xnys_sessions_still_open_with_a_gap(self):
        layout = next_trading_dates(TestIntraday.HISTORY, 2)
        assert list(layout.classes) == (["gap"] + ["regular"] * 7) * 2


class TestContinuousMode:
    """U3: future marks t_T + kΔ, n in bars, all regular."""

    HISTORY = pd.date_range("2025-06-06 20:00", periods=6, freq="h", tz="UTC")

    def test_continues_the_grid(self):
        layout = next_trading_dates(self.HISTORY, 4, calendar=None)
        assert list(layout.dates) == [self.HISTORY[-1] + pd.Timedelta(hours=k) for k in range(1, 5)]
        assert list(layout.classes) == ["regular"] * 4 and list(layout.tau) == [1.0] * 4

    def test_runs_through_weekends_and_keeps_naive_labels(self):
        days = pd.date_range("2025-06-05", periods=3, freq="D")
        layout = next_trading_dates(days, 3, calendar=None)
        assert list(layout.dates) == list(pd.date_range("2025-06-08", periods=3, freq="D"))

    def test_interval_overrides_the_step(self):
        layout = next_trading_dates(self.HISTORY[-1:], 2, interval="15min", calendar=None)
        assert list(layout.dates) == [self.HISTORY[-1] + pd.Timedelta(minutes=m) for m in (15, 30)]
        week = next_trading_dates(self.HISTORY[-1:], 1, interval="1wk", calendar=None)
        assert week.dates[0] == self.HISTORY[-1] + pd.Timedelta(days=7)

    def test_monthly_interval_raises(self):
        import pytest
        with pytest.raises(ValueError, match="months"):
            next_trading_dates(self.HISTORY, 2, interval="1mo", calendar=None)
