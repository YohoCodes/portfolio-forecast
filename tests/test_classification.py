"""Daily detection, classify_returns and to_session_values (MC1)."""
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast import BacktestResult
from portfolio_forecast.utils import classify_returns, to_session_values
from portfolio_forecast.utils.marks import history_marks, is_daily

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def series(labels):
    labels = pd.DatetimeIndex(labels)
    return pd.Series(100.0 + np.arange(len(labels)), index=labels)


def session_marks(day, opening=True):
    """An hourly session's marks: the 9:30 open mark, then 10:00 to 16:00."""
    marks = [ny(f"{day} {h}:00") for h in range(10, 17)]
    return ([ny(f"{day} 09:30")] if opening else []) + marks


FRI, MON = "2025-06-06", "2025-06-09"
INTRADAY = series(session_marks(FRI) + session_marks(MON))


class TestIsDaily:
    def test_our_fill_open_daily_output_is_daily(self):
        # history_marks' own daily output: 09:30 fill, then 16:00 closes
        days = pd.DatetimeIndex(["2025-06-05", "2025-06-06", "2025-06-09"])
        px = pd.DataFrame(1.0, index=days, columns=["A"])
        labels, _, _ = history_marks(px, px, ["A"])
        assert [t.strftime("%H:%M") for t in labels] == ["09:30", "16:00", "16:00", "16:00"]
        assert is_daily(series(labels))

    def test_session_closes_are_daily(self):
        assert is_daily(series([ny("2025-06-05 16:00"), ny("2025-06-06 16:00")]))

    def test_half_day_close_is_a_close(self):
        # 13:00 is the close on 2024-11-29 and mid-session on 2024-11-26
        assert is_daily(series([ny("2024-11-27 16:00"), ny("2024-11-29 13:00")]))
        assert not is_daily(series([ny("2024-11-25 16:00"), ny("2024-11-26 13:00"), ny("2024-11-27 16:00")]))

    def test_16_00_on_a_half_day_is_off_the_calendar(self):
        # Never from clock times alone: 16:00 is after the 2024-11-29 close
        with pytest.raises(ValueError, match="outside XNYS session hours"):
            is_daily(series([ny("2024-11-27 16:00"), ny("2024-11-29 16:00")]))

    def test_midnight_dates_are_daily(self):
        assert is_daily(series(pd.bdate_range("2025-06-02", periods=5)))
        assert is_daily(series(pd.bdate_range("2025-06-02", periods=5, tz="UTC")))

    def test_weekly_closes_are_daily(self):
        assert is_daily(series([ny("2024-11-18 09:30"), ny("2024-11-22 16:00"), ny("2024-11-29 13:00")]))

    def test_first_session_open_mark_only_stays_daily(self):
        # A single session's first mark is the fill, so one session is daily
        assert is_daily(series([ny(f"{FRI} 09:30"), ny(f"{FRI} 16:00")]))

    def test_intraday_is_not_daily(self):
        assert not is_daily(INTRADAY)

    def test_two_closes_one_session_apart_but_one_off_close(self):
        assert not is_daily(series([ny("2025-06-05 16:00"), ny("2025-06-06 15:00"), ny("2025-06-09 16:00")]))

    def test_accepts_a_backtest_result(self):
        result = BacktestResult(INTRADAY, INTRADAY.index, classify_returns(INTRADAY))
        assert not is_daily(result)

    def test_naive_times_raise(self):
        with pytest.raises(ValueError, match="localize"):
            is_daily(INTRADAY.tz_localize(None))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            is_daily(pd.Series([], index=pd.DatetimeIndex([], tz=NY), dtype=float))


class TestClassifyReturns:
    def test_matches_history_marks_when_every_price_is_fresh(self):
        index = pd.DatetimeIndex([ny(f"{d} {t}") for d in (FRI, MON)
                                  for t in ["09:30"] + [f"{h}:00" for h in range(10, 16)]])
        px = pd.DataFrame(1.0, index=index, columns=["A"])
        labels, _, classes = history_marks(px, px, ["A"])
        assert classify_returns(series(labels)).equals(classes)

    def test_gap_and_regular(self):
        classes = classify_returns(INTRADAY)
        assert classes.index.equals(INTRADAY.index[1:])
        assert list(classes) == ["regular"] * 7 + ["gap"] + ["regular"] * 7

    def test_first_mark_of_a_session_at_10_30_is_irregular(self):
        # B1.6: Monday's first mark is at 10:30, after the open; Tuesday
        # opens normally
        # (Monday runs to its close, so no scheduled mark is missing, B4.1)
        monday = [ny(f"{MON} 10:30")] + [ny(f"{MON} {h}:00") for h in range(11, 17)]
        values = series(session_marks(FRI) + monday + session_marks("2025-06-10"))
        classes = classify_returns(values)
        assert classes.loc[ny(f"{MON} 10:30")] == "irregular"
        assert list(classes.loc[ny(f"{MON} 11:00"):ny(f"{MON} 12:00")]) == ["regular", "regular"]
        assert classes.loc[ny("2025-06-10 09:30")] == "gap"

    def test_fill_close_history_is_inferable(self):
        values = series(session_marks(FRI, opening=False) + session_marks(MON))
        assert list(classify_returns(values)) == ["regular"] * 6 + ["gap"] + ["regular"] * 7

    def test_no_open_marks_raises(self):
        # 0.3.0-style intraday output: closes only after the first session
        values = series(session_marks(FRI) + session_marks(MON, opening=False))
        with pytest.raises(ValueError, match="no open marks"):
            classify_returns(values)

    def test_one_session_is_all_regular(self):
        assert list(classify_returns(series(session_marks(FRI)))) == ["regular"] * 7

    def test_daily_is_all_regular(self):
        values = series(pd.bdate_range("2025-06-02", periods=4))
        classes = classify_returns(values)
        assert list(classes) == ["regular"] * 3 and classes.index.equals(values.index[1:])

    def test_keeps_the_input_timezone(self):
        values = INTRADAY.tz_convert("UTC")
        assert classify_returns(values).index.equals(values.index[1:])


class TestToSessionValues:
    def test_first_mark_then_each_session_close(self):
        out = to_session_values(INTRADAY)
        assert list(out.index) == [ny(f"{FRI} 09:30"), ny(f"{FRI} 16:00"), ny(f"{MON} 16:00")]
        assert list(out) == [100.0, 107.0, 115.0]

    def test_returns_are_session_returns(self):
        out = to_session_values(INTRADAY)
        assert out.iloc[2] / out.iloc[1] == INTRADAY.loc[ny(f"{MON} 16:00")] / INTRADAY.loc[ny(f"{FRI} 16:00")]

    def test_missing_last_bar_is_labelled_at_the_close(self):
        # Friday ends at 15:00: its last value moves to the 16:00 close, with one warning
        values = series(session_marks(FRI)[:-1] + session_marks(MON))
        with pytest.warns(UserWarning) as record:
            out = to_session_values(values)
        assert [str(w.message) for w in record] == [
            f"1 session(s) end before their close, first {FRI}: each one's last value is labelled at its close"]
        assert list(out.index) == [ny(f"{FRI} 09:30"), ny(f"{FRI} 16:00"), ny(f"{MON} 16:00")]
        assert out.iloc[1] == values.loc[ny(f"{FRI} 15:00")]
        assert is_daily(out)

    def test_short_sessions_of_a_backtest_result(self):
        # Both sessions miss their last two bars: the count and the first date
        values = series(session_marks(FRI)[:-2] + session_marks(MON)[:-2])
        result = BacktestResult(values, values.index, classify_returns(values))
        with pytest.warns(UserWarning, match=rf"^2 session\(s\) end before their close, first {FRI}:"):
            out = to_session_values(result)
        assert list(out.index) == [ny(f"{FRI} 09:30"), ny(f"{FRI} 16:00"), ny(f"{MON} 16:00")]
        assert list(out) == [values.iloc[0], values.loc[ny(f"{FRI} 14:00")], values.iloc[-1]]

    def test_session_in_progress_is_left_out(self, recwarn):
        # Monday ends at 12:00 with its close (16:00) after now: left out, no warning
        values = series(session_marks(FRI) + session_marks(MON)[:4])
        out = to_session_values(values, now=ny(f"{MON} 12:30"))
        assert list(out.index) == [ny(f"{FRI} 09:30"), ny(f"{FRI} 16:00")]
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_now_in_another_timezone(self):
        # 20:30 UTC is 16:30 New York, after Monday's close: relabelled, with the warning
        values = series(session_marks(FRI) + session_marks(MON)[:4])
        with pytest.warns(UserWarning, match=rf"^1 session\(s\) end before their close, first {MON}:"):
            out = to_session_values(values, now=pd.Timestamp(f"{MON} 20:30", tz="UTC"))
        assert out.index[-1] == ny(f"{MON} 16:00")
        # 19:30 UTC is 15:30 New York, before the close: left out
        assert len(to_session_values(values, now=pd.Timestamp(f"{MON} 19:30", tz="UTC"))) == 2

    def test_naive_now_is_local_time(self):
        # A naive now is read in this machine's timezone, as in simulate_buy_and_hold
        from dateutil.tz import tzlocal
        values = series(session_marks(FRI) + session_marks(MON)[:4])
        before = ny(f"{MON} 15:59").tz_convert(tzlocal()).tz_localize(None)
        assert len(to_session_values(values, now=before)) == 2
        after = ny(f"{MON} 16:00").tz_convert(tzlocal()).tz_localize(None)
        with pytest.warns(UserWarning, match="end before their close"):
            out = to_session_values(values, now=after)
        assert out.index[-1] == ny(f"{MON} 16:00") and out.iloc[-1] == values.iloc[-1]

    def test_default_now_is_the_clock(self, recwarn):
        # A history far in the future ends mid-session: its last session is still to close
        days = pd.bdate_range("2099-06-01", periods=2).strftime("%Y-%m-%d")
        values = series(session_marks(days[0]) + session_marks(days[1])[:3])
        assert list(to_session_values(values).index) == [ny(f"{days[0]} 09:30"), ny(f"{days[0]} 16:00")]

    def test_full_sessions_do_not_warn(self, recwarn):
        to_session_values(INTRADAY)
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_fill_close_history(self):
        values = series(session_marks(FRI, opening=False) + session_marks(MON))
        assert list(to_session_values(values).index) == [ny(f"{FRI} 10:00"), ny(f"{FRI} 16:00"), ny(f"{MON} 16:00")]

    def test_backtest_result(self):
        result = BacktestResult(INTRADAY, INTRADAY.index, classify_returns(INTRADAY))
        assert to_session_values(result).equals(to_session_values(INTRADAY))

    def test_daily_comes_back_unchanged(self):
        values = series([ny("2025-06-05 09:30"), ny("2025-06-05 16:00"), ny("2025-06-06 16:00")])
        assert to_session_values(values).equals(values)
        dates = series(pd.bdate_range("2025-06-02", periods=3))
        assert to_session_values(dates).equals(dates)


class TestNeitherDailyNorIntraday:
    """MC1: one mark per session, not at the close (12:00 every session)."""

    NOON = series([ny("2025-06-05 12:00"), ny("2025-06-06 12:00"), ny("2025-06-09 12:00")])

    def test_is_not_daily(self):
        assert not is_daily(self.NOON)

    def test_with_a_fill_mark_is_not_daily(self):
        assert not is_daily(series([ny("2025-06-05 09:30")] + list(self.NOON.index)))

    def test_classify_returns_raises_naming_to_session_values(self):
        with pytest.raises(ValueError, match="can't be classified.*to_session_values"):
            classify_returns(self.NOON)

    def test_next_trading_dates_raises(self):
        from portfolio_forecast.utils import next_trading_dates
        with pytest.raises(ValueError, match="to_session_values"):
            next_trading_dates(self.NOON, 2)
        with pytest.raises(ValueError, match="to_session_values"):
            next_trading_dates(self.NOON, 2, interval="1h")

    def test_to_session_values_relabels_at_each_session_close(self, recwarn):
        # One mark per session is relabelled without a warning
        out = to_session_values(self.NOON)
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
        assert list(out.index) == [ny("2025-06-05 16:00"), ny("2025-06-06 16:00"), ny("2025-06-09 16:00")]
        assert list(out) == list(self.NOON)
        assert is_daily(out) and list(classify_returns(out)) == ["regular", "regular"]

    def test_to_session_values_half_day_close_and_input_timezone(self):
        values = series([ny("2024-11-27 12:00"), ny("2024-11-29 12:00")]).tz_convert("UTC")
        out = to_session_values(values)
        assert list(out.index) == [ny("2024-11-27 16:00"), ny("2024-11-29 13:00")]
        assert str(out.index.tz) == "UTC"

    def test_to_session_values_keeps_the_fill_mark(self):
        values = series([ny("2025-06-05 09:30")] + list(self.NOON.index))
        out = to_session_values(values)
        assert list(out.index) == [ny("2025-06-05 09:30"), ny("2025-06-05 16:00"),
                                   ny("2025-06-06 16:00"), ny("2025-06-09 16:00")]
        assert list(out) == list(values)


class TestMissingScheduledMarks:
    """B4.1 from labels alone: a return spanning a scheduled mark absent from
    the series is irregular; a holiday is not scheduled."""

    def test_mid_session_mark_missing(self):
        marks = session_marks(FRI)
        values = series(marks[:4] + marks[5:] + session_marks(MON))  # no 13:00 mark
        classes = classify_returns(values)
        assert list(classes[classes == "irregular"].index) == [ny(f"{FRI} 14:00")]
        assert classes.loc[ny(f"{MON} 09:30")] == "gap"

    def test_whole_session_missing(self):
        values = series(session_marks(FRI) + session_marks("2025-06-10"))  # Monday absent
        classes = classify_returns(values)
        assert classes.loc[ny("2025-06-10 09:30")] == "irregular"
        assert (classes == "irregular").sum() == 1

    def test_holiday_keeps_the_gap(self):
        values = series(session_marks("2025-06-18") + session_marks("2025-06-20"))  # Juneteenth between
        classes = classify_returns(values)
        assert classes.loc[ny("2025-06-20 09:30")] == "gap"
        assert (classes == "irregular").sum() == 0

    def test_half_day_short_last_bar_is_not_missing(self):
        # 2024-11-29 closes at 13:00: 12:00 -> 13:00 is its last bar
        values = series(session_marks("2024-11-27") + [ny("2024-11-29 09:30")]
                        + [ny(f"2024-11-29 {h}:00") for h in range(10, 14)])
        assert (classify_returns(values) == "irregular").sum() == 0

    def test_daily_missing_session(self):
        dates = pd.DatetimeIndex(["2025-07-02", "2025-07-03", "2025-07-07", "2025-07-09"])
        assert list(classify_returns(series(dates))) == ["regular", "regular", "irregular"]

    def test_daily_fill_open_output_missing_session(self):
        values = series([ny("2025-06-05 09:30"), ny("2025-06-05 16:00"), ny("2025-06-06 16:00"),
                         ny("2025-06-10 16:00")])
        assert list(classify_returns(values)) == ["regular", "regular", "irregular"]


class TestContinuousMode:
    """U3 with calendar=None."""

    HOURS = series(pd.date_range("2025-06-02 05:00", "2025-06-05 03:00", freq="h", tz="UTC"))

    def test_is_daily_iff_the_step_is_one_day(self):
        assert not is_daily(self.HOURS, calendar=None)
        assert is_daily(series(pd.date_range("2025-06-01", periods=10, freq="D", tz="UTC")), calendar=None)
        # Weekday-only daily data still steps one day most often
        assert is_daily(series(pd.bdate_range("2025-06-02", periods=10)), calendar=None)
        assert not is_daily(series(pd.date_range("2025-06-01", periods=5, freq="7D")), calendar=None)

    def test_classify_returns_all_regular_on_the_grid(self):
        classes = classify_returns(self.HOURS, calendar=None)
        assert classes.index.equals(self.HOURS.index[1:])
        assert set(classes) == {"regular"}

    def test_classify_returns_missing_grid_point(self):
        values = self.HOURS.drop(self.HOURS.index[10])
        classes = classify_returns(values, calendar=None)
        assert list(classes[classes == "irregular"].index) == [self.HOURS.index[11]]

    def test_weekday_only_data_warns_naming_24_5(self):
        values = series(pd.bdate_range("2025-06-02", periods=15))
        with pytest.warns(UserWarning, match=r'2 of 14 returns.*calendar="24/5"'):
            classes = classify_returns(values, calendar=None)
        assert (classes == "irregular").sum() == 2

    def test_to_session_values_takes_each_utc_midnight(self):
        out = to_session_values(self.HOURS, calendar=None)
        midnights = [pd.Timestamp(f"2025-06-0{d}", tz="UTC") for d in (3, 4, 5)]
        assert list(out.index) == [self.HOURS.index[0]] + midnights
        assert list(out) == [self.HOURS.iloc[0]] + [self.HOURS[m] for m in midnights]

    def test_to_session_values_uses_the_last_mark_before_midnight(self):
        # No mark at 2025-06-04 00:00: its value is the 23:00 mark's, still
        # labelled at the midnight, in the series' own timezone
        values = self.HOURS.drop(pd.Timestamp("2025-06-04", tz="UTC")).tz_convert("Asia/Tokyo")
        out = to_session_values(values, calendar=None)
        assert out.iloc[2] == self.HOURS[pd.Timestamp("2025-06-03 23:00", tz="UTC")]
        assert out.index[2] == pd.Timestamp("2025-06-04", tz="UTC") and str(out.index.tz) == "Asia/Tokyo"

    def test_to_session_values_first_mark_at_midnight_is_kept_once(self):
        values = self.HOURS[pd.Timestamp("2025-06-03", tz="UTC"):]
        out = to_session_values(values, calendar=None)
        assert list(out.index) == [pd.Timestamp(f"2025-06-0{d}", tz="UTC") for d in (3, 4, 5)]

    def test_to_session_values_daily_comes_back_unchanged(self):
        days = series(pd.date_range("2025-06-01", periods=5, freq="D", tz="UTC"))
        assert to_session_values(days, calendar=None).equals(days)

    def test_to_session_values_naive_intraday_raises(self):
        with pytest.raises(ValueError, match="localize"):
            to_session_values(self.HOURS.tz_localize(None), calendar=None)


def test_sparse_marks_learn_the_bar_size_from_the_open():
    # B4.1 on labels: 09:30 open mark, 10:30, 12:30, 13:00 on the 2024-11-29
    # half-day; the hourly grid's 11:30 mark is missing
    values = series([ny("2024-11-29 09:30"), ny("2024-11-29 10:30"), ny("2024-11-29 12:30"), ny("2024-11-29 13:00")])
    assert list(classify_returns(values)) == ["regular", "irregular", "regular"]
