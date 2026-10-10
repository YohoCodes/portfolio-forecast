"""history_marks: the marks, freshness and classes of a backtest (B1-B4).
Expected labels and classes are written out by hand from the rules."""
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast.utils import classify_returns
from portfolio_forecast.utils.marks import history_marks

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def hourly(day, first="09:30", last="15:00"):
    """Start labels of hourly bars: 9:30, then every hour to `last`."""
    hours = [f"{h}:00" for h in range(10, int(last[:2]) + 1)]
    times = ["09:30"] + hours
    return pd.DatetimeIndex([ny(f"{day} {t}") for t in times if t >= first])


def closes_of(day, last="16:00"):
    return [ny(f"{day} {h}:00") for h in range(10, int(last[:2]) + 1)]


def frames(index, symbols=("A",)):
    """Opens and closes of 1.0 for every bar and symbol."""
    px = pd.DataFrame(1.0, index=index, columns=list(symbols))
    return px.copy(), px.copy()


# Friday 2025-06-06 and Monday 2025-06-09: the gap spans a weekend
FRI, MON = "2025-06-06", "2025-06-09"
TWO_SESSIONS = hourly(FRI).append(hourly(MON))


class TestIntraday:
    def test_open_mark_then_bar_closes_with_a_weekend_gap(self):
        # B1.1-B1.4, B2 fill="open": each session's open, then its 7 closes
        labels, fresh, classes = history_marks(*frames(TWO_SESSIONS), ["A"])
        assert list(labels) == [ny(f"{FRI} 09:30"), *closes_of(FRI), ny(f"{MON} 09:30"), *closes_of(MON)]
        assert list(classes) == ["regular"] * 7 + ["gap"] + ["regular"] * 7
        assert classes.index.equals(labels[1:])
        assert fresh.to_numpy().all()

    def test_fill_close_has_no_first_open_mark(self):
        # B2 fill="close": session 1 starts at its first bar's close
        labels, _, classes = history_marks(*frames(TWO_SESSIONS), ["A"], fill="close")
        assert list(labels) == [*closes_of(FRI), ny(f"{MON} 09:30"), *closes_of(MON)]
        assert list(classes) == ["regular"] * 6 + ["gap"] + ["regular"] * 7

    def test_labels_are_on_the_exchange_clock(self):
        # B3: UTC input, labels tz-aware in New York time
        labels, _, _ = history_marks(*frames(TWO_SESSIONS.tz_convert("UTC")), ["A"])
        assert str(labels.tz) == NY and labels[0] == ny(f"{FRI} 09:30")

    def test_naive_input_read_in_tz(self):
        labels, _, _ = history_marks(*frames(TWO_SESSIONS.tz_localize(None)), ["A"], tz=NY)
        assert labels[0] == ny(f"{FRI} 09:30") and str(labels.tz) == NY

    def test_half_day_last_close_at_13_00(self):
        index = hourly("2024-11-27").append(hourly("2024-11-29", last="12:00"))
        labels, _, classes = history_marks(*frames(index), ["A"])
        assert labels[-1] == ny("2024-11-29 13:00")
        assert labels[8] == ny("2024-11-29 09:30") and classes.iloc[7] == "gap"

    def test_missing_first_bar_of_a_session(self):
        # B1.6: Monday's data starts at 10:00, so its open mark is at 10:00
        # and the return ending there is irregular; the next bar is regular
        index = hourly(FRI).append(hourly(MON, first="10:00"))
        labels, fresh, classes = history_marks(*frames(index), ["A"])
        assert labels[8] == ny(f"{MON} 10:00") and labels[9] == ny(f"{MON} 11:00")
        assert classes.loc[ny(f"{MON} 10:00")] == "irregular"
        assert classes.loc[ny(f"{MON} 11:00")] == "regular"
        assert fresh.to_numpy().all()

    def test_single_symbol_missing_open(self):
        # B4.1: A has no open on Monday's first bar; the gap and the bar after
        # it have A stale at one end
        opens, closes = frames(TWO_SESSIONS, ("A", "B"))
        opens.loc[ny(f"{MON} 09:30"), "A"] = np.nan
        labels, fresh, classes = history_marks(opens, closes, ["A", "B"])
        assert labels[8] == ny(f"{MON} 09:30")
        assert not fresh.loc[ny(f"{MON} 09:30"), "A"] and fresh.loc[ny(f"{MON} 09:30"), "B"]
        assert classes.loc[ny(f"{MON} 09:30")] == "irregular"
        assert classes.loc[ny(f"{MON} 10:00")] == "irregular"
        assert classes.loc[ny(f"{MON} 11:00")] == "regular"
        # Not held, A's staleness doesn't matter
        _, _, classes = history_marks(opens, closes, ["B"])
        assert classes.loc[ny(f"{MON} 09:30")] == "gap"

    def test_held_as_weights(self):
        opens, closes = frames(TWO_SESSIONS, ("A", "B"))
        opens.loc[ny(f"{MON} 09:30"), "A"] = np.nan
        _, _, classes = history_marks(opens, closes, pd.Series({"A": 0.0, "B": 1.0}))
        assert classes.loc[ny(f"{MON} 09:30")] == "gap"

    def test_mid_session_bar_missing_for_every_symbol(self):
        # B4.1: the 12:00 row is absent, so there is no 13:00 mark and the
        # 12:00 -> 14:00 return spans a scheduled mark: irregular, though
        # every symbol is fresh at both ends
        index = TWO_SESSIONS.delete(3)
        labels, _, classes = history_marks(*frames(index, ("A", "B")), ["A", "B"])
        assert ny(f"{FRI} 13:00") not in labels
        assert list(classes[classes == "irregular"].index) == [ny(f"{FRI} 14:00")]

    def test_mid_session_bar_missing_for_one_symbol(self):
        # A has no close on the 12:00 bar: A is stale at the 13:00 mark, so
        # the returns ending at 13:00 and 14:00 are irregular
        opens, closes = frames(TWO_SESSIONS, ("A", "B"))
        opens.loc[ny(f"{FRI} 12:00"), "A"] = np.nan
        closes.loc[ny(f"{FRI} 12:00"), "A"] = np.nan
        labels, fresh, classes = history_marks(opens, closes, ["A", "B"])
        assert not fresh.loc[ny(f"{FRI} 13:00"), "A"]
        assert list(classes[classes == "irregular"].index) == [ny(f"{FRI} 13:00"), ny(f"{FRI} 14:00")]

    def test_symbol_stopping_two_sessions_early_is_cash(self):
        # B4.3: A's prices end with Tuesday; Wednesday and Thursday it is
        # cash, always fresh, so the returns keep their class
        days = ["2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12"]
        index = hourly(days[0]).append([hourly(d) for d in days[1:]])
        opens, closes = frames(index, ("A", "B"))
        late = index > ny(f"{days[1]} 16:00")
        opens.loc[late, "A"] = np.nan
        closes.loc[late, "A"] = np.nan
        labels, fresh, classes = history_marks(opens, closes, ["A", "B"])
        assert fresh.to_numpy().all()
        assert list(classes) == (["regular"] * 7 + ["gap"]) * 3 + ["regular"] * 7

    def test_intraday_without_opens_raises(self):
        # B1.5
        _, closes = frames(TWO_SESSIONS)
        with pytest.raises(ValueError, match="need opens"):
            history_marks(None, closes, ["A"], fill="close")

    def test_unknown_held_symbol_raises(self):
        with pytest.raises(ValueError, match="Z"):
            history_marks(*frames(TWO_SESSIONS), ["Z"])


class TestDailyAndLonger:
    DAYS = pd.DatetimeIndex(["2025-06-05", "2025-06-06", "2025-06-09"])

    def test_daily_fill_open(self):
        # B2: the fill mark at session 1's open, then each session's close
        labels, _, classes = history_marks(*frames(self.DAYS), ["A"])
        assert list(labels) == [ny("2025-06-05 09:30"), ny("2025-06-05 16:00"),
                                ny("2025-06-06 16:00"), ny("2025-06-09 16:00")]
        assert list(classes) == ["regular"] * 3

    def test_daily_fill_close_needs_no_opens(self):
        _, closes = frames(self.DAYS)
        labels, _, classes = history_marks(None, closes, ["A"], fill="close")
        assert list(labels) == [ny("2025-06-05 16:00"), ny("2025-06-06 16:00"), ny("2025-06-09 16:00")]
        assert list(classes) == ["regular"] * 2

    def test_fill_open_without_opens_raises(self):
        _, closes = frames(self.DAYS)
        with pytest.raises(ValueError, match="fill='close'"):
            history_marks(None, closes, ["A"])

    def test_daily_stale_symbol_is_irregular(self):
        # B4.1 on daily bars: A has no close on 2025-06-06
        opens, closes = frames(self.DAYS, ("A", "B"))
        closes.loc["2025-06-06", "A"] = np.nan
        _, _, classes = history_marks(opens, closes, ["A", "B"])
        assert list(classes) == ["regular", "irregular", "irregular"]

    def test_weekly_fill_open_across_a_holiday_week(self):
        # B3: the fill at the first week's Monday open, then each week's
        # last close; Thanksgiving week ends Friday 13:00
        weeks = pd.DatetimeIndex(["2024-11-18", "2024-11-25", "2024-12-02"])
        labels, _, classes = history_marks(*frames(weeks), ["A"])
        assert list(labels) == [ny("2024-11-18 09:30"), ny("2024-11-22 16:00"),
                                ny("2024-11-29 13:00"), ny("2024-12-06 16:00")]
        assert list(classes) == ["regular"] * 3


class TestMissingScheduledMarks:
    """B4.1: a return spanning a scheduled mark that no symbol has is irregular."""

    def test_whole_session_missing(self):
        # Monday 2025-06-09 is an ordinary session with no data: the return
        # from Friday's close to Tuesday's open spans it
        index = hourly(FRI).append(hourly("2025-06-10"))
        labels, _, classes = history_marks(*frames(index), ["A"])
        assert classes.loc[ny("2025-06-10 09:30")] == "irregular"
        assert (classes == "irregular").sum() == 1

    def test_holiday_is_not_missing(self):
        # Thursday 2025-06-19 (Juneteenth) is not a session: Wednesday's
        # close to Friday's open is a gap
        index = hourly("2025-06-18").append(hourly("2025-06-20"))
        _, _, classes = history_marks(*frames(index), ["A"])
        assert classes.loc[ny("2025-06-20 09:30")] == "gap"
        assert (classes == "irregular").sum() == 0

    def test_missing_last_bar_makes_the_gap_irregular(self):
        # Friday's 15:00 bar is absent: its 16:00 close is scheduled, so the
        # return from 15:00 to Monday's open spans it
        index = TWO_SESSIONS.delete(6)
        labels, _, classes = history_marks(*frames(index), ["A"])
        assert labels[6] == ny(f"{FRI} 15:00")
        assert classes.loc[ny(f"{MON} 09:30")] == "irregular"

    def test_daily_missing_session_and_holiday(self):
        # 2025-07-03 to 2025-07-07: the 4th is a holiday (not missing); the
        # 8th is missing, so the return ending on the 9th is irregular
        days = pd.DatetimeIndex(["2025-07-02", "2025-07-03", "2025-07-07", "2025-07-09"])
        _, _, classes = history_marks(*frames(days), ["A"])
        assert list(classes) == ["regular", "regular", "regular", "irregular"]

    def test_weekly_missing_week(self):
        weeks = pd.DatetimeIndex(["2025-06-02", "2025-06-09", "2025-06-23"])
        _, _, classes = history_marks(*frames(weeks), ["A"], fill="close")
        assert list(classes) == ["regular", "irregular"]


class TestOutOfSessionBars:
    """B4.4: intraday bars outside every session are dropped, with
    one warning giving the count."""

    @staticmethod
    def clean_labels():
        labels, _, _ = history_marks(*frames(TWO_SESSIONS), ["A"])
        return labels

    def test_pre_market_bar(self):
        index = pd.DatetimeIndex([ny(f"{FRI} 08:00")]).append(TWO_SESSIONS)
        with pytest.warns(UserWarning, match=r"Dropped 1 intraday bar\(s\) outside XNYS sessions") as caught:
            labels, _, classes = history_marks(*frames(index), ["A"])
        assert len(caught) == 1
        assert labels.equals(self.clean_labels())
        assert (classes == "irregular").sum() == 0

    def test_post_market_bars(self):
        index = hourly(FRI).append(pd.DatetimeIndex([ny(f"{FRI} 16:00"), ny(f"{FRI} 17:00")])).append(hourly(MON))
        with pytest.warns(UserWarning, match=r"Dropped 2 intraday bar\(s\)"):
            labels, _, _ = history_marks(*frames(index), ["A"])
        assert labels.equals(self.clean_labels())

    def test_saturday_bar(self):
        index = hourly(FRI).append(pd.DatetimeIndex([ny("2025-06-07 10:00")])).append(hourly(MON))
        with pytest.warns(UserWarning, match=r"Dropped 1 intraday bar\(s\).*first 2025-06-07 10:00"):
            labels, _, classes = history_marks(*frames(index), ["A"])
        assert labels.equals(self.clean_labels())
        assert classes.loc[ny(f"{MON} 09:30")] == "gap"

    def test_one_warning_for_every_kind(self):
        index = (pd.DatetimeIndex([ny(f"{FRI} 08:00")]).append(hourly(FRI))
                 .append(pd.DatetimeIndex([ny(f"{FRI} 17:00"), ny("2025-06-08 12:00")])).append(hourly(MON)))
        with pytest.warns(UserWarning) as caught:
            history_marks(*frames(index), ["A"])
        assert [str(w.message)[:42] for w in caught] == ["Dropped 3 intraday bar(s) outside XNYS ses"]

    def test_daily_bar_on_a_holiday_raises(self):
        # B4.4: dates name sessions, so a holiday date is an error, not dropped
        days = pd.DatetimeIndex(["2025-07-03", "2025-07-04", "2025-07-07"])
        with pytest.raises(ValueError, match="outside XNYS sessions"):
            history_marks(*frames(days), ["A"])


def utc(text):
    return pd.Timestamp(text, tz="UTC")


class TestZeroLengthGaps:
    """B1.4 on "24/5" (sessions 00:00 to 00:00 UTC, Monday to Friday): only the
    weekend is a gap; weeknight session changes are ordinary bars."""

    # Hourly bars from Tuesday 2025-06-03 00:00 to Monday 2025-06-09 23:00 UTC
    INDEX = pd.date_range("2025-06-03", "2025-06-10", freq="h", tz="UTC", inclusive="left")
    INDEX = INDEX[INDEX.dayofweek < 5]

    def test_one_gap_on_monday_and_no_weeknight_open_marks(self):
        labels, fresh, classes = history_marks(*frames(self.INDEX), ["A"], calendar="24/5")
        # The fill at Tuesday 00:00, each bar's close, and Monday's open mark
        assert len(labels) == 1 + len(self.INDEX) + 1
        assert labels[0] == utc("2025-06-03 00:00")
        assert list(classes[classes != "regular"].index) == [utc("2025-06-09 00:00")]
        assert classes.loc[utc("2025-06-09 00:00")] == "gap"
        # Friday's close is Saturday 00:00, then Monday's open
        friday_close = labels.get_loc(utc("2025-06-07 00:00"))
        assert labels[friday_close + 1] == utc("2025-06-09 00:00")
        # Weeknights: Tuesday's close (Wednesday 00:00) is followed by 01:00
        assert labels[labels.get_loc(utc("2025-06-04 00:00")) + 1] == utc("2025-06-04 01:00")
        assert fresh.to_numpy().all()

    def test_classify_returns_agrees(self):
        labels, _, classes = history_marks(*frames(self.INDEX), ["A"], calendar="24/5")
        values = pd.Series(1.0, index=labels)
        assert classify_returns(values, calendar="24/5").equals(classes)

    def test_fill_close_starts_at_the_first_close(self):
        labels, _, classes = history_marks(*frames(self.INDEX), ["A"], calendar="24/5", fill="close")
        assert labels[0] == utc("2025-06-03 01:00")
        assert (classes == "gap").sum() == 1

    def test_xnys_output_unchanged(self):
        # XNYS has no zero-length gaps: every session keeps its open mark
        labels, _, classes = history_marks(*frames(TWO_SESSIONS), ["A"])
        assert list(classes) == ["regular"] * 7 + ["gap"] + ["regular"] * 7


class TestContinuousMode:
    """U3: calendar=None is a constant grid with no sessions."""

    HOURS = pd.date_range("2025-06-06 20:00", periods=10, freq="h", tz="UTC")  # across a weekend night
    # Long enough that one or two irregular returns stay under U3's 10% warning threshold
    LONG = pd.date_range("2025-06-06 20:00", periods=30, freq="h", tz="UTC")

    def test_marks_on_the_grid_all_regular(self):
        labels, fresh, classes = history_marks(*frames(self.HOURS), ["A"], calendar=None)
        # Fill at the first bar's start, then each bar's start + 1 hour
        assert labels.equals(self.HOURS.append(pd.DatetimeIndex([self.HOURS[-1] + pd.Timedelta(hours=1)])))
        assert list(classes) == ["regular"] * 10
        assert fresh.to_numpy().all()

    def test_fill_close_has_no_open_mark_and_needs_no_opens(self):
        _, closes = frames(self.HOURS)
        labels, _, classes = history_marks(None, closes, ["A"], calendar=None, fill="close")
        assert labels.equals(self.HOURS + pd.Timedelta(hours=1))
        assert list(classes) == ["regular"] * 9

    def test_missing_grid_point_is_irregular(self):
        labels, _, classes = history_marks(*frames(self.LONG.delete(4)), ["A"], calendar=None)
        assert list(classes[classes == "irregular"].index) == [self.LONG[5] + pd.Timedelta(hours=1)]

    def test_stale_symbol_is_irregular(self):
        opens, closes = frames(self.LONG, ("A", "B"))
        closes.iloc[3, 0] = np.nan
        _, _, classes = history_marks(opens, closes, ["A", "B"], calendar=None)
        assert (classes == "irregular").sum() == 2

    def test_naive_dates_are_localized_in_tz(self):
        days = pd.date_range("2025-06-01", periods=5, freq="D")
        labels, _, classes = history_marks(*frames(days), ["A"], calendar=None, tz="UTC", fill="close")
        assert labels[0] == utc("2025-06-02") and str(labels.tz) == "UTC"
        assert list(classes) == ["regular"] * 4

    def test_weekday_only_daily_data_warns(self):
        # U3: every Monday return spans a weekend, so 1 in 5 is irregular
        days = pd.bdate_range("2025-06-02", periods=15, tz="UTC")
        with pytest.warns(UserWarning, match=r'calendar="24/5"') as caught:
            _, _, classes = history_marks(*frames(days), ["A"], calendar=None)
        assert len(caught) == 1
        assert (classes == "irregular").sum() == 2

    def test_few_irregular_returns_do_not_warn(self, recwarn):
        history_marks(*frames(self.LONG.delete(4)), ["A"], calendar=None)  # 1 of 29 is under 10%
        assert not recwarn

    def test_single_bar_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            history_marks(*frames(self.HOURS[:1]), ["A"], calendar=None)
