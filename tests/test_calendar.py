"""Bars on the exchange calendar (B3, B6): session spans and the
completed-bar cutoff. Expected times are read off the NYSE calendar."""
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast.utils import bar_times
from portfolio_forecast.utils.calendar import close_times, completed_mask, session_spans

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def hourly(day, last="15:00"):
    """A session of hourly bars, start-labelled: 9:30, then every hour."""
    hours = [f"{h}:00" for h in range(10, int(last[:2]) + 1)]
    return pd.DatetimeIndex([ny(f"{day} 09:30")] + [ny(f"{day} {h}") for h in hours])


class TestSessionSpans:
    def test_columns(self):
        spans = session_spans(hourly("2025-06-04"))
        assert list(spans.columns) == ["first_open", "last_close", "n_sessions"]

    def test_half_day_2024_11_29_closes_at_13_00(self):
        # Day after Thanksgiving: 13:00 close; the 12:00 bar is the last
        index = hourly("2024-11-29", last="12:00")
        spans = session_spans(index)
        assert (spans["first_open"] == ny("2024-11-29 09:30")).all()
        assert (spans["last_close"] == ny("2024-11-29 13:00")).all()
        assert (spans["n_sessions"] == 1).all()
        assert close_times(index)[-1] == ny("2024-11-29 13:00")

    def test_first_session_after_spring_forward_2024_03_11(self):
        # Clocks went forward on Sunday 2024-03-10: the open moves from
        # 14:30 UTC to 13:30 UTC, and stays 09:30 on the exchange's clock
        index = hourly("2024-03-08").append(hourly("2024-03-11")).tz_convert("UTC")
        spans = session_spans(index)
        friday, monday = spans.iloc[0], spans.iloc[-1]
        assert friday["first_open"] == pd.Timestamp("2024-03-08 14:30", tz="UTC")
        assert monday["first_open"] == pd.Timestamp("2024-03-11 13:30", tz="UTC")
        assert monday["last_close"] == pd.Timestamp("2024-03-11 20:00", tz="UTC")
        assert str(monday["first_open"].tz) == NY
        closes = close_times(index)
        assert closes[7] == pd.Timestamp("2024-03-11 14:00", tz="UTC")  # 10:00 on the exchange clock

    def test_naive_intraday_read_in_tz(self):
        spans = session_spans(hourly("2025-06-04").tz_localize(None), tz=NY)
        assert spans["last_close"].iloc[0] == ny("2025-06-04 16:00")

    def test_daily_bars(self):
        spans = session_spans(pd.DatetimeIndex(["2024-11-27", "2024-11-29"]))
        assert list(spans["first_open"]) == [ny("2024-11-27 09:30"), ny("2024-11-29 09:30")]
        assert list(spans["last_close"]) == [ny("2024-11-27 16:00"), ny("2024-11-29 13:00")]
        assert list(spans["n_sessions"]) == [1, 1]

    def test_weekly_bar_spanning_a_holiday(self):
        # Week of 2024-11-25: Thanksgiving (28th) closed, 29th a half-day
        spans = session_spans(pd.DatetimeIndex(["2024-11-18", "2024-11-25", "2024-12-02"]))
        week = spans.loc["2024-11-25"]
        assert week["first_open"] == ny("2024-11-25 09:30")
        assert week["last_close"] == ny("2024-11-29 13:00")
        assert week["n_sessions"] == 4

    def test_monthly_bars_dated_on_the_first_of_the_month(self):
        # yfinance dates a month by its 1st: 2024-01-01 is a holiday.
        # January 2024: 23 weekdays less New Year's Day and MLK Day = 21.
        # March 2024: 21 weekdays less Good Friday (29th) = 20, last 28th.
        spans = session_spans(pd.DatetimeIndex(["2024-01-01", "2024-02-01", "2024-03-01"]))
        assert spans["first_open"].iloc[0] == ny("2024-01-02 09:30")
        assert spans["last_close"].iloc[0] == ny("2024-01-31 16:00")
        assert spans["n_sessions"].iloc[0] == 21
        assert spans["last_close"].iloc[2] == ny("2024-03-28 16:00")
        assert spans["n_sessions"].iloc[2] == 20

    def test_two_month_bars_labelled_at_their_last_session(self):
        spans = session_spans(pd.DatetimeIndex(["2025-01-31", "2025-03-31", "2025-05-30"]))
        assert spans["first_open"].iloc[1] == ny("2025-02-03 09:30")
        assert spans["last_close"].iloc[1] == ny("2025-03-31 16:00")

    def test_bar_on_a_saturday_raises(self):
        with pytest.raises(ValueError, match="outside XNYS sessions"):
            session_spans(pd.DatetimeIndex([ny("2025-06-06 10:00"), ny("2025-06-07 10:00")]))

    def test_daily_bar_on_a_holiday_raises(self):
        with pytest.raises(ValueError, match="outside XNYS sessions"):
            session_spans(pd.DatetimeIndex(["2025-07-03", "2025-07-04"]))

    def test_bar_after_the_close_raises(self):
        index = hourly("2025-06-04").append(pd.DatetimeIndex([ny("2025-06-04 16:00")]))
        with pytest.raises(ValueError, match="outside XNYS session hours"):
            session_spans(index)

    def test_unsorted_index_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            session_spans(hourly("2025-06-04")[::-1])

    def test_empty(self):
        assert len(session_spans(pd.DatetimeIndex([]))) == 0


class TestCompletedMask:
    def test_in_progress_hourly_bar_is_dropped(self):
        # At 12:30 the 12:00 bar is still open; the 11:00 bar closed at 12:00
        mask = completed_mask(hourly("2025-06-04", last="12:00"), now=ny("2025-06-04 12:30"))
        assert mask.tolist() == [True, True, True, False]

    def test_bar_closing_exactly_now_is_complete(self):
        mask = completed_mask(hourly("2025-06-04", last="12:00"), now=ny("2025-06-04 13:00"))
        assert mask.all()

    def test_short_last_bar_on_a_half_day(self):
        index = hourly("2024-11-29", last="12:00")
        assert completed_mask(index, now=ny("2024-11-29 13:00")).all()
        assert not completed_mask(index, now=ny("2024-11-29 12:59")).all()

    def test_daily_bar_completes_at_its_session_close(self):
        index = pd.DatetimeIndex(["2025-06-03", "2025-06-04"])
        assert completed_mask(index, now=ny("2025-06-04 15:59")).tolist() == [True, False]
        assert completed_mask(index, now=ny("2025-06-04 16:00")).tolist() == [True, True]

    def test_weekly_bar_completes_at_its_last_session_close(self):
        index = pd.DatetimeIndex(["2024-11-18", "2024-11-25"])
        assert completed_mask(index, now=ny("2024-11-29 12:00")).tolist() == [True, False]

    def test_naive_now_is_read_in_tz(self):
        index = hourly("2025-06-04", last="12:00")
        mask = completed_mask(index, tz=NY, now=pd.Timestamp("2025-06-04 12:30"))
        assert mask.tolist() == [True, True, True, False]

    def test_now_defaults_to_the_clock(self):
        assert isinstance(completed_mask(hourly("2025-06-04")), np.ndarray)
        assert completed_mask(hourly("2025-06-04")).all()


def test_bar_times_shim_still_imports():
    assert bar_times.close_times is close_times


class TestContinuousMode:
    """U3: calendar=None has no sessions; a bar closes one grid step after it starts."""

    HOURS = pd.date_range("2025-06-07 10:00", periods=5, freq="h", tz="UTC")  # a Saturday

    def test_close_times(self):
        assert close_times(self.HOURS, calendar=None).equals(self.HOURS + pd.Timedelta(hours=1))
        naive = self.HOURS.tz_localize(None)
        assert close_times(naive, calendar=None).equals(naive + pd.Timedelta(hours=1))

    def test_session_spans(self):
        spans = session_spans(self.HOURS, calendar=None)
        assert pd.DatetimeIndex(spans["first_open"]).equals(self.HOURS)
        assert pd.DatetimeIndex(spans["last_close"]).equals(self.HOURS + pd.Timedelta(hours=1))
        assert (spans["n_sessions"] == 1).all()

    def test_completed_mask(self):
        mask = completed_mask(self.HOURS, calendar=None, now=pd.Timestamp("2025-06-07 13:00", tz="UTC"))
        assert list(mask) == [True, True, True, False, False]


class TestSparseBarGrid:
    """A bar closes at its start + Δ even when the bar after it is missing."""

    def test_missing_bar_does_not_stretch_the_one_before(self):
        # 2024-11-29 half-day: bars 09:30, 10:30, 12:30 on a :30 hourly grid.
        # The 10:30 bar closes at 11:30, not 12:30; the 12:30 bar is short.
        index = pd.DatetimeIndex([ny("2024-11-29 09:30"), ny("2024-11-29 10:30"), ny("2024-11-29 12:30")])
        assert list(close_times(index)) == [ny("2024-11-29 10:30"), ny("2024-11-29 11:30"), ny("2024-11-29 13:00")]

    def test_on_the_hour_grid_keeps_its_partial_first_bar(self):
        # 09:30, 10:00, 11:00: hourly bars on :00, the 9:30 bar partial
        index = pd.DatetimeIndex([ny("2025-06-04 09:30"), ny("2025-06-04 10:00"), ny("2025-06-04 11:00")])
        assert list(close_times(index)) == [ny("2025-06-04 10:00"), ny("2025-06-04 11:00"), ny("2025-06-04 12:00")]
