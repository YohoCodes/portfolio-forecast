import pandas as pd
import pytest

from portfolio_forecast.utils import close_times

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def hourly(day):
    """A session of hourly bars as yfinance and IB label them: a 9:30 bar,
    then one on every hour to 15:00."""
    return pd.DatetimeIndex([ny(f"{day} 09:30")] + [ny(f"{day} {h}:00") for h in range(10, 16)])


class TestIntraday:
    def test_hourly_bars_close_on_the_grid_and_at_the_session_close(self):
        closes = close_times(hourly("2025-06-04"))
        assert [t.strftime("%H:%M") for t in closes] == [
            "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]

    def test_close_is_the_next_start_within_a_session(self):
        index = hourly("2025-06-04").append(hourly("2025-06-05"))
        closes = close_times(index)
        assert (closes[:6] == index[1:7]).all()
        assert closes[6] == ny("2025-06-04 16:00")

    def test_half_day_closes_at_13_00(self):
        # 2025-11-28, the day after Thanksgiving
        index = hourly("2025-11-28")[:4]
        assert close_times(index)[-1] == ny("2025-11-28 13:00")

    @pytest.mark.parametrize("day", ["2025-01-15", "2025-07-15"])
    def test_closes_stay_on_their_session_date(self, day):
        closes = close_times(hourly(day))
        assert (closes.date == hourly(day).date).all()
        assert closes[-1].hour == 16

    def test_missing_bar_does_not_stretch_the_one_before(self):
        index = hourly("2025-06-04").delete(3)
        assert close_times(index)[2] == ny("2025-06-04 12:00")

    @pytest.mark.parametrize("minutes", [5, 30])
    def test_other_bar_sizes(self, minutes):
        starts = pd.date_range("2025-06-04 09:30", "2025-06-04 15:59",
                               freq=f"{minutes}min", tz=NY)
        assert (close_times(starts) - starts == pd.Timedelta(minutes=minutes)).all()

    def test_keeps_the_input_timezone_and_name(self):
        index = hourly("2025-06-04").tz_convert("UTC").rename("Datetime")
        closes = close_times(index)
        assert str(closes.tz) == "UTC" and closes.name == "Datetime"
        assert closes[-1] == ny("2025-06-04 16:00")

    def test_naive_intraday_is_read_in_the_given_timezone(self):
        naive = hourly("2025-06-04").tz_localize(None)
        closes = close_times(naive, tz=NY)
        assert closes.tz is None
        assert closes[-1] == pd.Timestamp("2025-06-04 16:00")

    def test_aware_daily_bars_close_at_the_session_close(self):
        index = pd.DatetimeIndex([ny("2025-06-04 09:30"), ny("2025-11-28 09:30")])
        assert list(close_times(index)) == [ny("2025-06-04 16:00"), ny("2025-11-28 13:00")]


class TestDatesAndErrors:
    def test_daily_dates_close_at_their_session_close(self):
        closes = close_times(pd.DatetimeIndex(["2025-11-26", "2025-11-28"]))
        assert list(closes) == [ny("2025-11-26 16:00"), ny("2025-11-28 13:00")]

    def test_weekly_dates_pass_through(self):
        index = pd.DatetimeIndex(["2025-06-02", "2025-06-09", "2025-06-16"])
        assert close_times(index).equals(index)

    def test_bar_outside_a_session_raises(self):
        with pytest.raises(ValueError, match="outside XNYS sessions"):
            close_times(pd.DatetimeIndex([ny("2025-06-07 10:00")]))  # a Saturday

    def test_date_outside_a_session_raises(self):
        with pytest.raises(ValueError, match="outside XNYS sessions"):
            close_times(pd.DatetimeIndex(["2025-06-06", "2025-06-07"]))

    def test_empty(self):
        assert len(close_times(pd.DatetimeIndex([]))) == 0
