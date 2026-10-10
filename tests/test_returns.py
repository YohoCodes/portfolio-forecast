"""Return helpers (M1, M2, M9, M10, U3). Expected returns are computed
by hand from the values."""
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast import BacktestResult
from portfolio_forecast.performance import (
    calculate_daily_return,
    calculate_period_returns,
    calculate_session_returns,
)
from portfolio_forecast.performance.returns import _risk_periods_per_year

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def utc(text):
    return pd.Timestamp(text, tz="UTC")


def series(labels, values):
    return pd.Series([float(v) for v in values], index=pd.DatetimeIndex(labels))


# Two hourly sessions, fill="open": the 9:30 fill, 10:00 to 16:00, then the
# next session's open mark and closes
FILL_OPEN = series(
    [ny("2025-06-04 09:30")] + [ny(f"2025-06-04 {h}:00") for h in range(10, 17)]
    + [ny("2025-06-05 09:30")] + [ny(f"2025-06-05 {h}:00") for h in range(10, 17)],
    [100, 101, 102, 101, 103, 104, 105, 106, 104, 105, 106, 107, 108, 109, 110, 111],
)


class TestPeriodReturns:
    def test_between_consecutive_marks(self):
        r = calculate_period_returns(FILL_OPEN)
        assert r.index.equals(FILL_OPEN.index[1:])
        assert r.iloc[0] == pytest.approx(0.01) and r.iloc[7] == pytest.approx(104 / 106 - 1)

    def test_backtest_result(self):
        result = BacktestResult(FILL_OPEN, FILL_OPEN.index, pd.Series("regular", index=FILL_OPEN.index[1:]))
        assert calculate_period_returns(result).equals(calculate_period_returns(FILL_OPEN))

    def test_returns_after_ruin_are_nan(self):
        # M9: 0 -> 0 is 0/0, kept as NaN; the -100% return itself stays
        values = series(pd.bdate_range("2025-06-02", periods=4), [100, 50, 0, 0])
        r = calculate_period_returns(values)
        assert r.index.equals(values.index[1:])
        assert r.iloc[:2].tolist() == [-0.5, -1.0] and np.isnan(r.iloc[2])

    def test_intraday_ruin_mid_history_keeps_every_row(self):
        # M9: FILL_OPEN hits 0 at the 4th's 13:00 close and stays there: the
        # 13:00 return is -1, every later one is 0/0 = NaN, and the length
        # is still len(values) - 1
        values = FILL_OPEN.copy()
        values.iloc[4:] = 0.0
        r = calculate_period_returns(values)
        assert len(r) == len(values) - 1 and r.index.equals(values.index[1:])
        assert r.iloc[:3].tolist() == pytest.approx([0.01, 102 / 101 - 1, 101 / 102 - 1])
        assert r.iloc[3] == -1.0
        assert r.iloc[4:].isna().all()

    def test_deprecated_alias_warns_and_matches(self):
        with pytest.warns(DeprecationWarning, match="1.0.0; use calculate_period_returns"):
            old = calculate_daily_return(FILL_OPEN)
        assert old.equals(calculate_period_returns(FILL_OPEN))


class TestSessionReturns:
    def test_intraday_partial_first_session_excluded(self):
        # M1, M2: R_1 (fill 9:30 to the 4th's close) is partial; R_2 is close to close
        r = calculate_session_returns(FILL_OPEN)
        assert list(r.index) == [ny("2025-06-05 16:00")]
        assert r.iloc[0] == pytest.approx(111 / 106 - 1)

    def test_intraday_fill_close_gives_the_same_series(self):
        fill_close = FILL_OPEN.iloc[1:]
        pd.testing.assert_series_equal(calculate_session_returns(fill_close), calculate_session_returns(FILL_OPEN))

    def test_daily_fill_close_keeps_the_first_return(self):
        values = series([ny("2025-06-04 16:00"), ny("2025-06-05 16:00"), ny("2025-06-06 16:00")], [100, 110, 99])
        assert calculate_session_returns(values).tolist() == pytest.approx([0.1, -0.1])

    def test_daily_dates_keep_the_first_return(self):
        values = series(pd.DatetimeIndex(["2025-06-04", "2025-06-05", "2025-06-06"]), [100, 110, 99])
        assert calculate_session_returns(values).tolist() == pytest.approx([0.1, -0.1])

    def test_daily_fill_open_drops_the_open_to_close_return(self):
        values = series([ny("2025-06-04 09:30"), ny("2025-06-04 16:00"), ny("2025-06-05 16:00")], [100, 102, 110.16])
        assert calculate_session_returns(values).tolist() == pytest.approx([0.08])

    def test_weekly_own_bar_returns(self):
        # M2: fill="close" keeps R_1; fill="open" drops the first week's open-to-close
        closes = [ny("2025-06-06 16:00"), ny("2025-06-13 16:00"), ny("2025-06-20 16:00")]
        assert calculate_session_returns(series(closes, [100, 105, 102.9])).tolist() == pytest.approx([0.05, -0.02])
        with_fill = series([ny("2025-06-02 09:30")] + closes, [98, 100, 105, 102.9])
        assert calculate_session_returns(with_fill).tolist() == pytest.approx([0.05, -0.02])

    def test_ruin_keeps_the_minus_100_percent_session(self):
        # M9: the session after ruin is 0/0, kept as NaN
        values = series(pd.DatetimeIndex(["2025-06-04", "2025-06-05", "2025-06-06", "2025-06-09"]), [100, 50, 0, 0])
        r = calculate_session_returns(values)
        assert r.index.equals(values.index[1:])
        assert r.iloc[:2].tolist() == [-0.5, -1.0] and np.isnan(r.iloc[2])

    def test_intraday_ruin_mid_history_one_row_per_full_session(self):
        # M2, M9: three hourly sessions, fill="open"; the value hits 0 at the
        # 5th's 13:00 close. R_1 is partial and left out; R_2 (4th close to
        # 5th close) is -1; R_3 (5th close to 6th close) is 0/0 = NaN. One
        # row per full session either way
        labels = (list(FILL_OPEN.index)
                  + [ny("2025-06-06 09:30")] + [ny(f"2025-06-06 {h}:00") for h in range(10, 17)])
        values = series(labels, list(FILL_OPEN) + [0] * 8)
        values.iloc[12:] = 0.0
        r = calculate_session_returns(values)
        assert list(r.index) == [ny("2025-06-05 16:00"), ny("2025-06-06 16:00")]
        assert r.iloc[0] == -1.0 and np.isnan(r.iloc[1])

    def test_continuous_ruin_keeps_the_utc_day(self):
        # M9, U3: hits 0 at 2025-06-03 00:00, so the day ending then is -1
        # and the day ending 2025-06-04 is 0/0 = NaN; the partial first day
        # is left out as usual
        values = series([utc("2025-06-01 12:00"), utc("2025-06-02 00:00"), utc("2025-06-02 12:00"),
                         utc("2025-06-03 00:00"), utc("2025-06-03 12:00"), utc("2025-06-04 00:00")],
                        [100, 101, 102, 0, 0, 0])
        r = calculate_session_returns(values, calendar=None)
        assert list(r.index) == [utc("2025-06-03"), utc("2025-06-04")]
        assert r.iloc[0] == -1.0 and np.isnan(r.iloc[1])


class TestContinuousSessionReturns:
    """M1, U3: calendar-day returns at 00:00 UTC."""

    def test_utc_day_returns_without_the_partial_first_day(self):
        values = series([utc("2025-06-01 18:00"), utc("2025-06-02 00:00"), utc("2025-06-02 12:00"),
                         utc("2025-06-03 00:00"), utc("2025-06-03 12:00"), utc("2025-06-04 00:00")],
                        [100, 101, 102, 103, 104, 105])
        r = calculate_session_returns(values, calendar=None)
        assert list(r.index) == [utc("2025-06-03"), utc("2025-06-04")]
        assert r.tolist() == pytest.approx([103 / 101 - 1, 105 / 103 - 1])

    def test_first_day_is_excluded_even_from_midnight(self):
        # M2, U3: the first UTC day is always left out, so fill="open" at 00:00
        # and fill="close" an hour later give the same series
        values = series([utc("2025-06-02 00:00"), utc("2025-06-02 12:00"), utc("2025-06-03 00:00"),
                         utc("2025-06-03 12:00"), utc("2025-06-04 00:00")], [100, 101, 102, 103, 104])
        r = calculate_session_returns(values, calendar=None)
        assert list(r.index) == [utc("2025-06-04")]
        assert r.tolist() == pytest.approx([104 / 102 - 1])

    def test_bars_longer_than_a_day_use_their_own_returns(self):
        # M1, U3: weekly 24/7 bars, own returns annualized with 365 / 7
        values = series(pd.date_range("2025-06-01", periods=4, freq="7D", tz="UTC"), [100, 110, 99, 99])
        assert calculate_session_returns(values, calendar=None).tolist() == pytest.approx([0.1, -0.1, 0.0])
        assert _risk_periods_per_year(values, calendar=None) == pytest.approx(365 / 7)

    def test_daily_input_uses_its_own_returns(self):
        values = series(pd.date_range("2025-06-01 12:00", periods=4, freq="D", tz="UTC"), [100, 110, 99, 99])
        assert calculate_session_returns(values, calendar=None).tolist() == pytest.approx([0.1, -0.1, 0.0])


class TestAnnualization:
    @pytest.mark.parametrize("labels, calendar, expected", [
        (FILL_OPEN.index, "XNYS", 252),
        (pd.DatetimeIndex(["2025-06-04", "2025-06-05", "2025-06-06"]), "XNYS", 252),
        (pd.DatetimeIndex(["2025-06-06", "2025-06-13", "2025-06-20"]), "XNYS", 52),
        (pd.DatetimeIndex(["2025-03-31", "2025-04-30", "2025-05-30"]), "XNYS", 12),
        (pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC"), None, 365),
    ], ids=["intraday", "daily", "weekly", "monthly", "continuous"])
    def test_periods_per_year(self, labels, calendar, expected):
        assert _risk_periods_per_year(series(labels, np.ones(len(labels))), calendar) == expected

    def test_quarterly_bars_raise(self):
        labels = pd.DatetimeIndex(["2025-03-31", "2025-06-30", "2025-09-30"])
        with pytest.raises(ValueError, match="No annualization factor"):
            _risk_periods_per_year(series(labels, [1, 1, 1]))

    def test_interval_overrides_the_inferred_bar(self):
        labels = pd.DatetimeIndex(["2025-06-06", "2025-06-13", "2025-06-20"])
        assert _risk_periods_per_year(series(labels, [1, 1, 1]), interval="1 month") == 12
        with pytest.raises(ValueError, match="intraday"):
            _risk_periods_per_year(series(labels, [1, 1, 1]), interval="5 mins")
