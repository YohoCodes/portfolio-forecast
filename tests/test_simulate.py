import numpy as np
import pandas as pd
import pytest

from portfolio_forecast.performance import simulate_buy_and_hold
from portfolio_forecast.performance.simulate import get_closing_prices

DATES = pd.bdate_range("2025-01-02", periods=6)


@pytest.fixture
def two_assets():
    """A doubles over the sample and B is flat, so any mix is easy to price."""
    return pd.DataFrame({"A": [100, 110, 120, 150, 180, 200.0],
                         "B": [50, 50, 50, 50, 50, 50.0]}, index=DATES)


class TestGetClosingPrices:
    def test_multiindex_field_on_inner_level(self):
        cols = pd.MultiIndex.from_product([["A", "B"], ["open", "close"]])
        data = pd.DataFrame(np.arange(12.0).reshape(3, 4), columns=cols)
        assert get_closing_prices(data).columns.tolist() == ["A", "B"]

    def test_multiindex_field_on_outer_level(self):
        cols = pd.MultiIndex.from_product([["Close", "Open"], ["A", "B"]], names=["Price", "Ticker"])
        data = pd.DataFrame(np.arange(12.0).reshape(3, 4), columns=cols)
        out = get_closing_prices(data)
        assert out.columns.tolist() == ["A", "B"]
        assert out["A"].tolist() == [0.0, 4.0, 8.0]

    def test_flat_ohlcv_selects_close(self):
        data = pd.DataFrame({"open": [1.0], "close": [2.0], "volume": [3.0]})
        assert get_closing_prices(data).columns.tolist() == ["close"]

    def test_frame_of_closes_passes_through(self, two_assets):
        assert get_closing_prices(two_assets) is two_assets

    def test_multiindex_prefers_adjusted_close(self):
        # yfinance.download(..., auto_adjust=False) returns both; only Adj Close includes dividends
        cols = pd.MultiIndex.from_product([["Adj Close", "Close"], ["A", "B"]], names=["Price", "Ticker"])
        data = pd.DataFrame([[1.0, 2.0, 10.0, 20.0]], columns=cols)
        assert get_closing_prices(data).iloc[0].tolist() == [1.0, 2.0]

    def test_flat_ohlcv_prefers_adjusted_close(self):
        data = pd.DataFrame({"close": [2.0], "adj_close": [1.5], "volume": [3.0]})
        assert get_closing_prices(data).columns.tolist() == ["adj_close"]

    def test_multiindex_without_close_raises(self):
        cols = pd.MultiIndex.from_product([["A"], ["open", "high"]])
        with pytest.raises(KeyError):
            get_closing_prices(pd.DataFrame([[1.0, 2.0]], columns=cols))


class TestBuyAndHold:
    def test_keeps_the_first_period_return(self):
        # Once rebased on the first return bar and reported 1.21
        data = pd.DataFrame({"close": [100, 110, 121, 133.1]}, index=DATES[:4])
        values, dates = simulate_buy_and_hold(data, [1.0])
        assert values.iloc[-1] / values.iloc[0] == pytest.approx(1.331)
        assert dates[0] == DATES[0] and values.iloc[0] == 1.0

    def test_weights_drift_with_prices(self, two_assets):
        values, _ = simulate_buy_and_hold(two_assets, [0.5, 0.5], br0=100)
        # Half in A (doubles) and half in B (flat), never rebalanced
        assert values.iloc[-1] == pytest.approx(150)

    def test_series_weights_align_by_symbol_and_normalize(self, two_assets):
        by_label, _ = simulate_buy_and_hold(two_assets, pd.Series({"B": 3.0, "A": 1.0}))
        by_position, _ = simulate_buy_and_hold(two_assets, [0.25, 0.75])
        pd.testing.assert_series_equal(by_label, by_position)

    def test_dict_weights_match_series_weights(self, two_assets):
        by_dict, _ = simulate_buy_and_hold(two_assets, {"B": 3.0, "A": 1.0})
        by_series, _ = simulate_buy_and_hold(two_assets, pd.Series({"B": 3.0, "A": 1.0}))
        pd.testing.assert_series_equal(by_dict, by_series)

    def test_symbol_missing_from_series_weights_is_unheld(self, two_assets):
        values, _ = simulate_buy_and_hold(two_assets, pd.Series({"A": 1.0}))
        assert values.iloc[-1] == pytest.approx(2.0)

    def test_complete_history_does_not_warn(self, two_assets, recwarn):
        simulate_buy_and_hold(two_assets, [0.5, 0.5])
        assert not recwarn

    def test_late_listed_symbol_is_dropped_with_a_warning(self, two_assets):
        # C lists on the fourth bar, so its first three returns are missing
        data = two_assets.assign(C=[np.nan, np.nan, np.nan, 10, 11, 12.0])
        with pytest.warns(UserWarning, match=r"C \(3 of 5 returns missing, first price 2025-01-07\)"):
            values, _ = simulate_buy_and_hold(data, {"A": 0.25, "B": 0.25, "C": 0.5})
        # Dropping C renormalizes A and B to half each
        expected, _ = simulate_buy_and_hold(two_assets, [0.5, 0.5])
        pd.testing.assert_series_equal(values, expected)

    def test_gap_inside_history_is_filled_not_dropped(self, two_assets, recwarn):
        # A missing close mid-history is carried from the last price, so the
        # return across the gap is measured from 120 and the symbol is kept
        gapped = two_assets.copy()
        gapped.loc[DATES[3], "A"] = np.nan
        values, _ = simulate_buy_and_hold(gapped, [1.0, 0.0])
        filled, _ = simulate_buy_and_hold(two_assets.assign(A=[100, 110, 120, 120, 180, 200.0]), [1.0, 0.0])
        pd.testing.assert_series_equal(values, filled)
        assert not recwarn

    def test_zero_weights_raise(self, two_assets):
        with pytest.raises(ValueError, match="sum to zero"):
            simulate_buy_and_hold(two_assets, [0.0, 0.0])

    def test_wrong_number_of_positional_weights_raises(self, two_assets):
        with pytest.raises(ValueError, match="3 positional weights for 2 assets"):
            simulate_buy_and_hold(two_assets, [0.2, 0.3, 0.5])


class TestNoLookAhead:
    def test_values_only_use_prices_up_to_their_date(self):
        # Scrambling every close after a date leaves the values up to it alone
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2025-01-02", periods=60)
        data = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 3)), axis=0)),
                            index=dates, columns=["A", "B", "C"])
        values, _ = simulate_buy_and_hold(data, [0.5, 0.3, 0.2])
        for cut in (0, 10, 30, 58):
            scrambled = data.copy()
            scrambled.iloc[cut + 1:] *= rng.uniform(0.3, 3.0, scrambled.iloc[cut + 1:].shape)
            again, _ = simulate_buy_and_hold(scrambled, [0.5, 0.3, 0.2])
            pd.testing.assert_series_equal(values.iloc[:cut + 1], again.iloc[:cut + 1])

    def test_a_later_gap_does_not_change_earlier_values(self, two_assets, recwarn):
        values, _ = simulate_buy_and_hold(two_assets, [0.5, 0.5])
        gapped = two_assets.copy()
        gapped.loc[DATES[4]:, "B"] = np.nan
        again, _ = simulate_buy_and_hold(gapped, [0.5, 0.5])
        pd.testing.assert_series_equal(values.iloc[:4], again.iloc[:4])
        assert not recwarn

    def test_symbol_with_no_prices_is_reported(self, two_assets):
        with pytest.warns(UserWarning, match=r"C \(5 of 5 returns missing, no prices\)"):
            simulate_buy_and_hold(two_assets.assign(C=np.nan), {"A": 1.0, "C": 1.0})


NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


def hourly_index(*days):
    """Hourly bar starts as yfinance and IB label them: 9:30, then 10:00 to 15:00."""
    return pd.DatetimeIndex([ny(f"{d} {c}") for d in days
                             for c in ["09:30"] + [f"{h}:00" for h in range(10, 16)]])


def ohlc(index, opens, closes):
    """A yfinance-style (Price, Ticker) frame from {ticker: prices} dicts."""
    return pd.concat({"Open": pd.DataFrame(opens, index=index),
                      "Close": pd.DataFrame(closes, index=index)},
                     axis=1, names=["Price", "Ticker"])


@pytest.fixture
def hourly():
    """Two sessions of hourly bars; A opens each bar 1 below its close."""
    index = hourly_index("2025-06-04", "2025-06-05")
    closes = {"A": 100 + np.arange(14.0), "B": 50 + 0.5 * np.arange(14.0)}
    opens = {"A": closes["A"] - 1, "B": closes["B"] - 0.25}
    return ohlc(index, opens, closes)


class TestCloseFillLabels:
    def test_intraday_values_are_labelled_with_their_bar_close(self, hourly):
        values, dates = simulate_buy_and_hold(hourly, [1, 1])
        assert dates is values.index
        assert len(values) == len(hourly)
        # Bought at the 9:30 bar's close, which is 10:00
        assert values.index[0] == ny("2025-06-04 10:00")
        assert values.index[6] == ny("2025-06-04 16:00")
        assert values.index[-1] == ny("2025-06-05 16:00")

    def test_relabelling_leaves_the_numbers_alone(self, hourly):
        values, _ = simulate_buy_and_hold(hourly, [1, 1])
        closes = hourly["Close"]
        expected = 0.5 * closes["A"] / 100 + 0.5 * closes["B"] / 50
        np.testing.assert_allclose(values.to_numpy(), expected.to_numpy())

    def test_no_value_carries_the_start_of_its_bar(self, hourly):
        values, _ = simulate_buy_and_hold(hourly, [1, 1])
        assert (values.index > hourly.index).all()

    def test_daily_dates_are_kept(self, two_assets):
        values, _ = simulate_buy_and_hold(two_assets, [0.5, 0.5])
        assert values.index.equals(DATES)

    def test_calendar_none_keeps_the_input_labels(self, hourly):
        values, _ = simulate_buy_and_hold(hourly, [1, 1], calendar=None)
        assert values.index.equals(hourly.index)

    def test_naive_intraday_labels_are_read_in_tz(self, hourly):
        naive = hourly.tz_localize(None)
        values, _ = simulate_buy_and_hold(naive, [1, 1], tz=NY)
        assert values.index.tz is None
        assert values.index[0] == pd.Timestamp("2025-06-04 10:00")


class TestOpenFill:
    def test_first_value_is_br0_at_the_first_open(self, hourly):
        values, _ = simulate_buy_and_hold(hourly, [1, 1], br0=10, fill="open")
        assert values.index[0] == ny("2025-06-04 09:30")
        assert values.iloc[0] == 10
        assert len(values) == len(hourly) + 1

    def test_later_values_are_closes_over_the_first_open(self, hourly):
        values, _ = simulate_buy_and_hold(hourly, [1, 1], fill="open")
        closes = hourly["Close"]
        expected = 0.5 * closes["A"] / 99 + 0.5 * closes["B"] / 49.75
        np.testing.assert_allclose(values.iloc[1:].to_numpy(), expected.to_numpy())
        assert values.index[1:].equals(
            pd.DatetimeIndex([ny("2025-06-04 10:00")]).append(hourly.index[2:7])
            .append(pd.DatetimeIndex([ny("2025-06-04 16:00")]))
            .append(hourly.index[8:]).append(pd.DatetimeIndex([ny("2025-06-05 16:00")])))

    def test_the_first_bar_return_is_counted(self, hourly):
        # Close fill starts at the first close; open fill also earns bar 0
        by_open, _ = simulate_buy_and_hold(hourly, [1, 0], fill="open")
        assert by_open.iloc[1] == pytest.approx(100 / 99)

    def test_daily_bars_fill_at_the_session_open(self):
        dates = pd.DatetimeIndex(["2025-11-26", "2025-11-28"])
        data = ohlc(dates, {"A": [10.0, 11.0]}, {"A": [10.5, 12.0]})
        values, _ = simulate_buy_and_hold(data, [1.0], fill="open")
        assert list(values.index) == [ny("2025-11-26 09:30"), ny("2025-11-26 16:00"),
                                      ny("2025-11-28 13:00")]
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.05, 1.2])

    def test_flat_single_symbol_frame(self):
        dates = pd.DatetimeIndex(["2025-06-04", "2025-06-05"])
        data = pd.DataFrame({"open": [10.0, 11.0], "close": [11.0, 12.0]}, index=dates)
        values, _ = simulate_buy_and_hold(data, [1.0], fill="open")
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.1, 1.2])

    def test_opens_are_adjusted_like_the_close(self):
        # A 2:1 adjustment on day 1: the adjusted close is half the close, so
        # the open must be halved too or the first return reads as -50%
        dates = pd.DatetimeIndex(["2025-06-04", "2025-06-05"])
        data = pd.concat({"Open": pd.DataFrame({"A": [20.0, 21.0]}, index=dates),
                          "Close": pd.DataFrame({"A": [22.0, 24.0]}, index=dates),
                          "Adj Close": pd.DataFrame({"A": [11.0, 12.0]}, index=dates)},
                         axis=1, names=["Price", "Ticker"])
        values, _ = simulate_buy_and_hold(data, [1.0], fill="open")
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.1, 1.2])

    def test_a_gap_holds_the_last_quote(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[3:5, gapped.columns.get_loc(("Close", "A"))] = np.nan
        values, _ = simulate_buy_and_hold(gapped, [1, 0], fill="open")
        assert values.iloc[4] == values.iloc[3] == values.iloc[5]

    def test_a_missing_first_close_holds_the_fill(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[0, gapped.columns.get_loc(("Close", "A"))] = np.nan
        values, _ = simulate_buy_and_hold(gapped, [1, 0], fill="open")
        assert values.iloc[1] == 1.0

    def test_symbol_without_a_first_open_is_dropped_with_a_warning(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[0, gapped.columns.get_loc(("Open", "B"))] = np.nan
        with pytest.warns(UserWarning, match="B"):
            values, _ = simulate_buy_and_hold(gapped, {"A": 1.0, "B": 1.0}, fill="open")
        alone, _ = simulate_buy_and_hold(hourly, {"A": 1.0}, fill="open")
        pd.testing.assert_series_equal(values, alone)

    def test_later_gap_does_not_drop_a_symbol(self, hourly, recwarn):
        gapped = hourly.copy()
        gapped.iloc[-3:, gapped.columns.get_loc(("Close", "B"))] = np.nan
        values, _ = simulate_buy_and_hold(gapped, [1, 1], fill="open")
        clean, _ = simulate_buy_and_hold(hourly, [1, 1], fill="open")
        pd.testing.assert_series_equal(values.iloc[:-3], clean.iloc[:-3])
        assert not recwarn

    def test_values_only_use_prices_known_by_their_label(self, hourly):
        # An open is known at its bar's start, a close at its bar's close
        values, _ = simulate_buy_and_hold(hourly, [0.6, 0.4], fill="open")
        close_known = values.index[1:]
        rng = np.random.default_rng(0)
        for cut in values.index[[0, 3, 7, 12]]:
            scrambled = hourly.copy()
            for field, known in (("Open", hourly.index), ("Close", close_known)):
                rows = np.asarray(known > cut)
                cols = [c for c in scrambled.columns if c[0] == field]
                scrambled.loc[rows, cols] *= rng.uniform(0.3, 3, (rows.sum(), len(cols)))
            again, _ = simulate_buy_and_hold(scrambled, [0.6, 0.4], fill="open")
            pd.testing.assert_series_equal(values[values.index <= cut],
                                           again[again.index <= cut])

    def test_weekly_bars_raise(self):
        weeks = pd.DatetimeIndex(["2025-06-02", "2025-06-09", "2025-06-16"])
        data = ohlc(weeks, {"A": [1.0, 1, 1]}, {"A": [1.0, 1, 1]})
        with pytest.raises(ValueError, match="weekly and longer"):
            simulate_buy_and_hold(data, [1.0], fill="open")

    def test_prices_without_opens_raise(self, two_assets):
        with pytest.raises(ValueError, match="needs opening prices"):
            simulate_buy_and_hold(two_assets, [0.5, 0.5], fill="open")

    def test_needs_a_calendar(self, hourly):
        with pytest.raises(ValueError, match="needs a calendar"):
            simulate_buy_and_hold(hourly, [1, 1], fill="open", calendar=None)

    def test_unknown_fill_raises(self, hourly):
        with pytest.raises(ValueError, match="fill must be"):
            simulate_buy_and_hold(hourly, [1, 1], fill="vwap")

    @pytest.mark.parametrize("w, message", [([1, 1, 1], "positional weights"),
                                            ([0, 0], "sum to zero")])
    def test_bad_weights_raise(self, hourly, w, message):
        with pytest.raises(ValueError, match=message):
            simulate_buy_and_hold(hourly, w, fill="open")
