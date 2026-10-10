import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from portfolio_forecast import BacktestResult
from portfolio_forecast.performance import simulate_buy_and_hold
from portfolio_forecast.performance.simulate import get_closing_prices
from portfolio_forecast.utils.calendar import close_times

NY = "America/New_York"


def ny(text):
    return pd.Timestamp(text, tz=NY)


# The first six XNYS sessions of 2025 (the market closed on Thursday 9 January)
DATES = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-10"])


SESSIONS_2025 = xcals.get_calendar("XNYS", start="2025-01-01", end="2025-12-31").sessions


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
        values, dates, _ = simulate_buy_and_hold(data, [1.0], fill="close")
        assert values.iloc[-1] / values.iloc[0] == pytest.approx(1.331)
        assert dates[0] == ny("2025-01-02 16:00") and values.iloc[0] == 1.0

    def test_weights_drift_with_prices(self, two_assets):
        values = simulate_buy_and_hold(two_assets, [0.5, 0.5], br0=100, fill="close").values
        # Half in A (doubles) and half in B (flat), never rebalanced
        assert values.iloc[-1] == pytest.approx(150)

    def test_series_weights_align_by_symbol_and_normalize(self, two_assets):
        by_label = simulate_buy_and_hold(two_assets, pd.Series({"B": 3.0, "A": 1.0}), fill="close").values
        by_position = simulate_buy_and_hold(two_assets, [0.25, 0.75], fill="close").values
        pd.testing.assert_series_equal(by_label, by_position)

    def test_dict_weights_match_series_weights(self, two_assets):
        by_dict = simulate_buy_and_hold(two_assets, {"B": 3.0, "A": 1.0}, fill="close").values
        by_series = simulate_buy_and_hold(two_assets, pd.Series({"B": 3.0, "A": 1.0}), fill="close").values
        pd.testing.assert_series_equal(by_dict, by_series)

    def test_symbol_missing_from_series_weights_is_unheld(self, two_assets):
        values = simulate_buy_and_hold(two_assets, pd.Series({"A": 1.0}), fill="close").values
        assert values.iloc[-1] == pytest.approx(2.0)

    def test_complete_history_does_not_warn(self, two_assets, recwarn):
        simulate_buy_and_hold(two_assets, [0.5, 0.5], fill="close")
        assert not recwarn

    def test_late_listed_symbol_is_dropped_with_a_warning(self, two_assets):
        # C lists on the fourth bar, so its first three returns are missing
        data = two_assets.assign(C=[np.nan, np.nan, np.nan, 10, 11, 12.0])
        with pytest.warns(UserWarning, match=r"C \(first price 2025-01-07"):
            values = simulate_buy_and_hold(data, {"A": 0.25, "B": 0.25, "C": 0.5}, fill="close").values
        # Dropping C renormalizes A and B to half each
        expected = simulate_buy_and_hold(two_assets, [0.5, 0.5], fill="close").values
        pd.testing.assert_series_equal(values, expected)

    def test_gap_inside_history_is_filled_not_dropped(self, two_assets):
        # A missing close mid-history is carried from the last price, so the
        # return across the gap is measured from 120 and the symbol is kept;
        # the two returns touching the stale price are irregular (B4.1)
        gapped = two_assets.copy()
        gapped.loc[DATES[3], "A"] = np.nan
        with pytest.warns(UserWarning, match=r"2 return\(s\) are irregular.*stale"):
            values = simulate_buy_and_hold(gapped, [1.0, 0.0], fill="close").values
        filled = simulate_buy_and_hold(two_assets.assign(A=[100, 110, 120, 120, 180, 200.0]), [1.0, 0.0],
                                       fill="close").values
        pd.testing.assert_series_equal(values, filled)

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
        dates = SESSIONS_2025[:60]
        data = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 3)), axis=0)),
                            index=dates, columns=["A", "B", "C"])
        values = simulate_buy_and_hold(data, [0.5, 0.3, 0.2], fill="close").values
        for cut in (0, 10, 30, 58):
            scrambled = data.copy()
            scrambled.iloc[cut + 1:] *= rng.uniform(0.3, 3.0, scrambled.iloc[cut + 1:].shape)
            again = simulate_buy_and_hold(scrambled, [0.5, 0.3, 0.2], fill="close").values
            pd.testing.assert_series_equal(values.iloc[:cut + 1], again.iloc[:cut + 1])

    def test_a_later_gap_does_not_change_earlier_values(self, two_assets):
        values = simulate_buy_and_hold(two_assets, [0.5, 0.5], fill="close").values
        gapped = two_assets.copy()
        gapped.loc[DATES[4]:, "B"] = np.nan
        # B's prices stop early, so it is held as cash from there (B4.3)
        with pytest.warns(UserWarning, match=r"B \(last price 2025-01-07"):
            again = simulate_buy_and_hold(gapped, [0.5, 0.5], fill="close").values
        pd.testing.assert_series_equal(values.iloc[:4], again.iloc[:4])

    def test_symbol_with_no_prices_is_reported(self, two_assets):
        with pytest.warns(UserWarning, match=r"C \(no prices\)"):
            simulate_buy_and_hold(two_assets.assign(C=np.nan), {"A": 1.0, "C": 1.0}, fill="close")


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
        values, dates, _ = simulate_buy_and_hold(hourly, [1, 1], fill="close")
        assert dates is values.index
        # Each bar's close, plus session 2's open mark (B1)
        assert len(values) == len(hourly) + 1
        # Bought at the 9:30 bar's close, which is 10:00
        assert values.index[0] == ny("2025-06-04 10:00")
        assert values.index[6] == ny("2025-06-04 16:00")
        assert values.index[7] == ny("2025-06-05 09:30")
        assert values.index[-1] == ny("2025-06-05 16:00")

    def test_relabelling_leaves_the_numbers_alone(self, hourly):
        values = simulate_buy_and_hold(hourly, [1, 1], fill="close").values
        closes = hourly["Close"]
        expected = 0.5 * closes["A"] / 100 + 0.5 * closes["B"] / 50
        np.testing.assert_allclose(values.drop(ny("2025-06-05 09:30")).to_numpy(), expected.to_numpy())

    def test_no_value_carries_the_start_of_its_bar(self, hourly):
        # Close marks sit after their bar's start; only open marks sit at it
        values = simulate_buy_and_hold(hourly, [1, 1], fill="close").values
        closes = values.drop(ny("2025-06-05 09:30"))
        assert (closes.index > hourly.index).all()

    def test_daily_values_are_labelled_at_the_session_close(self, two_assets):
        # B3: daily output was dates; each value is now its session's close
        values = simulate_buy_and_hold(two_assets, [0.5, 0.5], fill="close").values
        assert list(values.index) == [ny(f"{d:%Y-%m-%d} 16:00") for d in DATES]

    def test_calendar_none_is_continuous_mode(self, hourly):
        # U3: session-hours data is not 24/7, so continuous mode warns, and
        # the returns across nights are irregular (B4.1)
        with pytest.warns(UserWarning) as caught:
            values = simulate_buy_and_hold(hourly, [1, 1], calendar=None, fill="close").values
        messages = [str(w.message) for w in caught]
        assert len(messages) == 2
        assert 'calendar="24/5"' in messages[0] and "held symbol is stale (no own price)" in messages[1]
        assert values.index.equals(hourly.index + pd.Timedelta(hours=1))

    def test_naive_intraday_labels_are_read_in_tz(self, hourly):
        # B3: labels are timezone-aware on the exchange clock, even for naive input
        naive = hourly.tz_localize(None)
        values = simulate_buy_and_hold(naive, [1, 1], tz=NY, fill="close").values
        assert str(values.index.tz) == NY
        assert values.index[0] == ny("2025-06-04 10:00")


class TestOpenFill:
    def test_first_value_is_br0_at_the_first_open(self, hourly):
        values = simulate_buy_and_hold(hourly, [1, 1], br0=10, fill="open").values
        assert values.index[0] == ny("2025-06-04 09:30")
        assert values.iloc[0] == 10
        # Every bar's close and both sessions' open marks
        assert len(values) == len(hourly) + 2

    def test_open_is_the_default(self, hourly):
        pd.testing.assert_series_equal(simulate_buy_and_hold(hourly, [1, 1]).values,
                                       simulate_buy_and_hold(hourly, [1, 1], fill="open").values)

    def test_later_values_are_prices_over_the_first_open(self, hourly):
        values = simulate_buy_and_hold(hourly, [1, 1], fill="open").values
        closes, opens = hourly["Close"], hourly["Open"]
        at_closes = 0.5 * closes["A"] / 99 + 0.5 * closes["B"] / 49.75
        open_mark = ny("2025-06-05 09:30")
        np.testing.assert_allclose(values.iloc[1:].drop(open_mark).to_numpy(), at_closes.to_numpy())
        assert values[open_mark] == pytest.approx(0.5 * opens["A"].iloc[7] / 99 + 0.5 * opens["B"].iloc[7] / 49.75)
        assert values.index[1:].equals(
            pd.DatetimeIndex([ny("2025-06-04 10:00")]).append(hourly.index[2:7])
            .append(pd.DatetimeIndex([ny("2025-06-04 16:00"), open_mark, ny("2025-06-05 10:00")]))
            .append(hourly.index[9:]).append(pd.DatetimeIndex([ny("2025-06-05 16:00")])))

    def test_the_first_bar_return_is_counted(self, hourly):
        # Close fill starts at the first close; open fill also earns bar 0
        by_open = simulate_buy_and_hold(hourly, [1, 0], fill="open").values
        assert by_open.iloc[1] == pytest.approx(100 / 99)

    def test_daily_bars_fill_at_the_session_open(self):
        dates = pd.DatetimeIndex(["2025-11-26", "2025-11-28"])
        data = ohlc(dates, {"A": [10.0, 11.0]}, {"A": [10.5, 12.0]})
        values = simulate_buy_and_hold(data, [1.0], fill="open").values
        assert list(values.index) == [ny("2025-11-26 09:30"), ny("2025-11-26 16:00"),
                                      ny("2025-11-28 13:00")]
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.05, 1.2])

    def test_flat_single_symbol_frame(self):
        dates = pd.DatetimeIndex(["2025-06-04", "2025-06-05"])
        data = pd.DataFrame({"open": [10.0, 11.0], "close": [11.0, 12.0]}, index=dates)
        values = simulate_buy_and_hold(data, [1.0], fill="open").values
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.1, 1.2])

    def test_opens_are_adjusted_like_the_close(self):
        # A 2:1 adjustment on day 1: the adjusted close is half the close, so
        # the open must be halved too or the first return reads as -50%
        dates = pd.DatetimeIndex(["2025-06-04", "2025-06-05"])
        data = pd.concat({"Open": pd.DataFrame({"A": [20.0, 21.0]}, index=dates),
                          "Close": pd.DataFrame({"A": [22.0, 24.0]}, index=dates),
                          "Adj Close": pd.DataFrame({"A": [11.0, 12.0]}, index=dates)},
                         axis=1, names=["Price", "Ticker"])
        values = simulate_buy_and_hold(data, [1.0], fill="open").values
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.1, 1.2])

    def test_a_gap_holds_the_last_quote(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[3:5, gapped.columns.get_loc(("Close", "A"))] = np.nan
        with pytest.warns(UserWarning, match=r"held symbol is stale \(no own price\)"):
            values = simulate_buy_and_hold(gapped, [1, 0], fill="open").values
        assert values.iloc[4] == values.iloc[3] == values.iloc[5]

    def test_a_missing_first_close_holds_the_fill(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[0, gapped.columns.get_loc(("Close", "A"))] = np.nan
        with pytest.warns(UserWarning, match=r"held symbol is stale \(no own price\)"):
            values = simulate_buy_and_hold(gapped, [1, 0], fill="open").values
        assert values.iloc[1] == 1.0

    def test_symbol_without_a_first_open_is_dropped_with_a_warning(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[0, gapped.columns.get_loc(("Open", "B"))] = np.nan
        with pytest.warns(UserWarning, match=r"B \(first price 2025-06-04 10:00"):
            values = simulate_buy_and_hold(gapped, {"A": 1.0, "B": 1.0}, fill="open").values
        alone = simulate_buy_and_hold(hourly, {"A": 1.0}, fill="open").values
        pd.testing.assert_series_equal(values, alone)

    def test_later_gap_does_not_drop_a_symbol(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[-3:, gapped.columns.get_loc(("Close", "B"))] = np.nan
        # B's closes stop three bars early, so it is held as cash (B4.3)
        with pytest.warns(UserWarning, match=r"B \(last price 2025-06-05 13:00"):
            values = simulate_buy_and_hold(gapped, [1, 1], fill="open").values
        clean = simulate_buy_and_hold(hourly, [1, 1], fill="open").values
        pd.testing.assert_series_equal(values.iloc[:-3], clean.iloc[:-3])

    def test_values_only_use_prices_known_by_their_label(self, hourly):
        # An open is known at its bar's start, a close at its bar's close
        values = simulate_buy_and_hold(hourly, [0.6, 0.4], fill="open").values
        close_known = pd.DatetimeIndex(close_times(hourly.index))
        rng = np.random.default_rng(0)
        for cut in values.index[[0, 3, 7, 8, 12]]:
            scrambled = hourly.copy()
            for field, known in (("Open", hourly.index), ("Close", close_known)):
                rows = np.asarray(known > cut)
                cols = [c for c in scrambled.columns if c[0] == field]
                scrambled.loc[rows, cols] *= rng.uniform(0.3, 3, (rows.sum(), len(cols)))
            again = simulate_buy_and_hold(scrambled, [0.6, 0.4], fill="open").values
            pd.testing.assert_series_equal(values[values.index <= cut],
                                           again[again.index <= cut])

    def test_weekly_bars_fill_at_the_first_week_open(self):
        # B3: weekly bars support fill="open"; Juneteenth week still ends Friday
        weeks = pd.DatetimeIndex(["2025-06-02", "2025-06-09", "2025-06-16"])
        data = ohlc(weeks, {"A": [10.0, 11, 12]}, {"A": [11.0, 12, 13]})
        values = simulate_buy_and_hold(data, [1.0], fill="open").values
        assert list(values.index) == [ny("2025-06-02 09:30"), ny("2025-06-06 16:00"),
                                      ny("2025-06-13 16:00"), ny("2025-06-20 16:00")]
        np.testing.assert_allclose(values.to_numpy(), [1.0, 1.1, 1.2, 1.3])

    def test_prices_without_opens_raise_naming_fill_close(self, two_assets):
        # B2: the default fill="open" needs opens
        with pytest.raises(ValueError, match="needs opening prices.*fill='close'"):
            simulate_buy_and_hold(two_assets, [0.5, 0.5])

    def test_unknown_fill_raises(self, hourly):
        with pytest.raises(ValueError, match="fill must be"):
            simulate_buy_and_hold(hourly, [1, 1], fill="vwap")

    @pytest.mark.parametrize("w, message", [([1, 1, 1], "positional weights"),
                                            ([0, 0], "sum to zero")])
    def test_bad_weights_raise(self, hourly, w, message):
        with pytest.raises(ValueError, match=message):
            simulate_buy_and_hold(hourly, w, fill="open")


def random_hourly(days, symbols=("A", "B"), seed=0):
    """Hourly OHLC on `days` with random walks, opens away from the last close."""
    index = hourly_index(*days)
    rng = np.random.default_rng(seed)
    closes = {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))) for s in symbols}
    opens = {s: closes[s] * np.exp(rng.normal(0, 0.005, len(index))) for s in symbols}
    return ohlc(index, opens, closes)


class TestBacktestResult:
    def test_three_fields_with_classes_ending_at_each_mark(self, hourly):
        result = simulate_buy_and_hold(hourly, [1, 1])
        assert isinstance(result, BacktestResult)
        assert result.dates is result.values.index
        assert result.classes.index.equals(result.dates[1:])
        assert list(result.classes) == (["regular"] * 7 + ["gap"] + ["regular"] * 7)

    def test_old_two_name_unpacking_fails_loudly(self, hourly):
        with pytest.raises(ValueError, match="too many values"):
            values, dates = simulate_buy_and_hold(hourly, [1, 1])


class TestValueSeries:
    """B1: point-in-time marks, gap and bars separated."""

    def test_session_identity(self):
        # (1 + g_d)(1 + r_d,1) Π_k≥2 (1 + r_d,k) = V(C_d) / V(C_d-1) on every session after the first
        data = random_hourly(["2025-06-04", "2025-06-05", "2025-06-06", "2025-06-09"])
        values, _, classes = simulate_buy_and_hold(data, [0.7, 0.3])
        returns = values.pct_change().iloc[1:]
        session = values.index.normalize()
        closes = values.groupby(session).last()
        for d in range(1, len(closes)):
            day = returns[session[1:] == closes.index[d]]
            assert classes[day.index[0]] == "gap"
            assert np.prod(1 + day) == pytest.approx(closes.iloc[d] / closes.iloc[d - 1], rel=1e-12)

    def test_gap_is_last_close_to_next_open(self):
        data = random_hourly(["2025-06-04", "2025-06-05"])
        values = simulate_buy_and_hold(data, [1, 0]).values
        opens, closes = data["Open"]["A"], data["Close"]["A"]
        gap = values[ny("2025-06-05 09:30")] / values[ny("2025-06-04 16:00")] - 1
        assert gap == pytest.approx(opens.iloc[7] / closes.iloc[6] - 1)
        first_bar = values[ny("2025-06-05 10:00")] / values[ny("2025-06-05 09:30")] - 1
        assert first_bar == pytest.approx(closes.iloc[7] / opens.iloc[7] - 1)

    def test_intraday_without_opens_raises(self, hourly):
        # B1.5
        with pytest.raises(ValueError, match="need opens"):
            simulate_buy_and_hold(hourly["Close"], [1, 1], fill="close")

    def test_missing_first_bar_is_warned_and_irregular(self, hourly):
        # B1.6: session 2 starts at 10:00; its open mark is that bar's open
        late = hourly.drop(ny("2025-06-05 09:30"))
        with pytest.warns(UserWarning, match=r"1 session\(s\) have no bar at the open.*irregular"):
            values, _, classes = simulate_buy_and_hold(late, [1, 0])
        assert ny("2025-06-05 10:00") in values.index
        assert classes[ny("2025-06-05 10:00")] == "irregular"
        assert values[ny("2025-06-05 10:00")] == pytest.approx(hourly["Open"]["A"].iloc[8] / 99)


class TestMissingPrices:
    def test_stale_symbol_count_is_warned(self, hourly):
        # B4.1: B has no 11:00 bar; the returns ending at 12:00 and 13:00
        # have B stale at one end
        gapped = hourly.copy()
        gapped.loc[ny("2025-06-04 11:00"), [("Open", "B"), ("Close", "B")]] = np.nan
        with pytest.warns(UserWarning, match=r"2 return\(s\) are irregular"):
            _, _, classes = simulate_buy_and_hold(gapped, [1, 1])
        assert list(classes[classes == "irregular"].index) == [ny("2025-06-04 12:00"), ny("2025-06-04 13:00")]

    def test_missing_scheduled_bar_counts_with_b4_1(self, hourly):
        # B4.1: the 11:00 row is gone for every symbol
        with pytest.warns(UserWarning, match=r"1 return\(s\) are irregular.*scheduled bar is missing"):
            simulate_buy_and_hold(hourly.drop(ny("2025-06-04 11:00")), [1, 1])

    def test_late_listing_dropped_and_renormalized(self):
        # B4.2: C has no open at the fill; it is dropped and A, B split 1:1
        data = random_hourly(["2025-06-04", "2025-06-05"], ("A", "B", "C"))
        data.loc[:ny("2025-06-04 12:00"), [("Open", "C"), ("Close", "C")]] = np.nan
        with pytest.warns(UserWarning, match=r"Dropped 1 symbol\(s\).*C \(first price 2025-06-04 13:00"):
            values = simulate_buy_and_hold(data, {"A": 1, "B": 1, "C": 2}).values
        expected = simulate_buy_and_hold(data.drop(columns="C", level=1), {"A": 1, "B": 1}).values
        pd.testing.assert_series_equal(values, expected)

    def test_no_held_symbol_priced_at_the_fill_raises(self, hourly):
        gapped = hourly.copy()
        gapped.iloc[0, gapped.columns.get_loc(("Open", "A"))] = np.nan
        with pytest.raises(ValueError, match="No held symbol has a price at the fill"):
            simulate_buy_and_hold(gapped, {"A": 1.0})

    def test_prices_stop_early_hold_cash(self):
        # B4.3: A's prices end on the 4th; from there it is cash at its last
        # close, always fresh, so later returns keep their class
        data = random_hourly(["2025-06-04", "2025-06-05"])
        data.loc[ny("2025-06-05 09:30"):, [("Open", "A"), ("Close", "A")]] = np.nan
        with pytest.warns(UserWarning, match=r"held as cash.*A \(last price 2025-06-04 16:00.*survivorship"):
            values, _, classes = simulate_buy_and_hold(data, [0.5, 0.5])
        a_cash = 0.5 / data["Open"]["A"].iloc[0] * data["Close"]["A"].iloc[6]
        b = 0.5 / data["Open"]["B"].iloc[0] * data["Close"]["B"]
        np.testing.assert_allclose(values[ny("2025-06-05 10:00"):].to_numpy(), a_cash + b.iloc[7:].to_numpy())
        assert (classes != "irregular").all()

    def test_pre_market_bar_dropped_before_the_fill(self, hourly):
        # B4.4: an 08:00 bar is not the first bar; the fill is at 09:30
        early = pd.concat([hourly.iloc[:1].set_axis([ny("2025-06-04 08:00")]) * 0.5, hourly])
        with pytest.warns(UserWarning, match=r"Dropped 1 intraday bar\(s\) outside XNYS sessions"):
            values = simulate_buy_and_hold(early, [1, 1]).values
        pd.testing.assert_series_equal(values, simulate_buy_and_hold(hourly, [1, 1]).values)


class TestAdjustedOpens:
    def test_each_open_uses_its_own_bars_factor(self):
        # B5: day 2 is the ex-date: AdjC/C is 0.98 on day 1 and 1 on day 2,
        # so the fill is O_1 * 0.98 = 98.0 and the dividend shows in the
        # close-to-close return (in the gap, on intraday data)
        dates = pd.DatetimeIndex(["2025-06-04", "2025-06-05"])
        data = pd.concat({"Open": pd.DataFrame({"A": [100.0, 99.0]}, index=dates),
                          "Close": pd.DataFrame({"A": [101.0, 100.0]}, index=dates),
                          "Adj Close": pd.DataFrame({"A": [98.98, 100.0]}, index=dates)},
                         axis=1, names=["Price", "Ticker"])
        values = simulate_buy_and_hold(data, [1.0]).values
        # Daily: the fill at day 1's open (98.0 adjusted), then each close
        np.testing.assert_allclose(values.to_numpy(), [1.0, 98.98 / 98.0, 100.0 / 98.0])


class TestCompletedBars:
    def test_in_progress_bar_dropped_without_a_warning(self, hourly, recwarn):
        # B6: at 14:30 on the 5th the 14:00 bar is still open
        result = simulate_buy_and_hold(hourly, [1, 1], now=ny("2025-06-05 14:30"))
        assert result.dates[-1] == ny("2025-06-05 14:00")
        assert not recwarn

    def test_bar_closing_at_now_is_kept(self, hourly):
        result = simulate_buy_and_hold(hourly, [1, 1], now=ny("2025-06-05 15:00"))
        assert result.dates[-1] == ny("2025-06-05 15:00")

    def test_nothing_closed_raises(self, hourly):
        with pytest.raises(ValueError, match="No bar has closed"):
            simulate_buy_and_hold(hourly, [1, 1], now=ny("2025-06-04 09:45"))


class TestContinuousMode:
    """U3: calendar=None on 24/7 data."""

    HOURS = pd.date_range("2025-06-06 22:00", periods=30, freq="h", tz="UTC")  # through a weekend

    def data(self):
        closes = {"BTC": 100 + np.arange(30.0)}
        opens = {"BTC": closes["BTC"] - 0.5}
        return ohlc(self.HOURS, opens, closes)

    def test_fill_close_starts_at_the_first_close(self):
        values, dates, classes = simulate_buy_and_hold(self.data(), [1.0], calendar=None, fill="close")
        assert dates[0] == self.HOURS[0] + pd.Timedelta(hours=1) and values.iloc[0] == 1.0
        assert dates.equals(self.HOURS + pd.Timedelta(hours=1))
        assert set(classes) == {"regular"}

    def test_fill_open_starts_at_the_first_bars_open(self):
        values, dates, classes = simulate_buy_and_hold(self.data(), [1.0], calendar=None)
        assert dates[0] == self.HOURS[0] and values.iloc[0] == 1.0
        assert values.iloc[1] == pytest.approx(100 / 99.5)
        assert classes.iloc[0] == "regular" and set(classes) == {"regular"}

    def test_missing_grid_point_is_irregular_and_warned(self):
        with pytest.warns(UserWarning, match=r"held symbol is stale \(no own price\)"):
            _, _, classes = simulate_buy_and_hold(self.data().drop(self.HOURS[10]), [1.0], calendar=None)
        assert (classes == "irregular").sum() == 1
