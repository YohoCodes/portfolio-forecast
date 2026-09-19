import numpy as np
import pandas as pd
import pytest

from portfolio_forecast.performance import simulate_buy_and_hold, simulate_rebalancing_MVO
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

    def test_symbol_missing_from_series_weights_is_unheld(self, two_assets):
        values, _ = simulate_buy_and_hold(two_assets, pd.Series({"A": 1.0}))
        assert values.iloc[-1] == pytest.approx(2.0)

    def test_zero_weights_raise(self, two_assets):
        with pytest.raises(ValueError, match="sum to zero"):
            simulate_buy_and_hold(two_assets, [0.0, 0.0])

    def test_wrong_number_of_positional_weights_raises(self, two_assets):
        with pytest.raises(ValueError, match="3 positional weights for 2 assets"):
            simulate_buy_and_hold(two_assets, [0.2, 0.3, 0.5])


class TestRebalancing:
    def test_single_trade_tracks_the_book(self, two_assets):
        w = pd.DataFrame({DATES[0]: [1.0, 0.0]}, index=["A", "B"])
        values, _, decisions = simulate_rebalancing_MVO(two_assets, w)
        assert values.tolist() == pytest.approx((two_assets["A"] / 100).tolist())
        assert decisions.tolist() == ["traded"]

    def test_flat_before_the_first_trade(self, two_assets):
        w = pd.DataFrame({DATES[2]: [1.0, 0.0]}, index=["A", "B"])
        values, _, _ = simulate_rebalancing_MVO(two_assets, w, br0=10)
        assert values.iloc[:3].tolist() == [10.0, 10.0, 10.0]
        assert values.iloc[-1] == pytest.approx(10 * 200 / 120)

    def test_decisions(self, two_assets):
        # Day 0: all A. Day 2: all A again (no turnover, held). Day 3: no
        # suggestion (skipped). Day 4: all B (full turnover, traded).
        w = pd.DataFrame({DATES[0]: [1.0, 0.0], DATES[2]: [1.0, 0.0],
                          DATES[3]: [np.nan, np.nan], DATES[4]: [0.0, 1.0]}, index=["A", "B"])
        values, _, decisions = simulate_rebalancing_MVO(two_assets, w, beta=0.1)
        assert decisions.tolist() == ["traded", "held", "skipped", "traded"]
        # A's gain to day 4 (100 -> 180), then flat in B
        assert values.iloc[-1] == pytest.approx(1.8)

    def test_beta_one_never_trades_after_the_first(self, two_assets):
        w = pd.DataFrame({DATES[0]: [1.0, 0.0], DATES[3]: [0.0, 1.0]}, index=["A", "B"])
        _, _, decisions = simulate_rebalancing_MVO(two_assets, w, beta=1)
        assert decisions.tolist() == ["traded", "held"]

    def test_weights_must_be_a_frame(self, two_assets):
        with pytest.raises(TypeError):
            simulate_rebalancing_MVO(two_assets, [1.0, 0.0])

    @pytest.mark.parametrize("beta", [-0.1, 1.5])
    def test_beta_outside_unit_interval_raises(self, two_assets, beta):
        w = pd.DataFrame({DATES[0]: [1.0, 0.0]}, index=["A", "B"])
        with pytest.raises(ValueError, match="beta"):
            simulate_rebalancing_MVO(two_assets, w, beta=beta)

    def test_partly_nan_column_raises(self, two_assets):
        w = pd.DataFrame({DATES[0]: [1.0, np.nan]}, index=["A", "B"])
        with pytest.raises(ValueError, match="partly NaN"):
            simulate_rebalancing_MVO(two_assets, w)

    def test_timestamp_outside_price_index_raises(self, two_assets):
        w = pd.DataFrame({pd.Timestamp("2030-01-01"): [1.0, 0.0]}, index=["A", "B"])
        with pytest.raises(ValueError, match="absent from the price index"):
            simulate_rebalancing_MVO(two_assets, w)

    def test_timestamps_out_of_order_raise(self, two_assets):
        w = pd.DataFrame({DATES[3]: [1.0, 0.0], DATES[1]: [0.0, 1.0]}, index=["A", "B"])
        with pytest.raises(ValueError, match="strictly increasing"):
            simulate_rebalancing_MVO(two_assets, w)
