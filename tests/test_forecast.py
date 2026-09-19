import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfolio_forecast.forecast import (nonparametric_monte_carlo, parametric_monte_carlo,
                                         regime_switching_monte_carlo)

# Every simulator, with keyword arguments that keep the fits quick and quiet
SIMULATORS = [
    pytest.param(nonparametric_monte_carlo, {}, id="nonparametric"),
    pytest.param(parametric_monte_carlo, {"distribution": stats.norm}, id="parametric"),
    pytest.param(regime_switching_monte_carlo, {"n_regimes": 1, "n_starts": 2}, id="regime"),
]


@pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
class TestSimsContract:
    """Every simulator returns the documented sims array."""

    def test_shape_start_and_values(self, simulate, kwargs, prices):
        sims = simulate(prices, sim_length=20, n_sims=30, random_state=1, **kwargs)
        assert sims.shape == (30, 21)
        assert np.all(sims[:, 0] == 1.0)
        assert np.isfinite(sims).all()

    def test_accepts_a_series(self, simulate, kwargs, prices):
        sims = simulate(prices["X"], sim_length=5, n_sims=4, random_state=1, **kwargs)
        assert sims.shape == (4, 6)

    def test_int_seed_gives_distinct_reproducible_paths(self, simulate, kwargs, prices):
        # An int seed once restarted the draws on every path, so all paths matched
        a = simulate(prices, sim_length=10, n_sims=50, random_state=7, **kwargs)
        b = simulate(prices, sim_length=10, n_sims=50, random_state=7, **kwargs)
        assert np.array_equal(a, b)
        assert len(np.unique(a[:, -1])) == 50

    def test_generator_seed_is_reproducible(self, simulate, kwargs, prices):
        a = simulate(prices, sim_length=10, n_sims=5, random_state=np.random.default_rng(3), **kwargs)
        b = simulate(prices, sim_length=10, n_sims=5, random_state=np.random.default_rng(3), **kwargs)
        assert np.array_equal(a, b)


class TestNonparametric:
    def test_every_draw_is_a_historical_return(self, prices):
        sims = nonparametric_monte_carlo(prices, sim_length=25, n_sims=20, random_state=0)
        history = prices["X"].pct_change().dropna().to_numpy()
        drawn = (sims[:, 1:] / sims[:, :-1] - 1).ravel()
        assert np.isclose(drawn[:, None], history[None, :]).any(axis=1).all()

    def test_rejects_more_than_one_column(self, prices):
        two = prices.assign(Y=prices["X"] * 2)
        with pytest.raises(ValueError):
            nonparametric_monte_carlo(two, sim_length=5, n_sims=2)


class TestParametric:
    def test_selects_by_aic_and_reports_it(self, prices, capsys):
        parametric_monte_carlo(prices, sim_length=5, n_sims=2, random_state=0)
        out = capsys.readouterr().out
        for name in ("norm", "t", "johnsonsu"):
            assert f"{name} AIC" in out
        assert "Selected:" in out

    def test_given_distribution_skips_selection(self, prices, capsys):
        parametric_monte_carlo(prices, distribution=stats.laplace, sim_length=5, n_sims=2,
                               random_state=0)
        assert "AIC" not in capsys.readouterr().out

    def test_mean_return_matches_fitted_normal(self, prices):
        # With a normal fit, simulated one-period returns center on the historical mean
        sims = parametric_monte_carlo(prices, distribution=stats.norm, sim_length=200,
                                      n_sims=200, random_state=0)
        drawn = sims[:, 1:] / sims[:, :-1] - 1
        history = prices["X"].pct_change().dropna()
        assert drawn.mean() == pytest.approx(history.mean(), abs=3 * history.std() / np.sqrt(drawn.size))
        assert drawn.std() == pytest.approx(history.std(), rel=0.05)


class TestRegimeSwitching:
    @pytest.fixture
    def two_regime_prices(self):
        """Alternating calm (1% vol) and turbulent (4% vol) stretches of 100 bars."""
        rng = np.random.default_rng(0)
        vols = np.tile(np.repeat([0.01, 0.04], 100), 3)
        values = 100 * np.exp(np.cumsum(rng.normal(0, vols)))
        return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)))

    def test_bootstrap_draws_are_historical_log_returns(self, two_regime_prices):
        sims = regime_switching_monte_carlo(two_regime_prices, n_regimes=2, bootstrap=True,
                                            n_starts=3, sim_length=15, n_sims=20, random_state=0)
        history = np.log(two_regime_prices).diff().dropna().to_numpy()
        drawn = np.diff(np.log(sims), axis=1).ravel()
        assert np.isclose(drawn[:, None], history[None, :]).any(axis=1).all()

    def test_selects_regime_count_by_bic(self, prices, capsys):
        regime_switching_monte_carlo(prices, n_starts=2, sim_length=5, n_sims=2, random_state=0)
        out = capsys.readouterr().out
        assert "BIC" in out and "Selected:" in out

    def test_unfittable_regime_count_raises(self, prices):
        # 50 regimes cannot each hold 1% of 400 returns
        with pytest.raises(ValueError, match="Could not fit 50 regimes"):
            regime_switching_monte_carlo(prices, n_regimes=50, n_starts=1, sim_length=5, n_sims=2,
                                         random_state=0)
