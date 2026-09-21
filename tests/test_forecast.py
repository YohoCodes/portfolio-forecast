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


@pytest.fixture
def two_regime_prices():
    """Alternating calm (1% vol) and turbulent (4% vol) stretches of 100 bars."""
    rng = np.random.default_rng(0)
    vols = np.tile(np.repeat([0.01, 0.04], 100), 3)
    values = 100 * np.exp(np.cumsum(rng.normal(0, vols)))
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)))


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


@pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
class TestDeprecatedPrices:
    """`prices` still works as an alias for `values` until 1.0.0, with a warning."""

    def test_values_keyword_does_not_warn(self, simulate, kwargs, prices, recwarn):
        simulate(values=prices, sim_length=5, n_sims=3, random_state=1, **kwargs)
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_prices_warns_and_matches_values(self, simulate, kwargs, prices):
        expected = simulate(prices, sim_length=5, n_sims=3, random_state=1, **kwargs)
        with pytest.warns(DeprecationWarning, match="removed in 1.0.0"):
            got = simulate(prices=prices, sim_length=5, n_sims=3, random_state=1, **kwargs)
        np.testing.assert_array_equal(got, expected)

    def test_warning_points_at_the_caller(self, simulate, kwargs, prices):
        with pytest.warns(DeprecationWarning) as record:
            simulate(prices=prices, sim_length=5, n_sims=3, random_state=1, **kwargs)
        assert record[0].filename == __file__

    def test_both_raise(self, simulate, kwargs, prices):
        with pytest.raises(TypeError, match="both"):
            simulate(prices, prices=prices, sim_length=5, n_sims=3, **kwargs)

    def test_neither_raises(self, simulate, kwargs):
        with pytest.raises(TypeError, match="missing"):
            simulate(sim_length=5, n_sims=3, **kwargs)


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


class TestParametricFit:
    def test_returns_only_sims_by_default(self, prices):
        out = parametric_monte_carlo(prices, distribution=stats.norm, sim_length=5, n_sims=3, random_state=0)
        assert isinstance(out, np.ndarray)

    def test_fit_does_not_change_the_paths(self, prices):
        plain = parametric_monte_carlo(prices, sim_length=10, n_sims=20, random_state=5)
        sims, _ = parametric_monte_carlo(prices, sim_length=10, n_sims=20, random_state=5, return_fit=True)
        np.testing.assert_allclose(sims, plain, rtol=1e-9)

    def test_selection_reports_the_lowest_aic(self, prices):
        _, fit = parametric_monte_carlo(prices, sim_length=5, n_sims=2, random_state=0, return_fit=True)
        aics = fit["Candidates"]
        assert set(aics) == {"norm", "t", "johnsonsu"}
        assert list(aics.values()) == sorted(aics.values())
        assert fit["Distribution"] == next(iter(aics))
        assert fit["AIC"] == pytest.approx(aics[fit["Distribution"]])

    def test_given_distribution_reports_its_fit(self, prices):
        _, fit = parametric_monte_carlo(prices, distribution=stats.t, sim_length=5, n_sims=2,
                                        random_state=0, return_fit=True)
        assert fit["Distribution"] == "t" and fit["Candidates"] is None
        assert list(fit["Parameters"]) == ["df", "loc", "scale"]
        data = prices["X"].pct_change().dropna().to_numpy()
        params = list(fit["Parameters"].values())
        assert fit["AIC"] == pytest.approx(2 * 3 - 2 * stats.t.logpdf(data, *params).sum())

    def test_parameters_are_named_for_each_distribution(self, prices):
        for dist, names in [(stats.norm, ["loc", "scale"]), (stats.johnsonsu, ["a", "b", "loc", "scale"])]:
            _, fit = parametric_monte_carlo(prices, distribution=dist, sim_length=2, n_sims=1,
                                            random_state=0, return_fit=True)
            assert list(fit["Parameters"]) == names


class TestRegimeFit:
    @pytest.fixture
    def fitted(self, two_regime_prices):
        return regime_switching_monte_carlo(two_regime_prices, n_starts=3, sim_length=10, n_sims=20,
                                            random_state=0, return_fit=True)

    def test_returns_only_sims_by_default(self, prices):
        out = regime_switching_monte_carlo(prices, n_regimes=1, n_starts=1, sim_length=5, n_sims=3,
                                           random_state=0)
        assert isinstance(out, np.ndarray)

    def test_fit_does_not_change_the_paths(self, two_regime_prices, fitted):
        plain = regime_switching_monte_carlo(two_regime_prices, n_starts=3, sim_length=10, n_sims=20,
                                             random_state=0)
        # The HMM fit can differ in the last bits with memory layout, so compare closely, not exactly
        np.testing.assert_allclose(fitted[0], plain, rtol=1e-9)

    def test_selected_count_has_the_lowest_bic(self, fitted):
        _, fit = fitted
        assert fit["Regime Count"] == min(fit["BIC"], key=fit["BIC"].get)
        assert set(fit["BIC"]) <= {1, 2, 3}
        # The history has a calm and a turbulent regime, so more than one is chosen
        assert fit["Regime Count"] >= 2

    def test_regime_table(self, fitted):
        _, fit = fitted
        regimes, P = fit["Regimes"], fit["Transition Matrix"]
        assert len(regimes) == fit["Regime Count"] == P.shape[0] == P.shape[1]
        assert list(regimes.columns) == ["Mean", "Volatility", "Expected Duration", "Current Probability"]
        assert regimes["Volatility"].is_monotonic_increasing
        assert regimes["Current Probability"].sum() == pytest.approx(1)
        np.testing.assert_allclose(P.sum(axis=1), 1)
        np.testing.assert_allclose(regimes["Expected Duration"], 1 / (1 - np.diag(P)))

    def test_given_regime_count_reports_only_its_bic(self, prices):
        _, fit = regime_switching_monte_carlo(prices, n_regimes=1, n_starts=1, sim_length=5, n_sims=2,
                                              random_state=0, return_fit=True)
        assert fit["Regime Count"] == 1 and list(fit["BIC"]) == [1]
        assert fit["Regimes"]["Current Probability"].iloc[0] == pytest.approx(1)


def prices_from_returns(returns):
    """A price series whose one-period simple returns are exactly `returns`."""
    return pd.Series(100 * np.cumprod(np.r_[1, 1 + np.asarray(returns)]))


def prices_from_log_returns(log_returns):
    """A price series whose one-period log returns are exactly `log_returns`."""
    return pd.Series(100 * np.exp(np.cumsum(np.r_[0, np.asarray(log_returns)])))


def exact_sample(dist, *args, n=1000):
    """n returns shaped exactly like `dist`: its quantiles at evenly spaced
    probabilities. Unlike a random sample, the best-fitting candidate is then
    not a matter of chance."""
    return dist.ppf((np.arange(n) + 0.5) / n, *args)


def regime_returns(vols, block, repeats, seed=0):
    """Log returns alternating between regimes of the given volatilities, in
    blocks of `block` bars, the cycle repeated `repeats` times."""
    return np.random.default_rng(seed).normal(0, np.tile(np.repeat(vols, block), repeats))


class TestAicSelection:
    """The parametric simulator's AIC search, checked against independent fits
    and against data whose best distribution is known."""

    def fit(self, returns, **kwargs):
        _, fit = parametric_monte_carlo(prices_from_returns(returns), sim_length=1, n_sims=1,
                                        random_state=0, return_fit=True, **kwargs)
        return fit

    def test_candidate_aics_match_an_independent_fit(self, prices):
        data = prices["X"].pct_change().dropna().to_numpy()
        fit = self.fit(data)
        for dist in (stats.norm, stats.t, stats.johnsonsu):
            params = dist.fit(data)
            expected = 2 * len(params) - 2 * dist.logpdf(data, *params).sum()
            assert fit["Candidates"][dist.name] == pytest.approx(expected, rel=1e-9), dist.name

    def test_penalty_counts_each_distributions_parameters(self):
        # On exactly normal data the extra parameters of t (df) and Johnson SU
        # (a, b in place of loc, scale shape) add no likelihood, so their AIC
        # exceeds the normal's by the penalty alone: 2 per extra parameter
        aics = self.fit(exact_sample(stats.norm, 0, 0.01))["Candidates"]
        assert aics["t"] - aics["norm"] == pytest.approx(2, abs=0.05)
        assert aics["johnsonsu"] - aics["norm"] == pytest.approx(4, abs=0.05)

    @pytest.mark.parametrize("dist, args, expected", [
        (stats.norm, (0, 0.01), "norm"),
        (stats.t, (3, 0, 0.01), "t"),
        (stats.johnsonsu, (-1.5, 2, 0, 0.01), "johnsonsu"),
    ], ids=["normal", "fat-tailed", "skewed"])
    def test_selects_the_distribution_the_data_came_from(self, dist, args, expected):
        assert self.fit(exact_sample(dist, *args))["Distribution"] == expected


class TestBicSelection:
    """The regime-switching simulator's BIC search, checked against closed-form
    values and against data with a known number of regimes."""

    def fit(self, log_returns, **kwargs):
        _, fit = regime_switching_monte_carlo(prices_from_log_returns(log_returns), sim_length=1,
                                              n_sims=1, random_state=0, return_fit=True, **kwargs)
        return fit

    def test_one_regime_matches_the_closed_form_normal_fit(self):
        # One regime is a single normal: its maximum-likelihood mean and
        # (population) volatility, log-likelihood and BIC have closed forms
        x = np.random.default_rng(1).normal(0.0005, 0.01, 600)
        fit = self.fit(x, n_regimes=1)
        log_lik = stats.norm.logpdf(x, x.mean(), x.std()).sum()
        assert fit["Log Likelihood"] == pytest.approx(log_lik, rel=1e-6)
        assert fit["BIC"][1] == pytest.approx(-2 * log_lik + 2 * np.log(len(x)), rel=1e-6)
        assert fit["Regimes"]["Mean"].iloc[0] == pytest.approx(x.mean(), rel=1e-4)
        assert fit["Regimes"]["Volatility"].iloc[0] == pytest.approx(x.std(), rel=1e-4)

    @pytest.mark.parametrize("n_regimes, free_parameters", [(1, 2), (2, 7), (3, 14)])
    def test_bic_counts_free_parameters(self, n_regimes, free_parameters):
        # K regimes: K - 1 initial probabilities, K(K - 1) transition
        # probabilities, and a mean and a variance per regime
        x = regime_returns([0.005, 0.02, 0.06], block=100, repeats=3)
        fit = self.fit(x, n_regimes=n_regimes)
        expected = -2 * fit["Log Likelihood"] + free_parameters * np.log(len(x))
        assert fit["BIC"][n_regimes] == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("log_returns, expected", [
        (np.random.default_rng(0).normal(0, 0.01, 600), 1),
        (regime_returns([0.01, 0.04], block=100, repeats=3), 2),
        (regime_returns([0.005, 0.02, 0.06], block=100, repeats=3), 3),
    ], ids=["one regime", "two regimes", "three regimes"])
    def test_selects_the_number_of_regimes_the_data_has(self, log_returns, expected):
        # Each case picked the right count for all of 40 seeds with the default
        # 10 restarts; the three-regime case needs them (3 restarts: 28 of 40)
        fit = self.fit(log_returns)
        assert fit["Regime Count"] == expected
        assert fit["Regime Count"] == min(fit["BIC"], key=fit["BIC"].get)
