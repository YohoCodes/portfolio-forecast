"""The bootstrap simulator (forecast/bootstrap.py; MC3, MC6), seeded.

Gap and regular log returns are drawn from disjoint sets, so every simulated
step can be traced to the pool it came from.
"""
import numpy as np
import pandas as pd
import pytest
from mc_helpers import from_log_returns, hourly_marks, log_steps, mark_kinds, sessions

from portfolio_forecast import BacktestResult
from portfolio_forecast.forecast import _common, nonparametric_monte_carlo

pytestmark = pytest.mark.filterwarnings("ignore:.*counts sessions:UserWarning")


@pytest.fixture(scope="module")
def traced():
    """60 hourly sessions: gaps in [0.02, 0.03], regular full bars in [-0.002, 0.002], short bars at √ρ scale."""
    index = hourly_marks(sessions("2025-03-03", 60))
    kinds, rho = mark_kinds(index)
    rng = np.random.default_rng(1)
    ell = np.where(kinds == "gap", rng.uniform(0.02, 0.03, len(kinds)),
                   np.sqrt(rho) * rng.uniform(-0.002, 0.002, len(kinds)))
    return from_log_returns(index, ell), kinds, rho, ell


def is_in(drawn, pool):
    """Return, for each drawn value, whether it equals some pool value to 1e-12."""
    return np.isclose(np.asarray(drawn).ravel()[:, None], np.asarray(pool)[None, :], rtol=0, atol=1e-12).any(axis=1)


class TestPools:
    """MC3: two pools, i.i.d. within class."""

    def test_draws_come_only_from_their_class_pool(self, traced):
        values, kinds, rho, ell = traced
        sims = nonparametric_monte_carlo(values, sim_length=5, n_sims=200, random_state=0)
        steps = log_steps(sims.paths)
        history = _common.prepare_history(values, None, "XNYS", 5, "test")
        gap_cols = sims.classes == "gap"
        full_cols = (sims.classes == "regular") & (history.future_rho == 1)
        assert is_in(steps[:, gap_cols], ell[kinds == "gap"]).all()
        assert is_in(steps[:, full_cols], history.unit_pool("regular")).all()
        # Gaps sit in [0.02, 0.03] and regular returns well inside ±0.002, so no
        # draw crossed over between classes
        assert (steps[:, gap_cols] >= 0.02).all() and (np.abs(steps[:, ~gap_cols]) < 0.005).all()

    def test_short_future_bars_are_mapped_regular_draws(self, traced):
        # MC3: a short future bar is ρ m̂ + √ρ (ℓ' − m̂) of a regular pool draw
        values, kinds, rho, ell = traced
        sims = nonparametric_monte_carlo(values, sim_length=5, n_sims=200, random_state=0)
        history = _common.prepare_history(values, None, "XNYS", 5, "test")
        short = history.future_rho < 1
        assert short.any()
        mapped = _common.from_unit(history.unit_pool("regular")[:, None], history.future_rho[short][None, :],
                                   history.m_hat)
        steps = log_steps(sims.paths)[:, short]
        for j in range(short.sum()):
            assert is_in(steps[:, j], mapped[:, j]).all()

    def test_every_pool_value_can_be_drawn(self, traced):
        # i.i.d. with replacement over the whole pool: 59 gaps, 2,000 paths x 5 gaps
        values, kinds, _, ell = traced
        sims = nonparametric_monte_carlo(values, sim_length=5, n_sims=2000, random_state=0)
        steps = log_steps(sims.paths)[:, sims.classes == "gap"]
        assert is_in(ell[kinds == "gap"], steps.ravel()).all()

    def test_irregular_never_drawn(self, traced):
        values, kinds, _, ell = traced
        classes = pd.Series(kinds, index=values.index[1:])
        # Mark a gap and a bar irregular; their values are distinctive by construction
        marked = [np.flatnonzero(kinds == "gap")[3], np.flatnonzero(kinds == "regular")[10]]
        classes.iloc[marked] = "irregular"
        sims = nonparametric_monte_carlo(BacktestResult(values, values.index, classes), sim_length=5,
                                         n_sims=3000, random_state=0)
        steps = log_steps(sims.paths)
        assert not is_in(steps, ell[marked]).any()

    def test_constant_pools_give_one_deterministic_path(self):
        # Known answer: every gap 0.01, every regular full bar 0.002 with short bars ρ x 0.002
        index = hourly_marks(sessions("2025-03-03", 40))
        kinds, rho = mark_kinds(index)
        ell = np.where(kinds == "gap", 0.01, rho * 0.002)
        sims = nonparametric_monte_carlo(from_log_returns(index, ell), sim_length=2, n_sims=4, random_state=3)
        tau = _common.prepare_history(from_log_returns(index, ell), None, "XNYS", 2, "test").future_rho
        expected = np.exp(np.cumsum(np.where(sims.classes == "gap", 0.01, tau * 0.002)))
        np.testing.assert_allclose(sims.paths[:, 1:], np.broadcast_to(expected, (4, len(expected))), rtol=1e-12)
        assert sims.fit is None


class TestContract:
    def test_seeded_and_reproducible(self, traced):
        values = traced[0]
        a = nonparametric_monte_carlo(values, sim_length=3, n_sims=50, random_state=7)
        b = nonparametric_monte_carlo(values, sim_length=3, n_sims=50, random_state=7)
        np.testing.assert_array_equal(a.paths, b.paths)
        assert len(np.unique(a.paths[:, -1])) > 40

    def test_min_sample_names_the_short_class(self):
        index = hourly_marks(sessions("2025-03-03", 10))
        kinds, rho = mark_kinds(index)
        values = from_log_returns(index, np.random.default_rng(0).normal(0, 0.01, len(kinds)))
        with pytest.raises(ValueError, match="The gap class has 9 returns"):
            nonparametric_monte_carlo(values, sim_length=2, n_sims=2)
        sims = nonparametric_monte_carlo(values, sim_length=2, n_sims=2, min_sample=0, random_state=0)
        assert sims.paths.shape[0] == 2

    def test_bad_n_sims_raises(self, traced):
        with pytest.raises(ValueError, match="n_sims"):
            nonparametric_monte_carlo(traced[0], sim_length=2, n_sims=0)
