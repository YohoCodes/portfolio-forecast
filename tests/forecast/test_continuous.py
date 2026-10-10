"""Continuous mode, calendar=None (U3): one pool, ρ = 1, HMM with C = 1, marks t_T + kΔ.

The history is hourly and 24/7 but labelled at times that XNYS would read
as sessions with gaps, so any calendar logic leaking into continuous mode
shows up as a gap class, a short bar or a session-counted horizon.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from mc_helpers import from_log_returns, log_steps
from scipy import stats

from portfolio_forecast.forecast import (
    _common,
    nonparametric_monte_carlo,
    parametric_monte_carlo,
    regime_switching_monte_carlo,
)

SIMULATORS = [
    pytest.param(nonparametric_monte_carlo, {}, id="bootstrap"),
    pytest.param(parametric_monte_carlo, {"distribution": stats.norm}, id="parametric"),
    pytest.param(regime_switching_monte_carlo, {"n_regimes": 1, "n_starts": 1}, id="regime"),
]

STEP = pd.Timedelta(hours=1)


@pytest.fixture(scope="module")
def crypto():
    """600 hourly marks, 24/7 from a Monday 09:30 New York (13:30 UTC), with distinct log returns."""
    index = pd.date_range("2025-03-03 13:30", periods=600, freq="h", tz="UTC")
    ell = np.random.default_rng(0).normal(0, 0.01, 599)
    return from_log_returns(index, ell), ell


class TestLayout:
    """MC2, MC8 analog (U3): future marks t_T + kΔ, sim_length in bars."""

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_marks_every_bar_from_the_last(self, simulate, kwargs, crypto):
        values = crypto[0]
        with warnings.catch_warnings():
            # No intraday sim_length warning: continuous mode counts bars, as 0.3.0 did
            warnings.simplefilter("error")
            sims = simulate(values, sim_length=30, n_sims=3, random_state=0, calendar=None, **kwargs)
        expected = pd.DatetimeIndex([values.index[-1] + k * STEP for k in range(1, 31)])
        assert sims.dates.equals(expected)
        assert list(sims.classes) == ["regular"] * 30
        assert sims.paths.shape == (3, 31) and (sims.paths[:, 0] == 1.0).all()


class TestOnePool:
    """MC3-MC7 analog (U3): one pool, no rescaling."""

    def test_history_is_one_class_with_unit_rho(self, crypto):
        history = _common.prepare_history(crypto[0], None, None, 5, "test")
        assert history.mode == "continuous"
        assert set(history.classes) == {"regular"}
        np.testing.assert_array_equal(history.rho, np.ones(599))
        np.testing.assert_array_equal(history.future_rho, np.ones(5))
        # ℓ = ln V_t − ln V_{t−1} of exp-built values: equal up to rounding
        np.testing.assert_allclose(history.ell, crypto[1], rtol=0, atol=1e-14)

    def test_bootstrap_draws_are_historical_returns_unscaled(self, crypto):
        values, ell = crypto
        sims = nonparametric_monte_carlo(values, sim_length=24, n_sims=100, random_state=0, calendar=None)
        drawn = log_steps(sims.paths).ravel()
        assert np.isclose(drawn[:, None], ell[None, :], rtol=0, atol=1e-12).any(axis=1).all()

    def test_parametric_fits_one_class(self, crypto):
        values, ell = crypto
        fit = parametric_monte_carlo(values, distribution=stats.norm, sim_length=2, n_sims=2, calendar=None).fit
        assert list(fit) == ["regular"]
        assert fit["regular"]["Parameters"]["loc"] == pytest.approx(ell.mean(), rel=1e-6)

    def test_hmm_has_regular_emissions_only(self, crypto):
        values, ell = crypto
        fit = regime_switching_monte_carlo(values, n_regimes=1, n_starts=1, sim_length=2, n_sims=2,
                                           calendar=None).fit
        assert "Gap Mean" not in fit["Regimes"] and "Gap" not in fit
        assert fit["Regimes"]["Mean"].iloc[0] == pytest.approx(ell.mean(), rel=1e-9)
        assert fit["Regimes"]["Volatility"].iloc[0] == pytest.approx(ell.std(), rel=1e-9)
        # C = 1: k = 2 at S = 1
        assert fit["BIC"][1] == pytest.approx(-2 * fit["Log Likelihood"] + 2 * np.log(599), rel=1e-12)

    def test_missing_grid_point_is_irregular_and_left_out(self, crypto):
        # U3: a return across a missing grid point is irregular, so in no pool
        values, ell = crypto
        gappy = values.drop(values.index[100])
        history = _common.prepare_history(gappy, None, None, 2, "test")
        assert len(history.ell) == 597
        two_bar = ell[99] + ell[100]
        assert not np.isclose(history.ell, two_bar, rtol=0, atol=1e-15).any()

    def test_same_data_on_xnys_has_gaps_and_short_bars(self, crypto):
        # The contrast: on XNYS these labels are an hourly session history whose
        # off-session marks raise, so calendar=None is what keeps them one pool
        with pytest.raises(ValueError):
            _common.prepare_history(crypto[0], None, "XNYS", 2, "test")


class TestDaily:
    def test_daily_grid_counts_bars(self):
        # Δ = 1 day: still t_T + kΔ, every calendar day, weekends included
        index = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
        values = from_log_returns(index, np.random.default_rng(1).normal(0, 0.02, 199))
        sims = nonparametric_monte_carlo(values, sim_length=7, n_sims=2, random_state=0, calendar=None)
        assert sims.dates.equals(pd.date_range(index[-1] + pd.Timedelta(days=1), periods=7, freq="D"))
