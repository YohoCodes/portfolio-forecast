"""Verification of the private HMM EM (forecast/_hmm.py; MC7).

Recovery from simulated data with known parameters, equivalence with
hmmlearn on single-class data, stickiness of the kappa prior, the parameter
count and criteria, and edge cases. All randomness is seeded.
"""

import warnings

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from portfolio_forecast.forecast import _hmm


def simulate(n, P, mu, sd, classes, seed):
    """Seeded draw of n marks from the model (H1): regimes by P from regime 0, emissions by (regime, class)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    cum = np.cumsum(P, axis=1)
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = np.searchsorted(cum[s[t - 1]], u[t], side="right")
    return rng.normal(mu[s, classes], sd[s, classes]), s


def single_class_data(n=3000, seed=1):
    """Two regimes, one class: calm (vol 0.8%) and turbulent (vol 2.5%)."""
    P = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([[5e-4], [-1e-3]])
    sd = np.array([[0.008], [0.025]])
    return simulate(n, P, mu, sd, np.zeros(n, dtype=int), seed)[0]


# Recovery model: rows are regimes, columns classes (regular, gap). Gap
# emissions differ from regular ones in both mean and volatility, so swapped
# classes cannot pass.
REC_N = 24_000
REC_P = np.array([[0.99, 0.01], [0.02, 0.98]])
REC_MU = np.array([[2e-4, 1e-3], [-5e-4, -3e-3]])
REC_SD = np.array([[0.002, 0.005], [0.006, 0.015]])


@pytest.fixture(scope="module")
def recovered():
    """Fit to 24,000 simulated marks, one gap per 8; fitted regimes ordered like the true ones."""
    classes = (np.arange(REC_N) % 8 == 7).astype(int)
    ell, states = simulate(REC_N, REC_P, REC_MU, REC_SD, classes, seed=0)
    fit = _hmm.fit_hmm(ell, classes, n_states=2, n_starts=5, random_state=0)
    # Label switching: the fitted regimes are numbered arbitrarily, so order
    # them by regular-mark variance, as the true regimes are (calm first)
    order = np.argsort(fit.var[:, 0])
    return fit, order, states


def expected_counts():
    """Expected marks per (regime, class) cell and per regime under the recovery model.

    Stationary regime share (P_10, P_01) / (P_01 + P_10) = (2/3, 1/3) times the
    class share (7/8, 1/8).
    """
    stat = np.array([REC_P[1, 0], REC_P[0, 1]]) / (REC_P[0, 1] + REC_P[1, 0])
    return REC_N * stat[:, None] * np.array([7 / 8, 1 / 8])[None, :], REC_N * stat


class TestRecovery:
    """Known parameters are recovered from 24,000 marks with one gap per 8 marks.

    Tolerances are 5 standard errors of the complete-data estimator (as if the
    regimes were observed), from the true parameters and expected counts; the
    regimes are well separated, so the extra error from inferring them is small.
    """

    def test_converged(self, recovered):
        assert recovered[0].converged

    def test_means(self, recovered):
        # Tolerance: 5 standard errors of a known-regime sample mean, sd / sqrt(N)
        fit, order, _ = recovered
        N, _ = expected_counts()
        np.testing.assert_array_less(np.abs(fit.mu[order] - REC_MU), 5 * REC_SD / np.sqrt(N))

    def test_volatilities(self, recovered):
        # Tolerance: 5 standard errors of a normal sample sd, sd / sqrt(2 N)
        fit, order, _ = recovered
        N, _ = expected_counts()
        np.testing.assert_array_less(np.abs(np.sqrt(fit.var[order]) - REC_SD), 5 * REC_SD / np.sqrt(2 * N))

    def test_transitions(self, recovered):
        # Tolerance: 5 standard errors of a binomial switching rate, sqrt(p (1 - p) / visits)
        fit, order, _ = recovered
        _, visits = expected_counts()
        P_hat = fit.P[np.ix_(order, order)]
        se = np.sqrt(REC_P * (1 - REC_P) / visits[:, None])
        np.testing.assert_array_less(np.abs(P_hat - REC_P), 5 * se)
        np.testing.assert_allclose(P_hat.sum(axis=1), 1.0, rtol=0, atol=1e-12)

    def test_regimes_decoded(self, recovered):
        # The regimes differ threefold in volatility, so most marks are decodable
        fit, order, states = recovered
        assert np.mean(fit.gamma[:, order].argmax(axis=1) == states) > 0.95
        np.testing.assert_allclose(fit.gamma.sum(axis=1), 1.0, rtol=0, atol=1e-12)


class TestEquivalence:
    """Single class, kappa = 0: the model is hmmlearn's GaussianHMM."""

    pi = np.array([0.3, 0.7])
    P = np.array([[0.97, 0.03], [0.06, 0.94]])
    mu = np.array([[1e-3], [-2e-3]])
    var = np.array([[1e-4], [4e-4]])

    def reference(self, pi, P, mu, var):
        model = GaussianHMM(n_components=len(pi), covariance_type="diag")
        model.startprob_, model.transmat_, model.means_, model.covars_ = pi, P, mu, var
        return model

    def test_log_likelihood_at_fixed_parameters(self):
        ell = single_class_data()
        ours = _hmm.log_likelihood(ell, self.pi, self.P, self.mu, self.var)
        ref = self.reference(self.pi, self.P, self.mu, self.var).score(ell[:, None])
        assert ours == pytest.approx(ref, rel=1e-6)

    def test_log_likelihood_three_regimes_long_series(self):
        # 20,000 marks: many blocks in the forward pass, and a long underflow-prone product
        ell = single_class_data(n=20_000, seed=2)
        pi = np.array([0.2, 0.5, 0.3])
        P = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.02, 0.08, 0.9]])
        mu = np.array([[0.0], [1e-3], [-2e-3]])
        var = np.array([[5e-5], [2e-4], [9e-4]])
        ours = _hmm.log_likelihood(ell, pi, P, mu, var)
        ref = self.reference(pi, P, mu, var).score(ell[:, None])
        assert ours == pytest.approx(ref, rel=1e-6)

    def test_posteriors_at_fixed_parameters(self):
        ell = single_class_data()
        ours = _hmm.posteriors(ell, self.pi, self.P, self.mu, self.var)
        ref = self.reference(self.pi, self.P, self.mu, self.var).predict_proba(ell[:, None])
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-8)

    def test_fitted_log_likelihood(self):
        ell = single_class_data()
        ours = _hmm.fit_hmm(ell, n_states=2, n_starts=5, random_state=0).log_likelihood
        best = -np.inf
        for seed in range(5):
            # covars_prior=0 and a negligible min_covar make hmmlearn's fit maximum likelihood
            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, tol=1e-6,
                                covars_prior=0.0, min_covar=1e-12, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(ell[:, None])
            best = max(best, model.score(ell[:, None]))
        assert ours == pytest.approx(best, rel=1e-4)


class TestStickiness:
    """The kappa prior (MC7, S4)."""

    def test_large_kappa_makes_regimes_persist(self):
        ell = single_class_data(n=2000)
        fit = _hmm.fit_hmm(ell, n_states=2, kappa=1e6, n_starts=3, random_state=0)
        assert (np.diag(fit.P) > 0.99).all()

    def test_map_fit_is_a_fixed_point(self):
        # With kappa > 0, EM climbs the log posterior J (H2), not log L; a fit stopped
        # on the wrong objective is not a fixed point of the penalised update (S4)
        ell = single_class_data(n=2000)
        fit = _hmm.fit_hmm(ell, n_states=2, kappa=500.0, n_starts=3, random_state=0)
        assert fit.converged
        cls = np.zeros(len(ell), dtype=np.intp)
        _, gamma, xi_sum, _ = _hmm._e_step(ell, cls, fit.pi, fit.P, fit.mu, fit.var)
        _, P, mu, var = _hmm._m_step(ell, cls, 1, gamma, xi_sum, 500.0)
        np.testing.assert_allclose(P, fit.P, rtol=0, atol=1e-5)
        np.testing.assert_allclose(np.sqrt(var), np.sqrt(fit.var), rtol=1e-4)

    def test_kappa_zero_is_the_unpenalized_fit(self):
        ell = single_class_data(n=2000)
        default = _hmm.fit_hmm(ell, n_states=2, n_starts=3, random_state=0)
        zero = _hmm.fit_hmm(ell, n_states=2, kappa=0.0, n_starts=3, random_state=0)
        for name in ("pi", "P", "mu", "var", "gamma", "filtered_last"):
            np.testing.assert_array_equal(getattr(zero, name), getattr(default, name))
        assert zero.log_likelihood == default.log_likelihood


class TestMStep:
    """Closed-form updates (S1)-(S5) on hand-made posteriors."""

    def test_emissions_initial_distribution_and_floor(self):
        # Two gaps then three regular marks, two regimes. Per (regime, class) cell:
        #   regular, regime 0: weights 1, 0.5, 0 on 0.01, -0.02, 0.04; N = 1.5,
        #     mu = (0.01 - 0.01) / 1.5 = 0, var = (1e-4 + 0.5 * 4e-4) / 1.5 = 2e-4
        #   regular, regime 1: weights 0, 0.5, 1; N = 1.5, mu = (-0.01 + 0.04) / 1.5 = 0.02,
        #     var = (0.5 * 0.04^2 + 0.02^2) / 1.5 = 8e-4
        #   gap, regime 0: weights 0.25, 1 on 0.03, 0.05; N = 1.25, mu = 0.0575 / 1.25 = 0.046,
        #     var = (0.25 * 0.016^2 + 0.004^2) / 1.25 = 6.4e-5
        #   gap, regime 1: weights 0.75, 0; mu = 0.03, var = 0, floored to 1e-12
        # pi is the first posterior, (0.25, 0.75), not the last, (0, 1).
        ell = np.array([0.03, 0.05, 0.01, -0.02, 0.04])
        cls = np.array([1, 1, 0, 0, 0], dtype=np.intp)
        gamma = np.array([[0.25, 0.75], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        pi, _, mu, var = _hmm._m_step(ell, cls, 2, gamma, np.ones((2, 2)), 0.0)
        np.testing.assert_array_equal(pi, [0.25, 0.75])
        np.testing.assert_allclose(mu, [[0.0, 0.046], [0.02, 0.03]], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(var, [[2e-4, 6.4e-5], [8e-4, 1e-12]], rtol=1e-12, atol=0)

    def test_transition_update_is_expected_counts_plus_kappa(self):
        # M-step on hand-made expected counts Xi: kappa = 0 is Xi row-normalised
        # exactly; kappa > 0 adds kappa to the diagonal only
        xi_sum = np.array([[30.0, 10.0], [5.0, 15.0]])
        gamma = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
        ell = np.array([0.01, -0.02, 0.03])
        cls = np.zeros(3, dtype=np.intp)
        P0 = _hmm._m_step(ell, cls, 1, gamma, xi_sum, 0.0)[1]
        np.testing.assert_array_equal(P0, np.array([[0.75, 0.25], [0.25, 0.75]]))
        P10 = _hmm._m_step(ell, cls, 1, gamma, xi_sum, 10.0)[1]
        np.testing.assert_allclose(P10, np.array([[40 / 50, 10 / 50], [5 / 30, 25 / 30]]), rtol=1e-15)


class TestCriteria:
    """Parameter count k and AIC/BIC (MC7, MC5)."""

    @pytest.mark.parametrize("S, k_regime, k_independent", [(2, 11, 7), (3, 20, 14)])
    def test_n_params_matches_spec_table(self, S, k_regime, k_independent):
        # MC7: "regime" k = (S-1) + S(S-1) + 4S; "independent" fits regular marks
        # only, the single-class k = (S-1) + S(S-1) + 2S (gap normal's +2 is the caller's)
        assert _hmm.n_params(S, 2) == k_regime
        assert _hmm.n_params(S, 1) == k_independent

    def test_fit_reports_its_parameter_count(self):
        ell = single_class_data(n=800)
        classes = (np.arange(800) % 8 == 7).astype(int)
        assert _hmm.fit_hmm(ell, n_states=2, n_starts=2, random_state=0).n_params == 7
        assert _hmm.fit_hmm(ell, classes, n_states=2, n_starts=2, random_state=0).n_params == 11

    @pytest.mark.parametrize("criterion", ["BIC", "bic", "Bic"])
    def test_bic(self, criterion):
        fit = _hmm.HMMFit(pi=None, P=None, mu=None, var=None, log_likelihood=100.0, gamma=None,
                          filtered_last=None, n_params=11, converged=True)
        assert _hmm.score(fit, 1000, criterion) == pytest.approx(-200.0 + 11 * np.log(1000), rel=1e-15)

    @pytest.mark.parametrize("criterion", ["AIC", "aic"])
    def test_aic(self, criterion):
        fit = _hmm.HMMFit(pi=None, P=None, mu=None, var=None, log_likelihood=100.0, gamma=None,
                          filtered_last=None, n_params=11, converged=True)
        assert _hmm.score(fit, 1000, criterion) == -200.0 + 22

    def test_unknown_criterion_raises(self):
        fit = _hmm.HMMFit(pi=None, P=None, mu=None, var=None, log_likelihood=0.0, gamma=None,
                          filtered_last=None, n_params=2, converged=True)
        with pytest.raises(ValueError, match="AIC"):
            _hmm.score(fit, 10, "HQC")


class TestFitContract:
    def test_deterministic_for_a_fixed_seed(self):
        ell = single_class_data(n=1500)
        a = _hmm.fit_hmm(ell, n_states=2, n_starts=3, random_state=7)
        b = _hmm.fit_hmm(ell, n_states=2, n_starts=3, random_state=7)
        for name in ("pi", "P", "mu", "var", "gamma", "filtered_last"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert a.log_likelihood == b.log_likelihood

    def test_shapes(self):
        ell = single_class_data(n=800)
        classes = (np.arange(800) % 8 == 7).astype(int)
        fit = _hmm.fit_hmm(ell, classes, n_states=2, n_starts=2, random_state=0)
        assert fit.mu.shape == fit.var.shape == (2, 2)
        assert fit.P.shape == (2, 2) and fit.pi.shape == fit.filtered_last.shape == (2,)
        assert fit.gamma.shape == (800, 2)
        assert fit.filtered_last.sum() == pytest.approx(1.0)


class TestEdgeCases:
    def test_fewer_returns_than_parameters_raises(self):
        # S = 3, one class: k = 2 + 6 + 6 = 14 > 13 returns (each cell still has 2 * S = 6)
        ell = np.random.default_rng(0).normal(0, 0.01, 13)
        with pytest.raises(ValueError, match="14 parameters but there are only 13 returns"):
            _hmm.fit_hmm(ell, n_states=3, random_state=0)

    def test_too_few_returns_of_a_class_raises(self):
        # S = 2 needs 2 * S = 4 gap returns for its two gap cells; only 3 given
        ell = np.random.default_rng(0).normal(0, 0.01, 200)
        classes = np.zeros(200, dtype=int)
        classes[[50, 100, 150]] = 1
        with pytest.raises(ValueError, match="at least 4 gap returns, got 3"):
            _hmm.fit_hmm(ell, classes, n_states=2, random_state=0)

    def test_constant_input_is_degenerate(self):
        with pytest.raises(ValueError, match="degenerate"):
            _hmm.fit_hmm(np.full(500, 0.001), n_states=2, n_starts=3, random_state=0)

    def test_constant_input_single_regime_is_degenerate(self):
        with pytest.raises(ValueError, match="degenerate"):
            _hmm.fit_hmm(np.zeros(500), n_states=1, n_starts=1, random_state=0)

    @pytest.mark.parametrize("classes", [np.array([0, 1, 2] * 100), np.zeros(10, dtype=int)])
    def test_bad_classes_raise(self, classes):
        with pytest.raises(ValueError, match="classes"):
            _hmm.fit_hmm(np.zeros(300) + np.arange(300) * 1e-4, classes, n_states=1)


# Short-bar recovery model (MC7): sessions of 8 marks, a gap then 7 regular bars, the last
# of them short (rho = 1/4, e.g. a 15-minute last bar under hourly bars). The
# regular means are large enough that ignoring rho in the mean is visible too.
RHO_N = 24_000
RHO_SHORT = 0.25
RHO_MU = np.array([[2e-3, 1e-3], [-3e-3, -3e-3]])
RHO_SD = REC_SD


@pytest.fixture(scope="module")
def recovered_rho():
    """Fit to 24,000 marks drawn from (H1) with rho = 1/4 on one regular bar in seven."""
    t = np.arange(RHO_N)
    classes = (t % 8 == 0).astype(int)
    rho = np.where(t % 8 == 7, RHO_SHORT, 1.0)
    rng = np.random.default_rng(3)
    _, states = simulate(RHO_N, REC_P, RHO_MU, RHO_SD, classes, seed=3)
    # (H1): a short bar has mean rho mu and variance rho var of its cell
    ell = rng.normal(rho * RHO_MU[states, classes], np.sqrt(rho) * RHO_SD[states, classes])
    fit = _hmm.fit_hmm(ell, classes, rho, n_states=2, n_starts=5, random_state=0)
    order = np.argsort(fit.var[:, 0])
    return fit, order, rho, classes


class TestRecoveryWithShortBars:
    """MC7: known per-unit parameters are recovered when one regular bar in seven is short.

    Tolerances are 5 complete-data standard errors. For a cell with N marks
    and length ratios rho_t, the mean estimator sum l / sum rho has standard
    error sd / sqrt(sum rho); each (l - rho mu)^2 / rho has expectation var,
    so the sd estimator keeps sd / sqrt(2 N).
    """

    @staticmethod
    def cell_counts():
        """Expected marks and expected sum of rho per (regime, class) cell."""
        N, _ = expected_counts()
        regular_rho = (6 + RHO_SHORT) / 7
        return N, N * np.array([regular_rho, 1.0])[None, :]

    def test_converged(self, recovered_rho):
        assert recovered_rho[0].converged

    def test_means(self, recovered_rho):
        fit, order, _, _ = recovered_rho
        _, rho_sum = self.cell_counts()
        np.testing.assert_array_less(np.abs(fit.mu[order] - RHO_MU), 5 * RHO_SD / np.sqrt(rho_sum))

    def test_volatilities(self, recovered_rho):
        fit, order, _, _ = recovered_rho
        N, _ = self.cell_counts()
        np.testing.assert_array_less(np.abs(np.sqrt(fit.var[order]) - RHO_SD), 5 * RHO_SD / np.sqrt(2 * N))

    def test_transitions(self, recovered_rho):
        fit, order, _, _ = recovered_rho
        _, visits = expected_counts()
        se = np.sqrt(REC_P * (1 - REC_P) / visits[:, None])
        np.testing.assert_array_less(np.abs(fit.P[np.ix_(order, order)] - REC_P), 5 * se)


class TestShortBarEmissions:
    """(H1), (S2), (S3) with rho != 1, checked by hand."""

    def test_single_regime_log_likelihood_is_a_sum_of_normal_densities(self):
        # S = 1: log L is the sum of log N(l_t; rho_t mu, rho_t var)
        ell = np.array([0.01, -0.004, 0.002, 0.03])
        rho = np.array([1.0, 0.5, 0.25, 1.0])
        mu, var = 0.004, 1e-4
        expected = -0.5 * np.sum(np.log(2 * np.pi * rho * var) + (ell - rho * mu) ** 2 / (rho * var))
        ours = _hmm.log_likelihood(ell, [1.0], [[1.0]], [[mu]], [[var]], rho=rho)
        assert ours == pytest.approx(expected, rel=1e-13)

    def test_m_step_with_short_bars(self):
        # One regime, regular marks 0.01, 0.006, -0.002 with rho 1, 0.5, 0.25, all weight 1:
        #   mu  = (0.01 + 0.006 - 0.002) / (1 + 0.5 + 0.25) = 0.014 / 1.75 = 0.008
        #   var = [ (0.01 - 0.008)^2 / 1 + (0.006 - 0.004)^2 / 0.5 + (-0.002 - 0.002)^2 / 0.25 ] / 3
        #       = [ 4e-6 + 8e-6 + 6.4e-5 ] / 3 = 7.6e-5 / 3
        # Gap mark 0.02 (rho 1) is its own cell: mu = 0.02, var floored.
        ell = np.array([0.02, 0.01, 0.006, -0.002])
        cls = np.array([1, 0, 0, 0], dtype=np.intp)
        rho = np.array([1.0, 1.0, 0.5, 0.25])
        gamma = np.ones((4, 1))
        _, _, mu, var = _hmm._m_step(ell, cls, 2, gamma, np.ones((1, 1)), 0.0, rho=rho)
        np.testing.assert_allclose(mu, [[0.008, 0.02]], rtol=1e-12)
        np.testing.assert_allclose(var, [[7.6e-5 / 3, 1e-12]], rtol=1e-12)

    def test_unit_rho_is_the_default(self):
        ell = single_class_data(n=1500)
        a = _hmm.fit_hmm(ell, n_states=2, n_starts=2, random_state=4)
        b = _hmm.fit_hmm(ell, rho=np.ones(1500), n_states=2, n_starts=2, random_state=4)
        for name in ("pi", "P", "mu", "var", "gamma", "filtered_last"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    @pytest.mark.parametrize("rho", [np.full(300, 0.0), np.full(300, np.nan), np.ones(299)])
    def test_bad_rho_raises(self, rho):
        with pytest.raises(ValueError, match="rho"):
            _hmm.fit_hmm(np.arange(300) * 1e-4, rho=rho, n_states=1)

    def test_rescaled_gap_raises(self):
        # U2: a gap is one unit whatever its clock length
        classes = (np.arange(300) % 8 == 0).astype(int)
        rho = np.where(classes == 1, 0.5, 1.0)
        with pytest.raises(ValueError, match="gap"):
            _hmm.fit_hmm(np.random.default_rng(0).normal(0, 0.01, 300), classes, rho, n_states=1)
