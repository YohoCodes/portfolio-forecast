"""The parametric simulator (forecast/parametric.py; MC3, MC4, MC5, MC6), seeded.

Samples shaped exactly like a distribution (its quantiles at evenly spaced
probabilities) make the best-fitting family a matter of shape, not chance.
"""
import numpy as np
import pytest
from mc_helpers import from_log_returns, hourly_marks, log_steps, mark_kinds, sessions
from scipy import stats

from portfolio_forecast.forecast import _common, parametric_monte_carlo

pytestmark = pytest.mark.filterwarnings("ignore:.*counts sessions:UserWarning")


def exact_sample(dist, *args, n=1000):
    """n values shaped exactly like `dist`: its quantiles at (i + 0.5) / n."""
    return dist.ppf((np.arange(n) + 0.5) / n, *args)


def plain(log_returns):
    """A plain array of values whose log returns are exactly `log_returns`."""
    return 100 * np.exp(np.cumsum(np.r_[0.0, log_returns]))


@pytest.fixture(scope="module")
def intraday():
    """80 hourly sessions: gaps t(4) at 1%, regular full bars normal at 0.3%, short bars at √ρ scale.

    The gaps are shuffled quantiles of t(4), the bars of a normal, so each
    class's best family is known.
    """
    index = hourly_marks(sessions("2025-03-03", 80))
    kinds, rho = mark_kinds(index)
    rng = np.random.default_rng(0)
    n_gap, n_reg = (kinds == "gap").sum(), (kinds == "regular").sum()
    ell = np.empty(len(kinds))
    ell[kinds == "gap"] = rng.permutation(exact_sample(stats.t, 4, 0, 0.01, n=n_gap))
    ell[kinds == "regular"] = np.sqrt(rho[kinds == "regular"]) * rng.permutation(
        exact_sample(stats.norm, 0, 0.003, n=n_reg))
    return from_log_returns(index, ell), kinds, rho, ell


class TestSelection:
    """MC5: per class, the lowest criterion of normal, t and Johnson SU wins."""

    @pytest.mark.parametrize("dist, args, expected", [
        (stats.norm, (0, 0.01), "norm"),
        (stats.t, (3, 0, 0.01), "t"),
        (stats.johnsonsu, (-1.5, 2, 0, 0.01), "johnsonsu"),
    ], ids=["normal", "fat-tailed", "skewed"])
    @pytest.mark.parametrize("criterion", ["BIC", "AIC"])
    def test_selects_the_family_the_data_came_from(self, dist, args, expected, criterion):
        sims = parametric_monte_carlo(plain(exact_sample(dist, *args)), sim_length=1, n_sims=1, random_state=0,
                                      criterion=criterion)
        assert sims.fit["regular"]["Distribution"] == expected

    def test_bic_is_the_default_and_case_insensitive(self):
        data = plain(exact_sample(stats.norm, 0, 0.01))
        default = parametric_monte_carlo(data, sim_length=1, n_sims=1, random_state=0).fit["regular"]
        lower = parametric_monte_carlo(data, sim_length=1, n_sims=1, random_state=0, criterion="bic").fit["regular"]
        assert default["Criterion"] == lower["Criterion"] == "BIC"
        assert default["BIC"] == lower["BIC"] and "AIC" not in default

    def test_penalties(self):
        # On exactly normal data t's df and Johnson SU's two shapes add (almost) no
        # likelihood, so scores differ by the penalty: 2 per extra parameter (AIC),
        # ln n per extra parameter (BIC), n = 1000
        data = plain(exact_sample(stats.norm, 0, 0.01))
        aic = parametric_monte_carlo(data, sim_length=1, n_sims=1, criterion="AIC").fit["regular"]["Candidates"]
        bic = parametric_monte_carlo(data, sim_length=1, n_sims=1, criterion="BIC").fit["regular"]["Candidates"]
        assert aic["t"] - aic["norm"] == pytest.approx(2, abs=0.05)
        assert aic["johnsonsu"] - aic["norm"] == pytest.approx(4, abs=0.05)
        assert bic["t"] - bic["norm"] == pytest.approx(np.log(1000), abs=0.05)
        assert bic["johnsonsu"] - bic["norm"] == pytest.approx(2 * np.log(1000), abs=0.05)

    @pytest.mark.parametrize("seed", [0, 7])
    def test_t_fit_reaches_the_maximum_on_fat_tails(self, seed):
        # MC5: on t_3 returns at a 0.2% scale scipy's default t fit stops near
        # df = 1.9 (about 20 log-likelihood units short), where Johnson SU wins;
        # the converged maximum has df near 3 and t wins
        ell = 0.0005 + 0.002 * np.random.default_rng(seed).standard_t(3, 2000)
        stalled = stats.t.fit(ell)
        assert stalled[0] < 2.0
        fit = parametric_monte_carlo(plain(ell), sim_length=1, n_sims=1, random_state=0).fit["regular"]
        assert fit["Distribution"] == "t"
        df = fit["Parameters"]["df"]
        assert 2.5 < df < 3.6
        params = list(fit["Parameters"].values())
        assert stats.t.logpdf(ell, *params).sum() > stats.t.logpdf(ell, *stalled).sum() + 10
        # A stationary point: a start at the true df finds nothing better
        near = stats.t.fit(ell, 3.0, loc=np.median(ell), scale=0.002)
        assert stats.t.logpdf(ell, *params).sum() >= stats.t.logpdf(ell, *near).sum() - 1e-6

    def test_johnson_su_fit_is_never_below_any_start(self):
        ell = np.random.default_rng(3).permutation(exact_sample(stats.johnsonsu, -1.5, 2, 0.0005, 0.002, n=3000))
        fit = parametric_monte_carlo(plain(ell), distribution=stats.johnsonsu, sim_length=1,
                                     n_sims=1).fit["regular"]
        ours = stats.johnsonsu.logpdf(ell, *fit["Parameters"].values()).sum()
        for a in (-1.0, 0.0, 1.0):
            for b in (0.7, 3.0):
                other = stats.johnsonsu.fit(ell, a, b, loc=np.median(ell), scale=0.002)
                assert ours >= stats.johnsonsu.logpdf(ell, *other).sum() - 1e-6

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            parametric_monte_carlo(plain(exact_sample(stats.norm, 0, 0.01)), criterion="HQC")

    def test_prints_scores_and_selection_per_class(self, intraday, capsys):
        parametric_monte_carlo(intraday[0], sim_length=1, n_sims=1, random_state=0)
        out = capsys.readouterr().out
        for cls in ("regular", "gap"):
            for name in ("norm", "t", "johnsonsu"):
                assert f"{cls} " in out and f"{name} BIC" in out
            assert f"{cls} Selected:" in out

    def test_classes_are_selected_independently(self, intraday):
        fit = parametric_monte_carlo(intraday[0], sim_length=1, n_sims=1, random_state=0).fit
        assert fit["gap"]["Distribution"] == "t"
        assert fit["regular"]["Distribution"] == "norm"


class TestFitReport:
    """MC5: per class the family, parameters, criterion and every candidate's score."""

    def test_scores_match_an_independent_fit_per_class(self, intraday):
        values, kinds, rho, ell = intraday
        sims = parametric_monte_carlo(values, sim_length=1, n_sims=1, random_state=0)
        history = _common.prepare_history(values, None, "XNYS", 1, "test")
        for cls in ("gap", "regular"):
            # Gaps are fitted as they are, regular bars per full bar (MC3)
            data = ell[kinds == "gap"] if cls == "gap" else history.unit_pool("regular")
            entry = sims.fit[cls]
            assert list(entry["Candidates"]) == sorted(entry["Candidates"], key=entry["Candidates"].get)
            for dist in (stats.norm, stats.t, stats.johnsonsu):
                # MC5 needs the maximum: never below scipy's default fit
                params = dist.fit(data)
                default = -2 * dist.logpdf(data, *params).sum() + len(params) * np.log(len(data))
                assert entry["Candidates"][dist.name] <= default + 1e-9 * abs(default), (cls, dist.name)
            # The reported score is BIC at the reported parameters
            chosen = getattr(stats, entry["Distribution"])
            params = list(entry["Parameters"].values())
            expected = -2 * chosen.logpdf(data, *params).sum() + len(params) * np.log(len(data))
            assert entry["BIC"] == pytest.approx(expected, rel=1e-12)
            assert entry["BIC"] == entry["Candidates"][entry["Distribution"]]

    def test_given_family_for_both_classes(self, intraday):
        fit = parametric_monte_carlo(intraday[0], distribution=stats.laplace, sim_length=1, n_sims=1).fit
        for cls in ("gap", "regular"):
            assert fit[cls]["Distribution"] == "laplace" and fit[cls]["Candidates"] is None
            assert list(fit[cls]["Parameters"]) == ["loc", "scale"]

    def test_dict_gives_a_family_per_class(self, intraday):
        fit = parametric_monte_carlo(intraday[0], distribution={"gap": stats.t, "regular": stats.norm},
                                     sim_length=1, n_sims=1).fit
        assert list(fit["gap"]["Parameters"]) == ["df", "loc", "scale"]
        assert list(fit["regular"]["Parameters"]) == ["loc", "scale"]

    def test_dict_none_selects_for_that_class(self, intraday):
        fit = parametric_monte_carlo(intraday[0], distribution={"gap": None, "regular": stats.norm},
                                     sim_length=1, n_sims=1).fit
        assert fit["gap"]["Candidates"] is not None and fit["regular"]["Candidates"] is None

    def test_dict_missing_a_class_raises(self, intraday):
        with pytest.raises(ValueError, match="no entry for the gap class"):
            parametric_monte_carlo(intraday[0], distribution={"regular": stats.norm}, sim_length=1, n_sims=1)

    def test_dict_unknown_key_raises(self, intraday):
        with pytest.raises(ValueError, match="'overnight'"):
            parametric_monte_carlo(intraday[0], distribution={"regular": stats.norm, "gap": stats.norm,
                                                              "overnight": stats.norm}, sim_length=1, n_sims=1)

    def test_daily_has_one_class(self):
        fit = parametric_monte_carlo(plain(exact_sample(stats.norm, 0, 0.01)), distribution=stats.norm,
                                     sim_length=1, n_sims=1).fit
        assert list(fit) == ["regular"]

    def test_return_fit_is_deprecated_and_ignored(self, intraday):
        a = parametric_monte_carlo(intraday[0], distribution=stats.norm, sim_length=2, n_sims=5, random_state=1)
        with pytest.warns(DeprecationWarning, match="return_fit"):
            b = parametric_monte_carlo(intraday[0], distribution=stats.norm, sim_length=2, n_sims=5,
                                       random_state=1, return_fit=True)
        np.testing.assert_array_equal(a.paths, b.paths)
        assert b.fit == a.fit


class TestDraws:
    """MC3, MC4: draws are log returns of the class's fit; short bars mapped."""

    def test_per_class_moments_match_the_normal_fit(self, intraday):
        # Normal fits: drawn ℓ of a class has the fitted loc and scale. Tolerance:
        # 5 standard errors, scale / sqrt(m) for the mean and scale / sqrt(2 m) for the sd
        values = intraday[0]
        sims = parametric_monte_carlo(values, distribution=stats.norm, sim_length=3, n_sims=4000, random_state=0)
        steps = log_steps(sims.paths)
        history = _common.prepare_history(values, None, "XNYS", 3, "test")
        for cls in ("gap", "regular"):
            loc, scale = sims.fit[cls]["Parameters"]["loc"], sims.fit[cls]["Parameters"]["scale"]
            cols = (sims.classes == cls) & (history.future_rho == 1)
            drawn = steps[:, cols].ravel()
            assert drawn.mean() == pytest.approx(loc, abs=5 * scale / np.sqrt(drawn.size))
            assert drawn.std() == pytest.approx(scale, abs=5 * scale / np.sqrt(2 * drawn.size))

    def test_short_bar_has_rho_drift_and_root_rho_scale(self, intraday):
        # MC3: ℓ = ρ m̂ + √ρ (ℓ' − m̂), ℓ' ~ N(loc, scale): mean ρ m̂ + √ρ (loc − m̂), sd √ρ scale
        values = intraday[0]
        sims = parametric_monte_carlo(values, distribution=stats.norm, sim_length=3, n_sims=20000, random_state=0)
        history = _common.prepare_history(values, None, "XNYS", 3, "test")
        short = np.flatnonzero(history.future_rho < 1)[0]
        rho = history.future_rho[short]
        loc, scale = sims.fit["regular"]["Parameters"]["loc"], sims.fit["regular"]["Parameters"]["scale"]
        drawn = log_steps(sims.paths)[:, short]
        sd = np.sqrt(rho) * scale
        assert drawn.mean() == pytest.approx(rho * history.m_hat + np.sqrt(rho) * (loc - history.m_hat),
                                             abs=5 * sd / np.sqrt(drawn.size))
        assert drawn.std() == pytest.approx(sd, abs=5 * sd / np.sqrt(2 * drawn.size))

    def test_fat_left_tail_paths_stay_positive(self):
        # MC4 (ISSUES: parametric negative paths): a Johnson SU with a heavy left
        # tail. Fitted to simple returns, as before 0.4.0, it puts about 200 of
        # 125,000 draws below -100%; fitted to log returns, every path stays positive
        ell = np.random.default_rng(0).permutation(exact_sample(stats.johnsonsu, 1.5, 0.8, 0.01, 0.01, n=2000))
        simple_fit = stats.johnsonsu.fit(np.expm1(ell))
        assert stats.johnsonsu.cdf(-1, *simple_fit) * 250 * 500 > 50
        sims = parametric_monte_carlo(plain(ell), distribution=stats.johnsonsu, sim_length=250, n_sims=500,
                                      random_state=0)
        assert np.isfinite(sims.paths).all() and (sims.paths > 0).all()
        # The fitted left tail is still there: some steps lose more than 90%
        assert (log_steps(sims.paths) < np.log(0.1)).any()

    def test_seeded_and_reproducible(self, intraday):
        a = parametric_monte_carlo(intraday[0], distribution=stats.norm, sim_length=2, n_sims=30, random_state=4)
        b = parametric_monte_carlo(intraday[0], distribution=stats.norm, sim_length=2, n_sims=30, random_state=4)
        np.testing.assert_array_equal(a.paths, b.paths)


class TestMinSample:
    """MC6 for the parametric fit."""

    def test_short_class_raises(self):
        index = hourly_marks(sessions("2025-03-03", 10))
        values = from_log_returns(index, np.random.default_rng(0).normal(0, 0.01, len(index) - 1))
        with pytest.raises(ValueError, match="The gap class has 9 returns"):
            parametric_monte_carlo(values, sim_length=1, n_sims=1)

    def test_fewer_returns_than_parameters_always_raises(self):
        # Johnson SU has 4 parameters; 3 returns can't fit it, whatever min_sample
        with pytest.raises(ValueError, match="fewer than the 4 parameters of johnsonsu"):
            parametric_monte_carlo(plain([0.01, -0.02, 0.005]), distribution=stats.johnsonsu, min_sample=0,
                                   sim_length=1, n_sims=1)

    def test_zero_disables_the_count(self):
        sims = parametric_monte_carlo(plain(exact_sample(stats.norm, 0, 0.01, n=10)), distribution=stats.norm,
                                      min_sample=0, sim_length=2, n_sims=3, random_state=0)
        assert sims.paths.shape == (3, 3)


class TestVerbose:
    """`verbose=True` (the default) prints the family selection or fit, False prints nothing."""

    def test_true_prints_each_class_selection(self, intraday, capsys):
        sims = parametric_monte_carlo(intraday[0], sim_length=1, n_sims=2, random_state=0, verbose=True)
        expected = []
        for cls in ("regular", "gap"):
            entry = sims.fit[cls]
            # Printed in CANDIDATES order, then the family selected
            for name in ("norm", "t", "johnsonsu"):
                expected.append(f"{cls} {name:>10} BIC: {entry['Candidates'][name]:,.2f}")
            # The family selected, with its fitted parameters by name
            shown = ", ".join(f"{name} {value:.3g}" for name, value in entry["Parameters"].items())
            expected.append(f"{cls} Selected: {entry['Distribution']}  ({shown})")
        out = capsys.readouterr().out.splitlines()
        assert out == expected
        # The gaps are t-shaped, so their line names df before loc and scale
        assert out[-1].startswith("gap Selected: t  (df ")

    def test_given_family_prints_one_fitted_line_per_class(self, intraday, capsys):
        fit = parametric_monte_carlo(intraday[0], distribution=stats.t, sim_length=1, n_sims=1,
                                     random_state=0).fit
        out = capsys.readouterr().out.splitlines()
        expected = []
        for cls in ("regular", "gap"):
            params = fit[cls]["Parameters"]
            assert list(params) == ["df", "loc", "scale"]
            expected.append(f"{cls} Fitted: t  (df {params['df']:.3g}, loc {params['loc']:.3g}, "
                            f"scale {params['scale']:.3g})")
        # No candidate scores and no selection, only the fit
        assert out == expected

    def test_daily_fitted_line_has_no_class_label(self, capsys):
        fit = parametric_monte_carlo(plain(exact_sample(stats.norm, 0.0005, 0.01)), distribution=stats.norm,
                                     sim_length=1, n_sims=1).fit["regular"]["Parameters"]
        assert capsys.readouterr().out.splitlines() == [
            f"Fitted: norm  (loc {fit['loc']:.3g}, scale {fit['scale']:.3g})"]

    def test_quiet_prints_no_fitted_line(self, intraday, capsys):
        parametric_monte_carlo(intraday[0], distribution=stats.t, sim_length=1, n_sims=1, verbose=False)
        assert capsys.readouterr().out == ""

    def test_default_is_verbose(self, intraday, capsys):
        parametric_monte_carlo(intraday[0], sim_length=1, n_sims=2, random_state=0)
        default = capsys.readouterr().out
        parametric_monte_carlo(intraday[0], sim_length=1, n_sims=2, random_state=0, verbose=True)
        assert default and capsys.readouterr().out == default

    def test_false_prints_nothing(self, intraday, capsys):
        loud = parametric_monte_carlo(intraday[0], sim_length=1, n_sims=2, random_state=0)
        capsys.readouterr()
        quiet = parametric_monte_carlo(intraday[0], sim_length=1, n_sims=2, random_state=0, verbose=False)
        assert capsys.readouterr().out == ""
        # Only the printing changes, not the fit or the draws
        assert quiet.fit == loud.fit
        np.testing.assert_array_equal(quiet.paths, loud.paths)

    def test_false_keeps_warnings(self, intraday, capsys):
        with pytest.warns(UserWarning, match="counts sessions"):
            parametric_monte_carlo(intraday[0], sim_length=1, n_sims=2, random_state=0, verbose=False)
        assert capsys.readouterr().out == ""
