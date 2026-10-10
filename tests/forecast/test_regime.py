"""The regime-switching simulator (forecast/regime.py; MC7, MC5, MC6), seeded.

One-regime fits have closed forms, so the wiring of classes, ρ and the gap
model into the EM is checked exactly; two-regime histories check the
selection, the start from the filtered regime and the per-cell min_sample.
"""
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mc_helpers import (
    daily_history,
    from_log_returns,
    hourly_marks,
    intraday_history,
    log_steps,
    mark_kinds,
    sessions,
)
from scipy import stats

from portfolio_forecast.forecast import _common, regime, regime_switching_monte_carlo

pytestmark = pytest.mark.filterwarnings("ignore:.*counts sessions:UserWarning")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def intraday():
    """150 hourly sessions, gaps N(0, 1%), full bars N(0, 0.3%), short bars at √ρ scale."""
    return intraday_history(150, seed=5)


def two_regime_daily(end_turbulent, seed=0):
    """Daily history alternating calm (0.5%) and turbulent (3%) blocks of 200, ending in the given one."""
    vols = np.tile(np.repeat([0.005, 0.03], 200), 3)
    if not end_turbulent:
        vols = vols[:-200]
    ell = np.random.default_rng(seed).normal(0, vols)
    return from_log_returns(np.arange(len(ell) + 1), ell).to_numpy()


def two_regime_intraday(seed=0):
    """120 hourly sessions in blocks of 30: calm (gaps 0.5%, bars 0.2%), then turbulent (gaps 2%, bars 0.8%)."""
    index = hourly_marks(sessions("2024-03-01", 120))
    kinds, rho = mark_kinds(index)
    day = np.cumsum(np.r_[0, kinds == "gap"])[1:]
    turbulent = (day // 30) % 2 == 1
    sd = np.where(kinds == "gap", np.where(turbulent, 0.02, 0.005), np.where(turbulent, 0.008, 0.002) * np.sqrt(rho))
    return from_log_returns(index, np.random.default_rng(seed).normal(0, sd))


def simulate(values, **kwargs):
    """Run the simulator with quick settings unless overridden."""
    kwargs = {"sim_length": 2, "n_sims": 4, "random_state": 0, "n_starts": 2, **kwargs}
    return regime_switching_monte_carlo(values, **kwargs)


class TestOneRegimeClosedForms:
    """S = 1: each emission is a weighted sample moment, so the fit is known exactly."""

    def test_regime_gap_model(self, intraday):
        values, kinds, rho = intraday
        ell = np.diff(np.log(values.to_numpy()))
        gap, reg = ell[kinds == "gap"], ell[kinds == "regular"]
        r = rho[kinds == "regular"]
        fit = simulate(values, n_regimes=1).fit
        row = fit["Regimes"].iloc[0]
        # (S2), (S3) with short bars (MC7): mean Σℓ/Σρ, variance Σ(ℓ − ρμ)²/ρ / n
        mu = reg.sum() / r.sum()
        assert row["Mean"] == pytest.approx(mu, rel=1e-9)
        assert row["Volatility"] == pytest.approx(np.sqrt(np.mean((reg - r * mu) ** 2 / r)), rel=1e-9)
        assert row["Gap Mean"] == pytest.approx(gap.mean(), rel=1e-9)
        assert row["Gap Volatility"] == pytest.approx(gap.std(), rel=1e-9)
        log_lik = (stats.norm.logpdf(reg, r * mu, np.sqrt(r) * row["Volatility"]).sum()
                   + stats.norm.logpdf(gap, gap.mean(), gap.std()).sum())
        assert fit["Log Likelihood"] == pytest.approx(log_lik, rel=1e-9)
        # MC7 "regime": k = (S-1) + S(S-1) + 4S = 4 at S = 1; n counts every mark
        assert fit["BIC"][1] == pytest.approx(-2 * log_lik + 4 * np.log(len(ell)), rel=1e-12)
        assert fit["Gap Model"] == "regime" and "Gap" not in fit

    def test_independent_gap_model(self, intraday):
        values, kinds, rho = intraday
        ell = np.diff(np.log(values.to_numpy()))
        gap, reg = ell[kinds == "gap"], ell[kinds == "regular"]
        r = rho[kinds == "regular"]
        fit = simulate(values, n_regimes=1, gap_model="independent").fit
        assert fit["Gap"]["Mean"] == pytest.approx(gap.mean(), rel=1e-12)
        assert fit["Gap"]["Volatility"] == pytest.approx(gap.std(), rel=1e-12)
        assert "Gap Mean" not in fit["Regimes"]
        mu = reg.sum() / r.sum()
        sd = np.sqrt(np.mean((reg - r * mu) ** 2 / r))
        log_lik = (stats.norm.logpdf(reg, r * mu, np.sqrt(r) * sd).sum()
                   + stats.norm.logpdf(gap, gap.mean(), gap.std()).sum())
        assert fit["Log Likelihood"] == pytest.approx(log_lik, rel=1e-9)
        # MC7 "independent": k = (S-1) + S(S-1) + 2S + 2 = 4 at S = 1, n = every mark
        assert fit["BIC"][1] == pytest.approx(-2 * log_lik + 4 * np.log(len(ell)), rel=1e-12)

    def test_independent_bootstrap_fits_no_gap_normal(self, intraday):
        # MC7: resampled gaps have no fitted normal, so k has no +2 and the
        # likelihood (and n) covers the regular marks only
        values, kinds, rho = intraday
        ell = np.diff(np.log(values.to_numpy()))
        reg, r = ell[kinds == "regular"], rho[kinds == "regular"]
        fit = simulate(values, n_regimes=1, gap_model="independent", bootstrap=True).fit
        assert "Gap" not in fit
        mu = reg.sum() / r.sum()
        sd = np.sqrt(np.mean((reg - r * mu) ** 2 / r))
        log_lik = stats.norm.logpdf(reg, r * mu, np.sqrt(r) * sd).sum()
        assert fit["Log Likelihood"] == pytest.approx(log_lik, rel=1e-9)
        # k = (S-1) + S(S-1) + 2S = 2 at S = 1, n = regular marks
        assert fit["BIC"][1] == pytest.approx(-2 * log_lik + 2 * np.log(len(reg)), rel=1e-12)

    @pytest.mark.parametrize("S, k_regime, k_independent", [(2, 11, 9), (3, 20, 16)])
    def test_parameter_counts(self, intraday, S, k_regime, k_independent):
        # MC7 table: (S-1) + S(S-1) + 4S, and (S-1) + S(S-1) + 2S + 2
        values = intraday[0]
        n = len(values) - 1
        for gap_model, k in (("regime", k_regime), ("independent", k_independent)):
            fit = simulate(values, n_regimes=S, gap_model=gap_model, min_sample=0).fit
            assert fit["BIC"][S] == pytest.approx(-2 * fit["Log Likelihood"] + k * np.log(n), rel=1e-12)

    def test_daily_history_is_one_class(self):
        fit = simulate(daily_history(300), n_regimes=1).fit
        assert list(fit["Regimes"].columns) == ["Mean", "Volatility", "Expected Duration", "Current Probability"]
        assert "Gap" not in fit


class TestSelection:
    """MC5, MC7: the regime count by criterion, among eligible counts."""

    def test_selects_two_regimes_by_bic(self):
        fit = simulate(two_regime_daily(True), n_starts=5).fit
        assert fit["Criterion"] == "BIC" and fit["Regime Count"] == 2
        assert fit["Regime Count"] == min(fit["BIC"], key=fit["BIC"].get)

    def test_aic_is_reported_under_its_name(self):
        fit = simulate(two_regime_daily(True), n_starts=3, criterion="aic").fit
        assert fit["Criterion"] == "AIC" and "BIC" not in fit
        S = fit["Regime Count"]
        k = (S - 1) + S * (S - 1) + 2 * S
        assert fit["AIC"][S] == pytest.approx(-2 * fit["Log Likelihood"] + 2 * k, rel=1e-12)

    def test_given_count_is_the_only_one_fitted(self):
        fit = simulate(two_regime_daily(True), n_regimes=2).fit
        assert list(fit["BIC"]) == [2]

    def test_unfittable_count_raises(self):
        with pytest.raises(ValueError, match="Could not fit 50 regimes"):
            simulate(daily_history(300), n_regimes=50, n_starts=1)

    @pytest.mark.parametrize("kwargs, match", [({"gap_model": "both"}, "gap_model"),
                                               ({"n_regimes": 0}, "n_regimes"),
                                               ({"kappa": -1.0}, "kappa"),
                                               ({"criterion": "HQC"}, "criterion")])
    def test_bad_settings_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            simulate(daily_history(300), **kwargs)


@pytest.fixture(scope="module")
def short():
    """40 sessions: 39 gaps, so two regimes can't each hold 30 gaps."""
    return intraday_history(40, seed=2)[0]


class TestMinSample:
    """MC6: per-regime gap occupancy under "regime", per class otherwise."""

    def test_two_regimes_ineligible_when_selecting(self, short, capsys):
        fit = simulate(short).fit
        out = capsys.readouterr().out
        assert "2 regime(s): not eligible" in out and "3 regime(s): not eligible" in out
        assert fit["Regime Count"] == 1 and list(fit["BIC"]) == [1]

    def test_given_count_raises_with_the_occupancies(self, short):
        with pytest.raises(ValueError, match=r"each regime needs at least min_sample=30 expected gap marks, but "
                                             r"the 2 regime\(s\) have [\d.]+, [\d.]+ expected gap marks;"):
            simulate(short, n_regimes=2)

    def test_counts_print_without_a_trailing_decimal(self, short):
        # One regime holds every gap: expected occupancy 39 exactly, shown as 39
        with pytest.raises(ValueError, match=r"1 regime\(s\) have 39 expected gap marks;"):
            simulate(short, n_regimes=1, min_sample=40)
        with pytest.raises(ValueError, match=r"1 regime\(s\) have 39 gap marks by most likely regime;"):
            simulate(short, n_regimes=1, min_sample=40, bootstrap=True)

    def test_bootstrap_uses_hard_counts(self, short):
        with pytest.raises(ValueError, match="gap marks by most likely regime"):
            simulate(short, n_regimes=2, bootstrap=True)

    def test_two_regimes_eligible_with_enough_gaps_each(self):
        # 120 sessions in blocks of 30 calm and 30 turbulent: about 60 gaps per regime
        sims = simulate(two_regime_intraday(), n_regimes=2, n_starts=5)
        assert sims.fit["Regime Count"] == 2
        gap_vols = sims.fit["Regimes"]["Gap Volatility"]
        assert gap_vols.iloc[0] < gap_vols.iloc[1]

    def test_independent_counts_each_class(self, short):
        # 39 gaps, 280 regular bars: min_sample = 40 fails on the gap class
        with pytest.raises(ValueError, match="The gap class has 39 returns, fewer than min_sample=40"):
            simulate(short, n_regimes=1, gap_model="independent", min_sample=40)

    def test_unsamplable_gap_class_always_raises(self, short):
        # Every gap marked irregular: the forecast has gap marks and no gap pool
        kinds, _ = mark_kinds(short.index)
        classes = np.where(kinds == "gap", "irregular", kinds)
        with pytest.raises(ValueError, match="no gap returns to draw from"):
            simulate(short, n_regimes=1, classes=classes, min_sample=0)

    def test_empty_regular_pool_under_bootstrap(self, intraday):
        # A fit that puts every mark in regime 0 leaves regime 1 nothing
        # regular to resample, which fails even with min_sample = 0
        history = _common.prepare_history(intraday[0], None, "XNYS", 2, "regime_switching_monte_carlo")
        data = regime._ChainData(history, "independent", True)
        gamma = np.zeros((len(data.ell), 2))
        gamma[:, 0] = 1.0
        fit = SimpleNamespace(pi=np.array([1.0, 0.0]), gamma=gamma)
        assert regime._cell_problem(fit, data, history, 0, True) == (
            "with bootstrap=True regime 1 of 2 has no regular marks to resample; use fewer regimes or "
            "bootstrap=False")


class TestOneClassMinSample:
    """MC6: with one class (daily, continuous, or intraday without gaps) the HMM
    uses the per-class rule of MC6, n >= min_sample, under either gap_model."""

    @pytest.mark.parametrize("gap_model", ["regime", "independent"])
    @pytest.mark.parametrize("calendar", ["XNYS", None], ids=["daily", "continuous"])
    def test_per_class_rule(self, gap_model, calendar):
        # 20 regular returns: below min_sample = 30 raises, naming the class and
        # its n; min_sample = 20 (n >= min_sample) and 0 pass
        values = daily_history(20)
        if calendar is None:
            # The same values on a 24/7 daily grid
            values = pd.Series(values.to_numpy(), index=pd.date_range("2025-01-01", periods=21, freq="D", tz="UTC"))
        with pytest.raises(ValueError, match="The regular class has 20 returns, fewer than min_sample=30"):
            simulate(values, n_regimes=1, gap_model=gap_model, calendar=calendar)
        for ok in (20, 0):
            sims = simulate(values, n_regimes=1, gap_model=gap_model, calendar=calendar, min_sample=ok)
            assert sims.fit["Regime Count"] == 1

    def test_applies_before_selection(self):
        # The rule is per class, not per regime count: every count is refused alike
        with pytest.raises(ValueError, match="regular class has 20 returns"):
            simulate(daily_history(20))


class TestSimulation:
    """MC7: start from the filtered regime, one step per mark, draws per (regime, class)."""

    @pytest.mark.parametrize("end_turbulent, vol", [(True, 0.03), (False, 0.005)])
    def test_starts_from_the_filtered_regime(self, end_turbulent, vol):
        # The history ends deep in one regime and regimes last ~200 bars, so the
        # first future draws come from that regime: sd within 10% of its volatility
        sims = simulate(two_regime_daily(end_turbulent), n_regimes=2, n_starts=5, sim_length=1, n_sims=4000)
        first = log_steps(sims.paths)[:, 0]
        assert first.std() == pytest.approx(vol, rel=0.1)

    def test_large_kappa_makes_regimes_persist(self):
        # MC7: Dir(1, .., 1 + κ, .., 1) rows; κ = 10^6 drives every P_ii above 0.99
        fit = simulate(two_regime_daily(True), n_regimes=2, kappa=1e6).fit
        assert (np.diag(fit["Transition Matrix"]) > 0.99).all() and fit["Kappa"] == 1e6

    def test_kappa_zero_is_the_default(self):
        a = simulate(two_regime_daily(True), n_regimes=2)
        b = simulate(two_regime_daily(True), n_regimes=2, kappa=0.0)
        np.testing.assert_array_equal(a.paths, b.paths)

    @pytest.mark.parametrize("gap_model", ["regime", "independent"])
    def test_one_regime_draw_moments(self, intraday, gap_model):
        # S = 1, parametric draws: a gap ~ N(gap mean, gap sd); a short bar of ratio ρ
        # ~ N(ρ μ, ρ σ²) (MC7, with μ_reg). Tolerance: 5 standard errors.
        values = intraday[0]
        sims = simulate(values, n_regimes=1, gap_model=gap_model, sim_length=2, n_sims=20000)
        history = _common.prepare_history(values, None, "XNYS", 2, "test")
        row = sims.fit["Regimes"].iloc[0]
        gap_mean, gap_sd = ((row["Gap Mean"], row["Gap Volatility"]) if gap_model == "regime"
                            else (sims.fit["Gap"]["Mean"], sims.fit["Gap"]["Volatility"]))
        steps = log_steps(sims.paths)
        g = steps[:, np.flatnonzero(sims.classes == "gap")[0]]
        assert g.mean() == pytest.approx(gap_mean, abs=5 * gap_sd / np.sqrt(g.size))
        assert g.std() == pytest.approx(gap_sd, abs=5 * gap_sd / np.sqrt(2 * g.size))
        j = np.flatnonzero(history.future_rho < 1)[0]
        rho = history.future_rho[j]
        sd = np.sqrt(rho) * row["Volatility"]
        assert steps[:, j].mean() == pytest.approx(rho * row["Mean"], abs=5 * sd / np.sqrt(g.size))
        assert steps[:, j].std() == pytest.approx(sd, abs=5 * sd / np.sqrt(2 * g.size))

    @pytest.mark.parametrize("gap_model", ["regime", "independent"])
    def test_bootstrap_draws_from_class_pools(self, intraday, gap_model):
        values, kinds, rho = intraday
        ell = np.diff(np.log(values.to_numpy()))
        sims = simulate(values, n_regimes=1, gap_model=gap_model, bootstrap=True, n_sims=500, min_sample=0)
        history = _common.prepare_history(values, None, "XNYS", 2, "test")
        steps = log_steps(sims.paths)
        gap_cols = sims.classes == "gap"
        full_cols = ~gap_cols & (history.future_rho == 1)
        # S = 1: the (regime, class) pools are the class pools; regular per full bar
        # normalized with the regime's own μ_reg (MC7)
        mu = sims.fit["Regimes"]["Mean"].iloc[0]
        regular_pool = _common.to_unit(ell[kinds == "regular"], rho[kinds == "regular"], mu)
        near = lambda x, pool: np.isclose(x.ravel()[:, None], pool[None, :], rtol=0, atol=1e-12).any(axis=1)  # noqa: E731
        assert near(steps[:, gap_cols], ell[kinds == "gap"]).all()
        assert near(steps[:, full_cols], regular_pool).all()

    def test_constant_gaps_resampled_under_independent(self):
        # MC7: with bootstrap=True every gap comes from the pool, so constant
        # gaps are fine and every simulated gap is that constant
        values, kinds, rho = intraday_history(60, seed=4)
        ell = np.diff(np.log(values.to_numpy()))
        constant = from_log_returns(values.index, np.where(kinds == "gap", 0.004, ell))
        sims = simulate(constant, n_regimes=1, gap_model="independent", bootstrap=True, sim_length=5, n_sims=20)
        gaps = log_steps(sims.paths)[:, sims.classes == "gap"]
        np.testing.assert_allclose(gaps, 0.004, rtol=0, atol=1e-12)

    def test_constant_gaps_raise_for_the_gap_normal(self):
        values, kinds, _ = intraday_history(60, seed=4)
        ell = np.diff(np.log(values.to_numpy()))
        constant = from_log_returns(values.index, np.where(kinds == "gap", 0.004, ell))
        with pytest.raises(ValueError, match="gap returns are constant.*bootstrap=True"):
            simulate(constant, n_regimes=1, gap_model="independent")

    def test_bootstrap_on_daily_history_draws_historical_returns(self):
        values = two_regime_daily(True)
        sims = simulate(values, n_regimes=2, bootstrap=True, n_sims=50, sim_length=10)
        history = np.diff(np.log(values))
        drawn = log_steps(sims.paths).ravel()
        assert np.isclose(drawn[:, None], history[None, :], rtol=0, atol=1e-12).any(axis=1).all()

    def test_return_fit_is_deprecated_and_ignored(self):
        values = daily_history(300)
        a = simulate(values, n_regimes=1)
        with pytest.warns(DeprecationWarning, match="return_fit"):
            b = simulate(values, n_regimes=1, return_fit=True)
        np.testing.assert_array_equal(a.paths, b.paths)
        assert b.fit["Regime Count"] == 1

    def test_seeded_and_reproducible(self, intraday):
        a = simulate(intraday[0], n_regimes=2, min_sample=0, n_sims=20)
        b = simulate(intraday[0], n_regimes=2, min_sample=0, n_sims=20)
        np.testing.assert_array_equal(a.paths, b.paths)


class TestDependency:
    """hmmlearn is a test-only reference, not a runtime dependency."""

    def test_hmmlearn_is_only_in_the_test_extra(self):
        # Plain-text read: tomllib needs Python 3.11 and the package supports 3.10
        text = (ROOT / "pyproject.toml").read_text()
        runtime = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M).group(1)
        test = re.search(r"^test = \[(.*?)\]", text, re.M).group(1)
        assert "hmmlearn" not in runtime
        assert '"hmmlearn"' in test

    def test_package_runs_without_hmmlearn(self):
        # Block hmmlearn at import, then import the package and run a regime simulation
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'hmmlearn':\n"
            "            raise ImportError('hmmlearn blocked')\n"
            "sys.meta_path.insert(0, Block())\n"
            "import numpy as np, portfolio_forecast, portfolio_forecast.forecast as f\n"
            "x = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200)))\n"
            "s = f.regime_switching_monte_carlo(x, n_regimes=1, n_starts=1, sim_length=3, n_sims=2, random_state=0)\n"
            "print(s.paths.shape, 'hmmlearn' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip().splitlines()[-1] == "(2, 4) False"


class TestVerbose:
    """`verbose=True` (the default) prints the scores and regimes, False prints nothing."""

    @staticmethod
    def regime_lines(fit):
        """The lines printed for each regime of a daily history, built from the fit dict."""
        return [f"Regime {k}: mean {row.Mean:+.4%}, vol {row.Volatility:.4%}, "
                f"expected duration {row._3:.1f} marks"
                for k, row in enumerate(fit["Regimes"][["Mean", "Volatility", "Expected Duration"]].itertuples())]

    def test_daily_line_has_one_unlabelled_pair(self, capsys):
        sims = simulate(daily_history(300, seed=1), n_regimes=2)
        lines = capsys.readouterr().out.splitlines()
        assert lines == self.regime_lines(sims.fit)
        # One class: no class label and no gap moments
        assert all(re.fullmatch(r"Regime \d: mean [+-]\d+\.\d{4}%, vol \d+\.\d{4}%, "
                                r"expected duration \d+\.\d marks", line) for line in lines)

    def test_intraday_regime_line_shows_regular_and_gap_moments(self, capsys):
        sims = simulate(two_regime_intraday(), n_regimes=2, min_sample=0)
        regimes = sims.fit["Regimes"]
        expected = [f"Regime {k}: regular mean {row.Mean:+.4%}, vol {row.Volatility:.4%} | "
                    f"gap mean {row._3:+.4%}, vol {row._4:.4%} | expected duration {row._5:.1f} marks"
                    for k, row in enumerate(regimes[["Mean", "Volatility", "Gap Mean", "Gap Volatility",
                                                     "Expected Duration"]].itertuples())]
        assert capsys.readouterr().out.splitlines() == expected

    def test_intraday_independent_line_labels_the_regular_pair(self, capsys):
        sims = simulate(two_regime_intraday(), n_regimes=2, gap_model="independent")
        regimes = sims.fit["Regimes"]
        # Gaps are off the chain, so the regime has no gap moments to show
        assert "Gap Mean" not in regimes
        expected = [f"Regime {k}: regular mean {row.Mean:+.4%}, vol {row.Volatility:.4%}, "
                    f"expected duration {row._3:.1f} marks"
                    for k, row in enumerate(regimes[["Mean", "Volatility", "Expected Duration"]].itertuples())]
        assert capsys.readouterr().out.splitlines() == expected

    def test_true_prints_selection_and_regimes(self, capsys):
        values = daily_history(300, seed=1)
        sims = simulate(values, min_sample=0, verbose=True)
        out = capsys.readouterr().out.splitlines()
        scores = sims.fit["BIC"]
        S = sims.fit["Regime Count"]
        selection = [line for line in out if not line.startswith("Regime ")]
        # One line per regime count fitted: its score, or why it was skipped
        assert len(selection) == 4 and selection[-1] == f"Selected: {S} regime(s)"
        for count, score in scores.items():
            assert f"{count} regime(s) BIC: {score:,.2f}" in selection
        assert out[len(selection):] == self.regime_lines(sims.fit)

    def test_true_with_given_count_prints_regimes_only(self, capsys):
        sims = simulate(daily_history(300, seed=1), n_regimes=2, verbose=True)
        assert capsys.readouterr().out.splitlines() == self.regime_lines(sims.fit)

    def test_true_prints_ineligible_counts(self, capsys):
        # 40 sessions hold 39 gaps: 2 and 3 regimes can't each see 30 of them
        values = intraday_history(40, seed=2)[0]
        simulate(values, verbose=True)
        out = capsys.readouterr().out
        assert "3 regime(s): not eligible, under gap_model='regime' each regime needs at least min_sample" in out
        assert "Selected: 1 regime(s)" in out

    def test_default_is_verbose(self, intraday, capsys):
        simulate(intraday[0])
        default = capsys.readouterr().out
        simulate(intraday[0], verbose=True)
        assert default and capsys.readouterr().out == default

    def test_false_prints_nothing(self, intraday, capsys):
        loud = simulate(intraday[0], min_sample=0)
        capsys.readouterr()
        quiet = simulate(intraday[0], min_sample=0, verbose=False)
        assert capsys.readouterr().out == ""
        # Only the printing changes, not the fit or the draws
        assert quiet.fit["BIC"] == loud.fit["BIC"]
        pd.testing.assert_frame_equal(quiet.fit["Regimes"], loud.fit["Regimes"])
        np.testing.assert_array_equal(quiet.paths, loud.paths)

    def test_false_silences_skipped_counts_and_keeps_warnings(self, capsys):
        values = intraday_history(40, seed=2)[0]
        with pytest.warns(UserWarning, match="counts sessions"):
            sims = simulate(values, verbose=False)
        assert sims.fit["Regime Count"] == 1
        assert capsys.readouterr().out == ""
