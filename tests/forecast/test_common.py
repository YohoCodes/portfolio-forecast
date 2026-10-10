"""The shared simulation path (forecast/_common.py; MC1, MC2, MC3, MC4, MC6).

Each step of `prepare_history` and the path assembly, on histories whose log
returns, classes and length ratios are known by construction.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from mc_helpers import (
    NY,
    daily_history,
    from_log_returns,
    hourly_marks,
    intraday_history,
    mark_kinds,
    sessions,
)

from portfolio_forecast import BacktestResult, SimulationResult
from portfolio_forecast.forecast import (
    _common,
    nonparametric_monte_carlo,
    parametric_monte_carlo,
    regime_switching_monte_carlo,
)
from portfolio_forecast.utils import next_trading_dates

SIMULATORS = [
    pytest.param(nonparametric_monte_carlo, {}, id="bootstrap"),
    pytest.param(parametric_monte_carlo, {"distribution": __import__("scipy").stats.norm}, id="parametric"),
    pytest.param(regime_switching_monte_carlo, {"n_regimes": 1, "n_starts": 1}, id="regime"),
]


def prepare(values, classes=None, calendar="XNYS", sim_length=3):
    """Run prepare_history, keeping the intraday sim_length warning out of the way."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*counts sessions", category=UserWarning)
        return _common.prepare_history(values, classes, calendar, sim_length, "test")


class TestInputResolution:
    """MC1: a Series, a BacktestResult or a plain array."""

    def test_log_returns_of_a_daily_series(self):
        values = daily_history(50)
        history = prepare(values)
        np.testing.assert_array_equal(history.ell, np.diff(np.log(values.to_numpy())))
        assert history.mode == "daily"
        assert set(history.classes) == {"regular"}
        np.testing.assert_array_equal(history.rho, np.ones(50))

    def test_single_column_frame_is_its_column(self):
        values = daily_history(40)
        a, b = prepare(values), prepare(values.to_frame("X"))
        np.testing.assert_array_equal(a.ell, b.ell)

    def test_two_columns_raise(self):
        values = daily_history(40).to_frame("X").assign(Y=1.0)
        with pytest.raises(ValueError, match="2 columns"):
            prepare(values)

    @pytest.mark.parametrize("make", [np.asarray, list, lambda v: pd.Series(v)], ids=["array", "list", "range"])
    def test_plain_array_has_one_class_and_no_layout(self, make):
        values = make(daily_history(40).to_numpy())
        history = prepare(values, sim_length=7)
        assert history.layout is None and history.mode == "array"
        assert list(history.future_classes) == ["regular"] * 7
        np.testing.assert_array_equal(history.future_rho, np.ones(7))

    def test_classes_with_a_plain_array_raise(self):
        with pytest.raises(TypeError, match="labelled values"):
            prepare(np.arange(1.0, 50.0), classes=["regular"] * 48)

    @pytest.mark.parametrize("bad, match", [(np.nan, "non-finite"), (0.0, "at or below zero"),
                                            (-1.0, "at or below zero")])
    def test_values_must_be_finite_and_positive(self, bad, match):
        values = daily_history(40)
        values.iloc[10] = bad
        with pytest.raises(ValueError, match=match):
            prepare(values)

    def test_one_value_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            prepare(daily_history(1).iloc[:1])


class TestClasses:
    """MC1: classes given, from a BacktestResult, or from classify_returns."""

    def test_inferred_from_open_marks(self):
        values, kinds, _ = intraday_history(4)
        history = prepare(values)
        np.testing.assert_array_equal(history.classes, kinds)
        assert history.mode == "intraday"

    def test_from_a_backtest_result_with_irregular_dropped(self):
        # B1.6, B4.1: irregular returns are kept in values but in no pool
        values, kinds, _ = intraday_history(4)
        classes = pd.Series(kinds, index=values.index[1:])
        classes.iloc[[3, 11]] = "irregular"
        result = BacktestResult(values, values.index, classes)
        history = prepare(result)
        keep = np.ones(len(kinds), dtype=bool)
        keep[[3, 11]] = False
        np.testing.assert_array_equal(history.ell, np.diff(np.log(values.to_numpy()))[keep])
        np.testing.assert_array_equal(history.classes, kinds[keep])

    def test_given_classes_override_the_result(self):
        values, kinds, _ = intraday_history(3)
        result = BacktestResult(values, values.index, pd.Series(kinds, index=values.index[1:]))
        given = kinds.copy()
        given[0] = "irregular"
        assert len(prepare(result, classes=given).ell) == len(kinds) - 1

    def test_classes_series_must_cover_every_return(self):
        values, kinds, _ = intraday_history(3)
        classes = pd.Series(kinds, index=values.index[1:]).iloc[1:]
        with pytest.raises(ValueError, match="no entry for 1 return"):
            prepare(values, classes=classes)

    def test_classes_array_length_and_values(self):
        values = daily_history(10)
        with pytest.raises(ValueError, match="9 entries but values has 10 returns"):
            prepare(values, classes=["regular"] * 9)
        with pytest.raises(ValueError, match="'stale'"):
            prepare(values, classes=["regular"] * 9 + ["stale"])

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_midday_marks_raise_naming_to_session_values(self, simulate, kwargs):
        # MC1: one mark per session, not at the close, is neither daily nor intraday
        days = sessions("2025-03-03", 60)
        index = pd.DatetimeIndex([pd.Timestamp(f"{d:%Y-%m-%d} 12:00", tz=NY) for d in days])
        values = from_log_returns(index, np.full(59, 1e-3))
        with pytest.raises(ValueError, match="to_session_values"):
            simulate(values, sim_length=2, n_sims=2, **kwargs)

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_midday_marks_raise_with_classes_given(self, simulate, kwargs):
        days = sessions("2025-03-03", 60)
        index = pd.DatetimeIndex([pd.Timestamp(f"{d:%Y-%m-%d} 12:00", tz=NY) for d in days])
        values = from_log_returns(index, np.full(59, 1e-3))
        with pytest.raises(ValueError, match="to_session_values"):
            simulate(values, sim_length=2, n_sims=2, classes=["regular"] * 59, **kwargs)

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_intraday_without_classes_or_open_marks_raises(self, simulate, kwargs):
        # MC1: bar closes only, so the gaps can't be told apart
        values, _, _ = intraday_history(10)
        closes_only = values[[(t.hour, t.minute) != (9, 30) for t in values.index]]
        with pytest.raises(ValueError, match="no open marks"):
            simulate(closes_only, sim_length=2, n_sims=2, **kwargs)


class TestShortBars:
    """U2, MC3: ρ = τ/Δ per return, and the location–scale maps."""

    def test_history_rho(self):
        # Hourly marks: 9:30-10:00 is half a bar; gaps and full bars are 1
        values, kinds, rho = intraday_history(3)
        history = prepare(values)
        np.testing.assert_array_equal(history.rho, rho)
        assert set(history.rho[history.classes == "gap"]) == {1.0}
        assert np.count_nonzero(history.rho == 0.5) == 3

    def test_m_hat_is_regular_log_sum_over_rho_sum(self):
        # MC3: m̂ = Σ ℓ_k / Σ ρ_k over regular returns, gaps left out
        values, kinds, rho = intraday_history(5)
        ell = np.diff(np.log(values.to_numpy()))
        regular = kinds == "regular"
        assert prepare(values).m_hat == pytest.approx(ell[regular].sum() / rho[regular].sum(), rel=1e-12)

    def test_round_trip_is_exact(self):
        # MC3: ℓ → ℓ' → ℓ is the identity; full bars are passed through bit for bit
        rng = np.random.default_rng(0)
        ell = rng.normal(1e-3, 0.01, 1000)
        rho = np.where(rng.random(1000) < 0.3, rng.uniform(0.05, 1.0, 1000), 1.0)
        back = _common.from_unit(_common.to_unit(ell, rho, 4e-4), rho, 4e-4)
        np.testing.assert_allclose(back, ell, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(back[rho == 1], ell[rho == 1])
        np.testing.assert_array_equal(_common.to_unit(ell, np.ones(1000), 4e-4), ell)

    def test_maps_by_hand(self):
        # m = 0.001, ρ = 0.25, ℓ = 0.003: ℓ' = 0.001 + (0.003 - 0.00025) / 0.5 = 0.0065;
        # back: 0.25 * 0.001 + 0.5 * (0.0065 - 0.001) = 0.003
        assert _common.to_unit(0.003, 0.25, 0.001) == pytest.approx(0.0065, rel=1e-14)
        assert _common.from_unit(0.0065, 0.25, 0.001) == pytest.approx(0.003, rel=1e-14)

    def test_pure_drift_short_bar_maps_to_rho_m_hat(self):
        # Every regular return is pure drift, ℓ = ρ m: then m̂ = m, every ℓ' = m,
        # and a short future bar gets ρ m̂
        m = 2e-3
        index = hourly_marks(sessions("2025-03-03", 40))
        kinds, rho = mark_kinds(index)
        ell = np.where(kinds == "gap", 0.01, rho * m)
        history = prepare(from_log_returns(index, ell))
        assert history.m_hat == pytest.approx(m, rel=1e-12)
        np.testing.assert_allclose(history.unit_pool("regular"), m, rtol=1e-12)
        with pytest.warns(UserWarning, match="counts sessions"):
            sims = nonparametric_monte_carlo(from_log_returns(index, ell), sim_length=2, n_sims=3, random_state=0)
        steps = np.diff(np.log(sims.paths), axis=1)
        layout_kinds, layout_rho = sims.classes, next_trading_dates(from_log_returns(index, ell), 2).tau
        expected = np.where(layout_kinds == "gap", 0.01, layout_rho * m)
        np.testing.assert_allclose(steps, np.broadcast_to(expected, steps.shape), rtol=1e-9)
        assert 0.5 in layout_rho


class TestMinSample:
    """MC6: per class, n below min_sample raises, naming the class, n and the history needed."""

    def test_message_names_class_count_and_history_needed(self):
        # 6 sessions: 5 gaps; 30 needs 25 more gaps, one per session
        values, _, _ = intraday_history(6)
        history = prepare(values)
        with pytest.raises(ValueError, match=r"gap class has 5 returns, fewer than min_sample=30; "
                                             r"about 25 more sessions of history"):
            _common.check_min_sample(history, 30)

    @pytest.mark.parametrize("start, n_gaps, more", [("open", 9, 21), ("close", 10, 20)])
    def test_gap_hint_counts_every_session_boundary(self, start, n_gaps, more):
        # 10 sessions from an open hold 9 gaps; from the previous session's
        # close, 10. Either way one gap per session boundary, so 30 needs
        # 30 - n more sessions
        values, _, _ = intraday_history(11)
        first_close = int(np.argmax(values.index.hour == 16))
        values = values.iloc[:-8] if start == "open" else values.iloc[first_close:]
        history = prepare(values)
        assert history.count("gap") == n_gaps
        with pytest.raises(ValueError, match=rf"gap class has {n_gaps} returns, fewer than min_sample=30; "
                                             rf"about {more} more sessions of history are needed "
                                             r"\(the history spans 10 sessions\)"):
            _common.check_min_sample(history, 30)

    def test_regular_class_is_counted_in_bars_per_session(self):
        # Daily: 20 regular returns over 20 bars; 30 needs 10 more bars
        history = prepare(daily_history(20))
        with pytest.raises(ValueError, match=r"regular class has 20 returns.*about 10 more bars"):
            _common.check_min_sample(history, 30)

    def test_zero_disables_the_count(self):
        values, _, _ = intraday_history(6)
        _common.check_min_sample(prepare(values), 0)

    def test_unsamplable_class_always_raises(self):
        # Every gap irregular: the forecast has gap marks but no gap pool
        values, kinds, _ = intraday_history(6)
        classes = np.where(kinds == "gap", "irregular", kinds)
        with pytest.raises(ValueError, match="no gap returns to draw from"):
            _common.check_min_sample(prepare(values, classes=classes), 0)

    @pytest.mark.parametrize("bad", [-1, 2.5, True])
    def test_bad_min_sample_raises(self, bad):
        with pytest.raises(ValueError, match="min_sample"):
            _common.check_min_sample(prepare(daily_history(40)), bad)


class TestFutureLayout:
    """MC2, MC8: the future marks are next_trading_dates', with their classes."""

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    @pytest.mark.parametrize("cut", [None, -3], ids=["ends at close", "ends mid-session"])
    def test_dates_and_classes_from_next_trading_dates(self, simulate, kwargs, cut):
        values, _, _ = intraday_history(40)
        values = values.iloc[:cut]
        layout = next_trading_dates(values, 3)
        with pytest.warns(UserWarning, match="counts sessions"):
            sims = simulate(values, sim_length=3, n_sims=4, random_state=0, min_sample=5, **kwargs)
        assert isinstance(sims, SimulationResult)
        assert sims.dates.equals(layout.dates)
        np.testing.assert_array_equal(sims.classes, layout.classes)
        assert sims.paths.shape == (4, len(layout.dates) + 1)

    def test_mid_session_history_counts_its_session_first(self):
        # MC2: history ends at 14:00 on day 40, so 2 sessions end at day 41's close
        values, _, _ = intraday_history(41)
        values = values.iloc[:-2]
        history = prepare(values, sim_length=2)
        assert history.layout.dates[-1] == pd.Timestamp(f"{sessions('2025-03-03', 42)[-1]:%Y-%m-%d} 16:00", tz=NY)
        assert list(history.future_classes[:3]) == ["regular", "regular", "gap"]

    def test_future_rho_is_the_layout_tau(self):
        values, _, _ = intraday_history(40)
        history = prepare(values, sim_length=2)
        np.testing.assert_array_equal(history.future_rho, history.layout.tau)
        assert 0.5 in history.future_rho

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_one_intraday_warning(self, simulate, kwargs):
        values, _, _ = intraday_history(40)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            simulate(values, sim_length=2, n_sims=2, random_state=0, **kwargs)
        user = [w for w in record if issubclass(w.category, UserWarning)]
        assert len(user) == 1
        assert "sim_length counts sessions" in str(user[0].message)
        assert user[0].filename == __file__

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_no_warning_on_daily_input(self, simulate, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sims = simulate(daily_history(60), sim_length=5, n_sims=2, random_state=0, **kwargs)
        assert sims.dates.equals(next_trading_dates(daily_history(60), 5).dates)

    @pytest.mark.parametrize("simulate, kwargs", SIMULATORS)
    def test_plain_array_gives_no_dates(self, simulate, kwargs):
        sims = simulate(daily_history(60).to_numpy(), sim_length=5, n_sims=2, random_state=0, **kwargs)
        assert sims.dates is None and sims.classes is None
        assert sims.paths.shape == (2, 6)


class TestAssemble:
    """MC4: V_t = exp(Σ ℓ), column 0 = 1.0."""

    def test_paths_compound_log_returns(self):
        history = prepare(np.arange(1.0, 50.0), sim_length=3)
        ell = np.array([[0.01, -0.02, 0.005], [0.0, 0.0, np.log(2)]])
        sims = _common.assemble(ell, history, None)
        np.testing.assert_allclose(sims.paths, [[1, np.exp(0.01), np.exp(-0.01), np.exp(-0.005)],
                                                [1, 1, 1, 2]], rtol=1e-14)
        assert (sims.paths[:, 0] == 1.0).all()
        assert sims.fit is None and sims.dates is None

    def test_zero_horizon(self):
        sims = nonparametric_monte_carlo(daily_history(60), sim_length=0, n_sims=3, random_state=0)
        assert sims.paths.shape == (3, 1) and len(sims.dates) == 0


class TestCriterion:
    @pytest.mark.parametrize("name", ["bic", "BIC", "Bic"])
    def test_case_insensitive(self, name):
        assert _common.normalize_criterion(name) == "BIC"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="AIC"):
            _common.normalize_criterion("HQC")

    def test_values(self):
        assert _common.information_criterion(100.0, 3, 1000, "AIC") == -194.0
        assert _common.information_criterion(100.0, 3, 1000, "BIC") == pytest.approx(-200 + 3 * np.log(1000))
