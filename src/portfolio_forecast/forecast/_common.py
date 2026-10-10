"""Shared simulation path of the three Monte Carlo simulators.

Every simulator turns a history into log returns per class, draws one log
return per future mark from its class's model, and compounds them. This
module holds the steps they share; the simulators (`bootstrap`,
`parametric`, `regime`) only differ in how a mark's log return is drawn.

Flow
----
1. `resolve_values` and `warn_return_fit` handle the deprecated keywords.
2. `prepare_history` resolves the input (a Series, a `BacktestResult` or a
   plain array), takes or infers the classes, computes
   ℓ = ln(1 + r), drops irregular returns, measures each kept return's
   length ratio ρ = τ/Δ, estimates the regular drift m̂ and lays out
   the future marks with `next_trading_dates`. It returns a
   `History`.
3. `check_min_sample` applies the minimum sample to the classes the simulation uses.
4. The simulator draws ℓ' per full unit and maps short future bars back
   with `from_unit`; `to_unit` is the inverse map for the history.
5. `assemble` compounds the draws into paths, V_t = V_0 exp(ℓ_1 + … + ℓ_t)
   with V_0 = 1.0 in column 0, and wraps them in a `SimulationResult`.

Conventions
-----------
- Returns are log returns per mark, ℓ = ln(V_t / V_{t-1}), never simple
  returns; paths are growth multiples of the last historical value.
- ρ = τ/Δ is a return's length over the bar size: 1.0 for full bars, every
  gap (one unit whatever its clock length), daily and longer bars and
  continuous mode; below 1 only for a short intraday bar, such as the
  last bar of a half-day.
- Short-bar location–scale maps, with m̂ = Σ_t ℓ_t / Σ_t ρ_t over regular returns:
  history ℓ' = m̂ + (ℓ − ρ m̂)/√ρ, future ℓ = ρ m̂ + √ρ (ℓ' − m̂). Drift
  scales with ρ and volatility with √ρ. Full bars (ρ = 1) are passed
  through untouched, so they are exact.
"""
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..types import BacktestResult, FutureLayout, SimulationResult
from ..utils import classify_returns, next_trading_dates
from ..utils.marks import _intraday_step, _mark_sessions, _session_values, is_daily

GAP, REGULAR, IRREGULAR = "gap", "regular", "irregular"
CLASSES = (REGULAR, GAP)
CRITERIA = ("AIC", "BIC")

# The kind of history: sets whether the horizon counts sessions or bars,
# whether short bars exist (intraday only) and the unit of the minimum-sample message
INTRADAY, DAILY, CONTINUOUS, ARRAY = "intraday", "daily", "continuous", "array"


@dataclass(frozen=True)
class History:
    """A history prepared for simulation, and the future marks to simulate.

    Attributes
    ----------
    ell : numpy.ndarray
        Log returns ℓ = ln(1 + r) of the history, oldest first, irregular
        returns left out.
    classes : numpy.ndarray of str
        ``"gap"`` or ``"regular"`` for each entry of `ell`.
    rho : numpy.ndarray
        Length ratio ρ = τ/Δ of each entry of `ell` (1.0 except short
        intraday bars).
    m_hat : float
        Regular drift per full bar, Σ_t ℓ_t / Σ_t ρ_t over the regular
        returns; 0.0 if there are none.
    layout : FutureLayout or None
        The future marks from `next_trading_dates`; None for plain arrays.
    future_classes : numpy.ndarray of str
        Class of the return ending at each future mark.
    future_rho : numpy.ndarray
        ρ of the return ending at each future mark.
    mode : str
        ``"intraday"``, ``"daily"``, ``"continuous"`` or ``"array"``.
    units : int
        Sessions (intraday) or bars (otherwise) the history spans, for the
        minimum-sample message.
    gap_spans : int
        Session boundaries the history crosses, the gaps it could hold
        (intraday; 0 otherwise), for the minimum-sample message.
    """

    ell: np.ndarray
    classes: np.ndarray
    rho: np.ndarray
    m_hat: float
    layout: FutureLayout | None
    future_classes: np.ndarray
    future_rho: np.ndarray
    mode: str
    units: int
    gap_spans: int

    @property
    def n_marks(self):
        """Number of future marks simulated."""
        return len(self.future_classes)

    @property
    def model_classes(self):
        """Return the classes the simulation models, regular first.

        These are the classes of the future marks and any class the history
        has returns of.
        """
        present = set(self.future_classes) | set(self.classes)
        return [c for c in CLASSES if c in present]

    def count(self, cls):
        """Return the number of kept history returns of class `cls`."""
        return int(np.count_nonzero(self.classes == cls))

    def unit_pool(self, cls):
        """Return the history's log returns of class `cls` per full unit.

        Gaps are one unit each whatever their clock length, so they are
        returned as they are; regular returns go through `to_unit` with m̂.
        """
        mask = self.classes == cls
        if cls == GAP:
            return self.ell[mask]
        return to_unit(self.ell[mask], self.rho[mask], self.m_hat)


def resolve_values(values, prices, func_name):
    """Return `values`, accepting the deprecated `prices` keyword in its place until 1.0.0.

    Call it directly from the public function, so the warning points at the
    caller's line. Raises TypeError if neither or both are given.
    """
    if prices is None:
        if values is None:
            raise TypeError(f"{func_name}() missing 1 required argument: 'values'")
        return values
    if values is not None:
        raise TypeError(f"{func_name}() got both 'values' and 'prices'; pass only 'values'")
    warnings.warn(f"{func_name}(prices=...) is deprecated and will be removed in 1.0.0; "
                  "use values=... instead", DeprecationWarning, stacklevel=3)
    return prices


def warn_return_fit(return_fit, func_name):
    """Warn that `return_fit` is deprecated and ignored when it is set.

    The fit is always returned in ``SimulationResult.fit``, so the flag does
    nothing. Call it directly from the public function, so the warning
    points at the caller.
    """
    if return_fit is not False:
        warnings.warn(f"{func_name}(return_fit=...) is deprecated, ignored and will be removed in 1.0.0: "
                      "the result is always a SimulationResult with the fit in .fit",
                      DeprecationWarning, stacklevel=3)


def normalize_criterion(criterion):
    """Return "AIC" or "BIC" for a case-insensitive `criterion`; raise ValueError otherwise."""
    name = str(criterion).upper()
    if name not in CRITERIA:
        raise ValueError(f'criterion must be "AIC" or "BIC" (any case), got {criterion!r}')
    return name


def information_criterion(log_lik, k, n, criterion):
    """Return AIC = −2 ln L + 2k or BIC = −2 ln L + k ln n; lower is better.

    Akaike (1974), Schwarz (1978). `log_lik` is the maximized log L, `k`
    counts the free parameters fitted (a family's loc and scale included)
    and `n` the returns fitted. `criterion` is already normalized.
    """
    penalty = math.log(n) if criterion == "BIC" else 2.0
    return -2.0 * float(log_lik) + k * penalty


def to_unit(ell, rho, m):
    """Return history log returns per full unit, ℓ' = m + (ℓ − ρ m)/√ρ.

    `m` may be a scalar or broadcast against `ell` (per-regime μ_s,reg).
    Full bars (ρ = 1) are returned exactly as given.
    """
    ell, rho = np.asarray(ell, dtype=float), np.asarray(rho, dtype=float)
    return np.where(rho == 1.0, ell, m + (ell - rho * m) / np.sqrt(rho))


def from_unit(ell_unit, rho, m):
    """Return log returns for bars of length ratio ρ from per-unit draws, ℓ = ρ m + √ρ (ℓ' − m).

    The inverse of `to_unit`. Full bars (ρ = 1) are returned exactly as given.
    """
    ell_unit, rho = np.asarray(ell_unit, dtype=float), np.asarray(rho, dtype=float)
    return np.where(rho == 1.0, ell_unit, rho * m + np.sqrt(rho) * (ell_unit - m))


def _value_input(values):
    """Return (series or None, array, classes from a BacktestResult or None).

    A Series on a DatetimeIndex is labelled input; anything else (a plain
    array, a list, a Series on a RangeIndex) is a plain array of values. A
    single-column DataFrame or ``(n, 1)`` array is read as its column; more
    columns or dimensions raise ValueError.
    """
    given = None
    # A BacktestResult brings its own classes
    if isinstance(values, BacktestResult):
        given = values.classes
        values = values.values
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"values has {values.shape[1]} columns; pass one value series")
        values = values.iloc[:, 0]
    # Labelled input: a Series on a DatetimeIndex
    if isinstance(values, pd.Series) and isinstance(values.index, pd.DatetimeIndex):
        return values, values.to_numpy(dtype=float), given
    # Anything else is a plain array, read as one column
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"values must be one value series, got an array of shape {array.shape}")
    return None, array, None


def _check_values(array):
    """Raise ValueError unless there are two or more finite, positive values.

    Log returns need V > 0. NaN is not filled, since a filled value
    would add a zero return and a two-bar return to the pools.
    """
    if len(array) < 2:
        raise ValueError(f"values needs at least two values to have a return, got {len(array)}")
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(f"values has {bad.sum()} missing or non-finite value(s); drop them, or pass classes "
                         "marking the returns across them irregular")
    if (array <= 0).any():
        raise ValueError(f"values has {(array <= 0).sum()} value(s) at or below zero; log returns need "
                         "positive values")


def _class_array(classes, index):
    """Return classes as an object array aligned to the returns of `index`.

    A Series must be indexed by the mark where each return ends,
    ``index[1:]``; an array-like must have one entry per return.
    """
    n = len(index) - 1
    if isinstance(classes, pd.Series):
        # Align on the mark where each return ends; a return with no entry raises
        aligned = classes.reindex(index[1:])
        if aligned.isna().any():
            raise ValueError(f"classes has no entry for {aligned.isna().sum()} return(s), first ending at "
                             f"{aligned.index[aligned.isna()][0]}; index classes by the mark where each "
                             "return ends (values.index[1:])")
        out = aligned.to_numpy(dtype=object)
    else:
        out = np.asarray(classes, dtype=object)
        if out.shape != (n,):
            raise ValueError(f"classes has {out.size} entries but values has {n} returns; give one per return")
    unknown = sorted(set(out) - {GAP, REGULAR, IRREGULAR})
    if unknown:
        raise ValueError(f"classes must be 'gap', 'regular' or 'irregular', got {unknown}")
    return out


def _history_rho(index, classes, mode, calendar):
    """Return ρ = τ/Δ of every history return.

    Only a short intraday regular bar has ρ < 1: its length over
    the bar size Δ learned from the marks, as for the future layout. A
    regular return as long as Δ or longer is left at 1, as for Bar Volatility.
    """
    rho = np.ones(len(index) - 1)
    # Only intraday history has short bars
    if mode != INTRADAY:
        return rho
    # Each return's clock length in bar sizes Δ
    step = _intraday_step(index, calendar)
    lengths = np.asarray((index[1:] - index[:-1]) / step, dtype=float)
    # Regular returns take it, capped at one full bar; gaps stay at 1
    regular = classes == REGULAR
    rho[regular] = np.minimum(lengths[regular], 1.0)
    return rho


def prepare_history(values, classes, calendar, sim_length, func_name):
    """Resolve a simulator's input into a `History`.

    Parameters
    ----------
    values : pandas.Series, single-column DataFrame, BacktestResult or array_like
        The value history, oldest first.
    classes : pandas.Series, array_like or None
        Class of each return; None takes a BacktestResult's, or infers them
        with `classify_returns`. Plain arrays have one class.
    calendar : str or None
        ``exchange_calendars`` code; None is continuous mode.
    sim_length : int
        Horizon: sessions on intraday history, bars otherwise.
    func_name : str
        The calling simulator, for messages.

    Returns
    -------
    History
        The kept log returns with their classes and ρ, m̂, and the future
        marks with their classes and ρ.

    Raises
    ------
    ValueError
        For a negative `sim_length`, fewer than two values, non-finite or
        non-positive values, more than one column, bad classes, a series
        that is neither daily nor intraday, and intraday labels with
        neither classes nor open marks.
    TypeError
        For classes given with a plain array, which has no labels to place
        gaps on.

    Warns
    -----
    UserWarning
        On intraday history: `sim_length` now counts sessions.
    """
    sim_length = int(sim_length)
    if sim_length < 0:
        raise ValueError(f"sim_length must be at least 0, got {sim_length}")
    # The labelled Series (None for a plain array), its values, and a BacktestResult's classes
    series, array, given = _value_input(values)
    _check_values(array)
    # ℓ = ln(V_t / V_{t-1}) of every return, irregular ones included for now
    ell_all = np.diff(np.log(array))

    if series is None:
        # A plain array is one class with no labels, so no layout
        if classes is not None:
            raise TypeError("classes needs labelled values (a Series on a DatetimeIndex or a BacktestResult); "
                            "a plain array is one class")
        cls_all = np.full(len(ell_all), REGULAR, dtype=object)
        rho_all = np.ones(len(ell_all))
        layout = None
        future_classes = np.full(sim_length, REGULAR, dtype=object)
        future_rho = np.ones(sim_length)
        mode, units = ARRAY, len(ell_all)
        gap_spans = 0
    else:
        index = series.index
        if classes is None:
            classes = given
        # Class per return, given or inferred; inferring raises for one mark per session
        # away from the close, and for intraday labels with no open marks
        cls_all = classify_returns(series, calendar).to_numpy(dtype=object) if classes is None \
            else _class_array(classes, index)
        # The future marks and their classes, from their one source; it also
        # raises for a series neither daily nor intraday when classes were given
        layout = next_trading_dates(series, sim_length, calendar=calendar)
        # The kind of history: continuous without a calendar, else daily or intraday by its marks
        if calendar is None:
            mode = CONTINUOUS
        else:
            mode = DAILY if is_daily(series, calendar) else INTRADAY
        # ρ = τ/Δ of every history return, below 1 only for a short intraday bar
        rho_all = _history_rho(index, cls_all, mode, calendar)
        future_classes = np.asarray(layout.classes, dtype=object)
        # Continuous mode has no short bars
        future_rho = np.ones(len(layout.tau)) if mode == CONTINUOUS else np.asarray(layout.tau, dtype=float)
        # Sessions (intraday) or bars the history spans, for the minimum-sample message
        units = len(_session_values(series, calendar)[0]) - 1 if mode == INTRADAY else len(ell_all)
        # A history touching d sessions holds at most d - 1 gaps, one more
        # than units - 1 when it starts at a session close
        gap_spans = _mark_sessions(index, calendar)["session"].nunique() - 1 if mode == INTRADAY else 0
        if mode == INTRADAY:
            warnings.warn(
                f"{func_name}: on intraday history sim_length counts sessions, not bars (changed in 0.4.0): "
                f"sim_length={sim_length} gives {len(layout.dates)} future marks, ending at the close of the "
                f"last of {sim_length} sessions. For a daily forecast, pass to_session_values(values)",
                UserWarning, stacklevel=3)

    # Irregular returns are in no pool and no fit
    keep = cls_all != IRREGULAR
    ell, cls, rho = ell_all[keep], cls_all[keep], rho_all[keep]
    regular = cls == REGULAR
    # Drift per full bar, so a short bar's share of it is ρ m̂
    m_hat = float(ell[regular].sum() / rho[regular].sum()) if regular.any() else 0.0
    return History(ell=ell, classes=cls, rho=rho, m_hat=m_hat, layout=layout, future_classes=future_classes,
                   future_rho=future_rho, mode=mode, units=max(units, 1), gap_spans=gap_spans)


def check_samplable(history):
    """Raise ValueError if a future mark's class has no history returns (always, whatever min_sample)."""
    for cls in CLASSES:
        if cls in set(history.future_classes) and history.count(cls) == 0:
            raise ValueError(f"The forecast has {cls} marks but the history has no {cls} returns to draw from, "
                             "whatever min_sample (irregular returns are left out); use a longer history or "
                             "pass classes")


def check_min_sample(history, min_sample):
    """Raise ValueError if a modelled class has fewer than `min_sample` returns.

    The message names the class, its n and the history needed.
    ``min_sample=0`` disables the count check, but a class the forecast
    needs and the history lacks always raises (`check_samplable`).
    """
    min_sample = check_min_sample_value(min_sample)
    # A class the forecast needs and the history lacks raises even with min_sample=0
    check_samplable(history)
    if min_sample == 0:
        return
    for cls in history.model_classes:
        n = history.count(cls)
        if n < min_sample:
            raise ValueError(min_sample_message(history, cls, n, min_sample))


def check_min_sample_value(min_sample):
    """Return `min_sample` as an int >= 0, or raise ValueError."""
    if isinstance(min_sample, bool) or int(min_sample) != min_sample or min_sample < 0:
        raise ValueError(f"min_sample must be an integer >= 0, got {min_sample!r}")
    return int(min_sample)


def min_sample_message(history, cls, n, min_sample):
    """Return the minimum-sample message: the class, its n and the history needed.

    The history needed is estimated from the class's rate per session (or
    bar) in the history.
    """
    unit = "sessions" if history.mode == INTRADAY else "bars"
    # The class's returns per session or bar so far; gaps can only occur at session boundaries
    spans = history.gap_spans if history.mode == INTRADAY and cls == GAP else history.units
    per_unit = max(history.count(cls), 1) / max(spans, 1)
    more = math.ceil((min_sample - n) / per_unit)
    return (f"The {cls} class has {n} returns, fewer than min_sample={min_sample}; about {more} more "
            f"{unit} of history are needed (the history spans {history.units} {unit}). Pass a longer history, "
            "or a lower min_sample (0 disables the check)")


def assemble(log_returns, history, fit):
    """Return the `SimulationResult` of per-mark log returns, shape ``(n_sims, n_marks)``.

    V_t = V_0 exp(ℓ_1 + … + ℓ_t), positive by construction, with
    V_0 = 1.0 in column 0, the last historical value. `fit` is stored as
    the result's ``fit`` (None for the bootstrap). A plain-array history
    gets no dates or classes.
    """
    n_sims = log_returns.shape[0]
    # Column 0 is V_0 = 1; each later column exponentiates the running sum of ℓ
    paths = np.ones((n_sims, history.n_marks + 1))
    paths[:, 1:] = np.exp(np.cumsum(log_returns, axis=1))
    # A plain array has no dates or classes to attach
    if history.layout is None:
        return SimulationResult(paths=paths, dates=None, classes=None, fit=fit)
    return SimulationResult(paths=paths, dates=history.layout.dates,
                            classes=np.asarray(history.layout.classes, dtype=object), fit=fit)


def check_n_sims(n_sims):
    """Return `n_sims` as an int >= 1, or raise ValueError."""
    if int(n_sims) != n_sims or n_sims < 1:
        raise ValueError(f"n_sims must be an integer >= 1, got {n_sims!r}")
    return int(n_sims)
