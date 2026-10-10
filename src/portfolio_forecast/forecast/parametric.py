"""`parametric_monte_carlo`: draw log returns from a distribution fitted per class.

Flow
----
1. `_common.prepare_history` turns the input into log returns per class,
   with irregular returns left out, and lays out the future marks.
2. `_common.check_min_sample` applies the minimum sample to each class; `_fit_class` also
   refuses a family with more parameters than the class has returns.
3. Each class is fitted by maximum likelihood on its log returns per unit
   (`_fit_class`, `_mle_fit`): gaps as they are, regular bars per full bar.
   With no distribution given, normal, Student's t and Johnson SU
   are fitted and the lowest AIC or BIC wins.
4. Each future mark draws from its class's fit; short bars are mapped by
   the short-bar location–scale map; `_common.assemble` compounds the draws into paths.
"""
import warnings

import numpy as np
from scipy import optimize, stats

from . import _common
from ._common import GAP, REGULAR

# The families compared when no distribution is given
CANDIDATES = (stats.norm, stats.t, stats.johnsonsu)

# Shape starts for `_mle_fit`: scipy's own t start (df = 1) stops short of the maximum on
# fat-tailed returns (df ≈ 1.9 on t_3 data at a 0.2% scale), so several starts are climbed
_SHAPE_STARTS = {
    "t": [(2.0,), (3.0,), (5.0,), (10.0,), (30.0,)],
    "johnsonsu": [(a, b) for a in (-1.0, 0.0, 1.0) for b in (0.7, 1.5, 3.0)],
}


def parametric_monte_carlo(values=None, distribution=None, sim_length=100, n_sims=1000, random_state=None,
                           return_fit=False, *, classes=None, calendar="XNYS", criterion="BIC", min_sample=30,
                           prices=None, verbose=True):
    """Simulate future value paths from distributions fitted to historical log returns, one per class.

    Each class (gaps and regular bars) gets its own distribution,
    fitted by maximum likelihood to its log returns ℓ = ln(1 + r), and
    each future mark draws from its class's fit. With `distribution=None`, a
    normal, a Student's t and a Johnson SU are fitted to each class and the
    one with the lowest `criterion` is used.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or array_like
        Historical portfolio values (or prices), oldest first, all positive.
        A Series on a DatetimeIndex, or a `BacktestResult`, gives labelled
        output; anything else (a NumPy array, a list, a Series on a
        RangeIndex) is a plain array: one class and no labels. A
        single-column DataFrame is read as its column.
    distribution : scipy.stats continuous distribution, dict or None, default None
        Family to fit to log returns, e.g. ``stats.t`` or ``stats.laplace``,
        used for every class; or a dict ``{"gap": ..., "regular": ...}``
        with one family (or None) per class the history has. None selects
        among normal, t and Johnson SU by `criterion`.
    sim_length : int, default 100
        Horizon. On intraday history, sessions: if the history ends
        mid-session, the rest of that session counts as the first. On daily
        and longer history, and in continuous mode, bars.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the draws. An int or a Generator gives
        reproducible paths; None gives a fresh, unseeded Generator.
    return_fit : bool, default False
        Deprecated and ignored: the fit is always in the result's ``fit``.

        .. deprecated:: 0.4.0
            `return_fit` will be removed in 1.0.0.
    classes : pandas.Series, array_like or None, default None
        Class of each return, ``"gap"``, ``"regular"`` or ``"irregular"``,
        indexed by the mark where it ends (or one per return, in order).
        None takes a `BacktestResult`'s classes, or infers them from the
        labels with `classify_returns`. Irregular returns are left out
        of every fit.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code for the future marks and the
        classes; None is continuous mode for 24/7 data: one class, no
        rescaling, future marks every bar size Δ.
    criterion : {"BIC", "AIC"}, default "BIC"
        Score that selects the family when `distribution` is None, case
        insensitive: BIC = −2 ln L + k ln n, AIC = −2 ln L + 2k, with k the
        family's parameter count and n the class's returns. BIC penalizes
        the extra parameters of t and Johnson SU more on long histories.
    min_sample : int, default 30
        Fewest returns a class may have. 0 disables the check; a
        class the forecast needs with no returns, or with fewer returns
        than the family has parameters, always raises.
    prices : pandas.Series or array_like, optional
        Deprecated alias for `values`, keyword only.

        .. deprecated:: 0.2.0
            `prices` will be removed in 1.0.0; use `values` instead.
    verbose : bool, default True
        Print the family selection and fitted parameters (see Notes).
        False prints nothing to stdout; warnings are unaffected.

    Returns
    -------
    SimulationResult
        ``paths``: shape ``(n_sims, n_marks + 1)``, one path per row; column
        0 is 1.0, so every value is a growth multiple of the last historical
        value. ``dates`` and ``classes``: the future marks and the class of
        the return ending at each; None for a plain array, whose
        ``n_marks`` is `sim_length`. ``fit``: a dict with one entry per
        modelled class, ``"regular"`` and (intraday) ``"gap"``, each a dict:

        ``"Distribution"``
            scipy.stats name of the family used, e.g. ``'johnsonsu'``.
        ``"Parameters"``
            dict of its fitted parameters by name: the shape parameters
            (e.g. ``a``, ``b``), then ``loc`` and ``scale``, of the log
            return per unit (per gap, or per full bar).
        ``"Criterion"``
            ``"BIC"`` or ``"AIC"``.
        ``"BIC"`` or ``"AIC"`` (the criterion's name)
            The family's score.
        ``"Candidates"``
            dict of every candidate's score by name, lowest first, when
            the family was selected; otherwise None.

    Raises
    ------
    ValueError
        If `values` has fewer than two values, a non-finite or non-positive
        value, or more than one column; if `classes` doesn't match it; if
        `n_sims` is not a positive integer, `sim_length` is negative or
        `min_sample` is not an integer >= 0; if a class has fewer than
        `min_sample` returns, naming the class, its count and the history
        needed, none at all, or fewer than the family's parameters;
        if `distribution` is a dict missing a modelled class or with an
        unknown key; if `criterion` is not AIC or BIC; if the series has one
        mark per session away from the close; or if intraday labels
        have neither classes nor open marks.
    TypeError
        If neither or both of `values` and `prices` are given, or `classes`
        is given with a plain array.

    Warns
    -----
    UserWarning
        On intraday history, that `sim_length` counts sessions.
    DeprecationWarning
        If `prices` or `return_fit` is given.

    See Also
    --------
    nonparametric_monte_carlo : Resamples historical returns instead.
    regime_switching_monte_carlo : Lets volatility switch between regimes.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.

    Notes
    -----
    With `verbose=True`, prints each class's candidate scores and the
    family selected with its fitted parameters by name, e.g.
    ``Selected: t  (df 4.12, loc 0.000912, scale 0.00981)``; with a family
    given, one ``Fitted:`` line per class in the same format instead. The
    parameters are those of the log return per unit (per gap, or per full
    bar), as in ``fit[cls]["Parameters"]``.

    Fitting log returns keeps every path positive,
    V_t = V_0 exp(ℓ_1 + … + ℓ_t), however heavy the fitted left tail.

    Short bars: regular returns are fitted per full bar,
    ℓ' = m̂ + (ℓ − ρ m̂)/√ρ with ρ = τ/Δ and m̂ = Σ_t ℓ_t / Σ_t ρ_t over the
    regular returns, and a draw for a short future bar is mapped back,
    ℓ = ρ m̂ + √ρ (ℓ' − m̂). Full bars and gaps are unchanged.

    Examples
    --------
    >>> from scipy import stats
    >>> sims = parametric_monte_carlo(values, distribution=stats.t, random_state=42)
    >>> sims = parametric_monte_carlo(values, random_state=42)
    >>> sims.fit["regular"]["Distribution"], sims.fit["regular"]["Criterion"]
    ('t', 'BIC')
    """
    # Check the arguments; the deprecated keywords warn
    values = _common.resolve_values(values, prices, "parametric_monte_carlo")
    _common.warn_return_fit(return_fit, "parametric_monte_carlo")
    n_sims = _common.check_n_sims(n_sims)
    criterion = _common.normalize_criterion(criterion)
    # Kept log returns, classes, ρ per mark and the future marks
    history = _common.prepare_history(values, classes, calendar, sim_length, "parametric_monte_carlo")
    # Each class in use needs min_sample returns; one the forecast needs can't be empty
    _common.check_min_sample(history, min_sample)
    # The family to fit per class, or None to select one
    families = _families(distribution, history.model_classes)
    rng = np.random.default_rng(random_state)

    # Fit each class on its own, per full unit, picking the family by criterion if none is given
    fit, fitted = {}, {}
    labelled = len(history.model_classes) > 1
    for cls in history.model_classes:
        # The chosen scipy family, its parameters and the class's fit dict entry
        dist, params, entry = _fit_class(history.unit_pool(cls), families[cls], criterion, cls,
                                         cls + " " if labelled else "", verbose)
        fit[cls], fitted[cls] = entry, (dist, params)

    # Each class's future marks draw i.i.d. from that class's fitted family
    log_returns = np.empty((n_sims, history.n_marks))
    for cls in (REGULAR, GAP):
        cols = np.flatnonzero(history.future_classes == cls)
        if cols.size == 0:
            continue
        dist, params = fitted[cls]
        draws = dist.rvs(*params, size=(n_sims, cols.size), random_state=rng)
        if cls == REGULAR:
            # Draws are per full bar; shrink them to each future bar's length about m̂
            draws = _common.from_unit(draws, history.future_rho[cols], history.m_hat)
        log_returns[:, cols] = draws
    # Paths V_t = V_0 exp(ℓ_1 + … + ℓ_t) with V_0 = 1, plus the per-class fit dict
    return _common.assemble(log_returns, history, fit)


def _families(distribution, model_classes):
    """Return the family (or None, to select) for each modelled class from `distribution`.

    `distribution` is one family (or None) for every class, or a dict keyed
    by ``"gap"`` and ``"regular"`` that must cover every modelled class.
    """
    # One family, or None, applies to every class
    if not isinstance(distribution, dict):
        return {cls: distribution for cls in model_classes}
    unknown = sorted(set(distribution) - {GAP, REGULAR})
    if unknown:
        raise ValueError(f'distribution dict keys must be "gap" and "regular", got {unknown}')
    missing = [cls for cls in model_classes if cls not in distribution]
    if missing:
        raise ValueError(f"distribution dict has no entry for the {' and '.join(missing)} class, which the "
                         "history has; add it (None selects a family)")
    return {cls: distribution[cls] for cls in model_classes}


def _n_params(dist):
    """Return a scipy family's fitted parameter count k: its shapes, then loc and scale."""
    return dist.numargs + 2


def _fit_class(data, family, criterion, cls, label, verbose):
    """Fit one class's log returns by maximum likelihood; return (dist, params, fit entry).

    `data` holds the class's log returns per unit, `cls` names the class
    in messages and `label` prefixes the printed lines. `family` None
    selects among `CANDIDATES` by `criterion`, printing each score
    and the winner with its parameters if `verbose`; a given family prints
    only its parameters. A family with more parameters than the class has
    returns can't be fitted and raises.
    """
    candidates = CANDIDATES if family is None else (family,)
    # A family needs at least as many returns as parameters, whatever min_sample
    for dist in candidates:
        if len(data) < _n_params(dist):
            raise ValueError(f"The {cls} class has {len(data)} returns, fewer than the {_n_params(dist)} "
                             f"parameters of {dist.name}, so it can't be fitted; use a longer history")
    # Fit and score every candidate; the lowest criterion wins
    fits, scores = {}, {}
    for dist in candidates:
        # Maximum-likelihood parameters and their log L
        params, log_lik = _mle_fit(dist, data)
        fits[dist] = params
        scores[dist] = _common.information_criterion(log_lik, len(params), len(data), criterion)
        if family is None and verbose:
            print(f"{label}{dist.name:>10} {criterion}: {scores[dist]:,.2f}")
    dist = min(scores, key=scores.get)
    params = fits[dist]
    # The fit dict entry; scipy orders the parameters shapes first, then loc and scale
    names = [n.strip() for n in (dist.shapes or "").split(",") if n.strip()] + ["loc", "scale"]
    named = {name: float(value) for name, value in zip(names, params, strict=True)}
    if verbose:
        # The family and its parameters by name; 3 significant figures keep tiny hourly scales readable
        shown = ", ".join(f"{name} {value:.3g}" for name, value in named.items())
        print(f"{label}{'Selected' if family is None else 'Fitted'}: {dist.name}  ({shown})")
    entry = {
        "Distribution": dist.name,
        "Parameters": named,
        "Criterion": criterion,
        criterion: float(scores[dist]),
        "Candidates": None,
    }
    if family is None:
        entry["Candidates"] = {d.name: float(s) for d, s in sorted(scores.items(), key=lambda item: item[1])}
    return dist, params, entry


def _log_lik(dist, data, params):
    """Return the log-likelihood of `data` under `dist` at `params`, -inf where undefined."""
    value = dist.logpdf(data, *params).sum()
    return float(value) if np.isfinite(value) else -np.inf


def _mle_fit(dist, data):
    """Return (params, log L) of the maximum-likelihood fit of `dist` to `data`.

    Family selection compares maximized likelihoods, so a fit stopped
    short of the maximum can pick the wrong family. The fit is climbed from
    scipy's own start and, for t and Johnson SU, from a grid of shape
    starts (loc at the median, scale at the normal-consistent MAD), and the
    best is polished by Nelder-Mead on (shapes, loc, ln scale) with tight
    tolerances. A step is kept only if it raises log L, so the result is
    never worse than scipy's default fit.
    """
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # Optimizer and overflow warnings from trial points far off the
        # maximum; only the best fit is kept
        warnings.simplefilter("ignore", RuntimeWarning)
        # Candidate fits: scipy's own start, then the shape grid from a robust loc and scale
        fits = [tuple(dist.fit(data))]
        loc = float(np.median(data))
        scale = float(stats.median_abs_deviation(data, scale="normal")) or float(np.std(data)) or 1.0
        for shapes in _SHAPE_STARTS.get(dist.name, []):
            try:
                fits.append(tuple(dist.fit(data, *shapes, loc=loc, scale=scale)))
            except (ValueError, RuntimeError, FloatingPointError):
                continue
        # The highest-likelihood candidate, then polished if that raises log L
        params = max(fits, key=lambda p: _log_lik(dist, data, p))
        params = _polish(dist, data, params)
    return tuple(float(p) for p in params), _log_lik(dist, data, params)


def _polish(dist, data, params):
    """Return `params` refined by Nelder-Mead on (shapes, loc, ln scale), if that raises log L."""
    k = dist.numargs

    def objective(z):
        """Return the negative log L at the unpacked point, or 1e300 where it is undefined."""
        value = -_log_lik(dist, data, (*z[:k + 1], np.exp(z[k + 1])))
        return value if np.isfinite(value) else 1e300

    # Searching on ln scale keeps the scale positive without a constraint,
    # which Nelder-Mead can't take
    start = np.r_[params[:k + 1], np.log(params[k + 1])]
    result = optimize.minimize(objective, start, method="Nelder-Mead",
                               options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000, "maxfev": 20_000,
                                        "adaptive": True})
    # Keep the polished point only if it beats the start
    polished = (*result.x[:k + 1], float(np.exp(result.x[k + 1])))
    return polished if _log_lik(dist, data, polished) > _log_lik(dist, data, params) else tuple(params)
