"""`regime_switching_monte_carlo`: simulate from a Gaussian hidden Markov model of regimes.

Flow
----
1. `_common.prepare_history` turns the input into log returns per class,
   with irregular returns left out, and lays out the future marks.
2. `_ChainData` picks the marks the regime chain runs on: every mark,
   gaps included, under ``gap_model="regime"``; regular marks only under
   ``"independent"`` and whenever the history has no gaps.
3. `_hmm.fit_hmm` fits each candidate regime count (one given, or 1-3);
   `_cell_problem` applies the per-regime `min_sample` rule and
   `_common.information_criterion` picks among the eligible counts.
4. `_simulate` starts each path from the filtered regime probabilities at
   the last historical mark, steps the chain once per future mark it runs
   on, and draws each mark from its (regime, class) emission or pool; a
   short bar's draw is shrunk to its length about the regime's own regular
   mean, ρ of the drift and √ρ of the volatility.
5. `_common.assemble` compounds the draws into paths.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import _common, _hmm
from ._common import GAP, REGULAR

GAP_MODELS = ("regime", "independent")
# The regime counts compared when n_regimes is None
_SELECTION_COUNTS = (1, 2, 3)


def regime_switching_monte_carlo(values=None, n_regimes=None, bootstrap=False, n_starts=10, sim_length=100,
                                 n_sims=1000, random_state=None, return_fit=False, *, classes=None,
                                 calendar="XNYS", gap_model="regime", kappa=0.0, criterion="BIC", min_sample=30,
                                 prices=None, verbose=True):
    """Simulate future value paths from a Gaussian hidden Markov model of regimes.

    A hidden Markov model is fitted to the history's log returns: one chain
    of hidden regimes (e.g. calm and rising, volatile and falling), each
    with its own normal emission per class. Each path starts from the
    model's regime probabilities at the last historical mark, moves between
    regimes by the fitted transition matrix, one step per mark, and draws
    each mark's log return from its regime. Unlike the other two
    simulators, this keeps volatility clustering.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or array_like
        Historical portfolio values (or prices), oldest first, all positive.
        A Series on a DatetimeIndex, or a `BacktestResult`, gives labelled
        output; anything else (a NumPy array, a list, a Series on a
        RangeIndex) is a plain array: one class and no labels. A
        single-column DataFrame is read as its column.
    n_regimes : int or None, default None
        Number of regimes S. None fits 1, 2 and 3 and uses the eligible one
        with the lowest `criterion`.
    bootstrap : bool, default False
        How a mark is drawn within its regime. False draws from the
        fitted normal of its (regime, class) cell, or under
        ``gap_model="independent"`` a gap from the single gap normal. True
        resamples the historical returns of that class whose most likely
        regime is that one (regular bars per full bar), or under
        ``"independent"`` a gap from the whole gap pool.
    n_starts : int, default 10
        EM starts per regime count; EM finds a local optimum, so the
        highest log-likelihood of `n_starts` is kept.
    sim_length : int, default 100
        Horizon. On intraday history, sessions: if the history ends
        mid-session, the rest of that session counts as the first. On daily
        and longer history, and in continuous mode, bars.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the EM starts and the draws. An int or a
        Generator gives reproducible paths; None gives a fresh, unseeded
        Generator.
    return_fit : bool, default False
        Deprecated and ignored: the fit is always in the result's ``fit``.

        .. deprecated:: 0.4.0
            `return_fit` will be removed in 1.0.0.
    classes : pandas.Series, array_like or None, default None
        Class of each return, ``"gap"``, ``"regular"`` or ``"irregular"``,
        indexed by the mark where it ends (or one per return, in order).
        None takes a `BacktestResult`'s classes, or infers them from the
        labels with `classify_returns`. Irregular returns are left out
        of the fit.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code for the future marks and the
        classes; None is continuous mode for 24/7 data: one class, no
        rescaling, future marks every bar size Δ.
    gap_model : {"regime", "independent"}, default "regime"
        How gaps relate to the regime. ``"regime"``: the chain steps
        at every mark, gaps included, and each regime has a gap emission as
        well as a regular one (2S emission pairs). ``"independent"``: the
        chain runs on regular marks only, and gaps are one normal (or one
        pool) whatever the regime. Ignored when the history has no gaps.
    kappa : float, default 0.0
        Sticky prior weight on staying in a regime: each row of the
        transition matrix has a Dir(1, .., 1 + kappa, .., 1) prior and the
        fit is its MAP estimate. Larger values give longer-lived regimes; 0
        is maximum likelihood.
    criterion : {"BIC", "AIC"}, default "BIC"
        Score that picks the regime count when `n_regimes` is None, case
        insensitive; BIC penalizes extra regimes more.
    min_sample : int, default 30
        Data each regime must have. Under ``"regime"`` with
        gaps: each regime's gap occupancy, the expected number of gap marks
        in it (the count of gap marks most likely in it when `bootstrap`),
        must be at least `min_sample`. Otherwise (``"independent"``, or one
        class under either `gap_model`): each class must have at least
        `min_sample` returns. With `n_regimes=None` a count
        failing it is not eligible; a given `n_regimes` raises. 0 disables
        the check.
    prices : pandas.Series or array_like, optional
        Deprecated alias for `values`, keyword only.

        .. deprecated:: 0.2.0
            `prices` will be removed in 1.0.0; use `values` instead.
    verbose : bool, default True
        Print the regime-count scores and the regimes (see Notes). False
        prints nothing to stdout; warnings are unaffected.

    Returns
    -------
    SimulationResult
        ``paths``: shape ``(n_sims, n_marks + 1)``, one path per row; column
        0 is 1.0, so every value is a growth multiple of the last historical
        value. ``dates`` and ``classes``: the future marks and the class of
        the return ending at each; None for a plain array, whose
        ``n_marks`` is `sim_length`. ``fit``: a dict:

        ``"Regime Count"``
            Number of regimes used.
        ``"Gap Model"``, ``"Kappa"``, ``"Criterion"``
            The settings used.
        ``"BIC"`` or ``"AIC"`` (the criterion's name)
            dict of the score by regime count: every count fitted and
            eligible when `n_regimes` is None, otherwise just `n_regimes`.
        ``"Log Likelihood"``
            Data log-likelihood of the model used, gap normal included
            under ``"independent"`` with ``bootstrap=False`` (at the MAP
            estimate when kappa > 0).
        ``"Regimes"``
            pandas.DataFrame, one row per regime ordered from lowest to
            highest regular volatility: ``Mean`` and ``Volatility`` of the
            log return per full bar, ``Gap Mean`` and ``Gap Volatility``
            (under ``"regime"`` with gaps), ``Expected Duration`` (marks the
            chain steps on, ``1 / (1 - P_ss)``) and ``Current Probability``
            (of the regime at the last historical mark).
        ``"Transition Matrix"``
            pandas.DataFrame of one-step transition probabilities, from the
            row regime to the column regime, in the same order.
        ``"Gap"``
            Under ``"independent"`` with gaps and ``bootstrap=False`` only:
            dict of the gap normal's ``Mean`` and ``Volatility``.

    Raises
    ------
    ValueError
        If `values` has fewer than two values, a non-finite or non-positive
        value, or more than one column; if `classes` doesn't match it; if
        `n_sims` or `n_regimes` is not a positive integer, `sim_length` is
        negative or `min_sample` is not an integer >= 0; if a given
        `n_regimes` can't be fitted (too few returns, every start
        degenerate) or fails `min_sample`, or no count of 1-3 is eligible;
        if a class the forecast needs has no returns; if the gaps are
        constant under ``"independent"`` with ``bootstrap=False`` (their
        normal is degenerate); for an unknown `gap_model` or `criterion`, or
        a negative `kappa`; if the series has one mark per session away from
        the close; or if intraday labels have neither classes nor open
        marks.
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
    nonparametric_monte_carlo : Resamples historical returns, no regimes.
    parametric_monte_carlo : Draws from one fitted distribution per class.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.

    Notes
    -----
    The model: regimes s_t follow one Markov chain with transition
    matrix P, stepping once per mark; given the regime, a mark of class c
    has log return ℓ_t ~ N(ρ_t μ[s, c], ρ_t σ²[s, c]), with
    ρ_t = τ/Δ below 1 only for a short bar. It is fitted by EM with a
    variance floor of 1e-12; a start is rejected if a (regime, class) cell's
    expected occupancy is below max(2, 1% of that class's marks), or its
    variance collapses to the floor.

    Free parameters k for S regimes: (S − 1) + S(S − 1) + 4S under
    ``"regime"`` with gaps; (S − 1) + S(S − 1) + 2S + 2 under
    ``"independent"`` with ``bootstrap=False`` (the gap normal's 2 do not
    change with S, so they cancel when comparing counts); (S − 1) + S(S − 1)
    + 2S with no gaps, or under ``"independent"`` with ``bootstrap=True``,
    where gaps are resampled and no gap normal is fitted. n is the number of
    marks the likelihood covers: every mark, except that resampled
    independent gaps are not counted.

    Tuning: more regimes or ``"regime"`` gap emissions need more data per
    cell, and `min_sample` refuses counts whose regimes see too few gaps;
    BIC prefers fewer regimes than AIC on long histories; `kappa` trades
    fit for persistence, and as it grows regimes last longer and switch
    less.

    With `verbose=True`, prints the score of each regime count (when
    selecting), then each regime's mean, volatility and expected duration,
    from lowest to highest volatility. Means and volatilities are of the
    log return per unit: regular bars per full bar, labelled ``regular`` on
    intraday history, and, under ``"regime"`` with gaps, each regime's gap
    moments after a ``|``, e.g. ``Regime 0: regular mean +0.0127%, vol
    0.2094% | gap mean +0.0500%, vol 0.7700% | expected duration 6.3
    marks``. The model numbers its regimes
    arbitrarily; ordering them by volatility only changes how they are
    reported, not the simulation.

    References
    ----------
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica 57(2),
    357-384.

    Examples
    --------
    >>> sims = regime_switching_monte_carlo(values, n_regimes=2, bootstrap=True,
    ...                                     random_state=42)
    >>> sims = regime_switching_monte_carlo(values, random_state=42)
    >>> sims.fit["Regime Count"]
    2
    >>> sims.fit["Regimes"]["Volatility"]  # calm to turbulent
    """
    # Check the arguments; the deprecated keywords warn
    values = _common.resolve_values(values, prices, "regime_switching_monte_carlo")
    _common.warn_return_fit(return_fit, "regime_switching_monte_carlo")
    n_sims = _common.check_n_sims(n_sims)
    criterion = _common.normalize_criterion(criterion)
    if gap_model not in GAP_MODELS:
        raise ValueError(f'gap_model must be "regime" or "independent", got {gap_model!r}')
    if n_regimes is not None and (int(n_regimes) != n_regimes or n_regimes < 1):
        raise ValueError(f"n_regimes must be an integer >= 1 or None, got {n_regimes!r}")
    min_sample = _common.check_min_sample_value(min_sample)
    # Kept log returns, classes, ρ per mark and the future marks
    history = _common.prepare_history(values, classes, calendar, sim_length, "regime_switching_monte_carlo")
    # Every class the forecast has marks of must have history returns
    _common.check_samplable(history)
    rng = np.random.default_rng(random_state)

    # The chain's marks, and the gap normal or pool when gaps are off the chain
    data = _ChainData(history, gap_model, bootstrap)
    if not data.joint:
        # With one class on the chain, min_sample applies per class, once
        _common.check_min_sample(history, min_sample)

    # Fit each candidate regime count by EM and score it; skip any that fail or lack data
    counts = [int(n_regimes)] if n_regimes is not None else list(_SELECTION_COUNTS)
    fits, scores, skipped = {}, {}, {}
    for S in counts:
        try:
            # Best of n_starts EM runs: π, P, μ and σ² per (regime, class), γ and log L
            fit = _hmm.fit_hmm(data.ell, data.cls, data.rho, n_states=S, kappa=kappa, n_starts=n_starts,
                               random_state=rng)
        except ValueError as err:
            if n_regimes is not None:
                raise ValueError(f"Could not fit {S} regimes: {err}") from err
            skipped[S] = str(err)
            if verbose:
                print(f"{S} regime(s): no stable fit, skipped")
            continue
        # Why this count fails min_sample or leaves a regime's pool empty; None if it passes
        problem = _cell_problem(fit, data, history, min_sample, bootstrap)
        if problem is not None:
            if n_regimes is not None:
                raise ValueError(problem)
            skipped[S] = problem
            if verbose:
                print(f"{S} regime(s): not eligible, {problem}")
            continue
        fits[S] = fit
        # Score the whole model: the chain plus any gap normal, in log L, k and n
        scores[S] = _common.information_criterion(fit.log_likelihood + data.gap_log_lik,
                                                  fit.n_params + data.gap_params, data.n_obs, criterion)
        if n_regimes is None and verbose:
            print(f"{S} regime(s) {criterion}: {scores[S]:,.2f}")
    if not fits:
        reasons = "; ".join(f"{S} regime(s): {why}" for S, why in skipped.items())
        raise ValueError(f"No regime count of {counts[0]}-{counts[-1]} could be used: {reasons}")
    # The eligible count with the lowest score is used
    S = min(scores, key=scores.get)
    if n_regimes is None and verbose:
        print(f"Selected: {S} regime(s)")
    fit = fits[S]

    # Log returns per path and future mark, drawn by stepping the chain
    log_returns = _simulate(fit, data, history, bootstrap, n_sims, rng)
    # Paths V_t = V_0 exp(ℓ_1 + … + ℓ_t) with V_0 = 1, plus the fit dict of Returns
    report = _report(fit, data, history.mode == "intraday", gap_model, kappa, criterion, scores, verbose)
    return _common.assemble(log_returns, history, report)


class _ChainData:
    """The marks the regime chain runs on, and the separate gap model under "independent".

    Built from the `History`, `gap_model` and `bootstrap` of the call.

    Attributes
    ----------
    joint : bool
        True if gaps are on the chain: ``gap_model="regime"`` and the
        history has gaps.
    ell, rho : numpy.ndarray
        Log return ℓ_t and length ratio ρ_t of each chain mark, t = 1..T.
    cls : numpy.ndarray of int or None
        Class code of each chain mark, 0 regular and 1 gap; None when the
        chain has one class.
    regular : numpy.ndarray of bool
        True for the chain's regular marks.
    gap_pool : numpy.ndarray
        Every kept gap log return, whether or not gaps are on the chain.
    independent_gaps : bool
        True if the history has gaps and they are off the chain.
    gap_normal : bool
        True if gaps are drawn from one fitted normal: off the chain and
        ``bootstrap=False``.
    gap_mean, gap_sd : float
        That normal's mean and standard deviation; NaN without one.
    gap_log_lik, gap_params : float and int
        Its log-likelihood and its 2 parameters, added to the chain's in
        the criterion; 0 without one.
    n_obs : int
        n of the criterion: the chain's T marks, plus the gaps when the gap
        normal is fitted.
    """

    def __init__(self, history, gap_model, bootstrap):
        """Split `history` per `gap_model`; fit the gap normal only if gaps are drawn from it."""
        has_gaps = history.count(GAP) > 0
        # Gaps ride the chain only under "regime" and only if the history has any
        self.joint = gap_model == "regime" and has_gaps
        # The chain's marks: all kept marks if joint, else regular marks only
        on_chain = np.ones(len(history.ell), dtype=bool) if self.joint else history.classes == REGULAR
        self.ell, self.rho = history.ell[on_chain], history.rho[on_chain]
        # Class code per chain mark (0 regular, 1 gap); None with one class
        self.cls = (history.classes[on_chain] == GAP).astype(int) if self.joint else None
        self.regular = history.classes[on_chain] == REGULAR
        # Every kept gap; gaps are one unit each, so the pool is their raw log returns
        self.gap_pool = history.unit_pool(GAP)
        self.independent_gaps = has_gaps and not self.joint
        # Off-chain gaps get a normal only without bootstrap, which resamples the pool instead
        self.gap_normal = self.independent_gaps and not bootstrap
        # No gap normal: nothing to add to log L or k, and no gap moments to report
        self.gap_log_lik, self.gap_params = 0.0, 0
        self.gap_mean = self.gap_sd = np.nan
        if self.gap_normal:
            # Fit the gap normal by maximum likelihood; constant gaps have none
            self.gap_mean = float(self.gap_pool.mean())
            var = float(self.gap_pool.var())
            if var <= _hmm.VAR_FLOOR:
                raise ValueError("The gap returns are constant, so their normal is degenerate; use bootstrap=True "
                                 'to resample them, gap_model="regime", or nonparametric_monte_carlo')
            self.gap_sd = float(np.sqrt(var))
            # Its log L and 2 parameters join the chain's in the criterion
            self.gap_log_lik = float(stats.norm.logpdf(self.gap_pool, self.gap_mean, self.gap_sd).sum())
            self.gap_params = 2
        # n of the criterion: every mark the likelihood covers, so gaps count
        # only where they are modelled (on the chain, or by the gap normal)
        self.n_obs = len(self.ell) + (len(self.gap_pool) if self.gap_normal else 0)


def _cell_problem(fit, data, history, min_sample, bootstrap):
    """Return why `fit` fails the minimum-sample rule, or None if it passes.

    `fit` is one regime count's fit to `data`'s chain. Under "regime" with
    gaps, each regime needs `min_sample` gap marks: expected ones, or with
    `bootstrap` those whose most likely regime it is. With `bootstrap`,
    every regime's gap and regular pools must also be non-empty. The
    returned string is the ValueError message.
    """
    S = len(fit.pi)
    # Each chain mark's most likely regime, arg max over s of γ_t(s)
    labels = fit.gamma.argmax(axis=1)
    if data.joint:
        gap = data.cls == 1
        # Gap marks per regime: expected (Σ γ_t(s) over gaps), or hard counts under bootstrap
        occupancy = np.bincount(labels[gap], minlength=S) if bootstrap else fit.gamma[gap].sum(axis=0)
        # An empty regime pool can't be drawn from, so it fails even with min_sample=0
        if (occupancy < min_sample).any() or (bootstrap and (occupancy == 0).any()):
            kind = "gap marks by most likely regime" if bootstrap else "expected gap marks"
            # :g shows hard counts as 19, not 19.0, and keeps the decimals of expected ones
            shown = ", ".join(f"{float(x):g}" for x in occupancy)
            return (f"under gap_model='regime' each regime needs at least min_sample={min_sample} "
                    f"{kind}, but the {S} regime(s) have {shown} {kind}; use fewer regimes, a longer "
                    "history, gap_model='independent', or a lower min_sample (0 disables the check)")
    if bootstrap:
        # Regular marks per regime; a regime with none has no pool to resample
        regular_counts = np.bincount(labels[data.regular], minlength=S)
        if (regular_counts == 0).any():
            return (f"with bootstrap=True regime {int(np.argmin(regular_counts))} of {S} has no regular marks "
                    "to resample; use fewer regimes or bootstrap=False")
    return None


def _simulate(fit, data, history, bootstrap, n_sims, rng):
    """Draw the log return of every path at every future mark from `fit`.

    Takes the chosen fit, the chain data, the history's future marks (their
    classes and ρ), `bootstrap`, `n_sims` and the generator. Returns shape
    ``(n_sims, n_marks)``: row a path, column j the log return ending at
    future mark j.
    """
    S = len(fit.pi)
    mu, sd = fit.mu, np.sqrt(fit.var)
    if bootstrap:
        labels = fit.gamma.argmax(axis=1)
        # Each regime's regular pool per full bar, about that regime's own mean μ_s,reg
        regular_pools = [_common.to_unit(data.ell[data.regular & (labels == s)],
                                         data.rho[data.regular & (labels == s)], mu[s, 0]) for s in range(S)]
        if data.joint:
            # Each regime's gap pool: the gaps whose most likely regime it is
            gap_pools = [data.ell[(data.cls == 1) & (labels == s)] for s in range(S)]

    # Each path's first regime, drawn from the filtered Pr(s_T | ℓ_1..ℓ_T);
    # renormalized so that rounding can't make rng.choice reject it
    start = fit.filtered_last / fit.filtered_last.sum()
    state = rng.choice(S, size=n_sims, p=start)
    # Cumulative rows of P, to step every path at once by inverse CDF
    cum_P = np.cumsum(fit.P, axis=1)
    out = np.empty((n_sims, history.n_marks))
    for j, (cls, rho) in enumerate(zip(history.future_classes, history.future_rho, strict=True)):
        if data.joint or cls == REGULAR:
            # One step per mark the chain runs on; the cap at S - 1 guards rows summing just under 1
            u = rng.random(n_sims)
            state = np.minimum((u[:, None] > cum_P[state]).sum(axis=1), S - 1)
        if cls == GAP and not data.joint:
            # Off-chain gap, whatever the regime: from the gap pool or the gap normal
            out[:, j] = data.gap_pool[rng.integers(len(data.gap_pool), size=n_sims)] if bootstrap \
                else rng.normal(data.gap_mean, data.gap_sd, size=n_sims)
            continue
        c = 1 if cls == GAP else 0
        # Paths in regime s draw from cell (s, c): its pool, or its normal per full unit
        for s in range(S):
            idx = np.flatnonzero(state == s)
            if idx.size == 0:
                continue
            if bootstrap:
                pool = gap_pools[s] if c == 1 else regular_pools[s]
                draws = pool[rng.integers(len(pool), size=idx.size)]
            else:
                draws = rng.normal(mu[s, c], sd[s, c], size=idx.size)
            # Regular draws shrunk to this bar's length: ρ of the regime's drift, √ρ of its volatility
            out[idx, j] = _common.from_unit(draws, rho, mu[s, c]) if c == 0 else draws
    return out


def _report(fit, data, intraday, gap_model, kappa, criterion, scores, verbose):
    """Return the simulator's fit dict (see its Returns), printing each regime if `verbose`.

    Takes the chosen fit, the chain data, whether the history is intraday
    (which labels the printed moments ``regular``), the settings to echo
    and `scores`, the criterion score of each eligible regime count.
    Regimes are listed calm to turbulent, by regular volatility.
    """
    S = len(fit.pi)
    P = fit.P
    mu, sd = fit.mu, np.sqrt(fit.var)
    # Expected stay in each regime, in chain marks, 1 / (1 - P_ss); infinite if it never leaves
    durations = np.array([np.inf if P[k, k] >= 1 else 1 / (1 - P[k, k]) for k in range(S)])
    # Regimes are numbered arbitrarily by the fit; report them calm to turbulent
    order = np.argsort(sd[:, 0], kind="stable")
    if verbose:
        for rank, k in enumerate(order):
            # Intraday moments are named by class, since gaps have their own; daily has one class
            moments = f"{'regular ' if intraday else ''}mean {mu[k, 0]:+.4%}, vol {sd[k, 0]:.4%}"
            if data.joint:
                # Gaps on the chain: each regime's gap moments too, the pairs set apart by |
                moments += f" | gap mean {mu[k, 1]:+.4%}, vol {sd[k, 1]:.4%} |"
            else:
                moments += ","
            print(f"Regime {rank}: {moments} expected duration {durations[k]:.1f} marks")
    # Regime table: regular moments, gap moments if gaps are on the chain, duration, current probability
    columns = {"Mean": mu[order, 0], "Volatility": sd[order, 0]}
    if data.joint:
        columns.update({"Gap Mean": mu[order, 1], "Gap Volatility": sd[order, 1]})
    columns.update({"Expected Duration": durations[order], "Current Probability": fit.filtered_last[order]})
    labels = pd.RangeIndex(S, name="Regime")
    report = {
        "Regime Count": int(S),
        "Gap Model": gap_model,
        "Kappa": float(kappa),
        "Criterion": criterion,
        criterion: {k: float(v) for k, v in sorted(scores.items())},
        "Log Likelihood": float(fit.log_likelihood + data.gap_log_lik),
        "Regimes": pd.DataFrame(columns, index=labels),
        "Transition Matrix": pd.DataFrame(P[np.ix_(order, order)], index=labels, columns=labels),
    }
    # The off-chain gap normal, when one was fitted
    if data.gap_normal:
        report["Gap"] = {"Mean": data.gap_mean, "Volatility": data.gap_sd}
    return report
