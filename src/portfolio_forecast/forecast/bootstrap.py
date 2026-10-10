"""`nonparametric_monte_carlo`: resample historical log returns, class by class.

Flow
----
1. `_common.prepare_history` turns the input into log returns per class,
   with irregular returns left out, and lays out the future marks.
2. `_common.check_min_sample` applies the minimum sample to each class.
3. Each future mark draws, with replacement, from its own class's pool:
   gaps from the gap pool, regular bars from the regular pool per full bar,
   mapped onto a short future bar by the short-bar location–scale map.
4. `_common.assemble` compounds the draws into paths.
"""
import numpy as np

from . import _common
from ._common import GAP, REGULAR


def nonparametric_monte_carlo(values=None, sim_length=100, n_sims=1000, random_state=None, *, classes=None,
                              calendar="XNYS", min_sample=30, prices=None):
    """Simulate future value paths by resampling historical log returns within each class.

    Each future mark draws one historical log return, with replacement, from
    the pool of its class: a gap mark from the gaps (last close to next
    open), a bar from the regular bars. The draws compound into a path. No
    distribution is assumed, so fat tails and skew in the history carry
    over, but every draw is independent: volatility clustering and
    autocorrelation do not, nor does the intraday U-shape of volatility.

    Parameters
    ----------
    values : pandas.Series, BacktestResult or array_like
        Historical portfolio values (or prices), oldest first, all positive.
        A Series on a DatetimeIndex, or a `BacktestResult`, gives labelled
        output; anything else (a NumPy array, a list, a Series on a
        RangeIndex) is a plain array: one class and no labels. A
        single-column DataFrame is read as its column.
    sim_length : int, default 100
        Horizon. On intraday history, sessions: if the history ends
        mid-session, the rest of that session counts as the first. On daily
        and longer history, and in continuous mode, bars.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the draws. An int or a Generator gives
        reproducible paths; None gives a fresh, unseeded Generator.
    classes : pandas.Series, array_like or None, default None
        Class of each return, ``"gap"``, ``"regular"`` or ``"irregular"``,
        indexed by the mark where it ends (or one per return, in order).
        None takes a `BacktestResult`'s classes, or infers them from the
        labels with `classify_returns`. Irregular returns are never
        drawn.
    calendar : str or None, default "XNYS"
        ``exchange_calendars`` calendar code for the future marks and the
        classes; None is continuous mode for 24/7 data: one pool, no
        rescaling, future marks every bar size Δ.
    min_sample : int, default 30
        Fewest returns a class may have. 0 disables the check; a
        class the forecast needs with no returns always raises.
    prices : pandas.Series or array_like, optional
        Deprecated alias for `values`, keyword only.

        .. deprecated:: 0.2.0
            `prices` will be removed in 1.0.0; use `values` instead.

    Returns
    -------
    SimulationResult
        ``paths``: shape ``(n_sims, n_marks + 1)``, one path per row; column
        0 is 1.0, so every value is a growth multiple of the last historical
        value. ``dates`` and ``classes``: the future marks and the class of
        the return ending at each, from `next_trading_dates`; None for
        a plain array, whose ``n_marks`` is `sim_length`. ``fit``: None.

    Raises
    ------
    ValueError
        If `values` has fewer than two values, a non-finite or non-positive
        value, or more than one column; if `classes` doesn't match it; if
        `n_sims` is not a positive integer, `sim_length` is negative or
        `min_sample` is not an integer >= 0; if a class has fewer than
        `min_sample` returns, naming the class, its count and the history
        needed, or none at all; if the series has one mark per session
        away from the close (see `to_session_values`); or if intraday
        labels have neither classes nor open marks.
    TypeError
        If neither or both of `values` and `prices` are given, or `classes`
        is given with a plain array.

    Warns
    -----
    UserWarning
        On intraday history, that `sim_length` counts sessions.
    DeprecationWarning
        If `prices` is given.

    See Also
    --------
    parametric_monte_carlo : Draws from a distribution fitted per class.
    regime_switching_monte_carlo : Lets volatility switch between regimes.
    portfolio_forecast.utils.next_trading_dates : The future marks.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.

    Notes
    -----
    Log returns ℓ = ln(1 + r) are drawn, so paths are
    V_t = V_0 exp(ℓ_1 + … + ℓ_t) with V_0 = 1 and stay positive.

    Short bars: a regular return of length ratio ρ = τ/Δ < 1 (a
    half-day's last bar) enters the pool per full bar,
    ℓ' = m̂ + (ℓ − ρ m̂)/√ρ, and a draw for a short future bar is mapped back,
    ℓ = ρ m̂ + √ρ (ℓ' − m̂), where m̂ = Σ_t ℓ_t / Σ_t ρ_t over the regular
    returns: drift scales with ρ, volatility with √ρ. Full bars are drawn
    as they are. Gaps are one unit whatever their clock length.

    Examples
    --------
    >>> sims = nonparametric_monte_carlo(values, sim_length=100, n_sims=1000,
    ...                                  random_state=42)
    >>> sims.paths.shape
    (1000, 101)
    """
    # Check the arguments; the deprecated prices keyword warns
    values = _common.resolve_values(values, prices, "nonparametric_monte_carlo")
    n_sims = _common.check_n_sims(n_sims)
    # Kept log returns, classes, ρ per mark and the future marks
    history = _common.prepare_history(values, classes, calendar, sim_length, "nonparametric_monte_carlo")
    # Each class in use needs min_sample returns; one the forecast needs can't be empty
    _common.check_min_sample(history, min_sample)
    rng = np.random.default_rng(random_state)

    # Each class's future marks draw i.i.d., with replacement, from that class's pool only
    log_returns = np.empty((n_sims, history.n_marks))
    for cls in (REGULAR, GAP):
        cols = np.flatnonzero(history.future_classes == cls)
        if cols.size == 0:
            continue
        # The class's history per full unit: regular bars stretched about m̂, gaps as they are
        pool = history.unit_pool(cls)
        draws = pool[rng.integers(len(pool), size=(n_sims, cols.size))]
        if cls == REGULAR:
            # Pool draws are per full bar; shrink them to each future bar's length.
            # Gaps count as one unit whatever their length, so only regular marks map
            draws = _common.from_unit(draws, history.future_rho[cols], history.m_hat)
        log_returns[:, cols] = draws
    # Paths V_t = V_0 exp(ℓ_1 + … + ℓ_t) with V_0 = 1; the bootstrap has no fit
    return _common.assemble(log_returns, history, None)
