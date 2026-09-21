import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats


def _resolve_values(values, prices, func_name):
    # Accepting the deprecated `prices` keyword in place of `values` until 1.0.0
    if prices is None:
        if values is None:
            raise TypeError(f"{func_name}() missing 1 required argument: 'values'")
        return values
    if values is not None:
        raise TypeError(f"{func_name}() got both 'values' and 'prices'; pass only 'values'")
    warnings.warn(f"{func_name}(prices=...) is deprecated and will be removed in 1.0.0; "
                  "use values=... instead", DeprecationWarning, stacklevel=3)
    return prices


def nonparametric_monte_carlo(values=None, sim_length=100, n_sims=1000, random_state=None, *, prices=None):
    """Simulate future value paths by resampling historical returns.

    Each path draws `sim_length` one-period returns, with replacement, from
    the history in `values` (a bootstrap) and compounds them. No distribution
    is assumed, so fat tails and skew in the history carry over, but every
    draw is independent: volatility clustering and autocorrelation do not.

    Parameters
    ----------
    values : pandas.Series or single-column pandas.DataFrame
        Historical prices or portfolio values, one row per bar, oldest first.
        The bar size of `values` is the length of one simulated period.
    sim_length : int, default 100
        Number of future periods to simulate per path.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the draws. An int or a Generator gives
        reproducible paths; None gives a fresh, unseeded Generator.
    prices : pandas.Series or single-column pandas.DataFrame, optional
        Deprecated alias for `values`, keyword only.

        .. deprecated:: 0.2.0
            `prices` will be removed in 1.0.0; use `values` instead.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sims, sim_length + 1)``, one path per row. Column
        0 is the starting value, 1.0, so every value is a growth multiple of
        the last historical price.

    Raises
    ------
    ValueError
        If `values` has more than one column.
    TypeError
        If neither or both of `values` and `prices` are given.

    Warns
    -----
    DeprecationWarning
        If `prices` is given.

    See Also
    --------
    parametric_monte_carlo : Draws returns from a fitted distribution instead.
    regime_switching_monte_carlo : Lets volatility switch between regimes.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.
    portfolio_forecast.plotting.plot_simulated_paths : Fan chart of the paths.

    Examples
    --------
    >>> sims = nonparametric_monte_carlo(values, sim_length=100, n_sims=1000,
    ...                                  random_state=42)
    >>> sims.shape
    (1000, 101)
    """
    values = _resolve_values(values, prices, 'nonparametric_monte_carlo')
    rng = np.random.default_rng(random_state)

    # Calculating the percent change between values
    returns = values.pct_change().dropna()

    # Creating a matrix to store the returns for each
    sims = np.zeros((n_sims, sim_length + 1))
    sims[:,0] = 1

    for i in range(n_sims):
        # Sampling from the returns with replacement
        ret_sample = returns.sample(n=sim_length, replace=True, random_state=rng).reset_index(drop=True)
        # Obtaining the single period accumulation factors
        acc_factor = ret_sample.to_numpy().flatten() + 1  # ensure 1D array, just in case
        # The cumulative returns over the simulation period
        value = np.cumprod(acc_factor)
        # Adding the simulated values to the simulation matrix
        sims[i, 1:] = value  # value should be shape (sim_length,)

    return sims



def parametric_monte_carlo(values=None, distribution=None, sim_length=100, n_sims=1000, random_state=None,
                           return_fit=False, *, prices=None):
    """Simulate future value paths from a distribution fitted to historical returns.

    One-period returns are fitted by maximum likelihood to a scipy.stats
    distribution, and each path compounds `sim_length` independent draws from
    it. With `distribution=None`, a normal, a Student's t and a Johnson SU are
    all fitted and the one with the lowest AIC is used.

    Parameters
    ----------
    values : pandas.Series or single-column pandas.DataFrame
        Historical prices or portfolio values, one row per bar, oldest first.
        The bar size of `values` is the length of one simulated period.
    distribution : scipy.stats continuous distribution or None, default None
        Distribution to fit, e.g. ``stats.t``, ``stats.norm`` or
        ``stats.laplace``. None selects among normal, t and Johnson SU by AIC.
    sim_length : int, default 100
        Number of future periods to simulate per path.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the draws. An int or a Generator gives
        reproducible paths; None gives a fresh, unseeded Generator.
    return_fit : bool, default False
        If True, also return the fitted distribution; see Returns.
    prices : pandas.Series or single-column pandas.DataFrame, optional
        Deprecated alias for `values`, keyword only.

        .. deprecated:: 0.2.0
            `prices` will be removed in 1.0.0; use `values` instead.

    Returns
    -------
    sims : numpy.ndarray
        Array of shape ``(n_sims, sim_length + 1)``, one path per row. Column
        0 is the starting value, 1.0, so every value is a growth multiple of
        the last historical price.
    fit : dict
        Only if `return_fit` is True:

        ``"Distribution"``
            scipy.stats name of the distribution used, e.g. ``'johnsonsu'``.
        ``"Parameters"``
            dict of its fitted parameters by name: the shape parameters
            (e.g. ``a``, ``b``), then ``loc`` and ``scale``.
        ``"AIC"``
            Its AIC, ``2k - 2 log L``.
        ``"Candidates"``
            dict of every candidate's AIC by name when `distribution` is
            None, lowest first; otherwise None.

    Raises
    ------
    TypeError
        If neither or both of `values` and `prices` are given.

    Warns
    -----
    DeprecationWarning
        If `prices` is given.

    Notes
    -----
    With `distribution=None`, prints each candidate's AIC
    (``2k - 2 log L``, k fitted parameters) and the one selected.

    The fitted distribution is unbounded, so a draw can fall below -100%
    and send a path to zero or below; heavy-tailed fits make this more
    likely on long horizons. `statistical_report` leaves CAGR undefined (NaN)
    for such paths.

    See Also
    --------
    nonparametric_monte_carlo : Resamples historical returns instead.
    regime_switching_monte_carlo : Lets volatility switch between regimes.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.

    Examples
    --------
    >>> from scipy import stats
    >>> sims = parametric_monte_carlo(values, distribution=stats.t,
    ...                               random_state=42)
    >>> sims, fit = parametric_monte_carlo(values, random_state=42, return_fit=True)
    >>> fit["Distribution"], fit["AIC"]
    ('johnsonsu', -13480.71...)
    """
    values = _resolve_values(values, prices, 'parametric_monte_carlo')
    rng = np.random.default_rng(random_state)

    # Calculating the percent change between values
    returns = values.pct_change().dropna()

    data = returns.to_numpy().flatten()

    def aic(dist, dist_params):
        # AIC = 2k - 2*log-likelihood
        return 2 * len(dist_params) - 2 * np.sum(dist.logpdf(data, *dist_params))

    if distribution is None:
        # Fitting each candidate via MLE and scoring it with AIC
        candidates = [stats.norm, stats.t, stats.johnsonsu]
        fits, aics = {}, {}
        for dist in candidates:
            fits[dist] = dist.fit(data)
            aics[dist] = aic(dist, fits[dist])
            print(f'{dist.name:>10} AIC: {aics[dist]:,.2f}')

        # Choosing the distribution with the lowest AIC
        distribution = min(aics, key=aics.get)
        params = fits[distribution]
        print(f'Selected: {distribution.name}')
        candidate_aics = {d.name: float(a) for d, a in sorted(aics.items(), key=lambda item: item[1])}
    else:
        # Fitting the distribution to the returns via MLE (returns shape params, then loc and scale)
        params = distribution.fit(data)
        candidate_aics = None

    # Creating a matrix to store the returns for each
    sims = np.zeros((n_sims, sim_length + 1))
    sims[:,0] = 1

    for i in range(n_sims):
        # Drawing returns from the fitted distribution
        ret_sample = distribution.rvs(*params, size=sim_length, random_state=rng)
        # Obtaining the single period accumulation factors
        acc_factor = ret_sample + 1
        # The cumulative returns over the simulation period
        value = np.cumprod(acc_factor)
        # Adding the simulated values to the simulation matrix
        sims[i, 1:] = value

    if not return_fit:
        return sims

    # Naming the fitted parameters: scipy orders them shapes, then loc and scale
    names = [n.strip() for n in (distribution.shapes or '').split(',') if n.strip()] + ['loc', 'scale']
    fit = {
        'Distribution': distribution.name,
        'Parameters': {name: float(value) for name, value in zip(names, params)},
        'AIC': float(aic(distribution, params)),
        'Candidates': candidate_aics,
    }
    return sims, fit



def regime_switching_monte_carlo(values=None, n_regimes=None, bootstrap=False, n_starts=10,
                                 sim_length=100, n_sims=1000, random_state=None, return_fit=False,
                                 *, prices=None):
    """Simulate future value paths from a Gaussian hidden Markov model of regimes.

    A hidden Markov model is fitted to one-period log returns, each hidden
    state (regime) having its own mean and volatility, e.g. calm and rising
    versus volatile and falling. Each path starts in a regime drawn from the
    model's probabilities for the last historical bar, moves between regimes
    by the fitted transition matrix, and draws each period's return from its
    current regime. Unlike the other two simulators, this keeps volatility
    clustering: high-volatility periods tend to follow each other.

    Parameters
    ----------
    values : pandas.Series or single-column pandas.DataFrame
        Historical prices or portfolio values, one row per bar, oldest first.
        Must be positive (log returns are taken). The bar size of `values` is
        the length of one simulated period.
    n_regimes : int or None, default None
        Number of hidden regimes. None fits 1, 2 and 3 regimes and uses the
        one with the lowest BIC.
    bootstrap : bool, default False
        How a return is drawn within a regime. False draws from the regime's
        fitted normal distribution. True resamples the historical returns
        whose most likely regime is that one, keeping their fat tails; a
        regime with no historical returns falls back to its normal.
    n_starts : int, default 10
        Random restarts per model fit. EM only finds a local optimum, so each
        model is fitted this many times and the highest log-likelihood kept.
    sim_length : int, default 100
        Number of future periods to simulate per path.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the restarts and the draws. An int or a
        Generator gives reproducible paths; None gives a fresh, unseeded
        Generator.
    return_fit : bool, default False
        If True, also return the fitted model; see Returns.
    prices : pandas.Series or single-column pandas.DataFrame, optional
        Deprecated alias for `values`, keyword only.

        .. deprecated:: 0.2.0
            `prices` will be removed in 1.0.0; use `values` instead.

    Returns
    -------
    sims : numpy.ndarray
        Array of shape ``(n_sims, sim_length + 1)``, one path per row. Column
        0 is the starting value, 1.0, so every value is a growth multiple of
        the last historical price.
    fit : dict
        Only if `return_fit` is True:

        ``"Regime Count"``
            Number of regimes used.
        ``"BIC"``
            dict of BIC by regime count: every count fitted when `n_regimes`
            is None, otherwise just `n_regimes`.
        ``"Log Likelihood"``
            Log-likelihood of the model used.
        ``"Regimes"``
            pandas.DataFrame, one row per regime ordered from lowest to
            highest volatility, with columns ``Mean`` and ``Volatility`` (of
            the one-period log return), ``Expected Duration`` (periods,
            ``1 / (1 - p_kk)``) and ``Current Probability`` (of being in that
            regime at the last historical bar).
        ``"Transition Matrix"``
            pandas.DataFrame of one-period transition probabilities, from
            the row regime to the column regime, in the same order.

    Raises
    ------
    ValueError
        If `n_regimes` is given and no restart produces a usable fit, or if
        `n_regimes` is None and none of 1, 2 or 3 regimes can be fitted.
    TypeError
        If neither or both of `values` and `prices` are given.

    Warns
    -----
    DeprecationWarning
        If `prices` is given.

    Notes
    -----
    Each regime is fitted by maximum likelihood (EM with no prior on the
    variances). A restart is rejected if any fitted parameter is non-finite
    or if a regime holds less than ``max(2, 1% of the history)`` of the
    expected occupancy, since such a regime's parameters are meaningless.

    With `n_regimes=None`, the BIC is ``-2 log L + p log n``, with
    ``p = (K - 1) + K(K - 1) + 2K`` free parameters for K regimes (initial
    probabilities, transitions, and a mean and variance per regime). Regime
    counts that cannot be fitted are skipped.

    Prints the BIC of each regime count (when selecting), then each regime's
    per-period mean, volatility and expected duration ``1 / (1 - p_kk)``,
    from lowest to highest volatility. The model numbers its regimes
    arbitrarily; ordering them by volatility only changes how they are
    reported, not the simulation.

    See Also
    --------
    nonparametric_monte_carlo : Resamples historical returns, no regimes.
    parametric_monte_carlo : Draws from one fitted distribution, no regimes.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.

    Examples
    --------
    >>> sims = regime_switching_monte_carlo(values, n_regimes=2, bootstrap=True,
    ...                                     random_state=42)
    >>> sims, fit = regime_switching_monte_carlo(values, random_state=42, return_fit=True)
    >>> fit["Regime Count"]
    3
    >>> fit["Regimes"]["Volatility"]  # calm to turbulent
    """
    values = _resolve_values(values, prices, 'regime_switching_monte_carlo')
    rng = np.random.default_rng(random_state)

    # Calculating log returns as a column vector (hmmlearn expects shape (n_obs, n_features))
    X = np.log(values).diff().dropna().to_numpy().reshape(-1, 1)

    def fit(K):
        # Fitting K regimes from several random starts and keeping the highest log-likelihood
        best, best_ll = None, -np.inf
        for seed in rng.integers(2**31 - 1, size=n_starts):
            # covars_prior=0: hmmlearn's default prior (1e-2) adds 0.01 to every variance
            # estimate, far larger than a daily return's variance (~1e-4), which inflated
            # regime volatilities and made the fit, and so the BIC, penalized rather than
            # maximum likelihood. Degenerate fits the prior guards against are rejected below.
            model = GaussianHMM(n_components=K, covariance_type='full', n_iter=1000, tol=1e-6,
                                covars_prior=0.0, random_state=int(seed))
            # Degenerate starts are common and are rejected below, so silence hmmlearn's warnings about them
            hmm_log = logging.getLogger('hmmlearn')
            level, hmm_log.level = hmm_log.level, logging.ERROR
            try:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    model.fit(X)
                    ll = model.score(X)
                    occupancy = model.predict_proba(X).sum(axis=0)
            except ValueError:
                continue
            finally:
                hmm_log.level = level
            # Rejecting starts that broke down: non-finite parameters, or a regime holding
            # (almost) none of the history, whose parameters are then meaningless
            params = (model.startprob_, model.transmat_, model.means_, model.covars_)
            if not (np.isfinite(ll) and all(np.isfinite(p).all() for p in params)):
                continue
            if occupancy.min() < max(2, 0.01 * len(X)):
                continue
            if ll > best_ll:
                best, best_ll = model, ll
        if best is None:
            raise ValueError(f'Could not fit {K} regimes to {len(X)} returns; try fewer regimes or more history')
        return best, best_ll

    def bic(K, log_lik):
        # BIC = -2*log-likelihood + n_params*log(n_obs). Free parameters for one series:
        # initial probs (K-1) + transitions K(K-1) + a mean and variance per regime
        n_params = (K - 1) + K * (K - 1) + 2 * K
        return -2 * log_lik + n_params * np.log(len(X))

    if n_regimes is None:
        # Scoring each regime count with BIC
        fits, bics = {}, {}
        for K in (1, 2, 3):
            try:
                fits[K] = fit(K)
            except ValueError:
                # Short or featureless histories may not support K distinct regimes
                print(f'{K} regime(s): no stable fit, skipped')
                continue
            bics[K] = bic(K, fits[K][1])
            print(f'{K} regime(s) BIC: {bics[K]:,.2f}')

        # Choosing the regime count with the lowest BIC
        n_regimes = min(bics, key=bics.get)
        model, log_lik = fits[n_regimes]
        print(f'Selected: {n_regimes} regime(s)')
    else:
        model, log_lik = fit(n_regimes)
        bics = {n_regimes: bic(n_regimes, log_lik)}

    # Each regime's per-period mean and volatility, how long it tends to last, and
    # how likely it is to be the current regime. The model numbers regimes arbitrarily,
    # so they are reported from calmest to most volatile; the simulation below uses
    # the model's own numbering.
    P = model.transmat_
    means, vols = model.means_[:, 0], np.sqrt(model.covars_[:, 0, 0])
    durations = np.array([np.inf if P[k, k] >= 1 else 1 / (1 - P[k, k]) for k in range(n_regimes)])
    current = model.predict_proba(X)[-1]
    order = np.argsort(vols, kind='stable')
    for rank, k in enumerate(order):
        print(f'Regime {rank}: mean {means[k]:+.4%}, vol {vols[k]:.4%}, '
              f'expected duration {durations[k]:.1f} periods')

    # Starting each path in a regime drawn from today's regime probabilities
    regime = rng.choice(n_regimes, size=n_sims, p=current)

    if bootstrap:
        # Historical returns grouped by their most likely regime
        labels = model.predict(X)
        pools = [X[labels == k, 0] for k in range(n_regimes)]

    # Simulating log returns one period at a time across all paths
    log_rets = np.empty((n_sims, sim_length))
    cum_P = np.cumsum(P, axis=1)
    for t in range(sim_length):
        # Markov transition: drawing each path's next regime from its current regime's row
        u = rng.random(n_sims)
        regime = np.minimum((u[:, None] > cum_P[regime]).sum(axis=1), n_regimes - 1)

        for k in range(n_regimes):
            idx = np.flatnonzero(regime == k)
            if idx.size == 0:
                continue
            if bootstrap and len(pools[k]) > 0:
                log_rets[idx, t] = rng.choice(pools[k], size=idx.size)
            else:
                # Parametric draw (also the fallback if no historical return was assigned to regime k)
                log_rets[idx, t] = rng.normal(means[k], vols[k], size=idx.size)

    # Creating a matrix to store the value of each path
    sims = np.ones((n_sims, sim_length + 1))
    # The cumulative returns over the simulation period (log returns add, so exponentiate the running sum)
    sims[:, 1:] = np.exp(np.cumsum(log_rets, axis=1))

    if not return_fit:
        return sims

    labels = pd.RangeIndex(n_regimes, name='Regime')
    regimes = pd.DataFrame({
        'Mean': means[order],
        'Volatility': vols[order],
        'Expected Duration': durations[order],
        'Current Probability': current[order],
    }, index=labels)
    fitted = {
        'Regime Count': int(n_regimes),
        'BIC': {K: float(b) for K, b in sorted(bics.items())},
        'Log Likelihood': float(log_lik),
        'Regimes': regimes,
        'Transition Matrix': pd.DataFrame(P[np.ix_(order, order)], index=labels, columns=labels),
    }
    return sims, fitted
