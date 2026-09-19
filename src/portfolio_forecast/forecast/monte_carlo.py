import logging
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy import stats


def nonparametric_monte_carlo(prices, sim_length=100, n_sims=1000, random_state=None):
    """Simulate future value paths by resampling historical returns.

    Each path draws `sim_length` one-period returns, with replacement, from
    the history in `prices` (a bootstrap) and compounds them. No distribution
    is assumed, so fat tails and skew in the history carry over, but every
    draw is independent: volatility clustering and autocorrelation do not.

    Parameters
    ----------
    prices : pandas.Series or single-column pandas.DataFrame
        Historical prices or portfolio values, one row per bar, oldest first.
        The bar size of `prices` is the length of one simulated period.
    sim_length : int, default 100
        Number of future periods to simulate per path.
    n_sims : int, default 1000
        Number of simulated paths.
    random_state : int, numpy.random.Generator or None, default None
        Seed or Generator for the draws. An int or a Generator gives
        reproducible paths; None gives a fresh, unseeded Generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sims, sim_length + 1)``, one path per row. Column
        0 is the starting value, 1.0, so every value is a growth multiple of
        the last historical price.

    Raises
    ------
    ValueError
        If `prices` has more than one column.

    See Also
    --------
    parametric_monte_carlo : Draws returns from a fitted distribution instead.
    regime_switching_monte_carlo : Lets volatility switch between regimes.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.
    portfolio_forecast.plotting.plot_simulated_paths : Fan chart of the paths.

    Examples
    --------
    >>> sims = nonparametric_monte_carlo(prices, sim_length=100, n_sims=1000,
    ...                                  random_state=42)
    >>> sims.shape
    (1000, 101)
    """
    rng = np.random.default_rng(random_state)

    # Calculating the percent change between prices
    returns = prices.pct_change().dropna()

    # Creating a matrix to store the returns for each
    sims = np.zeros((n_sims, sim_length + 1))
    sims[:,0] = 1

    for i in range(n_sims):
        # Sampling from the prices with replacement
        ret_sample = returns.sample(n=sim_length, replace=True, random_state=rng).reset_index(drop=True)
        # Obtaining the single period accumulation factors
        acc_factor = ret_sample.to_numpy().flatten() + 1  # ensure 1D array, just in case
        # The cumulative returns over the simulation period
        value = np.cumprod(acc_factor)
        # Adding the simulated values to the simulation matrix
        sims[i, 1:] = value  # value should be shape (sim_length,)

    return sims



def parametric_monte_carlo(prices, distribution=None, sim_length=100, n_sims=1000, random_state=None):
    """Simulate future value paths from a distribution fitted to historical returns.

    One-period returns are fitted by maximum likelihood to a scipy.stats
    distribution, and each path compounds `sim_length` independent draws from
    it. With `distribution=None`, a normal, a Student's t and a Johnson SU are
    all fitted and the one with the lowest AIC is used.

    Parameters
    ----------
    prices : pandas.Series or single-column pandas.DataFrame
        Historical prices or portfolio values, one row per bar, oldest first.
        The bar size of `prices` is the length of one simulated period.
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

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sims, sim_length + 1)``, one path per row. Column
        0 is the starting value, 1.0, so every value is a growth multiple of
        the last historical price.

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
    >>> sims = parametric_monte_carlo(prices, distribution=stats.t,
    ...                               random_state=42)
    """
    rng = np.random.default_rng(random_state)

    # Calculating the percent change between prices
    returns = prices.pct_change().dropna()

    data = returns.to_numpy().flatten()

    if distribution is None:
        # Fitting each candidate via MLE and scoring it with AIC = 2k - 2*log-likelihood
        candidates = [stats.norm, stats.t, stats.johnsonsu]
        fits, aics = {}, {}
        for dist in candidates:
            dist_params = dist.fit(data)
            fits[dist] = dist_params
            log_lik = np.sum(dist.logpdf(data, *dist_params))
            aics[dist] = 2 * len(dist_params) - 2 * log_lik
            print(f'{dist.name:>10} AIC: {aics[dist]:,.2f}')

        # Choosing the distribution with the lowest AIC
        distribution = min(aics, key=aics.get)
        params = fits[distribution]
        print(f'Selected: {distribution.name}')
    else:
        # Fitting the distribution to the returns via MLE (returns shape params, then loc and scale)
        params = distribution.fit(data)

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

    return sims



def regime_switching_monte_carlo(prices, n_regimes=None, bootstrap=False, n_starts=10,
                                 sim_length=100, n_sims=1000, random_state=None):
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
    prices : pandas.Series or single-column pandas.DataFrame
        Historical prices or portfolio values, one row per bar, oldest first.
        Must be positive (log returns are taken). The bar size of `prices` is
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

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sims, sim_length + 1)``, one path per row. Column
        0 is the starting value, 1.0, so every value is a growth multiple of
        the last historical price.

    Raises
    ------
    ValueError
        If `n_regimes` is given and no restart produces a usable fit, or if
        `n_regimes` is None and none of 1, 2 or 3 regimes can be fitted.

    Notes
    -----
    A restart is rejected if any fitted parameter is non-finite or if a
    regime holds less than ``max(2, 1% of the history)`` of the expected
    occupancy, since such a regime's parameters are meaningless.

    With `n_regimes=None`, the BIC is ``-2 log L + p log n``, with
    ``p = (K - 1) + K(K - 1) + 2K`` free parameters for K regimes (initial
    probabilities, transitions, and a mean and variance per regime). Regime
    counts that cannot be fitted are skipped.

    Prints the BIC of each regime count (when selecting), then each regime's
    per-period mean, volatility and expected duration ``1 / (1 - p_kk)``.

    See Also
    --------
    nonparametric_monte_carlo : Resamples historical returns, no regimes.
    parametric_monte_carlo : Draws from one fitted distribution, no regimes.
    portfolio_forecast.performance.statistical_report : Summarizes the paths.

    Examples
    --------
    >>> sims = regime_switching_monte_carlo(prices, n_regimes=2, bootstrap=True,
    ...                                     random_state=42)
    """
    rng = np.random.default_rng(random_state)

    # Calculating log returns as a column vector (hmmlearn expects shape (n_obs, n_features))
    X = np.log(prices).diff().dropna().to_numpy().reshape(-1, 1)

    def fit(K):
        # Fitting K regimes from several random starts and keeping the highest log-likelihood
        best, best_ll = None, -np.inf
        for seed in rng.integers(2**31 - 1, size=n_starts):
            model = GaussianHMM(n_components=K, covariance_type='full', n_iter=1000, tol=1e-6,
                                random_state=int(seed))
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

    if n_regimes is None:
        # Scoring each regime count with BIC = -2*log-likelihood + n_params*log(n_obs)
        # Free parameters for one series: initial probs (K-1) + transitions K(K-1) + a mean and variance per regime
        fits, bics = {}, {}
        for K in (1, 2, 3):
            try:
                fits[K] = fit(K)
            except ValueError:
                # Short or featureless histories may not support K distinct regimes
                print(f'{K} regime(s): no stable fit, skipped')
                continue
            n_params = (K - 1) + K * (K - 1) + 2 * K
            bics[K] = -2 * fits[K][1] + n_params * np.log(len(X))
            print(f'{K} regime(s) BIC: {bics[K]:,.2f}')

        # Choosing the regime count with the lowest BIC
        n_regimes = min(bics, key=bics.get)
        model = fits[n_regimes][0]
        print(f'Selected: {n_regimes} regime(s)')
    else:
        model = fit(n_regimes)[0]

    # Each regime's per-period mean and volatility, and how long it tends to last
    P = model.transmat_
    means, vols = model.means_[:, 0], np.sqrt(model.covars_[:, 0, 0])
    for k in range(n_regimes):
        duration = np.inf if P[k, k] >= 1 else 1 / (1 - P[k, k])
        print(f'Regime {k}: mean {means[k]:+.4%}, vol {vols[k]:.4%}, expected duration {duration:.1f} periods')

    # Starting each path in a regime drawn from today's regime probabilities
    regime = rng.choice(n_regimes, size=n_sims, p=model.predict_proba(X)[-1])

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

    return sims
