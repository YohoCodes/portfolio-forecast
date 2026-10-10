"""Gaussian hidden Markov model with class-specific emissions, fitted by EM.

Private module behind the regime-switching simulator: one regime chain
steps per mark, emissions are normal per (regime, class) cell, and the
transition estimate adds a sticky weight κ to staying in a regime. A short
regular bar (ρ_t = τ/Δ < 1) has its emission mean scaled by ρ_t and its
variance by ρ_t. `regime_switching_monte_carlo` (forecast/regime.py) is
the caller.

Flow
----
1. `fit_hmm` checks the input (`_validate`), then runs EM from `n_starts`
   random starts (`_init`, `_fit_one`) and keeps the best by log L.
2. Each EM iteration is an `_e_step` (scaled forward–backward through
   `_chain`, (E1)-(E5)) and an `_m_step` ((S1)-(S5)), until the log
   posterior J (H2) stops rising.
3. `n_params` and `score` give k (K1) and BIC or AIC (K2) for choosing the
   number of regimes; `log_likelihood` and `posteriors` evaluate fixed
   parameters.

Derivation
----------
The math below is plain text with Unicode symbols, as in the README; each
numbered equation is cited by the code that implements it, e.g.
``# (E3)``. The labels (H*), (E*), (S*) and (K*) number this derivation
only.

Model
~~~~~
Observations are log returns ℓ_1..ℓ_T, one per mark the chain runs on,
each with a known class c_t in {0 = regular, 1 = gap} (or a single class,
c_t = 0, when no classes are given), and a known length ratio
ρ_t = τ_t/Δ > 0: the return's length over the bar size, below 1 for a
short regular bar (a half-day's last bar) and 1 for full bars and every
gap, since a gap is one unit whatever its clock length (ρ_t = 1 throughout
when none is given). A hidden regime s_t in {1..S} follows a Markov chain
that steps once per mark, gaps included:

    Pr(s_1 = i) = π_i,      Pr(s_{t+1} = j | s_t = i) = P_ij.

Given the regime, the return is normal with the parameters of its
(regime, class) cell, location and scale stretched by ρ_t (drift grows with
the bar's length, variance too, so a bar of half the length has half the
mean and half the variance of a full one):

    ℓ_t | s_t = s  ~  N(ρ_t μ[s, c_t], ρ_t σ²[s, c_t]),
    log b_t(s) = −½ ( log(2π ρ_t σ²[s, c_t])
                      + (ℓ_t − ρ_t μ[s, c_t])² / (ρ_t σ²[s, c_t]) ).                    (H1)

So μ[s, c] and σ²[s, c] are per full unit (one full bar, one gap). The
parameters are θ = (π, P, μ, σ²); μ and σ² have shape (S, C) with C the
number of classes. With every ρ_t = 1, (H1) is the plain Gaussian HMM
(hmmlearn's GaussianHMM), term for term.

Prior. Each row of P has a sticky Dirichlet prior

    P_i·  ~  Dir(1, .., 1 + κ, .., 1)      (1 + κ at position i),

whose log density is κ log P_ii + const. EM maximises the log posterior

    J(θ) = log L(θ) + κ Σ_i log P_ii,                                                  (H2)

so κ = 0 gives exactly the maximum-likelihood estimate.

Likelihood and forward–backward (scaled, emissions in log space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Emission densities are kept in log space and shifted by their per-mark
maximum so that none underflows:

    m_t = max_s log b_t(s),      b̃_t(s) = exp(log b_t(s) − m_t)  in (0, 1].           (E1)

Write α_t(i) = p(ℓ_1..ℓ_t, s_t = i) / exp(m_1 + … + m_t) and
M_t = P diag(b̃_t) (an S × S matrix, t = 2..T). The forward recursion is

    α_1 = π ⊙ b̃_1,      α_t = α_{t−1} M_t      (row vectors),                           (E2)

so α_T = α_1 M_2 ⋯ M_T and the exact data log-likelihood is

    log L(θ) = log Σ_i α_T(i) + Σ_t m_t.                                               (E3)

The backward vectors are β_T = 1 and β_t = M_{t+1} β_{t+1}, i.e.
β_t = M_{t+1} ⋯ M_T 1, so that α_t(i) β_t(i) is proportional to
p(ℓ_1..ℓ_T, s_t = i).

Only the direction of each α_t and β_t is needed below, so each is stored
normalised to sum 1 ("scaled"): α̂_t = α_t / Σ α_t is the filtered
probability Pr(s_t = i | ℓ_1..ℓ_t), and α̂_T is ``filtered_last``. The
scale factors are accumulated in logs, which gives (E3) without underflow.
Because (E2) is a product of matrices, the chain is evaluated in blocks of
about √T marks: the running products inside every block are formed side by
side (each rescaled to a maximum entry of 1, its log scale kept), then the
blocks are joined in order. This is the same product as the mark-by-mark
recursion, regrouped by associativity; it only removes per-mark Python
overhead. β is the same computation on the reversed sequence of transposed
matrices: as a row vector, β_t is 1 times the transposes of M_T, …,
M_{t+1}, in that order.

E-step
~~~~~~
Posterior regime probabilities and pairwise transition probabilities:

    γ_t(i) = α̂_t(i) β_t(i) / Σ_s α̂_t(s) β_t(s),                                         (E4)
    ξ_t(i, j) = α̂_t(i) M_{t+1}(i, j) β_{t+1}(j)
                / Σ_{i′,j′} α̂_t(i′) M_{t+1}(i′, j′) β_{t+1}(j′),     t = 1..T−1.         (E5)

Each is normalised per t, so the arbitrary scale of the stored vectors
cancels.

M-step
~~~~~~
Maximising the expected complete-data log posterior (Q-function plus the
log prior) gives closed forms. With expected cell occupancy

    N[s, c] = Σ_{t: c_t = c} γ_t(s),                                                   (S1)

the part of the expected complete-data log posterior that holds cell (s, c)
is, from (H1),

    Q[s, c] = Σ_{t: c_t = c} γ_t(s) ( −½ log σ²[s, c]
              − (ℓ_t − ρ_t μ[s, c])² / (2 ρ_t σ²[s, c]) ) + const.

Setting ∂Q/∂μ = Σ γ_t(s) (ℓ_t − ρ_t μ) / σ² = 0 and
∂Q/∂σ² = −N / (2σ²) + Σ γ_t(s) (ℓ_t − ρ_t μ)² / (2 ρ_t σ⁴) = 0
gives the emission updates, class by class (only marks of class c inform
cell (s, c)):

    μ[s, c]  = Σ_{t: c_t = c} γ_t(s) ℓ_t / Σ_{t: c_t = c} γ_t(s) ρ_t,                  (S2)
    σ²[s, c] = max( Σ_{t: c_t = c} γ_t(s) (ℓ_t − ρ_t μ[s, c])² / ρ_t
                    / N[s, c], 1e-12 ).                                                (S3)

With every ρ_t = 1 these are the weighted sample mean and variance. Each
term (ℓ_t − ρ_t μ)² / ρ_t has expectation σ² whatever ρ_t, so a short bar
counts as one full observation of the variance, but only as ρ_t of an
observation of the mean.

The floor 1e-12 keeps the density finite; a start whose variance reaches
it is rejected as collapsed (see below). The transition update maximises
Σ_j (Ξ_ij + κ 1[i = j]) log P_ij subject to Σ_j P_ij = 1, with
Ξ_ij = Σ_t ξ_t(i, j), by a Lagrange multiplier:

    P_ij = (Ξ_ij + κ 1[i = j]) / (Σ_j′ Ξ_ij′ + κ),                                      (S4)

and the initial distribution is the first posterior,

    π_i = γ_1(i).                                                                       (S5)

EM never decreases J (H2); iteration stops when J rises by less than
``tol`` (absolute, as in hmmlearn) or after ``max_iter`` M-steps. The
returned parameters are those at which the last E-step was run, so
``log_likelihood``, ``gamma`` and ``filtered_last`` belong to them.
``log_likelihood`` is log L (E3), without the prior term: under κ > 0 it is
the data log-likelihood at the MAP estimate.

Starts and degenerate fits
~~~~~~~~~~~~~~~~~~~~~~~~~~
Each start seeds the means k-means++ style on ℓ, class by class: the first
seed is a uniform draw from the class's returns, each further seed a draw
with probability proportional to the squared distance to the nearest seed
so far. Every cell starts at its class's variance, π and P start uniform.
The best of ``n_starts`` starts by log L is kept. A start is rejected as
degenerate if any parameter or log L is non-finite, if any (regime, class)
cell holds an expected occupancy N[s, c] below max(2, 1% of that class's
marks) (an empty regime), or if any variance sits at the floor (a collapsed
variance).

Parameter count and criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Free parameters: S − 1 for π, S(S − 1) for P, and a mean and a variance per
(regime, class) cell, 2SC:

    k = (S − 1) + S(S − 1) + 2SC.                                                       (K1)

gap_model "regime": two classes, k = (S − 1) + S(S − 1) + 4S.
gap_model "independent": the caller fits the HMM to regular marks only, one
class, k = (S − 1) + S(S − 1) + 2S; the separate gap normal adds 2, which
is the same for every S and so cancels in regime-count selection. With n
the observations the likelihood covers,

    BIC = −2 log L + k ln n,      AIC = −2 log L + 2k.                                  (K2)

References
----------
Hamilton, J. D. (1989). A new approach to the economic analysis of
nonstationary time series and the business cycle. Econometrica 57(2),
357-384. (Regime-switching returns.)

Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). Maximum likelihood
from incomplete data via the EM algorithm. Journal of the Royal Statistical
Society B 39(1), 1-38. (EM never decreases the objective.)

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
applications in speech recognition. Proceedings of the IEEE 77(2), 257-286.
(Scaled forward–backward, γ and ξ.)

Arthur, D. and Vassilvitskii, S. (2007). k-means++: the advantages of
careful seeding. Proceedings of SODA 2007, 1027-1035. (Seeding the means.)
"""

from dataclasses import dataclass

import numpy as np

VAR_FLOOR = 1e-12
_CRITERIA = ("AIC", "BIC")


@dataclass(frozen=True)
class HMMFit:
    """Fitted HMM parameters and posteriors, see the module docstring.

    Attributes
    ----------
    pi : numpy.ndarray
        Initial regime distribution π, shape ``(S,)``.
    P : numpy.ndarray
        Transition matrix P, shape ``(S, S)``, rows sum to 1.
    mu, var : numpy.ndarray
        Emission mean μ and variance σ² of each (regime, class) cell, shape
        ``(S, C)``; class 0 is regular, class 1 gap.
    log_likelihood : float
        Data log-likelihood log L at these parameters (E3), without the
        prior term.
    gamma : numpy.ndarray
        Posterior regime probabilities γ, shape ``(T, S)`` (E4).
    filtered_last : numpy.ndarray
        Filtered Pr(s_T | ℓ_1..ℓ_T) at the last mark, shape ``(S,)``.
    n_params : int
        Free parameter count k (K1).
    converged : bool
        True if EM stopped on ``tol`` rather than ``max_iter``.
    """

    pi: np.ndarray
    P: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    log_likelihood: float
    gamma: np.ndarray
    filtered_last: np.ndarray
    n_params: int
    converged: bool


def n_params(n_states, n_classes):
    """Return the free parameter count k of an HMM with `n_states` regimes and `n_classes` classes (K1)."""
    S, C = int(n_states), int(n_classes)
    return (S - 1) + S * (S - 1) + 2 * S * C


def score(fit, n, criterion):
    """Return the information criterion of `fit` on `n` observations (K2).

    Parameters
    ----------
    fit : HMMFit
    n : int
        Number of observations the HMM was fitted to.
    criterion : {"AIC", "BIC"}
        Case-insensitive.

    Returns
    -------
    float
        ``-2 log L + k ln n`` (BIC) or ``-2 log L + 2 k`` (AIC); lower is better.

    Raises
    ------
    ValueError
        If `criterion` is neither AIC nor BIC.
    """
    name = str(criterion).upper()
    if name not in _CRITERIA:
        raise ValueError(f'criterion must be "AIC" or "BIC" (any case), got {criterion!r}')
    penalty = np.log(n) if name == "BIC" else 2.0
    return -2.0 * fit.log_likelihood + fit.n_params * penalty


def log_likelihood(ell, pi, P, mu, var, classes=None, rho=None):
    """Return the data log-likelihood log L (E3) at fixed parameters.

    `mu` and `var` have shape ``(S, C)``; `classes` and `rho` as in `fit_hmm`.
    """
    ell, cls, rho, _ = _validate(ell, classes, rho)
    return _e_step(ell, cls, *_as_params(pi, P, mu, var), rho=rho)[0]


def posteriors(ell, pi, P, mu, var, classes=None, rho=None):
    """Return the posterior regime probabilities γ (E4) at fixed parameters, shape ``(T, S)``."""
    ell, cls, rho, _ = _validate(ell, classes, rho)
    return _e_step(ell, cls, *_as_params(pi, P, mu, var), rho=rho)[1]


def fit_hmm(ell, classes=None, rho=None, *, n_states, kappa=0.0, n_starts=10, random_state=None, tol=1e-6,
            max_iter=1000):
    """Fit a Gaussian HMM with class-specific emissions by EM, best of several starts.

    Parameters
    ----------
    ell : array_like
        Log returns ℓ, one per mark, oldest first.
    classes : array_like of {0, 1} or None, default None
        Class of each return: 0 regular, 1 gap. None fits a single class,
        e.g. regular marks only under gap_model "independent".
    rho : array_like of float or None, default None
        Length of each return over the bar size, ρ = τ/Δ: below 1 for a
        short regular bar, 1 for full bars and every gap (one unit whatever
        its clock length). Emission
        mean and variance scale with it (H1). None means 1 for every return.
    n_states : int
        Number of regimes S, keyword only.
    kappa : float, default 0.0
        Sticky Dirichlet prior weight κ on staying in a regime (S4); 0 is
        maximum likelihood.
    n_starts : int, default 10
        Random starts; the one with the highest log-likelihood is kept.
    random_state : int, numpy.random.Generator or None, default None
        Seed for the starts; a fixed int gives a deterministic fit.
    tol : float, default 1e-6
        Stop when the log posterior J (H2) rises by less than this.
    max_iter : int, default 1000
        Maximum EM iterations per start.

    Returns
    -------
    HMMFit

    Raises
    ------
    ValueError
        If the input is invalid (including a `rho` that is not positive, or
        a gap with `rho` other than 1), if a class has fewer than 2S returns or
        there are fewer returns than parameters, or if every start is
        degenerate (e.g. constant input).
    """
    ell, cls, rho, C = _validate(ell, classes, rho)
    S = int(n_states)
    if S < 1:
        raise ValueError(f"n_states must be at least 1, got {n_states!r}")
    if not (np.isfinite(kappa) and kappa >= 0):
        raise ValueError(f"kappa must be a finite number >= 0, got {kappa!r}")
    if int(n_starts) < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts!r}")
    n, k = len(ell), n_params(S, C)
    counts = np.bincount(cls, minlength=C)
    names = ("regular ", "gap ") if C == 2 else ("",)
    for c in range(C):
        # Each (regime, class) cell needs at least two returns for a variance
        if counts[c] < 2 * S:
            raise ValueError(f"{S} regimes need at least {2 * S} {names[c]}returns, got {counts[c]}; "
                             f"use fewer regimes or more history")
    if n < k:
        raise ValueError(f"{S} regimes have {k} parameters but there are only {n} returns; "
                         f"use fewer regimes or more history")

    # Run EM from each random start and keep the one with the highest log L
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(int(n_starts)):
        # One EM run from a k-means++ start; None if it broke down or ended degenerate
        fit = _fit_one(ell, cls, rho, C, S, float(kappa), _init(ell, cls, C, S, rng), tol, int(max_iter), counts)
        if fit is not None and (best is None or fit.log_likelihood > best.log_likelihood):
            best = fit
    if best is None:
        raise ValueError(f"Could not fit {S} regimes to {n} returns: every start was degenerate "
                         f"(an empty regime or a collapsed variance); use fewer regimes or more history")
    return best


def _validate(ell, classes, rho=None):
    """Return float returns, integer classes, float length ratios ρ and the class count C (1 or 2)."""
    ell = np.asarray(ell, dtype=float)
    if ell.ndim != 1 or len(ell) == 0:
        raise ValueError(f"ell must be a non-empty 1-D array, got shape {ell.shape}")
    if not np.isfinite(ell).all():
        raise ValueError(f"ell has {np.count_nonzero(~np.isfinite(ell))} non-finite values")
    if rho is None:
        rho = np.ones(len(ell))
    else:
        rho = np.asarray(rho, dtype=float)
        if rho.shape != ell.shape:
            raise ValueError(f"rho has shape {rho.shape}, ell has {ell.shape}; give one ratio per return")
        if not (np.isfinite(rho).all() and (rho > 0).all()):
            raise ValueError("rho must hold finite ratios tau/Delta > 0")
    if classes is None:
        return ell, np.zeros(len(ell), dtype=np.intp), rho, 1
    cls = np.asarray(classes)
    if cls.shape != ell.shape:
        raise ValueError(f"classes has shape {cls.shape}, ell has {ell.shape}; give one class per return")
    if not np.isin(cls, (0, 1)).all():
        raise ValueError("classes must hold only 0 (regular) and 1 (gap)")
    # A gap is one unit whatever its clock length, so it is never rescaled
    if (rho[cls == 1] != 1).any():
        raise ValueError("rho must be 1 for every gap (class 1): a gap is one unit whatever its length")
    return ell, cls.astype(np.intp), rho, 2


def _as_params(pi, P, mu, var):
    """Return the parameters as float arrays, mu and var 2-D ``(S, C)``."""
    mu, var = np.asarray(mu, dtype=float), np.asarray(var, dtype=float)
    if mu.ndim == 1:
        mu, var = mu[:, None], var[:, None]
    return np.asarray(pi, dtype=float), np.asarray(P, dtype=float), mu, var


def _init(ell, cls, C, S, rng):
    """Return one random start: k-means++ seeded means per class, class variances, uniform π and P."""
    mu = np.empty((S, C))
    var = np.empty((S, C))
    for c in range(C):
        # k-means++ means: the first seed uniform, each next drawn with probability
        # proportional to the squared distance to its nearest seed
        x = ell[cls == c]
        seeds = [x[rng.integers(len(x))]]
        for _ in range(S - 1):
            d2 = np.min((x[:, None] - np.array(seeds)[None, :]) ** 2, axis=1)
            total = d2.sum()
            # Constant data: every distance is zero, fall back to a uniform draw
            p = d2 / total if total > 0 else None
            seeds.append(x[rng.choice(len(x), p=p)])
        mu[:, c] = seeds
        # Every cell of the class starts at the class's variance
        var[:, c] = max(x.var(), VAR_FLOOR)
    return np.full(S, 1.0 / S), np.full((S, S), 1.0 / S), mu, var


def _fit_one(ell, cls, rho, C, S, kappa, params, tol, max_iter, counts):
    """Run EM from one start; return its HMMFit, or None if it breaks down or ends degenerate."""
    pi, P, mu, var = params
    J_prev = -np.inf
    converged = False
    with np.errstate(all="ignore"):
        for it in range(max_iter + 1):
            # E-step at the current parameters: log L, γ, Ξ and the last filtered vector
            ll, gamma, xi_sum, last = _e_step(ell, cls, pi, P, mu, var, rho=rho)
            if not np.isfinite(ll):
                return None
            # (H2): the prior term only exists for κ > 0 (avoids 0 · log 0)
            J = ll + kappa * np.log(np.diag(P)).sum() if kappa > 0 else ll
            if J - J_prev < tol:
                converged = True
                break
            J_prev = J
            if it == max_iter:
                break
            # M-step: the closed-form π, P, μ and σ² given these posteriors
            pi, P, mu, var = _m_step(ell, cls, C, gamma, xi_sum, kappa, rho=rho)

    # Reject degenerate fits: non-finite, a near-empty cell, a collapsed variance
    if not all(np.isfinite(a).all() for a in (pi, P, mu, var, gamma)):
        return None
    # Expected occupancy N[s, c] of every cell (S1), against max(2, 1% of the class's marks)
    occupancy = np.stack([gamma[cls == c].sum(axis=0) for c in range(C)], axis=1)
    if (occupancy < np.maximum(2.0, 0.01 * counts)[None, :]).any():
        return None
    if (var <= VAR_FLOOR).any():
        return None
    return HMMFit(pi=pi, P=P, mu=mu, var=var, log_likelihood=float(ll), gamma=gamma, filtered_last=last,
                  n_params=n_params(S, C), converged=converged)


def _e_step(ell, cls, pi, P, mu, var, rho=None):
    """Return log L (E3), γ (E4), the sum over t of ξ (E5) and the last filtered vector.

    `rho` holds each mark's length ratio (H1); None means 1 everywhere.
    """
    if rho is None:
        rho = np.ones(len(ell))
    # (H1): log density of every mark under every regime, shape (T, S), from its class's
    # cell stretched by ρ_t; with ρ = 1 the products are exact, so full bars are untouched
    m_c, v_c = mu[:, cls].T * rho[:, None], var[:, cls].T * rho[:, None]
    log_b = -0.5 * (np.log(2 * np.pi * v_c) + (ell[:, None] - m_c) ** 2 / v_c)
    # (E1): densities shifted by their per-mark maximum, so none underflows
    m = log_b.max(axis=1)
    bb = np.exp(log_b - m[:, None])
    # (E2), (E3): M_t = P diag(bb_t) for t = 2..T, the scaled forward pass and log L
    M = P[None, :, :] * bb[1:, None, :]
    ahat, log_norm = _chain(pi * bb[0], M)
    ll = log_norm + m.sum()
    # Scaled backward pass: the same chain run over the reversed transposes
    beta = _chain(np.ones(len(pi)), np.swapaxes(M[::-1], 1, 2))[0][::-1]
    # (E4): posterior regime probabilities per mark
    gamma = ahat * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    # (E5): posterior regime pairs per step t to t + 1; their sum over t is Ξ
    xi = ahat[:-1, :, None] * M * beta[1:, None, :]
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    return ll, gamma, xi.sum(axis=0), ahat[-1]


def _m_step(ell, cls, C, gamma, xi_sum, kappa, rho=None):
    """Return the closed-form updates (S1)-(S5): π, P, μ, σ².

    `rho` holds each mark's length ratio (H1); None means 1 everywhere.
    """
    if rho is None:
        rho = np.ones(len(ell))
    S = gamma.shape[1]
    mu = np.empty((S, C))
    var = np.empty((S, C))
    for c in range(C):
        # Only marks of class c inform the cells (s, c); (S1) their expected occupancy
        g, x, r = gamma[cls == c], ell[cls == c], rho[cls == c][:, None]
        N = g.sum(axis=0)
        # (S2): the mean is per full unit, so a short bar weighs ρ_t in the denominator
        mu[:, c] = g.T @ x / (g * r).sum(axis=0)
        # (S3): the variance per full unit, floored
        var[:, c] = np.maximum((g * (x[:, None] - r * mu[:, c]) ** 2 / r).sum(axis=0) / N, VAR_FLOOR)
    # (S4): expected transitions plus κ pretend stays, normalized per row
    counts = xi_sum + kappa * np.eye(S)
    P = counts / counts.sum(axis=1, keepdims=True)
    # (S5): the initial distribution is the first posterior
    return gamma[0].copy(), P, mu, var


def _chain(v0, M):
    """Return the normalised prefix products of (E2): row t is v0 M_1 ... M_t / its sum, t = 0..K.

    Also returns log of the sum of v0 M_1 ... M_K. Computed in blocks of
    about √K matrices (module docstring): running products inside all
    blocks side by side, each rescaled to a maximum entry of 1 with its log
    scale kept, then the blocks joined in order.
    """
    K, S = M.shape[0], len(v0)
    out = np.empty((K + 1, S))
    # Row 0 is v0 normalized; its sum starts the log scale
    z = v0.sum()
    out[0] = v0 / z
    log_norm = np.log(z)
    if K == 0:
        return out, log_norm
    # nb blocks of b, about √K, matrices each
    b = int(np.ceil(np.sqrt(K)))
    nb = -(-K // b)
    # Identity matrices pad the last block; they change neither direction nor scale
    pad = np.broadcast_to(np.eye(S), (nb * b - K, S, S))
    Mb = np.concatenate([M, pad]).reshape(nb, b, S, S)

    # Running products inside every block at once, each rescaled to a maximum entry of 1;
    # log_q accumulates each block's log scale
    Q = np.empty_like(Mb)
    log_q = np.zeros(nb)
    q = Mb[:, 0]
    for r in range(b):
        if r:
            q = Q[:, r - 1] @ Mb[:, r]
        scale = q.max(axis=(1, 2))
        Q[:, r] = q / scale[:, None, None]
        log_q += np.log(scale)

    # Join the blocks in order: each block's start vector, and the log scale so far
    starts = np.empty((nb, S))
    v = out[0]
    for i in range(nb):
        starts[i] = v
        w = v @ Q[i, -1]
        z = w.sum()
        log_norm += np.log(z) + log_q[i]
        v = w / z

    # Every row: its block's start vector times its running product, normalized
    inner = np.einsum("bi,brij->brj", starts, Q).reshape(nb * b, S)[:K]
    out[1:] = inner / inner.sum(axis=1, keepdims=True)
    return out, log_norm
