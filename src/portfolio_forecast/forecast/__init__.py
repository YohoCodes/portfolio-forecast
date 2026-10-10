"""Monte Carlo simulators of future value paths.

Three simulators share one path (`_common`) and differ in how a
future mark's log return is drawn:

1. `nonparametric_monte_carlo` (`bootstrap.py`) resamples history per class.
2. `parametric_monte_carlo` (`parametric.py`) draws from a distribution
   fitted per class.
3. `regime_switching_monte_carlo` (`regime.py`) draws from a hidden Markov
   model of regimes, fitted by the EM in `_hmm.py`.

Each returns a `SimulationResult` on the future marks of
`next_trading_dates`.
"""
from .bootstrap import nonparametric_monte_carlo
from .parametric import parametric_monte_carlo
from .regime import regime_switching_monte_carlo

__all__ = ["nonparametric_monte_carlo", "parametric_monte_carlo", "regime_switching_monte_carlo"]
