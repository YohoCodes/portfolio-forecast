"""Result types shared by the backtest, the simulators and the future layout.

Each is a NamedTuple, so it unpacks positionally and its fields read by
name. Where each comes from:

1. `simulate_buy_and_hold` returns a `BacktestResult`: the values at every
   mark and the class of every return.
2. `next_trading_dates` returns a `FutureLayout`: the marks that continue a
   history, with each return's class and length.
3. The Monte Carlo simulators return a `SimulationResult`: paths on that
   layout, plus the fitted model.

`BacktestResult` has three fields on purpose: 0.3.0 code that unpacked
``values, dates = simulate_buy_and_hold(...)`` fails loudly instead of
silently binding a different meaning.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd


class BacktestResult(NamedTuple):
    """The value series of a backtest, with the class of every return.

    Attributes
    ----------
    values : pandas.Series
        Portfolio value at each mark: the book at the last prices known
        then, labelled with the moment it is true (a bar's value at its
        close), on the exchange's clock.
    dates : pandas.DatetimeIndex
        The marks, the same as ``values.index``.
    classes : pandas.Series
        Class of each return, indexed by the mark where the return ends
        (length ``len(values) - 1``): ``"gap"`` (last close to the next
        open), ``"regular"`` (a bar) or ``"irregular"`` (a late open
        mark, a stale held symbol or a missing scheduled bar). Daily and
        longer data have no gaps: every return is ``"regular"`` unless a
        held symbol is stale or a scheduled bar is missing.
    """

    values: pd.Series
    dates: pd.DatetimeIndex
    classes: pd.Series


class SimulationResult(NamedTuple):
    """Simulated paths from a Monte Carlo simulator.

    Attributes
    ----------
    paths : numpy.ndarray
        Shape ``(n_sims, n_marks + 1)``. Column 0 is 1.0: the paths are growth
        multiples of the last historical value.
    dates : pandas.DatetimeIndex or None
        Labels of the future marks, from `next_trading_dates`; None for
        plain-array input.
    classes : numpy.ndarray or None
        Class of the return ending at each future mark; None for
        plain-array input.
    fit : dict or None
        The fitted model, when the simulator fits one.
    """

    paths: np.ndarray
    dates: pd.DatetimeIndex | None
    classes: np.ndarray | None
    fit: dict | None


class FutureLayout(NamedTuple):
    """The future marks that continue a history on its grid and calendar.

    Attributes
    ----------
    dates : pandas.DatetimeIndex
        The future marks, oldest first, excluding the last historical mark.
    classes : numpy.ndarray
        Class of the return ending at each future mark: ``"gap"`` at a
        session's open mark, ``"regular"`` otherwise.
    tau : numpy.ndarray
        Length of the return ending at each mark as a fraction of the bar
        grid, τ/Δ: 1.0 for full bars and for gaps (a gap is one unit whatever
        its clock length), less than 1.0 for a short bar.
    """

    dates: pd.DatetimeIndex
    classes: np.ndarray
    tau: np.ndarray


# Field docstrings, so help() and IDEs show what each field holds
BacktestResult.values.__doc__ = "Portfolio value at each mark, labelled with the moment it is true."
BacktestResult.dates.__doc__ = "The marks, the same as values.index."
BacktestResult.classes.__doc__ = (
    "Class of each return ('gap', 'regular', 'irregular'), indexed by the mark where it ends."
)
SimulationResult.paths.__doc__ = "Simulated growth multiples, shape (n_sims, n_marks + 1); column 0 is 1.0."
SimulationResult.dates.__doc__ = "Future mark labels from next_trading_dates, or None for plain-array input."
SimulationResult.classes.__doc__ = "Class of the return ending at each future mark, or None for plain-array input."
SimulationResult.fit.__doc__ = "The fitted model, or None."
FutureLayout.dates.__doc__ = "The future marks, excluding the last historical mark."
FutureLayout.classes.__doc__ = "Class of the return ending at each future mark ('gap' or 'regular')."
FutureLayout.tau.__doc__ = "Length of the return ending at each mark relative to the bar grid (1.0 = full bar or gap)."
