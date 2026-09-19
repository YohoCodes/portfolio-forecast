import matplotlib
import numpy as np
import pandas as pd
import pytest

# Render figures off-screen so the plotting tests run headless
matplotlib.use("Agg")


@pytest.fixture
def prices():
    """400 daily closes of a random walk: a date-indexed, one-column frame."""
    rng = np.random.default_rng(0)
    values = 100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, 400)))
    return pd.DataFrame({"X": values}, index=pd.bdate_range("2024-01-02", periods=400))


def compounding(index, step_return, start=100.0):
    """A series that grows by exactly `step_return` every bar, so every
    return, CAGR and yearly figure computed from it is known in closed form."""
    return pd.Series(start * (1 + step_return) ** np.arange(len(index)), index=index)
