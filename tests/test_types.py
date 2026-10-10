import numpy as np
import pandas as pd
import pytest

import portfolio_forecast
from portfolio_forecast import BacktestResult, FutureLayout, SimulationResult, types


@pytest.mark.parametrize("cls, fields", [
    (BacktestResult, ("values", "dates", "classes")),
    (SimulationResult, ("paths", "dates", "classes", "fit")),
    (FutureLayout, ("dates", "classes", "tau")),
])
def test_field_names_and_order(cls, fields):
    assert cls._fields == fields
    for name in fields:
        assert getattr(cls, name).__doc__


def test_re_exported_from_the_package_root():
    assert portfolio_forecast.BacktestResult is types.BacktestResult
    assert portfolio_forecast.SimulationResult is types.SimulationResult
    assert portfolio_forecast.FutureLayout is types.FutureLayout


def test_backtest_result_construction_and_unpacking():
    dates = pd.DatetimeIndex(["2025-06-04 16:00", "2025-06-05 16:00"], tz="America/New_York")
    values = pd.Series([100.0, 101.0], index=dates)
    classes = pd.Series(["regular"], index=dates[1:])
    result = BacktestResult(values, dates, classes)
    assert result.values is values and result.dates is dates and result.classes is classes
    with pytest.raises(ValueError):
        v, d = result  # 0.3.0-style two-name unpacking fails loudly


def test_simulation_result_and_future_layout_construction():
    paths = np.ones((2, 3))
    sim = SimulationResult(paths=paths, dates=None, classes=None, fit=None)
    assert sim.paths is paths and sim.dates is None and sim.fit is None
    layout = FutureLayout(pd.DatetimeIndex([]), np.array([], dtype=object), np.array([]))
    assert len(layout.dates) == 0 and layout.tau.size == 0
