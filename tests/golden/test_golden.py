"""Run the hand-derived golden cases. Format: tests/golden/README.md.

Expected values in the YAML files come from the derivations written in their
comments, never from the package's output. A case is never edited to make the
code pass; a failing case means the code is wrong until shown otherwise.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolio_forecast.performance import performance_report, simulate_buy_and_hold, statistical_report

HERE = Path(__file__).parent
SIM_CASES = sorted((HERE / "simulate").glob("S*.yaml"))
METRIC_CASES = sorted((HERE / "metrics").glob("M*.yaml"))
ALL_CASES = SIM_CASES + METRIC_CASES

TOL_VALUE = 1e-6    # dollars and shares; expected values are exact or written to 9 decimals
RTOL_METRIC = 1e-9  # metrics are written to 12 significant digits

EXPECTED_SIM = {"S01", "S02", "S03", "S04", "S05", "S06", "S07a", "S07b", "S08", "S09"}
EXPECTED_METRICS = {"M01", "M02", "M03", "M04", "M05", "M06", "M07"}

# The metrics performance_report and statistical_report both give for one path
SHARED_METRICS = ["Final Bankroll", "Total Return", "CAGR", "Period Volatility", "Annualized Volatility",
                  "Downside Volatility", "Max Drawdown", "Sharpe Ratio", "Sortino Ratio"]


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ids(paths):
    return [p.stem for p in paths]


def _times(texts, tz):
    """Times from case strings, on `tz`'s wall clock, or naive if tz is None."""
    index = pd.DatetimeIndex([pd.Timestamp(t) for t in texts])
    return index if tz is None else index.tz_localize(tz)


def _prices(case):
    """The case's bars as a (field, symbol) frame, laid out like
    yfinance.download: fields on the outer level, symbols on the inner."""
    fields = case["fields"]
    frames = {}
    for symbol, rows in case["bars"].items():
        index = _times([r[0] for r in rows], case.get("bars_tz"))
        values = [[np.nan if v is None else float(v) for v in r[1:]] for r in rows]
        frames[symbol] = pd.DataFrame(values, index=index, columns=fields)
    data = pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)
    data.columns.names = ["Price", "Ticker"]
    return data.sort_index()


def _simulate(case):
    """Run simulate_buy_and_hold on a case; returns (values, UserWarning messages)."""
    kwargs = {"br0": case["br0"], "fill": case["fill"], "calendar": case.get("calendar", "XNYS")}
    if case.get("tz"):
        kwargs["tz"] = case["tz"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values, _ = simulate_buy_and_hold(_prices(case), case["weights"], **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return values, messages


def _metric_values(case):
    index = _times([t for t, _ in case["values"]], None)
    return pd.Series([float(v) for _, v in case["values"]], index=index)


def _case_values(path):
    """The value series a case's report is computed on: the package's output
    for a simulation case (so the report is chained to it), the given values
    for a metrics case."""
    case = _load(path)
    if "bars" in case:
        return case, _simulate(case)[0]
    return case, _metric_values(case)


def _report_kwargs(case):
    """The case's risk-free rate, and the exchange calendar and timezone its
    intraday labels are grouped into sessions by."""
    return {"risk_free_rate": case.get("risk_free_rate", 0.0), "calendar": case.get("calendar", "XNYS"),
            "tz": case.get("tz")}


def _lookup(report, key):
    for section in (report, report["Returns"], report["Risk"]):
        if key in section:
            return section[key]
    raise KeyError(key)


def _check_report(values, case, expected):
    report = performance_report(values, values.index, case["interval"], **_report_kwargs(case))
    for key, want in expected.items():
        if key in ("Period Returns", "Daily Returns"):
            np.testing.assert_allclose(report[key].to_numpy(), [float(v) for v in want],
                                       rtol=RTOL_METRIC, atol=1e-15, err_msg=key)
        elif key == "Yearly Returns":
            got = _lookup(report, "Actual Yearly Returns")
            assert list(got.index) == [year for year, *_ in want], key
            np.testing.assert_allclose(got["return"].astype(float), [float(r) for _, r, _, _ in want],
                                       rtol=RTOL_METRIC, err_msg=key)
            assert list(got["days"]) == [d for _, _, d, _ in want], f"{key} days"
            assert list(got["is_full"]) == [f for *_, f in want], f"{key} is_full"
        elif key in ("Periods", "Total Days"):
            assert report[key] == want, key
        else:
            np.testing.assert_allclose(_lookup(report, key), float(want), rtol=RTOL_METRIC, atol=1e-15,
                                       equal_nan=True, err_msg=key)


def test_all_cases_present():
    assert {_load(p)["case"] for p in SIM_CASES} >= EXPECTED_SIM
    assert {_load(p)["case"] for p in METRIC_CASES} >= EXPECTED_METRICS


@pytest.mark.parametrize("path", SIM_CASES, ids=_ids(SIM_CASES))
def test_case_arithmetic(path):
    """The case's own numbers add up, without running the package: each
    expected value is sum(shares x price) at the fill and at every close after
    it, and the shares split br0 in the target weights."""
    case = _load(path)
    expected = case["expected"]
    q, fill_px = expected["shares"], expected["fill_prices"]
    br0 = float(case["br0"])

    assert sum(q[s] * fill_px[s] for s in q) == pytest.approx(br0, abs=TOL_VALUE), "shares cost br0"
    total_w = sum(float(case["weights"][s]) for s in q)
    for s in q:
        assert q[s] * fill_px[s] / br0 == pytest.approx(float(case["weights"][s]) / total_w, abs=1e-12), s

    # The close is the adjusted one when the case carries it, held over a gap
    data = _prices(case)
    close_field = "Adj Close" if "Adj Close" in case["fields"] else "Close"
    closes = data[close_field].ffill()
    # After br0, an open fill has a value at every bar's close; a close fill
    # at every bar's close but the first, which is the fill itself
    marks = closes if case["fill"] == "open" else closes.iloc[1:]
    derived = [br0] + [sum(q[s] * row[s] for s in q) for _, row in marks.iterrows()]
    np.testing.assert_allclose(derived, [float(v) for _, v in expected["values"]], rtol=0, atol=TOL_VALUE)


@pytest.mark.parametrize("path", SIM_CASES, ids=_ids(SIM_CASES))
def test_simulate_case(path):
    case = _load(path)
    expected = case["expected"]
    values, messages = _simulate(case)

    # Every value's label, including its timezone (None: naive), then the value
    want_index = _times([t for t, _ in expected["values"]], expected.get("label_tz"))
    assert str(values.index.tz) == str(want_index.tz), "label timezone"
    assert list(values.index) == list(want_index), "labels"
    np.testing.assert_allclose(values.to_numpy(), [float(v) for _, v in expected["values"]],
                               rtol=0, atol=TOL_VALUE)

    # Exactly the warnings the case expects, each containing its text
    want_warnings = expected.get("warnings", [])
    assert len(messages) == len(want_warnings), messages
    for got, want in zip(messages, want_warnings, strict=True):
        assert want in got


@pytest.mark.parametrize(
    "path", [p for p in ALL_CASES if "report" in _load(p)["expected"] or "bars" not in _load(p)],
    ids=lambda p: p.stem,
)
def test_report_case(path):
    """performance_report on the case's values: for a simulation case, on
    what the package simulated, so the metrics are checked against what the
    simulation means (e.g. with an open fill, one period per bar)."""
    case, values = _case_values(path)
    expected = case["expected"]["report"] if "bars" in case else case["expected"]
    _check_report(values, case, expected)


@pytest.mark.parametrize("path", ALL_CASES, ids=_ids(ALL_CASES))
def test_compounding_identity(path):
    """Compounding the period returns, and the daily returns the risk
    figures use, gives back the total move: no return is dropped or counted
    twice."""
    case, values = _case_values(path)
    report = performance_report(values, values.index, case["interval"], **_report_kwargs(case))
    total = values.iloc[-1] / values.iloc[0]
    assert np.prod(1 + report["Period Returns"].to_numpy()) == pytest.approx(total, rel=1e-12)
    assert len(report["Period Returns"]) == len(values) - 1
    if report["Daily Returns"] is not None:
        assert np.prod(1 + report["Daily Returns"].to_numpy()) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("path", ALL_CASES, ids=_ids(ALL_CASES))
def test_statistical_report_matches(path):
    """statistical_report computes every metric again, vectorized across
    paths. On the case's values as a single path it must agree with
    performance_report, so the two can't drift apart."""
    case, values = _case_values(path)
    if len(values) < 3:
        pytest.skip("statistical_report needs at least two periods")
    kwargs = _report_kwargs(case)
    single = performance_report(values, values.index, case["interval"], **kwargs)
    paths = statistical_report(values.to_numpy()[None, :], case["interval"], dates=values.index, **kwargs)
    per_path = paths["Per Path Metrics"].iloc[0]
    for key in SHARED_METRICS:
        np.testing.assert_allclose(per_path[key], _lookup(single, key), rtol=1e-12, equal_nan=True, err_msg=key)
