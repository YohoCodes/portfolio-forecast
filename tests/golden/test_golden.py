"""Golden-case runner (R2).

Discovers every YAML case under ``backtest/``, ``metrics/`` and ``forecast/``,
calls the callable its ``kind`` names with its ``inputs``, and compares the
result with its ``expected`` block: labels and classes exactly, numbers within
the case's ``tolerance``. The case format is described in ``README.md`` next
to this file. Expected numbers are derived by hand in each case's comments,
never by running the package.

An unknown ``kind`` or a malformed case fails; nothing here ever skips.
"""

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

GOLDEN = Path(__file__).parent
CASE_DIRS = ("backtest", "metrics", "forecast")
REQUIRED_FIELDS = ("rule", "kind", "inputs", "expected", "tolerance")
FILE_NAME = re.compile(r"^[a-z0-9]+(?:\.[0-9]+)?(?:_[a-z0-9.]+)+\.yaml$")


def case_files(root=GOLDEN):
    """Every case file under the case directories of `root`, sorted."""
    return sorted(path for name in CASE_DIRS for path in (root / name).glob("*.y*ml"))


def load_case(path):
    """Parse one case file and check it has the required fields."""
    with open(path) as handle:
        case = yaml.safe_load(handle)
    if not isinstance(case, dict):
        pytest.fail(f"{path.name}: a case is a YAML mapping, got {type(case).__name__}")
    missing = [field for field in REQUIRED_FIELDS if field not in case]
    if missing:
        pytest.fail(f"{path.name}: missing field(s) {missing}")
    if not isinstance(case["rule"], list) or not case["rule"]:
        pytest.fail(f"{path.name}: `rule` must be a non-empty list of rule IDs")
    return case


# ---------------------------------------------------------------- parsing


def to_index(stamps, tz=None):
    """A DatetimeIndex from YAML timestamps (strings, dates or datetimes).

    All-naive stamps stay naive (midnight dates, or naive intraday input whose
    zone is passed to the callable separately). Offset-aware stamps are read as
    instants and expressed in `tz` (default UTC); comparisons are by instant.
    """
    stamps = [pd.Timestamp(stamp) for stamp in stamps]
    aware = [stamp.tz is not None for stamp in stamps]
    if any(aware) and not all(aware):
        pytest.fail("case mixes naive and offset-aware timestamps in one list")
    if not stamps or not aware[0]:
        if tz is not None and stamps:
            pytest.fail("`tz` given for naive timestamps; give offsets in the stamps instead")
        return pd.DatetimeIndex(stamps)
    return pd.DatetimeIndex([stamp.tz_convert("UTC") for stamp in stamps]).tz_convert(tz or "UTC")


def to_series(inputs):
    """The `values` input: `values` on the index `labels`."""
    index = to_index(inputs["labels"], inputs.get("labels_tz"))
    return pd.Series([float(v) for v in inputs["values"]], index=index)


def price_frame(columns, index):
    """A (bar x symbol) frame from {symbol: [price or null, ...]}."""
    return pd.DataFrame({symbol: [np.nan if p is None else float(p) for p in prices]
                         for symbol, prices in columns.items()}, index=index)


# ---------------------------------------------------------------- comparison


def tolerances(case):
    """(abs, rel) from `tolerance`: a number (absolute) or {abs:, rel:}."""
    tol = case["tolerance"]
    if isinstance(tol, (int, float)):
        return float(tol), 0.0
    if isinstance(tol, dict) and set(tol) <= {"abs", "rel"}:
        return float(tol.get("abs", 0.0)), float(tol.get("rel", 0.0))
    pytest.fail(f"bad tolerance {tol!r}: a number or a mapping with `abs` and/or `rel`")


def assert_labels(actual, expected, what, tz=None):
    """Labels are compared exactly, as instants; `tz` also pins the zone."""
    actual = pd.DatetimeIndex(actual)
    expected = to_index(expected)
    assert len(actual) == len(expected), (
        f"{what}: {len(actual)} labels, expected {len(expected)}\n"
        f"actual:   {list(actual)}\nexpected: {list(expected)}")
    if expected.tz is None:
        assert actual.tz is None, f"{what}: expected naive labels, got tz {actual.tz}"
    else:
        assert actual.tz is not None, f"{what}: expected tz-aware labels, got naive {list(actual)}"
    mismatch = [(i, a, e) for i, (a, e) in enumerate(zip(actual, expected, strict=True)) if a != e]
    assert not mismatch, f"{what}: labels differ at (position, actual, expected) {mismatch}"
    if tz is not None:
        assert str(actual.tz) == tz, f"{what}: labels in {actual.tz}, expected {tz}"


def assert_exact(actual, expected, what):
    """Classes, flags and counts are compared exactly, position by position."""
    actual = [a.item() if hasattr(a, "item") else a for a in actual]
    assert len(actual) == len(expected), (
        f"{what}: {len(actual)} entries, expected {len(expected)}\n"
        f"actual:   {actual}\nexpected: {expected}")
    mismatch = [(i, a, e) for i, (a, e) in enumerate(zip(actual, expected, strict=True)) if a != e]
    assert not mismatch, f"{what}: entries differ at (position, actual, expected) {mismatch}"


def assert_numbers(actual, expected, case, what):
    """Numbers within the case's tolerance; NaN matches NaN, inf matches inf."""
    abs_tol, rel_tol = tolerances(case)
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray([np.nan if e is None else float(e) for e in np.ravel(expected)])
    assert actual.shape == expected.shape, f"{what}: {actual.size} numbers, expected {expected.size}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if math.isnan(e) or math.isinf(e):
            ok = (math.isnan(a) and math.isnan(e)) or a == e
        else:
            ok = math.isfinite(a) and abs(a - e) <= abs_tol + rel_tol * abs(e)
        assert ok, f"{what}[{i}]: {a!r}, expected {e!r} (abs {abs_tol}, rel {rel_tol})"


def assert_indexed_by(obj, labels, what):
    """If `obj` carries an index, it must be `labels` (e.g. return end marks)."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        assert_labels(obj.index, labels, f"{what} index")


# ---------------------------------------------------------------- kinds
# Each handler calls one interface callable (core_engineer_zara.md, Phase 1 to
# Phase 3 interface tables; forecast_engineer_dex.md, Phase 3) and checks its
# output. Imports are inside the handlers so a callable that doesn't exist yet
# fails its cases instead of the whole module.


def run_close_times(case):
    from portfolio_forecast.utils.calendar import close_times

    inputs, expected = case["inputs"], case["expected"]
    index = to_index(inputs["index"], inputs.get("index_tz"))
    result = close_times(index, calendar=inputs.get("calendar", "XNYS"), tz=inputs.get("tz"))
    assert_labels(result, expected["labels"], "close_times", expected.get("tz"))


def run_session_spans(case):
    from portfolio_forecast.utils.calendar import session_spans

    inputs, expected = case["inputs"], case["expected"]
    index = to_index(inputs["index"], inputs.get("index_tz"))
    spans = session_spans(index, calendar=inputs.get("calendar", "XNYS"), tz=inputs.get("tz"))
    assert_labels(spans["first_open"], expected["first_open"], "first_open", expected.get("tz"))
    assert_labels(spans["last_close"], expected["last_close"], "last_close", expected.get("tz"))
    assert_exact(spans["n_sessions"], expected["n_sessions"], "n_sessions")


def run_completed_mask(case):
    from portfolio_forecast.utils.calendar import completed_mask

    inputs, expected = case["inputs"], case["expected"]
    index = to_index(inputs["index"], inputs.get("index_tz"))
    now = pd.Timestamp(inputs["now"])
    mask = completed_mask(index, calendar=inputs.get("calendar", "XNYS"), tz=inputs.get("tz"), now=now)
    assert_exact([bool(m) for m in np.asarray(mask)], expected["mask"], "completed_mask")


def run_history_marks(case):
    from portfolio_forecast.utils.marks import history_marks

    inputs, expected = case["inputs"], case["expected"]
    bars = to_index(inputs["bars"], inputs.get("bars_tz"))
    opens = price_frame(inputs["opens"], bars)
    closes = price_frame(inputs["closes"], bars)
    held = pd.Series(inputs["held"], dtype=bool)
    labels, fresh, classes = history_marks(opens, closes, held, calendar=inputs.get("calendar", "XNYS"),
                                           tz=inputs.get("tz"), fill=inputs["fill"])
    assert_labels(labels, expected["labels"], "labels", expected.get("tz"))
    # Only the symbols the case lists are checked, by position along the marks
    assert len(fresh) == len(expected["labels"]), f"fresh has {len(fresh)} rows, one per mark expected"
    for symbol, flags in expected["fresh"].items():
        assert_exact([bool(f) for f in np.asarray(fresh[symbol])], flags, f"fresh[{symbol}]")
    assert_exact(list(np.asarray(classes)), expected["classes"], "classes")
    assert_indexed_by(classes, expected["labels"][1:], "classes")


def run_is_daily(case):
    from portfolio_forecast.utils.marks import is_daily

    result = is_daily(to_series(case["inputs"]), calendar=case["inputs"].get("calendar", "XNYS"))
    assert isinstance(result, (bool, np.bool_)), f"is_daily returned {type(result).__name__}, not bool"
    assert bool(result) is case["expected"]["value"], f"is_daily: {result}, expected {case['expected']['value']}"


def run_classify_returns(case):
    from portfolio_forecast.utils.marks import classify_returns

    expected = case["expected"]
    classes = classify_returns(to_series(case["inputs"]), calendar=case["inputs"].get("calendar", "XNYS"))
    assert isinstance(classes, pd.Series), f"classify_returns returned {type(classes).__name__}, not Series"
    assert_labels(classes.index, expected["labels"], "classify_returns index")
    assert_exact(list(classes), expected["classes"], "classify_returns")


def run_to_session_values(case):
    from portfolio_forecast.utils.marks import to_session_values

    inputs, expected = case["inputs"], case["expected"]
    values = to_series(inputs)
    if inputs.get("as_backtest_result", False):
        from portfolio_forecast.types import BacktestResult

        classes = pd.Series(inputs["classes"], index=values.index[1:])
        values = BacktestResult(values=values, dates=values.index, classes=classes)
    # `now` is passed only when the case gives it, so other cases keep the default
    options = {"now": pd.Timestamp(inputs["now"])} if "now" in inputs else {}
    result = to_session_values(values, calendar=inputs.get("calendar", "XNYS"), **options)
    assert isinstance(result, pd.Series), f"to_session_values returned {type(result).__name__}, not Series"
    assert_labels(result.index, expected["labels"], "to_session_values index", expected.get("tz"))
    assert_numbers(result.to_numpy(), expected["values"], case, "to_session_values")


def run_next_trading_dates(case):
    from portfolio_forecast.utils.trading_dates import next_trading_dates

    inputs, expected = case["inputs"], case["expected"]
    layout = next_trading_dates(to_series(inputs), inputs["n"], interval=inputs.get("interval"),
                                calendar=inputs.get("calendar", "XNYS"), tz=inputs.get("tz"))
    assert_labels(layout.dates, expected["labels"], "layout.dates", expected.get("tz"))
    assert_exact(list(np.asarray(layout.classes)), expected["classes"], "layout.classes")
    assert_numbers(layout.tau, expected["tau"], case, "layout.tau")


def price_data(inputs, index):
    """The backtest's `data` in a 0.3.0 layout (`get_closing_prices`).

    `prices` is ``{field: {symbol: [price or null, ...]}}``. ``layout:
    fields`` (default) gives yfinance's MultiIndex ``(Price, Ticker)`` columns;
    ``layout: closes`` gives flat columns that are already closes, one per
    symbol, from the ``Close`` field (no opens: B1.5, B2).
    """
    layout = inputs.get("layout", "fields")
    prices = inputs["prices"]
    if layout == "fields":
        frames = {field: price_frame(columns, index) for field, columns in prices.items()}
        return pd.concat(frames, axis=1, names=["Price", "Ticker"])
    if layout == "closes":
        return price_frame(prices["Close"], index)
    pytest.fail(f"unknown layout {layout!r}: `fields` or `closes`")


def run_simulate_buy_and_hold(case):
    from portfolio_forecast.performance.simulate import simulate_buy_and_hold
    from portfolio_forecast.types import BacktestResult

    inputs, expected = case["inputs"], case["expected"]
    bars = to_index(inputs["bars"], inputs.get("bars_tz"))
    now = inputs.get("now")
    result = simulate_buy_and_hold(price_data(inputs, bars), inputs["w"], br0=inputs.get("br0", 1),
                                   fill=inputs["fill"], calendar=inputs.get("calendar", "XNYS"),
                                   tz=inputs.get("tz"), now=None if now is None else pd.Timestamp(now))
    assert isinstance(result, BacktestResult), f"returned {type(result).__name__}, not BacktestResult"
    assert_labels(result.values.index, expected["labels"], "values index", expected.get("tz"))
    assert_labels(result.dates, expected["labels"], "dates", expected.get("tz"))
    assert_numbers(result.values.to_numpy(), expected["values"], case, "values")
    assert_exact(list(np.asarray(result.classes)), expected["classes"], "classes")
    assert_indexed_by(result.classes, expected["labels"][1:], "classes")


def report_values(inputs):
    """The `values` argument of the metrics callables: a Series, or with
    ``as_backtest_result: true`` a `BacktestResult` carrying `classes`."""
    values = to_series(inputs)
    if not inputs.get("as_backtest_result", False):
        return values
    from portfolio_forecast.types import BacktestResult

    classes = pd.Series(inputs["classes"], index=values.index[1:])
    return BacktestResult(values=values, dates=values.index, classes=classes)


def find_key(report, key):
    """The one entry named `key` anywhere in the nested report dict.

    The spec names the figures ("Gap Volatility", "Sortino Ratio", ...) but not
    which sub-dict holds them, so cases name figures, not paths; a name found
    twice or not at all fails.
    """
    found = []

    def walk(node):
        if isinstance(node, dict):
            for name, item in node.items():
                if name == key:
                    found.append(item)
                walk(item)

    walk(report)
    if len(found) != 1:
        pytest.fail(f"report has {len(found)} entries named {key!r}, expected exactly one")
    return found[0]


def assert_series(actual, want, case, what):
    """A Series against ``{labels: [...], values: [...]}``; `labels` optional."""
    assert isinstance(actual, pd.Series), f"{what}: got {type(actual).__name__}, not Series"
    if "labels" in want:
        assert_labels(actual.index, want["labels"], f"{what} index", want.get("tz"))
    assert_numbers(actual.to_numpy(), want["values"], case, what)


def run_performance_report(case):
    from portfolio_forecast.performance.report import performance_report

    inputs, expected = case["inputs"], case["expected"]
    values = report_values(inputs)
    classes = None
    if "classes" in inputs and not inputs.get("as_backtest_result", False):
        classes = pd.Series(inputs["classes"], index=values.index[1:])
    options = {name: inputs[name] for name in ("interval", "risk_free_rate", "cagr_clock") if name in inputs}
    report = performance_report(values, classes, calendar=inputs.get("calendar", "XNYS"), **options)
    assert isinstance(report, dict), f"performance_report returned {type(report).__name__}, not dict"
    for key, want in expected.get("report", {}).items():
        got = find_key(report, key)
        if isinstance(want, dict):
            assert_series(got, want, case, key)
        else:
            assert_numbers([got], [want], case, key)
    if "yearly" in expected:
        # M8: one row per year, with its return and whether it is full
        want = expected["yearly"]
        table = find_key(report, "Actual Yearly Returns")
        assert_exact([int(year) for year in table.index], want["years"], "yearly years")
        assert_numbers(table["return"].to_numpy(), want["return"], case, "yearly return")
        assert_exact([bool(full) for full in table["is_full"]], want["is_full"], "yearly is_full")


def run_calculate_period_returns(case):
    from portfolio_forecast.performance.returns import calculate_period_returns

    assert_series(calculate_period_returns(report_values(case["inputs"])), case["expected"], case,
                  "calculate_period_returns")


def run_calculate_daily_return(case):
    from portfolio_forecast.performance.returns import calculate_daily_return

    assert_series(calculate_daily_return(report_values(case["inputs"])), case["expected"], case,
                  "calculate_daily_return")


def run_calculate_session_returns(case):
    from portfolio_forecast.performance.returns import calculate_session_returns

    inputs = case["inputs"]
    result = calculate_session_returns(report_values(inputs), calendar=inputs.get("calendar", "XNYS"))
    assert_series(result, case["expected"], case, "calculate_session_returns")


def run_returns_and_report(case):
    """M9 with M10: both return helpers and `performance_report` on one history,
    so a case can pin the helpers' NaN rows and the risk figures together."""
    from portfolio_forecast.performance.report import performance_report
    from portfolio_forecast.performance.returns import calculate_period_returns, calculate_session_returns

    inputs, expected = case["inputs"], case["expected"]
    calendar = inputs.get("calendar", "XNYS")
    values = report_values(inputs)
    if "period_returns" in expected:
        assert_series(calculate_period_returns(values), expected["period_returns"], case,
                      "calculate_period_returns")
    if "session_returns" in expected:
        assert_series(calculate_session_returns(values, calendar=calendar), expected["session_returns"],
                      case, "calculate_session_returns")
    if "report" in expected:
        report = performance_report(values, None, calendar=calendar)
        for key, want in expected["report"].items():
            assert_numbers([find_key(report, key)], [want], case, key)


# ---------------------------------------------------------------- Monte Carlo
# Phase 3 kinds: the three simulators (forecast_engineer_dex.md, Phase 3
# interface table) and `statistical_report` (core_engineer_zara.md, Phase 3).


def generated_history(spec):
    """A seeded synthetic history for the statistical cases (R2, MC3-MC7).

    Built here from the calendar and numpy alone, never from the package, so
    the population each class is drawn from is known exactly:

    - ``kind: sessions``: `sessions` consecutive sessions of `calendar` from
      `start`, an open mark (the first is the fill) then one mark per bar
      close on a `grid` starting at the open, the last bar ending at the
      session close (short if the close is off the grid, B3). The return
      ending at each later open mark is a gap; every other return is regular.
    - ``kind: continuous``: `bars` + 1 marks on ``start + k grid`` (U3).
    - ``kind: array``: a plain array of `returns` + 1 values (MC1).

    Log returns: a gap is ``mean + sd z``; a regular bar of length
    rho = tau / grid is ``rho mean + sqrt(rho) sd z`` (the MC3 location-scale
    model, so the pool's population is the same at every rho), with ``z``
    standard normal, or Student's t with `df` degrees of freedom (`sd` is then
    the scale), or ``constant: x`` for a zero-variance class (rho x on a
    short bar). ``regular: {regimes: {P, means, sds}}`` draws regular returns
    from a Markov chain started in state 0. Values are ``100 exp(sum l)``.

    Returns (values, classes, rho): a Series (an ndarray for ``array``), the
    classes Series indexed by the end mark of each return (None for
    ``array``) and each return's rho.
    """
    rng = np.random.default_rng(spec["seed"])
    kind = spec["kind"]
    grid = pd.Timedelta(spec["grid"]) if "grid" in spec else None
    if kind == "sessions":
        import exchange_calendars as xcals

        calendar = xcals.get_calendar(spec.get("calendar", "XNYS"))
        start = pd.Timestamp(spec["start"])
        sessions = calendar.sessions_in_range(start, start + pd.Timedelta(days=2 * spec["sessions"] + 10))
        sessions = sessions[: spec["sessions"]]
        assert len(sessions) == spec["sessions"], "calendar window too short for the requested sessions"
        marks, classes, rho = [], [], []
        for session in sessions:
            open_, close = calendar.session_open(session), calendar.session_close(session)
            if marks:  # the first open mark is the fill, with no return ending at it
                classes.append("gap")
                rho.append(1.0)
            marks.append(open_)
            bar_start = open_
            while bar_start < close:
                bar_end = min(bar_start + grid, close)
                marks.append(bar_end)
                classes.append("regular")
                rho.append((bar_end - bar_start) / grid)
                bar_start = bar_end
        index = pd.DatetimeIndex(marks).tz_convert(spec.get("tz", "America/New_York"))
    elif kind == "continuous":
        index = pd.date_range(pd.Timestamp(spec["start"]), periods=spec["bars"] + 1, freq=grid)
        classes, rho = ["regular"] * spec["bars"], [1.0] * spec["bars"]
    elif kind == "array":
        index = None
        classes, rho = ["regular"] * spec["returns"], [1.0] * spec["returns"]
    else:
        pytest.fail(f"unknown generate kind {kind!r}: sessions, continuous or array")
    classes, rho = np.array(classes, dtype=object), np.asarray(rho, dtype=float)
    n = len(classes)
    ell = np.zeros(n)
    for name in ("gap", "regular"):
        model = spec.get(name)
        where = classes == name
        if model is None:
            if where.any():
                pytest.fail(f"generate: the history has {name} returns but no `{name}` model")
            continue
        if "constant" in model:
            ell[where] = rho[where] * float(model["constant"])
            continue
        if "regimes" in model:
            regimes = model["regimes"]
            P = np.asarray(regimes["P"], dtype=float)
            states = np.zeros(n, dtype=int)
            for t in range(1, n):
                states[t] = rng.choice(len(P), p=P[states[t - 1]])
            mean = np.asarray(regimes["means"], dtype=float)[states]
            sd = np.asarray(regimes["sds"], dtype=float)[states]
        else:
            mean, sd = float(model["mean"]), float(model["sd"])
            mean, sd = np.full(n, mean), np.full(n, sd)
        z = rng.standard_t(model["df"], n) if "df" in model else rng.standard_normal(n)
        ell[where] = (rho * mean + np.sqrt(rho) * sd * z)[where]
    values = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(ell)]))
    if index is None:
        return values, None, rho
    return pd.Series(values, index=index), pd.Series(classes, index=index[1:]), rho


def scipy_distribution(name):
    """A scipy.stats distribution from its name (``norm``, ``t``, ``johnsonsu``)."""
    import scipy.stats

    dist = getattr(scipy.stats, name, None)
    if dist is None:
        pytest.fail(f"unknown scipy distribution {name!r}")
    return dist


def simulator_options(inputs):
    """Keyword arguments other than `values` and `classes`, from `inputs`."""
    options = {name: inputs[name] for name in ("sim_length", "n_sims", "random_state") if name in inputs}
    if "calendar" in inputs:
        options["calendar"] = inputs["calendar"]
    for name, value in (inputs.get("options") or {}).items():
        if name == "distribution":
            if isinstance(value, dict):
                value = {cls: scipy_distribution(dist) for cls, dist in value.items()}
            elif value is not None:
                value = scipy_distribution(value)
        options[name] = value
    return options


def simulator_input(inputs):
    """(values, classes, last historical label) for a simulator case."""
    if "generate" in inputs:
        values, classes, _ = generated_history(inputs["generate"])
        last = None if classes is None else values.index[-1]
        if not inputs["generate"].get("pass_classes", False):
            classes = None
        return values, classes, last
    if "array" in inputs:
        return np.asarray(inputs["array"], dtype=float), None, None
    values = to_series(inputs)
    classes = None
    if "classes" in inputs:
        classes = pd.Series(inputs["classes"], index=values.index[1:])
        if inputs.get("as_backtest_result", False):
            from portfolio_forecast.types import BacktestResult

            return BacktestResult(values=values, dates=values.index, classes=classes), None, values.index[-1]
    return values, classes, values.index[-1]


def future_rho(result, last, grid):
    """rho = tau / grid of each future return, read off the labels: a regular
    mark's bar runs from the previous mark (the last historical one for the
    first); gaps are one unit (U2). All 1.0 without labels."""
    n_marks = result.paths.shape[1] - 1
    if result.dates is None or grid is None:
        return np.ones(n_marks)
    dates = pd.DatetimeIndex(result.dates)
    previous = pd.DatetimeIndex([last]).append(dates[:-1])
    rho = np.asarray((dates - previous) / pd.Timedelta(grid), dtype=float)
    rho[np.asarray(result.classes) == "gap"] = 1.0
    return rho


FAMILIES = {"norm": "normal", "t": "t", "johnsonsu": "johnsonsu"}
SCIPY_NAMES = {family: name for name, family in FAMILIES.items()}


def selected_family(fit, cls):
    """The family the fit reports for class `cls` (MC5): ``fit[cls]
    ["Distribution"]``, a scipy.stats name (the key names are those of
    `parametric_monte_carlo`'s docstring)."""
    assert isinstance(fit, dict), f"fit is {type(fit).__name__}, not dict"
    per_class = fit.get(cls)
    assert isinstance(per_class, dict), f"fit has no per-class dict for {cls!r}: keys {list(fit)}"
    name = per_class.get("Distribution")
    assert isinstance(name, str), f"fit[{cls!r}]['Distribution'] is {name!r}, not a scipy name"
    family = FAMILIES.get(name)
    assert family is not None, f"fit[{cls!r}] family {name!r} is not norm, t or johnsonsu"
    return family


def history_pools(inputs):
    """{class: log returns} of a generated history whose bars are all full
    (rho = 1, so l' = l under MC3), or None if that doesn't hold."""
    if "generate" not in inputs:
        return None
    values, classes, rho = generated_history(inputs["generate"])
    if classes is None or not np.all(rho == 1.0):
        return None
    ell = np.log(values.to_numpy()[1:] / values.to_numpy()[:-1])
    return {cls: ell[np.asarray(classes) == cls] for cls in ("gap", "regular")}


def check_fit_report(fit, want, pools):
    """MC5: per class the fit reports the family, its parameters, the
    criterion and every candidate's score; the family's score is the
    criterion, -2 log L + k ln n (BIC) or + 2k (AIC), at the reported
    parameters on the class's pool, and the lowest of the candidates."""
    import scipy.stats

    for cls, spec in want.items():
        family = selected_family(fit, cls)
        entry = fit[cls]
        criterion = spec["criterion"]
        assert entry.get("Criterion") == criterion, f"fit[{cls!r}]['Criterion'] is {entry.get('Criterion')!r}"
        score = entry.get(criterion)
        assert isinstance(score, (int, float, np.floating)) and math.isfinite(score), (
            f"fit[{cls!r}][{criterion!r}] is {score!r}")
        dist = getattr(scipy.stats, SCIPY_NAMES[family])
        names = ([n.strip() for n in dist.shapes.split(",")] if dist.shapes else []) + ["loc", "scale"]
        params = entry.get("Parameters")
        assert isinstance(params, dict) and list(params) == names, (
            f"fit[{cls!r}]['Parameters'] {params!r}, expected keys {names}")
        assert all(math.isfinite(float(v)) for v in params.values()) and params["scale"] > 0
        if pools is not None:
            x = pools[cls]
            k = len(names)
            penalty = k * math.log(len(x)) if criterion == "BIC" else 2 * k
            shapes = [params[n] for n in names[:-2]]
            expected = -2 * dist.logpdf(x, *shapes, loc=params["loc"], scale=params["scale"]).sum() + penalty
            assert abs(score - expected) <= 1e-6 * abs(expected), (
                f"fit[{cls!r}] {criterion} {score!r}, but -2 log L + penalty at its parameters is {expected!r}")
        candidates = entry.get("Candidates")
        if spec["candidates"] is None:
            assert candidates is None, f"fit[{cls!r}]['Candidates'] should be None, got {candidates!r}"
            continue
        assert isinstance(candidates, dict), f"fit[{cls!r}]['Candidates'] is {candidates!r}"
        assert sorted(FAMILIES.get(n, n) for n in candidates) == sorted(spec["candidates"]), (
            f"fit[{cls!r}] candidates {list(candidates)}, expected {spec['candidates']}")
        scores = [float(v) for v in candidates.values()]
        assert all(math.isfinite(v) for v in scores) and scores == sorted(scores), (
            f"fit[{cls!r}] candidate scores {candidates!r} are not finite and lowest first")
        assert abs(candidates[SCIPY_NAMES[family]] - score) <= 1e-9 * abs(score) and score == min(scores), (
            f"fit[{cls!r}]: the selected {family} scores {score!r}, candidates {candidates!r}")


def check_moments(draws, groups, case):
    """Mean and variance of drawn log returns per group, within the case's
    per-group tolerances (stated with their derivation in the case)."""
    tolerance = case["tolerance"]
    for group, want in groups.items():
        values = draws.get(group)
        assert values is not None and values.size > 0, f"no draws in group {group!r}"
        tol = tolerance[group]
        mean, var = float(values.mean()), float(values.var())
        assert abs(mean - want["mean"]) <= tol["mean"], (
            f"{group}: mean of {values.size} draws {mean!r}, expected {want['mean']!r} +/- {tol['mean']!r}")
        assert abs(var - want["var"]) <= tol["var"], (
            f"{group}: variance of {values.size} draws {var!r}, expected {want['var']!r} +/- {tol['var']!r}")


def check_bound(name, value, want):
    """``{value, tol}`` or ``{max}`` / ``{min}`` on a computed statistic."""
    if "value" in want:
        assert abs(value - want["value"]) <= want["tol"], (
            f"{name}: {value!r}, expected {want['value']} +/- {want['tol']}")
    if "max" in want:
        assert value <= want["max"], f"{name}: {value!r}, expected at most {want['max']}"
    if "min" in want:
        assert value >= want["min"], f"{name}: {value!r}, expected at least {want['min']}"


def simulate_case(simulate, inputs):
    """(SimulationResult, last historical label) of one simulator call."""
    values, classes, last = simulator_input(inputs)
    options = simulator_options(inputs)
    if classes is not None:
        options["classes"] = classes
    return simulate(values, **options), last


def run_simulator(name):
    """A handler calling the simulator `name` and checking its SimulationResult."""

    def handler(case):
        import portfolio_forecast.forecast as forecast
        from portfolio_forecast.types import SimulationResult

        simulate = getattr(forecast, name)
        inputs, expected = case["inputs"], case["expected"]
        result, last = simulate_case(simulate, inputs)
        assert isinstance(result, SimulationResult), f"{name} returned {type(result).__name__}, not SimulationResult"
        paths = np.asarray(result.paths, dtype=float)
        assert paths.ndim == 2, f"paths has shape {paths.shape}"
        # MC4: column 0 is the growth multiple 1.0 of the last historical value
        assert np.all(paths[:, 0] == 1.0), f"column 0 is not 1.0: {paths[:, 0][:5]}"
        if "shape" in expected:
            assert list(paths.shape) == expected["shape"], f"paths shape {paths.shape}, expected {expected['shape']}"
        if expected.get("dates_none", False):
            assert result.dates is None, f"dates should be None for plain-array input, got {result.dates}"
            assert result.classes is None, f"classes should be None for plain-array input, got {result.classes}"
        if "labels" in expected:
            assert result.dates is not None, "dates is None"
            assert_labels(result.dates, expected["labels"], "SimulationResult.dates", expected.get("tz"))
        if "classes" in expected:
            assert result.classes is not None, "classes is None"
            assert_exact(list(np.asarray(result.classes)), expected["classes"], "SimulationResult.classes")
        if result.dates is not None:
            assert len(result.dates) == paths.shape[1] - 1, "one date per simulated mark expected"
        if "fit_none" in expected:
            assert (result.fit is None) is expected["fit_none"], f"fit is {result.fit!r}"
        if "path" in expected:
            # Deterministic case: every path is the same known one
            for row in paths:
                assert_numbers(row, expected["path"], case, "path")
        if "paths" in expected:
            assert_numbers(paths, expected["paths"], case, "paths")
        if "family" in expected:
            for cls, family in expected["family"].items():
                got = selected_family(result.fit, cls)
                assert got == family, f"fit[{cls!r}] selected {got}, expected {family}"
        if "fit_report" in expected:
            check_fit_report(result.fit, expected["fit_report"], history_pools(inputs))
        statistics = {"moments", "gap_draws", "switch_rate", "share_near_zero"}
        if statistics & set(expected):
            ell = np.log(paths[:, 1:] / paths[:, :-1])
            future = np.asarray(result.classes) if result.classes is not None else np.full(ell.shape[1], "regular")
            grid = (inputs.get("generate") or {}).get("grid")
            rho = future_rho(result, last, grid)
            draws = {"gap": ell[:, future == "gap"].ravel(),
                     "regular": ell[:, (future == "regular") & (rho == 1.0)].ravel(),
                     "regular_short": ell[:, (future == "regular") & (rho < 1.0)].ravel()}
            if "moments" in expected:
                check_moments(draws, expected["moments"], case)
            if "gap_draws" in expected:
                assert_numbers(draws["gap"], np.full(draws["gap"].size, expected["gap_draws"]), case, "gap draws")
            if "switch_rate" in expected:
                # MC7: the regime of a draw is the sign of l (regimes with
                # means of opposite sign far apart against their spread)
                sign = np.sign(ell)
                check_bound("switch rate", float((sign[:, 1:] != sign[:, :-1]).mean()), expected["switch_rate"])
            if "share_near_zero" in expected:
                want = expected["share_near_zero"]
                check_bound("share near zero", float((np.abs(ell) < want["threshold"]).mean()), want)

    handler.__name__ = f"run_{name}"
    return handler


def walk_report(node):
    """Every (key, item) pair in the nested report dict."""
    if isinstance(node, dict):
        for key, item in node.items():
            yield key, item
            yield from walk_report(item)


def per_path_column(report, figure, n_paths):
    """The per-path values of `figure`: the one table in the report with a
    `figure` column and one row per path."""
    found = [item[figure] for _, item in walk_report(report)
             if isinstance(item, pd.DataFrame) and figure in item.columns and len(item) == n_paths]
    if len(found) != 1:
        pytest.fail(f"report has {len(found)} per-path tables with a {figure!r} column, expected exactly one")
    return found[0].to_numpy(dtype=float)


SUMMARY = ("Mean", "Median", "Lower", "Upper")


def summary_row(report, figure):
    """`figure`'s Mean, Median, Lower and Upper across paths: its row in the
    one summary table that has it (M11: these keys are unchanged)."""
    found = []
    for key, item in walk_report(report):
        if isinstance(item, pd.DataFrame) and figure in item.index and set(SUMMARY) <= set(item.columns):
            found.append(item.loc[figure])
        elif key == figure and isinstance(item, dict) and set(SUMMARY) <= set(item):
            found.append(item)
    if len(found) != 1:
        pytest.fail(f"report has {len(found)} summaries of {figure!r}, expected exactly one")
    return found[0]


def run_simulate_then_report(case):
    """A simulator's SimulationResult passed straight to statistical_report
    (MC8 layout into M4): `inputs` as the simulator kinds plus `simulator`
    and `report_options`; `expected` as the statistical_report kind."""
    import portfolio_forecast.forecast as forecast
    from portfolio_forecast.performance.report import statistical_report

    inputs, expected = case["inputs"], case["expected"]
    result, _ = simulate_case(getattr(forecast, inputs["simulator"]), inputs)
    report = statistical_report(result, **(inputs.get("report_options") or {}))
    n_paths = np.asarray(result.paths).shape[0]
    for figure, want in expected.get("per_path", {}).items():
        assert_numbers(per_path_column(report, figure, n_paths), want, case, f"{figure} per path")
    for key, want in expected.get("report", {}).items():
        assert_numbers([find_key(report, key)], [want], case, key)


def run_statistical_report(case):
    import contextlib
    import io

    from portfolio_forecast.performance.report import statistical_report
    from portfolio_forecast.types import SimulationResult

    inputs, expected = case["inputs"], case["expected"]
    options = {name: inputs[name] for name in ("interval", "risk_free_rate", "confidence", "cagr_clock", "display")
               if name in inputs}
    if "calendar" in inputs:
        options["calendar"] = inputs["calendar"]
    if "simulation" in inputs:
        sim = inputs["simulation"]
        sims = SimulationResult(paths=np.asarray(sim["paths"], dtype=float),
                                dates=to_index(sim["dates"], sim.get("dates_tz")),
                                classes=np.asarray(sim["classes"], dtype=object), fit=None)
    else:
        sims = np.asarray(inputs["paths"], dtype=float)
        if "dates" in inputs:
            options["dates"] = to_index(inputs["dates"], inputs.get("dates_tz"))
    n_paths = np.asarray(sims.paths if isinstance(sims, SimulationResult) else sims).shape[0]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        report = statistical_report(sims, **options)
    assert isinstance(report, dict), f"statistical_report returned {type(report).__name__}, not dict"
    for figure, want in expected.get("per_path", {}).items():
        assert_numbers(per_path_column(report, figure, n_paths), want, case, f"{figure} per path")
    for figure, want in expected.get("summary", {}).items():
        row = summary_row(report, figure)
        for stat, value in want.items():
            assert_numbers([row[stat]], [value], case, f"{figure} {stat}")
    for key, want in expected.get("report", {}).items():
        assert_numbers([find_key(report, key)], [want], case, key)
    text = printed.getvalue()
    for phrase in expected.get("printed", []):
        assert phrase in text, f"printed report lacks {phrase!r}:\n{text}"
    for phrase in expected.get("not_printed", []):
        assert phrase not in text, f"printed report still says {phrase!r}:\n{text}"
    for phrase in expected.get("docstring", []):
        assert phrase in (statistical_report.__doc__ or ""), f"statistical_report docstring lacks {phrase!r}"


KINDS = {
    "close_times": run_close_times,
    "session_spans": run_session_spans,
    "completed_mask": run_completed_mask,
    "history_marks": run_history_marks,
    "is_daily": run_is_daily,
    "classify_returns": run_classify_returns,
    "to_session_values": run_to_session_values,
    "next_trading_dates": run_next_trading_dates,
    "simulate_buy_and_hold": run_simulate_buy_and_hold,
    "performance_report": run_performance_report,
    "calculate_period_returns": run_calculate_period_returns,
    "calculate_session_returns": run_calculate_session_returns,
    "calculate_daily_return": run_calculate_daily_return,
    "returns_and_report": run_returns_and_report,
    "nonparametric_monte_carlo": run_simulator("nonparametric_monte_carlo"),
    "parametric_monte_carlo": run_simulator("parametric_monte_carlo"),
    "regime_switching_monte_carlo": run_simulator("regime_switching_monte_carlo"),
    "statistical_report": run_statistical_report,
    "simulate_then_report": run_simulate_then_report,
}


def warning_matches(entry, message):
    """Whether one `warns` entry matches one warning message.

    A string entry is a substring of the message. A mapping ``{match: regex,
    count: N}`` is a case-insensitive regex search plus, if ``count`` is given,
    the integer N standing alone in the message (not inside a number or a date) (B1.6, B4.1, B4.4: a warning
    "gives the count"), since the exact wording is the implementer's.
    """
    if isinstance(entry, str):
        return entry in message
    if not isinstance(entry, dict) or "match" not in entry or set(entry) - {"match", "count"}:
        pytest.fail(f"bad `warns` entry {entry!r}: a substring or {{match: regex, count: N}}")
    if not re.search(entry["match"], message, re.I):
        return False
    if "count" not in entry:
        return True
    # The count stands alone: not part of a larger number, a decimal or a date
    return re.search(rf"(?<![\d.:/-]){int(entry['count'])}(?![\d.:/-])", message) is not None


def check_warnings(caught, category, entries, what):
    """The `category` warnings in `caught` must be exactly `entries`: each entry
    matches some warning and each warning matches some entry."""
    messages = [str(w.message) for w in caught if issubclass(w.category, category)]
    unmatched = [e for e in entries if not any(warning_matches(e, m) for m in messages)]
    unexpected = [m for m in messages if not any(warning_matches(e, m) for e in entries)]
    assert not unmatched, f"expected {what}(s) not raised: {unmatched}; raised: {messages}"
    assert not unexpected, f"unexpected {what}(s): {unexpected}"


def run_case(case):
    """Dispatch on `kind`, checking `raises` and `warns` if the case gives them.

    ``expected.raises`` (an exception name, optionally with ``expected.match``,
    a regex) means the call must raise it. ``expected.warns`` (a list of
    entries, see `warning_matches`) means the UserWarnings raised must be
    exactly these: each entry matches some warning and each warning matches
    some entry; ``warns: []`` asserts there is no UserWarning. Without
    ``warns``, warnings are not checked. ``expected.deprecation`` checks
    DeprecationWarnings the same way (M10).
    """
    handler = KINDS.get(case["kind"])
    if handler is None:
        pytest.fail(f"unknown kind {case['kind']!r}; known kinds: {sorted(KINDS)}")
    expected = case["expected"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "raises" in expected:
            error = {"ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError}.get(expected["raises"])
            if error is None:
                pytest.fail(f"unsupported `raises` {expected['raises']!r}")
            with pytest.raises(error, match=expected.get("match")):
                handler(case)
        else:
            handler(case)
    if "warns" in expected:
        check_warnings(caught, UserWarning, expected["warns"], "warning")
    if "deprecation" in expected:
        check_warnings(caught, DeprecationWarning, expected["deprecation"], "DeprecationWarning")


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("path", case_files(), ids=lambda p: str(p.relative_to(GOLDEN)))
def test_golden_case(path):
    assert FILE_NAME.match(path.name), f"{path.name}: name cases <ruleid>_<slug>.yaml, lowercase"
    run_case(load_case(path))


def test_cases_exist():
    assert case_files(), "no golden cases found"


def test_unknown_kind_fails():
    case = {"rule": ["B3"], "kind": "no_such_callable", "inputs": {}, "expected": {}, "tolerance": 0}
    with pytest.raises(pytest.fail.Exception, match="unknown kind"):
        run_case(case)


def test_case_missing_a_field_fails(tmp_path):
    path = tmp_path / "b3_no_tolerance.yaml"
    path.write_text("rule: [B3]\nkind: close_times\ninputs: {}\nexpected: {}\n")
    with pytest.raises(pytest.fail.Exception, match="tolerance"):
        load_case(path)


def test_label_comparison_is_exact_and_by_instant():
    stamps = ["2024-11-29T13:00:00-05:00"]
    assert_labels(pd.DatetimeIndex([pd.Timestamp("2024-11-29 18:00", tz="UTC")]), stamps, "same instant")
    with pytest.raises(AssertionError):
        assert_labels(pd.DatetimeIndex([pd.Timestamp("2024-11-29 13:00:01", tz="America/New_York")]),
                      stamps, "one second late")
    with pytest.raises(AssertionError):
        assert_labels(pd.DatetimeIndex([pd.Timestamp("2024-11-29 13:00")]), stamps, "naive")


def test_warning_entries_match_substrings_and_counts():
    message = "Dropped 3 bars outside the XNYS sessions (2024-11-29)"
    assert warning_matches("outside", message)
    assert warning_matches({"match": "outside|dropped"}, message)
    assert warning_matches({"match": "OUTSIDE", "count": 3}, message)
    assert not warning_matches({"match": "outside", "count": 2}, message)
    assert not warning_matches({"match": "outside", "count": 11}, message)
    assert not warning_matches({"match": "irregular", "count": 3}, message)
    with pytest.raises(pytest.fail.Exception, match="bad `warns` entry"):
        warning_matches({"count": 3}, message)


def test_warnings_must_match_exactly():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("Dropped 3 bars outside the sessions", UserWarning, stacklevel=1)
        warnings.warn("old name", DeprecationWarning, stacklevel=1)
    check_warnings(caught, UserWarning, [{"match": "outside", "count": 3}], "warning")
    check_warnings(caught, DeprecationWarning, ["old name"], "DeprecationWarning")
    with pytest.raises(AssertionError, match="unexpected"):
        check_warnings(caught, UserWarning, [], "warning")
    with pytest.raises(AssertionError, match="not raised"):
        check_warnings(caught, DeprecationWarning, ["old name", "another"], "DeprecationWarning")


def test_find_key_needs_exactly_one_entry():
    report = {"Total Years": 1.0, "Risk": {"Sharpe Ratio": 2.0}, "Paths": {"Risk": {"Sharpe Ratio": 3.0}}}
    assert find_key(report, "Total Years") == 1.0
    with pytest.raises(pytest.fail.Exception, match="2 entries"):
        find_key(report, "Sharpe Ratio")
    with pytest.raises(pytest.fail.Exception, match="0 entries"):
        find_key(report, "Gap Volatility")


def test_number_comparison_uses_the_case_tolerance():
    case = {"tolerance": {"abs": 1e-9}}
    assert_numbers([0.5 + 1e-10], [0.5], case, "within")
    with pytest.raises(AssertionError):
        assert_numbers([0.5 + 1e-8], [0.5], case, "outside")
    with pytest.raises(AssertionError):
        assert_numbers([np.nan], [0.5], case, "nan")


def test_generated_sessions_follow_the_calendar_and_location_scale_model():
    # 2024-11-27 (full day) and 11-29 (half-day, 09:30-13:00) on an hourly grid
    spec = {"kind": "sessions", "start": "2024-11-27", "sessions": 2, "grid": "1h", "seed": 0,
            "gap": {"constant": -0.01}, "regular": {"constant": 0.002}}
    values, classes, rho = generated_history(spec)
    expected = ["2024-11-27T09:30:00-05:00"] + [f"2024-11-27T{h}:30:00-05:00" for h in range(10, 16)] + [
        "2024-11-27T16:00:00-05:00", "2024-11-29T09:30:00-05:00", "2024-11-29T10:30:00-05:00",
        "2024-11-29T11:30:00-05:00", "2024-11-29T12:30:00-05:00", "2024-11-29T13:00:00-05:00"]
    assert_labels(values.index, expected, "generated labels", "America/New_York")
    assert list(classes) == ["regular"] * 7 + ["gap"] + ["regular"] * 4
    assert list(rho) == [1.0] * 6 + [0.5, 1.0] + [1.0] * 3 + [0.5]
    ell = np.log(values.to_numpy()[1:] / values.to_numpy()[:-1])
    np.testing.assert_allclose(ell, np.where(classes == "gap", -0.01, rho * 0.002), rtol=1e-12)


def test_generated_history_is_seeded():
    spec = {"kind": "array", "returns": 5, "seed": 3, "regular": {"mean": 0.0, "sd": 0.01}}
    first, classes, _ = generated_history(spec)
    np.testing.assert_array_equal(first, generated_history(spec)[0])
    assert classes is None and len(first) == 6 and first[0] == 100.0


def test_per_path_and_summary_lookups():
    per_path = pd.DataFrame({"CAGR": [0.1, 0.2, 0.3]})
    summary = pd.DataFrame({"Mean": [0.2], "Median": [0.2], "Lower": [0.105], "Upper": [0.295]}, index=["CAGR"])
    report = {"Per Path Metrics": per_path, "Returns": summary, "Total Years": 1.0}
    assert list(per_path_column(report, "CAGR", 3)) == [0.1, 0.2, 0.3]
    assert summary_row(report, "CAGR")["Lower"] == 0.105
    with pytest.raises(pytest.fail.Exception, match="0 per-path tables"):
        per_path_column(report, "CAGR", 4)
    with pytest.raises(pytest.fail.Exception, match="0 summaries"):
        summary_row(report, "Sortino Ratio")


def test_selected_family_reads_the_class_entry():
    import scipy.stats

    fit = {"gap": {"Distribution": "norm"}, "regular": {"Distribution": "t"}}
    assert selected_family(fit, "gap") == "normal"
    assert selected_family(fit, "regular") == "t"
    with pytest.raises(AssertionError):
        selected_family({"gap": {"family": "norm"}}, "gap")
    with pytest.raises(AssertionError):
        selected_family({"gap": {"Distribution": scipy.stats.norm}}, "gap")


def test_fit_report_check_bites():
    x = np.array([-0.01, 0.0, 0.02, 0.005, -0.004])
    loc, scale = float(x.mean()), float(x.std())
    bic = -2 * float(np.sum(-0.5 * ((x - loc) / scale) ** 2 - np.log(scale * math.sqrt(2 * math.pi)))) \
        + 2 * math.log(5)
    entry = {"Distribution": "norm", "Parameters": {"loc": loc, "scale": scale}, "Criterion": "BIC",
             "BIC": bic, "Candidates": {"norm": bic, "t": bic + 1.0, "johnsonsu": bic + 2.0}}
    want = {"gap": {"criterion": "BIC", "candidates": ["normal", "t", "johnsonsu"]}}
    check_fit_report({"gap": entry}, want, {"gap": x})
    for broken in ({"BIC": bic + 0.5, "Candidates": {"norm": bic + 0.5, "t": bic + 1, "johnsonsu": bic + 2}},
                   {"Candidates": {"t": bic - 1.0, "norm": bic, "johnsonsu": bic + 2.0}},
                   {"Criterion": "AIC"}, {"Parameters": {"mu": loc, "sigma": scale}}):
        with pytest.raises(AssertionError):
            check_fit_report({"gap": {**entry, **broken}}, want, {"gap": x})
