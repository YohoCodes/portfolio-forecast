# Golden cases

Hand-derived known-answer cases, one per YAML file, each tied to the rule IDs
of the spec (B1.1, M4, MC2, ...) that the top-level `README.md` lists in its
**Logic** section. `test_golden.py` runs every case; `test_traceability.py`
checks that every case cites known rules, that every rule has a case and
names cases that cite it on its `Cases:` line in the README, and that no rule
is left pending (R2). A new case named after a rule goes on that rule's `Cases:` line.

## Rules

1. **Hand-derived.** Every expected value is derived by hand from the spec,
   step by step, in the case's YAML comments. It is **never** obtained by
   running the package, nor adjusted to match its output. If the package
   disagrees with a case, the disagreement is reported with the derivation and
   the package's number; neither side is changed to fit the other.
2. **Frozen once merged.** A merged case is not edited. Changing one (inputs,
   expected values or tolerance) needs a maintainer's sign-off, recorded in
   the commit message.
3. **One case per file**, named `<ruleid>_<slug>.yaml`, lowercase, by the
   first rule it cites (`b1.6_missing_first_bar.yaml`,
   `mc2_daily_history.yaml`), in the directory of its rule family:
   `backtest/` (B, U), `metrics/` (M) or `forecast/` (MC).
4. **No skips.** An unknown `kind` or a malformed case fails. A case whose
   callable doesn't exist yet fails until it does.

## Fields

| Field | Content |
|-------|---------|
| `rule` | List of rule IDs the case enforces, e.g. `[B4.1, B1.4]`. Each must be a heading of the README Logic section. |
| `kind` | Which interface callable the runner calls (table below). |
| `inputs` | Arguments, in plain YAML. Missing prices are `null`. |
| `expected` | What the callable must return, plus the optional `warns` and `raises` checks below. |
| `tolerance` | For numbers: a number (absolute tolerance) or `{abs: ..., rel: ...}`; a number passes if $\lvert a - e \rvert \le \text{abs} + \text{rel}\thinspace\lvert e \rvert$. Labels, classes, flags and counts are always compared exactly; label-only cases use `tolerance: 0`. |

**Timestamps.** Write them as ISO strings. With a UTC offset
(`"2024-11-29T13:00:00-05:00"`) they are instants: inputs are converted to the
case's `index_tz` / `bars_tz` / `labels_tz` (default UTC) before the call, and
expected labels are compared as instants, so the offset you write is for the
reader. Without an offset they are naive: midnight dates for daily and longer
bars, or naive intraday times whose zone is passed as the callable's `tz`.
`expected.tz` additionally pins the output's time zone (B3: labels on the
exchange clock).

**Warnings and errors.** `expected.warns` is a list of entries: the
`UserWarning`s raised must be exactly those (each entry matches a warning,
each warning matches an entry); `warns: []` asserts there is none. Without
`warns`, warnings are not checked. An entry is a substring, or a mapping
`{match: <regex>, count: N}` when the spec fixes what a warning says but not
its wording (B1.6, B4.1, B4.4: a warning "gives the count"): the regex is
searched case-insensitively and N must stand alone in the message, not inside
a larger number or a date. `expected.deprecation` checks `DeprecationWarning`s
the same way (M10). `expected.raises` (`ValueError`,
`TypeError` or `KeyError`, with an optional regex `match`) means the call must
raise it.

## Kinds

Each kind calls one callable of the package, as the table gives it.

| `kind` | Call | `inputs` | `expected` |
|--------|------|----------|------------|
| `close_times` | `utils.calendar.close_times(index, calendar, tz)` | `index`, `index_tz`, `calendar`, `tz` | `labels`, `tz` |
| `session_spans` | `utils.calendar.session_spans(index, calendar, tz)` | as `close_times` | `first_open`, `last_close`, `n_sessions` |
| `completed_mask` | `utils.calendar.completed_mask(index, calendar, tz, now)` | as `close_times`, plus `now` | `mask` |
| `history_marks` | `utils.marks.history_marks(opens, closes, held, calendar, tz, fill)` | `bars`, `bars_tz`, `opens` and `closes` as `{symbol: [price or null, ...]}`, `held` as `{symbol: bool}`, `fill`, `tz` | `labels`, `tz`, `fresh` as `{symbol: [bool, ...]}` (listed symbols only), `classes` (one per return) |
| `is_daily` | `utils.marks.is_daily(values, calendar)` | `labels`, `labels_tz`, `values` | `value` |
| `classify_returns` | `utils.marks.classify_returns(values, calendar)` | `labels`, `labels_tz`, `values` | `labels` (the mark each return ends at), `classes` |
| `to_session_values` | `utils.marks.to_session_values(values, calendar, now=...)` | `labels`, `labels_tz`, `values`; `as_backtest_result: true` with `classes` wraps them in a `BacktestResult`; `now` (passed only when given) | `labels`, `values` |
| `next_trading_dates` | `utils.trading_dates.next_trading_dates(values, n, interval, calendar, tz)` | `labels`, `labels_tz`, `values`, `n`, `interval` | `labels`, `classes`, `tau` |
| `simulate_buy_and_hold` | `performance.simulate.simulate_buy_and_hold(data, w, br0, fill, calendar, tz, now)` | `bars`, `bars_tz`, `prices` as `{field: {symbol: [price or null, ...]}}` (fields `Open`, `Close`, `Adj Close`), `layout` (`fields`: yfinance's `(Price, Ticker)` columns, default; `closes`: flat close columns), `w` as `{symbol: weight}`, `br0`, `fill`, `tz`, `now` | `labels`, `tz`, `values`, `classes` (one per return) |
| `performance_report` | `performance.report.performance_report(values, classes, calendar=..., interval=..., risk_free_rate=..., cagr_clock=...)` | `labels`, `labels_tz`, `values`; `classes` (one per return), with `as_backtest_result: true` to pass a `BacktestResult` instead; `interval`, `risk_free_rate`, `cagr_clock`, `calendar` when given | `report` as `{figure name: number or {labels, values}}`, each name found exactly once anywhere in the nested dict; `yearly` as `{years, return, is_full}` for "Actual Yearly Returns" |
| `calculate_period_returns`, `calculate_daily_return` | `performance.returns.<name>(values)` | `labels`, `labels_tz`, `values` | `labels` (optional), `values`; `deprecation` for the alias (M10) |
| `calculate_session_returns` | `performance.returns.calculate_session_returns(values, calendar)` | `labels`, `labels_tz`, `values`, `calendar` | `labels` (optional), `values` |
| `returns_and_report` | `calculate_period_returns(values)`, `calculate_session_returns(values, calendar)` and `performance_report(values, calendar=...)` on the same history (M9 with M10) | `labels`, `labels_tz`, `values`, `calendar` | `period_returns` and `session_returns` as `{labels, values}` (`null` for a NaN row), `report` as `{figure name: number}` |
| `nonparametric_monte_carlo`, `parametric_monte_carlo`, `regime_switching_monte_carlo` | `forecast.<name>(values, sim_length=..., n_sims=..., random_state=..., classes=..., calendar=..., **options)` | the history as `labels`, `labels_tz`, `values` (with `classes`, or `as_backtest_result: true`), or `array` (a plain array), or `generate` (below); `sim_length`, `n_sims`, `random_state`, `calendar`; `options` for any other keyword (`min_sample`, `distribution` as a scipy name or `{gap: ..., regular: ...}`, `criterion`, `n_regimes`, `gap_model`, `kappa`, `bootstrap`, `return_fit`) | `shape`, `labels` (the result's `dates`), `classes`, `dates_none`, `fit_none`, `path` (every path equals it) or `paths`, `family` per class and `fit_report` as `{class: {criterion, candidates}}` (MC5: the score is recomputed as $-2\log L$ plus the penalty at the reported parameters on the class's pool, for a generated history of full bars), `moments`, `gap_draws`, `switch_rate`, `share_near_zero` (below); column 0 is always checked to be 1.0 |
| `statistical_report` | `performance.report.statistical_report(sims, calendar=..., interval=..., risk_free_rate=..., confidence=..., cagr_clock=..., display=...)` | `simulation` as `{paths, dates, dates_tz, classes}` (a hand-built `SimulationResult`, fit None), or `paths` with `interval` and optional `dates`; the keywords when given | `per_path` as `{figure: [one value per path]}` (the one per-path table with that column), `summary` as `{figure: {Mean, Median, Lower, Upper}}` (the one summary row of that figure), `report` as `{key: number}`, `printed` / `not_printed` phrases (with `display: true`), `docstring` phrases |
| `simulate_then_report` | `forecast.<simulator>(...)`, then `statistical_report(result, **report_options)` | as the simulator kinds, plus `simulator` (its name) and `report_options` | `per_path`, `report` as for `statistical_report` |

`values` inputs are a series of `values` on the index `labels`. `calendar`
defaults to `"XNYS"` everywhere; `calendar: null` is continuous mode (U3).

**Statistical cases (MC3-MC7).** A simulator's paths are random, so besides
the known-answer cases (zero-variance pools: every draw forced, every path
known) the seeded cases check statistics of the draws, read back as
$\ell = \ln(V_t/V_{t-1})$ at each future mark and grouped by class and by
$\rho = \tau/\Delta$ from the result's dates. Their history is generated by
the runner (`generate`, see `generated_history` in `test_golden.py`) from
`exchange_calendars` and numpy alone, never from the package, so the
population each class is drawn from is known exactly:

- `kind: sessions` (`calendar`, `start`, `sessions`, `grid`), `continuous`
  (`start`, `bars`, `grid`) or `array` (`returns`), with a `seed`;
- per class, `{mean, sd}` (normal), `{mean, sd, df}` (Student's $t$, `sd` the
  scale), `{constant: x}`, or for `regular` `{regimes: {P, means, sds}}`; a
  regular bar of ratio $\rho$ gets $\rho\thinspace\text{mean} + \sqrt\rho\thinspace\text{sd}\thinspace z$
  (the location–scale map of MC3); `pass_classes: true` passes the classes instead of letting
  the simulator infer them.

The expected values of a statistical case are population values derived by
hand from the spec, and each tolerance is derived in the case's comments from
the sampling error of the history and of the draws (four standard errors).
`moments: {group: {mean, var}}` with groups `gap`, `regular` ($\rho = 1$) and
`regular_short` ($\rho < 1$) takes its tolerances from `tolerance: {group:
{mean, var}}`; `switch_rate` (share of consecutive draws changing sign, for
regimes with means of opposite sign) and `share_near_zero` (share of
$\lvert\ell\rvert$ below `threshold`) take `{value, tol}`, `{max}` or `{min}`.
A case may record that the expected winner of a model selection holds on its
seed with a margin, checked by hand outside the package (scipy fits of the
generated data); that is a check of the case, not a source of its numbers.

**Assumptions about output layout.** The interface tables fix the callables
and the result types, not every key. Where a case relies on more, the runner
says so: `family` and `fit_report` read the `fit` keys documented in
`parametric_monte_carlo`'s docstring (`fit[class]` with `"Distribution"`,
`"Parameters"`, `"Criterion"`, the criterion's score and `"Candidates"`); `per_path` and
`summary` look for the one table holding the figure (0.3.0's "Per Path
Metrics" and the Mean/Median/Lower/Upper tables, keys unchanged under M11);
`Lower`/`Upper` are percentiles by linear interpolation between order
statistics, $h = (n-1)p$ (0.3.0's method; M7 adds only inf-safety).

## Worked example

`backtest/b3_half_day_hourly_labels.yaml` checks B3's half-day rule. The
derivation lives in the comments: the calendar facts it relies on, then one
line per expected value.

```yaml
# B3: hourly bar labels on a half-day, 2024-11-29 (the day after Thanksgiving).
#
# Derivation
# 1. XNYS closes early at 13:00 on 2024-11-29; the session runs 09:30-13:00
#    America/New_York, which is EST (UTC-05:00) after 2024-11-03.
# 2. The input bars are labelled at their start on a :30 hourly grid:
#    09:30, 10:30, 11:30, 12:30. The bar size is 1 hour.
# 3. B3: each value is labelled at its bar's close, "session close on a
#    half-day; the bar's actual end for a short last bar".
#    - 09:30 bar -> 09:30 + 1h = 10:30
#    - 10:30 bar -> 11:30
#    - 11:30 bar -> 12:30
#    - 12:30 bar -> 12:30 + 1h = 13:30 is after the 13:00 close, so the bar
#      ends at the session close: 13:00.
# 4. The input is tz-aware on the exchange clock, so the labels stay on it.
rule: [B3]
kind: close_times
inputs:
  index: ["2024-11-29T09:30:00-05:00", "2024-11-29T10:30:00-05:00",
          "2024-11-29T11:30:00-05:00", "2024-11-29T12:30:00-05:00"]
  index_tz: America/New_York
expected:
  labels: ["2024-11-29T10:30:00-05:00", "2024-11-29T11:30:00-05:00",
           "2024-11-29T12:30:00-05:00", "2024-11-29T13:00:00-05:00"]
  tz: America/New_York
tolerance: 0  # labels only, compared exactly
```

The runner converts `index` to a `DatetimeIndex` in `America/New_York`, calls
`close_times(index, calendar="XNYS", tz=None)`, and requires the four labels
exactly (as instants, with no tolerance), in `America/New_York`. A case with
numbers states its tolerance instead, e.g. `tolerance: {abs: 1.0e-12}` for
values that are copied rather than computed.

## Adding a case

1. Pick the rule and write the derivation first, in comments, from the spec
   alone.
2. Fill in `inputs` and `expected` from the derivation.
3. Check the case bites: it must fail against a deliberately wrong answer
   (an off-by-one label, a flipped class).
4. When a rule's cases cover all of it, remove its `_(pending)_` marker in the
   README Logic section.
