# Contributing to portfolio-forecast

Thanks for helping out. Bug reports, fixes, new simulators and documentation
improvements are all welcome. This guide covers setting up, the conventions
the code follows, and what a pull request needs.

## Reporting a bug or asking for a feature

Open an issue at
[github.com/YohoCodes/portfolio-forecast/issues](https://github.com/YohoCodes/portfolio-forecast/issues).
For a bug, include:

- the package version (`pip show portfolio-forecast`) and Python version
- a short, self-contained snippet that reproduces it, ideally on synthetic
  data rather than a download
- what you expected and what happened, with the full traceback

For a larger change, such as a new simulator or a change to a public
signature, open an issue to discuss it before writing the code.

## Setting up

You need Python 3.10 or later. From a fork of the repository:

```bash
git clone https://github.com/<your-username>/portfolio-forecast.git
cd portfolio-forecast
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[test,lint]"
```

Add `notebook` to the extras (`".[test,lint,notebook]"`) to run the demo
notebooks. The PDF report tests also need `pdflatex` on the PATH; without it
they are skipped.

## Before opening a pull request

Both of these must pass:

```bash
pytest
ruff check .
```

`ruff check --fix .` fixes import order and other mechanical issues for you.
Ruff's configuration is in `pyproject.toml`.

## Where the code lives

Everything is under `src/portfolio_forecast/`:

| Module | Contents |
| --- | --- |
| `types.py` | `BacktestResult`, `SimulationResult`, `FutureLayout` |
| `forecast/bootstrap.py` | `nonparametric_monte_carlo` |
| `forecast/parametric.py` | `parametric_monte_carlo` |
| `forecast/regime.py` | `regime_switching_monte_carlo` |
| `forecast/_common.py` | The simulators' shared path: input checks, classes, future layout, short-bar rescaling, path assembly, `resolve_values` |
| `forecast/_hmm.py` | The hidden Markov model's EM fit (private) |
| `performance/simulate.py` | `simulate_buy_and_hold`, `get_closing_prices` |
| `performance/returns.py` | `calculate_period_returns`, `calculate_session_returns`, `calculate_daily_return` (deprecated) |
| `performance/metrics.py` | CAGR, volatility, Sharpe, Sortino, drawdown, per-class volatility, yearly returns |
| `performance/report.py` | `performance_report`, `statistical_report` |
| `plotting/paths.py` | `plot_simulated_paths`, `plot_path_comparison` |
| `reporting/pdf.py` | `compile_statistical_reports` |
| `utils/calendar.py` | `close_times`, `session_spans`, `completed_mask` |
| `utils/marks.py` | Marks and return classes: `classify_returns`, `to_session_values`, `is_daily`, `history_marks`, the future layout |
| `utils/trading_dates.py` | `next_trading_dates` |
| `utils/periods.py` | `PERIODS_PER_YEAR` and bar-size parsing |
| `utils/bar_times.py` | Deprecated location of `close_times`, kept for 0.3.0 imports |

The README's [Project layout](README.md#project-layout) lists the same
files; keep both in step with `src/`.

## Code style

- **PEP 8**, checked by Ruff, with a line length of 120.
- **`zip()` always takes `strict=`.** Use `strict=True` when the inputs
  should always be the same length, so a mismatch raises instead of silently
  dropping items. Use `strict=False` only where truncating is intended, such
  as zipping with `itertools.cycle`.
- **Comments and docstrings** follow the
  [Commenting standard](#commenting-standard) below.
- **Private helpers** start with an underscore and are listed in the
  README's [Internal helpers](README.md#internal-helpers) table.
- **Randomness** goes through a `random_state` argument passed to
  `np.random.default_rng`, so every result can be reproduced from a seed.

## Commenting standard

Anyone new to the code, quant or maintainer, should be able to follow each
module's flow from its docstrings and comments alone. Tags such as [O13] point
to the sources at the end of this section.

### Docstrings

- **Every module** opens with a docstring: one line on its purpose, then its
  flow in a few numbered steps naming the functions involved [PEP257, O13].
- **Every function**, public or private, has a docstring in triple double
  quotes. The first line is one imperative sentence ("Return bars per year
  for a bar size."), not a restated signature [PEP257, pandas].
- **Public functions** use the full [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
  layout, in this order, omitting sections that don't apply: summary,
  extended description, Parameters, Returns, Raises, Warns, See Also,
  Notes, References, Examples [numpydoc]. Private helpers, classes and
  nested functions may drop the section headers, but still say inputs →
  outputs and what each attribute means [O13].
- **Parameters** read `name : type, default x` (or `type, optional`) and say
  what the value means, not just its type. Examples are seeded and
  deterministic [pandas].

### Units and conventions

State every unit and convention a reader could get wrong [O13]:

- **Returns:** simple ($r = V_t/V_{t-1} - 1$) or log ($\ell = \ln(1+r)$);
  per bar, per session or annualized; fraction or percent.
- **Time:** what a label means (a bar's value is labelled at its close), on
  which clock (exchange time, tz-aware), and how time is counted (sessions,
  bars or calendar days).
- **Annualizers:** what is counted (252 sessions, 52 weeks, 12 months), and
  that these are conventions, not counts of a real year.

### Math

Docstrings are read raw in terminals and editors, so math there is plain text
or Unicode (`ℓ = ln(1 + r)`, `σ√252`), not LaTeX, which reads badly raw and
needs raw strings for its backslashes [numpydoc]. Derivations go in Notes.
Markdown files (README, docs) use TeX.

### Inline comments: what and why

Comments say **what** a group of code does, in domain terms, and **why**
where it isn't obvious, so a reader can follow layered code without opening
every helper [OM, O13]. Each sits on its own line(s) **above** the group it
explains, indented with the code [PEP8, Google 3.8.5].

- **Public functions:** a short comment opens each stage, saying what the
  stage produces. Stages are never numbered, so inserting one renumbers nothing [O16].
- **Calls to private or local helpers:** a comment above the call says what
  comes back (and why, if that isn't obvious), so the reader needn't open
  the helper. Never a trailing comment on the call [OM, PEP8].
- **Private and local functions, classes, nested functions** get the most:
  a docstring with inputs → outputs and what each attribute means, and a
  one-line comment above each group of related statements or attributes
  [O13, Google 3.8.5].
- **Length:** one line by default; two only when the why won't fit; three
  rarely. Trivial code gets no comment. If code needs a long comment on
  *how* it works, make the code clearer instead [Wilson 7b].
- **Notation** matches the README: $t$ indexes marks, $T$ is the chain
  length, $k$ is only a parameter count, and a path is
  $V_t = V_0 \exp(\ell_1 + \dots + \ell_t)$ [Wilson 7a].

### Citing rules and papers

- Where code implements a rule from the README's [Logic](README.md#logic)
  section, state the rule in words, so the comment stands alone:
  `# The gap runs from the last close to the next open`. Never cite the
  rule's ID: an ID means nothing to a reader who hasn't left the page. Rule
  IDs appear only in the README's Logic section and the golden tests
  [Wilson 7a, O13].
- Never cite a decision number, a design or specification document, or
  any file outside the repository [Wilson 7a].
- Cite a paper or book by author and year where it is used
  (`# Hamilton (1989): ...`) and give the full reference under the
  docstring's References. The docstring must make sense without them; don't
  cite web pages that may move [numpydoc].

### Invariants and edge cases

Write down what must stay true, and the cases that break naive code, where
the code relies on them [O13]: inclusive or exclusive boundaries, empty or
one-element input, half-day sessions, DST changes, holidays, stale
(forward-filled) prices, partial first or last bars.

### Comments to avoid

- Restating syntax: `# loop over classes`, `# Creating a matrix to store the
  returns`. Say what the group yields instead [PEP8, OM].
- Change history and authorship; they go in the commit message and changelog
  [numpydoc, O16].
- Vague hedges (`# just in case`): name the case or remove the code [O13].

### TODOs

`# TODO: <issue link> - <what, and when it becomes actionable>.` No TODO is
merged without an issue [Google 3.12].

### Keeping comments true

A comment that contradicts the code is worse than none [PEP8]. Update comments
and docstrings in the same commit as the code they describe, keep each next to
the code it explains, and read their diff before committing [O16, Wilson 7c].

### Examples from this codebase

These are the reference style. The third is closest to what every part of
the code should read like: short lines that walk the reader through
non-trivial logic.

A public-function stage: one short line naming the stage and what it yields
(`forecast/regime.py`).

```python
# Fit each candidate regime count by EM and score it; skip any that fail or lack data
counts = [int(n_regimes)] if n_regimes is not None else list(_SELECTION_COUNTS)
```

A private-helper call: what comes back and, where it isn't obvious, why;
above the call, not trailing it (`forecast/bootstrap.py`).

```python
if cls == REGULAR:
    # Pool draws are per full bar; shrink them to each future bar's length.
    # Gaps count as one unit whatever their length, so only regular marks map
    draws = _common.from_unit(draws, history.future_rho[cols], history.m_hat)
```

A private class body: one line above each group of related statements or
attributes, saying what the group holds or does (`forecast/regime.py`).

```python
# Gaps ride the chain only under "regime" and only if the history has any
self.joint = gap_model == "regime" and has_gaps
# The chain's marks: all kept marks if joint, else regular marks only
on_chain = np.ones(len(history.ell), dtype=bool) if self.joint else history.classes == REGULAR
self.ell, self.rho = history.ell[on_chain], history.rho[on_chain]
# Class code per chain mark (0 regular, 1 gap); None with one class
self.cls = (history.classes[on_chain] == GAP).astype(int) if self.joint else None
```

### Sources

- [PEP8] [PEP 8, Comments](https://peps.python.org/pep-0008/#comments): full
  sentences; block comments indented with the code; inline comments sparingly;
  contradicting comments worse than none.
- [PEP257] [PEP 257](https://peps.python.org/pep-0257/): triple double quotes;
  imperative summary line, not a signature; module docstrings summarize.
- [numpydoc] [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html):
  section order; plain-text math over LaTeX; References supplement, never
  carry, the text; no temporary web references; no author info.
- [pandas] [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html):
  infinitive summary; `type, default x` and `optional`; seeded Examples. The
  [NumPy](https://numpy.org/devdocs/dev/howto-docs.html) and
  [SciPy](https://docs.scipy.org/doc/scipy/dev/contributor/rendering_documentation.html)
  guides defer to numpydoc, so we add no house variant.
- [O13, O16] J. Ousterhout, [*A Philosophy of Software Design*](https://web.stanford.edu/~ouster/cgi-bin/book.php),
  2nd ed., 2021. Ch. 13: comment what isn't obvious from the code, precisely
  (units, boundaries, invariants) and at the level of intent. Ch. 16: keep
  comments by their code and current with it; history goes in the commit log.
- [Wilson] G. Wilson et al., [Best Practices for Scientific Computing](https://doi.org/10.1371/journal.pbio.1001745),
  *PLOS Biology*, 2014, practice 7: (a) document interfaces and reasons, not
  implementations; (b) refactor rather than explain; (c) keep docs in code.
- [OM] J. Ousterhout and R. Martin, [*A Philosophy of Software Design* vs
  *Clean Code*](https://github.com/johnousterhout/aposd-vs-clean-code):
  a comment should "say what the code is doing in a logical sense, not how it
  does it"; readers shouldn't have to load every helper to follow a method.
- [Google] [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  §3.8.5: a comment before each complicated operation, saying what you'd
  explain in review; §3.12: TODOs carry an issue.

## README reference

The README is the package's API reference. When you add or change a public
function, update its entry under [API reference](README.md#api-reference) in
the same pull request: the signature block, a Parameters table (Name, Type,
Default, Description), then **Returns**, **Raises**, **Notes** and
**See also** as needed. Follow the existing entries.

## Tests

- Tests live in `tests/test_<module>.py` (the simulators' in
  `tests/forecast/`), grouped into classes by the behaviour they cover
  (`class TestDeprecatedPrices:`). Hand-derived known-answer cases for the
  README's Logic rules are in `tests/golden/`; see its README.
- Shared fixtures are in `tests/conftest.py`. Use the `prices` fixture or
  another small synthetic series; tests must not download data.
- Test names state the behaviour being checked:
  `test_prices_warns_and_matches_values`, not `test_prices_2`.
- A bug fix comes with a test that fails without the fix.
- Keep tests fast: small `sim_length` and `n_sims`, and few `n_starts` for
  the regime-switching model.

## Demo notebooks

The demo notebooks live in `demos/`, named `NN_topic.ipynb` with a
two-digit number giving the reading order (`01_quick_start_daily.ipynb`).
They share one layout, so follow it in a new demo:

- A title cell, `# **Portfolio Forecast Demo: <Topic>**`, with a sentence on
  what the demo does and a numbered list of its steps.
- An `## **Imports**` section first, then one `##` section per step, with
  `###` for subsections. Headings are bold and in Title Case.
- Every code cell except the imports starts with a comment saying what it
  does or why.
- Single-quoted strings, and seeds passed as
  `random_state=np.random.default_rng(42)`.
- Commit the notebook with its outputs, run top to bottom, and with no empty
  trailing cells. Ruff checks notebooks too.

## Changing a public signature

Public names don't break without warning. To rename or remove a parameter:

1. Add the new name and keep the old one working, as a keyword-only
   argument that raises a `DeprecationWarning` naming the version it will be
   removed in. Use `stacklevel` so the warning points at the caller's code.
   See `resolve_values` in `forecast/_common.py`, which the simulators
   call for their deprecated `prices` keyword, for an example.
2. Document the old name in the docstring with a
   `.. deprecated:: <version>` note, and add it to the README table.
3. Add tests that the old name warns and gives identical results.
4. Remove the old name at the next major version.

## Pull requests

- Branch from `main` and keep each pull request to one change.
- Write the description as what changed and why, and link the issue it
  addresses.
- If users would notice the change, add a line under `## [Unreleased]` in
  [CHANGELOG.md](CHANGELOG.md), in the right section (Breaking, Added,
  Changed, Deprecated, Removed or Fixed), written for users: what they will see and
  what, if anything, they need to do. Internal refactors and test-only
  changes don't need one.
- Don't bump the version number in `pyproject.toml`; that happens at
  release.
- `PYPI.md` is the description shown on PyPI, which doesn't render math:
  a short introduction, installation and the README's daily quick start,
  with no math. If you change the README's quick start or installation,
  update `PYPI.md` to match in the same pull request, and check it before a
  release with `python -m build` and `twine check dist/*`.

By contributing, you agree that your work is licensed under the project's
[MIT License](LICENSE).
