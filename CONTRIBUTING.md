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

Add `notebook` to the extras (`".[test,lint,notebook]"`) to run `demo.ipynb`.
The PDF report tests also need `pdflatex` on the PATH; without it they are
skipped.

## Before opening a pull request

Both of these must pass:

```bash
pytest
ruff check .
```

`ruff check --fix .` fixes import order and other mechanical issues for you.
Ruff's configuration is in `pyproject.toml`.

## Code style

- **PEP 8**, checked by Ruff, with a line length of 120.
- **`zip()` always takes `strict=`.** Use `strict=True` when the inputs
  should always be the same length, so a mismatch raises instead of silently
  dropping items. Use `strict=False` only where truncating is intended, such
  as zipping with `itertools.cycle`.
- **Comments** explain why a step is there, not just what it does. Match the
  density of the surrounding code.
- **Private helpers** start with an underscore and are listed in the
  README's [Internal helpers](README.md#internal-helpers) table.
- **Randomness** goes through a `random_state` argument passed to
  `np.random.default_rng`, so every result can be reproduced from a seed.

## Docstrings

Every public function has a [NumPy-style docstring](https://numpydoc.readthedocs.io/en/latest/format.html)
with these sections, in this order, omitting any that don't apply:

```
Summary line.

Extended description.

Parameters
Returns
Raises
Warns
Notes
See Also
Examples
```

Describe each parameter's type and default (`sim_length : int, default 100`)
and say what the value means, not just what type it is.

## README reference

The README is the package's API reference. When you add or change a public
function, update its entry under [API reference](README.md#api-reference) in
the same pull request: the signature block, a Parameters table (Name, Type,
Default, Description), then **Returns**, **Raises**, **Notes** and
**See also** as needed. Follow the existing entries.

## Tests

- Tests live in `tests/test_<module>.py`, grouped into classes by the
  behaviour they cover (`class TestDeprecatedPrices:`).
- Shared fixtures are in `tests/conftest.py`. Use the `prices` fixture or
  another small synthetic series; tests must not download data.
- Test names state the behaviour being checked:
  `test_prices_warns_and_matches_values`, not `test_prices_2`.
- A bug fix comes with a test that fails without the fix.
- Keep tests fast: small `sim_length` and `n_sims`, and few `n_starts` for
  the regime-switching model.

## Changing a public signature

Public names don't break without warning. To rename or remove a parameter:

1. Add the new name and keep the old one working, as a keyword-only
   argument that raises a `DeprecationWarning` naming the version it will be
   removed in. Use `stacklevel` so the warning points at the caller's code.
   See `_resolve_values` in `forecast/monte_carlo.py` for an example.
2. Document the old name in the docstring with a
   `.. deprecated:: <version>` note, and add it to the README table.
3. Add tests that the old name warns and gives identical results.
4. Remove the old name at the next major version.

## Pull requests

- Branch from `main` and keep each pull request to one change.
- Write the description as what changed and why, and link the issue it
  addresses.
- Don't bump the version number in `pyproject.toml`; that happens at
  release.

By contributing, you agree that your work is licensed under the project's
[MIT License](LICENSE).
