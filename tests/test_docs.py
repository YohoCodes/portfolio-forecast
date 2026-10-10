"""Checks on the README and the PyPI description.

Some mistakes are easy to make when editing the docs and invisible
until they are published, so the tests catch them first:

- Math rendering on GitHub. The README writes math as ``$...$`` and
  ``$$...$$``, which GitHub renders only after its markdown pass. A table
  pipe inside a span, a double backslash, or an ``_`` or ``*`` between two
  spans on the same line is taken by markdown first and breaks the math.
  PyPI renders no math at all, so ``PYPI.md`` must contain no ``$``.
- PyPI description drift. ``PYPI.md`` is the description shown on PyPI and
  repeats the README's quick start. Its first ``python`` block under
  ``## Quick start`` must equal the README's, character for character, so
  an edit to one cannot silently leave the other stale.
- Undefined symbols. "How the simulators work", "Tuning the simulators" and
  each Logic subsection (Backtest, Return classes and units, Metrics, Monte
  Carlo) have a Notation table. Every symbol in a section's math (a letter
  or Greek command, with its accent and primes; the letters in its
  subscripts count as symbols too) must have a row in that section's table,
  or be named in the "Conventions" line under it. A table that opens "as in
  Notation" adds to the simulators' table, and its own rows come first.
- A stale table of contents. Every ``#anchor`` link in the README's table of
  contents must reach a heading (by GitHub's anchor rules), and every heading
  at levels 2 to 4 must be listed there.

If one of these tests fails, fix the doc it names; the failure message gives
the file, the line and the offending text.
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Static math lint on the README and the PyPI description: GitHub renders
# ``$...$`` spans only after its markdown pass, so table pipes, double
# backslashes and emphasis markers between two spans break the rendering;
# PyPI renders no math at all.

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"(`+)(.+?)\1")
MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$")


def _blank(match):
    """The matched text replaced by spaces, keeping column positions."""
    return " " * len(match.group(0))


def math_problems(text):
    """[(line number, problem)] of math that GitHub would render wrongly.

    Fenced code and inline code are skipped. A ``$$`` line opens or closes
    a display block; other lines are split into ``$...$`` (or ``$$...$$``)
    spans in order.
    """
    problems, in_fence, in_display = [], False, False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.strip() == "$$":
            in_display = not in_display
            continue
        if in_display:
            if "\\\\" in line:
                problems.append((number, "'\\\\' inside display math"))
            continue
        clean = INLINE_CODE.sub(_blank, line).replace("\\$", "  ")
        spans = list(MATH_SPAN.finditer(clean))
        table_row = clean.lstrip().startswith("|")
        for span in spans:
            body = span.group(1) if span.group(1) is not None else span.group(2)
            if "\\\\" in body:
                problems.append((number, f"'\\\\' inside math {span.group(0)!r}"))
            if table_row and "|" in body:
                problems.append((number, f"'|' inside math in a table row {span.group(0)!r}"))
        for left, right in zip(spans, spans[1:], strict=False):
            between = clean[left.end():right.start()]
            for marker in "_*":
                if marker in between:
                    problems.append((number, f"{marker!r} between math spans: {between.strip()!r}"))
    if in_display:
        problems.append((len(text.splitlines()), "unclosed '$$' display block"))
    return problems


@pytest.mark.parametrize("name", ["README.md", "PYPI.md"])
def test_math_renders_on_github(name):
    """No table pipe, double backslash or emphasis marker breaks the math."""
    found = [f"{name}:{n}: {problem}" for n, problem in math_problems((ROOT / name).read_text())]
    assert not found, "math that will not render:\n" + "\n".join(found)


def test_pypi_description_has_no_math():
    """PyPI renders no math, so its description has no '$' at all."""
    found = [
        f"PYPI.md:{n}: {line.strip()}"
        for n, line in enumerate((ROOT / "PYPI.md").read_text().splitlines(), start=1)
        if "$" in line
    ]
    assert not found, "'$' in PYPI.md:\n" + "\n".join(found)


@pytest.mark.parametrize(
    "text",
    [
        "| a | $|x|$ | b |",
        "| a | $\\lVert x \\rVert + |y|$ |",
        "inline $a \\\\ b$ here",
        "$$\na \\\\\nb\n$$",
        "both $a$ and_b and $c$",
        "both $a$ *and* $c$",
        "both $$a$$ x_y $c$",
    ],
)
def test_math_lint_catches_examples(text):
    """Each math rule catches a typical violation."""
    assert math_problems(text)


@pytest.mark.parametrize(
    "text",
    [
        "| a | $\\tau/\\Delta$ | b |",
        "a line $|x|$ outside a table",
        "code `$a$ x_y $b$` is literal",
        "```\n$a$ x_y $b$ \\\\\n```",
        "$a_1$ and $b_2^*$ keep markers inside math",
        "a cost of \\$5 and $x$ and \\$6_",
        "$$\n\\sum_t \\ell_t\n$$",
    ],
)
def test_math_lint_spares_valid_math(text):
    """Markers inside math, code and escaped dollars are left alone."""
    assert not math_problems(text)


def quick_start_code(text):
    """The first ``python`` block under ``## Quick start``, or None."""
    section = re.search(r"^## Quick start\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if section is None:
        return None
    block = re.search(r"^```python\n(.*?)^```", section.group(1), re.M | re.S)
    return block.group(1) if block else None


def test_pypi_quick_start_matches_readme():
    """PyPI's quick start is the README's first quick-start block, verbatim."""
    readme = quick_start_code((ROOT / "README.md").read_text())
    pypi = quick_start_code((ROOT / "PYPI.md").read_text())
    assert readme is not None, "README.md has no python block under '## Quick start'"
    assert pypi is not None, "PYPI.md has no python block under '## Quick start'"
    assert pypi == readme, "PYPI.md's quick start differs from the README's"


def test_quick_start_code_takes_the_first_block_of_the_section():
    """Blocks before the section and after the first one are ignored."""
    text = (
        "```python\nbefore\n```\n## Quick start\n```bash\nx\n```\n"
        "```python\nfirst\n```\n```python\nsecond\n```\n## Next\n"
    )
    assert quick_start_code(text) == "first\n"
    assert quick_start_code("## Quick start\ntext only\n## Next\n```python\nlater\n```\n") is None


# Notation: every symbol in the math of "How the simulators work" has a row in
# its Notation table, so a reader never meets a letter the section leaves
# undefined.

GREEK = frozenset(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi "
    "pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega "
    "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega ell".split()
)
ACCENTS = frozenset("hat widehat bar overline tilde widetilde dot ddot check vec breve acute grave".split())
FONTS = frozenset("mathcal mathbb mathbf mathrm mathit mathsf boldsymbol".split())
TEXT_ARG = re.compile(r"\\(?:text|textrm|textit)\s*\{([^{}]*)\}")
MATH_TOKEN = re.compile(r"\x00[^\x00]*\x00|\\[A-Za-z]+|\\.|\s+|.", re.S)
WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
HEADING = re.compile(r"^(#{1,6}) +(.+?) *#*$")


class Symbol(NamedTuple):
    """One symbol of some math: its `name` (``\\hat\\sigma``), whether it
    sits in a script, how it is `written` with its scripts (``\\hat\\sigma^2_{s,c}``),
    and for a script letter, how the symbol it indexes is written."""

    name: str
    in_script: bool
    written: str
    parent: str | None


def math_tokens(math):
    """The math split into commands, single characters and braces, no spaces;
    each ``\\text{...}`` is one token, ``"\\x00"`` + its text + ``"\\x00"``."""
    math = TEXT_ARG.sub(lambda m: "\x00" + m.group(1) + "\x00", math)
    return [token for token in MATH_TOKEN.findall(math) if not token.isspace()]


def _group(tokens, i):
    """(tokens of the atom at `i`, index after it): a braced group or one token."""
    if i >= len(tokens):
        return [], i
    if tokens[i] != "{":
        return [tokens[i]], i + 1
    depth, j = 0, i
    while j < len(tokens):
        depth += {"{": 1, "}": -1}.get(tokens[j], 0)
        if depth == 0:
            return tokens[i + 1:j], j + 1
        j += 1
    return tokens[i + 1:], len(tokens)


def _symbol_name(prefix, base, primes):
    """A symbol written back as TeX: ``\\hat\\sigma``, ``\\hat m``, ``\\ell'``."""
    joiner = " " if prefix and not base.startswith("\\") else ""
    return f"{prefix}{joiner}{base}{primes}"


def _written(tokens):
    """Tokens joined back into TeX without spaces, for comparing symbols as written."""
    return "".join(f"\\text{{{t.strip(chr(0))}}}" if t.startswith("\x00") else t for t in tokens)


def _parse(tokens, found, index_mode=False):
    """Append a `Symbol` to `found` for every symbol in `tokens`.

    A symbol is a Latin letter or a Greek command (``\\ell`` counts as one),
    with its accent or font command and its primes; its subscripts and
    superscripts are scanned for symbols of their own (the index letters).
    A ``\\text{...}`` word is a symbol, except inside a script, where it is a
    label such as ``gap``; text with spaces is prose. Every other command
    (``\\frac``, ``\\sum``, ``\\le``, ...) is notation, not a symbol.
    """
    i = 0
    while i < len(tokens):
        token, start = tokens[i], i
        base = prefix = None
        if token in ("_", "^"):
            script, i = _group(tokens, i + 1)
            _parse(script, found, index_mode=True)
            continue
        if token.startswith("\x00"):
            word, i = token.strip("\x00"), i + 1
            if not index_mode and WORD.fullmatch(word):
                found.append(Symbol(f"\\text{{{word}}}", index_mode, _written([token]), None))
            continue
        if token.startswith("\\") and token[1:] in ACCENTS | FONTS:
            atom, i = _group(tokens, i + 1)
            if len(atom) == 1:
                prefix, base = token, atom[0]
            else:
                _parse(atom, found, index_mode)
                continue
        elif token.startswith("\\") and token[1:] in GREEK:
            base, i = token, i + 1
        elif len(token) == 1 and token.isalpha():
            base, i = token, i + 1
        else:
            i += 1
            continue
        primes, scripts = "", []
        while i < len(tokens) and tokens[i] in ("'", "_", "^"):
            if tokens[i] == "'":
                primes += "'"
                i += 1
            else:
                script, i = _group(tokens, i + 1)
                _parse(script, scripts, index_mode=True)
        written = _written(tokens[start:i])
        found.extend(item._replace(parent=item.parent or written) for item in scripts)
        found.append(Symbol(_symbol_name(prefix or "", base, primes), index_mode, written, None))


def math_symbols(math):
    """Every symbol in one piece of math, in order, with repeats."""
    return [item.name for item in parsed(math)]


def parsed(math):
    """Every `Symbol` in one piece of math, in order."""
    found = []
    _parse(math_tokens(math), found)
    return found


def head_symbol(math):
    """The first symbol outside a script, so ``$\\gamma_t(s)$`` gives ``\\gamma``."""
    return next((item for item in parsed(math) if not item.in_script), None)


def math_in(lines):
    """[(line number, math)] for each ``$...$`` span and ``$$`` display block.

    `lines` is [(line number, text)]. Fenced and inline code are skipped and
    ``\\$`` is a literal dollar.
    """
    spans, in_fence, display, start = [], False, None, 0
    for number, line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.strip() == "$$":
            if display is None:
                display, start = [], number
            else:
                spans.append((start, " ".join(display)))
                display = None
            continue
        if display is not None:
            display.append(line)
            continue
        clean = INLINE_CODE.sub(_blank, line).replace("\\$", "  ")
        for span in MATH_SPAN.finditer(clean):
            spans.append((number, span.group(1) if span.group(1) is not None else span.group(2)))
    return spans


def headings(text):
    """[(line index, level, title)] of the markdown headings outside code."""
    found, in_fence = [], False
    for index, line in enumerate(text.splitlines()):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (match := HEADING.match(line)):
            found.append((index, len(match.group(1)), match.group(2)))
    return found


def section(text, title):
    """[(line number, text)] of the section headed `title`, heading excluded,
    up to the next heading of the same or a higher level."""
    lines = text.splitlines()
    marks = headings(text)
    for k, (index, level, name) in enumerate(marks):
        if name == title:
            end = next((i for i, lv, _ in marks[k + 1:] if lv <= level), len(lines))
            return [(n + 1, lines[n]) for n in range(index + 1, end)]
    raise AssertionError(f"README.md has no heading {title!r}")


def table_rows(lines):
    """The body rows (header and rule skipped) of the first table in `lines`,
    as [(line number, [cells])], and the index in `lines` just after it."""
    start = next((k for k, (_, line) in enumerate(lines) if line.lstrip().startswith("|")), None)
    if start is None:
        return [], 0
    end = start
    while end < len(lines) and lines[end][1].lstrip().startswith("|"):
        end += 1
    rows = []
    for number, line in lines[start + 2:end]:
        clean = INLINE_CODE.sub(_blank, line)
        cells = [cell.strip() for cell in clean.strip().strip("|").split(" | ")]
        rows.append((number, cells))
    return rows, end


def defined_symbols(cell):
    """The symbols a Symbol cell defines: the head of each math span, both
    by name and as written, and each plain word (``BIC, AIC``) as
    ``\\text{...}``. A row written with its scripts, such as ``$r_f$``, also
    covers the letters of those scripts wherever it is written that way."""
    heads = [head_symbol(m.group(2) or m.group(1)) for m in MATH_SPAN.finditer(cell)]
    defined = {item.name for item in heads if item} | {item.written for item in heads if item}
    words = MATH_SPAN.sub(" ", cell).replace(",", " ").split()
    return defined | {f"\\text{{{word}}}" for word in words if WORD.fullmatch(word)}


LEAD_IN = "**Notation.**"
INHERITS = "as in [Notation](#notation)"
LOGIC_SCOPES = ("Backtest", "Return classes and units", "Metrics", "Monte Carlo")


def notation(lines):
    """(defined, allowed, rest, inherits) for a scope whose Notation table is in `lines`.

    The table is the first after a line starting ``**Notation.**`` (or the
    first in `lines` when there is no such line). `defined` holds the symbols
    of its Symbol column; `allowed` those in the math of the paragraph right
    after it (the "Conventions" line, or one starting "Also"), which names
    operators and fixed meanings; `rest` is `lines` without the lead-in, the
    table and that paragraph; `inherits` says whether the lead-in reads
    "as in Notation", so the scope may also use the simulators' symbols.
    """
    lead = next((k for k, (_, line) in enumerate(lines) if line.startswith(LEAD_IN)), None)
    first = 0 if lead is None else lead
    start = next((k for k in range(first, len(lines)) if lines[k][1].lstrip().startswith("|")), None)
    assert start is not None, "no Notation table found"
    rows, end = table_rows(lines[start:])
    end += start
    defined = set().union(*(defined_symbols(cells[0]) for _, cells in rows))
    k = end
    while k < len(lines) and not lines[k][1].strip():
        k += 1
    stop = k
    if k < len(lines) and re.match(r"(Conventions:|Also )", lines[k][1]):
        while stop < len(lines) and lines[stop][1].strip():
            stop += 1
    allowed = {item.name for _, math in math_in(lines[k:stop]) for item in parsed(math)}
    inherits = lead is not None and INHERITS in lines[lead][1]
    return defined, allowed, lines[:first] + lines[stop:], inherits


def undefined_symbols(scope, defined, allowed):
    """[(symbol, first line)] of symbols in `scope` math neither defined nor
    allowed. A script letter is excused when the symbol it indexes has a row
    written the same way (``$r_f$`` covers the ``f``)."""
    first = {}
    for number, math in math_in(scope):
        for item in parsed(math):
            if item.name in defined or item.name in allowed or (item.parent and item.parent in defined):
                continue
            first.setdefault(item.name, number)
    return sorted(first.items(), key=lambda pair: pair[1])


def simulators_scope(text):
    """(defined, allowed, scope lines) of "How the simulators work": the
    ``### Notation`` subsection holds the table, the rest is checked."""
    defined, allowed, _, _ = notation(section(text, "Notation"))
    body = section(text, "How the simulators work")
    table = {number for number, _ in section(text, "Notation")}
    heading = next(number for number, line in body if line == "### Notation")
    scope = [(n, line) for n, line in body if n not in table and n != heading]
    return defined, allowed, scope


def notation_scopes(text):
    """{scope name: (defined, allowed, scope lines)} for each section with a
    Notation table: "How the simulators work", "Tuning the simulators" and the
    four Logic subsections. Each is checked against its own table. A scope
    whose table opens "as in Notation" may also use the simulators' symbols;
    its own rows come first, so a letter it redefines (a row of its own for a
    letter the simulators' table also has) takes its own meaning there."""
    base_defined, base_allowed, base_scope = simulators_scope(text)
    scopes = {"How the simulators work": (base_defined, base_allowed, base_scope)}
    for name in ("Tuning the simulators", *LOGIC_SCOPES):
        defined, allowed, scope, inherits = notation(section(text, name))
        if inherits:
            defined, allowed = defined | base_defined, allowed | base_allowed
        scopes[name] = (defined, allowed, scope)
    return scopes


def row_names(lines):
    """The symbol names (not written forms) of the first Notation table in `lines`."""
    names = set()
    for _, cells in table_rows(lines)[0]:
        heads = [head_symbol(m.group(2) or m.group(1)) for m in MATH_SPAN.finditer(cells[0])]
        names |= {item.name for item in heads if item}
    return names


def overrides(text, name):
    """Symbols the scope `name` gives a row of its own although the
    simulators' table has one too: there its own row takes precedence."""
    lines = section(text, name)
    lead = next(k for k, (_, line) in enumerate(lines) if line.startswith(LEAD_IN))
    return sorted(row_names(lines[lead:]) & row_names(section(text, "Notation")))


def missing_report(name, missing):
    """A failure message naming each undefined symbol and where it is first used."""
    return f"{name}: symbol(s) missing from its Notation table:\n" + "\n".join(
        f"  {symbol} (README.md:{number})" for symbol, number in missing)


SCOPE_NAMES = ("How the simulators work", "Tuning the simulators", *LOGIC_SCOPES)


@pytest.mark.parametrize("name", SCOPE_NAMES)
def test_notation_is_complete(name):
    """Every symbol in the section's math has a row in its Notation table."""
    defined, allowed, scope = notation_scopes((ROOT / "README.md").read_text())[name]
    missing = undefined_symbols(scope, defined, allowed)
    assert not missing, missing_report(name, missing)


def test_math_symbols_reads_accents_primes_scripts_and_text():
    """Scripts give their index letters; labels, prose and operators give none."""
    assert math_symbols(r"\hat\sigma^2_{s,c} = \ell'_t") == ["s", "c", r"\hat\sigma", "t", r"\ell'"]
    assert math_symbols(r"\mathcal N(\mu_{s,\text{gap}}, 1)") == [r"\mathcal N", "s", r"\mu"]
    assert math_symbols(r"\text{BIC} = -2\log L + k\ln n \qquad\text{(a full bar)}") == [
        r"\text{BIC}", "L", "k", "n"]
    assert math_symbols(r"\{s_\text{first},\dots\}") == ["s"]
    assert math_symbols(r"V(\text{last mark of } y)") == ["V", "y"]
    assert math_symbols(r"\sum_{t=1}^{n-1} \xi_t(i,j) \le \frac{1}{\sqrt\rho}") == [
        "t", "n", "t", r"\xi", "i", "j", r"\rho"]
    assert head_symbol(r"\gamma_t(s)").name == r"\gamma"
    assert head_symbol(r"\log L").name == "L"
    assert head_symbol(r"\hat\sigma^2_{s,c}").written == r"\hat\sigma^2_{s,c}"
    assert defined_symbols(r"$\theta$, $\hat\theta_c$") == {r"\theta", r"\hat\theta", r"\hat\theta_c"}
    assert defined_symbols("BIC, AIC") == {r"\text{BIC}", r"\text{AIC}"}


PLANT_AFTER = {
    "How the simulators work": "### What all three share\n",
    "Tuning the simulators": "### `kappa` (regime-switching)\n",
    "Backtest": "#### B2 Purchase\n",
    "Return classes and units": "#### U1 Classes\n",
    "Metrics": "#### M6 Max drawdown\n",
    "Monte Carlo": "#### MC8 Future marks\n",
}


@pytest.mark.parametrize("name", SCOPE_NAMES)
def test_planted_undefined_symbol_is_named(name):
    """A symbol with no row in the scope's table fails, and the message names it."""
    text = (ROOT / "README.md").read_text().replace(
        PLANT_AFTER[name], PLANT_AFTER[name] + "\nA rate $\\zeta_t$ and $\\hat r$.\n", 1)
    defined, allowed, scope = notation_scopes(text)[name]
    missing = undefined_symbols(scope, defined, allowed)
    assert [symbol for symbol, _ in missing] == [r"\zeta", r"\hat r"]
    assert r"\zeta" in missing_report(name, missing)


def test_only_an_inheriting_scope_uses_the_simulators_symbols():
    """$\\kappa$ has a row in the simulators' table only: Monte Carlo, which
    opens "as in Notation", may use it; Backtest may not."""
    text = (ROOT / "README.md").read_text()
    for name, ok in (("Monte Carlo", True), ("Backtest", False)):
        planted = text.replace(PLANT_AFTER[name], PLANT_AFTER[name] + "\nWith $\\kappa$.\n", 1)
        defined, allowed, scope = notation_scopes(planted)[name]
        assert (undefined_symbols(scope, defined, allowed) == []) is ok, name


def test_a_row_written_with_scripts_covers_their_letters():
    """``$r_f$`` covers the f of $r_f$, not a lone $f$."""
    lines = list(enumerate(["$r_f$ and $f$"], start=1))
    assert undefined_symbols(lines, defined_symbols("$r_f$"), set()) == [("f", 1)]


# Table of contents: every link resolves to a heading, and every heading at
# levels 2 to 4 is linked, so the ToC can neither break nor fall behind.

TOC_TITLE = "Table of contents"
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)\)")


def github_slug(title):
    """The anchor GitHub gives a heading: its text without markup, lowercased,
    punctuation other than ``-`` and ``_`` dropped, spaces turned to ``-``."""
    text = MD_LINK.sub(r"\1", title).replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def heading_anchors(text):
    """[(anchor, level, title)] of every heading, in order; a repeated slug
    gets ``-1``, ``-2``, ... as on GitHub."""
    seen, anchors = {}, []
    for _, level, title in headings(text):
        slug = github_slug(title)
        anchor = slug if slug not in seen else f"{slug}-{seen[slug]}"
        seen[slug] = seen.get(slug, 0) + 1
        anchors.append((anchor, level, title))
    return anchors


def toc_problems(text):
    """Messages for each ToC link with no heading and each level 2-4 heading
    (the ToC's own excepted) the ToC does not link."""
    anchors = heading_anchors(text)
    known = {anchor for anchor, _, _ in anchors}
    links = [target[1:] for line in section(text, TOC_TITLE) for _, target in MD_LINK.findall(line[1])
             if target.startswith("#")]
    problems = [f"ToC link #{link} matches no heading" for link in links if link not in known]
    problems += [f"heading {'#' * level} {title} (#{anchor}) is not in the ToC"
                 for anchor, level, title in anchors
                 if 2 <= level <= 4 and title != TOC_TITLE and anchor not in links]
    return problems


def test_toc_links_resolve_and_cover_every_heading():
    """The README's ToC links every level 2-4 heading and nothing else."""
    problems = toc_problems((ROOT / "README.md").read_text())
    assert not problems, "README.md table of contents:\n" + "\n".join(problems)


def test_github_slug_follows_github():
    """Markup and punctuation go, underscores and hyphens stay, repeats are numbered."""
    assert github_slug("MC7 Regime-switching (HMM)") == "mc7-regime-switching-hmm"
    assert github_slug("`min_sample` (all three simulators)") == "min_sample-all-three-simulators"
    assert github_slug("B1.1 Point in time") == "b11-point-in-time"
    assert github_slug("M11 Lower/Upper wording") == "m11-lowerupper-wording"
    text = "## A\n### Notes\n## B\n### Notes\n```\n## Not a heading\n```\n### Notes\n"
    assert [anchor for anchor, _, _ in heading_anchors(text)] == ["a", "notes", "b", "notes-1", "notes-2"]


TOC_EXAMPLE = """# Title
## Table of contents
- [Usage](#usage)
  - [Notes](#notes): [`run`](#run) · [Notes again](#notes-1)
## Usage
### Notes
#### `run`
#### Notes
"""


def test_toc_check_passes_a_complete_toc():
    assert toc_problems(TOC_EXAMPLE) == []


def test_toc_check_names_a_bad_anchor():
    """A link to an anchor no heading has fails, naming the anchor."""
    problems = toc_problems(TOC_EXAMPLE.replace("(#run)", "(#runn)"))
    assert "ToC link #runn matches no heading" in problems


def test_toc_check_names_an_unlisted_heading():
    """A level 2-4 heading missing from the ToC fails, naming the heading."""
    problems = toc_problems(TOC_EXAMPLE + "### Limits\n")
    assert problems == ["heading ### Limits (#limits) is not in the ToC"]


def test_monte_carlo_overrides_are_its_own_rows():
    """Monte Carlo's own rows for letters of the simulators' table are exactly
    those where it adds to them: N_{s,c} per class, the regime-free gap
    normal's mean and variance, and T, whose s_T is the last historical mark."""
    assert overrides((ROOT / "README.md").read_text(), "Monte Carlo") == ["N", "T", r"\mu", r"\sigma"]
