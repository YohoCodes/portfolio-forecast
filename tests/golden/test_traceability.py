"""Traceability between the README Logic section and the golden cases (R2).

Rule IDs come from the ``#### <ID> <title>`` headings of the ``## Logic``
section of the top-level README; a rule whose section contains ``_(pending)_``
does not need a case yet. Every case must cite only known IDs, and every rule
not marked pending must have at least one case. Each such rule's section
cites at least one case by file name (a ``Cases:`` line), and each case it
cites exists and cites that rule. No rule is left pending.
"""

import re
from pathlib import Path

import yaml
from test_golden import GOLDEN, case_files

README = GOLDEN.parent.parent / "README.md"
CASE_NAME = re.compile(r"`([\w.\-]+\.yaml)`")
PENDING = "_(pending)_"
RULE_HEADING = re.compile(r"^#### (\S+)\s")

# Every rule ID the README Logic section must list (B, U, M, MC)
CANONICAL = (
    [f"B1.{i}" for i in range(1, 7)] + ["B2", "B3"] + [f"B4.{i}" for i in range(1, 5)] + ["B5", "B6"]
    + ["U1", "U2", "U3"] + [f"M{i}" for i in range(1, 13)] + [f"MC{i}" for i in range(1, 9)]
)


def logic_rules(readme_text):
    """{rule ID: pending?} from the README's ``## Logic`` section."""
    match = re.search(r"^## Logic\n(.*?)(?=^## |\Z)", readme_text, re.M | re.S)
    if match is None:
        return {}
    rules, current = {}, None
    for line in match.group(1).splitlines():
        heading = RULE_HEADING.match(line)
        if heading:
            current = heading.group(1)
            if current in rules:
                raise ValueError(f"rule {current} has two headings in the Logic section")
            rules[current] = False
        elif line.startswith("#"):
            current = None
        elif current is not None and PENDING in line:
            rules[current] = True
    return rules


def logic_section(readme_text):
    """The text of the README's ``## Logic`` section, or an empty string."""
    match = re.search(r"^## Logic\n(.*?)(?=^## |\Z)", readme_text, re.M | re.S)
    return match.group(1) if match else ""


def rule_sections(readme_text):
    """{rule ID: body text} of each ``#### <ID>`` rule of the Logic section."""
    sections, current = {}, None
    for line in logic_section(readme_text).splitlines():
        heading = RULE_HEADING.match(line)
        if heading:
            current = heading.group(1)
            sections[current] = []
        elif line.startswith("#") or line == "---":
            current = None
        elif current is not None:
            sections[current].append(line)
    return {rule: "\n".join(lines) for rule, lines in sections.items()}


def readme_citation_errors(readme_text, paths):
    """Every way the Logic section's case citations fail (4.3, R2).

    A rule not marked pending must cite at least one case by file name; each
    cited name must be a case file, and that case must cite the rule.
    """
    cited = cited_rules(paths)
    errors = []
    for rule, body in rule_sections(readme_text).items():
        if PENDING in body:
            continue
        names = CASE_NAME.findall(body)
        if not names:
            errors.append(f"rule {rule} cites no golden case")
        for name in names:
            if name not in cited:
                errors.append(f"rule {rule} cites {name}, which is not a golden case")
            elif rule not in cited[name]:
                errors.append(f"rule {rule} cites {name}, which does not cite {rule}")
    return errors


def cited_rules(paths):
    """{case file name: [rule IDs]} from each case's ``rule`` list."""
    cited = {}
    for path in paths:
        with open(path) as handle:
            case = yaml.safe_load(handle) or {}
        cited[path.name] = [str(rule) for rule in case.get("rule") or []]
    return cited


def traceability_errors(readme_text, paths):
    """Every way the README and the cases fail to trace to each other."""
    rules = logic_rules(readme_text)
    if not rules:
        return ["README has no `## Logic` section with `#### <ID>` rule headings"]
    cited = cited_rules(paths)
    errors = []
    for name, ids in cited.items():
        if not ids:
            errors.append(f"{name} cites no rule")
        errors += [f"{name} cites unknown rule {rule}" for rule in ids if rule not in rules]
    covered = {rule for ids in cited.values() for rule in ids}
    errors += [f"rule {rule} is not marked pending but has no case"
               for rule, pending in rules.items() if not pending and rule not in covered]
    return errors


def test_readme_logic_lists_every_canonical_rule():
    rules = logic_rules(README.read_text())
    assert sorted(rules) == sorted(CANONICAL), (
        f"missing: {sorted(set(CANONICAL) - set(rules))}, unknown: {sorted(set(rules) - set(CANONICAL))}")


def test_every_rule_cites_its_cases():
    errors = readme_citation_errors(README.read_text(), case_files())
    assert not errors, "\n".join(errors)


def test_no_rule_is_pending():
    pending = [rule for rule, is_pending in logic_rules(README.read_text()).items() if is_pending]
    assert not pending, f"rules still marked pending: {pending}"


def test_cases_and_rules_trace_both_ways():
    errors = traceability_errors(README.read_text(), case_files())
    assert not errors, "\n".join(errors)


# The checker itself must bite: on a bogus ID and on an unmarked rule with no case

LOGIC = """# Title

## Logic

#### B1.4 Gap

$g_d = V(O_{d,1})/V(C_{d-1,K_{d-1}}) - 1$.

Cases: `b1.4_gap.yaml`.

#### B6 Completed bars only

Kept iff $c_j \\le \\text{now}$.
_(pending)_

## License
"""


def write_case(directory: Path, name, rules):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(yaml.safe_dump({"rule": rules, "kind": "close_times", "inputs": {},
                                    "expected": {}, "tolerance": 0}))
    return path


def test_logic_parser_reads_ids_and_pending_markers():
    assert logic_rules(LOGIC) == {"B1.4": False, "B6": True}


def test_clean_fixture_passes(tmp_path):
    assert traceability_errors(LOGIC, [write_case(tmp_path, "b1.4_gap.yaml", ["B1.4"])]) == []


def test_case_citing_a_bogus_id_fails(tmp_path):
    paths = [write_case(tmp_path, "b1.4_gap.yaml", ["B1.4"]), write_case(tmp_path, "b9_bogus.yaml", ["B9.9"])]
    assert traceability_errors(LOGIC, paths) == ["b9_bogus.yaml cites unknown rule B9.9"]


def test_unmarked_rule_without_a_case_fails(tmp_path):
    paths = [write_case(tmp_path, "b6_cutoff.yaml", ["B6"])]
    assert traceability_errors(LOGIC, paths) == ["rule B1.4 is not marked pending but has no case"]


def test_case_citing_nothing_fails(tmp_path):
    paths = [write_case(tmp_path, "b1.4_gap.yaml", ["B1.4"]), write_case(tmp_path, "b1.4_empty.yaml", [])]
    assert traceability_errors(LOGIC, paths) == ["b1.4_empty.yaml cites no rule"]


def test_clean_fixture_citations_pass(tmp_path):
    assert readme_citation_errors(LOGIC, [write_case(tmp_path, "b1.4_gap.yaml", ["B1.4"])]) == []


def test_rule_citing_no_case_fails(tmp_path):
    readme = LOGIC.replace("Cases: `b1.4_gap.yaml`.", "")
    assert readme_citation_errors(readme, [write_case(tmp_path, "b1.4_gap.yaml", ["B1.4"])]) == [
        "rule B1.4 cites no golden case"]


def test_rule_citing_a_missing_case_fails(tmp_path):
    assert readme_citation_errors(LOGIC, [write_case(tmp_path, "b1.4_other.yaml", ["B1.4"])]) == [
        "rule B1.4 cites b1.4_gap.yaml, which is not a golden case"]


def test_rule_citing_a_case_of_another_rule_fails(tmp_path):
    assert readme_citation_errors(LOGIC, [write_case(tmp_path, "b1.4_gap.yaml", ["B6"])]) == [
        "rule B1.4 cites b1.4_gap.yaml, which does not cite B1.4"]


def test_missing_logic_section_fails():
    assert traceability_errors("# Title\n\n## License\n", []) != []
