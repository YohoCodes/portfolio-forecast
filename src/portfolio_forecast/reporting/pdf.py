import datetime as dt
import math
import numbers
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from matplotlib.figure import Figure

# How each statistical_report metric is printed: percentages for returns and
# risk, two decimals for multiples and ratios
_PERCENT = {"Total Return", "CAGR", "Period Volatility", "Annualized Volatility",
            "Downside Volatility", "Max Drawdown", "Probability of Loss",
            "Value at Risk", "Conditional VaR"}

# Characters LaTeX treats as commands, and their literal forms
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_SPECIALS_RE = re.compile("|".join(re.escape(c) for c in _LATEX_SPECIALS))

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\definecolor{accent}{HTML}{1F3A5F}
\hypersetup{pdftitle={%(pdf_title)s}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{accent}%(title)s}
\fancyhead[R]{\small\color{accent}%(date)s}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
"""


def _escape(text):
    """Make plain text safe to typeset: LaTeX special characters print
    literally, and blank lines separate paragraphs."""
    escaped = _LATEX_SPECIALS_RE.sub(lambda m: _LATEX_SPECIALS[m.group()], str(text))
    return "\n\n".join(p.strip() for p in re.split(r"\n\s*\n", escaped) if p.strip())


def _number(value, metric):
    """Format one metric value for a table, with a true minus sign."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    text = f"{value:.2%}" if metric in _PERCENT else f"{value:,.2f}"
    text = text.replace("%", r"\%")
    return r"\textminus{}" + text[1:] if text.startswith("-") else text


def _slug(title):
    """A filesystem-safe file name from the report title."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return slug or "Report"


def _interval_table(summary, caption, level):
    """A booktabs table of Mean, Median and the interval for each metric row
    of a statistical_report summary frame."""
    rows = "\n".join(
        f"{_escape(metric)} & {_number(row['Mean'], metric)} & {_number(row['Median'], metric)} & "
        f"{_number(row['Lower'], metric)} & {_number(row['Upper'], metric)} \\\\"
        for metric, row in summary.iterrows()
    )
    return rf"""\begin{{table}}[H]
\centering
\caption{{{caption}}}
\begin{{tabular}}{{lrrrr}}
\toprule
 & & & \multicolumn{{2}}{{c}}{{{level} interval}} \\
\cmidrule(lr){{4-5}}
Metric & Mean & Median & Lower & Upper \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def _results_section(results, name):
    """Tables for one statistical_report result: the simulation setup, the
    returns and risk summaries, and tail risk."""
    level = rf"{results['Confidence']:.0%}".replace("%", r"\%")
    setup = [("Simulated paths", f"{results['Paths']:,}"),
             ("Horizon", f"{results['Periods']:,} periods ({results['Total Years']:.2f} years)")]
    if results.get("Total Days") is not None:
        setup.append(("Calendar span", f"{results['Total Days']:,} days"))
    setup.append(("Confidence level", level))
    setup_rows = "\n".join(f"{label} & {value} \\\\" for label, value in setup)

    tail = results["Tail Risk"]
    tail_rows = "\n".join([
        f"Probability of loss & {_number(tail['Probability of Loss'], 'Probability of Loss')} \\\\",
        f"Value at Risk ({level}) & {_number(tail['Value at Risk'], 'Value at Risk')} \\\\",
        f"Conditional VaR ({level}) & {_number(tail['Conditional VaR'], 'Conditional VaR')} \\\\",
    ])

    return rf"""\begin{{table}}[H]
\centering
\caption{{{name}: simulation setup}}
\begin{{tabular}}{{lr}}
\toprule
{setup_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

{_interval_table(results['Returns'], f'{name}: returns', level)}

{_interval_table(results['Risk'], f'{name}: risk', level)}

\begin{{table}}[H]
\centering
\caption{{{name}: tail risk of the total return}}
\begin{{tabular}}{{lr}}
\toprule
{tail_rows}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def _validate(spec):
    """Check the report dictionary's shape, so a mistake is reported by name
    rather than as a LaTeX error."""
    if not isinstance(spec, dict):
        raise TypeError(f"expected a report dictionary, got {type(spec).__name__}")
    if not str(spec.get("Title", "")).strip():
        raise ValueError('the report dictionary needs a non-empty "Title"')
    reports = spec.get("Reports")
    if not isinstance(reports, dict) or not reports:
        raise ValueError('the report dictionary needs a non-empty "Reports" dictionary')

    for name, report in reports.items():
        results = report.get("Results") if isinstance(report, dict) else None
        if not isinstance(results, dict) or not {"Returns", "Risk", "Tail Risk"} <= results.keys():
            raise ValueError(
                f'report {name!r} needs "Results" from statistical_report '
                '(a dict with "Returns", "Risk" and "Tail Risk")'
            )
        for fig_name, figure in (report.get("Figures") or {}).items():
            image = figure.get("Image") if isinstance(figure, dict) else None
            if not isinstance(image, Figure):
                raise TypeError(
                    f'figure {fig_name!r} in report {name!r} needs a matplotlib Figure as "Image", '
                    f"got {type(image).__name__}"
                )
            width = figure.get("Width", 1.0)
            if not isinstance(width, numbers.Real) or isinstance(width, bool):
                raise TypeError(
                    f'figure {fig_name!r} in report {name!r} needs a number as "Width", '
                    f"got {type(width).__name__}"
                )
            if not 0 < width <= 1:
                raise ValueError(
                    f'figure {fig_name!r} in report {name!r} has "Width" {width!r}; it is a share of '
                    "the text width, so it must be greater than 0 and at most 1"
                )


def compile_statistical_reports(spec, output_dir="Reporting", filename=None, engine="pdflatex",
                                table_of_contents=False):
    """Typeset statistical reports and their figures as a PDF, using LaTeX.

    Builds one document from a report dictionary: a title and introduction,
    an optional table of contents, then a section per report with its
    description, tables of the `statistical_report` results and its figures.
    LaTeX runs in a temporary directory, so the PDF is the only file written.

    Parameters
    ----------
    spec : dict
        The report dictionary::

            {
                "Title": "Generic Title",
                "Introduction": "Text before the first report.",
                "Reports": {
                    "Report 1 Name": {
                        "Description": "Text before the report's tables.",
                        "Results": results,  # from statistical_report
                        "Figures": {
                            "Fig1": {"Image": fig, "Caption": "...", "Width": 0.8},
                        },
                    },
                },
            }

        ``"Title"`` and a non-empty ``"Reports"`` are required. Reports and
        figures appear in dictionary order. ``"Introduction"``,
        ``"Description"``, ``"Figures"`` and ``"Caption"`` may be empty or
        left out. ``"Width"`` is the figure's width as a share of the text
        width, greater than 0 and at most 1 (the default, full width); the
        figure is centered and keeps its aspect ratio. Text is typeset literally (LaTeX special characters such as
        ``%``, ``&`` and ``_`` are escaped), and a blank line starts a new
        paragraph. The figure keys (``"Fig1"``) only name the figures in
        error messages.
    output_dir : str or path-like, default 'Reporting'
        Directory for the PDF, created if missing. A relative path is
        relative to the current working directory, normally the project
        folder.
    filename : str, optional
        Name of the PDF. Defaults to the title with non-alphanumeric runs
        replaced by underscores, e.g. ``Generic_Title.pdf``. An existing file
        of the same name is replaced.
    engine : str, default 'pdflatex'
        LaTeX engine to run; ``xelatex`` or ``lualatex`` also work.
    table_of_contents : bool, default False
        If True, list the reports with their page numbers after the
        introduction. Each entry links to its section.

    Returns
    -------
    pathlib.Path
        Absolute path of the PDF.

    Raises
    ------
    TypeError
        If `spec` is not a dict, a figure's ``"Image"`` is not a matplotlib
        Figure, or its ``"Width"`` is not a number.
    ValueError
        If ``"Title"`` or ``"Reports"`` is missing or empty, a report's
        ``"Results"`` is not a `statistical_report` result, or a figure's
        ``"Width"`` is not in (0, 1].
    RuntimeError
        If the LaTeX engine is not installed, or LaTeX fails to compile the
        document (the end of its log is included).

    Notes
    -----
    Requires a LaTeX distribution with the booktabs, caption, fancyhdr,
    float, geometry, hyperref, lmodern and microtype packages, all part of
    TeX Live, MacTeX and MiKTeX. Figures are embedded as vector PDF, so they
    stay sharp at any zoom; the Figure objects are not modified or closed.

    Examples
    --------
    >>> results = statistical_report(sims, interval="1 day")
    >>> fig, ax = plot_simulated_paths(sims, method="Non-parametric Monte Carlo")
    >>> compile_statistical_reports({
    ...     "Title": "AAPL 100-Day Outlook",
    ...     "Introduction": "Simulated from ten years of daily closes.",
    ...     "Reports": {"Non-parametric Monte Carlo": {
    ...         "Description": "Bootstrap of historical daily returns.",
    ...         "Results": results,
    ...         "Figures": {"Paths": {"Image": fig, "Caption": "1,000 simulated paths."}},
    ...     }},
    ... }, table_of_contents=True)
    PosixPath('/path/to/project/Reporting/AAPL_100_Day_Outlook.pdf')
    """
    _validate(spec)
    if shutil.which(engine) is None:
        raise RuntimeError(
            f"{engine} was not found. Install a LaTeX distribution (TeX Live, MacTeX on macOS, "
            "or MiKTeX on Windows) and make sure its binaries are on PATH."
        )

    title = str(spec["Title"]).strip()
    output_dir = Path(output_dir).resolve()
    pdf_name = filename or f"{_slug(title)}.pdf"
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    with tempfile.TemporaryDirectory() as build:
        build = Path(build)
        body = []

        intro = _escape(spec.get("Introduction") or "")
        if intro:
            body.append(intro)
        if table_of_contents:
            body.append(r"\tableofcontents")

        n_figure = 0
        for name, report in spec["Reports"].items():
            section = [rf"\section{{{_escape(name)}}}"]
            description = _escape(report.get("Description") or "")
            if description:
                section.append(description)
            section.append(_results_section(report["Results"], _escape(name)))

            for figure in (report.get("Figures") or {}).values():
                n_figure += 1
                image = build / f"figure{n_figure}.pdf"
                figure["Image"].savefig(image, bbox_inches="tight")
                caption = _escape(figure.get("Caption") or "")
                section.append(
                    "\\begin{figure}[H]\n\\centering\n"
                    f"\\includegraphics[width={float(figure.get('Width', 1.0)):.4f}\\linewidth]{{{image.name}}}\n"
                    + (f"\\caption{{{caption}}}\n" if caption else "")
                    + "\\end{figure}"
                )
            body.append("\n\n".join(section))

        today = dt.date.today()
        date = f"{today:%B} {today.day}, {today.year}"
        escaped_title = _escape(title)
        document = (
            _PREAMBLE % {"title": escaped_title, "pdf_title": escaped_title, "date": date}
            + "\\begin{document}\n"
            + "\\begin{center}\n"
            + f"{{\\LARGE\\bfseries\\color{{accent}} {escaped_title}\\par}}\n\\vspace{{0.4em}}\n"
            + f"{{\\large {date}\\par}}\n"
            + "\\end{center}\n\\vspace{0.5em}\n\n"
            + "\n\n".join(body)
            + "\n\n\\end{document}\n"
        )
        tex = build / "report.tex"
        tex.write_text(document, encoding="utf-8")

        # Twice, so the table of contents and cross-references settle
        for _ in range(2):
            run = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=build, capture_output=True, text=True, errors="replace",
            )
            if run.returncode != 0:
                log = (build / "report.log")
                tail = log.read_text(errors="replace")[-3000:] if log.exists() else run.stdout[-3000:]
                raise RuntimeError(f"{engine} failed to compile the report:\n{tail}")

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf = output_dir / pdf_name
        shutil.copyfile(build / "report.pdf", pdf)

    return pdf
