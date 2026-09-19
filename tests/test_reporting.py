import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from portfolio_forecast.performance import statistical_report
from portfolio_forecast.plotting import plot_simulated_paths
from portfolio_forecast.reporting import compile_statistical_reports
from portfolio_forecast.reporting.pdf import _escape, _number, _slug

needs_latex = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")


@pytest.fixture
def sims():
    rng = np.random.default_rng(0)
    return np.hstack([np.ones((50, 1)), np.cumprod(1 + rng.normal(0, 0.01, (50, 20)), axis=1)])


@pytest.fixture
def spec(sims):
    fig, _ = plot_simulated_paths(sims, method="Test")
    yield {
        "Title": "Generic Title",
        "Introduction": "An introduction.",
        "Reports": {
            "Report 1": {
                "Description": "A description.",
                "Results": statistical_report(sims, interval="1 day"),
                "Figures": {"Fig1": {"Image": fig, "Caption": "A caption."}},
            },
        },
    }
    plt.close("all")


class TestFormatting:
    def test_escape_makes_special_characters_literal(self):
        assert _escape("50% & $5_000 #1 {x} ~ ^ \\") == (
            r"50\% \& \$5\_000 \#1 \{x\} \textasciitilde{} \textasciicircum{} \textbackslash{}")

    def test_escape_keeps_paragraphs(self):
        assert _escape("One.\n\n\nTwo.\n") == "One.\n\nTwo."

    def test_number_formats_by_metric(self):
        assert _number(0.1234, "Total Return") == r"12.34\%"
        assert _number(-0.05, "Max Drawdown") == r"\textminus{}5.00\%"
        assert _number(1.2345, "Sharpe Ratio") == "1.23"
        assert _number(float("nan"), "CAGR") == "---"

    def test_slug(self):
        assert _slug("AAPL: 100-Day Outlook (95%)") == "AAPL_100_Day_Outlook_95"
        assert _slug("???") == "Report"


class TestValidation:
    def test_title_is_required(self, spec):
        spec["Title"] = "  "
        with pytest.raises(ValueError, match="Title"):
            compile_statistical_reports(spec)

    def test_reports_are_required(self, spec):
        spec["Reports"] = {}
        with pytest.raises(ValueError, match="Reports"):
            compile_statistical_reports(spec)

    def test_results_must_come_from_statistical_report(self, spec):
        spec["Reports"]["Report 1"]["Results"] = {"Returns": 1}
        with pytest.raises(ValueError, match="statistical_report"):
            compile_statistical_reports(spec)

    def test_image_must_be_a_figure(self, spec):
        spec["Reports"]["Report 1"]["Figures"]["Fig1"]["Image"] = "chart.png"
        with pytest.raises(TypeError, match="'Fig1' in report 'Report 1'"):
            compile_statistical_reports(spec)

    def test_missing_engine_raises(self, spec, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="pdflatex was not found"):
            compile_statistical_reports(spec)


@needs_latex
class TestCompile:
    def test_writes_only_the_pdf_to_reporting_in_the_working_directory(self, spec, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pdf = compile_statistical_reports(spec)
        assert pdf == tmp_path / "Reporting" / "Generic_Title.pdf"
        assert pdf.read_bytes().startswith(b"%PDF")
        assert [p.name for p in tmp_path.rglob("*")] == ["Reporting", "Generic_Title.pdf"]

    def test_custom_directory_and_filename(self, spec, tmp_path):
        pdf = compile_statistical_reports(spec, output_dir=tmp_path / "out", filename="summary")
        assert pdf == tmp_path / "out" / "summary.pdf" and pdf.exists()

    def test_replaces_an_existing_report(self, spec, tmp_path):
        first = compile_statistical_reports(spec, output_dir=tmp_path)
        first.write_bytes(b"stale")
        second = compile_statistical_reports(spec, output_dir=tmp_path)
        assert second == first and second.read_bytes().startswith(b"%PDF")

    def test_special_characters_and_optional_fields(self, spec, sims, tmp_path):
        spec["Title"] = "Returns & Risk: 95% of $100_000 {#1} ~^\\"
        spec["Introduction"] = ""
        spec["Reports"]["Report 1"]["Figures"]["Fig1"]["Caption"] = ""
        spec["Reports"]["Report 2 & more"] = {"Results": statistical_report(sims, "5 mins")}
        pdf = compile_statistical_reports(spec, output_dir=tmp_path)
        assert pdf.read_bytes().startswith(b"%PDF")

    def test_figures_are_left_open_and_unchanged(self, spec, tmp_path):
        fig = spec["Reports"]["Report 1"]["Figures"]["Fig1"]["Image"]
        size = tuple(fig.get_size_inches())
        compile_statistical_reports(spec, output_dir=tmp_path)
        assert plt.fignum_exists(fig.number)
        assert tuple(fig.get_size_inches()) == size

    def test_latex_failure_reports_the_log_and_writes_nothing(self, spec, tmp_path):
        # pdflatex has no glyph for an emoji
        spec["Introduction"] = "Rocket \U0001F680"
        with pytest.raises(RuntimeError, match="failed to compile"):
            compile_statistical_reports(spec, output_dir=tmp_path / "out")
        assert not (tmp_path / "out").exists()
