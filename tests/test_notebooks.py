"""Check the builder and optionally execute without a paper tree."""

import shutil
from pathlib import Path

import nbformat
import pytest

import build_compositional_notebooks as builder


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("include_plots", [False, True])
def test_builder_passes_source_and_working_directory_to_executor(tmp_path, monkeypatch, include_plots):
    class RecordingClient:
        def __init__(self, notebook, **kwargs):
            self.nb = notebook
            assert kwargs["resources"]["metadata"]["path"] == str(tmp_path)

        def execute(self):
            expected = builder.COMMON_CELLS + (builder.PLOT_CELLS if include_plots else [])
            assert [(c.cell_type, c.source) for c in self.nb.cells] == [
                (c.cell_type, c.source) for c in expected]

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "NotebookClient", RecordingClient)
    # This is a builder plumbing test, not evidence of notebook execution.
    builder.build_notebook(tmp_path / "generated.ipynb", include_plots=include_plots)


@pytest.mark.notebook
def test_analysis_notebook_executes_without_paper_checkout(tmp_path, monkeypatch):
    shutil.copy2(ROOT / "compositional_regime_contracts.py", tmp_path)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.delenv("BIOSPEAK_FIGURES_DIR", raising=False)
    output = tmp_path / "analysis.ipynb"
    builder.build_notebook(output, include_plots=False)
    notebook = nbformat.read(output, as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    assert not (tmp_path / "paper_ACC_2027").exists()


def test_failed_execution_preserves_existing_notebook(tmp_path, monkeypatch):
    output = tmp_path / "analysis.ipynb"
    original = "existing notebook bytes"
    output.write_text(original)

    def fail_execution(self):
        raise RuntimeError("execution failed")

    monkeypatch.setattr(builder.NotebookClient, "execute", fail_execution)
    with pytest.raises(RuntimeError, match="execution failed"):
        builder.build_notebook(output, include_plots=False)
    assert output.read_text() == original
