"""Test 1: Compilation and notebook validity."""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compile_all_python_files() -> None:
    """Compile every Python file in src and scripts without syntax errors."""
    python_files = list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").rglob("*.py"))
    assert len(python_files) >= 10, f"Expected at least 10 python files, found {len(python_files)}"

    for py_file in python_files:
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as exc:
            assert False, f"Compilation failed on {py_file}: {exc}"


def test_validate_notebooks_syntax() -> None:
    """Validate the final Account A notebook and all four private shards."""
    notebooks = list((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5, f"Expected exactly 5 notebooks, found {len(notebooks)}"

    for nb_path in notebooks:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        assert data.get("nbformat") == 4
        cells = data.get("cells", [])
        assert len(cells) >= 5, f"Notebook {nb_path.name} has too few cells: {len(cells)}"

        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                code_text = "".join(cell.get("source", []))
                try:
                    ast.parse(code_text, filename=f"{nb_path.name} cell {i}")
                except SyntaxError as exc:
                    assert False, f"Syntax error in {nb_path.name} cell {i}: {exc}"


def test_private_notebook_preflight_pins_and_imports_runtime_dependencies() -> None:
    """Check the shipped private notebooks against frozen runtime pins."""
    config = json.loads(
        (PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "sentence-transformers": "5.4.1",
        "faiss-cpu": "1.15.0",
        "numpy": "2.0.2",
    }
    for distribution, version in expected.items():
        assert config["runtime"][distribution] == version
    private_notebooks = list((PROJECT_ROOT / "notebooks").glob("E45-PRIVATE-*.ipynb"))
    assert len(private_notebooks) == 4
    for notebook in private_notebooks:
        notebook_data = json.loads(notebook.read_text(encoding="utf-8"))
        source = "".join(
            "".join(cell.get("source", [])) for cell in notebook_data["cells"]
        )
        assert '"faiss-cpu": "1.15.0"' in source
        assert "import faiss" in source
        assert 'pip", "uninstall", "-y", "torchao"' in source
        assert "EXPECTED_SYSTEM_SHA256" in source
        assert "RESUME_CHECKPOINT" in source
