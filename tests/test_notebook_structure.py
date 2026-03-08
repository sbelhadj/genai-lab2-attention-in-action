"""
Auto-grading tests for Lab 2 — Notebook Structure

Verifies that the student's notebook is complete and has been worked on.
"""

import pytest
import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "lab2_attention_in_action.ipynb"
)


def load_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        pytest.skip(f"Notebook not found at {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_source(nb):
    parts = []
    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            parts.append("".join(source))
        else:
            parts.append(source)
    return "\n".join(parts)


def count_code_cells_with_output(nb):
    count = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs", []):
                count += 1
    return count


class TestNotebookExists:

    def test_notebook_file_exists(self):
        assert os.path.exists(NOTEBOOK_PATH), \
            f"Notebook not found at {NOTEBOOK_PATH}"

    def test_notebook_valid_json(self):
        nb = load_notebook()
        assert "cells" in nb, "Notebook missing 'cells' key"
        assert len(nb["cells"]) > 0, "Notebook has no cells"


class TestNotebookContent:

    def test_has_minimum_cells(self):
        nb = load_notebook()
        assert len(nb["cells"]) >= 18, \
            f"Notebook has {len(nb['cells'])} cells, expected at least 18"

    def test_code_cells_executed(self):
        nb = load_notebook()
        executed = count_code_cells_with_output(nb)
        assert executed >= 5, \
            f"Only {executed} code cells have outputs. Run your notebook before submitting."


class TestAnalysisComplete:

    def test_todo_count_reduced(self):
        nb = load_notebook()
        todo_count = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                cell_source = "".join(cell.get("source", []))
                for line in cell_source.split("\n"):
                    if line.strip() == "TODO":
                        todo_count += 1
        assert todo_count <= 5, \
            f"Found {todo_count} unfilled TODO markers. Please complete your analyses."

    def test_mini_report_written(self):
        nb = load_notebook()
        source = get_all_source(nb)
        has_report = ("mini-report" in source.lower() or
                      "mini report" in source.lower() or
                      "context influences" in source.lower())
        assert has_report, "Mini-report section not found."

    def test_observations_present(self):
        nb = load_notebook()
        source = get_all_source(nb)
        observation_markers = source.lower().count("observation")
        assert observation_markers >= 3, \
            f"Expected at least 3 observation sections, found references to {observation_markers}"
