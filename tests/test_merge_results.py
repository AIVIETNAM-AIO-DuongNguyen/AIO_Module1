"""Tests for consolidating results CSV files."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_results.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_results", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(run, auroc):
    return {"run": run, "dataset": "d", "auroc": auroc}


def test_merge_dedups_and_prefers_earlier_input():
    module = _load_script()
    first = pd.DataFrame([_row("a", 0.90), _row("b", 0.80)])
    second = pd.DataFrame([_row("b", 0.10), _row("c", 0.70)])

    merged, conflicts = module.merge_results([first, second])

    assert list(merged["run"]) == ["a", "b", "c"]
    # 'b' conflicts; the earlier input (0.80) wins.
    assert conflicts == ["b"]
    assert merged.loc[merged["run"] == "b", "auroc"].iloc[0] == 0.80


def test_merge_no_conflict_when_rows_identical():
    module = _load_script()
    frame = pd.DataFrame([_row("a", 0.90)])
    merged, conflicts = module.merge_results([frame, frame.copy()])

    assert conflicts == []
    assert len(merged) == 1


def test_merge_requires_run_column():
    module = _load_script()
    with pytest.raises(ValueError):
        module.merge_results([pd.DataFrame({"dataset": ["d"]})])
