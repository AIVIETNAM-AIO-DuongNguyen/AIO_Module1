"""Tests for the Phase 1 baseline selection (no training)."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_baselines.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_phase1_baselines", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_configs_select_p0_ft_full_data():
    module = _load_script()
    configs = module._baseline_configs()

    assert len(configs) == 6
    for config in configs:
        assert config["data"]["preprocessing"] == "p0"
        assert config["model"]["transfer"] == "ft"
        assert float(config["data"]["regime"]) == 1.0

    assert {config["data"]["dataset"] for config in configs} == {
        "pneumoniamnist",
        "dermamnist",
    }
    assert {config["run"]["seed"] for config in configs} == {0, 1, 2}


def test_baseline_configs_build_into_valid_run_configs():
    module = _load_script()
    for config_dict in module._baseline_configs():
        run_config = module.build_run_config(config_dict)
        assert run_config.model.backbone == "resnet50"
        assert run_config.data.regime == 1.0
