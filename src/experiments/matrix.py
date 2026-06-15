"""Generate run configs from a matrix spec and assign them to compute accounts.

A matrix spec has two parts: ``base`` (settings shared by every run) and
``axes`` (the dimensions to sweep). The Cartesian product of the axes is merged
onto the base to produce one config per run, each validated against the run
schema so a bad spec fails before any run is launched.
"""

import copy
import itertools
import os
from typing import Any

import yaml

from ..utils.config import build_run_config

# Each sweepable axis maps to its location in the run config. The canonical
# order also fixes the generation order and the run-name token order.
_AXIS_PATHS = {
    "dataset": ("data", "dataset"),
    "backbone": ("model", "backbone"),
    "preprocessing": ("data", "preprocessing"),
    "transfer": ("model", "transfer"),
    "regime": ("data", "regime"),
    "seed": ("run", "seed"),
}
_AXIS_ORDER = ["dataset", "backbone", "preprocessing", "transfer", "regime", "seed"]


def _set_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    section = config
    for key in path[:-1]:
        section = section.setdefault(key, {})
    section[path[-1]] = value


def _name_token(axis: str, value: Any) -> str:
    if axis == "regime":
        return f"r{int(round(float(value) * 100))}"
    if axis == "seed":
        return f"s{value}"
    return str(value)


def generate_configs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one validated config dict per point in the matrix."""
    base = spec.get("base")
    axes = spec.get("axes")
    if not isinstance(base, dict):
        raise ValueError("Matrix spec must contain a 'base' mapping.")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("Matrix spec must contain a non-empty 'axes' mapping.")

    unknown = set(axes) - set(_AXIS_PATHS)
    if unknown:
        allowed = ", ".join(sorted(_AXIS_PATHS))
        raise ValueError(f"Unknown axes {sorted(unknown)}. Allowed: {allowed}.")

    present = [axis for axis in _AXIS_ORDER if axis in axes]
    for axis in present:
        if not isinstance(axes[axis], list) or not axes[axis]:
            raise ValueError(f"Axis '{axis}' must be a non-empty list.")

    configs = []
    seen_names: set[str] = set()
    for combination in itertools.product(*(axes[axis] for axis in present)):
        config = copy.deepcopy(base)
        tokens = []
        for axis, value in zip(present, combination):
            _set_path(config, _AXIS_PATHS[axis], value)
            tokens.append(_name_token(axis, value))
        name = "_".join(tokens)
        if name in seen_names:
            raise ValueError(f"Duplicate run name '{name}' from the matrix.")
        seen_names.add(name)
        _set_path(config, ("run", "name"), name)
        build_run_config(config)
        configs.append(config)
    return configs


def write_configs(configs: list[dict[str, Any]], out_dir: str) -> list[str]:
    """Write each config to ``out_dir/<run name>.yaml`` and return the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for config in configs:
        path = os.path.join(out_dir, f"{config['run']['name']}.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        paths.append(path)
    return paths


def assign_slices(run_names: list[str], n_accounts: int) -> list[dict[str, Any]]:
    """Assign runs to accounts round-robin so heavy and light runs spread out."""
    if n_accounts <= 0:
        raise ValueError(f"n_accounts must be positive, got {n_accounts}.")
    return [
        {"run": name, "account": index % n_accounts}
        for index, name in enumerate(run_names)
    ]
