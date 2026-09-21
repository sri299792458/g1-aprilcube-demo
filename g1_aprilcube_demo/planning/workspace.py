"""Deterministic bounded assembly and placement pose samples."""

from __future__ import annotations

import itertools
from typing import Any, Iterable

import numpy as np
import trimesh.transformations as tra


def _ordered(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    center = 0.5 * (min(values) + max(values))
    return sorted(values, key=lambda value: (abs(value - center), value))


def workspace_samples(config: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Return center-first root poses from explicit bounded sample values."""

    axes = [_ordered(config[key]) for key in ("x_m", "y_m", "z_m", "yaw_deg")]
    result = []
    for index, (x, y, z, yaw) in enumerate(itertools.product(*axes)):
        matrix = tra.rotation_matrix(np.deg2rad(yaw), [0.0, 0.0, 1.0])
        matrix[:3, 3] = [x, y, z]
        result.append((f"sample_{index:03d}", matrix))
    maximum = int(config.get("maximum_candidate_count", len(result)))
    return result[:maximum]


def placement_samples(
    config: dict[str, Any], table_top_z: float, assembly_min_z_in_root: float
) -> list[tuple[str, np.ndarray]]:
    derived_z = table_top_z - assembly_min_z_in_root + float(config["clearance_m"])
    expanded = dict(config)
    expanded["z_m"] = [derived_z]
    return workspace_samples(expanded)
