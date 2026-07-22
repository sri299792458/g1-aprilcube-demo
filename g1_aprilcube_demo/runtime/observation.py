"""Parse and validate measured loose-part poses independently of the planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml


ROOT = Path(__file__).resolve().parents[2]


class RuntimeObservationError(ValueError):
    """The observed scene cannot be passed safely to motion planning."""


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def matrix_from_pose(value: Mapping[str, Any], label: str) -> np.ndarray:
    translation = np.asarray(value.get("translation"), dtype=np.float64)
    quaternion = np.asarray(value.get("quaternion_xyzw"), dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise RuntimeObservationError(f"{label}.translation must be three finite numbers")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise RuntimeObservationError(f"{label}.quaternion_xyzw must be four finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise RuntimeObservationError(f"{label} quaternion norm is {norm}, expected 1")
    result = tra.quaternion_matrix(
        [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    )
    result[:3, 3] = translation
    return result


@dataclass(frozen=True)
class TableObservation:
    center: tuple[float, float, float]
    dimensions: tuple[float, float, float]

    @property
    def top_z(self) -> float:
        return self.center[2] + 0.5 * self.dimensions[2]


@dataclass(frozen=True)
class RuntimeObservation:
    observation_id: str
    source_path: Path
    planning_frame: str
    table: TableObservation
    world_T_objects: Mapping[str, np.ndarray]


def load_observation(
    path: str | Path,
    object_meshes: Mapping[str, str | Path],
    *,
    support_tolerance_m: float = 0.004,
    overlap_tolerance_m: float = 0.001,
) -> RuntimeObservation:
    """Load poses and reject malformed, unsupported, or overlapping objects.

    This validation deliberately does not impose prepared XY locations or yaw
    values. It checks only the measured geometry that must be true before a
    collision planner can receive the scene.
    """

    source = project_path(path)
    payload = yaml.safe_load(source.read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeObservationError("Unsupported runtime-observation schema")
    table_doc = payload.get("table", {})
    center = np.asarray(table_doc.get("center"), dtype=np.float64)
    dimensions = np.asarray(table_doc.get("dimensions"), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise RuntimeObservationError("table.center must be three finite numbers")
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise RuntimeObservationError("table.dimensions must be three positive finite numbers")
    table = TableObservation(tuple(center), tuple(dimensions))

    object_docs = payload.get("objects", {})
    expected = set(object_meshes)
    actual = set(object_docs)
    if actual != expected:
        raise RuntimeObservationError(
            f"Observed object IDs differ from task: missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}"
        )

    transforms: dict[str, np.ndarray] = {}
    bounds: dict[str, np.ndarray] = {}
    for object_id in sorted(expected):
        world_T_object = matrix_from_pose(object_docs[object_id], f"objects.{object_id}")
        mesh_path = project_path(object_meshes[object_id])
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = trimesh.transform_points(mesh.vertices, world_T_object)
        bound = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        bottom_error = float(bound[0, 2] - table.top_z)
        if bottom_error < -support_tolerance_m:
            raise RuntimeObservationError(
                f"{object_id} penetrates the table by {-bottom_error:.4f} m"
            )
        if bottom_error > support_tolerance_m:
            raise RuntimeObservationError(
                f"{object_id} floats {bottom_error:.4f} m above the table"
            )
        table_low_xy = center[:2] - 0.5 * dimensions[:2]
        table_high_xy = center[:2] + 0.5 * dimensions[:2]
        if np.any(bound[0, :2] < table_low_xy - support_tolerance_m) or np.any(
            bound[1, :2] > table_high_xy + support_tolerance_m
        ):
            raise RuntimeObservationError(
                f"{object_id} is not fully supported inside the tabletop XY bounds"
            )
        transforms[object_id] = world_T_object
        bounds[object_id] = bound

    names = sorted(bounds)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = np.minimum(bounds[first][1], bounds[second][1]) - np.maximum(
                bounds[first][0], bounds[second][0]
            )
            if np.all(overlap > overlap_tolerance_m):
                raise RuntimeObservationError(
                    f"Observed objects {first} and {second} overlap by {overlap.tolist()} m"
                )

    return RuntimeObservation(
        observation_id=str(payload.get("observation_id", source.stem)),
        source_path=source,
        planning_frame=str(payload.get("planning_frame", "world")),
        table=table,
        world_T_objects=transforms,
    )
