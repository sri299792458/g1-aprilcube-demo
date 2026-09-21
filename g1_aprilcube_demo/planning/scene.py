"""Load the seated G1 planning scene without importing a motion backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PlanningSceneError(ValueError):
    """The planning-scene configuration is incomplete or inconsistent."""


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise PlanningSceneError(f"{label} must contain {size} finite numbers")
    return result


def matrix_from_xyzw(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-7):
        raise PlanningSceneError(f"Quaternion norm is {norm}, expected 1")
    transform = trimesh.transformations.quaternion_matrix(
        [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    )
    transform[:3, 3] = translation
    return transform


@dataclass(frozen=True)
class PartPlacement:
    part_id: str
    mesh: Path
    xy_m: tuple[float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    color_rgb: tuple[int, int, int]

    def world_transform(self, table_top_z_m: float) -> np.ndarray:
        mesh = trimesh.load(self.mesh, force="mesh", process=False)
        rotation = matrix_from_xyzw(
            np.zeros(3), np.asarray(self.quaternion_xyzw, dtype=np.float64)
        )
        rotated = trimesh.transform_points(mesh.vertices, rotation)
        center_z = float(table_top_z_m - rotated[:, 2].min())
        result = rotation.copy()
        result[:3, 3] = [self.xy_m[0], self.xy_m[1], center_z]
        return result


@dataclass(frozen=True)
class PlanningSceneSpec:
    scene_id: str
    source_path: Path
    planning_frame: str
    curobo_config: Path
    urdf: Path
    start_joint_positions: Mapping[str, float]
    active_joint_groups: Mapping[str, tuple[str, ...]]
    tool_frames: Mapping[str, str]
    table_top_z_m: float
    table_center_xy_m: tuple[float, float]
    table_size_xy_m: tuple[float, float]
    table_thickness_m: float
    minimum_robot_clearance_m: float
    parts: Mapping[str, PartPlacement]
    camera_eye_m: tuple[float, float, float]
    camera_target_m: tuple[float, float, float]
    image_size_px: tuple[int, int]
    output: Path

    def world_part_transforms(self) -> dict[str, np.ndarray]:
        return {
            part_id: part.world_transform(self.table_top_z_m)
            for part_id, part in self.parts.items()
        }


def _load_start_state(robot: Mapping[str, Any]) -> dict[str, float]:
    seated = yaml.safe_load(_path(robot["seated_state_reference"]).read_text())
    observed = seated["observed_start_state"]
    state: dict[str, float] = {}
    for section in ("lower_body_and_waist_rad", "left_arm_rad", "right_arm_rad"):
        state.update({name: float(value) for name, value in observed[section].items()})

    descriptor = json.loads(_path(robot["hand_state_reference"]).read_text())
    finger_profile = descriptor["finger_profile"]
    for side in ("left", "right"):
        state.update(
            {name: float(value) for name, value in finger_profile[side]["open"].items()}
        )
    if len(state) != 43:
        raise PlanningSceneError(
            f"Expected all 43 physical G1 joints in the start state, got {len(state)}"
        )
    return state


def load_planning_scene(path: str | Path) -> PlanningSceneSpec:
    source_path = _path(path)
    payload = yaml.safe_load(source_path.read_text())
    if payload.get("schema_version") != 1:
        raise PlanningSceneError("Unsupported planning-scene schema")
    robot = payload["robot"]
    table = payload["table"]

    seated_reference = yaml.safe_load(
        _path(robot["seated_state_reference"]).read_text()
    )
    source_field = str(table["source_field"])
    section, field = source_field.split(".", 1)
    source_table_top_z_m = float(seated_reference[section][field])
    duplicated_reference = float(table["source_top_z_reference_m"])
    if not np.isclose(source_table_top_z_m, duplicated_reference, atol=5e-9):
        raise PlanningSceneError(
            "Planning-scene table height drifted from its named reference: "
            f"{duplicated_reference} != {source_table_top_z_m}"
        )
    table_top_z_m = source_table_top_z_m + float(
        table["dex3_clearance_adjustment_m"]
    )
    if not np.isclose(table_top_z_m, float(table["planned_top_z_m"]), atol=5e-9):
        raise PlanningSceneError("Derived Dex3-clear table height does not match config")

    parts: dict[str, PartPlacement] = {}
    for part_id, value in payload["parts"].items():
        xy = _vector(value["xy_m"], 2, f"parts.{part_id}.xy_m")
        quaternion = _vector(
            value["quaternion_xyzw"], 4, f"parts.{part_id}.quaternion_xyzw"
        )
        matrix_from_xyzw(np.zeros(3), quaternion)
        color = tuple(int(item) for item in value["color_rgb"])
        if len(color) != 3 or any(item < 0 or item > 255 for item in color):
            raise PlanningSceneError(f"Invalid color for {part_id}")
        mesh = _path(value["mesh"])
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        parts[part_id] = PartPlacement(
            part_id=part_id,
            mesh=mesh,
            xy_m=(float(xy[0]), float(xy[1])),
            quaternion_xyzw=tuple(float(item) for item in quaternion),
            color_rgb=color,
        )

    active_groups = {
        group: tuple(str(name) for name in names)
        for group, names in robot["active_joint_groups"].items()
    }
    start_state = _load_start_state(robot)
    for group, names in active_groups.items():
        missing = sorted(set(names) - set(start_state))
        if missing:
            raise PlanningSceneError(f"{group} references missing joints: {missing}")

    table_center = _vector(table["center_xy_m"], 2, "table.center_xy_m")
    table_size = _vector(table["size_xy_m"], 2, "table.size_xy_m")
    if np.any(table_size <= 0.0) or float(table["thickness_m"]) <= 0.0:
        raise PlanningSceneError("Table dimensions must be positive")
    visual = payload["visual"]
    image_size = tuple(int(item) for item in visual["image_size_px"])
    if len(image_size) != 2 or min(image_size) <= 0:
        raise PlanningSceneError("visual.image_size_px must be positive width/height")

    return PlanningSceneSpec(
        scene_id=str(payload["scene_id"]),
        source_path=source_path,
        planning_frame=str(payload["frames"]["planning_frame"]),
        curobo_config=_path(robot["curobo_config"]),
        urdf=_path(robot["urdf"]),
        start_joint_positions=start_state,
        active_joint_groups=active_groups,
        tool_frames={key: str(value) for key, value in robot["tool_frames"].items()},
        table_top_z_m=table_top_z_m,
        table_center_xy_m=(float(table_center[0]), float(table_center[1])),
        table_size_xy_m=(float(table_size[0]), float(table_size[1])),
        table_thickness_m=float(table["thickness_m"]),
        minimum_robot_clearance_m=float(table["minimum_robot_clearance_m"]),
        parts=parts,
        camera_eye_m=tuple(float(x) for x in visual["camera_eye_m"]),
        camera_target_m=tuple(float(x) for x in visual["camera_target_m"]),
        image_size_px=image_size,
        output=_path(visual["output"]),
    )
