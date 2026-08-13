"""Deterministic task annotations for immutable GraspGenX candidates.

The intrinsic Isaac atlas deliberately groups grasps using only trustworthy
body-level contact evidence.  Assembly still needs a coarse answer to a
different question: which named region of our own AprilCube CAD does the
candidate approach?

This module answers that question from geometry we control.  It does not
change ``object_T_G``, infer grasp success, or replace cuRobo collision
checking.  The GraspGenX contract approaches the terminal grasp along local
``+Z`` when viewed from the terminal ``G`` frame, so the first owned CAD voxel
hit by that ray is the task-facing target region.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .grasp_goalset import GraspCandidate


@dataclass(frozen=True)
class SemanticVoxel:
    """One non-overlapping voxel with a single named component owner."""

    index: tuple[int, int, int]
    component: str
    center: np.ndarray
    half_extents: np.ndarray


@dataclass(frozen=True)
class RayHit:
    """One first-surface event on a semantic voxel."""

    distance_m: float
    voxel_index: tuple[int, int, int]
    component: str
    surface_sector: str
    point_object_m: tuple[float, float, float]


@dataclass(frozen=True)
class TaskGraspAnnotation:
    """Coarse CAD-region annotation; never an intrinsic physics verdict."""

    candidate_id: str
    target_components: tuple[str, ...]
    target_voxels: tuple[tuple[int, int, int], ...]
    target_surface_sectors: tuple[str, ...]
    hit_points_object_m: tuple[tuple[float, float, float], ...]
    capture_components: tuple[str, ...]
    capture_voxels: tuple[tuple[int, int, int], ...]
    resolved: bool
    ambiguous: bool

    def allowed_by(self, allowed_components: Iterable[str]) -> bool:
        """Return true only when every equally-first hit is task-allowed."""

        allowed = set(allowed_components)
        if not allowed:
            return self.resolved
        return (
            self.resolved
            and set(self.target_components) <= allowed
            and set(self.capture_components) <= allowed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_components": list(self.target_components),
            "target_voxels": [list(value) for value in self.target_voxels],
            "target_surface_sectors": list(self.target_surface_sectors),
            "hit_points_object_m": [
                list(value) for value in self.hit_points_object_m
            ],
            "capture_components": list(self.capture_components),
            "capture_voxels": [list(value) for value in self.capture_voxels],
            "resolved": self.resolved,
            "ambiguous": self.ambiguous,
            "source": (
                "first owned AprilCube CAD voxel intersected by the "
                "GraspGenX terminal-frame local +Z ray, plus owned voxels "
                "overlapped by the descriptor capture volumes"
            ),
        }


def load_semantic_voxels(path: str | Path) -> tuple[SemanticVoxel, ...]:
    """Expand overlapping named cuboids into non-overlapping owned voxels.

    Later cuboids own shared voxels.  This matches the physical part intent:
    ``hip_bridge`` owns the U junction voxels and ``shoulder_crossbar`` owns
    the T's top junction voxel.
    """

    document: Mapping[str, Any] = yaml.safe_load(Path(path).read_text())
    shape = document["shape"]
    if shape.get("type") != "voxel_cuboids":
        raise ValueError(f"{path} is not a voxel_cuboids geometry")
    voxel_size = float(shape["voxel_size_mm"]) / 1000.0
    owners: dict[tuple[int, int, int], str] = {}
    for cuboid in shape["cuboids"]:
        origin = np.asarray(cuboid["origin"], dtype=np.int64)
        size = np.asarray(cuboid["size"], dtype=np.int64)
        if np.any(size <= 0):
            raise ValueError(f"{path} contains a non-positive cuboid size")
        for delta in itertools.product(*(range(int(value)) for value in size)):
            index = tuple((origin + np.asarray(delta, dtype=np.int64)).tolist())
            owners[index] = str(cuboid["name"])
    if not owners:
        raise ValueError(f"{path} contains no semantic voxels")

    indices = np.asarray(list(owners), dtype=np.float64)
    union_center = 0.5 * (indices.min(axis=0) + indices.max(axis=0) + 1.0)
    half_extents = np.full(3, 0.5 * voxel_size, dtype=np.float64)
    half_extents.setflags(write=False)
    output = []
    for index in sorted(owners):
        center = (
            np.asarray(index, dtype=np.float64) + 0.5 - union_center
        ) * voxel_size
        center.setflags(write=False)
        output.append(
            SemanticVoxel(
                index=index,
                component=owners[index],
                center=center,
                half_extents=half_extents,
            )
        )
    return tuple(output)


def _sector(axis: int, sign: float) -> str:
    return f"{'+' if sign > 0 else '-'}{'XYZ'[axis]}"


def _ray_voxel_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    voxel: SemanticVoxel,
    *,
    tolerance_m: float,
) -> RayHit | None:
    """Return the first non-negative slab hit on one axis-aligned voxel."""

    low = voxel.center - voxel.half_extents - tolerance_m
    high = voxel.center + voxel.half_extents + tolerance_m
    t_near = -float("inf")
    t_far = float("inf")
    near_sector = ""
    far_sector = ""
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) < 1e-12:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return None
            continue
        low_t = float((low[axis] - origin[axis]) / component)
        high_t = float((high[axis] - origin[axis]) / component)
        low_sector = _sector(axis, -1.0)
        high_sector = _sector(axis, 1.0)
        if low_t > high_t:
            low_t, high_t = high_t, low_t
            low_sector, high_sector = high_sector, low_sector
        if low_t > t_near:
            t_near = low_t
            near_sector = low_sector
        if high_t < t_far:
            t_far = high_t
            far_sector = high_sector
        if t_near > t_far:
            return None
    if t_far < -tolerance_m:
        return None
    if t_near >= -tolerance_m:
        distance = max(0.0, t_near)
        sector = near_sector
    else:
        # G may lie inside a voxel.  The first non-negative event is then the
        # exit surface, which remains a deterministic coarse region label.
        distance = max(0.0, t_far)
        sector = far_sector
    point = origin + distance * direction
    return RayHit(
        distance_m=distance,
        voxel_index=voxel.index,
        component=voxel.component,
        surface_sector=sector,
        point_object_m=tuple(float(value) for value in point),
    )


def annotate_candidate(
    candidate: GraspCandidate,
    voxels: Iterable[SemanticVoxel],
    *,
    capture_boxes_G: Iterable[tuple[np.ndarray, np.ndarray]] = (),
    tie_tolerance_m: float = 1e-6,
) -> TaskGraspAnnotation:
    """Annotate one unchanged candidate with all equally-first CAD hits."""

    origin = np.asarray(candidate.object_T_G[:3, 3], dtype=np.float64)
    direction = np.asarray(candidate.object_T_G[:3, 2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{candidate.candidate_id} has an invalid approach axis")
    direction = direction / norm
    hits = [
        hit
        for voxel in voxels
        if (
            hit := _ray_voxel_hit(
                origin,
                direction,
                voxel,
                tolerance_m=tie_tolerance_m,
            )
        )
        is not None
    ]
    if not hits:
        return TaskGraspAnnotation(
            candidate_id=candidate.candidate_id,
            target_components=(),
            target_voxels=(),
            target_surface_sectors=(),
            hit_points_object_m=(),
            capture_components=(),
            capture_voxels=(),
            resolved=False,
            ambiguous=False,
        )
    first_distance = min(hit.distance_m for hit in hits)
    first = [
        hit
        for hit in hits
        if abs(hit.distance_m - first_distance) <= tie_tolerance_m
    ]
    first.sort(
        key=lambda hit: (
            hit.component,
            hit.voxel_index,
            hit.surface_sector,
        )
    )
    components = tuple(sorted({hit.component for hit in first}))
    capture_voxels = tuple(
        voxel.index
        for voxel in voxels
        if any(
            _oriented_box_overlaps_voxel(
                candidate.object_T_G,
                center_G,
                half_extents_G,
                voxel,
            )
            for center_G, half_extents_G in capture_boxes_G
        )
    )
    voxel_by_index = {voxel.index: voxel for voxel in voxels}
    capture_components = tuple(
        sorted({voxel_by_index[index].component for index in capture_voxels})
    )
    return TaskGraspAnnotation(
        candidate_id=candidate.candidate_id,
        target_components=components,
        target_voxels=tuple(hit.voxel_index for hit in first),
        target_surface_sectors=tuple(hit.surface_sector for hit in first),
        hit_points_object_m=tuple(hit.point_object_m for hit in first),
        capture_components=capture_components,
        capture_voxels=capture_voxels,
        resolved=True,
        ambiguous=len(components) > 1,
    )


def load_capture_boxes(
    descriptor_config: str | Path,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Load the one or two descriptor capture boxes in frame G."""

    document: Mapping[str, Any] = yaml.safe_load(
        Path(descriptor_config).read_text()
    )
    sweep = document.get("sweep_volume")
    if not isinstance(sweep, Mapping):
        raise ValueError(f"{descriptor_config} has no sweep_volume")
    boxes = []
    suffixes = ("", "2")
    for suffix in suffixes:
        extents_key = f"extents{suffix}"
        offset_key = f"offset{suffix}"
        if extents_key not in sweep and offset_key not in sweep:
            continue
        extents = np.asarray(sweep[extents_key], dtype=np.float64)
        center = np.asarray(sweep[offset_key], dtype=np.float64)
        if extents.shape != (3,) or center.shape != (3,) or np.any(extents <= 0):
            raise ValueError(
                f"{descriptor_config} contains an invalid capture box {suffix or '1'}"
            )
        center.setflags(write=False)
        half = 0.5 * extents
        half.setflags(write=False)
        boxes.append((center, half))
    if not boxes:
        raise ValueError(f"{descriptor_config} contains no capture boxes")
    return tuple(boxes)


def _oriented_box_overlaps_voxel(
    object_T_G: np.ndarray,
    center_G: np.ndarray,
    half_extents_G: np.ndarray,
    voxel: SemanticVoxel,
) -> bool:
    """Exact 15-axis separating-axis test for a G-oriented box and voxel."""

    rotation = np.asarray(object_T_G[:3, :3], dtype=np.float64)
    center_object = (
        np.asarray(object_T_G, dtype=np.float64)
        @ np.array([*center_G, 1.0], dtype=np.float64)
    )[:3]
    delta = center_object - voxel.center
    absolute_rotation = np.abs(rotation) + 1e-12

    # The three object/voxel axes.
    for axis in range(3):
        radius_voxel = voxel.half_extents[axis]
        radius_capture = float(
            absolute_rotation[axis, :] @ half_extents_G
        )
        if abs(delta[axis]) > radius_voxel + radius_capture:
            return False

    # The three capture-box axes.
    delta_capture = rotation.T @ delta
    for axis in range(3):
        radius_capture = half_extents_G[axis]
        radius_voxel = float(
            absolute_rotation[:, axis] @ voxel.half_extents
        )
        if abs(delta_capture[axis]) > radius_capture + radius_voxel:
            return False

    # The nine pairwise cross-product axes.
    for object_axis in range(3):
        object_next = (object_axis + 1) % 3
        object_last = (object_axis + 2) % 3
        for capture_axis in range(3):
            capture_next = (capture_axis + 1) % 3
            capture_last = (capture_axis + 2) % 3
            radius_voxel = (
                voxel.half_extents[object_next]
                * absolute_rotation[object_last, capture_axis]
                + voxel.half_extents[object_last]
                * absolute_rotation[object_next, capture_axis]
            )
            radius_capture = (
                half_extents_G[capture_next]
                * absolute_rotation[object_axis, capture_last]
                + half_extents_G[capture_last]
                * absolute_rotation[object_axis, capture_next]
            )
            projected_delta = abs(
                delta[object_last] * rotation[object_next, capture_axis]
                - delta[object_next] * rotation[object_last, capture_axis]
            )
            if projected_delta > radius_voxel + radius_capture:
                return False
    return True


def annotate_pool(
    candidates: Iterable[GraspCandidate],
    geometry_config: str | Path,
    descriptor_config: str | Path | None = None,
) -> dict[str, TaskGraspAnnotation]:
    """Return a complete candidate-ID keyed annotation ledger."""

    values = tuple(candidates)
    voxels = load_semantic_voxels(geometry_config)
    capture_boxes = (
        () if descriptor_config is None else load_capture_boxes(descriptor_config)
    )
    output = {
        candidate.candidate_id: annotate_candidate(
            candidate, voxels, capture_boxes_G=capture_boxes
        )
        for candidate in values
    }
    if len(output) != len(values):
        raise ValueError("Candidate annotation identity collision")
    return output
