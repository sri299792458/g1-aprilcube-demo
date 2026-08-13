"""Condition immutable GraspGenX proposals on stable tabletop supports.

This module deliberately stops before grasp physics.  It answers a geometric
question for every ``object_T_G`` proposal:

    Can the exact open hand occupy the proposed pose and traverse the named
    straight pregrasp corridor while the object rests in this stable support?

Survivors are placed into proposal buckets using support orientation, the
semantic object component intersected by the canonical approach ray, its broad
surface relation, and the object-frame approach sector.  These buckets organize
physics evaluation; they never replace or prune the concrete candidates.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import fcl
import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml
from trimesh.collision import mesh_to_BVH


AXIS_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
AXIS_VECTOR = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_axis(vector: Sequence[float]) -> str:
    """Return the dominant signed Cartesian axis of a nonzero vector."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Axis vector must contain three finite values")
    index = int(np.argmax(np.abs(value)))
    if abs(float(value[index])) < 1e-12:
        raise ValueError("A zero vector has no signed axis")
    return ("+" if value[index] >= 0 else "-") + "XYZ"[index]


def matrix_from_isaac_grasp(entry: Mapping[str, Any]) -> np.ndarray:
    orientation = entry["orientation"]
    matrix = tra.quaternion_matrix(
        [float(orientation["w"]), *map(float, orientation["xyz"])]
    )
    matrix[:3, 3] = np.asarray(entry["position"], dtype=np.float64)
    return matrix


def pose_document(matrix: np.ndarray) -> dict[str, Any]:
    quaternion = tra.quaternion_from_matrix(matrix)
    return {
        "position": np.asarray(matrix[:3, 3], dtype=np.float64).tolist(),
        "orientation": {
            "w": float(quaternion[0]),
            "xyz": np.asarray(quaternion[1:], dtype=np.float64).tolist(),
        },
    }


@dataclass(frozen=True)
class RawCandidate:
    candidate_id: str
    score: float
    object_T_G: np.ndarray
    generation_seed: int
    sample_index: int
    content_sha256: str | None


def load_raw_candidates(raw_directory: Path) -> tuple[RawCandidate, ...]:
    """Load every raw shard exactly once without thresholding or ranking."""

    candidates: list[RawCandidate] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    paths = sorted(raw_directory.glob("shard_*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No raw grasp shards under {raw_directory}")
    for path in paths:
        document = yaml.safe_load(path.read_text())
        for candidate_id, entry in document["grasps"].items():
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate ID: {candidate_id}")
            generation = entry.get("graspgenx_generation", {})
            content = generation.get("candidate_content_sha256")
            if content is not None and content in seen_content:
                raise ValueError(f"Duplicate candidate content: {content}")
            seen_ids.add(candidate_id)
            if content is not None:
                seen_content.add(content)
            matrix = matrix_from_isaac_grasp(entry)
            matrix.setflags(write=False)
            candidates.append(
                RawCandidate(
                    candidate_id=str(candidate_id),
                    score=float(entry["confidence"]),
                    object_T_G=matrix,
                    generation_seed=int(generation["generation_seed"]),
                    sample_index=int(generation["sample_index"]),
                    content_sha256=None if content is None else str(content),
                )
            )
    return tuple(candidates)


def load_raw_candidates_many(
    raw_directories: Sequence[Path],
) -> tuple[RawCandidate, ...]:
    """Merge immutable raw sources while rejecting cross-source duplicates."""

    candidates: list[RawCandidate] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for raw_directory in raw_directories:
        for candidate in load_raw_candidates(raw_directory):
            if candidate.candidate_id in seen_ids:
                raise ValueError(
                    f"Duplicate candidate ID across raw sources: "
                    f"{candidate.candidate_id}"
                )
            if (
                candidate.content_sha256 is not None
                and candidate.content_sha256 in seen_content
            ):
                raise ValueError(
                    "Duplicate candidate content across raw sources: "
                    f"{candidate.content_sha256}"
                )
            seen_ids.add(candidate.candidate_id)
            if candidate.content_sha256 is not None:
                seen_content.add(candidate.content_sha256)
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError("No raw grasp candidates in configured sources")
    return tuple(candidates)


@dataclass(frozen=True)
class SupportCondition:
    support_id: str
    label: str
    table_up_object: np.ndarray
    table_up_sector: str
    symmetry_class: str
    support_T_object: np.ndarray


def configured_support_conditions(
    mesh: trimesh.Trimesh,
    *,
    entries: Sequence[Mapping[str, str]],
) -> tuple[SupportCondition, ...]:
    """Build the six explicit supports of a known voxel-aligned U.

    The configuration names the physical support and its geometric equivalence
    class.  The only computation here aligns the named object axis with table
    up and translates the exact mesh onto z=0.
    """

    supports = []
    seen_sectors: set[str] = set()
    for entry in entries:
        sector = str(entry["table_up_object"])
        if sector not in AXIS_VECTOR:
            raise ValueError(f"Unsupported configured support axis: {sector}")
        if sector in seen_sectors:
            raise ValueError(f"Duplicate configured support axis: {sector}")
        seen_sectors.add(sector)
        table_up_object = AXIS_VECTOR[sector].copy()
        world_up = np.array([0.0, 0.0, 1.0])
        if np.allclose(table_up_object, world_up):
            matrix = np.eye(4)
        elif np.allclose(table_up_object, -world_up):
            matrix = tra.rotation_matrix(math.pi, [1.0, 0.0, 0.0])
        else:
            rotation_axis = np.cross(table_up_object, world_up)
            rotation_axis /= np.linalg.norm(rotation_axis)
            matrix = tra.rotation_matrix(
                math.acos(float(np.clip(table_up_object @ world_up, -1.0, 1.0))),
                rotation_axis,
            )
        minimum_z = float(
            trimesh.transform_points(mesh.vertices, matrix)[:, 2].min()
        )
        matrix[2, 3] -= minimum_z
        support_id = f"table_up_object_{sector.replace('+', 'pos_').replace('-', 'neg_').lower()}"
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix.setflags(write=False)
        table_up_object.setflags(write=False)
        supports.append(
            SupportCondition(
                support_id=support_id,
                label=str(entry["label"]),
                table_up_object=table_up_object,
                table_up_sector=sector,
                symmetry_class=str(entry["symmetry_class"]),
                support_T_object=matrix,
            )
        )
    if seen_sectors != set(AXIS_ORDER):
        missing = [axis for axis in AXIS_ORDER if axis not in seen_sectors]
        raise ValueError(f"Configured U supports are incomplete: {missing}")
    supports.sort(key=lambda item: AXIS_ORDER.index(item.table_up_sector))
    return tuple(supports)


@dataclass(frozen=True)
class SemanticVoxel:
    index: tuple[int, int, int]
    component: str
    center: np.ndarray
    half_extents: np.ndarray

    def contains(self, point: np.ndarray, tolerance_m: float) -> bool:
        return bool(
            np.all(np.abs(point - self.center) <= self.half_extents + tolerance_m)
        )


def semantic_voxels(geometry_config: Path) -> tuple[SemanticVoxel, ...]:
    """Expand named, possibly overlapping cuboids into owned atomic voxels.

    Later cuboids own shared junction voxels.  For the U, ``hip_bridge`` is
    listed after the two legs, so its corner junctions remain bridge regions
    rather than being counted twice.
    """

    document = yaml.safe_load(geometry_config.read_text())
    shape = document["shape"]
    voxel_size = float(shape["voxel_size_mm"]) / 1000.0
    owners: dict[tuple[int, int, int], str] = {}
    for cuboid in shape["cuboids"]:
        origin = np.asarray(cuboid["origin"], dtype=np.int64)
        size = np.asarray(cuboid["size"], dtype=np.int64)
        for delta in itertools.product(*(range(int(value)) for value in size)):
            index = tuple((origin + np.asarray(delta, dtype=np.int64)).tolist())
            owners[index] = str(cuboid["name"])
    indices = np.asarray(list(owners), dtype=np.float64)
    union_center = 0.5 * (indices.min(axis=0) + indices.max(axis=0) + 1.0)
    half = np.full(3, 0.5 * voxel_size, dtype=np.float64)
    output = []
    for index in sorted(owners):
        center = (np.asarray(index, dtype=np.float64) + 0.5 - union_center) * voxel_size
        center.setflags(write=False)
        output.append(
            SemanticVoxel(
                index=index,
                component=owners[index],
                center=center,
                half_extents=half,
            )
        )
    return tuple(output)


class TargetRegionClassifier:
    """Classify the object surface targeted by GraspGenX local +Z."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        voxels: Sequence[SemanticVoxel],
        *,
        surface_tolerance_m: float,
    ):
        self.mesh = mesh
        self.voxels = tuple(voxels)
        self.surface_tolerance_m = float(surface_tolerance_m)
        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        self.hull = mesh.convex_hull

    def _component(self, hit: np.ndarray, normal: np.ndarray) -> str:
        inward = hit - normal * max(self.surface_tolerance_m, 1e-6)
        containing = [
            voxel
            for voxel in self.voxels
            if voxel.contains(inward, self.surface_tolerance_m)
        ]
        if not containing:
            return "unresolved"
        containing.sort(key=lambda voxel: float(np.linalg.norm(inward - voxel.center)))
        return containing[0].component

    def classify(
        self, object_T_G: np.ndarray, table_up_object: np.ndarray
    ) -> dict[str, Any]:
        origin = np.array(object_T_G[:3, 3], dtype=np.float64, copy=True)
        direction = np.array(object_T_G[:3, 2], dtype=np.float64, copy=True)
        direction /= np.linalg.norm(direction)
        locations, _, triangle_ids = self.ray.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=True,
        )
        if len(locations) == 0:
            return {
                "resolved": False,
                "component": "unresolved",
                "surface_sector": "unresolved",
                "surface_relation": "unresolved",
                "support_relation": "unresolved",
                "ray_event": "miss",
            }
        distances = (locations - origin) @ direction
        valid = np.flatnonzero(distances >= -self.surface_tolerance_m)
        if len(valid) == 0:
            return {
                "resolved": False,
                "component": "unresolved",
                "surface_sector": "unresolved",
                "surface_relation": "unresolved",
                "support_relation": "unresolved",
                "ray_event": "miss",
            }
        selected = int(valid[np.argmin(distances[valid])])
        hit = np.asarray(locations[selected], dtype=np.float64)
        normal = np.array(
            self.mesh.face_normals[int(triangle_ids[selected])],
            dtype=np.float64,
            copy=True,
        )
        normal /= np.linalg.norm(normal)
        _, hull_distance, _ = trimesh.proximity.closest_point(
            self.hull, hit.reshape(1, 3)
        )
        relation = (
            "exterior"
            if float(hull_distance[0]) <= self.surface_tolerance_m
            else "cavity"
        )
        support_dot = float(normal @ table_up_object)
        if support_dot >= math.sqrt(0.5):
            support_relation = "upward_exposed"
        elif support_dot <= -math.sqrt(0.5):
            support_relation = "table_facing"
        else:
            support_relation = "lateral"
        ray_dot = float(direction @ normal)
        ray_event = "entry" if ray_dot < -1e-6 else "exit" if ray_dot > 1e-6 else "tangent"
        return {
            "resolved": True,
            "component": self._component(hit, normal),
            "surface_sector": signed_axis(normal),
            "surface_relation": relation,
            "support_relation": support_relation,
            "ray_event": ray_event,
            "hit_object_m": hit.tolist(),
            "normal_object": normal.tolist(),
            "distance_from_G_m": float(max(0.0, distances[selected])),
        }


class MeshCollisionGate:
    """Cached exact-mesh binary collision queries for one hand and object."""

    def __init__(self, hand_mesh: trimesh.Trimesh, object_mesh: trimesh.Trimesh):
        self.hand = fcl.CollisionObject(mesh_to_BVH(hand_mesh))
        self.object = fcl.CollisionObject(mesh_to_BVH(object_mesh))
        self.request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)

    @staticmethod
    def _transform(matrix: np.ndarray) -> fcl.Transform:
        return fcl.Transform(
            np.asarray(matrix[:3, :3], dtype=np.float64),
            np.asarray(matrix[:3, 3], dtype=np.float64),
        )

    def set_object_pose(self, support_T_object: np.ndarray) -> None:
        self.object.setTransform(self._transform(support_T_object))

    def intersects(self, support_T_G: np.ndarray) -> bool:
        self.hand.setTransform(self._transform(support_T_G))
        result = fcl.CollisionResult()
        fcl.collide(self.hand, self.object, self.request, result)
        return bool(result.is_collision)


def local_z_offset(matrix: np.ndarray, distance_m: float) -> np.ndarray:
    return matrix @ tra.translation_matrix([0.0, 0.0, float(distance_m)])


def minimum_world_z_values(
    mesh: trimesh.Trimesh,
    matrices: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute exact mesh-vertex Z minima for many rigid transforms.

    A linear minimum over a point set is attained by a convex-hull vertex.
    Evaluating the hull therefore gives the same answer as transforming every
    dense triangle-mesh vertex, while batched matrix products avoid one NumPy
    call per candidate.
    """

    transforms = np.asarray(matrices, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("Expected transforms with shape (N, 4, 4)")
    if batch_size <= 0:
        raise ValueError("Minimum-Z batch size must be positive")
    hull_vertices = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    output = np.empty(len(transforms), dtype=np.float64)
    for start in range(0, len(transforms), batch_size):
        stop = min(start + batch_size, len(transforms))
        # With row-vector mesh coordinates, transformed z is
        # vertex @ rotation_row_z + translation_z.
        support = (
            transforms[start:stop, 2, :3] @ hull_vertices.T
        )
        output[start:stop] = (
            support.min(axis=1) + transforms[start:stop, 2, 3]
        )
    return output


def _bucket_key(
    support: SupportCondition,
    region: Mapping[str, Any],
    object_T_G: np.ndarray,
) -> dict[str, str]:
    return {
        "support_id": support.support_id,
        "support_symmetry_class": support.symmetry_class,
        "component": str(region["component"]),
        "surface_sector": str(region["surface_sector"]),
        "surface_relation": str(region["surface_relation"]),
        "support_relation": str(region["support_relation"]),
        "approach_sector": signed_axis(object_T_G[:3, 2]),
    }


def bucket_id(key: Mapping[str, str]) -> str:
    payload = json.dumps(dict(key), sort_keys=True, separators=(",", ":"))
    return "proposal_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def evaluate_support(
    *,
    support: SupportCondition,
    candidates: Sequence[RawCandidate],
    hand_mesh: trimesh.Trimesh,
    collision: MeshCollisionGate,
    classifier: TargetRegionClassifier,
    approach_offset_m: float,
    corridor_step_m: float,
    table_tolerance_m: float,
) -> dict[str, Any]:
    """Evaluate all candidates under one support without physics pruning."""

    if approach_offset_m >= 0:
        raise ValueError("Pregrasp offset must be negative local Z")
    if corridor_step_m <= 0:
        raise ValueError("Corridor step must be positive")
    collision.set_object_pose(support.support_T_object)
    counts: Counter[str] = Counter()
    rejection: Counter[str] = Counter()
    survivors = []
    steps = int(math.ceil(abs(approach_offset_m) / corridor_step_m))
    interior_fractions = np.linspace(0.0, 1.0, steps + 1)[1:-1]
    object_T_G = np.stack([candidate.object_T_G for candidate in candidates])
    final_matrices = np.einsum(
        "ij,njk->nik", support.support_T_object, object_T_G
    )
    final_minimum_z = minimum_world_z_values(hand_mesh, final_matrices)
    # A negative-local-Z pregrasp changes only translation. Its world-Z
    # displacement is offset * the final frame's local-Z world component.
    pregrasp_minimum_z = (
        final_minimum_z
        + float(approach_offset_m) * final_matrices[:, 2, 2]
    )

    for candidate_index, candidate in enumerate(candidates):
        counts["raw"] += 1
        final = final_matrices[candidate_index]
        final_min_z = float(final_minimum_z[candidate_index])
        pregrasp_min_z = float(pregrasp_minimum_z[candidate_index])
        if final_min_z < -table_tolerance_m:
            rejection["final_table"] += 1
            continue
        counts["final_table_clear"] += 1
        if pregrasp_min_z < -table_tolerance_m:
            rejection["pregrasp_table"] += 1
            continue
        counts["pregrasp_table_clear"] += 1
        pregrasp = local_z_offset(final, approach_offset_m)
        if collision.intersects(final):
            rejection["final_object"] += 1
            continue
        counts["final_object_clear"] += 1
        if collision.intersects(pregrasp):
            rejection["pregrasp_object"] += 1
            continue
        counts["pregrasp_object_clear"] += 1
        corridor_collision = False
        for fraction in interior_fractions:
            distance = approach_offset_m * (1.0 - float(fraction))
            if collision.intersects(local_z_offset(final, distance)):
                corridor_collision = True
                break
        if corridor_collision:
            rejection["approach_object"] += 1
            continue
        counts["approach_corridor_clear"] += 1

        region = classifier.classify(
            candidate.object_T_G, support.table_up_object
        )
        key = _bucket_key(support, region, candidate.object_T_G)
        identifier = bucket_id(key)
        survivors.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_content_sha256": candidate.content_sha256,
                "generation_seed": candidate.generation_seed,
                "sample_index": candidate.sample_index,
                "graspgenx_score": candidate.score,
                "object_T_G": pose_document(candidate.object_T_G),
                "support_T_G": pose_document(final),
                "support_T_pregrasp_G": pose_document(pregrasp),
                "final_table_clearance_m": final_min_z,
                "pregrasp_table_clearance_m": pregrasp_min_z,
                "target_region": region,
                "approach_sector_object": signed_axis(candidate.object_T_G[:3, 2]),
                "proposal_bucket_id": identifier,
                "proposal_bucket_key": key,
            }
        )
        counts["region_resolved"] += int(bool(region["resolved"]))
        counts["region_unresolved"] += int(not bool(region["resolved"]))

    return {
        "support": {
            "support_id": support.support_id,
            "label": support.label,
            "table_up_object": support.table_up_object.tolist(),
            "table_up_sector": support.table_up_sector,
            "symmetry_class": support.symmetry_class,
            "support_T_object": support.support_T_object.tolist(),
        },
        "stage_counts": dict(counts),
        "first_rejection_reason_counts": dict(rejection),
        "survivor_count": len(survivors),
        "survivors": survivors,
    }


def build_buckets(support_results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group every survivor exactly once without selecting representatives."""

    grouped: dict[str, dict[str, Any]] = {}
    total_survivors = 0
    for support in support_results:
        for survivor in support["survivors"]:
            total_survivors += 1
            identifier = survivor["proposal_bucket_id"]
            if identifier not in grouped:
                grouped[identifier] = {
                    "proposal_bucket_id": identifier,
                    "key": survivor["proposal_bucket_key"],
                    "member_ids": [],
                }
            elif grouped[identifier]["key"] != survivor["proposal_bucket_key"]:
                raise RuntimeError(f"Bucket hash collision: {identifier}")
            grouped[identifier]["member_ids"].append(survivor["candidate_id"])
    if sum(len(value["member_ids"]) for value in grouped.values()) != total_survivors:
        raise RuntimeError("Proposal bucketing lost a support-conditioned survivor")
    output = []
    for identifier in sorted(grouped):
        value = grouped[identifier]
        value["member_ids"].sort()
        value["member_count"] = len(value["member_ids"])
        output.append(value)
    return output
