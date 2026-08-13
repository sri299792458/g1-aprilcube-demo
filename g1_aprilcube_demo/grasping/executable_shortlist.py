"""Build a small runtime shortlist from recorded grasp-closure evidence.

The intrinsic Isaac atlas answers a deliberately broad question: did the hand
eventually retain the object through the disturbance sequence?  Runtime motion
planning needs a stricter contract.  A candidate admitted here must also start
without hand/object intersection, clear the known tabletop support at its
grasp and pregrasp endpoints, establish opposing digit-group contact as soon
as closure finishes, and avoid large object rearrangement during closure.

No pose is generated, averaged, corrected, or clustered here.  Every output
``object_T_G`` is copied unchanged from GraspGenX.  The output order is only
for deterministic serialization; all candidates are intended to enter one
cuRobo goal set simultaneously.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import fcl
import numpy as np
import trimesh
from trimesh.collision import mesh_to_BVH
import trimesh.transformations as tra
import yaml
import yourdfpy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_matrix(document: Mapping[str, Any]) -> np.ndarray:
    orientation = document["orientation"]
    matrix = tra.quaternion_matrix(
        [float(orientation["w"]), *map(float, orientation["xyz"])]
    )
    matrix[:3, 3] = np.asarray(document["position"], dtype=np.float64)
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


def relative_pose_change(
    initial_object_T_G: np.ndarray,
    closed_object_T_G: np.ndarray,
) -> tuple[float, float]:
    """Return the object's world motion while the hand root remains fixed.

    The intrinsic Isaac closure keeps ``world_T_G`` fixed and records the
    changing ``object_T_G = inverse(world_T_object) @ world_T_G``.  With the
    initial object frame chosen as world, the moved object pose is therefore
    ``initial_object_T_G @ inverse(closed_object_T_G)``.
    """

    world_T_closed_object = initial_object_T_G @ np.linalg.inv(closed_object_T_G)
    cosine = float(
        np.clip(
            (np.trace(world_T_closed_object[:3, :3]) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )
    return (
        float(np.linalg.norm(world_T_closed_object[:3, 3])),
        math.degrees(math.acos(cosine)),
    )


def contact_group_force(
    contacts: Iterable[Mapping[str, Any]],
    links: set[str],
) -> float:
    """Maximum trustworthy body-level object-contact magnitude in a group."""

    return max(
        (
            float(contact["contact_force_magnitude_N"])
            for contact in contacts
            if str(contact["hand_link"]) in links
        ),
        default=0.0,
    )


@dataclass(frozen=True)
class GeometryEvidence:
    open_hand_object_collision_free: bool
    grasp_table_clearance_m: float
    pregrasp_table_clearance_m: float


class OpenHandGeometry:
    """Exact descriptor-open meshes used for collision and support gates."""

    def __init__(
        self,
        *,
        object_mesh: trimesh.Trimesh,
        hand_visual_mesh: trimesh.Trimesh,
        hand_collision_mesh: trimesh.Trimesh,
        hand_urdf_path: Path,
        approach_distance_m: float,
        numerical_tolerance_m: float,
    ) -> None:
        self.object_mesh = object_mesh
        self.object_vertices_h = np.column_stack(
            (
                np.asarray(object_mesh.vertices, dtype=np.float64),
                np.ones(len(object_mesh.vertices), dtype=np.float64),
            )
        )
        self.hand_vertices_h = np.column_stack(
            (
                np.asarray(hand_visual_mesh.vertices, dtype=np.float64),
                np.ones(len(hand_visual_mesh.vertices), dtype=np.float64),
            )
        )
        self.hand_collision = mesh_to_BVH(hand_collision_mesh)
        self.object_collision = fcl.CollisionObject(mesh_to_BVH(object_mesh))
        self.hand_robot = yourdfpy.URDF.load(
            str(hand_urdf_path),
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )
        self.approach_distance_m = float(approach_distance_m)
        self.numerical_tolerance_m = float(numerical_tolerance_m)
        self.support_plane_object_z = float(object_mesh.bounds[0, 2])

    def evaluate(self, object_T_G: np.ndarray) -> GeometryEvidence:
        hand = fcl.CollisionObject(
            self.hand_collision,
            fcl.Transform(object_T_G[:3, :3], object_T_G[:3, 3]),
        )
        collision_count = fcl.collide(
            self.object_collision,
            hand,
            fcl.CollisionRequest(num_max_contacts=1, enable_contact=False),
            fcl.CollisionResult(),
        )

        object_T_pregrasp_G = object_T_G @ tra.translation_matrix(
            [0.0, 0.0, -self.approach_distance_m]
        )
        grasp_vertices = self.hand_vertices_h @ object_T_G.T
        pregrasp_vertices = self.hand_vertices_h @ object_T_pregrasp_G.T
        return GeometryEvidence(
            open_hand_object_collision_free=collision_count == 0,
            grasp_table_clearance_m=float(
                grasp_vertices[:, 2].min() - self.support_plane_object_z
            ),
            pregrasp_table_clearance_m=float(
                pregrasp_vertices[:, 2].min() - self.support_plane_object_z
            ),
        )

    def table_clear(self, evidence: GeometryEvidence) -> bool:
        # The local-Z approach is a pure translation.  Every vertex's height
        # is affine along it, so checking both endpoints proves the complete
        # straight corridor clears the horizontal support plane.
        threshold = -self.numerical_tolerance_m
        return (
            evidence.grasp_table_clearance_m >= threshold
            and evidence.pregrasp_table_clearance_m >= threshold
        )

    def closed_table_clearance(
        self,
        fixed_world_T_G: np.ndarray,
        closed_q: Mapping[str, float],
    ) -> float:
        """Minimum exact closed-hand collision-mesh height over the support."""

        self.hand_robot.update_cfg(dict(closed_q))
        minimum_z = math.inf
        scene = self.hand_robot.collision_scene
        for geometry_name, mesh in scene.geometry.items():
            G_T_geometry = scene.graph.get(
                frame_from=scene.graph.base_frame,
                frame_to=geometry_name,
            )[0]
            vertices_h = np.column_stack(
                (
                    np.asarray(mesh.vertices, dtype=np.float64),
                    np.ones(len(mesh.vertices), dtype=np.float64),
                )
            )
            vertices_object = vertices_h @ (
                fixed_world_T_G @ G_T_geometry
            ).T
            minimum_z = min(minimum_z, float(vertices_object[:, 2].min()))
        if not math.isfinite(minimum_z):
            raise ValueError("Dex3 URDF has no collision geometry")
        return minimum_z - self.support_plane_object_z

    def moved_object_table_clearance(
        self,
        world_T_object: np.ndarray,
    ) -> float:
        vertices_world = self.object_vertices_h @ world_T_object.T
        return float(vertices_world[:, 2].min() - self.support_plane_object_z)


def load_trace_records(patterns: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(value) for value in glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError("No contact traces matched the configured patterns")
    for path in sorted(set(paths)):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                candidate_id = str(record["candidate_id"])
                if candidate_id in records:
                    raise ValueError(f"Duplicate contact-trace candidate: {candidate_id}")
                records[candidate_id] = record
    return records


def _closed_phase(record: Mapping[str, Any], phase_name: str) -> Mapping[str, Any]:
    matches = [phase for phase in record["phases"] if phase["name"] == phase_name]
    if len(matches) != 1:
        raise ValueError(
            f"{record['candidate_id']} has {len(matches)} {phase_name!r} phases"
        )
    return matches[0]


def build_shortlist(
    *,
    config: Mapping[str, Any],
    source_pool: Mapping[str, Any],
    trace_records: Mapping[str, Mapping[str, Any]],
    geometry: OpenHandGeometry,
    object_mesh_path: Path,
    source_paths: Mapping[str, Any],
) -> dict[str, Any]:
    contract = config["execution_contract"]
    phase_name = str(contract["closure_phase"])
    force_epsilon = float(contract["contact_force_numerical_epsilon_N"])
    max_translation = 0.5 * float(np.min(geometry.object_mesh.extents))
    max_rotation_deg = float(contract["max_closure_rotation_deg"])
    tolerance = float(contract["pose_match_tolerance"])
    group_links = {
        name: set(map(str, values))
        for name, values in config["hand"]["contact_groups"].items()
    }
    required_groups = tuple(map(str, contract["required_contact_groups"]))
    if set(required_groups) != set(group_links):
        raise ValueError("required_contact_groups must exactly name contact_groups")

    rejection_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    source_candidates = source_pool["candidates"]
    for source_index, source in enumerate(source_candidates):
        candidate_id = str(source["candidate_id"])
        record = trace_records.get(candidate_id)
        reasons: list[str] = []
        if record is None:
            reasons.append("missing_contact_trace")
        elif not bool(record["result"]["passed"]):
            reasons.append("intrinsic_retention_failed")
        if reasons:
            rejection_counts.update(reasons)
            continue

        assert record is not None
        source_pose = pose_matrix(source["object_T_G"])
        input_pose = np.asarray(record["input"]["object_T_G"], dtype=np.float64)
        if not np.allclose(source_pose, input_pose, atol=tolerance, rtol=0.0):
            raise ValueError(f"Immutable object_T_G changed for {candidate_id}")
        if source.get("candidate_content_sha256") != record.get(
            "candidate_content_sha256"
        ):
            raise ValueError(f"Candidate provenance changed for {candidate_id}")

        closed = _closed_phase(record, phase_name)
        closed_pose = np.asarray(closed["object_T_G"], dtype=np.float64)
        closed_world_T_object = input_pose @ np.linalg.inv(closed_pose)
        translation_m, rotation_deg = relative_pose_change(input_pose, closed_pose)
        forces = {
            name: contact_group_force(closed["contacts"], links)
            for name, links in group_links.items()
        }
        geometry_evidence = geometry.evaluate(input_pose)
        closed_table_clearance_m = geometry.closed_table_clearance(
            input_pose,
            closed["q"],
        )
        closed_object_table_clearance_m = geometry.moved_object_table_clearance(
            closed_world_T_object
        )

        if not geometry_evidence.open_hand_object_collision_free:
            reasons.append("initial_open_hand_object_collision")
        if not geometry.table_clear(geometry_evidence):
            reasons.append("table_or_pregrasp_corridor_collision")
        if closed_table_clearance_m < -geometry.numerical_tolerance_m:
            reasons.append("closed_hand_table_collision")
        if closed_object_table_clearance_m < -geometry.numerical_tolerance_m:
            reasons.append("closed_object_table_collision")
        for group in required_groups:
            if forces[group] <= force_epsilon:
                reasons.append(f"missing_{group}_contact_after_closure")
        if translation_m > max_translation:
            reasons.append("closure_translation_exceeds_object_half_width")
        if rotation_deg > max_rotation_deg:
            reasons.append("closure_rotation_exceeds_half_face_turn")
        if reasons:
            rejection_counts.update(reasons)
            continue

        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_content_sha256": source.get(
                    "candidate_content_sha256"
                ),
                "source_pool_index": source_index,
                "graspgenx_score": float(source["graspgenx_score"]),
                "object_T_G": pose_document(input_pose),
                "execution_evidence": {
                    "intrinsic_retention_passed": True,
                    "closure_phase": phase_name,
                    "closure_translation_m": translation_m,
                    "closure_translation_object_width_fraction": (
                        translation_m / float(np.min(geometry.object_mesh.extents))
                    ),
                    "closure_rotation_deg": rotation_deg,
                    "closure_contact_group_max_force_N": forces,
                    "closure_contact_links": sorted(
                        str(contact["hand_link"])
                        for contact in closed["contacts"]
                        if float(contact["contact_force_magnitude_N"])
                        > force_epsilon
                    ),
                    "open_hand_object_collision_free": (
                        geometry_evidence.open_hand_object_collision_free
                    ),
                    "grasp_table_clearance_m": (
                        geometry_evidence.grasp_table_clearance_m
                    ),
                    "pregrasp_table_clearance_m": (
                        geometry_evidence.pregrasp_table_clearance_m
                    ),
                    "closed_hand_table_clearance_m": closed_table_clearance_m,
                    "closed_object_table_clearance_m": (
                        closed_object_table_clearance_m
                    ),
                    "isaac_closed_object_T_G": closed_pose.tolist(),
                    "isaac_closed_world_T_object": closed_world_T_object.tolist(),
                    "isaac_closed_q": closed["q"],
                },
            }
        )

    # Deterministic serialization only.  The runtime contract submits every
    # admitted candidate in one goal set; this order is not a planner cost.
    candidates.sort(
        key=lambda item: (
            item["execution_evidence"]["closure_translation_m"],
            item["execution_evidence"]["closure_rotation_deg"],
            -min(
                item["execution_evidence"][
                    "closure_contact_group_max_force_N"
                ].values()
            ),
            -item["graspgenx_score"],
            item["candidate_id"],
        )
    )
    if not candidates:
        raise ValueError("The executable-grasp contract admitted no candidates")

    return {
        "format": "g1_aprilcube_executable_grasp_shortlist",
        "format_version": 1,
        "shortlist_id": str(config["shortlist_id"]),
        "hand_side": str(config["hand_side"]),
        "object_id": str(config["object"]["id"]),
        "object_mesh": str(config["object"]["mesh"]),
        "object_mesh_sha256": sha256(object_mesh_path),
        "candidate_count": len(candidates),
        "runtime_policy": {
            "candidate_grouping": "none",
            "candidate_submission": "all_candidates_in_one_curobo_goalset",
            "curobo_selection_role": "reachability_and_collision_only",
            "serialization_order_is_planner_cost": False,
        },
        "execution_contract": {
            "applicability": "upright_cube_with_arbitrary_tabletop_yaw",
            "source_retention_pass_required": True,
            "initial_open_hand_object_collision_free": True,
            "table_and_straight_pregrasp_corridor_clear": True,
            "closed_hand_table_clear": True,
            "closed_object_table_clear": True,
            "approach_distance_m": geometry.approach_distance_m,
            "required_contact_groups_after_closure": list(required_groups),
            "contact_force_numerical_epsilon_N": force_epsilon,
            "max_closure_translation_m": max_translation,
            "max_closure_translation_basis": "half_of_smallest_object_extent",
            "max_closure_rotation_deg": max_rotation_deg,
            "max_closure_rotation_basis": (
                "half_of_90_degree_interval_to_adjacent_cube_face_orientation"
            ),
        },
        "audit": {
            "source_candidate_count": len(source_candidates),
            "admitted_candidate_count": len(candidates),
            "rejection_counts_nonexclusive": dict(sorted(rejection_counts.items())),
        },
        "source": dict(source_paths),
        "candidates": candidates,
    }
