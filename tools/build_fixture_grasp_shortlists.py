#!/usr/bin/env python3
"""Build exact tripod/table-conditioned Dex3 grasp proposal sets.

This reuses the ordinary VIRAL-profile Isaac retention evidence and the
existing executable-grasp gates. It does not rerun inference or physics and it
does not alter any GraspGenX pose. The additional gate is purely geometric:
the open-hand approach and recorded closing motion must clear the manufactured
tripod and tabletop.
"""

from __future__ import annotations

from collections import Counter
import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import fcl
import numpy as np
import trimesh
from trimesh.collision import mesh_to_BVH
import trimesh.transformations as tra
import yaml
import yourdfpy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.grasping.executable_shortlist import (  # noqa: E402
    OpenHandGeometry,
    build_shortlist,
    load_trace_records,
    pose_matrix,
    sha256,
)


DEFAULT_CONFIG = ROOT / "config/grasp_shortlists/cube_tripod_right_v1.yaml"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def fcl_transform(matrix: np.ndarray) -> fcl.Transform:
    return fcl.Transform(matrix[:3, :3], matrix[:3, 3])


def collision(
    moving_geometry: fcl.CollisionGeometry,
    moving_pose: np.ndarray,
    obstacle: fcl.CollisionObject,
) -> bool:
    moving = fcl.CollisionObject(moving_geometry, fcl_transform(moving_pose))
    result = fcl.CollisionResult()
    return bool(
        fcl.collide(
            moving,
            obstacle,
            fcl.CollisionRequest(num_max_contacts=1, enable_contact=False),
            result,
        )
    )


def continuous_collision(
    moving_geometry: fcl.CollisionGeometry,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    obstacle: fcl.CollisionObject,
    *,
    motion_type: int,
    iterations: int,
    toc_tolerance: float,
) -> bool:
    moving = fcl.CollisionObject(moving_geometry, fcl_transform(start_pose))
    request = fcl.ContinuousCollisionRequest(
        num_max_iterations=iterations,
        toc_err=toc_tolerance,
        ccd_motion_type=motion_type,
        ccd_solver_type=fcl.CCDSolverType.CCDC_NAIVE,
    )
    result = fcl.ContinuousCollisionResult()
    fcl.continuousCollide(
        moving,
        fcl_transform(end_pose),
        obstacle,
        fcl.Transform(),
        request,
        result,
    )
    return bool(result.is_collide)


class FixtureCollisionGate:
    def __init__(
        self,
        *,
        fixture_mesh: trimesh.Trimesh,
        table_width_m: float,
        table_depth_m: float,
        table_thickness_m: float,
        hand_open_mesh: trimesh.Trimesh,
        hand_urdf_path: Path,
        open_q: Mapping[str, float],
        closure_segments: int,
        ccd_iterations: int,
        toc_tolerance: float,
    ) -> None:
        table = trimesh.creation.box(
            extents=[table_width_m, table_depth_m, table_thickness_m]
        )
        table.apply_translation([0.0, 0.0, -table_thickness_m / 2.0])
        obstacle_mesh = trimesh.util.concatenate([fixture_mesh, table])
        self.obstacle = fcl.CollisionObject(mesh_to_BVH(obstacle_mesh), fcl.Transform())
        self.open_hand_geometry = mesh_to_BVH(hand_open_mesh)
        self.robot = yourdfpy.URDF.load(
            str(hand_urdf_path),
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )
        self.geometry_bvhs = {
            name: mesh_to_BVH(mesh)
            for name, mesh in self.robot.collision_scene.geometry.items()
        }
        self.geometry_names = tuple(sorted(self.geometry_bvhs))
        self.open_q = {str(name): float(value) for name, value in open_q.items()}
        self.closure_segments = int(closure_segments)
        if self.closure_segments < 1:
            raise ValueError("Closure CCD requires at least one segment")
        self.ccd_iterations = int(ccd_iterations)
        self.toc_tolerance = float(toc_tolerance)

    def approach_clear(
        self,
        world_T_G: np.ndarray,
        approach_distance_m: float,
    ) -> bool:
        pregrasp = world_T_G @ tra.translation_matrix(
            [0.0, 0.0, -float(approach_distance_m)]
        )
        if collision(self.open_hand_geometry, pregrasp, self.obstacle):
            return False
        if collision(self.open_hand_geometry, world_T_G, self.obstacle):
            return False
        return not continuous_collision(
            self.open_hand_geometry,
            pregrasp,
            world_T_G,
            self.obstacle,
            motion_type=fcl.CCDMotionType.CCDM_TRANS,
            iterations=self.ccd_iterations,
            toc_tolerance=self.toc_tolerance,
        )

    def _geometry_poses(
        self,
        world_T_G: np.ndarray,
        q: Mapping[str, float],
    ) -> dict[str, np.ndarray]:
        self.robot.update_cfg(dict(q))
        scene = self.robot.collision_scene
        poses = {}
        for name in self.geometry_names:
            G_T_geometry = scene.graph.get(
                frame_from=scene.graph.base_frame,
                frame_to=name,
            )[0]
            poses[name] = world_T_G @ G_T_geometry
        return poses

    def closure_clear(
        self,
        world_T_G: np.ndarray,
        closed_q: Mapping[str, float],
    ) -> bool:
        closed = {str(name): float(value) for name, value in closed_q.items()}
        if set(closed) != set(self.open_q):
            raise ValueError("Recorded closed q does not match the descriptor joints")
        previous = self._geometry_poses(world_T_G, self.open_q)
        for step in range(1, self.closure_segments + 1):
            fraction = step / self.closure_segments
            q = {
                name: self.open_q[name]
                + fraction * (closed[name] - self.open_q[name])
                for name in self.open_q
            }
            current = self._geometry_poses(world_T_G, q)
            for name in self.geometry_names:
                geometry = self.geometry_bvhs[name]
                if collision(geometry, current[name], self.obstacle):
                    return False
                if continuous_collision(
                    geometry,
                    previous[name],
                    current[name],
                    self.obstacle,
                    motion_type=fcl.CCDMotionType.CCDM_LINEAR,
                    iterations=self.ccd_iterations,
                    toc_tolerance=self.toc_tolerance,
                ):
                    return False
            previous = current
        return True


def compact_candidate(
    candidate: Mapping[str, Any],
    valid_approaches: list[float],
) -> dict[str, Any]:
    evidence = candidate["execution_evidence"]
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_content_sha256": candidate["candidate_content_sha256"],
        "source_pool_index": candidate["source_pool_index"],
        "graspgenx_score": candidate["graspgenx_score"],
        "object_T_G": candidate["object_T_G"],
        "valid_approach_distances_m": valid_approaches,
        "execution_evidence": {
            "intrinsic_retention_passed": True,
            "closure_translation_m": evidence["closure_translation_m"],
            "closure_rotation_deg": evidence["closure_rotation_deg"],
            "closure_contact_group_max_force_N": evidence[
                "closure_contact_group_max_force_N"
            ],
            "closure_contact_links": evidence["closure_contact_links"],
            "open_hand_object_collision_free": True,
            "fixture_and_table_approach_clear": True,
            "fixture_and_table_closure_clear": True,
        },
    }


def load_fixture(path: Path, units: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise ValueError(f"Fixture is not one valid watertight mesh: {path}")
    if units == "millimeter":
        mesh.apply_scale(0.001)
    elif units != "meter":
        raise ValueError(f"Unsupported fixture mesh unit: {units}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    base_config_path = project_path(config["base_shortlist_config"])
    base_config = yaml.safe_load(base_config_path.read_text())
    object_path = project_path(base_config["object"]["mesh"])
    hand_visual_path = project_path(base_config["hand"]["visual_mesh"])
    hand_collision_path = project_path(base_config["hand"]["collision_mesh"])
    hand_urdf_path = project_path(base_config["hand"]["urdf"])
    source_pool_path = project_path(base_config["source"]["arm_grasp_pool"])
    source_pool = yaml.safe_load(source_pool_path.read_text())
    trace_patterns = [
        str(project_path(value))
        for value in base_config["source"]["contact_trace_globs"]
    ]
    traces = load_trace_records(trace_patterns)
    object_mesh = trimesh.load(object_path, force="mesh", process=True)
    hand_visual = trimesh.load(hand_visual_path, force="mesh", process=True)
    hand_collision = trimesh.load(hand_collision_path, force="mesh", process=True)
    descriptor = json.loads(project_path("config/dex3_rev1_descriptor.json").read_text())
    open_q = descriptor["finger_profile"]["right"]["open"]

    approaches = [
        float(value)
        for value in config["execution_contract"]["approach_distances_m"]
    ]
    if sorted(set(approaches), reverse=True) != approaches:
        raise ValueError("Approach distances must be unique and longest-first")
    shortest_approach = min(approaches)
    output_root = project_path(config["output_root"])

    for fixture_spec in config["fixture"]["variants"]:
        fixture_id = str(fixture_spec["id"])
        fixture_path = project_path(fixture_spec["mesh"])
        height = float(fixture_spec["support_height_m"])
        fixture_mesh = load_fixture(fixture_path, str(config["fixture"]["mesh_units"]))
        if not math.isclose(float(fixture_mesh.bounds[1, 2]), height, abs_tol=1e-7):
            raise ValueError(
                f"{fixture_id} mesh height {fixture_mesh.bounds[1, 2]} != {height}"
            )

        variant_config = copy.deepcopy(base_config)
        variant_config["shortlist_id"] = (
            f"{config['shortlist_id_prefix']}_{fixture_id}"
        )
        variant_config["execution_contract"]["approach_distance_m"] = (
            shortest_approach
        )
        geometry = OpenHandGeometry(
            object_mesh=object_mesh,
            hand_visual_mesh=hand_visual,
            hand_collision_mesh=hand_collision,
            hand_urdf_path=hand_urdf_path,
            approach_distance_m=shortest_approach,
            numerical_tolerance_m=float(
                base_config["execution_contract"]["numerical_geometry_tolerance_m"]
            ),
            support_plane_offset_below_object_m=height,
        )
        height_candidates = build_shortlist(
            config=variant_config,
            source_pool=source_pool,
            trace_records=traces,
            geometry=geometry,
            object_mesh_path=object_path,
            source_paths={
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256(config_path),
                "base_shortlist_config": str(base_config_path.relative_to(ROOT)),
                "base_shortlist_config_sha256": sha256(base_config_path),
                "arm_grasp_pool": str(source_pool_path.relative_to(ROOT)),
                "arm_grasp_pool_sha256": sha256(source_pool_path),
            },
        )

        gate = FixtureCollisionGate(
            fixture_mesh=fixture_mesh,
            table_width_m=float(config["table"]["width_m"]),
            table_depth_m=float(config["table"]["depth_m"]),
            table_thickness_m=float(config["table"]["collision_thickness_m"]),
            hand_open_mesh=hand_collision,
            hand_urdf_path=hand_urdf_path,
            open_q=open_q,
            closure_segments=int(
                config["execution_contract"]["closure_ccd_segments"]
            ),
            ccd_iterations=int(
                config["execution_contract"]["ccd_max_iterations"]
            ),
            toc_tolerance=float(
                config["execution_contract"][
                    "ccd_time_of_contact_tolerance"
                ]
            ),
        )
        world_T_object = tra.euler_matrix(
            0.0,
            0.0,
            math.radians(float(config["fixture"]["cube_yaw_deg"])),
            axes="sxyz",
        )
        world_T_object[:2, 3] = np.asarray(config["fixture"]["cube_xy_m"])
        world_T_object[2, 3] = height - float(object_mesh.bounds[0, 2])

        rejection = Counter()
        count_by_approach = Counter()
        admitted = []
        for candidate in height_candidates["candidates"]:
            object_T_G = pose_matrix(candidate["object_T_G"])
            world_T_G = world_T_object @ object_T_G
            closed_q = candidate["execution_evidence"]["isaac_closed_q"]
            if not gate.closure_clear(world_T_G, closed_q):
                rejection["fixture_or_table_collision_during_closure"] += 1
                continue
            valid_approaches = []
            for distance in approaches:
                if gate.approach_clear(world_T_G, distance):
                    valid_approaches.append(distance)
                    count_by_approach[f"{distance:.2f}"] += 1
                else:
                    rejection[f"fixture_or_table_collision_approach_{distance:.2f}m"] += 1
            if not valid_approaches:
                rejection["no_runtime_approach_distance_clear"] += 1
                continue
            admitted.append(compact_candidate(candidate, valid_approaches))

        if not admitted:
            raise RuntimeError(f"{fixture_id} admitted no fixture-qualified candidates")
        valid_sets = {
            distance: {
                item["candidate_id"]
                for item in admitted
                if distance in item["valid_approach_distances_m"]
            }
            for distance in approaches
        }
        for longer, shorter in zip(approaches, approaches[1:]):
            if not valid_sets[longer].issubset(valid_sets[shorter]):
                raise RuntimeError(
                    f"Fixture CCD violated nested approach paths: {longer} vs {shorter}"
                )

        output = {
            "format": "g1_aprilcube_fixture_grasp_shortlist",
            "format_version": 1,
            "shortlist_id": variant_config["shortlist_id"],
            "hand_side": "right",
            "object_id": base_config["object"]["id"],
            "object_mesh": base_config["object"]["mesh"],
            "object_mesh_sha256": sha256(object_path),
            "fixture": {
                "id": fixture_id,
                "mesh": str(fixture_path.relative_to(ROOT)),
                "mesh_sha256": sha256(fixture_path),
                "support_height_m": height,
                "world_T_object": world_T_object.tolist(),
                "cube_pose_contract": "centred_and_yaw_aligned",
            },
            "candidate_count": len(admitted),
            "candidate_counts_by_exact_approach_distance_m": {
                f"{distance:.2f}": int(count_by_approach[f"{distance:.2f}"])
                for distance in approaches
            },
            "runtime_policy": {
                "candidate_submission": "all_candidates_in_one_curobo_goalset",
                "curobo_selection_role": "arm_reachability_and_full_scene_collision",
                "measured_cube_pose_recheck_required": True,
            },
            "execution_contract": {
                "source_retention_pass_required": True,
                "open_hand_object_collision_free": True,
                "exact_fixture_and_table_mesh": True,
                "continuous_open_hand_approach_ccd": True,
                "approach_distances_m": approaches,
                "dex3_closure_ccd_segments": int(
                    config["execution_contract"]["closure_ccd_segments"]
                ),
                "target_fixture_contact": "intentional_and_not_rejected",
            },
            "audit": {
                "retention_pass_source_count": len(source_pool["candidates"]),
                "shortest_approach_height_gate_count": height_candidates[
                    "candidate_count"
                ],
                "admitted_candidate_count": len(admitted),
                "height_gate_rejection_counts_nonexclusive": height_candidates[
                    "audit"
                ]["rejection_counts_nonexclusive"],
                "fixture_rejection_counts_nonexclusive": dict(sorted(rejection.items())),
            },
            "source": {
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256(config_path),
                "base_shortlist_config": str(base_config_path.relative_to(ROOT)),
                "base_shortlist_config_sha256": sha256(base_config_path),
                "arm_grasp_pool": str(source_pool_path.relative_to(ROOT)),
                "arm_grasp_pool_sha256": sha256(source_pool_path),
            },
            "candidates": admitted,
        }
        output_path = output_root / fixture_id / "shortlist.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(output, sort_keys=False))
        print(
            f"{fixture_id}: {len(admitted)} valid; "
            + ", ".join(
                f"{distance:.2f}m={count_by_approach[f'{distance:.2f}']}"
                for distance in approaches
            )
            + f" -> {output_path}"
        )


if __name__ == "__main__":
    main()
