#!/usr/bin/env python3
"""Numerically audit the GraspGenX-to-Dex3 frame contract.

This script deliberately does not render anything and does not claim that a
neural grasp is physically valid.  It answers the narrower questions which can
be proved exactly:

* what transform the generated descriptor uses for ``G_T_P``;
* whether that descriptor is the pinned Unitree hand expressed in ``G``;
* whether the canonical-origin translation is reproducibly derived;
* whether raw GraspGenX matrices survive centering and YAML serialization; and
* whether ``O_T_P = O_T_G @ G_T_P`` is internally self-consistent.

Notation:
    O  object mesh / point-cloud frame
    C  centered point-cloud frame
    G  GraspGenX canonical grasp frame
    P  Unitree Dex3 palm frame
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml
import yourdfpy
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config/dex3_rev1_descriptor.json"
DEFAULT_SOURCE = PROJECT_ROOT / ".cache/unitree_xr_teleoperate"
DEFAULT_DESCRIPTORS = PROJECT_ROOT / "third_party/GraspGenX/assets/x_grippers"
DEFAULT_CANDIDATES = PROJECT_ROOT / "artifacts/aprilcube_raw_grasps/cube_head.yaml"
DEFAULT_CUBE = PROJECT_ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/dex3_validation/frame_audit.json"

DISTAL_LINKS = {
    "right": (
        "right_hand_thumb_2_link.STL",
        "right_hand_middle_1_link.STL",
        "right_hand_index_1_link.STL",
    ),
    "left": (
        "left_hand_thumb_2_link.STL",
        "left_hand_middle_1_link.STL",
        "left_hand_index_1_link.STL",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rigid_errors(matrix: np.ndarray) -> dict[str, float]:
    rotation = matrix[:3, :3]
    return {
        "bottom_row_max_error": float(
            np.max(np.abs(matrix[3] - np.array([0.0, 0.0, 0.0, 1.0])))
        ),
        "orthonormal_max_error": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        ),
        "determinant_error": float(abs(np.linalg.det(rotation) - 1.0)),
    }


def fixed_parent_T_child(urdf_path: Path, child_link: str) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or child.attrib.get("link") != child_link:
            continue
        if joint.attrib.get("type") != "fixed":
            raise RuntimeError(f"Parent of {child_link} is not fixed in {urdf_path}")
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        result = tra.translation_matrix(xyz)
        result[:3, :3] = tra.euler_matrix(*rpy, axes="sxyz")[:3, :3]
        return result
    raise RuntimeError(f"No fixed parent joint found for {child_link} in {urdf_path}")


def load_robot(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(path),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )


def scene_vertices(
    robot: yourdfpy.URDF,
    joints: dict[str, float],
    allowed_links: set[str] | None = None,
    frame_from: str | None = None,
) -> np.ndarray:
    robot.update_cfg(joints)
    vertices = []
    for geometry_name, geometry in robot.scene.geometry.items():
        parent_link = robot.scene.graph.transforms.parents[geometry_name]
        if allowed_links is not None and parent_link not in allowed_links:
            continue
        base_T_geometry = robot.scene.graph.get(
            frame_from=frame_from or robot.scene.graph.base_frame,
            frame_to=geometry_name,
        )[0]
        vertices.append(tra.transform_points(geometry.vertices, base_T_geometry))
    if not vertices:
        raise RuntimeError("URDF has no visual geometry")
    return np.concatenate(vertices)


def geometry_centroid(
    robot: yourdfpy.URDF,
    joints: dict[str, float],
    geometry_name: str,
    frame_from: str | None = None,
) -> np.ndarray:
    robot.update_cfg(joints)
    geometry = robot.scene.geometry[geometry_name]
    base_T_geometry = robot.scene.graph.get(
        frame_from=frame_from or robot.scene.graph.base_frame,
        frame_to=geometry_name,
    )[0]
    return tra.transform_points(geometry.vertices, base_T_geometry).mean(axis=0)


def symmetric_nearest_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if len(a) != len(b):
        raise RuntimeError(f"Vertex-count mismatch: {len(a)} != {len(b)}")
    a_to_b = cKDTree(b).query(a, workers=-1)[0]
    b_to_a = cKDTree(a).query(b, workers=-1)[0]
    return {
        "max_m": float(max(a_to_b.max(), b_to_a.max())),
        "rms_m": float(
            np.sqrt((np.square(a_to_b).sum() + np.square(b_to_a).sum()) / (len(a) + len(b)))
        ),
    }


def load_isaac_grasps(path: Path) -> list[tuple[str, float, np.ndarray]]:
    data = yaml.safe_load(path.read_text())
    result = []
    for name, entry in data["grasps"].items():
        q = entry["orientation"]
        matrix = tra.quaternion_matrix([q["w"], *q["xyz"]])
        matrix[:3, 3] = entry["position"]
        result.append((name, float(entry["confidence"]), matrix))
    return result


def quaternion_roundtrip_error(matrix: np.ndarray) -> float:
    q = tra.quaternion_from_matrix(matrix)
    reconstructed = tra.quaternion_matrix(q)
    reconstructed[:3, 3] = matrix[:3, 3]
    return float(np.max(np.abs(reconstructed - matrix)))


def check(ok: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(ok), **evidence}


def audit_side(
    side: str,
    manifest: dict,
    source_root: Path,
    descriptor_root: Path,
    tolerance: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    source_urdf = source_root / f"assets/unitree_hand/unitree_dex3_{side}.urdf"
    descriptor_dir = descriptor_root / manifest["descriptor_names"][side]
    descriptor_urdf = descriptor_dir / "gripper.urdf"
    palm = f"{side}_hand_palm_link"
    G_T_P = fixed_parent_T_child(descriptor_urdf, palm)

    configured = manifest["canonical_frame"][side]
    configured_rotation = np.asarray(configured["rotation"], dtype=float)
    fixed_joint_error = float(
        np.max(np.abs(G_T_P[:3, :3] - configured_rotation))
    )

    profile = manifest["finger_profile"][side]
    source_robot = load_robot(source_urdf)
    descriptor_robot = load_robot(descriptor_urdf)
    selected_links = {
        palm,
        *(
            joint.child
            for joint in descriptor_robot.robot.joints
            if joint.type != "fixed"
        ),
    }

    overlay = {}
    for state in ("open", "close"):
        source_P = scene_vertices(
            source_robot, profile[state], selected_links, frame_from=palm
        )
        descriptor_G = scene_vertices(descriptor_robot, profile[state], selected_links)
        descriptor_P = tra.transform_points(descriptor_G, np.linalg.inv(G_T_P))
        overlay[state] = symmetric_nearest_error(source_P, descriptor_P)

    # Reproduce the origin shift from the unmodified Unitree palm-rooted hand:
    # rotate the three entire distal-link meshes into canonical axes, then put
    # the midpoint between thumb and the two opposing-link centroids at G.x=0.
    R_G0_P = G_T_P.copy()
    R_G0_P[:3, 3] = 0.0
    closed_centroids_P = [
        geometry_centroid(source_robot, profile["close"], name, frame_from=palm)
        for name in DISTAL_LINKS[side]
    ]
    closed_centroids_G0 = [
        tra.transform_points(np.asarray([point]), R_G0_P)[0]
        for point in closed_centroids_P
    ]
    thumb_x = float(closed_centroids_G0[0][0])
    opponent_x = float(np.mean([closed_centroids_G0[1][0], closed_centroids_G0[2][0]]))
    unshifted_midpoint_x = 0.5 * (thumb_x + opponent_x)
    derived_translation_x = -unshifted_midpoint_x
    origin_error = abs(float(G_T_P[0, 3]) - derived_translation_x)

    result = {
        "source_urdf": str(source_urdf.relative_to(PROJECT_ROOT)),
        "source_urdf_sha256": sha256(source_urdf),
        "descriptor_urdf": str(descriptor_urdf.relative_to(PROJECT_ROOT)),
        "descriptor_urdf_sha256": sha256(descriptor_urdf),
        "G_T_P": G_T_P.tolist(),
        "G_T_P_rigid": check(max(rigid_errors(G_T_P).values()) <= tolerance, **rigid_errors(G_T_P)),
        "fixed_joint_rotation_matches_manifest": check(
            fixed_joint_error <= tolerance,
            max_rotation_matrix_error=fixed_joint_error,
        ),
        "source_descriptor_geometry_overlay": {
            state: check(values["max_m"] <= tolerance, **values)
            for state, values in overlay.items()
        },
        "canonical_origin_derivation": check(
            origin_error <= tolerance,
            method="whole-mesh centroids of closed distal thumb, index, and middle links",
            unshifted_thumb_centroid_x_m=thumb_x,
            unshifted_opponents_mean_centroid_x_m=opponent_x,
            unshifted_pinch_midpoint_x_m=unshifted_midpoint_x,
            derived_translation_x_m=derived_translation_x,
            descriptor_translation_x_m=float(G_T_P[0, 3]),
            absolute_error_m=origin_error,
        ),
    }
    return result, G_T_P, scene_vertices(
        descriptor_robot, profile["open"], selected_links
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--descriptor-root", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=2.0e-8)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "static_frame_audit",
        "notation": {
            "O_T_G": "GraspGenX canonical frame G expressed in object frame O",
            "G_T_P": "Unitree palm frame P expressed in GraspGenX frame G",
            "O_T_P": "physical palm target, computed as O_T_G @ G_T_P",
        },
        "manifest": str(args.manifest.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256(args.manifest),
        "sides": {},
    }

    side_data = {}
    for side in ("right", "left"):
        side_result, G_T_P, open_vertices = audit_side(
            side,
            manifest,
            args.source_root,
            args.descriptor_root,
            args.tolerance,
        )
        result["sides"][side] = side_result
        side_data[side] = (G_T_P, open_vertices)

    canonical_overlay = symmetric_nearest_error(
        side_data["right"][1], side_data["left"][1]
    )
    result["left_right_open_geometry_overlay"] = check(
        canonical_overlay["max_m"] <= 5.0e-5,
        **canonical_overlay,
    )

    cube = trimesh.load(args.cube, force="mesh", process=False)
    points = np.asarray(cube.vertices, dtype=float)
    C_T_O = tra.translation_matrix(-points.mean(axis=0))
    O_T_C = np.linalg.inv(C_T_O)
    candidates = load_isaac_grasps(args.candidates)
    if not candidates:
        raise RuntimeError(f"No candidates in {args.candidates}")

    candidate_errors = []
    physical_chain_errors = {"right": [], "left": []}
    object_origins_in_G = []
    for _name, _confidence, O_T_G in candidates:
        errors = rigid_errors(O_T_G)
        errors["yaml_quaternion_roundtrip"] = quaternion_roundtrip_error(O_T_G)
        C_T_G = C_T_O @ O_T_G
        restored_O_T_G = O_T_C @ C_T_G
        errors["center_uncenter_max_error"] = float(
            np.max(np.abs(restored_O_T_G - O_T_G))
        )
        candidate_errors.append(errors)
        object_origins_in_G.append(np.linalg.inv(O_T_G)[:3, 3])
        for side in ("right", "left"):
            G_T_P = side_data[side][0]
            O_T_P = O_T_G @ G_T_P
            P_T_O = np.linalg.inv(O_T_P)
            physical_chain_errors[side].append(
                float(np.max(np.abs(O_T_P @ P_T_O - np.eye(4))))
            )

    maxima = {
        key: max(entry[key] for entry in candidate_errors)
        for key in candidate_errors[0]
    }
    result["candidate_contract"] = {
        "path": str(args.candidates.relative_to(PROJECT_ROOT)),
        "sha256": sha256(args.candidates),
        "count": len(candidates),
        "matrix_and_serialization": check(
            max(maxima.values()) <= args.tolerance,
            maximum_errors=maxima,
        ),
        "physical_palm_chain_roundtrip": {
            side: check(
                max(errors) <= args.tolerance,
                maximum_identity_error=max(errors),
            )
            for side, errors in physical_chain_errors.items()
        },
        "object_origin_in_G_m": {
            "minimum": np.min(object_origins_in_G, axis=0).tolist(),
            "mean": np.mean(object_origins_in_G, axis=0).tolist(),
            "maximum": np.max(object_origins_in_G, axis=0).tolist(),
        },
    }

    def all_passed(value: Any) -> bool:
        if isinstance(value, dict):
            if "passed" in value and not value["passed"]:
                return False
            return all(all_passed(item) for item in value.values())
        if isinstance(value, list):
            return all(all_passed(item) for item in value)
        return True

    result["static_audit_passed"] = all_passed(result)
    result["claim_boundary"] = (
        "A pass proves coordinate and descriptor consistency only. It does not "
        "prove that any GraspGenX candidate makes stable physical contact."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "static_audit_passed": result["static_audit_passed"],
        "output": str(args.output),
    }, indent=2))
    if not result["static_audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
