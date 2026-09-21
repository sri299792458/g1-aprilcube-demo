#!/usr/bin/env python3
"""Build object-centric Dex3 grasp families from Isaac validation traces.

This program is deliberately downstream of physics. It never changes a grasp
pose or decides whether a trial passed. Detailed solver contact points are kept
as diagnostics only. Family decisions use coarse digit-chain participation,
palm participation, and approach sector. Broad AprilCube surface neighborhoods
and persistence through the validator's tug phases remain annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/grasp_atlas/cube_v1.yaml"
AXES = ("X", "Y", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported atlas schema: {config.get('schema_version')}")
    return config


def region_id(object_id: str, marker: dict) -> str:
    voxel = marker["voxel"]
    return (
        f"{object_id}/v_{int(voxel[0])}_{int(voxel[1])}_{int(voxel[2])}/"
        f"{marker['face']}"
    )


def cuboid_markers(source: dict) -> list[dict]:
    """Adapt the released cuboid detector schema to voxel surface records."""
    dimensions = np.asarray(source["box_dims"], dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError(f"Invalid cuboid box_dims: {source.get('box_dims')}")
    half = dimensions / 2.0
    normals = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    corners = {
        "+X": [[half[0], -half[1], half[2]], [half[0], half[1], half[2]],
               [half[0], half[1], -half[2]], [half[0], -half[1], -half[2]]],
        "-X": [[-half[0], half[1], half[2]], [-half[0], -half[1], half[2]],
               [-half[0], -half[1], -half[2]], [-half[0], half[1], -half[2]]],
        "+Y": [[half[0], half[1], half[2]], [-half[0], half[1], half[2]],
               [-half[0], half[1], -half[2]], [half[0], half[1], -half[2]]],
        "-Y": [[-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
               [half[0], -half[1], -half[2]], [-half[0], -half[1], -half[2]]],
        "+Z": [[-half[0], -half[1], half[2]], [-half[0], half[1], half[2]],
               [half[0], half[1], half[2]], [half[0], -half[1], half[2]]],
        "-Z": [[-half[0], half[1], -half[2]], [-half[0], -half[1], -half[2]],
               [half[0], -half[1], -half[2]], [half[0], half[1], -half[2]]],
    }
    markers = []
    for face, ids in source["faces"].items():
        if face not in normals or len(ids) != 1:
            raise ValueError(f"Unsupported cuboid face assignment: {face}={ids}")
        markers.append(
            {
                "id": int(ids[0]),
                "face": face,
                "voxel": [0, 0, 0],
                "normal": normals[face].tolist(),
                "face_corners_mm": np.asarray(corners[face]).tolist(),
            }
        )
    return markers


def build_surface_regions(config: dict) -> dict:
    object_cfg = config["object"]
    source_path = project_path(object_cfg["aprilcube_config"])
    source = json.loads(source_path.read_text())
    regions = []
    seen = set()
    markers = source.get("markers")
    if markers is None:
        if source.get("target", {}).get("type") != "cuboid":
            raise ValueError(f"Unsupported AprilCube config schema: {source_path}")
        markers = cuboid_markers(source)
    for marker in markers:
        identity = region_id(object_cfg["id"], marker)
        if identity in seen:
            raise ValueError(f"Duplicate generated surface region: {identity}")
        seen.add(identity)
        corners = np.asarray(marker["face_corners_mm"], dtype=np.float64) / 1000.0
        if corners.shape != (4, 3):
            raise ValueError(f"Invalid corners for {identity}: {corners.shape}")
        edge_u = corners[1] - corners[0]
        edge_v = corners[3] - corners[0]
        length_u = float(np.linalg.norm(edge_u))
        length_v = float(np.linalg.norm(edge_v))
        if length_u <= 0 or length_v <= 0:
            raise ValueError(f"Degenerate region: {identity}")
        regions.append(
            {
                "region_id": identity,
                "object_id": object_cfg["id"],
                "voxel": [int(value) for value in marker["voxel"]],
                "face": marker["face"],
                "normal_object": [float(value) for value in marker["normal"]],
                "face_corners_object_m": corners.tolist(),
                "april_tag_id": int(marker["id"]),
                "role": "unassigned",
                "basis": {
                    "origin_object_m": corners[0].tolist(),
                    "u_axis_object": (edge_u / length_u).tolist(),
                    "v_axis_object": (edge_v / length_v).tolist(),
                    "u_length_m": length_u,
                    "v_length_m": length_v,
                },
            }
        )
    return {
        "schema_version": 1,
        "object_id": object_cfg["id"],
        "source": str(source_path.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(source_path),
        "fillet_radius_m": float(object_cfg["fillet_radius_m"]),
        "regions": sorted(regions, key=lambda item: item["region_id"]),
    }


def closest_point_on_region(point: np.ndarray, region: dict) -> dict:
    basis = region["basis"]
    origin = np.asarray(basis["origin_object_m"], dtype=np.float64)
    u_axis = np.asarray(basis["u_axis_object"], dtype=np.float64)
    v_axis = np.asarray(basis["v_axis_object"], dtype=np.float64)
    u_length = float(basis["u_length_m"])
    v_length = float(basis["v_length_m"])
    relative = point - origin
    u_m = float(np.dot(relative, u_axis))
    v_m = float(np.dot(relative, v_axis))
    u = u_m / u_length
    v = v_m / v_length
    u_clamped = float(np.clip(u, 0.0, 1.0))
    v_clamped = float(np.clip(v, 0.0, 1.0))
    closest = origin + u_clamped * u_length * u_axis + v_clamped * v_length * v_axis
    return {
        "region_id": region["region_id"],
        "distance_m": float(np.linalg.norm(point - closest)),
        "uv_unclamped": [u, v],
        "uv_clamped": [u_clamped, v_clamped],
        "closest_region_point_object_m": closest.tolist(),
    }


def map_contact_point(
    point_object_m: Iterable[float],
    *,
    regions: list[dict],
    mesh: trimesh.Trimesh,
    mesh_tolerance_m: float,
    fillet_radius_m: float,
    floating_point_tolerance_m: float,
) -> dict:
    point = np.asarray(point_object_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"Invalid contact point: {point_object_m}")
    closest_mesh, mesh_distance, triangle_id = trimesh.proximity.closest_point(
        mesh, point.reshape(1, 3)
    )
    mesh_distance_value = float(mesh_distance[0])
    mapping_valid = mesh_distance_value <= mesh_tolerance_m
    candidates = [closest_point_on_region(point, region) for region in regions]
    candidates.sort(key=lambda item: (item["distance_m"], item["region_id"]))
    primary = candidates[0]
    nearby_limit = (
        primary["distance_m"] + fillet_radius_m + floating_point_tolerance_m
    )
    nearby = [
        candidate["region_id"]
        for candidate in candidates
        if candidate["distance_m"] <= nearby_limit
    ]
    return {
        "primary_region": primary["region_id"],
        "distance_to_primary_region_m": primary["distance_m"],
        "nearby_regions_within_fillet_radius": nearby,
        "uv_unclamped": primary["uv_unclamped"],
        "uv_clamped": primary["uv_clamped"],
        "closest_region_point_object_m": primary[
            "closest_region_point_object_m"
        ],
        "mesh_distance_m": mesh_distance_value,
        "closest_mesh_point_object_m": closest_mesh[0].tolist(),
        "closest_mesh_triangle": int(triangle_id[0]),
        "mapping_valid": mapping_valid,
    }


def canonical_link_name(link: str, hand_side: str) -> str:
    prefix = f"{hand_side}_hand_"
    if not link.startswith(prefix) or not link.endswith("_link"):
        raise ValueError(f"Unexpected {hand_side} Dex3 link name: {link}")
    return link[len(prefix) : -len("_link")]


def digit_for_link(link: str) -> str | None:
    for digit in ("thumb", "index", "middle"):
        if link.startswith(digit + "_"):
            return digit
    return None


def approach_vector_and_sector(object_T_G: list[list[float]]) -> tuple[list[float], str]:
    transform = np.asarray(object_T_G, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected object_T_G 4x4, received {transform.shape}")
    vector = transform[:3, :3] @ np.array([0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Degenerate approach vector")
    vector /= norm
    axis_index = int(np.argmax(np.abs(vector)))
    sign = "+" if vector[axis_index] >= 0 else "-"
    return vector.tolist(), f"{sign}{AXES[axis_index]}"


def closed_phase(trial: dict) -> dict:
    matches = [
        phase for phase in trial["phases"] if phase["name"] == "closed_before_tug"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{trial['candidate_id']} has {len(matches)} closed_before_tug phases"
        )
    return matches[0]


def qualified_phase(trial: dict) -> dict:
    """Return the post-tug state whose contacts determine physics PASS."""
    matches = [
        phase for phase in trial["phases"] if phase["name"] == "after_tug_5_final"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{trial['candidate_id']} has {len(matches)} after_tug_5_final phases"
        )
    return matches[0]


def coarse_phase_contacts(phase: dict, hand_side: str) -> dict[str, set[str]]:
    """Return chain participation and broad face neighborhoods.

    The aggregate object-filtered normal force decides whether a chain is
    present. Detailed contact points only annotate that participating chain
    with broad cube face/edge neighborhoods; their exact coordinates are not a
    decision signal.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for contact in phase["contacts"]:
        link = canonical_link_name(contact["hand_link"], hand_side)
        group = "palm" if link == "palm" else digit_for_link(link)
        if group is None:
            continue
        if "contact_force_magnitude_N" in contact:
            force_magnitude = float(contact["contact_force_magnitude_N"])
        else:
            net_force = np.asarray(
                contact["net_normal_force_world_N"], dtype=np.float64
            )
            force_magnitude = float(np.linalg.norm(net_force))
        if force_magnitude <= 0.0:
            continue
        # Preserve participation even when detailed PhysX point buffers are
        # empty or fail the mesh-consistency diagnostic.
        groups[group]
        for point in contact["points"]:
            surface = point["surface"]
            if not surface["mapping_valid"]:
                continue
            groups[group].update(surface["nearby_regions_within_fillet_radius"])
    return groups


def build_signature(trial: dict) -> dict:
    hand_side = trial["hand_side"]
    # The family describes the contact state that actually survived all five
    # disturbances and satisfied the simulator's PASS test. Closure remains
    # available through contact_persistence, but it is not guaranteed to have
    # the same participating chains as the qualified final state.
    phase = qualified_phase(trial)
    grouped = coarse_phase_contacts(phase, hand_side)
    ordered_contacts = {
        group: sorted(regions) for group, regions in sorted(grouped.items())
    }
    digits = sorted(group for group in grouped if group != "palm")
    vector, sector = approach_vector_and_sector(trial["input"]["object_T_G"])
    signature = {
        "digit_chains": digits,
        "palm_contact": "palm" in grouped,
        "approach_sector": sector,
    }
    # Detailed PhysX points are useful visual diagnostics, but their exact
    # face/edge assignment is not reliable enough to split contact families.
    # The family decision uses only body-level chain participation, palm
    # participation, and the candidate's geometric approach sector.
    trial["diagnostic_broad_faces_by_chain"] = ordered_contacts
    trial["approach_vector_object"] = vector
    return signature


def annotate_persistence(trial: dict) -> dict[str, str]:
    initial_groups = sorted(
        coarse_phase_contacts(closed_phase(trial), trial["hand_side"])
    )
    phase_groups = [
        coarse_phase_contacts(phase, trial["hand_side"])
        for phase in trial["phases"]
    ]
    return {
        group: "".join("1" if group in groups else "0" for groups in phase_groups)
        for group in initial_groups
    }


def map_trial_contacts(
    trial: dict,
    *,
    regions: list[dict],
    mesh: trimesh.Trimesh,
    mapping_config: dict,
    fillet_radius_m: float,
) -> dict:
    for phase in trial["phases"]:
        for contact in phase["contacts"]:
            for point in contact["points"]:
                point["surface"] = map_contact_point(
                    point["position_object_m"],
                    regions=regions,
                    mesh=mesh,
                    mesh_tolerance_m=float(mapping_config["mesh_tolerance_m"]),
                    fillet_radius_m=fillet_radius_m,
                    floating_point_tolerance_m=float(
                        mapping_config["floating_point_tolerance_m"]
                    ),
                )
    trial["contact_persistence"] = annotate_persistence(trial)
    if trial["result"]["passed"]:
        signature = build_signature(trial)
        if len(signature["digit_chains"]) < 2:
            raise ValueError(
                f"Passing trial {trial['candidate_id']} has fewer than two digit "
                "chains in its qualified final phase"
            )
        trial["contact_signature"] = signature
        trial["family_key_sha256"] = stable_hash(signature)
    else:
        trial["contact_signature"] = None
        trial["family_key_sha256"] = None
    return trial


def persistence_score(trial: dict) -> int:
    return sum(mask.count("1") for mask in trial["contact_persistence"].values())


def better_tie_break(trial: dict) -> tuple:
    return (
        -persistence_score(trial),
        -float(trial["input"]["graspgenx_score"]),
        trial["candidate_id"],
    )


def rotation_distance(a: list[list[float]], b: list[list[float]]) -> float:
    rotation_a = np.asarray(a, dtype=np.float64)[:3, :3]
    rotation_b = np.asarray(b, dtype=np.float64)[:3, :3]
    cosine = (float(np.trace(rotation_a.T @ rotation_b)) - 1.0) / 2.0
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def choose_representatives(trials: list[dict], voxel_size_m: float) -> list[dict]:
    if not trials:
        return []
    primary_index = min(
        range(len(trials)), key=lambda index: better_tie_break(trials[index])
    )
    selections = [("primary", primary_index)]
    if len(trials) >= 2:
        primary_translation = np.asarray(
            trials[primary_index]["input"]["object_T_G"], dtype=np.float64
        )[:3, 3]
        distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(trial["input"]["object_T_G"], dtype=np.float64)[:3, 3]
                    - primary_translation
                )
                / voxel_size_m
                for trial in trials
            ]
        )
        maximum = float(distances.max())
        candidates = [
            index
            for index in np.flatnonzero(
                np.isclose(distances, maximum, rtol=0, atol=1e-12)
            ).tolist()
            if index != primary_index
        ]
        contact_index = min(candidates, key=lambda index: better_tie_break(trials[index]))
        selections.append(("translation_diverse_backup", contact_index))
    if len(trials) >= 3:
        selected_indices = {index for _, index in selections}
        # The first representative is selected from coarse contact persistence
        # and neural score. Rotation is used only to retain a geometrically
        # different concrete backup inside the same coarse family.
        rotations = [
            rotation_distance(
                trials[primary_index]["input"]["object_T_G"],
                trial["input"]["object_T_G"],
            )
            for trial in trials
        ]
        maximum = max(
            value for index, value in enumerate(rotations) if index not in selected_indices
        )
        candidates = [
            index
            for index, value in enumerate(rotations)
            if index not in selected_indices
            and math.isclose(value, maximum, rel_tol=0, abs_tol=1e-12)
        ]
        pose_index = min(candidates, key=lambda index: better_tie_break(trials[index]))
        selections.append(("pose_diverse_backup", pose_index))
    return [
        {
            "role": role,
            "candidate_id": trials[index]["candidate_id"],
            "candidate_content_sha256": trials[index].get(
                "candidate_content_sha256"
            ),
            "graspgenx_score": float(trials[index]["input"]["graspgenx_score"]),
            "object_T_G": trials[index]["input"]["object_T_G"],
            "final_q": trials[index]["result"]["final_q"],
            "contact_persistence": trials[index]["contact_persistence"],
            "diagnostic_broad_faces_by_chain": trials[index].get(
                "diagnostic_broad_faces_by_chain", {}
            ),
        }
        for role, index in selections
    ]


def build_families(trials: list[dict], voxel_size_m: float, hand_side: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    signatures = {}
    for trial in trials:
        if not trial["result"]["passed"]:
            continue
        key = trial["family_key_sha256"]
        grouped[key].append(trial)
        signatures[key] = trial["contact_signature"]
    families = []
    representative_entries = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda trial: trial["candidate_id"])
        family_id = f"{hand_side}_{key[:12]}"
        representatives = choose_representatives(members, voxel_size_m)
        for representative in representatives:
            representative_entries.append(
                {
                    "family_id": family_id,
                    "family_size": len(members),
                    **representative,
                }
            )
        families.append(
            {
                "family_id": family_id,
                "family_key_sha256": key,
                "hand_side": hand_side,
                "signature": signatures[key],
                "diagnostic_broad_faces_by_chain": {
                    group: sorted(
                        {
                            region
                            for trial in members
                            for region in trial.get(
                                "diagnostic_broad_faces_by_chain", {}
                            ).get(group, [])
                        }
                    )
                    for group in sorted(
                        {
                            group
                            for trial in members
                            for group in trial.get(
                                "diagnostic_broad_faces_by_chain", {}
                            )
                        }
                    )
                },
                "member_count": len(members),
                "member_ids": [trial["candidate_id"] for trial in members],
                "representatives": representatives,
            }
        )
    return {
        "schema_version": 1,
        "hand_side": hand_side,
        "trial_count": len(trials),
        "pass_count": sum(bool(trial["result"]["passed"]) for trial in trials),
        "fail_count": sum(not bool(trial["result"]["passed"]) for trial in trials),
        "family_count": len(families),
        "families": families,
        "representatives": representative_entries,
    }


def load_jsonl(paths: list[Path]) -> list[dict]:
    trials = []
    seen = set()
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                trial = json.loads(line)
                identity = (trial.get("hand_side"), trial.get("candidate_id"))
                if identity in seen:
                    raise ValueError(f"Duplicate trial {identity} in {path}:{line_number}")
                seen.add(identity)
                trials.append(trial)
    return trials


def write_jsonl(path: Path, records: list[dict]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )


def write_representatives(path: Path, summary: dict) -> None:
    grasps = {}
    for representative in summary["representatives"]:
        candidate_id = representative["candidate_id"]
        transform = np.asarray(representative["object_T_G"], dtype=np.float64)
        quaternion = trimesh.transformations.quaternion_from_matrix(transform)
        grasps[candidate_id] = {
            "family_id": representative["family_id"],
            "family_size": representative["family_size"],
            "representative_role": representative["role"],
            "candidate_content_sha256": representative[
                "candidate_content_sha256"
            ],
            "graspgenx_score": representative["graspgenx_score"],
            "position": transform[:3, 3].tolist(),
            "orientation": {
                "w": float(quaternion[0]),
                "xyz": quaternion[1:].tolist(),
            },
            "final_q": representative["final_q"],
            "contact_persistence": representative["contact_persistence"],
            "diagnostic_broad_faces_by_chain": representative[
                "diagnostic_broad_faces_by_chain"
            ],
        }
    output = {
        "format": "g1_aprilcube_grasp_atlas_representatives",
        "format_version": 1,
        "hand_side": summary["hand_side"],
        "grasps": grasps,
    }
    atomic_write_text(path, yaml.safe_dump(output, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build a smoke atlas from the currently available trace shards.",
    )
    parser.add_argument(
        "--hand-side",
        choices=("right", "left"),
        help=(
            "Build only one hand's atlas. This supports a one-sided visual "
            "gate without requiring the other hand's physics run."
        ),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifacts_root = project_path(config["artifacts_root"])
    surface = build_surface_regions(config)
    surface_path = artifacts_root / "surface_regions.json"
    atomic_write_text(surface_path, json.dumps(surface, indent=2) + "\n")
    mesh_path = project_path(config["object"]["mesh"])
    # Merge coincident vertices split only for OBJ texture/normal seams before
    # checking the physical surface topology.
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise ValueError(f"Expected one watertight object mesh: {mesh_path}")

    summaries = {}
    expected_count = len(config["generation"]["seeds"]) * int(
        config["generation"]["batch_size"]
    )
    hand_sides = [args.hand_side] if args.hand_side else config["physics"]["hands"]
    for hand_side in hand_sides:
        output_directory = config["physics"].get(
            "qualification_output_directory", "physics_outputs"
        )
        trace_paths = sorted(
            (artifacts_root / hand_side / output_directory).glob("*.jsonl")
        )
        if not trace_paths:
            raise FileNotFoundError(
                f"No {hand_side} contact traces under "
                f"{artifacts_root / hand_side / output_directory}"
            )
        trials = load_jsonl(trace_paths)
        if not args.allow_incomplete and len(trials) != expected_count:
            raise ValueError(
                f"{hand_side} has {len(trials)} trials; expected {expected_count}. "
                "Use --allow-incomplete only for the 256-candidate visual gate."
            )
        mapped = [
            map_trial_contacts(
                trial,
                regions=surface["regions"],
                mesh=mesh,
                mapping_config=config["surface_mapping"],
                fillet_radius_m=float(config["object"]["fillet_radius_m"]),
            )
            for trial in trials
        ]
        mapped.sort(key=lambda trial: trial["candidate_id"])
        output_dir = artifacts_root / hand_side
        trials_path = output_dir / "contact_trials.jsonl"
        write_jsonl(trials_path, mapped)
        summary = build_families(
            mapped,
            voxel_size_m=float(config["object"]["voxel_size_m"]),
            hand_side=hand_side,
        )
        families_path = output_dir / "families.json"
        atomic_write_text(families_path, json.dumps(summary, indent=2) + "\n")
        representatives_path = output_dir / "representatives.yaml"
        write_representatives(representatives_path, summary)
        summaries[hand_side] = {
            "trace_files": [str(path.relative_to(PROJECT_ROOT)) for path in trace_paths],
            "trace_file_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in trace_paths},
            "trials": len(mapped),
            "passes": summary["pass_count"],
            "families": summary["family_count"],
            "contact_trials": str(trials_path.relative_to(PROJECT_ROOT)),
            "contact_trials_sha256": sha256(trials_path),
            "families_file": str(families_path.relative_to(PROJECT_ROOT)),
            "families_sha256": sha256(families_path),
            "representatives": str(representatives_path.relative_to(PROJECT_ROOT)),
            "representatives_sha256": sha256(representatives_path),
        }
        print(
            f"{hand_side}: {len(mapped)} trials, {summary['pass_count']} passes, "
            f"{summary['family_count']} contact families"
        )

    complete = (
        set(summaries) == set(config["physics"]["hands"])
        and all(summary["trials"] == expected_count for summary in summaries.values())
    )
    manifest = {
        "schema_version": 1,
        "atlas_id": config["atlas_id"],
        "status": "complete" if complete else "smoke_incomplete",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(config_path),
        "revisions": {
            "project": git_revision(PROJECT_ROOT),
            "graspgenx": git_revision(PROJECT_ROOT / "third_party/GraspGenX"),
            "graspdatagen": git_revision(PROJECT_ROOT / "third_party/GraspDataGen"),
        },
        "object_mesh": str(mesh_path.relative_to(PROJECT_ROOT)),
        "object_mesh_sha256": sha256(mesh_path),
        "surface_regions": str(surface_path.relative_to(PROJECT_ROOT)),
        "surface_regions_sha256": sha256(surface_path),
        "expected_trials_per_hand": expected_count,
        "hands": summaries,
    }
    manifest_path = artifacts_root / "manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path} ({manifest['status']})")


if __name__ == "__main__":
    main()
