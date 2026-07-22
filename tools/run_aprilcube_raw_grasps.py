"""Run the released GraspGenX model on the clean AprilCube meshes.

The default current-Dex3 path verifies that the left and right descriptors have
identical canonical geometry and sweep-volume conditioning.  A named descriptor
can also be selected for controlled cross-hand baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = PROJECT_ROOT / "third_party/GraspGenX"
sys.path.insert(0, str(GRASPGENX_ROOT))

from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format  # noqa: E402
from graspgenx.grasp_server import GraspGenXSampler  # noqa: E402
from graspgenx.utils.checkpoint_io import load_model_cfg  # noqa: E402


DEFAULT_ASSETS = GRASPGENX_ROOT / "assets"
DEFAULT_PARTS = PROJECT_ROOT / "generated/aprilcube_parts"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/aprilcube_raw_grasps"
PARTS = ("t_body", "u_legs", "cube_head")
CHECKPOINT_COMMIT = "7c834043c11a11417e31d6d5ea9355801e40a2c1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a complete text artifact without exposing a partial shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def resolved_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def descriptor_config(assets: Path, side: str) -> tuple[Path, dict]:
    path = assets / "x_grippers" / f"dex3_rev1_{side}" / "config.json"
    return path, json.loads(path.read_text())


def descriptor_audit(assets: Path, gripper_name: str) -> dict:
    selected_path = assets / "x_grippers" / gripper_name / "config.json"
    if not selected_path.is_file():
        raise FileNotFoundError(f"Missing GraspGenX descriptor: {selected_path}")

    if gripper_name != "dex3_rev1_right":
        return {
            "gripper_name": gripper_name,
            "config": str(selected_path),
            "config_sha256": sha256(selected_path),
        }

    right_path, right = descriptor_config(assets, "right")
    left_path, left = descriptor_config(assets, "left")
    fields = ("sweep_volume", "fingertip", "standoff", "symmetric", "type")
    mismatches = [field for field in fields if right[field] != left[field]]
    if mismatches:
        raise RuntimeError(f"Left/right canonical descriptors differ: {mismatches}")
    required_status = "physics_validated_release_checkpoint_proxy"
    if right.get("review_status") != required_status:
        raise RuntimeError(
            "Dex3 descriptor has not passed the named physics-validation gate: "
            f"expected {required_status!r}, received {right.get('review_status')!r}"
        )
    return {
        "gripper_name": gripper_name,
        "config": str(selected_path),
        "config_sha256": sha256(selected_path),
        "right_config_sha256": sha256(right_path),
        "left_config_sha256": sha256(left_path),
        "identical_conditioning_fields": list(fields),
        "canonical_bbox_max_error_m": float(
            np.max(np.abs(np.asarray(right["bbox"]) - np.asarray(left["bbox"])))
        ),
        "canonical_candidate_compatible_hands": ["right", "left"],
    }


def sample_centered(mesh_path: Path, count: int, seed: int):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise RuntimeError(f"Expected one watertight physical mesh: {mesh_path}")
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, count)
    center_transform = tra.translation_matrix(-points.mean(axis=0))
    return tra.transform_points(points, center_transform), center_transform, mesh


def candidate_content_hash(
    *,
    object_mesh_sha256: str,
    descriptor_sha256: str,
    generator_checkpoint_sha256: str,
    discriminator_checkpoint_sha256: str,
    point_cloud_sha256: str,
    generation_seed: int,
    object_T_G: np.ndarray,
) -> str:
    payload = {
        "object_mesh_sha256": object_mesh_sha256,
        "descriptor_sha256": descriptor_sha256,
        "generator_checkpoint_sha256": generator_checkpoint_sha256,
        "discriminator_checkpoint_sha256": discriminator_checkpoint_sha256,
        "point_cloud_sha256": point_cloud_sha256,
        "generation_seed": int(generation_seed),
        "object_T_G_float64": np.asarray(object_T_G, dtype=np.float64).tolist(),
    }
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def run_atlas(args: argparse.Namespace) -> None:
    config_path = args.atlas_config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported atlas config schema: {config.get('schema_version')}")

    object_cfg = config["object"]
    generation_cfg = config["generation"]
    part = str(object_cfg["id"])
    if part not in PARTS:
        raise ValueError(f"Unsupported configured AprilCube part: {part}")
    checkpoint_dir = resolved_project_path(generation_cfg["checkpoint_dir"])
    assets_dir = args.assets_dir.resolve()
    descriptor_name = str(generation_cfg["descriptor"])
    descriptor = descriptor_audit(assets_dir, descriptor_name)
    descriptor_path = Path(descriptor["config"])
    mesh_path = resolved_project_path(object_cfg["mesh"])
    output_root = resolved_project_path(config["artifacts_root"]) / "raw"
    output_root.mkdir(parents=True, exist_ok=True)

    generator_checkpoint = checkpoint_dir / "gen/epoch_736.pth"
    discriminator_checkpoint = checkpoint_dir / "dis/epoch_1056.pth"
    for required in (generator_checkpoint, discriminator_checkpoint, mesh_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    point_count = int(generation_cfg["point_count"])
    point_seed = int(generation_cfg["point_seed"])
    centered_points, center_transform, mesh = sample_centered(
        mesh_path, point_count, point_seed
    )
    point_cloud_sha = sha256_bytes(
        np.ascontiguousarray(centered_points, dtype=np.float64).tobytes()
    )
    mesh_sha = sha256(mesh_path)
    descriptor_sha = sha256(descriptor_path)
    generator_sha = sha256(generator_checkpoint)
    discriminator_sha = sha256(discriminator_checkpoint)

    model_cfg = load_model_cfg(
        str(checkpoint_dir / "gen"),
        str(checkpoint_dir / "dis"),
    )
    sampler = GraspGenXSampler(
        model_cfg,
        descriptor_name,
        assets_dir=str(assets_dir),
    )

    seeds = [int(value) for value in generation_cfg["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Atlas generation seeds must be unique")
    if args.batch_index is not None:
        if not 0 <= args.batch_index < len(seeds):
            raise IndexError(f"--batch-index outside [0, {len(seeds) - 1}]")
        selected_batches = [(args.batch_index, seeds[args.batch_index])]
    else:
        selected_batches = list(enumerate(seeds))
        if args.max_batches:
            selected_batches = selected_batches[: args.max_batches]

    batch_size = int(generation_cfg["batch_size"])
    generation_manifest = {
        "schema_version": 1,
        "atlas_id": config["atlas_id"],
        "status": "raw_diffusion_candidates",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(config_path),
        "object_id": part,
        "object_mesh": str(mesh_path.relative_to(PROJECT_ROOT)),
        "object_mesh_sha256": mesh_sha,
        "mesh_extents_m": mesh.extents.tolist(),
        "descriptor": descriptor,
        "descriptor_sha256": descriptor_sha,
        "generator_checkpoint": str(generator_checkpoint.relative_to(PROJECT_ROOT)),
        "generator_checkpoint_sha256": generator_sha,
        "discriminator_checkpoint": str(discriminator_checkpoint.relative_to(PROJECT_ROOT)),
        "discriminator_checkpoint_sha256": discriminator_sha,
        "point_count": point_count,
        "point_seed": point_seed,
        "point_cloud_sha256": point_cloud_sha,
        "batch_size": batch_size,
        "configured_seeds": seeds,
        "batches": [],
    }

    def load_matching_provenance(batch_index: int, seed: int) -> dict | None:
        output_path = output_root / f"shard_{batch_index:03d}.yaml"
        provenance_path = output_root / f"shard_{batch_index:03d}.provenance.json"
        if not output_path.exists() and not provenance_path.exists():
            return None
        if not output_path.is_file() or not provenance_path.is_file():
            raise RuntimeError(f"Incomplete existing shard {batch_index:03d}")
        provenance = json.loads(provenance_path.read_text())
        if not provenance.get("complete"):
            raise RuntimeError(f"Existing shard {batch_index:03d} is not complete")
        expected = {
            "seed": seed,
            "candidates": batch_size,
            "object_mesh_sha256": mesh_sha,
            "descriptor_sha256": descriptor_sha,
            "point_cloud_sha256": point_cloud_sha,
            "output_sha256": sha256(output_path),
        }
        mismatches = {
            key: (provenance.get(key), value)
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing shard {batch_index:03d} provenance mismatch: {mismatches}"
            )
        return provenance

    for batch_index, seed in selected_batches:
        output_path = output_root / f"shard_{batch_index:03d}.yaml"
        provenance_path = output_root / f"shard_{batch_index:03d}.provenance.json"
        provenance = load_matching_provenance(batch_index, seed)
        if provenance is not None:
            print(f"shard {batch_index:03d}: complete and matching; skipped")
            continue

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        grasps, confidences = GraspGenXSampler.run_inference(
            centered_points,
            sampler,
            grasp_threshold=float(generation_cfg["grasp_threshold"]),
            num_grasps=batch_size,
            topk_num_grasps=batch_size,
            min_grasps=batch_size,
            max_tries=1,
            remove_outliers=bool(generation_cfg["remove_outliers"]),
        )
        if len(grasps) != batch_size:
            raise RuntimeError(
                f"Expected {batch_size} candidates in shard {batch_index:03d}, "
                f"received {len(grasps)}"
            )
        grasp_array = grasps.detach().cpu().numpy()
        confidence_array = confidences.detach().cpu().numpy()
        grasp_array[:, 3, 3] = 1.0
        object_grasps = np.asarray(
            [np.linalg.inv(center_transform) @ grasp for grasp in grasp_array]
        )
        data = save_to_isaac_grasp_format(object_grasps, confidence_array, None)
        renamed = {}
        content_hashes = set()
        for sample_index, (old_id, entry) in enumerate(data["grasps"].items()):
            del old_id
            candidate_id = (
                f"{part}__seed_{seed:010d}__sample_{sample_index:03d}"
            )
            content_sha = candidate_content_hash(
                object_mesh_sha256=mesh_sha,
                descriptor_sha256=descriptor_sha,
                generator_checkpoint_sha256=generator_sha,
                discriminator_checkpoint_sha256=discriminator_sha,
                point_cloud_sha256=point_cloud_sha,
                generation_seed=seed,
                object_T_G=object_grasps[sample_index],
            )
            if content_sha in content_hashes:
                raise RuntimeError(
                    f"Duplicate candidate content within shard {batch_index:03d}"
                )
            content_hashes.add(content_sha)
            entry["graspgenx_generation"] = {
                "candidate_id": candidate_id,
                "candidate_content_sha256": content_sha,
                "batch_index": batch_index,
                "generation_seed": seed,
                "sample_index": sample_index,
            }
            renamed[candidate_id] = entry
        data["grasps"] = renamed
        atomic_write_text(output_path, yaml.safe_dump(data, sort_keys=False))
        provenance = {
            "schema_version": 1,
            "complete": True,
            "batch_index": batch_index,
            "seed": seed,
            "candidates": len(renamed),
            "object_mesh_sha256": mesh_sha,
            "descriptor_sha256": descriptor_sha,
            "point_cloud_sha256": point_cloud_sha,
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "output_sha256": sha256(output_path),
            "confidence_min": float(confidence_array.min()),
            "confidence_max": float(confidence_array.max()),
        }
        atomic_write_text(provenance_path, json.dumps(provenance, indent=2) + "\n")
        print(
            f"shard {batch_index:03d}: {len(renamed)} diffusion candidates, "
            f"confidence {confidence_array.min():.3f}–{confidence_array.max():.3f}"
        )

    # A single-shard resume must not overwrite the manifest with a misleading
    # one-entry view. Re-scan every configured shard and record all completed,
    # provenance-matching batches.
    generation_manifest["batches"] = [
        provenance
        for batch_index, seed in enumerate(seeds)
        if (provenance := load_matching_provenance(batch_index, seed)) is not None
    ]
    generation_manifest["completed_batch_count"] = len(
        generation_manifest["batches"]
    )
    generation_manifest["complete"] = (
        generation_manifest["completed_batch_count"] == len(seeds)
    )
    manifest_path = output_root / "generation_manifest.json"
    atomic_write_text(manifest_path, json.dumps(generation_manifest, indent=2) + "\n")
    print(f"generation manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-config",
        type=Path,
        help="Run the reproducible sharded atlas-generation contract.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        help="With --atlas-config, generate or verify exactly one configured batch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="With --atlas-config, process only the first N batches (0 means all).",
    )
    parser.add_argument("--checkpoints", type=Path,
                        default=GRASPGENX_ROOT / "ext/graspgenx_checkpoints/release",
                        help="Release directory containing gen/ and dis/")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--gripper-name", default="dex3_rev1_right")
    parser.add_argument("--parts-root", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=PARTS,
        default=list(PARTS),
        help="AprilCube parts to process; their deterministic seeds remain fixed.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-grasps", type=int, default=240)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-points", type=int, default=3500)
    args = parser.parse_args()

    if args.atlas_config is not None:
        run_atlas(args)
        return

    args.checkpoints = args.checkpoints.resolve()
    args.assets_dir = args.assets_dir.resolve()
    args.parts_root = args.parts_root.resolve()
    args.output_root = args.output_root.resolve()

    args.output_root.mkdir(parents=True, exist_ok=True)
    hand_audit = descriptor_audit(args.assets_dir, args.gripper_name)
    model_cfg = load_model_cfg(
        str(args.checkpoints / "gen"),
        str(args.checkpoints / "dis"),
    )
    sampler = GraspGenXSampler(
        model_cfg,
        args.gripper_name,
        assets_dir=str(args.assets_dir),
    )

    provenance = {
        "schema_version": 1,
        "status": "raw_unqualified_candidates",
        "graspgenx_commit": "b9429097728cb1c430dd78b92edf17ba318aad03",
        "checkpoint_commit": CHECKPOINT_COMMIT,
        "generator_checkpoint": os.path.basename(str(model_cfg.eval.gen_checkpoint)),
        "discriminator_checkpoint": os.path.basename(str(model_cfg.eval.dis_checkpoint)),
        "seed": args.seed,
        "sample_points": args.sample_points,
        "requested_grasps": args.num_grasps,
        "retained_top_k": args.top_k,
        "gripper_descriptor": hand_audit,
        "parts": {},
    }

    for part in args.parts:
        mesh_path = args.parts_root / part / "grasp_mesh.obj"
        part_seed = args.seed + PARTS.index(part)
        np.random.seed(part_seed)
        torch.manual_seed(part_seed)
        torch.cuda.manual_seed_all(part_seed)
        points, center_transform, mesh = sample_centered(
            mesh_path, args.sample_points, part_seed
        )
        grasps, confidences = GraspGenXSampler.run_inference(
            points,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.top_k,
            min_grasps=args.top_k,
            max_tries=1,
            remove_outliers=False,
        )
        if len(grasps) != args.top_k:
            raise RuntimeError(f"Expected {args.top_k} {part} candidates, got {len(grasps)}")
        grasp_array = grasps.detach().cpu().numpy()
        confidence_array = confidences.detach().cpu().numpy()
        grasp_array[:, 3, 3] = 1.0
        object_grasps = np.asarray(
            [np.linalg.inv(center_transform) @ grasp for grasp in grasp_array]
        )
        output_path = args.output_root / f"{part}.yaml"
        save_to_isaac_grasp_format(object_grasps, confidence_array, str(output_path))
        provenance["parts"][part] = {
            "seed": part_seed,
            "mesh": str(mesh_path.relative_to(PROJECT_ROOT)),
            "mesh_sha256": sha256(mesh_path),
            "mesh_extents_m": mesh.extents.tolist(),
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "output_sha256": sha256(output_path),
            "candidates": int(len(object_grasps)),
            "confidence_min": float(confidence_array.min()),
            "confidence_max": float(confidence_array.max()),
        }
        print(
            f"{part}: {len(object_grasps)} candidates, "
            f"confidence {confidence_array.min():.3f}–{confidence_array.max():.3f}"
        )

    provenance_path = args.output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"provenance: {provenance_path}")


if __name__ == "__main__":
    main()
