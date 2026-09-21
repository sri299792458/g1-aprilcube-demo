#!/usr/bin/env python3
"""Empirically test the released GraspGenX frame and descriptor contracts.

Two different questions are kept separate:

1. Translating the input point cloud must translate every returned grasp by
   exactly the same amount.  This is a hard contract enforced by GraspGenX's
   center/uncenter code.
2. The generated descriptor consumed by inference must contain exactly the
   manifest's physics-qualified 12-number conditioning vector.  The vector is
   not shifted when the descriptor URDF's fixed palm transform changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = PROJECT_ROOT / "third_party/GraspGenX"
sys.path.insert(0, str(GRASPGENX_ROOT))

from graspgenx.grasp_server import GraspGenXSampler  # noqa: E402
from graspgenx.utils.checkpoint_io import load_model_cfg  # noqa: E402


DEFAULT_CHECKPOINTS = GRASPGENX_ROOT / "ext/graspgenx_checkpoints/release"
DEFAULT_MANIFEST = PROJECT_ROOT / "config/dex3_rev1_descriptor.json"
DEFAULT_DESCRIPTOR = (
    GRASPGENX_ROOT / "assets/x_grippers/dex3_rev1_right"
)
DEFAULT_CUBE = PROJECT_ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/dex3_validation/graspgenx_frame_contract.json"


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(
    sampler: GraspGenXSampler,
    points: np.ndarray,
    seed: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_all(seed)
    tensor = torch.from_numpy(points).cuda().float()
    grasps, scores, _ = sampler.sample(
        tensor,
        threshold=-1.0,
        num_grasps=count,
        remove_outliers=False,
    )
    result = grasps.detach().cpu().numpy()
    result[:, 3, 3] = 1.0
    return result, scores.detach().cpu().numpy()


def rotation_error_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a[:, :3, :3], 1, 2) @ b[:, :3, :3]
    cosines = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    translation = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1)
    rotation = rotation_error_degrees(actual, expected)
    return {
        "translation_error_m": {
            "maximum": float(translation.max()),
            "mean": float(translation.mean()),
            "p95": float(np.quantile(translation, 0.95)),
        },
        "rotation_error_degrees": {
            "maximum": float(rotation.max()),
            "mean": float(rotation.mean()),
            "p95": float(np.quantile(rotation, 0.95)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--num-grasps", type=int, default=60)
    parser.add_argument("--sample-points", type=int, default=3500)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    resolved = json.loads((args.descriptor / "config.json").read_text())
    sweep_spec = manifest["sweep_volume"]
    cfg = load_model_cfg(
        str(args.checkpoints / "gen"),
        str(args.checkpoints / "dis"),
    )

    sampler = GraspGenXSampler(
        cfg,
        args.descriptor.name,
        assets_dir=str(args.descriptor.parents[1]),
    )

    conditioning_fields = ("extents", "offset", "extents2", "offset2")
    descriptor_errors = {
        field: float(
            np.max(
                np.abs(
                    np.asarray(resolved["sweep_volume"][field], dtype=float)
                    - np.asarray(sweep_spec[field], dtype=float)
                )
            )
        )
        for field in conditioning_fields
    }
    descriptor_contract_passed = (
        max(descriptor_errors.values()) <= 1.0e-12
        and resolved.get("review_status") == sweep_spec["status"]
    )

    mesh = trimesh.load(args.cube, force="mesh", process=False)
    seed_all(args.seed)
    points, _ = trimesh.sample.sample_surface(mesh, args.sample_points)
    points -= points.mean(axis=0)

    centered, centered_scores = sample(sampler, points, args.seed, args.num_grasps)

    # Hard translation contract. GraspGenX centers internally, then restores
    # the input mean to the returned translation.
    input_translation = np.array([0.173, -0.081, 0.249])
    translated, translated_scores = sample(
        sampler, points + input_translation, args.seed, args.num_grasps
    )
    expected_translated = centered.copy()
    expected_translated[:, :3, 3] += input_translation
    translation_contract = comparison(translated, expected_translated)
    score_translation_error = float(
        np.max(np.abs(translated_scores - centered_scores))
    )
    translation_passed = (
        translation_contract["translation_error_m"]["maximum"] <= 5.0e-6
        and translation_contract["rotation_error_degrees"]["maximum"] <= 0.05
        and score_translation_error <= 2.0e-4
    )

    result = {
        "schema_version": 1,
        "status": "graspgenx_frame_contract_test",
        "seed": args.seed,
        "candidate_count": args.num_grasps,
        "input_translation_contract": {
            "passed": translation_passed,
            "applied_translation_m": input_translation.tolist(),
            "candidate_comparison": translation_contract,
            "maximum_confidence_error": score_translation_error,
            "interpretation": (
                "A pass establishes that returned transforms are expressed in "
                "the input point-cloud frame and use left-multiplied frame changes."
            ),
        },
        "descriptor_conditioning_contract": {
            "passed": descriptor_contract_passed,
            "manifest_status": sweep_spec["status"],
            "resolved_status": resolved.get("review_status"),
            "maximum_error_by_field": descriptor_errors,
            "interpretation": (
                "The generated config contains the manifest's exact released-"
                "checkpoint morphology proxy. This checks file plumbing only; "
                "the separate Isaac/PhysX test establishes retention."
            ),
        },
        "claim_boundary": (
            "Neither test establishes contact or grasp stability."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not translation_passed or not descriptor_contract_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
