#!/usr/bin/env python3
"""Add the Isaac physics contract to raw GraspGenX candidates without changing poses."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_usd_package(root_usd: Path) -> str:
    """Hash the root USD and generated USD sublayers it references."""
    files = [root_usd]
    configuration = root_usd.parent / "configuration"
    if configuration.is_dir():
        files.extend(sorted(configuration.glob("*.usd")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root_usd.parent)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-config",
        type=Path,
        help="Build one side-specific Isaac shard from the cube-atlas contract.",
    )
    parser.add_argument(
        "--hand-side",
        choices=("right", "left"),
        help="Required with --atlas-config.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Required with --atlas-config.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Replace an existing atlas input when its physics asset contract "
            "has intentionally changed; candidate transforms are still copied "
            "verbatim from the raw shard."
        ),
    )
    parser.add_argument(
        "--raw-grasps",
        type=Path,
        default=repo / "artifacts/aprilcube_raw_grasps/cube_head.yaml",
    )
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=repo / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right/config.json",
    )
    parser.add_argument(
        "--gripper-usd",
        type=Path,
        default=repo / "third_party/GraspDataGen/bots/dex3_rev1_right/dex3_rev1_right.usd",
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=repo / "generated/aprilcube_parts/cube_head/grasp_mesh.obj",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "artifacts/isaac_grasp_validation/dex3_cube_input.yaml",
    )
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Optional comma-separated GraspGenX indices to retain, in display order.",
    )
    parser.add_argument(
        "--finger-colliders",
        default="right_hand_thumb_2_link,right_hand_middle_1_link",
        help="Comma-separated opposing terminal links used by the upstream validator.",
    )
    parser.add_argument("--object-mass", type=float, default=0.12)
    parser.add_argument("--approach-axis", type=int, default=2)
    parser.add_argument("--open-limit", choices=("lower", "upper"), default="lower")
    parser.add_argument(
        "--finger-contact-groups",
        default="",
        help=(
            "Optional JSON list of finger-chain link lists. When supplied, the "
            "validator requires object contact in --min-contact-groups chains."
        ),
    )
    parser.add_argument(
        "--contact-trace-links",
        default="",
        help=(
            "Optional comma-separated hand links observed by the validator's "
            "diagnostic contact trace. This does not change pass/fail."
        ),
    )
    parser.add_argument(
        "--contact-trace-link-aliases",
        default="",
        help=(
            "Optional JSON object mapping imported PhysX body names to logical "
            "hand-link names in trace records."
        ),
    )
    parser.add_argument("--min-contact-groups", type=int, default=2)
    return parser.parse_args()


def project_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def side_contract(repo: Path, side: str) -> dict:
    prefix = f"{side}_hand"
    contact_trace_links = [
        f"{prefix}_palm_link",
        f"{prefix}_thumb_0_link",
        f"{prefix}_thumb_1_link",
        f"{prefix}_thumb_2_link",
        f"{prefix}_index_0_link",
        f"{prefix}_index_1_link",
        f"{prefix}_middle_0_link",
        f"{prefix}_middle_1_link",
    ]
    return {
        "hand_config": repo
        / f"third_party/GraspGenX/assets/x_grippers/dex3_rev1_{side}/config.json",
        "gripper_usd": repo
        / f"third_party/GraspDataGen/bots/dex3_rev1_{side}/dex3_rev1_{side}.usd",
        "finger_colliders": [
            f"{prefix}_thumb_2_link",
            f"{prefix}_middle_1_link",
        ],
        "finger_contact_groups": [
            [
                f"{prefix}_thumb_0_link",
                f"{prefix}_thumb_1_link",
                f"{prefix}_thumb_2_link",
            ],
            [f"{prefix}_index_0_link", f"{prefix}_index_1_link"],
            [f"{prefix}_middle_0_link", f"{prefix}_middle_1_link"],
        ],
        "contact_trace_links": contact_trace_links,
        # Fixed joints are deliberately preserved in the hand-only Isaac
        # asset, so PhysX exposes the palm under its actual URDF link name.
        "contact_trace_link_aliases": {},
    }


def build_output(
    *,
    raw_path: Path,
    hand_config: Path,
    gripper_usd: Path,
    object_mesh: Path,
    object_mass: float,
    finger_colliders: list[str],
    finger_contact_groups: list[list[str]] | None,
    min_contact_groups: int,
    approach_axis: int,
    open_limit: str,
    atlas_metadata: dict | None = None,
    contact_trace_links: list[str] | None = None,
    contact_trace_link_aliases: dict[str, str] | None = None,
    requested_candidate_ids: list[str] | None = None,
) -> dict:
    raw = yaml.safe_load(raw_path.read_text())
    hand = json.loads(hand_config.read_text())
    grasps = copy.deepcopy(raw["grasps"])
    if requested_candidate_ids is not None:
        missing = [candidate_id for candidate_id in requested_candidate_ids if candidate_id not in grasps]
        if missing:
            raise KeyError(f"Unknown candidate IDs: {missing}")
        grasps = {candidate_id: grasps[candidate_id] for candidate_id in requested_candidate_ids}

    for candidate_id, grasp in grasps.items():
        generation = copy.deepcopy(grasp.pop("graspgenx_generation", {}))
        if generation and generation.get("candidate_id") != candidate_id:
            raise ValueError(
                f"Candidate identity mismatch: key={candidate_id}, metadata={generation}"
            )
        grasp["graspgenx_source"] = {
            "candidate_id": candidate_id,
            "candidate_content_sha256": generation.get("candidate_content_sha256"),
            "batch_index": generation.get("batch_index"),
            "generation_seed": generation.get("generation_seed"),
            "sample_index": generation.get("sample_index"),
            "confidence": float(grasp["confidence"]),
            "object_T_G": {
                "position": copy.deepcopy(grasp["position"]),
                "orientation": copy.deepcopy(grasp["orientation"]),
            },
        }
        # Keep the historical field while consumers migrate to the precise name.
        grasp["graspgenx_source"]["object_T_grasp"] = copy.deepcopy(
            grasp["graspgenx_source"]["object_T_G"]
        )
        grasp["pregrasp_cspace_position"] = copy.deepcopy(hand["open"])
        grasp["cspace_position"] = copy.deepcopy(hand["close"])

    output = {
        "format": "isaac_grasp",
        "format_version": "1.0",
        "created_with": "graspgenx",
        "object_file": str(object_mesh.resolve()),
        "object_scale": 1.0,
        "obj2usd_use_existing_usd": False,
        "obj2usd_mass": object_mass,
        "gripper_file": str(gripper_usd.resolve()),
        "finger_colliders": finger_colliders,
        "open_limit": open_limit,
        "use_cspace_position_as_target": True,
        "approach_axis": approach_axis,
        "source": {
            "raw_grasps": str(raw_path.resolve()),
            "raw_grasps_sha256": sha256(raw_path),
            "hand_config": str(hand_config.resolve()),
            "hand_config_sha256": sha256(hand_config),
            "gripper_usd": str(gripper_usd.resolve()),
            "gripper_usd_sha256": sha256(gripper_usd),
            "gripper_usd_package_sha256": sha256_usd_package(gripper_usd),
            "object_mesh": str(object_mesh.resolve()),
            "object_mesh_sha256": sha256(object_mesh),
            "transform_policy": "copied_verbatim_from_raw_graspgenx_output",
        },
        "grasps": grasps,
    }
    if atlas_metadata is not None:
        output["grasp_atlas"] = atlas_metadata
    if contact_trace_links is not None:
        output["contact_trace_links"] = contact_trace_links
    if contact_trace_link_aliases is not None:
        output["contact_trace_link_aliases"] = contact_trace_link_aliases
    if finger_contact_groups is not None:
        output["finger_contact_groups"] = finger_contact_groups
        output["min_contact_groups"] = min_contact_groups
    return output


def run_atlas(args: argparse.Namespace, repo: Path) -> None:
    if args.hand_side is None or args.shard_index is None:
        raise ValueError("--atlas-config requires --hand-side and --shard-index")
    config_path = args.atlas_config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported atlas config schema: {config.get('schema_version')}")
    seeds = [int(seed) for seed in config["generation"]["seeds"]]
    if not 0 <= args.shard_index < len(seeds):
        raise IndexError(f"--shard-index outside [0, {len(seeds) - 1}]")
    artifacts_root = project_path(repo, config["artifacts_root"])
    raw_path = artifacts_root / "raw" / f"shard_{args.shard_index:03d}.yaml"
    output_path = (
        artifacts_root
        / args.hand_side
        / "isaac_inputs"
        / f"shard_{args.shard_index:03d}.yaml"
    )
    side = side_contract(repo, args.hand_side)
    object_mesh = project_path(repo, config["object"]["mesh"])
    for required in (raw_path, side["hand_config"], side["gripper_usd"], object_mesh):
        if not required.is_file():
            raise FileNotFoundError(required)
    output = build_output(
        raw_path=raw_path,
        hand_config=side["hand_config"],
        gripper_usd=side["gripper_usd"],
        object_mesh=object_mesh,
        object_mass=float(config["object"]["mass_kg"]),
        finger_colliders=side["finger_colliders"],
        finger_contact_groups=side["finger_contact_groups"],
        min_contact_groups=int(config["physics"]["min_contact_groups"]),
        approach_axis=2,
        open_limit="lower",
        atlas_metadata={
            "schema_version": 1,
            "atlas_id": config["atlas_id"],
            "config": str(config_path.relative_to(repo)),
            "config_sha256": sha256(config_path),
            "object_id": config["object"]["id"],
            "hand_side": args.hand_side,
            "shard_index": args.shard_index,
            "generation_seed": seeds[args.shard_index],
            "detailed_point_budget_per_body_environment": int(
                config["physics"]["detailed_point_budget_per_body_environment"]
            ),
        },
        contact_trace_links=side["contact_trace_links"],
        contact_trace_link_aliases=side["contact_trace_link_aliases"],
    )
    rendered = yaml.safe_dump(output, sort_keys=False)
    if output_path.exists():
        existing = yaml.safe_load(output_path.read_text())
        existing_without_atlas = copy.deepcopy(existing)
        output_without_atlas = copy.deepcopy(output)
        existing_without_atlas.pop("grasp_atlas", None)
        output_without_atlas.pop("grasp_atlas", None)
        contract_changed = existing_without_atlas != output_without_atlas
        if contract_changed:
            if not args.replace_existing:
                raise RuntimeError(
                    f"Existing Isaac shard differs outside atlas metadata: {output_path}; "
                    "pass --replace-existing only for an intentional asset-contract change"
                )
            output_path.write_text(rendered)
            print(
                "replaced physics asset contract without changing raw candidate "
                f"transforms: {output_path}"
            )
            return
        if existing == output:
            print(f"matching Isaac input already exists: {output_path}")
            return
        output_path.write_text(rendered)
        print(f"refreshed atlas metadata without changing candidates: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(
        f"wrote {len(output['grasps'])} unchanged {args.hand_side} candidates "
        f"to {output_path}"
    )


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.atlas_config is not None:
        run_atlas(args, repo)
        return

    finger_colliders = [
        value.strip() for value in args.finger_colliders.split(",") if value.strip()
    ]
    if len(finger_colliders) != 2:
        raise ValueError("--finger-colliders must name exactly two opposing links")
    finger_contact_groups = None
    if args.finger_contact_groups:
        finger_contact_groups = json.loads(args.finger_contact_groups)
        if not isinstance(finger_contact_groups, list) or not all(
            isinstance(group, list) and group and all(isinstance(link, str) for link in group)
            for group in finger_contact_groups
        ):
            raise ValueError("--finger-contact-groups must be a JSON list of nonempty string lists")
        if not 1 <= args.min_contact_groups <= len(finger_contact_groups):
            raise ValueError("--min-contact-groups must be between one and the group count")
    contact_trace_links = None
    if args.contact_trace_links:
        contact_trace_links = [
            value.strip()
            for value in args.contact_trace_links.split(",")
            if value.strip()
        ]
        if len(contact_trace_links) != len(set(contact_trace_links)):
            raise ValueError("--contact-trace-links contains duplicates")
    contact_trace_link_aliases = None
    if args.contact_trace_link_aliases:
        contact_trace_link_aliases = json.loads(args.contact_trace_link_aliases)
        if not isinstance(contact_trace_link_aliases, dict) or not all(
            isinstance(source, str)
            and source
            and isinstance(target, str)
            and target
            for source, target in contact_trace_link_aliases.items()
        ):
            raise ValueError("--contact-trace-link-aliases must be a JSON string map")
        if contact_trace_links is None or not set(contact_trace_link_aliases).issubset(
            contact_trace_links
        ):
            raise ValueError("Every contact-trace alias key must name a traced body")

    requested_candidate_ids = None
    if args.candidate_ids:
        requested_candidate_ids = [
            f"grasp_{int(value.strip())}"
            for value in args.candidate_ids.split(",")
            if value.strip()
        ]
    output = build_output(
        raw_path=args.raw_grasps,
        hand_config=args.hand_config,
        gripper_usd=args.gripper_usd,
        object_mesh=args.object_mesh,
        object_mass=args.object_mass,
        finger_colliders=finger_colliders,
        finger_contact_groups=finger_contact_groups,
        min_contact_groups=args.min_contact_groups,
        approach_axis=args.approach_axis,
        open_limit=args.open_limit,
        contact_trace_links=contact_trace_links,
        contact_trace_link_aliases=contact_trace_link_aliases,
        requested_candidate_ids=requested_candidate_ids,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False))
    print(f"wrote {len(output['grasps'])} unchanged GraspGenX poses to {args.output}")


if __name__ == "__main__":
    main()
