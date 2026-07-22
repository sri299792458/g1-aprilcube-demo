"""Generate the three physical AprilCube task parts from pinned YAML specs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APRILCUBE_SOURCE = PROJECT_ROOT / "third_party/aprilcube/src"
SPEC_ROOT = PROJECT_ROOT / "config/aprilcube_parts"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated/aprilcube_parts"
PARTS = {
    "t_body": "t_body.yaml",
    "u_legs": "u_legs.yaml",
    "cube_head": "cube_head.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(APRILCUBE_SOURCE) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    audit = {"schema_version": 1, "parts": {}}
    for name, spec_name in PARTS.items():
        spec = SPEC_ROOT / spec_name
        destination = args.output_root / name
        command = [
            sys.executable,
            "-m",
            "aprilcube.cli",
            "generate",
            str(spec),
            "--output",
            str(destination),
        ]
        print(" ".join(command))
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
        required = [
            destination / "cube.3mf",
            destination / "config.json",
            destination / "thumbnail.png",
            destination / "mujoco/cube.obj",
            destination / "mujoco/cube.mtl",
            destination / "mujoco/cube_atlas.png",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"AprilCube did not generate required files: {missing}")
        config = json.loads((destination / "config.json").read_text())
        audit["parts"][name] = {
            "spec": str(spec.relative_to(PROJECT_ROOT)),
            "object_mesh": str((destination / "mujoco/cube.obj").relative_to(PROJECT_ROOT)),
            "target_type": config["target"]["type"],
            "voxel_size_mm": config["target"]["voxel_size_mm"],
            "tag_size_mm": config["tag_size_mm"],
            "edge_radius_mm": config["edge_radius_mm"],
            "tag_ids": config["tag_ids"],
        }

    audit_path = PROJECT_ROOT / "artifacts/aprilcube_parts/audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"audit: {audit_path}")


if __name__ == "__main__":
    main()
