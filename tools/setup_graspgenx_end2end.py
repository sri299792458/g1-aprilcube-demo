#!/usr/bin/env python3
"""Materialize the pinned assets needed by GraspGenX end-to-end demos.

Run this with the project interpreter after ``uv sync``::

    .venv/bin/python tools/setup_graspgenx_end2end.py

The upstream GraspGenX setup script also builds an unrelated UR10e example and
invokes ``uv`` from inside the submodule. This project needs neither behavior.
This wrapper performs only the two setup operations used by our verified
Franka baseline and future G1 adapter:

* materialize the Git-LFS gripper meshes; and
* materialize pinned cuRobo assets and install that checkout editable into the
  project environment.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRASPGENX = ROOT / "third_party" / "GraspGenX"
GRIPPER_DESCRIPTIONS = GRASPGENX / "ext" / "gripper_descriptions"
CUROBO = GRASPGENX / "ext" / "curobo"
FRANKA_MESH = (
    GRIPPER_DESCRIPTIONS
    / "gripper_descriptions"
    / "assets"
    / "x_grippers"
    / "franka_panda"
    / "vis_mesh.obj"
)


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def load_upstream_setup():
    path = GRASPGENX / "end2end" / "setup_end2end_deps.py"
    spec = importlib.util.spec_from_file_location("graspgenx_e2e_setup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not GRASPGENX.is_dir():
        raise SystemExit(
            "GraspGenX submodule is missing; run "
            "`git submodule update --init third_party/GraspGenX`."
        )

    # Importing GraspGenX invokes its pinned gripper-description checkout
    # bootstrap when the checkout is absent.
    run(sys.executable, "-c", "import graspgenx", cwd=ROOT)
    if not (GRIPPER_DESCRIPTIONS / ".git").is_dir():
        raise SystemExit(f"gripper_descriptions checkout missing at {GRIPPER_DESCRIPTIONS}")
    run("git", "lfs", "pull", cwd=GRIPPER_DESCRIPTIONS)
    if not FRANKA_MESH.is_file() or FRANKA_MESH.stat().st_size < 100_000:
        raise SystemExit(f"Git-LFS gripper mesh was not materialized: {FRANKA_MESH}")

    upstream = load_upstream_setup()
    upstream._ensure_curobo_assets()
    run(
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "-e",
        str(CUROBO),
        "--no-deps",
        cwd=ROOT,
    )

    run(
        sys.executable,
        "-c",
        "from cuda.core import Device; import curobo, newton; print('E2E imports OK')",
        cwd=ROOT,
    )
    print(f"Franka mesh: {FRANKA_MESH.stat().st_size} bytes")
    print(
        "cuRobo commit:",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CUROBO, text=True
        ).strip(),
    )


if __name__ == "__main__":
    main()
