# G1 AprilCube Demo

Clean implementation of a seated Unitree G1 + Dex3 tabletop assembly demo.

The repository has passed the **current Dex3 / 45 mm cube intrinsic
grasp gate**. It contains pinned GraspGenX and AprilCube dependencies,
reproducible descriptors for the official current Unitree Dex3, the real
T/U/cube print geometry, and Isaac/PhysX-qualified neural grasp candidates.
There is no hand-authored grasp.

## Fixed demo scope

- One hand picks and holds an AprilCube T by its central stem.
- The other hand attaches the U legs and cube head.
- The assembled figure is placed upright on the table.
- Grasp candidates come from GraspGenX; cuRoboV2 is the motion-planning backend.

The implemented runtime contract and every implementation-driven correction
are recorded in [the runtime cuRobo specification](docs/runtime_curobo_assembly_spec.md).
The earlier fixed-scene design remains in
[the execution-stack specification](docs/execution_stack.md) as historical
regression context.

## Current runtime checkpoint

[Watch the runtime-conditioned T/U/cube assembly plan](docs/assets/t_u_cube_runtime_curobo_v2.mp4).
The video is reconstructed from the saved successful 14-joint cuRobo
trajectories and attachment state—not from a separately animated preview. It
uses the current Unitree Dex3 meshes, actual 45 mm AprilCube T/U/cube meshes,
and the exact grasp candidates selected for the observed loose-part poses.

Plan the nominal observation and reproduce the video from the repository root:

```bash
.venv/bin/python tools/run_runtime_assembly.py \
  --observation config/observations/t_u_cube_nominal_v1.yaml \
  --output artifacts/runtime_assembly/t_u_cube_v2/nominal
.venv/bin/python tools/render_full_assembly.py \
  --config config/planning/t_u_cube_runtime_v2.yaml \
  --run-dir artifacts/runtime_assembly/t_u_cube_v2/nominal
```

Run the same executable against the second separated XY/yaw arrangement by
changing only `--observation` to
`config/observations/t_u_cube_shuffled_v1.yaml`. Both observations complete
all six compiler-checked task transitions. Generated reports, trajectories,
render state, timelines, and MP4s live under
`artifacts/runtime_assembly/t_u_cube_v2/`; the reviewed nominal MP4 above is
the committed visual checkpoint.

![Selected 45 mm task scale](docs/assets/aprilcube_45mm_scale.png)

The task uses 45 mm voxels and produces a 360 mm completed figure. This image
is a size reference only; it does not claim a grasp or contact result.

![Raw GraspGenX proposals on the task parts](docs/assets/aprilcube_raw_grasp_audit.png)

The image shows only the open hand. A fixed terminal-close overlay was removed
because it passes through objects that should stop the real fingers and cannot
validate a grasp. [The raw AprilCube grasp checkpoint](docs/aprilcube_raw_grasp_checkpoint.md)
is retained as historical context.

![Current Unitree Dex3 descriptor states](docs/assets/dex3_rev1_descriptor_states.png)

The boxes are the released checkpoint's physics-qualified morphology proxy,
not a literal enclosure claim for the current L-shaped open posture. Read [the
current Dex3 descriptor root-cause audit](docs/dex3_rev1_descriptor.md).

The canonical named pipeline sent all 120 cube proposals unchanged into the
exact current hands. 118 survived closure and five directional tugs on the
right; 116 survived when the same proposals were replayed on the mirrored left.
[Watch the right-hand sequential close-up review of eight passes and both
right-hand failures](docs/assets/dex3_descriptor_corrected_review10_sequential.mp4).

Using masses scaled from the 30 g cube print, the same intrinsic right-hand
screen retained 43/120 T proposals at 181 g and 9/120 U proposals at 211 g.
[T passing-grasp review](docs/assets/dex3_t_body_passing_grasps_grid.mp4) ·
[U passing-grasp review](docs/assets/dex3_u_legs_passing_grasps_grid.mp4)

![Upstream descriptor and candidate](docs/assets/graspgenx_unitree_upstream_probe.png)

![One candidate from several views](docs/assets/graspgenx_unitree_candidate_multiview.png)

Read [the verified GraspGenX contract](docs/graspgenx_contract.md) before using
the candidates in robot planning code.

## Python environment

The Python stack belongs to this repository. From the repository root, install
`uv` once if necessary, then reproduce the exact locked environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
~/.local/bin/uv sync --frozen
source .venv/bin/activate
```

Python 3.11 is managed automatically from `.python-version`. The lock contains
GraspGenX, cuRobo, PyTorch 2.6.0+cu124, Newton 1.0.0, Warp 1.15.0, MuJoCo
3.5.0, and MuJoCo-Warp 3.5.0.2. The runtime assembly planner uses cuRobo and
does not depend on Newton or MuJoCo. Those packages remain for isolated grasp
physics validation; when that path is used, Newton owns the model while
MuJoCo-Warp is the GPU solver behind Newton's `SolverMuJoCo`.

Verify imports, CUDA, and an actual rigid-contact solve:

```bash
python tools/check_sim_stack.py
```

## Build the current Dex3 descriptors

From the activated repository environment:

```bash
PYOPENGL_PLATFORM=egl python tools/build_dex3_rev1_descriptors.py \
  --source-root /path/to/xr_teleoperate
```

Omit `--source-root` to download and hash-check only the pinned official hand
assets. Runtime files are generated where GraspGenX resolves them, under
`third_party/GraspGenX/assets/x_grippers/`; the numerical audit is
[here](artifacts/dex3_rev1_descriptor/audit.json).

## Pinned upstream dependency

```text
GraspGenX b9429097728cb1c430dd78b92edf17ba318aad03
AprilCube fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5
```

Clone with submodules:

```bash
git clone --recurse-submodules <this-repository>
```

The GraspGenX checkout downloads model checkpoints and gripper descriptions on
first import. Its Unitree hand meshes are stored with Git LFS; verify that the
actual meshes, rather than 130-byte pointer files, are materialized before
trusting a render.

## Reproduce checkpoint 0

Run the locked environment setup above, then set both asset variables **before
importing GraspGenX**:

```bash
export GRASPGENX_CHECKPOINT_DIR=/absolute/path/to/graspgenx_checkpoints
export GRASPGENX_GRIPPER_CFG_DIR=/absolute/path/to/gripper_descriptions
```

Generate a fresh raw candidate set with the upstream script:

```bash
python third_party/GraspGenX/scripts/demo_object_mesh.py \
  --mesh_file third_party/GraspGenX/assets/sample_data/object_mesh/box.obj \
  --checkpoints "$GRASPGENX_CHECKPOINT_DIR/release" \
  --gripper_name unitree_g1 \
  --grasp_threshold -1.0 --return_topk --topk_num_grasps 20 \
  --num_grasps 80 --num_sample_points 3500 --no-visualization \
  --output_file artifacts/upstream_probe/unitree_box_grasps.yml
```

The model is stochastic, so a fresh YAML need not match the committed snapshot.
Render the generated set:

```bash
PYOPENGL_PLATFORM=egl python tools/render_upstream_unitree_probe.py
```

The renderer refuses to run when the palm mesh still looks like a Git LFS
pointer. [Probe provenance](artifacts/upstream_probe/provenance.yaml) records the
exact revisions and input hash used for the committed images.

## Phase gate

The authoritative current-right-Dex3 contact atlases defined in
[the implementation specification](docs/dex3_aprilcube_grasp_atlas_spec.md)
are complete for the cube, T, and U. Each contains 4,096 unchanged GraspGenX
diffusion proposals qualified with the executed Isaac/PhysX and finger-control
contract from GR00T-VisualSim2Real commit `92bf086`, then grouped by coarse
body-level contact:

| Object | Isaac PASS | PASS rate | Contact families |
|---|---:|---:|---:|
| 45 mm cube | 2,437 / 4,096 | 59.50% | 40 |
| T body | 1,240 / 4,096 | 30.27% | 39 |
| U legs | 675 / 4,096 | 16.48% | 28 |

The sequential videos show one primary representative per family. A replay
failure remains visibly labeled as a diagnostic; it does not rewrite the
original 256-environment qualification verdict.

[Cube VIRAL-profile review](docs/assets/dex3_cube_grasp_families_right_viral.mp4) ·
[T VIRAL-profile review](docs/assets/dex3_t_body_grasp_families_right_viral.mp4) ·
[U VIRAL-profile review](docs/assets/dex3_u_legs_grasp_families_right_viral.mp4)

Left-hand qualification is complete for the T holder pool used by the fixed
demo; the right hand uses the qualified U and cube pools. The complete
collision-aware arm plan, dual attachment transfers, placement, and visual
replay now pass. Physical magnetic connectors, ROS 2 trajectory/hand execution,
AprilTag perception, camera calibration, and real seated-G1 validation remain
separate hardware gates.
