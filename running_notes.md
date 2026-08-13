# Running notes

## 2026-07-18 — clean restart, checkpoint 0

All implementation artifacts from the earlier prototype are excluded. They are
not migration inputs and are not compatibility requirements.

Pinned and audited:

- GraspGenX: `b9429097728cb1c430dd78b92edf17ba318aad03`
- Gripper descriptions used for the probe:
  `19a03c00d19aeaf052d0f6801f0041982d676e8a`
- Gripper: upstream `unitree_g1`, right hand only
- Probe object: upstream 0.10 m cube mesh
- Model output: 40 candidate `4x4` transforms and 40 confidence scores

Confirmed by code trace and render:

1. A returned matrix is the pose of the GraspGenX canonical root `G` in the
   input point-cloud frame `F`: `F_T_G`.
2. It is not a palm pose, wrist pose, pregrasp, or arm trajectory.
3. The Unitree descriptor URDF contains the fixed transform
   `G_T_right_palm_link` as its `world_joint`.
4. Therefore `F_T_right_palm_link = F_T_G @ G_T_right_palm_link`.
5. The released checkpoint is conditioned by the 12 open/half-open sweep-box
   numbers. It does not read the URDF or seven joint endpoints during inference.
6. The descriptor's open/close joint dictionaries are separate geometry and
   execution endpoints. Closing is not synthesized per object by GraspGenX.

Problems found before any project code was written:

- The upstream mesh demo README mentions a `--planner` flag that its current
  argument parser does not accept.
- Unitree hand meshes initially existed only as Git LFS pointer files. Inference
  could still run from the sweep descriptor, but the resulting empty visual
  meshes would have made visual validation invalid.
- The shipped Unitree descriptor is right-hand-only. Left-hand handling remains
  an explicit open item; it will not be guessed.
- A high confidence is not proof of collision-free closure or arm reachability.

Next gate: review the visual probe and contract. After approval, generate and
render candidates for the actual AprilCube meshes before introducing MoveIt.

## 2026-07-18 — official Dex3 left/right geometry check

Checked the current official Unitree `xr_teleoperate` hand assets at commit
`7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`.

- In their native palm frames, the complete zero-joint left-hand visual
  geometry equals the right-hand geometry rotated 180 degrees about palm +X:
  `left = Rx(pi) @ right`. The measured maximum mesh discrepancy is about
  2 micrometres (STL rounding).
- The same equality holds at a nonzero test configuration when left thumb
  joints negate the right thumb joints and left index/middle negate and swap
  right middle/index.
- Palm +X points longitudinally from the wrist/palm base toward the fingers.
- Consequently one verified actual-right canonical mapping is sufficient;
  the equivalent actual-left mapping is derived as
  `G_T_left_palm = G_T_right_palm @ Rx(pi)`.
- The GraspGenX-bundled `unitree_g1` hand is not geometrically identical to
  the current official Unitree Dex3 asset. The bundled mapping remains valid
  for its own probe, but the actual G1 mapping still requires an explicit
  overlay check before MoveIt integration.
- Provenance audit: all eight bundled right-hand STL files are byte-for-byte
  identical to the files under GR00T-VisualSim2Real's `g1/old/mesh/G1`, and
  all seven right-hand joint origins, axes, limits, and parent/child pairs
  match its old `g1_unitree.urdf`. GraspGenX changes those old fixed joints to
  revolute joints and supplies convex collision meshes for its descriptor.
- This establishes the same old Unitree G1 hand asset lineage, but not a direct
  GR00T-to-GraspGenX copy: the bundled asset's README explicitly names Unitree
  `unitree_ros/g1_description` as its source.
- Our selected mode-5 robot instead uses the newer rev-1.0 Dex3 model with
  `right_hand_*` / `left_hand_*` link names. The old and rev-1.0 hands are not
  related by one scale factor; palm proportions and finger-base locations,
  masses, inertias, and joint naming all differ.
- Rev-1.0 provenance is official Unitree, not NVIDIA-authored geometry:
  Unitree updated the current hand meshes in `unitree_ros` commit `9a7481d`
  on 2024-09-25 and added `g1_29dof_with_hand_rev_1_0` in commit `e6b9e8c`
  on 2024-10-22. GR00T's G1 assets appeared publicly in April 2026; its USD
  metadata names Unitree's rev-1.0 model as the source, and all eight current
  right-hand mesh LFS hashes exactly match the official Unitree files. NVIDIA
  converted/packaged and simulation-configured that model for GR00T.

Visual explanation: `docs/assets/dex3_rx_pi_explainer.png`; reproducible
renderer: `tools/render_dex3_rx_pi_explainer.py`.

## 2026-07-18 — GraspGenX training/checkpoint audit

- The committed upstream probe used the authors' public post-CVPR `release`
  checkpoint at model-repository commit
  `7c834043c11a11417e31d6d5ea9355801e40a2c1`: generator
  `epoch_736.pth` and discriminator `epoch_1056.pth`.
- Both released configs specify `train_gripper_split: proc_v1_train_32`.
  This is the latest model described in the paper as trained on 32 procedural
  grippers, 8.5K objects, and more than 2 billion simulated grasps.
- Unitree G1 is a held-out real test gripper in the paper, not one of those
  training grippers. The paper's per-target supervised-fine-tuning experiment
  is an adaptation study; no G1-specific fine-tuned checkpoint is present in
  the public model repository, and we are not using one.
- Therefore the old Unitree hand bundled in `gripper_descriptions` is an
  inference/evaluation embodiment definition, not evidence that the released
  network weights were trained or fine-tuned on that exact hand.
- For our rev-1.0 Dex3, use the same released cross-embodiment checkpoint but
  create a new descriptor from the exact rev-1.0 open and half-open sweep
  volumes and its own canonical-root-to-palm transform. The old 12 numbers can
  seed the annotation, but must not be copied unchanged; its transform and
  collision geometry are not valid for the current hand.
- This is the model's intended zero-shot use, but it is not a guarantee. Its
  12-number swept-volume representation omits detailed kinematics, contact
  friction, and the three-finger asymmetric contact pattern. Candidate
  ranking must therefore be followed by exact-hand geometry/closure,
  collision, IK, and motion-planning filters.

## 2026-07-18 — current Dex3 descriptors, visual gate

Built reproducible `dex3_rev1_right` and `dex3_rev1_left` GraspGenX descriptor
assets from the exact official Unitree `xr_teleoperate` hand source at commit
`7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`.

- The source manifest pins and SHA-256 verifies both URDFs, all 16 meshes, and
  the Apache-2.0 license.
- The build strips nonphysical standalone-teleoperation auxiliary links. Each
  generated URDF contains only canonical `world`, the palm, seven joints, and
  seven physical finger links.
- Right and left get separate fixed canonical-to-palm transforms. Their
  canonical open-hand bounding boxes agree within `1.32e-10 m`.
- Open/close joint endpoints are the provisional GR00T-VisualSim2Real profiles
  at commit `92bf086357156f04273cc5a3e9559e6b1415c8c7`; all values pass the current
  official URDF limits with at least `0.3216 rad` margin at close.
- The old GraspGenX Unitree boxes were used as a semantic and scale template,
  then fitted to current fingertip geometry. They remain marked
  `visual_review_required`.
- The upstream wizard's automatic initializer was evaluated but rejected for
  these boxes: the L-shaped zero-joint posture makes its largest-centroid-spread
  heuristic select canonical `+Z` as the closing axis, while the verified
  physical closing direction is canonical `+X`. The interactive/manual box
  review is therefore not optional for this hand.
- Both descriptors load through GraspGenX's project-asset resolution with the
  expected 12 sweep values and materialized visual/collision meshes. Missing
  point-cloud, TSDF, and VAE representation files fall back to dummy data as
  upstream permits; the released `sweep_volume_v2` checkpoint does not consume
  those legacy representations.

Reproducible builder: `tools/build_dex3_rev1_descriptors.py`.
Numerical audit: `artifacts/dex3_rev1_descriptor/audit.json`.
Visual gate: `docs/assets/dex3_rev1_descriptor_states.png`.

The descriptor visual gate was subsequently accepted for raw inference. This
acceptance is not a claim about closure, contact, reachability, or execution.

## 2026-07-18 — actual AprilCube parts and raw GraspGenX candidates

Pinned `sri299792458/aprilcube` at
`fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5` and generated the task's actual
T body, U legs, and cube head from project-owned YAML specs.

- The first 30 mm voxel revision was rejected after the exact open-Dex3 render
  showed that the objects were visibly underscaled. The physical design now
  uses the normal 45 mm AprilCube scale, 36 mm AprilTag 36h11 markers, and a
  3 mm edge radius.
- T: 6 occupied voxels, 135 × 45 × 180 mm, tag IDs 0–25.
- U: 7 occupied voxels, 135 × 45 × 135 mm, tag IDs 64–93.
- cube: 1 voxel, 45 × 45 × 45 mm, tag IDs 128–133.
- The complete no-gap figure is 360 mm tall. Magnet pockets are intentionally
  absent from this geometry revision.
- The textured AprilCube OBJ is retained for rendering/perception but is
  materially partitioned into marker patches. A second mesh builder repeats
  AprilCube's rounded manifold union without material partitioning and exports
  clean watertight positive-volume `grasp_mesh.obj` files for point sampling.
- Left/right descriptors have identical canonical conditioning, so the neural
  model generates one hand-neutral `object_T_G` set per part. Physical palm
  conversion and arm assignment remain side-specific downstream.
- The released model was rerun on the 45 mm meshes. It generated 240 raw
  proposals per part and retained 20. Best confidences were T `0.991`, U
  `0.978`, and cube `0.879`.
- The initial audit mistakenly rendered the fixed terminal-close vector through
  each object and used that penetrative result to judge the U proposal. That
  conclusion was unsupported and has been retracted. The replacement audit
  shows only the open hand and returned frame.

Visual gate: `docs/assets/aprilcube_raw_grasp_audit.png`.
Provenance: `artifacts/aprilcube_raw_grasps/provenance.json`.

The 45 mm physical scale is now established and rendered beside the exact open
Dex3 in `docs/assets/aprilcube_45mm_scale.png`. Stop before MoveIt. Next
implement a deterministic candidate qualifier that stops fingers at first
contact, checks prohibited-link penetration and opposing contacts, and renders
explicit pass/fail reasons.

## 2026-07-20 — clean locked simulation environment

Started the simulation/contact phase in the new repository rather than reusing
the old demo environment.

- Installed `uv 0.11.29` at `~/.local/bin/uv`; shell profiles were not changed.
- Added a Python 3.11 root project and reproducible lock contract in
  `pyproject.toml`, `.python-version`, and `uv.lock`.
- The selected compatible core stack is GraspGenX at the pinned local
  submodule, PyTorch `2.6.0+cu124`, Newton `1.0.0`, Warp `1.15.0`, MuJoCo
  `3.5.0`, and MuJoCo-Warp `3.5.0.2`.
- `uv pip check` reports all 153 installed packages compatible. GraspGenX
  imports from this repository's checkout, and PyTorch/Warp see both RTX A5500
  GPUs.
- `tools/check_sim_stack.py` exercises the same architectural path used by
  GraspGenX dynamic playback: Newton `CollisionPipeline` produces contacts and
  Newton `SolverMuJoCo` delegates the solve to MuJoCo-Warp. A 50 mm-radius
  sphere dropped from 300 mm settled at 48.83 mm instead of falling through
  the ground plane.
- Restored the old `/home/srinivas/Desktop/demo/.venv/graspgenx` environment to
  its observed pre-install state: MuJoCo `3.10.0`, with Newton, Warp, and
  MuJoCo-Warp absent.

This establishes the environment only. It does not yet qualify any Dex3 grasp.

## 2026-07-20 — untouched GraspGenX end-to-end baseline passes

Ran the pinned upstream Franka single-object example without source or physics
parameter changes. The complete GraspGenX -> scene collision filter -> cuRobo
-> Newton/MuJoCo-Warp path succeeded and dropped 1/1 objects into the bin with
zero retries.

- The first two attempts exposed setup faults rather than grasp failures: the
  Franka collision OBJ was an unmaterialized Git-LFS pointer, then cuRobo's
  editable `--no-deps` installation lacked `cuda.core`.
- Materialized the gripper-description LFS assets and installed GraspGenX's
  declared `cuda-core[cu12]>=0.7,<1.0` range. The accepted run used
  `cuda-core 0.7.0`; `uv pip check` reports all 165 packages compatible.
- 36 scene-collision-free candidates reached cuRobo. It selected original
  GraspGenX candidate 7 at confidence `0.783`.
- Newton reported a `0.196 m` lift at the lift checkpoint. The exported
  trajectory's peak rise was `0.240 m`; its final XY error from bin center was
  `0.053 m, 0.039 m`.
- The 1,532-frame trajectory and 25.5 second visual confirm physical closure,
  carrying, release, and final settling. See
  `docs/graspgenx_newton_baseline.md` and
  `docs/assets/graspgenx_franka_newton_baseline.mp4`.

This validates the reference pipeline, not Dex3. GraspGenX has no shipped
G1/Dex3 end-to-end profile. The next adapter must add current-Dex3 per-joint
gain maps and compare an explicitly named upstream-default profile against a
separate VIRAL-derived profile (`thumb_0 2.0/0.1`, other six `0.5/0.1`) in the
same Newton scene. Do not mix those gains into the Franka reference.

Added `config/dex3_newton_control_profiles.yaml` with those two profiles and
their separate provenance. The VIRAL profile also records its 200 Hz physics,
4-step control decimation, raw per-joint effort limits, and side-specific
armatures after the source Isaac Sim adapter's explicit `x3` scaling. The
values match the pinned source programmatically. The structural validator
confirms exact seven-joint coverage for both current left/right descriptor
URDFs while explicitly reporting that neither profile is physics-qualified.

## 2026-07-20 — controlled current-Dex3 tabletop test restored

Restored the originally agreed first Dex3 experiment after an unhelpful
floating-hand/zero-gravity detour. The controlled scene is one current right
Dex3 hand, the actual generated 45 mm AprilCube cube, a flat table, and normal
gravity. There is no Franka/bin scene, G1 arm/body, chair, magnet, MoveIt,
cuRobo, or assembly logic in this calibration test.

The intended sequence at 60 recorded frames/s is:

1. let the cube settle on the table;
2. hold the exact open hand at a collision-free pregrasp;
3. follow the selected GraspGenX approach to `world_T_G`;
4. hold open at the grasp pose for one second;
5. close the seven Dex3 joints over 20 frames (~0.33 s);
6. hold closed for one second;
7. lift the prescribed hand base by 0.20 m over four seconds;
8. hold elevated for one second.

The pregrasp is searched backward along the candidate's own approach axis and
must clear the exact open-hand collision mesh, cube, table, and sampled linear
approach corridor. No fixed 14 cm offset is used. A pass requires the cube to
rise with the hand, stay stable through the final hold, avoid numerical
ejection, and avoid collision by palm/proximal links that are not intended
contacts. After a cube profile passes, repeat with actual T/U candidates and
both hands.

### Upstream boundary

This remains an adaptation of the vendored GraspGenX end-to-end code under
`third_party/GraspGenX`, not an independent simulator. The implementation
reuses upstream `run_graspgen`, checkpoint loading, `scene_builder`,
`DynamicSession`, its Newton `CollisionPipeline` + MuJoCo-Warp solver/contact
settings, trajectory JSON exporter, and `render_trajectory_mp4.py`. Project
additions are limited to the current-Dex3 descriptor/profile, the tabletop
cube environment, candidate/experiment orchestration, and prescribed motion
of a hand-only fixed root.

A separate custom static-preview renderer was judged unnecessary bloat. The
single controlled evaluator should run Newton and derive its video and review
frames from the same exported simulated trajectory. Geometric qualification
still runs before stepping physics, but it is a validation stage inside that
one evaluator rather than a second visualization pipeline.

### Candidate/table result

The first deterministic inference used seed 17, 3,500 object points, 480 raw
samples, and the top 60 discriminator-ranked candidates. Nine of those 60 met
the requested top-down score (`-G.z_world >= 0.85`), but every one of those
nine placed the exact open current-Dex3 collision mesh in the table. Only one
of the top 60 was both cube- and table-clear; it was a side approach, not
top-down. This is a meaningful geometry result: do not silently send a
table-intersecting top-down pose into Newton merely to preserve the original
sketch. The evaluator should prefer top-down when feasible and otherwise
explicitly record any fallback to the best fully qualified GraspGenX
candidate.

### Newton adapter faults found and fixed

- `dynamic_playback.py`'s deterministic-inertia branch assigned NumPy arrays
  into Newton's typed `ModelBuilder.body_inertia/body_com` lists. This made
  `builder.finalize()` fail with a heterogeneous-array error. The same values
  are now assigned as `wp.mat33` and `wp.vec3`.
- Upstream dynamic playback hard-coded `collapse_fixed_joints=True`. On the
  standalone Dex3 descriptor this removes the moving palm and exposes thumb,
  middle, and index as three separate world-rooted articulations. The importer
  option is now profile-controlled; the hand-only Dex3 profile sets it to
  `false`, preserving one canonical fixed root, the palm body, and all three
  connected finger chains.
- With the palm preserved, MuJoCo rejected its inertia. Unitree's official
  palm inertia is valid but is expressed in a rotated inertial frame; Newton
  1.0's URDF import produced a non-symmetric matrix for that case. The
  descriptor generator now performs the equivalent basis change
  `I_link = R * I_inertial * R^T` and writes zero inertial-frame RPY. Mass,
  COM, and physical inertia are unchanged. Both descriptors were rebuilt and
  the complete hand-only Newton model now constructs successfully.
- `DynamicSession` now records the robot's fixed root joints and supports a
  prescribed base transform through Newton/MuJoCo's supported mocap update:
  update `model.joint_X_p`, notify `SolverNotifyFlags.JOINT_PROPERTIES`, and
  preserve each root's local transform. Per-frame exported hand meshes use
  this same moving base transform. This is the minimal missing capability for
  the arm-free approach/lift isolation test.

The generic GraspGenX control profile remains the first run: 1 ms requested
physics step, collision refresh every four substeps, 100 solver iterations,
50 line-search iterations, `impratio=1000`, finger `kp=2000`, `kd=200`, and
the existing explicit contact/object settings. Only after that exact baseline
runs should the otherwise identical scene be repeated with the recorded VIRAL
per-joint gains, effort limits, and armatures.

Current status: the descriptor and Newton model construction faults are fixed,
but no Dex3 cube grasp simulation has completed and no Dex3 grasp has passed.

### Correction after the first completed current-Dex3 simulations

The earlier top-60 table-collision count was produced with one merged visual
mesh and was too conservative. Candidate qualification now uses cached,
per-link collision elements from the exact descriptor URDF—the same collision
geometry family imported into Newton. Across the top 240 candidates, 30 meet
the nominal top-down score and 27 of those collide with the table. The three
remaining table-clear top-down proposals do not place the cube in the
descriptor's open/mid sweep volumes. Therefore there is currently no
top-down proposal that is both capture-qualified and table-clear for the
45 mm cube in this flat-table scene. The evaluator records an explicit side
approach fallback rather than modifying a neural pose.

The first table-clear side candidate selected only by confidence was candidate
148. It had zero meaningful sweep-volume occupancy: in canonical G its cube
center lay at approximately `[0.008, 0.004, 0.163] m`, beyond the descriptor's
finger sweep. In the full 560-frame Newton run the close ejected the cube;
measured final lift rise was `-0.491 m`, hand-relative drift was `4.54 m`, and
maximum one-frame cube translation was `0.214 m`. This exposed a missing
candidate-contract check, not a controller-tuning result.

The evaluator now samples the actual cube surface and checks it against the
same open and mid sweep boxes used to condition GraspGenX. The default gate is
at least 10% surface occupancy in each box and 20% in their union. It remains
a coarse descriptor-consistency gate, not a claim of physical force closure.
Pregrasp clearance is also separated into 30 mm from the object and 2 mm from
the table. Requiring 30 mm from the table made every valid horizontal approach
impossible because backing away horizontally cannot increase vertical table
clearance.

With those corrections, five of the deterministic top 240 candidates pass the
open-hand, table, capture-volume, and complete linear-corridor checks: 149,
158, 182, 189, and 205. Candidate 149 is the highest-confidence member. It is
a side approach (confidence `0.579`) with a computed 5 mm pregrasp, 32.9 mm
object clearance, and 6.9 mm table clearance. The full upstream-timing Newton
run completed, but it also failed physically: the 20-frame close pushed the
cube sideways and the hand lifted alone. Metrics were 0.34 mm cube rise,
207 mm maximum hand-relative drift, 0.68 mm final-hold drop, and 43.7 mm
maximum one-frame cube motion. Visual evidence is in
`artifacts/dex3_tabletop_generic_capture_filter/simulation.mp4` and
`phase_contact_sheet.png`.

This failure demonstrates why neither a static render nor sweep occupancy is
the final grasp test. Before changing gains, the same Newton scene and exact
timing must evaluate all five geometrically qualified GraspGenX candidates.
Candidate identity can now be requested explicitly, but only after it passes
the common geometric gate. The first physics-passing candidate—if one
exists—will be the selected baseline and the only one rendered as the accepted
result. If none passes, the next question is the current-Dex3 descriptor/open-
close trajectory contract, not MoveIt or assembly code.

### Immediate retraction: physics admission must not be heuristic

The previous paragraph's plan to send only five sweep/collision-qualified
candidates to Newton is retracted. It repeated the same conceptual mistake in
a more elaborate form: a project-written occupancy or approach heuristic was
still deciding which neural outputs were permitted to receive the real test.
The sweep-occupancy numbers remain useful diagnostics for explaining a result,
but they are not an admission filter and must not select the grasp.

The corrected contract is:

1. GraspGenX inference returns the candidate poses and discriminator scores.
2. Every returned candidate is placed in an otherwise identical Newton world.
3. Every world receives the same actual current-Dex3 model, 45 mm AprilCube,
   table, gravity, open/close trajectory, simulation parameters, and motion
   timing.
4. Newton contact dynamics determines which candidates close on, retain, and
   carry the cube. A render, sweep box, top-down score, table-distance query,
   or hand-written pose correction cannot remove a candidate first.
5. Only after the complete physical result table exists may discriminator
   confidence break ties among physics-successful candidates.

This is also the architecture used by NVIDIA's released `GraspDataGen`:
candidate grasps are replicated into many physics environments and validated
in batches (its implementation uses PhysX/Isaac Lab and allows up to 1,024
environments). For this project the equivalent mechanism is Newton 1.0's
`ModelBuilder.replicate`, which preserves isolated collision worlds while
Warp executes them on the GPU. We will process the full GraspGenX output in
GPU-sized chunks and rerun only the physics-selected winner through the same
Newton exporter for the review video. This is not a second preview path.

The partially rendered candidate-158 run was stopped as soon as this boundary
was corrected. Its output is not a result and must not be used for selection.

### All-candidate Newton evaluator implemented

`third_party/GraspGenX/end2end/hand_only_grasp_eval.py` now implements the
corrected boundary. It passes every pose actually returned by
`run_graspgen(...)` directly into Newton. There is no render gate, sweep-box
gate, top-down rule, table-distance rule, collision admission query, manually
selected candidate, or manually altered grasp transform. The discriminator
score is retained as metadata and may only rank candidates after physics has
reported successful retention.

Each candidate receives an isolated copy of the same hand/object/environment
world using Newton's `ModelBuilder.replicate`. The reusable world template is
built once, including the exact current right-Dex3 descriptor, the generated
45 mm AprilCube mesh, table, collision decomposition, material parameters,
gravity, and solver settings. Candidate batches copy that one template rather
than rebuilding collision geometry, so batch boundaries cannot silently
change the tested scene. The only per-world difference is the GraspGenX pose.

Every world then receives the same prescribed experiment: settle the cube,
move the open hand from the common upstream-style local-Z pregrasp offset to
the returned grasp pose, hold open, command the descriptor's close trajectory,
hold closed, lift the hand by 0.20 m, and hold again. Newton simulates all
contacts and the cube trajectory. The result file records the complete pose,
confidence, cube lift, cube-to-hand relative drift, final-hold drop, numerical
health, and the resulting retention decision for every candidate. Only after
that table is complete is one physics result replayed through the same Newton
scene/exporter to produce a review video. If no candidate retains the cube,
the replay is explicitly labelled a failure diagnostic rather than a selected
grasp.

The first batching smoke test also exposed an important Newton indexing issue.
A floating root has seven position coordinates (three translations plus a
quaternion) but six velocity/control coordinates. Reusing `joint_q` indices
for `joint_target_pos` therefore shifted later replicated worlds and placed
the last finger target one element beyond the controller array. The evaluator
now obtains finger state indices from each finalized replica's
`joint_q_start`, and controller indices independently from `joint_qd_start`.
No world-stride assumption remains.

The corrected compressed smoke test asked upstream GraspGenX for four grasps.
Its built-in minimum-grasp retry behavior returned 24 poses over six
iterations, and all 24 were simulated—not truncated back to four. They ran in
three eight-world batches; every batch contained exactly 80 bodies, 80 joints,
and 2,024 shapes, all 24 result indices were present, and no world produced a
NaN. The deliberately compressed 18-frame motion yielded no accepted grasp,
as expected; it validates batching and accounting only and is not grasp
evidence. Its audit output is
`artifacts/dex3_all_raw_newton_smoke_cached3/physics_results.json`.

The first full-timing run used that same deterministic 24-pose inference
result. All 24 candidates completed the 560-frame Newton experiment in three
identical eight-world batches with no NaNs. None passed the physical
lift/retention criteria. Candidate 16 was replayed only as an explicitly
labelled `no_physics_pass_diagnostic`; it did not grasp the cube. Its contact
launched the cube out of the camera view before the empty hand lifted, which
is also reflected by 22.95 m apparent cube rise, 63.33 m hand-relative drift,
and a 0.555 m one-frame cube displacement. Those values reject the candidate
as an ejection; they are not evidence of lift success. The complete result
table and Newton replay are in
`artifacts/dex3_all_raw_newton_generic_24/physics_results.json` and
`artifacts/dex3_all_raw_newton_generic_24/simulation.mp4`.

This result means only that none of this 24-candidate sample retained the cube.
It does not justify a geometric prefilter and does not yet establish that the
full 480-candidate inference pool fails. A temporary 64-world compressed test
also completed successfully on the A5500 (640 bodies, 640 joints, 15,808
shapes), confirming that larger GPU batches are structurally possible. Batch
size is therefore only a throughput/memory parameter; it must never alter the
candidate set or outcome contract.

### Production run stopped: environment contact was missing from success

The first 480-candidate production run was stopped during its second batch
after reviewing the earlier candidate-16 diagnostic more carefully. The hand
visibly collides with the table. That candidate was never marked as a grasp
success—the retention metrics rejected its cube ejection—but the visual
exposed a missing part of the evaluator's physical success contract.

The current hand-only scene prescribes the grasp-frame trajectory by moving a
kinematic hand root. Newton computes hand-table contacts, but the table cannot
push that prescribed root away or stop its commanded approach. Consequently,
a table-infeasible raw grasp can be driven into the table and create very large
contact impulses. Checking cube retention alone is insufficient because a
different table-colliding candidate might conceivably retain the cube after an
unphysical forced approach.

The correction must not reintroduce a geometric prefilter. Every GraspGenX
candidate must still enter Newton. Newton's simulated contact result must
instead report at least two independent physical outcomes for every world:

1. whether any Dex3 collision shape contacted the table during approach,
   closure, or lift; and
2. whether the cube stayed with the hand through lift and final hold.

A task-valid tabletop grasp requires retention **and** no hand-table contact.
This is a post-simulation contact outcome, not a render judgment or a
candidate-admission heuristic. A raw candidate that contacts the table remains
in the complete result table with an explicit physical failure reason. The
evaluator must also stop choosing the largest apparent cube rise as its
no-success replay because a numerical/contact ejection can maximize that
quantity. If no candidate passes, it must report no selected grasp; any video
must be labelled by its specific failure reason rather than presented as the
best candidate.

No result file was produced by the interrupted 480-candidate run, and it must
not be resumed or interpreted until the Newton hand-table contact measurement
is implemented and validated.

### Deeper correction: the grasp test and tabletop motion test were conflated

The preceding correction is necessary but incomplete. Candidate 16 was not
merely a lateral hand that happened to touch the table. It was a bottom-up raw
grasp. Its canonical grasp-frame origin was at `z=0.368 m` while the tabletop
was at `z=0.500 m`; its local `+Z` axis was nearly world-up. Applying the copied
cuRobo `-0.10 m` local-Z pregrasp therefore placed the hand still lower and
commanded it upward through the table.

Across the deterministic 24-candidate sample, the raw approach orientations
were 8 bottom-up, 12 lateral, and 4 top-down. Eleven of 24 grasp-frame origins
were below the tabletop. This is expected from object-only GraspGenX inference:
it proposes final object-relative grasps in free space and does not know about
the supporting table. The original downstream cuRobo path couples the local-Z
offset to collision-aware planning; copying the offset while replacing that
planner with an unstoppable prescribed hand-root trajectory was incorrect.

The 24-candidate tabletop result must therefore be discarded as evidence about
GraspGenX grasp quality or the current Dex3 descriptor. It combined three
different questions into one invalid experiment:

1. Is the returned object-relative grasp physically stable for this hand?
2. Is the final grasp and approach compatible with the table?
3. Can the robot arm reach and execute that approach?

The upstream NVIDIA GraspDataGen validator keeps question 1 isolated: gravity
is disabled, the gripper is fixed, the object is placed at the candidate's
relative transform, the fingers close, and external tug forces are applied in
multiple directions. It does not insert a table or turn the grasp pose into an
unplanned approach trajectory. Contact sensors and object-relative motion
determine success.

The correct next baseline is to reproduce that intrinsic validation contract
in Newton for the exact current Dex3 and 45 mm cube. Every GraspGenX candidate
still enters physics. Only after stable candidates exist should tabletop scene
feasibility and MoveIt/OMPL approach planning be evaluated as a separate layer.
This separation is not a heuristic filter; it restores the responsibilities
that the previous test accidentally collapsed.

### Stop inventing the missing middle: use the released end-to-end path

The proposed custom free-space Newton validator is not the next project step.
It would be another replacement for functionality already released around
GraspGenX. The repository's own `end2end/README.md` defines the intended stack:

`GraspGenX -> scene collision filtering -> cuRobo plan_grasp -> Newton replay`.

`e2e_grasp_demo.py` already implements candidate-frame conversion, robot-base
conversion, goal-set construction, local-tool-frame approach offsets,
collision-aware `plan_grasp`, chosen-candidate indexing, approach/grasp/lift
trajectory extraction, task sequencing, Newton/MuJoCo dynamic playback, and
trajectory/video export. The hand-only evaluator copied pieces of this
contract and omitted the planner. It must not remain an implementation path.

The installed upstream cuRobo checkout also already ships the exact relevant
robot assets:

- `curobo/content/configs/robot/unitree_g1.yml`;
- `robot/g1/g1_29dof_with_hand_rev_1_0.urdf`;
- current left/right Dex3 meshes and collision spheres; and
- Unitree G1 coverage in cuRobo's IK benchmark and reactive-control example.

Therefore the next implementation boundary is declarative adaptation only:
an end-to-end GraspGenX robot YAML referencing the shipped `unitree_g1.yml`,
the chosen palm as `tool_frame`, the descriptor-established grasp-to-palm
transform, the known seated start configuration/base placement, and the
AprilCube table scene YAML. We should exercise the existing
`e2e_grasp_demo.py` entry point before adding or changing any planning or
physics algorithm.

This also retracts the rule that every raw inference pose must be replayed in
Newton for the end-to-end pick. In the released architecture, all generated
poses may enter the collision/planning candidate set, but only a
collision-free, reachable grasp selected by cuRobo receives an executable
trajectory and Newton replay. Newton validates that planned execution; it is
not a replacement for scene-aware motion planning.

### Scope clarification: do not integrate the seated scene yet

The released end-to-end path above is the correct later pickup pipeline, but
it is not the immediate milestone. The current unanswered question is
narrower: does the exact current-Dex3 embodiment physically retain the 45 mm
cube at poses returned by GraspGenX? That test needs the descriptor/hand, cube
mesh, candidate transforms, closing command, and a released grasp-physics
validator. It does **not** need the G1 arms, seated joint state, robot base,
chair, table, or motion planner.

The table and seated G1 configuration enter only after intrinsic grasp
retention is demonstrated. At that point the released collision/planning path
answers whether a retained grasp is reachable in the actual tabletop scene.
Do not collapse these milestones again.

## 2026-07-20 — cuRoboV2 execution and attachment audit

The first demo has a specified sequence: one hand holds the T, the worker
attaches U, the holder's carried model becomes T+U, the worker picks and
attaches the cube, and the holder's model becomes T+U+cube before place and
detach. The implementation problem is reliable motion and explicit scene-state
transitions, not task-sequence search.

The motion/attachment side was tested against official cuRoboV2 v0.8.0 commit
`4ea77366ca48ee453e7df139e39fa6532af49f3b`. Its upstream attachment tests (18)
and motion-planner tests (72) passed. A separate exact-G1 contract probe loaded
the shipped rev-1.0 G1/Dex3 model and verified multi-tool planning, two
independent named attachment slots, planning with both slots active, explicit
detach of one without disturbing the other, and successive holder-model
updates for T -> T+U -> T+U+cube. It also verified that disabling a named
target world object leaves the table enabled. The probe used safe synthetic
spheres and small motions; it establishes API capability, not actual demo
reachability or grasp success.

Important v0.8 adapter constraints were also found. The public
`MotionPlanner.attachment_manager` convenience property is broken in the tag,
the underlying manager's no-argument bookkeeping assumes one attachment, and
its world-pose-offset helper always uses the first tool frame. We must access
the underlying manager deliberately, always name the hand/object on update and
detach, and supply collision spheres already expressed in the correct
hand-specific attachment-link frame. Collision permissions are object- or
link-wide rather than pair-specific. For final grasp contact, disable only the
target object's world copy; never globally disable the Dex3 contact links,
because doing that would also hide the table from those links.

Decision: use cuRoboV2 as the G1 motion backend and write a thin fixed-sequence
executor that owns explicit scene effects (`attach`, `snap_and_transfer`,
`replace_composite`, `place_and_detach`). Keep GraspGenX as the candidate
generator, Newton as the offline physical grasp qualifier, and add the ROS 2
hardware bridge later. Do not add task-sequence search or MoveIt merely for
sequence authoring. Full evidence and the next visual checkpoint are in
`docs/execution_stack.md`.

## 2026-07-21 — RETRACTED: current-Dex3 cube grasp validation

This entry is retained as an audit trail, but its validation conclusion and
pass/fail visuals must not be used. The imported hand USD had intra-hand
self-collision enabled, unlike the released GraspGenX Newton playback. During
finger closure, overlapping or adjacent hand collision shapes generated
unstable impulses. Batched copies could eject the cube metres away. The
corrected experiment is recorded in the next entry.

The immediate intrinsic-grasp milestone is complete for the right hand. The
validator uses the exact current official Unitree Dex3-1 right URDF and meshes
from `xr_teleoperate` commit
`7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`. The generated
`dex3_rev1_right` GraspGenX descriptor and the Isaac USD both derive from that
same geometry. This is not the older hand bundled with the original GraspGenX
demo.

The object is the actual generated 45 mm AprilCube mesh at
`generated/aprilcube_parts/cube_head/grasp_mesh.obj`. Its watertight volume is
90.10 cm3. The validator uses 0.12 kg: a conservative value for a fully solid
PLA print (approximately 0.112 kg) plus marker/connector allowance. Replace
this value with the measured finished-part mass before hardware calibration.
The upstream GraspDataGen default was a hard-coded 1.0 kg, which explained the
earlier 0/120 result and was not representative of this printed cube.

`tools/build_dex3_isaac_grasp_input.py` copies every GraspGenX
`object_T_grasp` transform verbatim. It adds only the exact current-Dex3 open
and close joint dictionaries and the object/validator configuration. Each
entry now preserves its original candidate ID, neural confidence and neural
transform under `graspgenx_source`; GraspDataGen is then free to store its
binary physics result and final post-tug transform without destroying
provenance.

The released GraspDataGen/Isaac/PhysX validation contract is preserved:

- 250 Hz physics;
- one second for the fingers to close and settle;
- five 0.5-second object-force tugs in `+Z`, `+Y/+Z`, `-Y/+Z`, `+X/+Z`, and
  `-X/+Z` directions;
- each direction normalized by the upstream parser and applied at one object
  gravity-equivalent (`mass * 9.81`);
- success only when both configured opposing contact sensors still report
  nonzero force after every tug.

All 120 raw GraspGenX candidates entered a single parallel physics batch.
Fifteen passed: `17, 25, 34, 43, 47, 48, 54, 64, 66, 74, 87, 104, 110,
112, 116`. The highest neural-ranked passing proposal is `grasp_17`, with
GraspGenX confidence `0.7723388671875`. The qualified result is
`artifacts/isaac_grasp_validation/dex3_cube_mass012_final_v2/grasp_sim_data/dex3_rev1_right/grasp_mesh.yaml`.

Candidate 17 was rerun alone with an Isaac Lab RGB camera. After the full tug
sequence, the two monitored contact-force vectors were approximately
`[0.678, 0.621, -1.449] N` and `[-1.288, -1.214, 1.929] N`; the same run was
recorded as a physics success. Visual styling uses USD Preview Surface
bindings and directional lighting only; it does not author or change any
physics material, mass, collision or actuator property.

Visual evidence:

- sequential close-up review:
  `docs/assets/dex3_cube_review10_sequential_closeups.mp4` (1280x960,
  30 fps, 45.33 s). It presents the same ten candidates one at a time for
  their complete closure-and-tug runs, labeled with candidate ID and the
  final physics verdict;
- close-up pass video:
  `docs/assets/dex3_cube_grasp17_isaac_validation.mp4` (960x720, 30 fps,
  4.53 s);
- readable comparison video:
  `docs/assets/dex3_cube_review10_isaac_validation.mp4` (1600x960, 30 fps,
  4.53 s). It contains the five highest-confidence failures (`0-4`) and five
  highest-confidence successes (`17, 25, 34, 43, 47`), with final red/green
  borders taken directly from the physics result;
- complete 120-candidate overview, retained for audit although too dense for
  routine viewing: `docs/assets/dex3_cube_all120_isaac_validation.mp4`;
- final candidate-17 still:
  `docs/assets/dex3_cube_isaac_grasp17_pass.png`.

The RGB frames come from Isaac Lab cameras in the same validator run, not from
an independently reconstructed renderer. Headless mode suppresses only the
GUI window; `--enable_cameras` still loads Isaac's RTX rendering experience.
It was used because the local headed viewport path produced black frames and
Vulkan `DEVICE_LOST` crashes.

Scope limit: this proves intrinsic simulated retention for the exact hand,
cube, controller profile and assumed cube mass. It does not yet prove table
clearance, G1 arm reachability, pickup execution, hardware grip reliability or
magnetic assembly. Those remain later planning and hardware milestones.

## 2026-07-21 — self-collision regression isolated; matching-hand A/B rerun

The low-success investigation first returned to the released GraspGenX
`unitree_g1` descriptor and its matching older Unitree hand geometry. The
upstream two-finger validator is deliberately batched, so serial execution was
not accepted as a fix. Instead, `grasp_0` was duplicated ten times in one
Isaac/PhysX batch.

With our original imported USD, ten identical inputs evolved into different
physical states: several cubes were ejected metres away and only one retained
the cube. The result remained unstable when the custom explicit `q_close`
target path was bypassed in favor of GraspDataGen's untouched joint-limit
target branch. This cleared the batching kernel and explicit target code.

The regression was in `tools/import_dex3_isaac_asset.py`: it had set
`self_collision=True`. GraspGenX's released Newton playback passes
`enable_self_collisions=False` when it imports the robot. Reimporting the same
older Unitree URDF with only intra-hand self-collision disabled made all ten
identical environments pass. The retained final object poses and joint states
vary slightly from GPU contact nondeterminism, but no object was numerically
ejected and the binary result was 10/10.

Both canonical Isaac hand assets have now been regenerated with
`self_collision: false`. Hand-object collisions remain enabled; this setting
only prevents collision shapes belonging to the same hand articulation from
applying forces to one another. The importer keeps an explicit
`--self-collision` diagnostic override, but its default now matches the
released GraspGenX Newton contract.

The root USD file alone was an insufficient provenance key because Isaac's
converter stores articulation physics in referenced files under
`configuration/`. The root file hash therefore did not change when the
self-collision policy changed. `tools/build_dex3_isaac_grasp_input.py` now
also writes `gripper_usd_package_sha256`, which hashes the root USD and all of
its generated USD sublayers.

Corrected, one-batch, 120-candidate **pipeline** comparison using the same
45 mm cube, object point sample/seed, released checkpoints, 0.12 kg mass,
controller gains, explicit descriptor close configuration, tug sequence, and
requirement for object contact in at least two of three finger chains:

- released matching older Unitree hand and released descriptor: 110/120;
- current Dex3-1 hand and our regenerated current-hand descriptor: 16/120.

This is deliberately not a pose-matched comparison of hand mechanics. Each
descriptor conditions GraspGenX separately, so it produces a different ranked
set of `object_T_G` transforms and the two hands visibly approach the cube in
different ways. Also, the canonical `G` frame has a different fixed mapping to
each physical hand. Applying the same numeric `object_T_G` to both assets would
therefore not create the same object-relative palm pose. The comparison says
that the complete released hand+descriptor pipeline is healthy while our
current-hand hand+descriptor pipeline remains weak; by itself it cannot assign
that weakness specifically to inference, the descriptor geometry/frame, or
the current hand's closure mechanics.

The current-hand run was repeated from a fresh Isaac process. It again
produced exactly the same 16 passing candidate IDs: `17, 25, 26, 34, 43, 47,
48, 54, 64, 66, 74, 87, 104, 110, 112, 116`. This establishes that the
remaining current-hand drop is not the self-collision regression and is not a
failure of upstream batching. It points back to the current-hand
descriptor/checkpoint contract and must be investigated before calling the
current Dex3 grasp pipeline complete.

Evidence:

- identical-input failure with self-collision enabled:
  `artifacts/isaac_grasp_validation/original_unitree_cube/repeat10_explicit_target/grasp_sim_data/unitree_g1_original/grasp_mesh.yaml`;
- identical-input 10/10 control with self-collision disabled:
  `artifacts/isaac_grasp_validation/original_unitree_cube/repeat10_no_self_collision/grasp_sim_data/unitree_g1_original_no_self_collision/grasp_mesh.yaml`;
- corrected released matching-hand result:
  `artifacts/isaac_grasp_validation/original_unitree_cube/physics_no_self_collision/grasp_sim_data/unitree_g1_original/grasp_mesh.yaml`;
- corrected current-hand results:
  `artifacts/isaac_grasp_validation/current_dex3_cube/physics_no_self_collision/grasp_sim_data/dex3_rev1_right/grasp_mesh.yaml` and
  `artifacts/isaac_grasp_validation/current_dex3_cube/physics_no_self_collision_repeat/grasp_sim_data/dex3_rev1_right/grasp_mesh.yaml`.

## 2026-07-21 — current-Dex3 descriptor root cause and corrected contract

The descriptor investigation is now separated into evidence, rejected
hypotheses, and the contract we will use. This entry supersedes the claims in
the 2026-07-18 “current Dex3 descriptors, visual gate” entry; that earlier
entry remains above only as an audit trail.

### Retraction

The earlier statement that the upstream GraspGenX wizard “incorrectly” chose
canonical Z as the current hand's open closing axis was false. Using the exact
current terminal meshes, the upstream estimator reproducibly finds its largest
open terminal separation on current-frame Z. That is a real consequence of
the current zero-joint hand's L-shaped open posture.

The mistake was ours: after observing that L-shaped geometry, the first builder
created a visually fitted axis-aligned vector

```text
open: [0.05, 0.06, 0.10] at [-0.02858, 0, 0.074]
half: [0.04, 0.06, 0.06] at [-0.00458, 0, 0.091]
```

while still claiming it preserved the released checkpoint's convention. It
did not. In the pinned GraspGenX code, the `sweep_volume_v2` networks consume
only these 12 numbers. They do not see the current URDF, finger meshes, joint
positions, or trajectory. X is used as aperture/width and Z as
approach/fingertip depth; our first vector encoded a 50 mm X aperture and a
100 mm Z depth. It was a literal-looking box around an asymmetric posture but
a bad learned-morphology input.

### Isolation experiment 1: cross proposals and physical hands

One 120-environment Isaac/PhysX batch was run for every pairing:

```text
released proposals -> released hand     110/120
released proposals -> current hand      118/120
first-current proposals -> released hand  5/120
first-current proposals -> current hand  16/120
```

The current physical hand succeeds on the released proposal distribution; the
first-current proposals fail on both hands. This isolates the large regression
to the descriptor-conditioned candidate distribution rather than current-hand
contact mechanics or batched physics.

Evidence is under `artifacts/isaac_grasp_validation/cross_matrix/`, with the
matching-hand controls under the adjacent `current_dex3_cube/` and
`original_unitree_cube/` directories.

### Isolation experiment 2: complete four-group factorial

All 16 combinations of current versus released `extents`, `offset`,
`extents2`, and `offset2` were inferred with the same seed and evaluated in a
single 1,920-environment physics run. Mask bits are ordered by those four
groups, with `1` selecting released:

```text
0000  16    0001  64    0010  10    0011  56
0100  15    0101  65    0110  10    0111  58
1000 111    1001 116    1010 111    1011 118
1100 108    1101 116    1110 112    1111 118
```

Replacing only the open extents (`1000`) raises retention from 16/120 to
111/120. Replacing only the half-open center (`0001`) raises it to 64/120.
The dominant defect is therefore the open-box conditioning, with an additional
half-open-center effect. Complete inputs, raw outputs, provenance, and physics
results are in `artifacts/dex3_sweep_ablation/`.

### Rejected explanation: a simple canonical-frame rotation

The exact open terminal centroids imply a 53.1326-degree Y rotation that aligns
the thumb-to-opponents separation with X. `tools/run_dex3_frame_ablation.py`
ran inference in that frame and converted every pose back into the unchanged
current execution frame before physics. Results were:

```text
rotation only                         0/120
rotation + XY centering               0/120
rotation + XY centering + Z=70 mm    31/120
```

This rejects that specific, straightforward frame realignment as the fix. It
does not establish that no other descriptor/frame redesign could work.

### Current-geometry semantic check and fresh seeds

A second vector placed the wizard-measured current open and half-open gaps
(84.728 mm and 38.543 mm) in the descriptor's X aperture slots while retaining
the released transverse/depth dimensions and offsets. It was compared with the
exact released Unitree 12-vector on the exact current hand:

```text
seed       19    29    39    49      aggregate
current aperture semantic vector
          116   115   118   118     467/480 (97.29%)
released Unitree vector
          118   119   115   120     472/480 (98.33%)
```

The five-grasp aggregate difference is small and sampling-dependent. Both
results confirm the diagnosis: the failed first descriptor supplied the wrong
learned semantics; the current physical hand and closing trajectory are able
to retain properly conditioned candidates.

### Selected contract and its exact limitation

The manifest now uses the released Unitree vector unchanged:

```text
open: extent [0.10, 0.06, 0.04], center [0.000, 0.000, 0.070]
half: extent [0.04, 0.06, 0.04], center [0.007, 0.000, 0.060]
```

Its status is `physics_validated_release_checkpoint_proxy`. “Proxy” is
intentional: the vector is compatible with what the released network learned;
it is not claimed to be the exact axis-aligned physical swept volume of the
current L-shaped hand. The exact current Unitree URDF, current `G_T_palm`,
GR00T open/close trajectory, collision meshes, and current Isaac asset remain
unchanged and authoritative downstream.

The descriptor builder now copies those final offsets directly. It no longer
shifts the conditioning box when it laterally centers the physical palm under
`G`; the old shift had implicitly assumed neural invariance to a descriptor
origin reparameterization that GraspGenX does not promise.

### Canonical-path rerun

After rebuilding the normal named `dex3_rev1_right` descriptor, the ordinary
project inference command—not an ablation helper—generated 120 cube proposals.
`tools/build_dex3_isaac_grasp_input.py` copied every `object_T_G` unchanged into
one current-hand Isaac/PhysX batch. Exactly 118 survived the full close and
five-tug sequence. The two non-retained IDs were `grasp_72` and `grasp_97`.

The validator saves only successes unless `--output_failed_grasp_locations` is
requested; therefore its console message “118 successes and 0 fails” means
118 saved successes, not that 120 were tested successfully.

The exact current left descriptor and hand were then tested as a mirror
contract. The left USD was imported from the generated current left descriptor
with the same current gains and `self_collision=false`. The identical 120
canonical `object_T_G` poses and the signed left close profile were replayed in
one batch. The left hand retained 116/120; failures were `37`, `96`, `97`, and
`98`. The right hand's failures were `72` and `97`. A grossly wrong left
`G_T_palm` would have produced the old near-zero-retention failure mode, so this
strong mirrored result validates the left conversion at the level required for
the current checkpoint. The four-candidate right/left pass-set disagreement is
recorded without a claimed cause; exact symmetry would require its own
controlled contact/tolerance study.

Canonical evidence:

- `artifacts/dex3_descriptor_canonical_validation/raw/provenance.json`;
- `artifacts/dex3_descriptor_canonical_validation/isaac_input.yaml`;
- `artifacts/dex3_descriptor_canonical_validation/physics/grasp_sim_data/dex3_rev1_right/grasp_mesh.yaml`;
- `artifacts/dex3_descriptor_canonical_validation/left_physics/grasp_sim_data/dex3_rev1_left/grasp_mesh.yaml`;
- `artifacts/dex3_validation/graspgenx_frame_contract.json`; and
- `docs/dex3_rev1_descriptor.md`.

A camera-enabled rerun selected eight retained candidates (`0–4`, `117–119`)
and the two non-retained candidates (`72`, `97`) from that exact canonical
set. It reproduced 8 passes and 2 failures. The camera ran inside the same
Isaac physics simulation; the readable 45.33-second sequence is
`docs/assets/dex3_descriptor_corrected_review10_sequential.mp4`. Each candidate
is shown for its complete 4.53-second closure/tug run, with the final green or
red border taken directly from the validator result. The synchronized grid
source is `docs/assets/dex3_descriptor_corrected_review10_grid.mp4`.

Claim boundary: this resolves the current-Dex3 descriptor failure for both
hands on the 45 mm cube in intrinsic simulation. It does not yet qualify T/U
parts, tabletop approaches, arm reachability, hardware grip force, or assembly.

## 2026-07-21 — quick intrinsic T/U grasp screen

The corrected named `dex3_rev1_right` descriptor was run on the actual
watertight T and U meshes. For each part, GraspGenX generated 480 stochastic
proposals and the top 120 by neural confidence entered one Isaac/PhysX batch
unchanged. The number 120 is a project evaluation budget chosen for useful
diversity and convenient GPU batching; it is not a GraspGenX constant or an
acceptance threshold. A future offline library should sample more candidates
across multiple seeds and retain qualified grasps by region and reachability.

The cube print uses 30 g of filament. Assuming the same effective material
density and print strategy, mesh-volume ratios give provisional masses:

```text
T / cube volume ratio = 6.0353 -> T = 181.1 g
U / cube volume ratio = 7.0427 -> U = 211.3 g
```

These are estimates until the finished parts are weighed. Earlier solid-volume
estimates of 724 g and 845 g are discarded. A 120 g diagnostic was also run to
separate load sensitivity from geometry; it is not the expected deployment
mass.

Right-hand close-and-five-tug results:

```text
part       120 g control     30 g cube-scaled mass
T          65/120            43/120 at 181.1 g
U          43/120             9/120 at 211.3 g
```

Therefore both parts have intrinsic candidates, but U is substantially more
load/contact-sensitive. This is not evidence of another descriptor collapse.

A camera-enabled replay covered eight diverse passing T candidates and all
nine passing U candidates. Every selected candidate reproduced its pass. T
candidate `grasp_67` visibly holds the central stem while leaving the shoulder
crossbar exposed, satisfying the holder-grasp requirement at the intrinsic
hand/object level. The U set contains leg-side grasps that leave the bridge
available, but their eventual connector and approach clearance still needs the
assembly-scene filter.

Evidence:

- `artifacts/dex3_tu_canonical_validation/raw/provenance.json`;
- `artifacts/dex3_tu_canonical_validation/t_body_right_physics_mass_from_30g_cube/grasp_sim_data/dex3_rev1_right/grasp_mesh.yaml`;
- `artifacts/dex3_tu_canonical_validation/u_legs_right_physics_mass_from_30g_cube/grasp_sim_data/dex3_rev1_right/grasp_mesh.yaml`;
- `docs/assets/dex3_t_body_passing_grasps_grid.mp4`; and
- `docs/assets/dex3_u_legs_passing_grasps_grid.mp4`.

Claim boundary: these are free-space intrinsic right-hand grasps. They do not
yet prove table-clear approaches, G1 arm reachability, left-hand T/U retention,
connector clearance, or hardware grip force.

## 2026-07-21 — G1 arm in the unchanged upstream Franka tabletop scene

The first upper-body scene placed the table and pelvis using provisional
hand-selected heights. That layout has been superseded for the current control
experiment. The table geometry, table pose, object offset, and camera now match
GraspGenX's `tabletop_single_nobin.yaml` exactly; only the object scale and
pre-rotation differ because the AprilCube mesh is already in metres and already
upright.

The G1 root placement is derived from the robot models rather than guessed:

```text
upstream Franka base                    (0.0000000, 0.00000, 0.30000)
upstream Franka joint-1/shoulder world  (0.0000000, 0.00000, 0.63300)
G1 right shoulder in pelvis frame       (-0.0000072, -0.10021, 0.29178)
derived G1 pelvis world                 (0.0000072, 0.10021, 0.34122)
```

This overlays the G1 right shoulder position on the Franka shoulder while
leaving both robots' base axes aligned. A collision-aware cuRobo IK solution
places the G1 palm at the upstream Franka home-palm pose:

```text
world palm position  (0.1612830, 0.0000001, 0.9368466)
right-arm q          [-3.0834400654, -1.0238929987, 2.1113719940,
                       0.2953709960, -0.5985820293, -0.4270730019,
                       0.7260659933]
```

The derived start state passed cuRobo self- and table-collision checking. The
generated single-arm derivative now removes the unused left-arm subtree. This
was necessary because the previously fixed left hand—not the active right
arm—hung inside the unchanged Franka table. Unitree's source URDF is not
modified. The generated model has 21 links and exactly 14 movable joints: seven
right-arm joints plus seven current-Dex3 joints; cuRobo plans only the seven arm
joints and locks the fingers open.

With the actual 45 mm cube, the unchanged upstream scene is:

```text
tabletop world z     0.5000 m
cube center world    (0.5000, 0.0500, 0.5225) m
G1 pelvis world      (0.0000072, 0.10021, 0.34122) m
```

The first exact upstream-style run used `graspmoe`, 200 diffusion proposals,
the top 80 combined candidates (44 diffusion + 36 OBB in that run), cuRobo's
collision world, the `pick_and_lift` task, Newton dynamic playback, and the
measured 30 g cube mass. It stopped correctly before Newton because cuRobo
could not find a valid approach.

A repeated seed-0 geometric/IK audit isolated the cause:

```text
combined GraspMoE candidates                         80
exact open-Dex3 final poses clearing the tabletop     0
best exact open-hand minimum world z                 0.48954 m
tabletop world z                                     0.50000 m
best remaining penetration                           10.46 mm
collision-free exact IK without the table
  pregrasp / final                                    2 / 2
collision-free exact IK with the table
  pregrasp / final                                    0 / 0
```

Therefore the shoulder-aligned baseline is valid and should not yet be shifted
forward. G1 reach is restrictive—only two candidate orientations converged in
free space—but the immediate blocker is stronger: every returned exact
open-hand pose intersects the tabletop. Moving the robot cannot repair that
object/hand/table geometry. No trajectory or MP4 was produced because planning
correctly rejected all candidates before simulation.

The upstream closure contract remains intact. cuRobo plans with the hand open;
`PickAndLiftTask` appends `close_fingers` before the lift; Newton drives the
gripper and resolves contact. Upstream two-finger profiles use one master plus
URDF/profile mimic coupling as appropriate. The current Dex3 profile applies
the same task contract to seven independently commanded finger joints.

## 2026-07-21 — contact-centric cube grasp atlas specified

The project is intentionally returning from the incomplete G1/curobo scene to
the last proven boundary: canonical GraspGenX proposals and exact-hand
Isaac/PhysX qualification. Before further implementation, the researched
contact-centric grasp-library idea has been reduced to a concrete project
specification at `docs/dex3_aprilcube_grasp_atlas_spec.md`.

The first implementation is deliberately only a cube vertical slice:

```text
4,096 diffusion-only canonical cube proposals
        ↓ every proposal unchanged
current right and left Dex3 Isaac/PhysX close-and-five-tug trials
        ↓ measured per-link contacts at six phase boundaries
six AprilCube face regions
        ↓
deterministic side-specific contact families
        ↓
real passing representatives + static HTML + sequential MP4s
```

Important scope decisions:

- use the existing GraspGenX wrapper and GraspDataGen simulator rather than a
  parallel implementation;
- inference runs once because the canonical left/right conditioning is shared,
  while physics runs independently for both exact hands;
- use 16 reproducible batches of 256 proposals, retain every diffusion output,
  and do not admit GraspMoE/OBB candidates;
- preserve the existing physics success rule and add contact tracing only as
  an optional observer;
- use the cube's existing generated voxel-face/AprilTag metadata instead of
  generic segmentation;
- group by body-level digit participation, palm contact, and object-frame
  approach sector; keep detailed link-to-face mappings diagnostic only;
- select only real physics-passing members as medoid/backups; and
- require a visual cube gate before scaling the same schema to T/U.

The specification explicitly defers perturbation sweeps, magnet/connector
roles, stable placements, a compatibility graph, table/arm reachability,
cuRobo, and hardware. Those are useful later, but none is required to prove
that the contact atlas is correct. No atlas implementation was performed in
this step.

## 2026-07-21 — preserve the real palm body in hand-only Isaac qualification

The first Isaac hand assets inherited GraspDataGen's/default Isaac Lab
`merge_fixed_joints: true`. In our descriptor URDF, the canonical `world` link
is fixed to `right_hand_palm_link` or `left_hand_palm_link`. Merging therefore
kept the collision geometry in the correct place but exposed it through the
runtime body name `world`. The initial atlas trace compensated with a
`world -> *_hand_palm_link` semantic alias.

That representation was unnecessary for the hand-only atlas. Newton's
hand-only validator already uses `collapse_fixed_joints=False`, and the atlas
needs clear body-level contact semantics more than it needs to eliminate one
fixed body. A controlled Isaac comparison established:

```text
same current right-Dex3 URDF and descriptor
same first 10 raw GraspGenX cube transforms
same open/close state, object and physics validator

merged asset:      10 pass, 0 fail
non-merged asset:  10 pass, 0 fail
```

The non-merged run also completed the optional six-phase trace. Its PhysX body
names resolved directly to all eight requested physical links, including
`right_hand_palm_link`; no alias was required. This is evidence that preserving
the fixed palm does not break the upstream batched validator and improves the
atlas contact contract.

Both canonical Isaac hand assets are now imported with
`merge_fixed_joints: false`. `tools/import_dex3_isaac_asset.py` defaults to that
hand-only policy, while retaining an explicit `--merge-fixed-joints` option.
`tools/build_dex3_isaac_grasp_input.py` traces the real palm link and emits an
empty alias map. The raw `object_T_G` candidates are still copied byte-for-byte
at the numeric YAML level; only the hashed USD physics asset and contact-link
contract changed. The four atlas unit tests still pass.

Before this correction, the first right-hand 256-candidate smoke run completed
with 203 passes and 53 failures. That result used the merged-palm USD and is now
superseded. The preserved-palm rerun completed with 202 passes and 54 failures.
Exactly one borderline proposal changed verdict:

```text
cube_head__seed_0000000019__sample_223
merged fixed joint:       pass
preserved fixed joint:    fail
all other 255 verdicts:   unchanged
```

The two representations are therefore geometrically equivalent but not
numerically identical in the PhysX constraint solver: preserving the palm adds
an explicit fixed-body constraint. The project accepts the preserved-palm
result as canonical because body-level palm semantics are required by the
atlas, the pass-rate change is one borderline candidate out of 256, and the
topology now matches the Newton hand-only validator. The old merged trace and
result were retained, not deleted, under
`artifacts/diagnostics/merged_palm_right_256/`.

## 2026-07-21 — right-hand smoke atlas and sequential Isaac review MP4

Per the current project decision, left-hand qualification is paused. The atlas
continued only with the preserved-palm right Dex3 result; no left-hand physics
run or review media was started.

The first right-hand smoke atlas uses the existing 256 canonical GraspGenX
cube proposals from seed 19. Every proposal entered the exact current-right-
Dex3 Isaac/PhysX close-and-five-tug validator without pose modification:

```text
raw proposals                     256
physics PASS                      202
physics FAIL                       54
right-hand coarse families         20
primary representatives replayed   20
representative replay PASS          20
representative replay FAIL           0
```

The family key is deliberately limited to body-level signals that the asset
exposes reliably:

```text
participating digit chains + palm-contact bit + object-frame approach sector
```

Detailed PhysX contact points are still mapped to broad cube faces and shown
as diagnostics, but neither those point locations nor derived face labels
split families. This avoids allowing unreliable fine contact geometry to
control the atlas while preserving it for frame-error diagnosis and human
inspection. The 202 passing trials form 20 families: six approach sectors for
each of the three common digit patterns (thumb+index, thumb+middle, and all
three digits), plus two singleton palm-contact families.

`tools/render_grasp_atlas.py` selects the real physics-passing primary member
of every family, copies its original Isaac input transform verbatim, checks
its provenance hash, and replays all 20 representatives in one upstream
batched simulation. The sequential presentation is generated from the Isaac
camera frames from that same run; it is not a pose-only or hand-authored
renderer. Each segment shows family/candidate identity, family size, coarse
signature, diagnostic face annotation, phase, and the final physics verdict.

The first capture attempt exposed a real wrapper defect: GraspDataGen parsed
Isaac Lab's `--enable_cameras` argument but its `LabStarter` reconstructed
`AppLauncher` without passing that value. The wrapper now propagates
`enable_cameras` explicitly. A one-representative camera smoke test then
produced a visible 960x720 video and reproduced PASS before the full capture
was admitted.

Final review artifact:

```text
docs/assets/dex3_cube_grasp_families_right.mp4
960x720, 24 fps, 2,180 frames, 90.833333 seconds
20/20 representative replays reproduced PASS
```

The machine-readable selection, replay result, and media probe are retained
under `artifacts/grasp_atlas/cube_v1/right/review/`, including
`review_manifest.json`. Sampled early, middle, last, and per-family frames were
checked for blank output, missing geometry, and unreadable overlays. The
engineering capture gate passed; the user visual-review gate remains pending.

## 2026-07-21 — full right-hand cube, T, and U grasp atlases

After the 256-candidate cube review looked correct, production was expanded in
the agreed direction: the current official right Dex3 only. Left-hand
qualification remains deliberately paused. No table, G1 arm, cuRobo, OBB
proposal, hand-authored grasp, geometric admission heuristic, or terminal
closed-hand render was introduced.

Each object used the same fixed generation and qualification contract:

```text
16 seeds × 256 diffusion proposals                   4,096/object
three objects                                        12,288 total proposals
GraspGenX threshold                                  -1.0 (retain all)
pre-physics pose changes                             none
Isaac/PhysX rate                                     250 Hz
initial closed hold                                  1.0 s
disturbance test                                     five 1 g tugs
authoritative PASS signal                            final object-filtered
                                                     contact in >=2 digits
```

The exact generated meshes were used: 45 mm cube, 135 × 45 × 180 mm T, and
135 × 45 × 135 mm U. The cube used the measured 0.030 kg filament mass. Until
the larger finished prints are weighed, T and U use explicitly provisional
mesh-volume-scaled masses of 0.1810585309129197 kg and
0.2112813226371547 kg. Those values are recorded in their configs and must not
be mistaken for measurements.

All 48 raw shards, 48 right-hand Isaac inputs, 48 ordinary result YAMLs, and
48 six-phase contact traces completed. The production provenance audit checked
candidate IDs, content hashes, object/hand/shard identity, exact 256-trial
cardinality, unchanged `object_T_G`, and ordinary-result/trace verdict
agreement at every boundary. Its summary is
`artifacts/grasp_atlas/production_right_audit.json`.

Production results:

| Object | Raw | Physics PASS | Physics FAIL | PASS rate | Coarse families |
|---|---:|---:|---:|---:|---:|
| cube | 4,096 | 3,223 | 873 | 78.69% | 36 |
| T | 4,096 | 1,616 | 2,480 | 39.45% | 35 |
| U | 4,096 | 764 | 3,332 | 18.65% | 32 |

One family-construction assumption was corrected before these atlases were
accepted. The family signature initially used `closed_before_tug`, even though
the existing validator awards PASS at `after_tug_5_final`. Across passing
trials, the participating-body pattern changed between those states for 105
cube, 280 T, and 97 U grasps. One cube PASS had fewer than two participating
digits before the tugs but acquired its physics-qualified two-digit contact by
the final state. Families therefore use the final qualified state consistently;
closure and intermediate phases remain persistence diagnostics. A regression
test fixes this contract.

Detailed solver points remain diagnostic only. All 196,608 requested
body/phase trace slots were returned for each object. Broad-surface mapping was
valid for 263,483/271,968 cube points (96.88%), 223,766/454,973 T points
(49.18%), and 122,356/404,397 U points (30.26%). The lower T/U mapping rates are
why fine point/face labels do not decide PASS or split families. Coarse
object-filtered body contacts remain the authoritative family signal.

Every family's real primary representative was rerun through the same
camera-enabled Isaac validator to make the sequential videos. The rerun result
is recorded as a presentation diagnostic:

| Object | Families shown | Camera rerun PASS | Camera rerun FAIL |
|---|---:|---:|---:|
| cube | 36 | 36 | 0 |
| T | 35 | 32 | 3 |
| U | 32 | 24 | 8 |

The three red T primaries are singleton families and have no alternative
member. Of the eight red U primaries, seven families have stored backup
representatives and one is a singleton. No red result was hidden, relabelled,
or replaced in the video. It does not overturn the original production
physics PASS or remove that candidate from the atlas. Downstream scene
planning may consider any production physics-passing member; the candidate it
selects is validated in the actual arm/table Newton execution.

Review media:

- `docs/assets/dex3_cube_grasp_families_right.mp4` — 960×720, 24 fps,
  3,924 frames, 163.500 s;
- `docs/assets/dex3_t_body_grasp_families_right.mp4` — 960×720, 24 fps,
  3,815 frames, 158.958 s; and
- `docs/assets/dex3_u_legs_grasp_families_right.mp4` — 960×720, 24 fps,
  3,488 frames, 145.333 s.

All three videos decode without error. Sampled early, middle, last, and one
post-tug frame per family show the current right Dex3, the correct object mesh,
readable overlays, and explicit green/red verdicts. Machine-readable family,
representative, replay, and media manifests live under
`artifacts/grasp_atlas/<atlas_id>/right/`.

One failed runner invocation formatted shard indices as `00` rather than the
required `000`. The output guard rejected it before reading or overwriting any
valid shard. Its log is preserved at
`artifacts/diagnostics/atlas_runner_index_format_guard/`; the corrected runner
then completed all 48 production shards.

Verification at this checkpoint: five atlas unit tests pass, the modified
Python programs compile, family membership/representative invariants pass for
all three objects, and `git diff --check` is clean. The remaining gate is the
user's inspection of the three sequential MP4s. Arm/table/assembly filtering
does not resume until that visual evidence is accepted.

## 2026-07-21 — right-arm pick-and-lift completed for cube, T, and U

The next vertical slice now works for all three printed parts. This is not a
manually authored grasp demo: every selected pose is an unchanged
`object_T_G` from the corresponding right-Dex3 atlas. The working execution
contract is:

```text
physics-passing right-Dex3 atlas poses
        ↓ unchanged object_T_G candidates
cuRobo: choose a reachable candidate and plan
        start → pregrasp → grasp → world-Z lift
        ↓ exact arm joint trajectory
Newton: prescribe the planned rigid arm/palm motion
        + dynamically simulate all seven Dex3 joints
        + dynamically simulate object/table contact, gravity and friction
        ↓
measured object retention through the lift and final hold
```

The target object is present in cuRobo's collision world for the transfer from
the starting pose to the pregrasp, preventing the arm or hand from knocking it
away. The existing named-obstacle contact permission hides only that target
during the intended final approach and lift. The table remains active. Only
the three terminal finger links receive the narrowly scoped grasp-time table
contact permission; the palm, proximal fingers, wrist and arm remain collision
checked. The old hand/table AABB calculation is retained as a diagnostic count
only and admits or rejects nothing.

### Why the arm is prescribed inside Newton

The first connected-arm attempts drove the generic Newton arm with a generic
PD controller. The resulting 10–15 degree arm tracking lag moved otherwise
valid T/U hand approaches into the objects before finger closure. That was not
evidence against the atlas pose or cuRobo path; it was an uncalibrated actuator
model changing the commanded path. The available GR00T/VIRAL gains also are
not a safe drop-in replacement because they assume their own armature, torque
limits, control loop and robot model.

For this grasp-validation stage, cuRobo therefore owns arm path feasibility
and Newton follows its arm trajectory exactly. In the Newton model, the rigid
chain through the palm is kinematic/prescribed. The seven finger bodies remain
dynamic and use the same current-Dex3 controller profile as the hand-only
qualification; the part remains a dynamic free body. This isolates the
question being tested now: can a collision-feasible G1 arm path deliver the
real Dex3 to an exact atlas grasp, close through contact, and retain the part?
It does **not** claim that G1 hardware arm dynamics or tracking have been
validated. Those belong to the later hardware-controller/bridge stage.

### Final runs

All three runs use a 15 cm approach along the candidate's local approach axis
and a 20 cm commanded lift along world +Z. These values are explicit planning
parameters, not a hidden palm-frame correction. The real selected candidates,
reports and measured outcomes are:

| Part | Exact atlas candidate | Pool index | Motion | Newton retention measurements |
|---|---|---:|---|---|
| 45 mm cube | `cube_head__seed_0000000119__sample_082` | 791 | PASS | 0.2060 m rise; 1.030 follow fraction; 0.0036 m max palm-relative drift; 0.00042 m final drop |
| T body | `t_body__seed_0000000149__sample_143` | 19 | PASS | 0.1770 m rise; 0.885 follow fraction; 0.0393 m max palm-relative drift; 0.00682 m final drop |
| U legs | `u_legs__seed_0000000159__sample_107` | 298 | PASS | 0.1606 m rise; 0.803 follow fraction; 0.0664 m max palm-relative drift; 0.00123 m final drop |

The machine-readable reports are:

- `artifacts/right_arm_pick/cube_v1/prescribed_arm_regression/planning_report.json`;
- `artifacts/right_arm_pick/t_body_v1/prescribed_arm_trial_01/planning_report.json`;
- `artifacts/right_arm_pick/u_legs_v1/prescribed_arm_trial_09/planning_report.json`.

The corresponding Newton camera videos are:

- `docs/assets/g1_right_cube_pick_and_lift.mp4` — 960×720, 60 fps,
  692 frames, 11.533 s;
- `docs/assets/g1_right_t_body_pick_and_lift.mp4` — 960×720, 60 fps,
  712 frames, 11.867 s; and
- `docs/assets/g1_right_u_legs_pick_and_lift.mp4` — 960×720, 60 fps,
  712 frames, 11.867 s.

Contact sheets sampled across each video were inspected. In all three, the
arm approaches from above the tabletop, the fingers close around the intended
part, the part rises with the hand, and it remains held during the final
pause. This supersedes the earlier disconnected-hand cube video, which is
retained under its explicitly failed filename for provenance.

### U-part diagnosis and selection

The U was the hardest part and was not accepted on the first object-motion
metric that happened to pass. Several attempted exact atlas poses were
retained as diagnostics. Some produced large, visibly invalid finger-joint
excursions even though the coarse object-rise metric alone could pass; those
runs were rejected rather than relabelled as successes. Other poses planned
but lost or insufficiently lifted the U in the complete Newton sequence.

The family `right_b9e0f10ac744` was then screened systematically: all 77 of
its unchanged atlas members entered the hand/table Newton approach test. The
screen is recorded at
`artifacts/right_hand_approach_screen/u_legs_v1_family_b9e0_full/physics_results.json`.
Candidate pool index 298 from that family was subsequently planned by cuRobo
and passed the full arm/table close-and-lift run. Its final video shows a
coherent grasp. Newton/MuJoCo represents joint limits as soft constraints, so
the exported joint histories were audited separately rather than assuming the
object-retention verdict proved strict limit compliance. The cube stays inside
its URDF limits apart from 3 microradians of numerical noise. The accepted T
run exceeds four limits briefly: thumb-1 by 0.070 rad, thumb-2 by 0.085 rad,
and the middle/index distal lower bounds by 0.023 rad each. The accepted U run
exceeds thumb-1 by 0.079 rad and thumb-2 by 0.228 rad. These excursions are
recorded rather than hidden. Runs with much larger, visibly invalid finger
excursions were rejected. Consequently T/U have passed the current contact
retention contract and visual gate, but this checkpoint does **not** certify
strict hardware joint-limit feasibility. A later hardware/controller test—or
a separately justified hard-limit simulation—must close that remaining gap.

### Current boundary

Completed here:

- current right Dex3, current descriptor and exact AprilCube meshes;
- full right-hand atlases for cube, T and U;
- cuRobo reachability/collision planning for one real atlas pose per part; and
- Newton contact-aware close, 20 cm lift and hold for all three parts.

Still deliberately outside this checkpoint: left-hand qualification,
bimanual coordination, magnetic attachment, assembly/disassembly sequencing,
perception, ROS 2 hardware bridging and physical G1 execution. The immediate
result is three reusable right-arm pick-and-lift primitives, not a completed
assembly demo.

## 2026-07-22 — checkpoint before authoritative Isaac requalification

The current work was checkpointed locally before changing the simulator
contract:

- root repository: `3bd0062 Checkpoint Dex3 grasp atlas and assembly task`;
- GraspGenX submodule: `fac1352 Add current Dex3 grasp validation checkpoint`;
  and
- GraspDataGen submodule: `defd62a Extend Isaac grasp validation for Dex3`.

Nothing was pushed. Existing Newton, Isaac, atlas, and visual outputs were
kept. The VIRAL-profile work writes to new atlas IDs and output directories so
the earlier evidence remains inspectable.

## 2026-07-22 — reproduce the executed VIRAL Isaac contract

The local clean GR00T-VisualSim2Real checkout at commit
`92bf0863d4a9b6ee29849736152b7769bd45c49c` was audited line by line. This
changed an important assumption: its YAML names `idealpd`, but the released
Isaac adapter actually creates `ImplicitActuatorCfg`. Likewise, some declared
fields are not consumed on the released path. The new named simulator profile
copies executed behavior and records dormant declarations separately.

Executed robot/simulator contract:

- Isaac/PhysX at 200 Hz; the source policy command loop has decimation 4,
  hence it may update targets at 50 Hz;
- TGS, four articulation position iterations, zero velocity iterations;
- self-collision disabled by the released G1 mapping, CCD disabled, maximum
  depenetration velocity 1 m/s;
- thumb-0: `kp=2.0`, `kd=0.1`, effort limit 2.45 Nm, velocity limit
  6.857 rad/s, applied armature 0.01275;
- the other six finger joints: `kp=0.5`, `kd=0.1`, effort limit 1.4 Nm,
  velocity limit 12 rad/s;
- applied armatures 0.010829175 for index-0/middle-0 and 0.03 for the four
  remaining distal joints; and
- joint friction zero and no command interpolation. The qualifier sends one
  constant `q_close`; Isaac's implicit drive retains it through every physics
  step, so command-refresh rate does not alter this static target.

The adapter's explicit `armature × 3` and `friction × 0` transforms are
already included in those values. The YAML's `idealpd` label, global
`contact_offset=0.01`, global `rest_offset=0`, and effort-scale `0.95` are
recorded as declared-but-not-applied instead of being silently copied into a
different program path. `tools/audit_viral_isaac_profile.py` verifies this
transcription directly against the pinned source checkout.

Task-object physics is explicit rather than borrowed from an unrelated VIRAL
prop: exact AprilCube meshes, convex decomposition, friction 1.0, restitution
0, object contact/rest offsets 0.002/0 m, zero damping, and maximum
depenetration velocity 1 m/s. The intrinsic hand-only qualifier keeps gravity
off and applies the existing five one-gravity directional tugs. Table
clearance, approach feasibility, and arm motion remain later task-scene tests.
The cube uses the measured 0.030 kg print mass. T and U retain density-scaled
estimates; their exact mass is not a current blocker because the tug force is
also scaled by object weight, though inertia remains part of the recorded
simulation provenance.

The implementation remains an upstream extension, not a second simulator:

- `simulation_profiles.py` owns the named, source-pinned profile;
- GraspDataGen's existing `grasp_sim.py` applies it and emits observational
  contact traces;
- the root runner provides resumable 256-candidate shards and validates every
  returned ID, profile, hash, phase, and PASS label; and
- the atlas builder and arm-pool exporter consume those validated records.

## 2026-07-22 — contact-trace cancellation found and corrected

The first VIRAL-profile full run produced valid simulator PASS/FAIL verdicts,
but the new coarse contact trace stored the vector sum of all object-filtered
contact-pair forces on each hand link. Opposing force vectors on one link can
cancel to zero. This is unsuitable for saying whether that link participated
in a contact family.

This did not corrupt the simulator verdict. Its existing PASS path computes
the norm of every filtered body-pair force, takes the maximum per link, and
requires at least two active digit groups. The contradiction was discovered
when a passing cube trial appeared to contain only one active digit in its
serialized final phase. Earlier work did not expose it because it consumed the
binary simulator verdict and did not attempt contact-family clustering.

The trace now records `contact_force_magnitude_N` as the maximum norm over the
same object-filtered body pairs. At the qualified final phase, that scalar is
overwritten from the exact tensor used by the PASS decision. The runner
reconstructs the digit-group verdict from the serialized scalars and refuses
the shard if it disagrees. A regression test explicitly creates equal and
opposite forces whose vector sum is zero and verifies that contact remains
present.

The lossy old vectors could not be repaired after the fact, so all 12,288
trials were rerun into `physics_outputs_body_scalar_v2`. The earlier
`physics_outputs` directories were preserved. Comparing the two independent
production runs found zero PASS/FAIL flips, which confirms that only family
metadata—not grasp qualification—was affected. Very small smoke runs with a
different environment count are not treated as reproducibility checks because
GPU PhysX layout changes with the batch layout.

## 2026-07-22 — authoritative right-Dex3 grasp atlases

The corrected full run and atlas build completed:

| Part | Proposals | Isaac PASS | PASS rate | Contact families | Arm-pool entries |
|---|---:|---:|---:|---:|---:|
| 45 mm cube | 4,096 | 2,437 | 59.50% | 40 | 2,437 |
| T body | 4,096 | 1,240 | 30.27% | 39 | 1,240 |
| U legs | 4,096 | 675 | 16.48% | 28 | 675 |

Every arm-pool entry is an unchanged GraspGenX candidate that passed the
VIRAL-profile intrinsic Isaac qualifier. The task configuration now references
these `*_viral_v1/right/arm_grasp_pool.yaml` files. Left-hand qualification
remains intentionally deferred.

The family-review MP4s replay one primary family representative sequentially
through the same named profile. Their replay result is shown explicitly but is
only a visual diagnostic: changing from the production 256-environment layout
to a one-environment-per-family review batch can change marginal PhysX cases,
so it never overwrites the production atlas label.

The generated reviews are all 960×720 at 24 fps and decode without errors:

- cube: 40 families, 39 replay passes, 1 explicit replay failure, 181.67 s;
- T: 39 families, 30 replay passes, 9 explicit replay failures, 177.13 s; and
- U: 28 families, 20 replay passes, 8 explicit replay failures, 127.17 s.

The files are `docs/assets/dex3_*_grasp_families_right_viral.mp4`.

## 2026-07-22 — UniBot-V1 seated/tabletop reference audit

The official Unitree UniBot-V1 Challenge collection is useful for our setup,
but only along clearly separated evidence boundaries. Across its 32 listed
G1-Dex1 tabletop datasets, the released metadata totals 33,276 episodes and
18,282,246 frames. The datasets expose named lower-body and waist joints,
both seven-joint arms, torso and `d435` poses, calibrated head stereo, wrist
cameras, and 30 Hz timestamps. Therefore this is much stronger evidence for a
seated seed and demonstrated arm/table workspace than a pose invented by eye.

Across the 32 datasets, the median of each dataset's reported joint median is:

```text
left  hip pitch -0.403860, knee +0.652441, ankle pitch -0.251860 rad
right hip pitch -0.409678, knee +0.647261, ankle pitch -0.258802 rad
waist pitch +0.171450 rad
```

For an internally coherent seed, the project stores one state that actually
occurred—ArrangePlates episode 0, frame 0—rather than combining independent
coordinate-wise medians into a pose that may never have existed. The exact
joint values, dataset revision, and source indices are in
`config/setup/unibot_arrangeplates_reference_v1.yaml`.

The camera-frame audit resolves the ambiguous part of that first estimate.
Across all 1,735 frames of ArrangePlates episode 0,
`inverse(state_torso) * state_d435` is the constant transform
`xyz=(0.0576235, 0.01753, 0.42987) m`, `rpy=(0, 0.830776724, 0) rad` to
numerical precision. Those values exactly match the official G1 URDF's fixed
`torso_link -> d435_link` joint. The end-effector poses obey the same
`gripper_base` parent convention, so `state_d435` is now identified as
`gripper_base_T_d435_link` rather than an unnamed optical-frame pose.

There is an important stream and mounting distinction—not a claim that the
D435i hardware is monocular. The D435i is physically a stereo-depth camera,
with two infrared imagers in addition to its RGB sensor and IMU. Unitree calls
the built-in D435i option "monocular" in its teleoperation equipment table
because that application consumes one head-view image. In the same table,
Unitree identifies a separately mounted RGB stereo camera for stereo viewing
and dataset capture. The UniBot dataset's two `head_stereo` RGB views are from
that external unit. The dataset publishes the pair's intrinsics and
left-to-right calibration, but no rigid transform from its optical frames to
`d435_link`. We therefore must not equate `head_stereo_left` with the built-in
D435 optical frame or claim an official Unitree mount extrinsic.

For the nominal simulation seed, the strongest released reference is NVIDIA's
fixed G1 RealSense real-to-sim alignment. It attaches the simulated camera to
`d435_link` with translation `(0, 0.035, 0) m` and quaternion
`wxyz=(0.99955, 0, 0.0299955, 0)`, equivalent to `Ry(+0.060000002 rad)`.
NVIDIA's accompanying projection code defines its camera axes as X forward,
Y left, Z up and maps them to OpenCV as `(x, y, z)=(-Y, -Z, X)`. Combining
those contracts gives the following transform, which maps a left OpenCV
optical-frame point into `d435_link`:

```text
d435_link_T_left_optical =
[ 0.000000  -0.059964   0.998201   0.000 ]
[-1.000000   0.000000   0.000000   0.035 ]
[ 0.000000  -0.998201  -0.059964   0.000 ]
[ 0.000000   0.000000   0.000000   1.000 ]
```

This alignment also passes an independent check against the UniBot stereo
data. Rectified stereo plane fits at frames 0 and 60 produce table normals
within 1.299 and 0.721 degrees of vertical after applying the transform. Their
plane offsets in `gripper_base` are 0.04139694 m and 0.04164816 m, agreeing
within 0.26 mm despite being two seconds apart. With no +0.06 rad correction,
the frame-60 residual tilt is 3.277 degrees; with the opposite sign it is
6.677 degrees. We will therefore use NVIDIA's transform as a nominal simulation
seed—not as our physical camera calibration. The released NVIDIA repository
contains the fixed offset and code that applies and randomizes it, but no
routine or documented procedure that measures it. Extrinsic randomization
makes their learned policy less sensitive to mounting error; it does not tell
us our robot's transform. The real G1 therefore needs a one-time target-based
robot-camera extrinsic calibration before camera observations can be placed
accurately in the planning frame.

The Dex1/Dex3 difference does not invalidate the lower-body seated seed or the
coarse table-relative arm workspace. It does invalidate blindly copying hand
clearance, grasp, and wrist-collision assumptions because the current Dex3 is
larger and differently shaped. Adoption therefore requires mapping the named
joints into our selected official G1 URDF, rendering that observed state with
the current Dex3 collision model, confirming forearm/hand table clearance,
and then setting the real height-adjustable table from the measured seated
robot. AprilTags can correct object/table pose at runtime; they do not correct
a poor nominal seated collision geometry.

### Camera calibration actually required by the assembly demo

The UniBot external stereo is only evidence for seated posture and coarse
table workspace. It is not part of our runtime perception chain, so recovering
its unpublished mount extrinsic is unnecessary. Our built-in D435i can run a
single RGB stream for AprilTag detection; depth is optional for scene checking.
The RealSense driver supplies factory intrinsics and the internal transforms
among the D435i's color, infrared, and depth sensor frames.

For hardware assembly, the required pose chain is:

```text
planning_T_object =
    planning_T_d435_link(q)
  * d435_link_T_color_optical       # calibrate once on our G1
  * color_optical_T_tag             # measured every camera frame
  * tag_T_object                    # exact from AprilCube CAD

planning_T_grasp = planning_T_object * object_T_grasp
planning_T_connector = planning_T_object * object_T_connector
```

Only `d435_link_T_color_optical` is a robot-specific camera-mount calibration.
It should be measured once after the camera or head cover is installed or
disturbed, stored as a versioned static transform, and verified with an
independent target pose. NVIDIA's fixed value is only its initial guess. A
ChArUco or AprilTag board rigidly mounted at a known robot-link transform and
observed from multiple diverse arm poses gives the measurements needed for a
standard robot-camera/hand-eye solve. The released NVIDIA repositories do not
contain that calibration routine; they apply a fixed transform and randomize
around it during policy training.

At each demo setup, a tag board fixed to the height-adjustable table should
register the table collision plane and task frame in the robot planning frame.
AprilCube face tags then locate the individual parts. CAD provides every
`tag_T_object`, grasp-atlas entries provide every `object_T_grasp`, and the
connector design provides every `object_T_connector`. The camera timestamps
must be paired with the corresponding robot joint state because the waist can
move `d435_link` even while the robot remains seated.

This means camera work is deliberately staged:

1. Simulation motion planning uses exact object poses and needs no camera.
2. Hardware bring-up verifies the RealSense driver's internal frame tree.
3. Calibrate our one `d435_link_T_color_optical` static transform.
4. Register the table at each setup and detect AprilCube poses at runtime.

We do not need to calibrate the D435i's stereo pair ourselves, reproduce the
UniBot external stereo, or use depth to estimate grasp poses for this tagged,
known-object demo.

## 2026-07-22 — UniBot-seeded cuRobo scene contract

The first assembly planning scene is now a concrete, versioned configuration
at `config/planning/unibot_seated_aprilcube_v1.yaml`. It uses the exact lower
body, waist, and 14 arm joint values from UniBot ArrangePlates episode 0,
frame 0. The UniBot `gripper_base` frame is numerically the official G1
cuRobo/URDF `base_link` frame for this sample: applying official forward
kinematics from `base_link` reproduces the recorded `state_torso` transform.
No guessed pelvis or torso offset is inserted.

UniBot used Dex1 grippers, so only its body and arm pose is transferred. The
two grippers are replaced with the open configuration from our versioned
current-Dex3 descriptor, and all collision checks use Unitree's official
`g1_29dof_with_hand_rev_1_0.urdf` meshes.

The UniBot-derived table top is `z=0.04152595 m` in `base_link`. At the exact
observed arm pose, that surface intersects the physically larger open Dex3.
Because our real table is height-adjustable, the planning seed preserves the
observed seated posture and lowers the table by 35 mm, to
`z=0.00652595 m`. This gives the exact full G1/Dex3 collision geometry a
minimum table clearance of 10.16 mm. This is a measured compatibility
adjustment, not a new arbitrary posture or a claim that UniBot used this
height. The physical table value remains configurable and must be verified on
our seated robot.

The scene places the actual 45 mm AprilCube T, U, and cube meshes directly on
the tabletop, with T and U lying horizontally. Their initial positions are
explicit task seeds and are separated from each other and the robot. Runtime
AprilTag estimates will later replace these nominal poses. There are no trays,
cradles, magnets, or simulated grasp assumptions in this scene.

The visual checkpoint is
`docs/assets/unibot_seated_aprilcube_scene_v1.png`, generated by
`tools/render_curobo_scene.py`. Automated geometry tests require all 43
physical joint values, verify that every object rests on the table without
overlap, and check the full robot against the table and all loose objects.
This scene is the fixed input to the next checkpoint: right-arm cuRobo planning
from a qualified right-Dex3 T grasp candidate. No trajectory has been planned
or accepted yet.

## 2026-07-22 — full T/U/cube cuRobo assembly is planned and rendered

This section supersedes the final sentence above. The UniBot-derived full-body
scene was useful evidence, but it is not the scene used by the completed
planning checkpoint. At the user's direction, implementation restarted from a
clean cuRobo v0.8.0 checkpoint with only the fixed torso, both seven-joint
arms, and both current Dex3 hands. The nested cuRobo repository remains clean;
all demo policy is project-owned.

### Concrete scene

The robot model is generated by `tools/build_g1_dual_arm_model.py` as
`generated/robot/g1_fixed_torso_dual_dex3.{urdf,yml}`. It has 28 movable
joints: 14 arm joints and 14 Dex3 finger joints. Lower-body joints are absent
from this planning model, and the base is fixed at world `(0, 0, 0.75) m`.
This isolates the tabletop arm problem; it is not yet a claim that the final
chair/base transform is calibrated on the real seated G1.

The versioned planning scene is
`config/planning/t_u_cube_full_assembly_v1.yaml`:

- table center `(0.55, 0, 0.68) m`, size `0.80 x 0.80 x 0.04 m`, top at
  `z=0.70 m`;
- T upright at `(0.40, 0.28, 0.79) m` so its central stem is accessible;
- U upright at `(0.36, -0.22, 0.7675) m` on its two legs;
- 45 mm cube centered at `(0.38, -0.32, 0.7425) m`; and
- a `15 x 15 x 20 mm` locating peg below the cube.

The peg is not a tray or cradle and does not constrain the cube laterally. A
measured candidate-reachability sweep found no single qualified cube grasp
that worked at both a bare-table pickup (`cube z=0.7225`) and the future head
mate. Raising the cube 20 mm produced two shared candidates; this is the
smallest tested height that makes the fixed task feasible. The cube bottom and
peg top both lie at `z=0.72 m`. This prepared fixture can be redesigned with
the physical magnet connector later.

The ready arm state is an exact current-model IK result. Its two GraspGenX tool
frames start at approximately `(0.30, +0.24, 1.02) m` and
`(0.30, -0.24, 1.02) m`, collision-clear of the table and all loose parts.

### Grasp selection contract

No grasp is authored in the task code. The planner searches only unchanged
GraspGenX candidates that passed the Isaac/PhysX VIRAL-profile qualifier. The
successful run selected:

| Part | Hand | Candidate | GraspGenX score |
|---|---|---|---:|
| T | left | `t_body__seed_0000000159__sample_088` | 0.867316 |
| U | right | `u_legs__seed_0000000119__sample_043` | 0.963534 |
| cube | right | `cube_head__seed_0000000119__sample_201` | 0.166805 |

The T candidate approaches the upper central stem immediately below the
shoulder crossbar. The candidate-region filter does not mistake the invisible
GraspGenX origin for a contact point. It intersects the candidate's declared
positive-local-Z approach ray with the exact union-of-cuboid part model and
uses the first hit only as a coarse task-region label. Isaac/PhysX remains the
grasp-retention authority.

For every U or cube candidate, selection intersects four reachability sets:
pickup pregrasp, pickup exact grasp, future mate precontact, and future exact
mate. This prevents choosing an excellent pickup grasp that cannot perform its
later assembly operation. The exact feasible joint target found during
selection is reused by the planner instead of solving the same stochastic IK
problem again.

The target object stays collision-live during the free-space move to pregrasp.
Only that target's world copy is absent for the final straight contact
approach, where contact is intentional. At connector contact, the holder's
attachment slot is hidden only for the exact endpoint because cuRobo v0.8 has
named-link collision enable/disable but no pairwise Allowed Collision Matrix.
The table, all other loose objects, robot self-collision, and the other arm
remain collision-live.

### Planning and attachment implementation

`g1_aprilcube_demo/planning/assembly_runner.py` owns the fixed choreography and
uses clean cuRobo public components. OMPL, MoveIt, Newton, and MuJoCo are not
part of this planning run.

Each arm gets a side-specific Cartesian IK solver with 256 seeds. The main
planner owns collision-aware 14-joint transfers. A critical bug was found in
the right-side solver: its solution joint order is right arm then left arm,
whereas the main model is left then right. Positional slicing silently copied
the wrong seven values. The adapter now maps every solution by joint name and
reorders the current seed into each solver's declared order. Side IK also has
CUDA graph capture disabled because selection alternates between 48-pose
goalsets and singleton waypoints, shapes that the released standalone solver
cannot recapture dynamically.

The two arms move sequentially. A full 14-joint plan may otherwise drift the
nominally stationary arm, so each accepted one-arm trajectory is projected
onto the exact stationary seven-joint values and every projected sample is
collision-checked before acceptance. This provides the desired one-arm-at-a-
time choreography without giving up whole-robot collision checking.

Loose part collision geometry is represented as its exact 45 mm voxel-cuboid
union. This avoids mesh-SDF behavior that was inappropriate for these small
generated parts. Carried parts use cuRobo's released MORPHIT sphere fitting in
independent `left_attached_object` and `right_attached_object` slots. On each
snap, the child is removed from the worker slot and the holder slot is replaced
with the full T-relative composite: T, then T+U, then T+U+cube.

The fixed execution is:

1. left hand picks T at its qualified central-stem grasp and raises it;
2. left stages T at `z=1.02 m`;
3. right picks U, retracts, approaches the bottom connector from below, and
   snaps it to T;
4. right releases and retreats 75 mm; left now carries T+U;
5. right parks while left lowers the composite to T `z=0.95 m`;
6. right picks the elevated cube, retracts, approaches from above, and snaps;
7. right releases and retreats; left now carries T+U+cube;
8. left places the complete 360 mm figure with its lowest geometry at
   `z=0.703 m`, 3 mm above the collision tabletop;
9. left opens, all three parts return to world state, and left retreats
   75 mm for the reveal.

"Magnetic snap" here is an explicit scene-state transition at the exact
declared connector transform. This checkpoint does not simulate magnetic
attraction, connector compliance, or part settling. Those belong to physical
connector testing, not motion planning.

### Successful run and exact visual replay

Run the planner with:

```bash
PYTHONPATH=.:third_party/GraspGenX/ext/curobo \
  .venv/bin/python tools/run_full_assembly.py
```

It completes 56 successful planning events and records 174 state-bearing
segments. The generated files are:

- `artifacts/full_assembly/t_u_cube_v1/planning_report.json`;
- `arm_trajectories.npz` with the exact 14-joint cuRobo results; and
- `render_state.json` with each segment's loose/attached object rules and hand
  closure fraction.

`tools/render_full_assembly.py` does not invent a preview trajectory. It reads
those saved arm trajectories, reconstructs the current URDF's 35 visual meshes
with `yourdfpy`, interpolates the versioned current-Dex3 open/close profiles,
computes every attached object as `world_T_G * grasp_T_object`, and passes the
result to GraspGenX's existing EGL renderer. Long solver paths are sampled with
both endpoints preserved; state transitions are never interpolated away.

The reviewed result is 582 frames, 960 x 720, 24 fps, and 24.25 seconds:
`docs/assets/t_u_cube_full_assembly_curobo_v1.mp4`.

The first final-reveal attempt requested an accidental hard-coded 100 mm
holder retreat. The completed object had already been placed and released,
but the fourth 25 mm waypoint failed. The final code uses the config's verified
75 mm retreat, and all four waypoints pass.

The root test command is now unambiguous (`pytest.ini` excludes vendored test
suites). The final project result is `19 passed`; warnings are upstream
`yourdfpy`/NumPy deprecations.

### Evidence boundary and next hardware gate

This checkpoint proves that one collision-aware kinematic sequence exists for
the exact current model, exact current grasp-frame contract, actual AprilCube
meshes, qualified grasp candidates, declared table, and explicit attachment
semantics. The video is a replay of that result.

It does not yet prove ROS 2 execution, Dex3 command tracking, seated-base pose
repeatability, real magnet capture tolerance, AprilTag pose accuracy, table
registration, camera extrinsics, or grasp survival under arm acceleration.
The next implementation gate is the ROS 2 arm/hand execution boundary plus a
slow single-part hardware pick using the same candidate and measured runtime
object pose—not another change to the planning architecture.

## 2026-07-22 — runtime-conditioned cuRobo assembly replacement completed

### Why the fixed runner was not extended

The previous `assembly_runner.py` proved one collision-aware choreography for
one prepared world. It selected grasp IDs and world poses in configuration,
used project-side IK/trajectory handling, and therefore did not answer the
actual demo requirement: receive three separated AprilCube poses at runtime,
select ordinary physics-qualified atlas candidates at those poses, and plan
the complete task without trays, pegs, cradles, or silently restored nominal
poses.

The old runner and its video were retained as regression evidence. Nothing
was deleted. The replacement is a separate path whose contract is
`docs/runtime_curobo_assembly_spec.md`.

### Implemented boundary

The new path consists of:

- `config/observations/t_u_cube_{nominal,shuffled}_v1.yaml`: versioned table
  and loose-object measurements, independent of planner configuration;
- `g1_aprilcube_demo/runtime/observation.py`: finite/normalized transform,
  exact mesh support, tabletop-XY support, ID, and overlap validation;
- `planning/grasp_goalset.py`: immutable atlas loading and only the declared
  transform `world_T_object * object_T_G`;
- `planning/workspace.py`: bounded, center-first work and placement samples;
- `planning/curobo_backend.py`: the narrow adapter over released
  `MotionPlanner`, `GoalToolPose`, `ToolPoseCriteria`, and the existing
  attachment managers;
- `planning/runtime_assembly.py`: connector-mode qualification, complete
  in-memory planning, task-state assertions, backtracking, and reports; and
- `tools/run_runtime_assembly.py`: one observation/config/task CLI.

No project Cartesian interpolator, project IK solver, manually authored
grasp, fixed loose-object pose, or render-based grasp decision exists in this
path. `MotionPlanner.plan_grasp` owns candidate selection, approach, and exact
grasp planning. `plan_pose`/`plan_cspace` own transfers, and
`ToolPoseCriteria.linear_motion` owns connector/descent/retreat constraints.

### Implementation findings and the reasons for each correction

1. **Goal-set capacity is 32 in this pinned solver.** Batches of 8, 16, and 32
   preserved a known good candidate; a 48-entry request failed even with that
   candidate first. All candidates remain eligible, but are sent in
   deterministic 32-entry goal sets. A warmed planner is reused and its random
   seed reset for every slice so feasibility does not depend on preceding
   failures.

2. **A multi-tool goal set does not preserve pair identity.** cuRobo returns a
   goal-set index per tool, while `plan_grasp` applies the first tool's index to
   every tool. Every holder/worker mate hypothesis is therefore a singleton
   two-tool `GoalToolPose`; left and right cannot be independently selected
   from different supposed rows.

3. **The public attachment facade is broken in this checkout.** It forwards
   to a nonexistent `TrajOptSolver.attachment_manager`. The already-created
   managers live at `ik_solver.core.attachment_manager` and
   `trajopt_solver.core.attachment_manager`, and IK and TrajOpt have distinct
   kinematics instances. The adapter fits one immutable sphere tensor per
   AprilCube mesh using the released manager, then sends the identical tensor
   and transform to both managers.

4. **Finger geometry must match task state.** The generated robot originally
   locked all 14 Dex3 finger joints at zero. That is correct before pickup but
   wrong while carrying and mating. Each short-lived planner now locks both
   hands at the exact rev-1.0 descriptor open/close interpolation: open for
   pickup, closed after attachment, open after release. The arm planner does
   not pretend to simulate closure; the separate finger command changes the
   task state, and the next planner sees the correct collision geometry.

5. **Stationary means locked in the robot model.** A one-arm stage locks the
   other seven arm joints before cuRobo builds kinematics. The code never plans
   two arms and projects one away afterward. Coupled endpoint qualification is
   the only genuinely 14-arm-DOF solve; execution moves the two arms
   sequentially.

6. **Intentional support/contact boundaries need narrow permissions.** The
   newly attached pickup object begins in table contact, so only `table` is
   disabled during its constrained upward separation. At a mate, the holder
   composite slot is hidden only during the worker's exact connector
   separation. During final support descent only the table is hidden. After
   release, the hand begins in contact with the newly published T/U/cube world
   copies, so only those named part cuboids are hidden during the constrained
   upward empty-hand retreat. The permissions are restored after each call;
   self-collision and unrelated obstacles remain active.

7. **Fixed placement lift offsets were geometrically wrong.** Trial 60 mm and
   150 mm offsets asked a near-limit holder to move farther upward even though
   the assembly was already above its derived support pose. Placement now
   keeps the actual current root height for the horizontal transfer, then
   descends to `table_top - assembly_min_z + 3 mm`.

8. **A cached paired IK vector is a witness, not a trajectory.** The varied
   scene showed that the collision-free paired precontact joint vector could
   have no path from the realized post-cube-pick state. Execution tries that
   joint target first. If it fails, upstream `plan_pose` may find another
   joint branch at the identical grasp-frame precontact pose; the exact linear
   connector approach is then replanned from the branch actually reached. No
   object target or waypoint changes.

9. **Bounded backtracking needs geometric diversity.** The first four-mode cap
   came from a depth-first Cartesian product, so every mode reused one T and
   one cube grasp and changed only U. The corrected list takes at most one
   complete connector mode per distinct T grasp. This is diversity among
   already cuRobo-qualified modes, not a new hand-authored grasp heuristic.

10. **Mode caches must be safe but not punish unrelated iteration.** Cache
    keys cover a versioned qualification contract, backend/goal/workspace
    source, planner config, observation, task, robot config/URDF, both hand
    profiles, every atlas, and every part mesh/config. Execution/report-only
    edits do not invalidate minutes of identical endpoint work; qualification
    logic edits must bump `MODE_CACHE_CONTRACT`.

### Runtime-variation evidence

The same executable and `t_u_cube_runtime_v2.yaml` completed two observations:

- nominal: T `t_body__seed_0000000079__sample_089`, U
  `u_legs__seed_0000000119__sample_043`, cube
  `cube_head__seed_0000000119__sample_201`, 33 events, 138 segments, no
  endpoint fallback;
- varied XY/yaw: T `t_body__seed_0000000139__sample_125`, U
  `u_legs__seed_0000000119__sample_043`, cube
  `cube_head__seed_0000000089__sample_161`, 34 events, 138 segments, one
  same-pose alternate-IK fallback.

Both reports assert the compiler's state after all six steps: `pick_t`,
`pick_u`, `mate_u_to_t`, `pick_head`, `mate_head_to_t`, and
`place_complete`. Both terminate with `success: true` and `complete`.

An earlier deliberately broad shuffled observation was not made to pass by
weakening collisions. At 45-degree cube yaw, all four diverse connector modes
failed to connect the realized cube-pick arm state to head precontact. A later
cube location near the rotated U admitted no collision-free cube pickup at
all. The accepted second scene remains separated and changes T/U XY and yaw,
but keeps the cube in a tested clear envelope. This is the intended meaning
of runtime scattering for the first demo.

### Reproduction and visual evidence

```bash
.venv/bin/python tools/run_runtime_assembly.py \
  --observation config/observations/t_u_cube_nominal_v1.yaml \
  --output artifacts/runtime_assembly/t_u_cube_v2/nominal

.venv/bin/python tools/run_runtime_assembly.py \
  --observation config/observations/t_u_cube_shuffled_v1.yaml \
  --output artifacts/runtime_assembly/t_u_cube_v2/shuffled

.venv/bin/python tools/render_full_assembly.py \
  --config config/planning/t_u_cube_runtime_v2.yaml \
  --run-dir artifacts/runtime_assembly/t_u_cube_v2/nominal \
  --motion-frames 10 --hold-frames 10 \
  --output docs/assets/t_u_cube_runtime_curobo_v2.mp4
```

The committed video is 408 frames at 960×720 and 24 fps (17.0 seconds). It was
checked as a contact sheet and at the terminal frame: loose parts, T pickup, U
pickup/mate, cube pickup/mate, support placement, release, and both open-hand
retreats are visible. The complete root test result is `25 passed`; the 132
warnings are existing `yourdfpy`/NumPy deprecations.

### Evidence boundary

This proves collision-aware kinematic planning for two observed scenes, the
current G1/Dex3 collision model, exact 45 mm AprilCube geometry, immutable
physics-qualified atlas grasps, and explicit attachment semantics. It does
not prove magnet capture, grasp survival under real acceleration, ROS 2
trajectory execution, seated base repeatability, AprilTag accuracy, table
registration, or camera extrinsics. Those remain later hardware gates and
must consume this planner's observation/report/trajectory contracts rather
than forcing another planning rewrite.

## 2026-07-24 — pause: intrinsic grasps were confused with tabletop pickups

The flat-part observation exposed a conceptual error, not a shortage of raw
GraspGenX samples. Implementation is paused; no further table or object-pose
tuning should be used to hide the error.

### What the thousands actually prove

The arm pools contain every Isaac/PhysX pass from the VIRAL-profile,
unsupported-object qualification:

- left T: 1,240 poses in 40 coarse families;
- right U: 675 poses in 28 coarse families; and
- right cube: 2,437 poses in 40 coarse families.

Each pose proves that the isolated hand can close on and retain that object
under the recorded free-space physics test. It does not prove that the open
hand clears a support surface, that a pregrasp can be reached, that the G1 arm
has IK, that a path exists from the seated ready state, or that the grasp
leaves an assembly connector usable.

As a non-authoritative but revealing diagnostic, the exact current-Dex3 open
visual mesh was transformed by every unchanged `object_T_G` for the present
flat observation. Requiring both the terminal mesh and its 10 cm local-Z
pregrasp to remain above the tabletop left only:

- 6 / 1,240 T poses;
- 4 / 675 U poses; and
- 57 / 2,437 cube poses.

The full cuRobo robot/collision model must make the real admission decision,
but this count explains why changing the adjustable table height did not
create a broad solution set. Object and tabletop move together, so their
relative clearance is unchanged; height only changes arm reachability.

### Concrete implementation defects found in the audit

1. `allowed_grasp_cuboids` and `keep_clear_connections` are parsed and schema
   checked, but the runtime never applies them to candidate admission or
   ranking. The written task says “hold the T by its stem and preserve both
   connector corridors”; the search currently does not enforce that claim.
2. The task schema says physical hands are assigned after reachability, but
   runtime pool loading and execution hard-code left T, right U, and right
   cube.
3. Contact families describe digit-chain participation, palm contact, and a
   coarse approach sector. They do not partition the T/U by grasped voxel or
   guarantee support-surface accessibility. Thousands of members are highly
   correlated samples, not thousands of independent tabletop modes.
4. `_qualified_pick_representatives()` submits family-ordered batches to one
   opaque `plan_grasp` call and records at most one selected pose per
   successful batch. A failed call does not reveal whether the cause was
   terminal table collision, pregrasp collision, IK, start-to-pregrasp
   connectivity, or the constrained final approach.
5. The project deliberately used diffusion-only generation. The released
   GraspGenX default also offers an OBB expert for top-down candidates and its
   scene-point-cloud demo filters gripper/scene collisions before downstream
   planning. The scene filter is directly relevant. The OBB expert is only an
   optional A/B source: it technically accepts the Dex3 `revolute_3f`
   descriptor and scores its base-pose proposals with the Dex3-conditioned
   discriminator, but its whole-object bounding-box prior does not model
   three-finger contact modes and can be particularly weak on non-convex T/U
   geometry. It must not replace the diffusion atlas by assumption.
6. The previous committed MP4 demonstrates upright prepared objects. It is
   not evidence for the newly requested flat/random stable orientations.
   Current flat observation edits remain uncommitted while this correction is
   reviewed.

### Corrected boundary to evaluate next

Keep the intrinsic physics atlas as evidence; do not delete or relabel it.
Add a separate scene-conditioned feasibility ledger:

```text
GraspGenX diffusion candidates
    (+ released OBB candidates only as a separately tagged fallback experiment)
    -> exact flat/stable object pose in the measured scene
    -> target-excluded hand/scene clearance
    -> connector/allowed-region keep-out checks
    -> batched left/right endpoint IK
    -> start-to-pregrasp and constrained approach planning
    -> Isaac/PhysX supported pickup and lift qualification where needed
    -> downstream mate/placement compatibility
```

Every candidate must retain its immutable `object_T_G` and receive a result at
each gate with a named rejection reason. A visual audit should show the exact
open hand, target, table, approach segment, assigned arm, and pass/fail gate
before the full assembly runner is changed again.

The assembly sequence is already fixed, so this does not require cuTAMP.
After per-stage feasible sets exist, the discrete problem is a small layered
compatibility graph: choose a holder grasp that survives pickup and both mate
poses, plus worker grasps that survive their pickup and mate poses, then ask
cuRobo to verify the complete continuous paths. This replaces early
representative truncation with task-conditioned mode selection.

### First reproducible scene-conditioned checkpoint

The admission ledger is now implemented by
`tools/audit_runtime_grasps.py`; its visual review is generated by
`tools/render_runtime_grasp_audit.py`. The authoritative nominal report is
`artifacts/runtime_grasp_audit/nominal/audit.json`, and its browsable visual
index is
`artifacts/runtime_grasp_audit/nominal/visual/index.html`.

The current nominal flat observation produced:

| Part | intrinsic atlas | open hand clears final + pregrasp | complete upstream `plan_grasp` |
|---|---:|---:|---:|
| T with left hand | 1,240 | 6 | 1 |
| U with right hand | 675 | 4 | 0 |
| cube with right hand | 2,437 | 57 | 13 |

The analytic support gate uses the exact released open descriptor collision
mesh at the candidate's unchanged `world_T_G` and at a -0.10 m
descriptor-local-Z pregrasp. Every support survivor is then submitted
individually to the same upstream cuRobo `MotionPlanner.plan_grasp` contract
used by the runtime, from the configured seated-ready arm state and with the
complete table/object collision scene.

An audit implementation mistake was caught during this checkpoint:
standalone endpoint IK was initially used to gate whether `plan_grasp` would
run. One cube pose failed that diagnostic but passed native `plan_grasp`.
The corrected implementation always treats the complete upstream
`plan_grasp` result as authoritative. Standalone final/pregrasp IK is now run
only after a failed pickup request to explain likely endpoint infeasibility;
it can never veto a successful native pickup plan.

All four support-clear U candidates fail both collision-aware endpoint
diagnostics. Independent probes also showed the same four wrist orientations
fail with the collision world removed, after lateral relocation, and across
table-top heights from 0.64 m through 0.86 m. Therefore this checkpoint does
not justify moving the U, lowering the table, or changing cuRobo settings.
It establishes that the current physics-qualified diffusion subset simply
does not contain a flat-U/right-arm wrist orientation usable by this G1 arm.

This checkpoint deliberately does **not** evaluate allowed grasp cuboids,
connector keep-outs, the later mate poses, or alternative left/right
assignment. Those are explicit `not_evaluated` fields in the report. No OBB
or GraspMoE candidates, manual poses, atlas transforms, object locations, or
assembly choreography were introduced.

## 2026-07-25 — explicit U supports replace generic stable-pose reasoning

The U support problem was simplified to the known geometry. The printed U is
an axis-aligned 3×1×3 AprilCube voxel solid, so its tabletop supports are
declared directly rather than inferred with a convex-hull or generic
stable-pose algorithm:

| Table-up object axis | Physical support | Symmetry class |
|---|---|---|
| `+X` | left outer leg down | outer-leg side |
| `-X` | right outer leg down | outer-leg side |
| `+Y` | broad `-Y` face down | broad face |
| `-Y` | broad `+Y` face down | broad face |
| `+Z` | upright on both leg ends | leg ends |
| `-Z` | inverted on the hip bridge | hip bridge |

The implementation is in
`g1_aprilcube_demo/grasping/support_atlas.py`, configured by
`config/grasp_support/u_legs_right_v1.yaml`, and driven by
`tools/build_support_conditioned_grasp_atlas.py`. Configuration names the six
supports; code only aligns the named axis with table up and translates the
exact U mesh so its minimum world z is zero.

All 4,096 immutable right-Dex3 U proposals were evaluated in all six supports.
Each candidate/support pair passed through:

1. exact open-hand final table clearance;
2. exact open-hand pregrasp table clearance;
3. exact FCL hand/U nonintersection at final and pregrasp;
4. the complete 10 cm negative-local-Z approach, sampled every 1 mm; and
5. semantic annotation of the approached U component, surface/cavity relation,
   support relation, and object-frame approach sector.

The mesh-based continuous collision API was not used because the available
FCL binding does not reliably report continuous collision for these BVH mesh
objects. The 1 mm sampling resolution is explicit provenance, not a claim of
mathematical continuous collision.

Results:

| Support | Geometry-clear proposals |
|---|---:|
| left outer leg down | 1,691 |
| right outer leg down | 1,496 |
| broad `-Y` face down | 33 |
| broad `+Y` face down | 9 |
| upright on leg ends | 1,837 |
| inverted on hip bridge | 1,564 |
| **Total candidate/support pairs retained** | **6,630** |

The 6,630 survivors occupy 164 proposal buckets. Every survivor appears
exactly once and no representative selection or pre-physics pruning is
performed. Of those labels, 6,586 approach rays resolve to the U and 44 miss;
the unresolved members remain explicit rather than being silently discarded.

This explains the earlier 33-grasp result: it was correct for one broad-face
support, but that support was incorrectly treated as the complete U problem.
The support image is
`docs/assets/u_legs_six_tabletop_supports.png`, the readable audit is
`docs/u_legs_support_conditioned_grasp_audit.md`, and the full ignored ledger
is `artifacts/grasp_support/u_legs_right_v1/support_atlas.json`.

These 6,630 are not called successful grasps. The next admission boundary is
the corrected table-supported Isaac test: collision-free pregrasp, complete
approach, closure, vertical lift, and hold under gravity. Final families will
combine named support, semantic U region/surface, approach direction, and
measured post-lift digit/palm participation. cuRobo must later iterate those
families explicitly instead of treating family order as incidental pool
sorting.

## 2026-07-25 — broad-face U supported-pickup Isaac result

The corrected table-supported test is now implemented as a guarded
`supported_pickup` mode in the upstream GraspDataGen
`scripts/graspgen/grasp_sim.py`; its ordinary intrinsic close-and-tug path is
left as the default. Project orchestration is in
`tools/run_isaac_supported_pickup.py`, configured by
`config/grasp_support/u_legs_right_broad_face_isaac_v1.yaml`.

The simulator receives the 42 exact broad-face survivors from the immutable
support ledger and the corresponding raw GraspGenX records. Neural confidence
is retained as provenance but does not prune this physical run. Each batched
trial contains:

1. the current right Dex3 and its exact GraspGenX descriptor contract;
2. the actual U mesh at its named broad-face support;
3. a 1.0 × 1.0 m table at `z=0`, gravity, and the VIRAL-profile 200 Hz
   controller/PhysX configuration;
4. 0.5 s support settling, 1.0 s stored pregrasp-to-grasp translation, 1.0 s
   closure/settling, a 20 cm vertical lift over 4.0 s, and a 1.0 s final hold;
5. object-filtered digit contact, object/table contact, and every traced
   hand-link/table contact; and
6. world-frame object and `G` transforms at five named phases.

The runner independently rejects a trace unless the hand moved exactly 20 cm
between `approach_complete` and `lift_complete` and held its commanded final
pose. All 42 production traces recorded zero final position and orientation
error.

The physical pass contract is conjunctive:

- at least two Dex3 digit chains contact the U after lift and final hold;
- no traced hand link contacted the table at any time; and
- the U did not contact the table during the final elevated hold.

Measured result:

| Broad support | Trials | Digit-contact pass | Full PASS |
|---|---:|---:|---:|
| broad `-Y` face down | 33 | 0 | 0 |
| broad `+Y` face down | 9 | 0 | 0 |
| **Total** | **42** | **0** | **0** |

Fourteen of the 42 trials additionally contacted the table with the hand. All
42 U parts remained on the table during the final hold. The filtered U/table
contact magnitude ranged from 2.070 to 2.079 N, consistent with the full
weight of the provisional 211.3 g U rather than a marginal retained grasp.
The largest transient object rise at the `closed_before_lift` phase was
3.51 mm; by `lift_complete` every U had returned to its support height.

This result empties the right-hand broad-face runtime library for the current
4,096 raw proposals. It does not prove that every conceivable flat-U strategy
is impossible; it proves that none of these unchanged GraspGenX proposals,
under this descriptor-local straight approach and controller profile, is an
admissible supported pickup. cuRobo cannot repair a failed physical grasp, so
these 42 must not enter motion planning.

The complete six-minute sequential review is
`docs/assets/dex3_u_broad_face_supported_pickup_all42.mp4`. The readable
summary is `docs/u_legs_broad_face_supported_pickup.md`; the ignored
machine-readable ledger is
`artifacts/grasp_support/u_legs_right_broad_face_isaac_v1/report.json`.

An ordinary cube input with no `supported_pickup` block was also rerun through
the same modified upstream executable. Its default intrinsic
close-and-five-tug path completed and retained the cube (1/1 PASS), with
`physics.mode=intrinsic_close_and_tug` in
`artifacts/grasp_support/intrinsic_regression_after_supported_pickup.trace.jsonl`.
This is the direct Isaac regression that the new mode remains opt-in.

The next U pickup checkpoint should intentionally use a non-broad support,
preferably upright on both leg ends, and run the same admission contract before
cuRobo integration.

## 2026-07-25 — upright U succeeds; exhaustive replay removes marginal passes

The same supported-pickup contract was applied to the U standing conventionally
on both leg ends. Configuration
`config/grasp_support/u_legs_right_upright_isaac_v1.yaml` selects all 1,837
geometry-clear `upright_on_leg_ends` proposals and changes no object,
controller, timing, contact, table, or pass setting from the broad-face test.
The exhaustive run is camera-free and uses 256-environment physics batches.

Discovery result:

| Trials | Digit-contact pass | Hand/table contact | U/table final contact | Full PASS |
|---:|---:|---:|---:|---:|
| 1,837 | 428 | 33 | 1,408 | 405 |

The 405 passes cover:

- 179 hip-bridge, 153 left-leg, and 73 right-leg proposal labels;
- approach sectors `+X` (192), `+Y` (19), `-X` (116), `-Y` (21), and `-Z`
  (57); and
- 17 proposal buckets.

A first 17-member camera review selected the highest neural score from every
successful proposal bucket, strictly for visualization. Only 11 passed again.
This was not hidden or explained away: the same physical candidates can cross
a marginal contact boundary when environment layout and floating-point solver
ordering change. Therefore one Isaac PASS is discovery evidence, not the final
admission rule.

All 405 discovery passes were replayed through
`config/grasp_support/u_legs_right_upright_replay1_isaac_v1.yaml` in a
deliberately different 64-environment layout. The source report is content
addressed by SHA-256 in that configuration, so the selected set cannot silently
change.

Replay result:

| Input discovery PASS | Replayed PASS | Replayed FAIL |
|---:|---:|---:|
| 405 | 365 | 40 |

The 365 twice-passing candidates contain 149 hip-bridge, 147 left-leg, and 69
right-leg labels across all five discovery approach sectors and 14 proposal
buckets. The 40 non-reproducing candidates are excluded from the upright
physics library.

For a third, visual execution,
`config/grasp_support/u_legs_right_upright_review2_isaac_v1.yaml` selected the
highest-score twice-passing candidate in each of the 14 surviving buckets.
Thirteen passed again; one hip-bridge `-Z` member
(`u_legs__seed_0000000149__sample_039`,
`proposal_bfc42f5cbc01`) returned to the table. The complete 14-trial video is
preserved at `docs/assets/dex3_u_upright_supported_pickup_review14.mp4`. A
derived, non-destructive 13-pass sequence is
`docs/assets/dex3_u_upright_supported_pickup_review13_passes.mp4`.

The outcome for the demo is now concrete:

- do not place the U on either broad face;
- place it upright on both leg ends at any reachable, separated tabletop XY
  and yaw;
- keep all 365 twice-passing candidate identities for task-conditioned cuRobo
  feasibility rather than reducing the runtime library to the 13 visual
  examples; and
- treat the videos as human review, not as the data-admission mechanism.

The primary exhaustive summaries are
`docs/u_legs_upright_supported_pickup.md` and
`docs/u_legs_upright_supported_pickup_replay1.md`. Their full ignored ledgers
are under
`artifacts/grasp_support/u_legs_right_upright_{isaac_v1,replay1_isaac_v1}/report.json`.

## 2026-07-25 — bounded 100K broad-face U experiment: 0/983 PASS

The earlier 0/42 broad-face result left a reasonable question: was the
4,096-candidate raw pool simply too small to contain a rare table-clear pickup?
We ran one bounded larger-sample experiment to answer that question before
fixing the runtime support choice.

The exact raw-candidate contract was:

```text
preserved u_legs_v1 source                       4,096
new independent diffusion candidates           95,904
                                                ───────
exact combined pool                            100,000
```

The extension is configured by
`config/grasp_atlas/u_legs_broad100k_extension_v1.yaml`. It uses 374 complete
256-candidate shards plus one 160-candidate final shard across 375 new,
non-overlapping seeds. `tools/run_aprilcube_raw_grasps.py` was extended, rather
than replaced, with:

- compact deterministic seed schedules;
- an explicit total candidate count;
- one resumable partial final shard; and
- per-shard provenance checks using that shard's exact expected count.

The original and extension manifests match on U mesh, right-Dex3 descriptor,
generator checkpoint, discriminator checkpoint, centered point-cloud hash,
point count, and point seed. Direct combined validation measured:

| Check | Result |
|---|---:|
| candidate records | 100,000 |
| unique candidate IDs | 100,000 |
| unique content hashes | 100,000 |
| unique transforms rounded to 1e-10 | 100,000 |
| independent generation seeds | 391 |

All six signed approach sectors are populated: `+X` 21,236, `+Y` 9,244, `+Z`
19,180, `-X` 20,281, `-Y` 10,278, and `-Z` 19,781. No neural-confidence
threshold or top-k filter was introduced.

The first 100K geometry attempt exposed a scale-only performance defect in the
4K support tool: it transformed all 76,675 dense hand-mesh vertices for every
candidate merely to compute minimum world Z. That interrupted attempt wrote no
support artifact. The corrected implementation evaluates the exact same
linear minimum on the 2,579 convex-hull vertices and batches the matrix
products. A regression test compares it against every dense vertex over saved
poses and agrees at `1e-12`. Exact FCL hand/U collision queries are unchanged.
The completed two-support audit took 79.55 s.

`config/grasp_support/u_legs_right_broad100k_v1.yaml` checked only the two
explicit broad-face supports, while retaining the complete six-support
orientation declaration. Results:

| Broad support | Raw trials | Final table-clear | Final object-clear | Full 1 mm corridor-clear |
|---|---:|---:|---:|---:|
| broad `-Y` face down | 100,000 | 1,619 | 796 | 794 |
| broad `+Y` face down | 100,000 | 722 | 190 | 189 |
| **Total physics inputs** | **200,000** |  |  | **983** |

All original 42 broad-face survivors occur in the 983 set with identical IDs,
content hashes, and `object_T_G`. The new survivors span 85 proposal buckets.

Every one of the 983 entered the same camera-free Isaac `supported_pickup`
contract used before: current right Dex3, exact U, 1 m table, VIRAL-profile
200 Hz controller/PhysX settings, settle, stored pregrasp approach, close,
20 cm lift over four seconds, and final hold. The report is complete across
four chunks of at most 256 environments:

| Physical outcome | Count |
|---|---:|
| full PASS | 0 |
| final two-digit contact | 1 |
| any hand/table contact | 340 |
| U/table contact in final hold | 982 |

Verdict combinations were 643 insufficient-contact/object-on-table, 339
insufficient-contact/hand-table/object-on-table, and one hand-table-only
failure. That sole near-miss,
`u_legs__seed_0001002849__sample_143`, visibly lifted and retained the U but
recorded a 203.103 N peak hand/table contact, so it is correctly inadmissible.
There is no PASS set to replay.

This is the stopping condition for unconditioned broad-face sampling.
Increasing the raw pool by 24.4× increased the geometry-clear physics trials
from 42 to 983 without producing one admissible pickup. The implementation
must keep the U upright on its two leg ends and use the 365 twice-passing
upright candidates; it must not ask cuRobo to repair a physically failed grasp.

Readable evidence:

- geometry: `docs/u_legs_broad100k_support_conditioned_grasp_audit.md`;
- exhaustive physics: `docs/u_legs_broad100k_supported_pickup.md`;
- ten-trial visual diagnostic:
  `docs/assets/dex3_u_broad100k_supported_pickup_review10.mp4`; and
- review explanation: `docs/u_legs_broad100k_supported_pickup_review.md`.

The exhaustive ignored ledgers are
`artifacts/grasp_support/u_legs_right_broad100k_v1/support_atlas.json` and
`artifacts/grasp_support/u_legs_right_broad100k_isaac_v1/report.json`.

## 2026-07-27 — Lightning-Grasp broad-face U assessment

Cloned the official `zhaohengyin/lightning-grasp` repository at commit
`af43818e864b0389c97b73429e5e60de2a2de593` into
`third_party/lightning-grasp` and retained its working Python 3.9/CUDA
environment at `third_party/lightning-grasp/.venv`. The release is CC BY-NC
4.0 and currently distributes its core CUDA extensions as compiled binaries.

Lightning-Grasp was worth testing because it jointly returns `G_T_object` and
a grasp-specific articulated joint vector `q`. This differs materially from
our GraspGenX use, where every pose is paired with a generic Dex3 closing
profile. Upstream supports Allegro, Shadow, LEAP, and DClaw but not Dex3. The
narrow adaptation added a current right-Dex3 robot interface, an explicit
external URDF path, deterministic seeding, and NPZ output. It reuses the exact
current descriptor at
`third_party/GraspGenX/assets/x_grippers/dex3_rev1_right/gripper.urdf`,
including its established synthetic G-to-palm transform; no new palm offset
was inferred.

A 1,024 outer × 128 inner run with 4,096 U surface points produced 287 final
solutions. An exact project-side audit evaluated each solution under both
broad-face U support transforms:

| Check | Result |
|---|---:|
| candidate/support pairs | 574 |
| final poses clear of table | 18 |
| table-clear and within 2 mm hand/U penetration | 14 |
| unique eligible candidates | 14 |
| maximum table clearance | 27.29 mm |

This is a real improvement at the final-geometry level. The visual is
`docs/assets/lightning_grasp_dex3_u_broad_face_audit_large.png`, and the
machine-readable audit is
`artifacts/lightning_grasp/u_legs_broad_face_audit_large.json`.

Lightning-Grasp does not return a pregrasp, approach, closing trajectory,
table/scene check, dynamics, or retention result. All 14 eligible
configurations therefore entered the existing VIRAL-faithful Isaac/PhysX
right-Dex3 test. The base remained at the returned final pose, the fingers
closed from the standard open configuration to the returned `q`, then lifted
20 cm and held:

| Physical outcome | Count |
|---|---:|
| full PASS | 0 / 14 |
| final two-digit contact | 0 / 14 |
| any hand/table contact | 10 / 14 |
| U still on table at final hold | 14 / 14 |

A second diagnostic selected the four closing motions with no hand/table
contact and continued their individual open-to-contact joint displacement by
5%, 10%, 20%, 35%, and 50%, clipped to exact limits. All 20 remained
table-clean, but all 20 displaced the U during closure, lost digit contact, and
left it on the table. This rules out both table contact and a simple lack of
position-controller preload as complete explanations.

Verdict: retain Lightning-Grasp as a promising second offline candidate
backend because its per-grasp `q` reaches useful flat-U final configurations,
but do not treat its final analytic outputs as executable pickups. It has not
solved the broad-face U. Using it for pickup would require support-aware
pre-shape/closure construction followed by the same Isaac qualification. The
current implementation should continue with the upright U and the existing
365 twice-qualified GraspGenX grasps.

Full assessment and evidence:

- `docs/lightning_grasp_u_assessment.md`;
- `docs/lightning_grasp_u_isaac_close_lift.md`;
- `docs/assets/lightning_grasp_dex3_u_close_lift_isaac14.mp4`;
- `docs/lightning_grasp_u_isaac_overclosure.md`; and
- `docs/assets/lightning_grasp_dex3_u_overclosure_isaac20.mp4`.

## 2026-07-29 — Corrected cuRobo grasp-domain search

The first runtime rewrite incorrectly mapped every atlas candidate to an
independent `BatchMotionPlanner` trajectory problem. Direct RTX A5500
measurements showed why that is the wrong abstraction:

| Independent CUDA problem batch | Result |
|---:|---|
| 4 | warmup and planning succeed |
| 8 | warmup and planning succeed |
| 16 | CUDA out of memory during graph warmup |
| 32 | CUDA out of memory during graph warmup |

An exhaustive left-T diagnostic using batches of eight took 156 seconds for
1,240 independent plans and returned nine successes. That experiment has been
removed from the runtime. The number eight is not a grasp limit and is not a
runtime configuration value.

The corrected implementation uses cuRobo's distinct goal-set dimension:

```text
one MotionPlanner problem
    └── up to 32 alternative world_T_G targets
            └── cuRobo returns one selected goalset_index
```

Every physics-qualified atlas entry belongs to exactly one deterministic
32-entry partition. The first lazy round submits every partition once. If
cuRobo selects a candidate, that candidate is exposed to the exact assembly
constraint search and removed from its partition. If the selected candidates
cannot complete the task, a later round requests another remaining
alternative from those partitions. A cuRobo no-solution result exhausts the
whole current partition. There is no family, neural-score, CAD-region, or
project-authored reachability gate.

The finite coordinator separates three concerns:

1. Isaac/PhysX retention determines which neural grasps enter each atlas.
2. Native cuRobo goal sets choose scene-reachable pickup candidates.
3. Singleton two-tool cuRobo IK plus locked-holder linear approaches establish
   exact T/U and T/head compatibility.

Complete sequential planning can still invalidate a connector-qualified mode
because its cached endpoint witness may not be path-connected to the realized
post-pick arm state. Backtracking is therefore scoped to the earliest failed
decision:

- T-pick failure excludes that T;
- U-pick or U-mate failure excludes that T+U prefix; and
- head-pick, head-mate, or placement failure excludes the full triple.

This matters in practice. The nominal run first selected:

```text
T    t_body__seed_0000000139__sample_125
U    u_legs__seed_0000000089__sample_089
head cube_head__seed_0000000029__sample_164
```

The exact pair qualification passed, but the realized sequential U mate
failed. The coordinator excluded that T+U prefix, did not waste time trying
six cube variants behind the identical failed U operation, and selected:

```text
T    t_body__seed_0000000139__sample_125
U    u_legs__seed_0000000169__sample_077
head cube_head__seed_0000000029__sample_164
```

That plan completed T pickup, U pickup and attachment, cube pickup and
attachment, support placement, release, and empty-hand retreat. It recorded
all six compiler state assertions and a 14-arm-joint arc-length cost of
22.3608.

A final source replay exercised one deeper backtrack: after the successful
T+U prefix, cube `...0029...164` had no pickup plan from that run's realized
joint branch. The coordinator preserved the T+U work and changed only the
cube decision to `cube_head__seed_0000000089__sample_161`. That triple
completed with joint-space arc-length cost 21.6303 and replaced the
provisional cache. A subsequent identical run loaded this one cached mode
directly and completed all stages again.

The first round considered the entire relevant atlas through alternatives
while materializing only the candidates cuRobo selected:

| Domain | Atlas entries | Goal-set requests | Selected |
|---|---:|---:|---:|
| left T | 1,240 | 39 | 18 |
| right U | 675 | 22 | 10 |
| right cube | 2,437 | 77 | 6 |

The final saved exact trajectory artifact is
`artifacts/runtime_assembly/t_u_cube_v2/nominal_goalset_v10_cached/`; its
reviewed MP4 is `full_assembly.mp4` (408 frames, 960×720, 24 fps, 17 seconds).
A one-second contact sheet was inspected: the left hand retains the T, the
right hand picks and attaches the U and cube, the completed figure is placed
on the U legs, and both hands separate. The current root suite reports 50
passing tests.

## 2026-07-29 — HERO cuRobo/Dex3 audit

Downloaded and read the full 27-page June 2026 v3 revision of HERO
(arXiv:2602.16705), including its implementation appendix. The PDF is retained
at `docs/references/papers/hero_humanoid_2026.pdf`; the detailed project
comparison is `docs/hero_curobo_dex3_audit.md`.

The decisive finding is that HERO reports approximately 20 ms for warmed
cuRobo replanning on an RTX 5070 Ti laptop. It does not use cuRobo to search
all grasps. Online AnyGrasp parallel-jaw candidates are filtered by segmented
object membership, hand-side approach, gravity-relative height, ground
parallelism, and confidence. One result is rotated 45 degrees around its
gripper z axis, yaw-clipped to 70 degrees, and sent to cuRobo as one 17-DoF
arms+waist end-effector target.

HERO therefore validates the architecture correction already identified:
offline/fast grasp selection must precede full motion generation, and cuRobo
must remain resident and CUDA-warm. Their 45-degree retarget is specific to
AnyGrasp's parallel-jaw frame and must not be applied to our exact GraspGenX
`object_T_G`.

The paper also reports real G1 analytical-FK error, learned FK/odometry,
six-second closed-loop replanning, a 1.5 cm hand-close threshold, and
MOCAP-assisted calibration of the built-in D435i necessitated by the passive
neck-pitch joint. These are important hardware lessons, but HERO's standing
29-DoF learned controller is not a drop-in seated assembly executor.

The code remains officially “Coming Soon.” The exact cuRobo version, robot
configuration, collision world, hand joint profiles, warmup lifecycle, and
the meaning of its stated `planning dt=7.25e-6` cannot yet be audited.

## 2026-08-13 — Cube-to-box milestone: kinematic visualization only

The immediate hardware milestone is now deliberately narrow: the current G1
right arm and Dex3 hand pick the actual 45 mm AprilCube cube and drop it into
an open box. Newton is not part of this milestone. cuRobo owns collision-aware
arm planning; the already existing GraspGenX MP4 renderer is used only to
inspect the saved plan.

The successful saved run is:

```text
artifacts/cube_to_box/seed7_kinematic_v4/
```

It contains all 14 planned stages from pregrasp through finger opening and
selects the VIRAL/Isaac-qualified candidate
`cube_head__seed_0000000059__sample_037`. The cube observation was randomized
within the configured tabletop region rather than placed at a single prepared
pose.

The visualization reuses the established attachment contract from
`tools/render_full_assembly.py`:

```text
world_T_object = world_T_right_hand_grasp_frame @ grasp_frame_T_object
```

Forward kinematics at the grasp agrees with the saved cuRobo
`selected_world_T_G` within `9e-8 m` translation and `1.6e-7` matrix-norm
rotation error. The cube therefore stays at its observed pose during approach
and closure, becomes rigidly attached after closure, follows the hand during
lift and transport, and detaches while the fingers open above the box.

The final downward motion into the box is explicitly a kinematic visualization,
not a physics or grasp-retention prediction: it preserves the release
orientation and interpolates vertically to a geometrically derived pose on the
box floor. The authoritative grasp qualification remains the prior Isaac test;
real contact success remains a hardware milestone.

Review artifacts:

- `docs/assets/g1_right_cube_to_box_kinematic.mp4` — 244 frames, 960×720,
  30 fps, 8.13 seconds;
- `artifacts/cube_to_box/seed7_kinematic_v4/trajectory_kinematic_attached_visible.json`;
- `artifacts/cube_to_box/seed7_kinematic_v4/timeline_kinematic_attached.json`.

## 2026-08-13 — Corrected cube runtime grasp contract

The first cube-to-box run demonstrated that the old family-balanced runtime
pool did not solve grasp selection. Families changed candidate ordering, but
cuRobo still optimized only arm reachability and motion. The selected old
candidate `cube_head__seed_0000000059__sample_037` had already moved the cube
28.3 mm and rotated it 50.7 degrees at Isaac's `closed_before_tug` phase. It
only developed a broader contact set during later tugs. That explains the bad
grasp visible in the prior kinematic video: the broad retention PASS contract
was too weak for direct execution.

Implemented a separate deterministic executable-shortlist stage in
`g1_aprilcube_demo/grasping/executable_shortlist.py`. It does not create or
modify grasp poses and it does not use family labels. It joins the immutable
arm pool to the saved Isaac traces by candidate ID, exact `object_T_G`, and
content hash, then requires:

- intrinsic retention PASS;
- no exact open collision-mesh intersection with the cube;
- exact open full-hand geometry above the tabletop at the final pose and
  along the configured straight local-Z pregrasp;
- exact closed-before-tug collision geometry above the tabletop;
- reconstructed moved-cube collision geometry above the tabletop;
- thumb and opposing-digit body contact immediately after closure;
- closure translation no greater than 22.5 mm, half the cube width; and
- closure rotation no greater than 45 degrees, half a cube-face turn.

Frame reconstruction was explicitly corrected during the audit. Isaac holds
`world_T_G` fixed during closure and stores the changing
`object_T_G = inverse(world_T_object) @ world_T_G`. Consequently:

```text
world_T_object_after_close =
    initial_object_T_G @ inverse(closed_object_T_G)
```

The object-motion and moved-cube/table gates use that pose. Closed hand/table
clearance uses the unchanged initial hand root plus the recorded closed joint
positions. An intermediate seven-candidate result had incorrectly treated the
closed relative transform as a world hand pose; it was discarded and all
reported final counts below use the corrected reconstruction.

The 2,437 right-cube retention passes reduce to 15 executable candidates.
The rejection counts are nonexclusive because one grasp may violate several
requirements:

| Gate | Rejections |
|---|---:|
| open grasp/pregrasp table corridor | 2,371 |
| closed-hand table collision | 2,147 |
| moved cube/table collision after closure | 1,669 |
| closure translation over 22.5 mm | 897 |
| closure rotation over 45 degrees | 710 |
| initial open hand/cube collision | 103 |
| missing thumb contact after closure | 97 |
| missing opposing contact after closure | 96 |

The contact force epsilon is `1e-6 N` only to distinguish a body contact from
numeric zero; force magnitude is not a quality score. The exact output is
`artifacts/grasp_shortlists/cube_right_executable_v1/shortlist.yaml` and the
complete paired proposal/Isaac-closure visual review is
`artifacts/grasp_shortlists/cube_right_executable_v1/visual/contact_sheet.png`.

The upstream end-to-end cuRobo example now accepts this shortlist format. All
15 candidates enter one goal set; there are no runtime families, no
40-candidate batches, and no 240-candidate sweep. For randomized seed 7,
cuRobo selected the first shortlist member,
`cube_head__seed_0000000159__sample_170`:

| Evidence | Value |
|---|---:|
| closure translation | 12.16 mm |
| closure rotation | 16.47 degrees |
| thumb / opposing body contact | 1.55 / 1.56 N |
| exact closed-hand table clearance | 23.81 mm |
| exact moved-cube table clearance | 3.63 mm |
| selected approach | 10 cm local-Z pregrasp |
| required lift | 20 cm world +Z |

The previous strategy table redundantly retried an identical failed 15 cm
approach for three different lift heights. It now preserves the required 20 cm
lift and varies only the pregrasp distance: 15, 10, then 7 cm. The same seed
selects the same candidate and succeeds at 10 cm.

Measured wall time from
`artifacts/cube_to_box/shortlist_final_v1/planning_report.json`:

| Boundary | Time |
|---|---:|
| cuRobo construction and CUDA warmup | 14.122 s |
| complete 15-goal grasp planning after warmup | 0.547 s |

Fresh-process warmup varied from 5.6 to 14.1 seconds in the measured runs. The
planner must therefore remain resident on hardware. Grasp planning is
approximately
10× faster than the prior 5.43 s grasp search over the 240-candidate runtime
path, but it is not yet HERO's approximately 20 ms closed-loop rate.

Current milestone truth:

- corrected selection + collision-aware pick + 20 cm lift: PASS;
- complete cube-to-box transfer with this corrected grasp: NOT YET PASS.

The configured box transport failed all eight whole-payload yaw variants and
fell back to pick-and-lift. This is a separate endpoint/transport planning
issue and must not be hidden by the older full-run visualization. The newest
pick-and-lift artifact is `artifacts/cube_to_box/shortlist_final_v1/`.

## 2026-08-13 — Constraint-driven cube-to-box transfer completed

The earlier eight-yaw transport retry was removed. It preserved the
lift-end pitch and roll, so every target inherited an arbitrary orientation;
all eight failed before trajectory optimization because no usable IK seed was
found. This did not indicate a bad grasp or an unreachable box.

The replacement is a declarative placement pipeline:

1. `task.placement_goal` declares `drop_inside`, target `bin`, free
   axis-aligned orientation, containment margin, and release clearance.
2. `end2end/placement_goals.py` generates up to 24 geometry-valid object
   poses from the actual object mesh and procedural-bin dimensions.
3. The exact selected `object_T_tool` converts every object pose to a Dex3
   tool pose without changing the grasp.
4. All 24 tool poses enter one upstream cuRobo `plan_pose` goal set.
5. cuRobo selects one reachable pose and plans the complete attached-payload
   transfer.

The open bin is now one floor plus four flared collision walls. A full AABB is
not used because it would fill the container's empty interior.

An implementation check showed that a release only 4 cm above the rim has no
collision-free right-arm/tool endpoint for the selected side grasp. The pose
12 cm above the rim is reachable and is a valid drop endpoint; gravity, not
the hand, completes `drop_inside`. This is an explicit task-geometry setting,
not a joint-space or wrist-pose hardcode.

Authoritative plan evidence is:

```text
artifacts/cube_to_box/constraint_goalset_v5/planning_report.json
artifacts/cube_to_box/constraint_goalset_v5/trajectory.json
artifacts/cube_to_box/constraint_goalset_v5/review.mp4
artifacts/cube_to_box/constraint_goalset_v5/review_contact_sheet.png
```

For randomized seed 7:

| Result | Value |
|---|---:|
| executable grasp alternatives | 15 |
| selected grasp shortlist index | 4 |
| selected grasp | `cube_head__seed_0000000159__sample_170` |
| selected approach | 10 cm local Z |
| commanded lift | 20 cm world Z |
| placement alternatives | 24 |
| selected placement | `axis_23_offset_00` |
| planner construction/warmup | 5.863 s |
| grasp goal-set planning | 0.674 s |

The same source and configuration also completed seed 19 with the cube at a
different XY/yaw observation. cuRobo selected a different shortlist member
(`cube_head__seed_0000000089__sample_163`), a 7 cm approach, and placement
`axis_14_offset_00`. The evidence is
`artifacts/cube_to_box/constraint_goalset_seed19/`.

The kinematic exporter was also corrected. It assigns the real task phase to
every frame, removes the cube from the static scene, and carries it using
`world_T_tool * tool_T_object`. After finger opening it visualizes a vertical
drop to the bin floor and records that this is symbolic kinematics, not a
physics prediction. The reviewed 18.73-second MP4 contains 281 frames at
960x720.

Focused placement tests cover the 24 unique proper rotations, preservation of
the exact selected object-to-tool transform, and the five-piece open-bin
collision model. The complete root suite reports `55 passed`.

This supersedes the immediately preceding note that the corrected shortlist
only completed pick-and-lift. The current boundary is now:

- randomized-pose collision-aware pick/lift/transfer/release plan: PASS;
- kinematic visual review with explicit non-physics drop: PASS;
- physical full-arm retention and real cube landing in the box: not yet
  tested.

## 2026-08-13 — regenerate the runtime shortlist for the printed 40 mm cube

The first pushed shortlist was found to reference the obsolete 45 mm task
mesh. It is not valid evidence for the physical cube printed on July 14. The
canonical runtime shortlist was regenerated from scratch against the released
AprilCube asset at
`third_party/aprilcube/models/dex3_safe_cube/mujoco/cube.obj`:

- exact processed mesh bounds: 40 x 40 x 40 mm;
- rounded geometry: R3 mm;
- mesh SHA-256:
  `27c8460e40a85475e87c3cc0d6090c3c9500de4fa7f5728a2462fc099ef3d927`;
- measured physical mass: 0.030 kg;
- raw GraspGenX candidates: 4,096 newly inferred poses, not rescaled 45 mm
  poses;
- VIRAL-profile Isaac retention passes: 3,178 (77.59%);
- coarse retained contact families: 46; and
- executable candidates after open-hand, table, straight-pregrasp,
  closed-hand, moved-object, closure-motion, and digit-contact gates: 15.

The printed target's textured OBJ duplicates vertices at UV/normal seams.
Trimesh processing now merges only those coincident vertices before checking
watertightness; it does not rescale or alter the authored surface. The Isaac
runner and arm-pool builder now derive the result filename from the configured
object mesh stem (`cube.yaml`) instead of assuming the old
`grasp_mesh.yaml` name. The atlas surface reader also accepts the released
cuboid detector schema directly, so the actual detector config remains the
source of face dimensions and marker IDs.

The corrected canonical output is
`artifacts/grasp_shortlists/cube_right_executable_v1/shortlist.yaml`. It
references the exact 40 mm mesh hash, records a 20 mm maximum closure
translation (half the actual cube width), contains 15 unique immutable
`object_T_G` candidates, and supersedes the 45 mm shortlist at the same
runtime path.
