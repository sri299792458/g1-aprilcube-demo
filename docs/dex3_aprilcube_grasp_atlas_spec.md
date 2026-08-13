# Dex3–AprilCube grasp atlas: implementation specification

Status: upright U has 365 twice-passing candidates; broad-face U has 0/983 PASS after a 100K raw-candidate test
Scope owner: G1 AprilCube demo
Objects: 45 mm-voxel `cube_head`, `t_body`, and `u_legs`
Current hand scope: current official Unitree Dex3-1 right; left explicitly deferred

## 0A. 2026-08-13 correction: runtime uses an executable shortlist, not families

The intrinsic atlas and its coarse contact families remain useful offline
evidence, but they are not a runtime grasp-selection policy. The first
cube-to-box prototype sent as many as 240 retention-pass candidates to cuRobo
in family-balanced order. That confused two separate questions:

1. **Can the hand establish and retain this grasp without badly rearranging
   the object?**
2. **Can the arm reach that already-acceptable grasp through the current
   collision scene?**

cuRobo answers the second question. It does not score Dex3 contact quality,
object motion during closure, or how convincing the resulting grasp looks.
The old run therefore selected a motion-reachable candidate whose Isaac trace
moved the 45 mm cube by 28.3 mm and rotated it by 50.7 degrees during closure.
Its apparent success came only after the object was rearranged and caught
during the later disturbance sequence. That candidate is not executable for
the cube-pick milestone even though it passed the broad retention test.

The corrected cube boundary is:

```text
2,437 intrinsic right-hand retention passes
    -> immutable candidate/provenance match
    -> exact open Dex3/cube non-intersection
    -> exact open Dex3/table clearance at grasp and pregrasp
    -> exact closed-before-tug Dex3/table clearance
    -> reconstructed closed cube/table nonpenetration
    -> thumb + opposing-digit body contact immediately after closure
    -> cube translation during closure <= 22.5 mm
    -> cube rotation during closure <= 45 degrees
    -> 15 execution-qualified candidates
    -> one cuRobo goal set
```

The two pose-change bounds are explicit coarse task semantics rather than
learned quality scores:

- 22.5 mm is half the cube's smallest 45 mm extent. Crossing it means the
  object's center has left its original half-width neighborhood.
- 45 degrees is half the 90-degree interval between adjacent cube-face
  orientations. Crossing it makes the final cube face assignment ambiguous.

The contact magnitude threshold is only `1e-6 N`, used to distinguish a
reported body contact from numerical zero. Contact force is not ranked. The
open and closed table checks use exact descriptor collision meshes. The
straight local-Z approach check uses the complete descriptor visual geometry;
because the approach is a pure translation, endpoint plane clearance proves
clearance for every point along the segment.

The Isaac trace stores `object_T_G` at each phase while `world_T_G` remains
fixed. It does not store a changing world hand pose. The closed cube pose is
therefore reconstructed as:

```text
world_T_object_after_close =
    initial_object_T_G @ inverse(closed_object_T_G)
```

Object displacement, rotation, and moved-cube/table clearance are computed
from that reconstructed pose. Closed-hand/table clearance uses the unchanged
initial `world_T_G` together with the recorded closed finger joints. A prior
draft incorrectly treated `closed_object_T_G` as a world hand pose; its
provisional seven-candidate result was discarded before finalizing this
contract.

No pose is generated, averaged, corrected, or clustered by this stage. All
15 output `object_T_G` transforms are copied unchanged from GraspGenX. The
serialization order is deterministic for inspection only. At runtime all
15 enter one cuRobo goal set simultaneously, and cuRobo selects only by arm
reachability and collision-aware motion feasibility.

Machine-readable contract and evidence:

- `config/grasp_shortlists/cube_right_executable_v1.yaml`;
- `artifacts/grasp_shortlists/cube_right_executable_v1/shortlist.yaml`; and
- `artifacts/grasp_shortlists/cube_right_executable_v1/visual/contact_sheet.png`.

The seed-7 right-arm scene selected shortlist member
`cube_head__seed_0000000159__sample_170`. Its recorded closure moved the cube
12.16 mm and 16.47 degrees, established thumb/middle contact, left 23.81 mm
of exact closed-hand table clearance, and left the moved cube 3.63 mm above
the support plane. Warm grasp planning took 0.547 seconds. Planner construction
and CUDA warmup varied from 5.6 to 14.1 seconds across the measured fresh
processes and therefore belong at process startup, not inside each perception
update.

The corrected run currently establishes collision-aware pick and 20 cm lift.
The subsequent box-transfer pose is a separate unresolved planning problem:
all configured transport-yaw variants failed for the current box placement.
The runtime must not describe that fallback as a complete cube-to-box plan.

## 0. 2026-07-25 correction: a tabletop grasp is support-conditioned

The original v1 atlas answered a free-space question: can an isolated Dex3
retain an unsupported object after closure and tugs? It did not establish that
the open hand starts outside the target, clears a supporting table, or can
approach the same pose while the object remains in its initial support.

The first runtime audit exposed two related defects:

1. the U qualification admitted trials without an explicit initial
   hand–target nonpenetration gate; and
2. runtime treated one broad-face placement as though it represented the
   graspability of the whole U.

The 675 legacy U passes and their 28 coarse contact families therefore remain
inspectable historical evidence, but they are not admissible runtime pickup
grasps. They must not seed cuRobo until they pass the replacement
support-conditioned physics path.

The replacement order is:

```text
4,096 immutable raw GraspGenX object_T_G proposals
    × six explicit U tabletop supports
    -> exact open-hand/table clearance at final and pregrasp
    -> exact open-hand/U clearance at final and pregrasp
    -> complete sampled straight local-Z approach corridor
    -> proposal buckets by support + U region + surface + approach
    -> corrected table-supported Isaac closure/lift/hold test
    -> final families from support semantics + measured hand contacts
    -> explicit per-family iteration in cuRobo
```

The U is a known axis-aligned 3×1×3 voxel part. Stable support discovery is
therefore configuration, not an inference problem:

| Table-up axis in object frame | Physical support | Symmetry class |
|---|---|---|
| `+X` | left outer leg face | outer-leg side |
| `-X` | right outer leg face | outer-leg side |
| `+Y` | broad `-Y` face | broad face |
| `-Y` | broad `+Y` face | broad face |
| `+Z` | two leg ends | leg ends |
| `-Z` | hip bridge | hip bridge |

The two members of a symmetry class remain distinct because AprilTags can
distinguish them in the real scene. Absolute table height, tabletop XY, and
in-plane yaw do not affect object-relative hand/table clearance; those enter
later as G1 reachability variables.

The first deterministic geometry pass checked all 24,576 candidate/support
pairs and retained 6,630:

| Support | Survivors |
|---|---:|
| left outer leg down | 1,691 |
| right outer leg down | 1,496 |
| broad `-Y` face down | 33 |
| broad `+Y` face down | 9 |
| upright on leg ends | 1,837 |
| inverted on hip bridge | 1,564 |

These are only geometry-clear proposals, not successful grasps. No member is
ranked or removed within the 164 pre-physics buckets. The machine-readable
ledger is generated by `tools/build_support_conditioned_grasp_atlas.py`; the
human audit and exact-support render are
`docs/u_legs_support_conditioned_grasp_audit.md` and
`docs/assets/u_legs_six_tabletop_supports.png`.

### Broad-face supported-pickup result

All 42 geometry-clear proposals from the two broad-face supports were then
executed in the table-supported Isaac/PhysX test:

- the exact U first settled under gravity on a 1 m square table;
- the exact open right Dex3 moved from the stored pregrasp to the unchanged
  GraspGenX `G` pose;
- the VIRAL-profile finger controller closed;
- the hand root rose 20 cm over four seconds; and
- physics continued for a one-second final hold.

A pass required at least two digit chains still contacting the U, no hand link
ever contacting the table, and no U/table contact during the final hold.
The measured result was **0/42 PASS**:

| Broad-face support | Trials | Final digit-contact pass | Full PASS |
|---|---:|---:|---:|
| broad `-Y` face down | 33 | 0 | 0 |
| broad `+Y` face down | 9 | 0 | 0 |

Fourteen approaches also contacted the table. All 42 U parts were again
supported by the table in the final hold; their measured table-contact force
was 2.070–2.079 N. The commanded hand lift and final orientation were verified
from independent world-frame phase poses with zero recorded command error.

The exact result ledger is
`artifacts/grasp_support/u_legs_right_broad_face_isaac_v1/report.json`; the
complete sequential review is
`docs/assets/dex3_u_broad_face_supported_pickup_all42.mp4`, summarized in
`docs/u_legs_broad_face_supported_pickup.md`.

Therefore the replacement right-U library contains no broad-face member from
the current 4,096 raw proposals. Runtime must not seed cuRobo with these flat-U
poses. This result is scoped to the current proposal set, right Dex3,
descriptor-local straight approach, and controller profile; it is not a claim
that every possible flat-U strategy is physically impossible.

#### Bounded 100K broad-face follow-up

Because 42 physical trials were too few to rule out a rare broad-face proposal,
one controlled larger-sample experiment was run before abandoning that
placement. It kept all original 4,096 transforms and generated 95,904 new
diffusion candidates from 375 non-overlapping seeds:

```text
4,096 preserved candidates + 95,904 new candidates = 100,000
100,000 candidates × two broad-face supports = 200,000 support trials
200,000 exact geometric trials -> 983 Isaac pickup trials
983 Isaac pickup trials -> 0 PASS
```

The extension used exactly the original U mesh, current-right-Dex3 descriptor,
generator/discriminator checkpoints, sampled point cloud, thresholds, and
outlier policy. All 100,000 candidate IDs, content hashes, and numerical
transforms are unique. The approach sectors cover all six signed object axes.
The geometry gate retained 794 candidates for the broad `-Y` face and 189 for
the broad `+Y` face, distributed over 85 proposal buckets.

Isaac measured:

| Gate | Count |
|---|---:|
| Physics trials | 983 |
| At least two final digit chains | 1 |
| Any hand/table contact | 340 |
| U/table contact in final hold | 982 |
| Full PASS | 0 |

The sole retention trial contacted the table with the hand and recorded a
203.1 N peak hand/table force, so it is not admissible. With no PASS set, there
is nothing to replay. This is the stopping condition for unconditioned
GraspGenX sampling on a broad-face U: increasing the raw pool 24.4× increased
the physical trial set from 42 to 983 but did not produce one valid pickup.

The immutable extension and geometry configurations are
`config/grasp_atlas/u_legs_broad100k_extension_v1.yaml` and
`config/grasp_support/u_legs_right_broad100k_v1.yaml`. Readable results are
`docs/u_legs_broad100k_support_conditioned_grasp_audit.md` and
`docs/u_legs_broad100k_supported_pickup.md`. The compact ten-trial diagnostic,
including the retention/table-collision near-miss, is
`docs/assets/dex3_u_broad100k_supported_pickup_review10.mp4`.

### Upright-on-leg-ends supported-pickup result

The same unchanged physical contract was applied to all 1,837 geometry-clear
proposals for the U standing upright on both leg ends. The discovery run, in
256-environment batches, produced **405 PASS**. Every one of those 405 was then
replayed under a different 64-environment layout:

| Gate | Input | PASS |
|---|---:|---:|
| Upright discovery | 1,837 | 405 |
| Exhaustive replay of discovery passes | 405 | 365 |

The 40 non-reproducing discoveries are excluded. The 365 twice-passing records
cover the hip bridge (149), left leg (147), and right leg (69), five approach
sectors, and 14 proposal buckets. This is now the support-conditioned
right-Dex3 U library eligible for later cuRobo reachability and connector
compatibility checks.

For human review, the highest-score twice-passing member of each bucket was
executed a third time with Isaac cameras enabled. Thirteen of 14 lifted and
held the U again; the one failed segment remains preserved in the complete
review. The readable pass-only sequence is
`docs/assets/dex3_u_upright_supported_pickup_review13_passes.mp4`.

The exhaustive ledgers are
`artifacts/grasp_support/u_legs_right_upright_isaac_v1/report.json` and
`artifacts/grasp_support/u_legs_right_upright_replay1_isaac_v1/report.json`.
The readable summaries are `docs/u_legs_upright_supported_pickup.md` and
`docs/u_legs_upright_supported_pickup_replay1.md`.

For the replacement final U library, the family key must include the support
orientation, semantic U component (`left_leg`, `right_leg`, or `hip_bridge`),
surface/cavity relation, approach sector, and the measured post-lift
digit/palm participation. The semantic ray label before physics is only a
proposal-bucket label; it is not a claim about actual finger contact.

## 1. Decision

The milestone is a **coarse-contact, physics-qualified grasp atlas for the
cube, T body, and U legs**, built from GraspGenX diffusion proposals and the
existing Isaac/PhysX validator.

The atlas is not a directory of wrist poses and it is not a task planner. It
answers a smaller, concrete question:

> For the exact current right Dex3 and the 45 mm-voxel AprilCube parts, which neural
> grasp proposals survive physical closure and tugs, where do the fingers
> participate in holding the object, which broad object surfaces are involved,
> and what distinct kinds of grasp do those robust signals represent?

The fixed pipeline is:

```text
GraspGenX diffusion proposals
        │
        │ every proposal, unchanged
        ▼
existing GraspDataGen Isaac/PhysX close-and-five-tug simulation
        │
        ├── pass/fail and final hand/object state
        └── measured body/chain contact presence at phase boundaries
             (raw solver points retained only as diagnostics)
                    │
                    ▼
deterministic AprilCube surface mapping
                    │
                    ▼
contact signatures and grasp families
                    │
                    ▼
sequential MP4 + machine-readable representatives
```

No OBB grasps, hand-written grasp poses, render-based closure tests, robot arm,
cuRobo, or assembly logic belongs in the intrinsic-atlas milestone. The
replacement support-conditioned admission stage adds only the supporting table
and prescribed approach/lift needed to answer whether an intrinsic proposal is
a real pickup from a named support.

## 2. Why the implementation started with the cube

The final demo needs grasps for the T, U, and cube. The cube was the right
first object because it isolated the atlas machinery:

- it already has a validated current-Dex3 baseline on both hands;
- its six physical surface regions are unambiguous;
- its measured mass is known: 30 g;
- it does not introduce T/U component naming or connector-clearance policy;
- a bad contact transform or family label is easier to see on a cube; and
- the same schemas and programs can process T/U after the visual and numerical
  cube gate passes.

The right-hand 256-candidate cube vertical slice passed its engineering and
visual-output checks. The same object-neutral implementation was then applied
unchanged to the full cube, T, and U proposal sets. Code may depend on the
existing AprilCube voxel/marker metadata, but it must not contain conditionals
such as `if object == "cube_head"` in the contact or family logic.

## 3. Existing facts this specification treats as fixed

These are already established and are not reopened by the atlas work:

1. The physical parts are the generated watertight meshes under
   `generated/aprilcube_parts/{cube_head,t_body,u_legs}/grasp_mesh.obj`.
2. The cube mass is the measured 0.030 kg filament mass. T and U currently use
   explicitly provisional mesh-volume-scaled masses; finished prints and any
   future magnets require new recorded masses rather than silent changes.
3. The current physics hand is the generated official Unitree Dex3-1 right
   descriptor at
   `third_party/GraspGenX/assets/x_grippers/dex3_rev1_right`.
4. Their exact URDFs, collision meshes, signed joint names, open/close
   profiles, `G_T_palm` transforms, and Isaac assets are authoritative.
5. GraspGenX outputs `object_T_G`; it does not output a palm pose, finger
   contact state, pregrasp, or trajectory.
6. The right and left descriptors have the same canonical GraspGenX
   conditioning, so this canonical proposal set can later be reused for the
   left hand. The present physics run is right-only by explicit project choice.
7. The released Unitree 12-number conditioning vector remains the selected
   checkpoint-compatible proxy. The descriptor audit is not repeated here.
8. Physics, not a terminal closed-hand render, decides whether the fingers are
   stopped by and retain the object.
9. The existing validator's success rule remains unchanged: after closure and
   the complete five-tug sequence, the object must still produce object-filtered
   contact in at least two of the thumb, index, and middle chains.
10. The known canonical checkpoint remains a regression fixture: 118/120
    right and 116/120 left at its recorded configuration.

The frame contract is defined in `docs/graspgenx_contract.md`; the descriptor
root-cause evidence is defined in `docs/dex3_rev1_descriptor.md`. This document
uses those contracts rather than restating or replacing them.

## 4. What v1 implements and what it deliberately defers

| Research proposal | Atlas v1 | Reason |
|---|---|---|
| Thousands of diffusion proposals | Yes | Required to discover contact diversity |
| GraspMoE OBB branch | No | It is not a learned Dex3 contact proposal and previously obscured the pipeline |
| Physics evaluation of every proposal | Yes | This is the qualification boundary |
| Per-body contact presence | Yes | Robust coarse evidence of which hand region participates |
| Exact contact points, normals, and forces | Diagnostic only | PhysX exposes them, but they are too noisy to drive atlas decisions |
| Contact snapshots through the tug test | Yes, at named phase boundaries | Enough to measure persistence without recording every physics frame |
| AprilCube surface regions | Yes, generated voxel faces | Required for object-centric labels |
| Deterministic contact families | Yes | Required to expose diversity to later planning |
| Representative real grasps per family | Yes | Required for visual review and later goal sets |
| Pose/friction/mass/controller perturbation campaign | No | The real uncertainty distributions are not yet measured |
| Learned or general human grasp taxonomy | No | Unnecessary for three-finger Dex3 |
| Connector exposure and collision corridors | No | Magnet faces and connector geometry are not fixed |
| Stable-placement graph | No | Not needed to prove the grasp atlas |
| Pick/place/attach/regrasp compatibility graph | No | Belongs after T/U regions and connectors exist |
| T and U production atlases | Yes, using the same pipeline | Confirms the implementation is object-neutral |
| Table collision and pregrasp planning | No | Scene-aware cuRobo gate |
| G1 arm reachability | No | Scene-aware cuRobo gate |
| Hardware execution | No | Later ROS 2 bridge and calibration gate |

Deferred fields must not be filled with guesses. In particular, v1 must not
invent connector roles, friction distributions, robustness percentages, or
assembly compatibility labels.

## 5. Terms and transform convention

### 5.1 Candidate

A candidate is one stochastic neural proposal with:

```text
object_T_G          4 × 4 rigid transform
graspgenx_score     discriminator score
generation_seed     exact torch/NumPy seed
batch_id            inference batch identity
sample_id           identity within that batch
```

`G` is the descriptor's canonical GraspGenX root. It is not the palm, wrist,
MoveIt world, or G1 pelvis.

### 5.2 Physics trial

A physics trial is one candidate replayed with one physical hand side. The
right and left trials share the candidate identity but have different hand
assets, signed close trajectories, contacts, and pass/fail results.

### 5.3 Contact observation

A contact observation is solver-reported hand-to-object contact for one named
Dex3 body at one named phase. Aggregate object-filtered contact presence is the
authoritative signal. Raw points are transformed from simulator world
coordinates into the instantaneous object frame and saved for diagnosis only.

It is not inferred from mesh overlap or finger proximity.

### 5.4 Contact signature

A contact signature separates robust body-level signals used by the family key
from detailed-point annotations:

```text
participating digit chains
palm contact present/absent
object-frame approach sector

diagnostic only:
digit chain or palm → zero or more broad AprilCube surface neighborhoods
```

### 5.5 Grasp family

A family contains physics-passing trials with the same coarse contact
signature. Exact link, point, normal, force, finger state, and small pose
differences do not create families.

## 6. Candidate generation contract

### 6.1 Production budget

The first production run generates **4,096 canonical cube proposals**:

```text
16 deterministic inference batches × 256 proposals = 4,096 proposals
```

4,096 is a practical initial coverage budget inside the researched 1,000–5,000
range, not a scientific threshold. v1 does not implement an automatic
family-saturation stopping rule; one run must be reproducible before adaptive
stopping is useful.

### 6.2 Fixed inference behavior

Extend the existing `tools/run_aprilcube_raw_grasps.py`; do not create a second
GraspGenX wrapper.

For every batch:

- use the pinned released generator and discriminator checkpoints;
- use `dex3_rev1_right` only as the canonical conditioning descriptor;
- use the same deterministic 3,500-point sample of the cube surface in all 16
  batches;
- vary only the recorded inference seed;
- request 256 generated proposals;
- set `grasp_threshold=-1.0`;
- retain all 256 proposals (`topk_num_grasps=256`);
- set `remove_outliers=False`;
- do not invoke GraspMoE;
- do not generate OBB candidates; and
- do not deduplicate or reject candidates before physics.

The discriminator score is recorded but is not an admission gate. If a batch
returns fewer than 256 proposals, the batch fails loudly; it is not silently
padded or replaced.

### 6.3 Stable identity

Rank names such as `grasp_17` are not globally unique across batches. Every
candidate receives this stable identity:

```text
cube_head__seed_<10-digit seed>__sample_<3-digit index>
```

The manifest also stores a SHA-256 content identity computed from:

```text
object mesh hash
descriptor config hash
checkpoint hashes
point-cloud sample hash
generation seed
float64 object_T_G values
```

The readable ID is used in videos and reports. The content hash detects
accidental mutation or collision.

### 6.4 Sharding and resumability

One 256-proposal inference batch is one raw YAML shard. A completed shard has:

- the candidate YAML;
- a JSON provenance record;
- hashes for all inputs and outputs; and
- a `complete: true` marker written only after validation of its count and
  transforms.

Runs resume by accepting only complete shards whose hashes match the current
manifest. A mismatched shard causes an error; it is never overwritten
implicitly.

## 7. Physics qualification contract

### 7.1 Reuse boundary

The simulation remains the released GraspDataGen
`scripts/graspgen/grasp_sim.py`, including the local changes already required
for exact multi-joint Dex3 targets, mass propagation, per-digit contact groups,
and camera capture.

The only new upstream change permitted by this specification is a narrow,
optional **contact-trace output**. It may observe and serialize simulation
state; it must not modify control targets, collision behavior, solver
configuration, timing, tug forces, or success logic.

The root repository remains responsible for orchestration, manifests, surface
mapping, grouping, and reports. It must not reimplement the simulator.

### 7.2 Exact hand trials

The current milestone evaluates each of the 4,096 canonical proposals once:

```text
4,096 × current right Dex3
```

Left-hand qualification is deliberately deferred until the task actually
requires it. The candidate transform is copied unchanged into the Isaac input.
The descriptor URDF supplies the fixed `G_T_palm` relationship.

Each input records:

- the raw shard and hash;
- the exact hand descriptor and hash;
- the complete generated USD package hash, not only the root USD hash;
- the object mesh and hash;
- mass 0.030 kg;
- the exact open and close dictionaries;
- `use_cspace_position_as_target: true`;
- the three digit contact groups; and
- `min_contact_groups: 2`.

### 7.3 Solver and test sequence

Production uses the named `viral_g1_43dof_92bf086` profile. It reproduces the
settings that the released GR00T-VisualSim2Real Isaac adapter actually applies,
not every field merely declared in its YAML:

```text
physics frequency              200 Hz (dt = 0.005 s)
source command decimation      4 (50 Hz target updates)
solver                         TGS
articulation iterations        4 position, 0 velocity
initial close/settle duration  1.0 s
disturbance magnitude          1 × object weight
gravity                        disabled in the hand-only test
self-collision                  disabled as in the released G1 mapping
object collision approximation convex decomposition
object friction                1.0 static, 1.0 dynamic
object contact/rest offsets    0.002 m / 0.0 m
hand–object collision           enabled
```

The seven finger actuators use the released implicit actuator path: thumb-0
`kp=2.0`, `kd=0.1`, `2.45 Nm`; the other six joints `kp=0.5`, `kd=0.1`,
`1.4 Nm`. Velocity limits, the adapter's applied `3×` armatures, and zero
joint friction are also copied. VIRAL may update targets at 50 Hz; this static
qualifier sends one constant `q_close`, which Isaac's implicit drive retains
through every 200 Hz physics step. The declared global
`contact_offset=0.01`, global `rest_offset=0`, effort-scale `0.95`, and
`idealpd` label are recorded as dormant because the released adapter does not
apply them on this path.

The five 0.5 s object-frame tugs are unchanged:

```text
+Z
+2Y + Z
-2Y + Z
+2X + Z
-2X + Z
```

The direction vector is normalized by the upstream parser and scaled by the
configured force multiple. The existing validator decides pass/fail at the end
of the complete sequence. Contact tracing must not strengthen, weaken, or
replace that rule.

### 7.4 Batch size

Inference shard size and physics parallelism are separate. Physics starts with
at most 256 environments per process because that scale is already practical
for the exact hand. The driver may reduce this after an out-of-memory failure,
record the actual value, and resume the shard. It may not change any physical
parameter as a recovery action.

### 7.5 Required phase snapshots

The optional trace records these six phase boundaries:

```text
closed_before_tug
after_tug_1
after_tug_2
after_tug_3
after_tug_4
after_tug_5_final
```

Recording every 200 Hz frame is deliberately out of scope. Six snapshots are
enough to distinguish a persistent contact family from a grasp that changes or
loses contacts during a particular disturbance.

### 7.6 Links instrumented

Object-filtered contact sensors cover the imported articulation's eight
physical bodies:

```text
world  → logical palm
thumb_0, thumb_1, thumb_2
index_0, index_1
middle_0, middle_1
```

The URDF's empty `world` link is fixed to the palm, and the importer is
configured with `merge_fixed_joints: true`; therefore PhysX carries the palm
collision mesh on the runtime body `world`. The input records an explicit
`world → {side}_hand_palm_link` alias. This was verified from the runtime body
list and a traced physics run, not inferred from naming. All other names are
side-specific and come from the descriptor URDF.

### 7.7 Data recorded per trial

Every trial, pass or fail, produces one record containing:

```yaml
candidate_id: cube_head__seed_...__sample_...
candidate_content_sha256: ...
object_id: cube_head
hand_side: right

input:
  object_T_G: [[...], [...], [...], [...]]
  graspgenx_score: 0.0
  open_q: {...}
  target_close_q: {...}

result:
  passed: true
  final_q: {...}
  final_object_T_G: [[...], [...], [...], [...]]

phases:
  - name: closed_before_tug
    object_T_G: [[...], [...], [...], [...]]
    contacts:
      - hand_link: right_hand_thumb_2_link
        physx_body: right_hand_thumb_2_link
        net_normal_force_world_N: [fx, fy, fz]
        contact_force_magnitude_N: 0.75  # max norm over object-filtered pairs
        points:                         # diagnostic, never a pass/family signal
          - position_object_m: [x, y, z]
            normal_object: [nx, ny, nz]
            normal_force_N: 0.0
            separation_m: 0.0
```

`object_T_G` naming follows the established project contract. If the upstream
simulator internally stores the inverse relationship, the adapter must convert
once at serialization and unit-test the conversion.

Body/chain participation uses `contact_force_magnitude_N > 0`, where the
scalar is the maximum norm over that body's object-filtered contact pairs.
It is the same non-cancelling quantity used by the simulator's PASS test.
The aggregate vector remains diagnostic only because opposing contacts can
cancel when summed. Exact solver points, normals, separations, and scalar
force sizes are provenance/debugging data; the scalar is interpreted only as
zero versus nonzero.

### 7.8 Contact-point semantics

Use Isaac Lab's object-filtered aggregate contact data for body/chain presence.
Detailed contact tracking is enabled only to annotate a participating body
with a broad cube-face neighborhood and to diagnose frame errors. Save only
solver-populated slots; zero-padded slots are not contacts.

For each contact point:

```text
p_object = inverse(world_T_object_at_that_phase) × p_world
```

The object pose and contact point must come from the same simulation step. A
bad or absent detailed point never changes a validator pass/fail result or
removes the associated body/chain presence.

## 8. AprilCube surface-region contract

### 8.1 Reuse existing metadata

The cube's generated `config.json` already supplies, for every exposed voxel
face:

- voxel index;
- face name (`+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`);
- face normal;
- face corners; and
- AprilTag ID.

Do not introduce generic semantic segmentation. Build a small deterministic
sidecar from this existing source.

### 8.2 Stable region identity

The six cube region IDs are:

```text
cube_head/v_0_0_0/+X
cube_head/v_0_0_0/-X
cube_head/v_0_0_0/+Y
cube_head/v_0_0_0/-Y
cube_head/v_0_0_0/+Z
cube_head/v_0_0_0/-Z
```

Each region record also carries its current AprilTag ID. The region identity
does not use the tag ID, because tags may be reassigned without changing the
physical surface.

### 8.3 Rounded edges and corners

The physical mesh has a 3 mm fillet, so not every diagnostic point lies on an
ideal planar face. Map each point to the nearest planar face rectangle and
save:

```text
primary_region
distance_to_primary_region_m
nearby_regions_within_fillet_radius
normalized face coordinates (u, v), unclamped and clamped
```

The nearby-region radius is the object's declared 3 mm fillet radius plus a
documented floating-point tolerance. A corner observation may therefore retain
three broad face labels. Exact primary face, `(u,v)`, and distance remain
diagnostic and never form a family key.

The mapper also compares the point with the watertight grasp mesh. A solver
point farther than 0.5 mm from the mesh is marked invalid and excluded from
the broad-face annotation, but the atlas build and the body's coarse presence
remain valid. Mapping rates are reported so systematic frame errors remain
visible.

### 8.4 No connector roles in v1

All six cube faces have role `unassigned`. We do not label a magnetic
connector, tag-safe face, or grasp-forbidden face until the printed connector
design is fixed. Because raw region IDs are retained, roles can later be added
without rerunning GraspGenX or physics.

## 9. Contact signature and family construction

### 9.1 Which phase defines the family

The family signature uses `after_tug_5_final`, the same post-disturbance state
whose aggregate body contacts determine physics PASS. This guarantees that a
family describes the contact pattern that actually survived qualification.

`closed_before_tug` and every intermediate snapshot produce a persistence and
contact-transition annotation. They remain available to distinguish a grasp
that began with its final pattern from one that acquired or lost a digit/palm
contact during the tugs, but they do not split the qualified family.

### 9.2 Discrete family key

For each physics-passing trial, construct this exact family key:

```yaml
digit_chains: [thumb, index, middle]
palm_contact: false
approach_sector: +X
```

Store the detailed-point mapping separately as an annotation, for example:

```yaml
diagnostic_broad_faces_by_chain:
  thumb: [cube_head/v_0_0_0/+Y]
  index: [cube_head/v_0_0_0/-Y]
  middle: [cube_head/v_0_0_0/-Y]
```

Rules:

1. Broad regions for a chain are a sorted union of all valid detailed-point
   neighborhoods on that chain, but are diagnostic and do not split families.
2. Proximal/distal exact link identity does not split a family.
3. `digit_chains` comes from aggregate object-contact presence even if detailed
   point buffers are empty or invalid.
4. Palm contact is explicit rather than inferred from missing finger contact.
5. The continuous approach vector is the canonical GraspGenX approach axis
   transformed into the object frame.
6. `approach_sector` is the signed object axis with the largest absolute vector
   component. The full unit vector is retained separately.
7. Right and left trials are never merged into the same family record. They
   may share a human-readable signature but remain side-specific libraries.
8. Cube symmetries are not collapsed; different tag-bearing physical faces
   remain distinct in the diagnostic annotation.

This lexicographic key is intentionally simple. v1 does not run K-means, learn
embeddings, or force Dex3 into a human-hand taxonomy.

### 9.3 Contact persistence

For every initially participating digit chain or palm, store a six-bit phase
mask indicating whether that same coarse hand region still has any
object-filtered contact. Broad face annotations may also be displayed at each
phase but do not define persistence.

Examples:

```text
111111  present before and after every tug
111000  lost after tug 3
101111  temporarily changed at tug 2
```

This is descriptive. v1 pass/fail continues to come from the upstream
validator's final rule.

### 9.4 Continuous members and representatives

All passing members are preserved. For visual review and later planning, each
family exposes at most three actual members:

1. **primary** — greatest coarse contact persistence, then neural score;
2. **translation-diverse backup** — the remaining real member farthest from
   the primary in object-frame grasp translation; and
3. **pose-diverse backup** — the remaining real member with the largest
   rotation-geodesic distance from the primary's `object_T_G`.

Exact PhysX contact points or centroids do not choose representatives.

Tie-breaking order is:

```text
greater total coarse presence across the six phase snapshots
higher GraspGenX score
lexicographically smaller candidate ID
```

Representatives are never averaged transforms. Each one is a candidate that
actually passed physics. Families with one or two members expose only the
members they have.

## 10. Machine-readable artifacts

The implementation writes under:

```text
artifacts/grasp_atlas/<cube_v1|t_body_v1|u_legs_v1>/
├── manifest.json
├── surface_regions.json
├── raw/
│   ├── shard_000.yaml
│   ├── shard_000.provenance.json
│   └── ... shard_015 ...
├── right/
│   ├── isaac_inputs/
│   ├── physics_outputs/
│   ├── contact_trials.jsonl
│   ├── families.json
│   └── representatives.yaml
```

### 10.1 Why YAML, JSON, and JSONL are all used

- YAML remains only at the GraspGenX/GraspDataGen boundaries because those
  upstream tools already consume it.
- JSON stores manifests and family summaries that are read as one document.
- JSONL stores one independent physics trial per line, supports streaming, and
  avoids adding a Parquet/HDF5 dependency for 4,096 trials per object and
  12,288 trials in the current right-hand production set.

### 10.2 Provenance manifest

The top-level manifest records:

- git commits for the root repo, GraspGenX, and GraspDataGen;
- checkpoint filenames and hashes;
- mesh, descriptor, URDF, USD-package, and object-config hashes;
- point-cloud sample hash;
- all seeds and candidate counts;
- all physics parameters and actual batch sizes;
- contact sensor configuration;
- surface-mapping tolerance;
- family schema version;
- stage status and timestamps; and
- hashes of every final summary artifact.

No result is called reproducible if this manifest is incomplete.

### 10.3 Repository policy

Commit:

- the config;
- source code;
- manifests;
- surface metadata;
- family summaries;
- representative YAML;
- readable review media.

Do not commit all simulator caches, USD conversion intermediates, or thousands
of redundant camera frames. Raw and physics shards are reproducible evidence
and may remain local unless a later data-release decision says otherwise.

## 11. Visual review contract

Numbers alone are insufficient for this gate. The current right-hand build
produces:

```text
docs/assets/dex3_cube_grasp_families_right_viral.mp4
docs/assets/dex3_t_body_grasp_families_right_viral.mp4
docs/assets/dex3_u_legs_grasp_families_right_viral.mp4
```

### 11.1 Machine-readable atlas

Each object/hand atlas records:

- total generated, passed, failed, and mapped trials;
- number and size distribution of contact families;
- one record per family;
- the coarse family signature;
- the primary and available pose-diverse backups;
- GraspGenX score, final finger state, and persistence mask;
- the named phase observations through all five tugs; and
- diagnostic broad object-face mappings derived from the saved contact data.

The report must clearly label neural confidence, physics pass, and family size
as different quantities.

### 11.2 Sequential MP4s

Each object receives one readable right-hand MP4. Family representatives
appear one after another, not in an unreadable simultaneous grid. Each segment
includes:

```text
hand side
family ID and member count
candidate ID
contacted broad object-face neighborhoods
participating digits/palm
approach sector
phase name
final PASS result
```

The frames come from a camera-enabled rerun through the same Isaac validator,
candidate, hand asset, and physics configuration used for qualification. Its
green/red result is retained as a video-generation diagnostic only. It does
not create a second qualification category, change the original atlas verdict,
or gate downstream planning. A separate hand-authored preview renderer is not
introduced.

### 11.3 Visual gate

Before arm-planning work resumes, review must establish that:

1. contact dots/labels agree with visible finger/object contacts;
2. broad object-face names agree with the displayed object orientation;
3. family cards that claim different contacts visibly differ;
4. representatives are real passing simulations; and
5. the right hand is not mirrored or transformed incorrectly.

## 12. Configuration and program boundaries

### 12.1 One project configuration

Use one checked-in file per object:

```text
config/grasp_atlas/cube_viral_v1.yaml
config/grasp_atlas/t_body_viral_v1.yaml
config/grasp_atlas/u_legs_viral_v1.yaml
```

It names inputs and reproducibility parameters. It must not duplicate the hand
joint values already stored in the descriptor configs; it references and
hashes those configs.

### 12.2 Existing programs to extend

1. `tools/run_aprilcube_raw_grasps.py`
   - add deterministic multi-seed sharding;
   - retain every requested diffusion proposal;
   - emit stable candidate IDs and per-shard provenance.

2. `tools/build_dex3_isaac_grasp_input.py`
   - accept a hand side and raw shard;
   - preserve candidate IDs and transforms;
   - set the measured mass and side-specific contact groups;
   - emit one Isaac input per shard.

3. `third_party/GraspDataGen/scripts/graspgen/grasp_sim.py`
   - add optional contact-point tracking;
   - snapshot phase-boundary contact and object state;
   - write a sidecar record for every trial;
   - leave physics and pass/fail logic unchanged.

### 12.3 New project programs

Keep new code to two focused programs:

1. `tools/build_grasp_atlas.py`
   - build surface metadata from generated AprilCube config;
   - normalize physics sidecars;
   - map contacts to regions;
   - form families and choose real representatives;
   - validate all invariants and write summaries.

2. `tools/render_grasp_atlas.py`
   - select representative trials;
   - invoke the existing Isaac camera replay for named candidates;
   - assemble the sequential MP4 and machine-readable review manifest.

Do not add a general database server, ROS package, web framework, plugin
system, or task-planning abstraction for this milestone.

## 13. Tests and acceptance gates

### Gate A — pure-data tests, no GPU

Required tests:

1. Multi-seed candidate IDs are unique and stable.
2. A raw candidate's `object_T_G` is numerically unchanged in the right and
   left Isaac input.
3. Candidate content hashes change if and only if a declared input changes.
4. World-to-object contact conversion passes synthetic transform cases.
5. The centers of all six cube faces map to the correct region and `(u,v)`.
6. Rounded edge/corner examples report the expected nearby region sets.
7. Family construction is independent of input record order.
8. Every representative ID names a passing member of its family.
9. No right-hand joint/link name appears in a left-hand common record, or vice
   versa.

### Gate B — contact instrumentation regression

Run the existing ten-candidate canonical visual fixture: eight known right-hand
passes plus the two known right-hand failures. Use its original recorded mass
and physics configuration.

Acceptance:

- enabling contact traces produces the same eight/two result;
- disabling contact traces produces byte-equivalent ordinary physics output
  where timestamps are excluded;
- passing trials have at least two final digit groups, matching the existing
  rule;
- diagnostic contact points have a reported mesh-mapping rate; and
- every record preserves its original `graspgenx_source`.

This test proves that observation did not alter simulation behavior.

### Gate C — 256-proposal production smoke test

Run one cube shard through the current right hand at 0.030 kg.

Acceptance:

- exactly 256 canonical proposals enter the right-hand physics run;
- no pose is discarded between inference and physics;
- 256 pass/fail trial records are written;
- point mapping statistics are reported without changing any verdict;
- family construction and rendering complete; and
- the short sequential MP4 is reviewed before the remaining shards run.

### Gate D — complete right-hand cube/T/U atlases

Run all 16 shards for each object.

Acceptance:

- exactly 4,096 unique canonical candidates exist per object;
- exactly 12,288 right-hand physics outcomes exist in total;
- every outcome is traceable to one unchanged raw proposal;
- every passing outcome has a valid contact signature;
- each family has one to three actual representative candidates;
- manifests and artifact hashes are complete;
- all three full sequential MP4s are readable; and
- the machine-readable summaries expose every family without requiring raw
  simulator-log inspection.

Passing Gate D completes intrinsic right-hand grasp generation. It does not
authorize execution by itself: physics-passing atlas members must still pass
task-scene, approach, and reachability checks.

## 14. Failure behavior

The pipeline fails loudly on:

- a missing or changed pinned input;
- fewer proposals than requested;
- duplicate candidate IDs or hashes;
- a transform changed between stages;
- missing physics results;
- an invalid or non-rigid transform at an artifact boundary;
- unknown hand link names;
- a passing trial without a valid contact signature;
- a representative that did not pass physics; or
- an attempt to resume from a shard with mismatched provenance.

A simulator crash leaves the current shard incomplete and resumable. The
driver never deletes prior artifacts automatically and never silently starts a
new experiment under an existing atlas ID.

## 15. Uncertainty ledger

### Known knowns

- The GraspGenX frame contract is verified.
- The current exact right and left hand assets are available.
- Both hands retain most of the existing cube proposal set.
- The cube geometry, face metadata, and measured mass are known.
- The upstream simulator already batches candidates and exposes per-link
  object-filtered forces.

### Known unknowns resolved by this milestone

- How many distinct contact families 4,096 proposals produce.
- How often detailed point buffers are missing or inconsistent even when the
  aggregate body contact signal is present.
- How often contact patterns change during the five tugs.
- How the right-hand family distribution changes across cube, T, and U.
- Whether 256 exact-hand environments remain the best stable production batch
  size with all link sensors active.

### Known unknowns deliberately left for later

- Real hardware friction and grip-force calibration.
- Perception error distributions.
- Magnet locations and attachment wrench directions.
- Which cube faces must remain exposed for the final assembly design.
- Table-clear approach geometry and G1 arm reachability.

### Unknown-unknown containment

- preserve all raw proposals and physics outcomes;
- hash every boundary artifact;
- keep pass/fail separate from family labels;
- use real physics-passing candidates rather than synthesized averages;
- make phase traces optional so they can be A/B tested; and
- require a human-readable visual gate before arm planning.

## 16. Definition of done

This milestone is complete only when the machine-readable atlases and MP4s can
answer, for the current right Dex3 and each of cube, T, and U:

1. Which of the 4,096 GraspGenX proposals passed physics?
2. Which digit chains/palm participated, and near which broad object faces?
3. Which coarse hand-region contacts persisted through each disturbance?
4. What distinct contact families exist?
5. Which real, physics-passing candidates represent each family?
6. Can we visually inspect every family without trusting a pose-only render?
7. Can every displayed result be reproduced from pinned, hashed inputs?

Once those answers are visually approved, the next specification revision can
add only the task-scene filters needed to choose among these intrinsic grasps:
table and approach clearance, G1 reachability, and connector exposure.
