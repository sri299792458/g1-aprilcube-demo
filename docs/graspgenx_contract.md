# Verified GraspGenX integration contract

This document records only contracts supported by the pinned code and our
reproduced tests.

## 1. Input to GraspGenX

For the released checkpoint:

```text
segmented object point cloud in frame F
          +
12-number gripper conditioning vector
```

The vector order is:

```text
open box extents xyz
open box center xyz
half-open box extents xyz
half-open box center xyz
```

Both checkpoint networks select `sweep_volume_v2`: a three-layer MLP whose
input dimension is 12. The URDF, meshes, collision shapes, joint values, and
closing trajectory are not consumed by inference. They are downstream assets.

For current Dex3 we use the released Unitree 12-vector as a
physics-qualified checkpoint-compatibility proxy. The exact reasoning and
ablation evidence are in `docs/dex3_rev1_descriptor.md`.

## 2. Output from GraspGenX

```text
candidate transforms: K x 4 x 4
confidence scores:     K
```

For point-cloud frame `F`, candidate `i` is `F_T_G[i]`, where `G` is the
descriptor's canonical GraspGenX root. It maps points written in `G` into `F`.

The output does **not** contain:

- a palm or wrist pose;
- seven candidate-specific finger joint values;
- a pregrasp or retract pose;
- arm IK or a trajectory;
- scene collision clearance; or
- proof of contact stability.

Confidence ranks proposals. It is not a grasp-success certificate.

## 3. Why G is not the palm

Every descriptor URDF has a root link named `world`; here that link means `G`,
not the MoveIt world or robot base. A fixed URDF joint defines `G_T_palm`.
Upstream places every hand geometry as:

```text
F_T_geometry = F_T_G @ G_T_geometry
```

Therefore the physical right palm pose is exactly:

```text
F_T_right_palm = F_T_G @ G_T_right_palm
```

There is no hand-tuned offset and no inverse. The current right and left fixed
rotations are:

```text
G_T_right_palm rotation = [[0,  1,  0],
                           [0,  0,  1],
                           [1,  0,  0]]

G_T_left_palm rotation  = [[0, -1,  0],
                           [0,  0, -1],
                           [1,  0,  0]]
```

The left transform incorporates physical mirroring so the two full open hands
occupy the same canonical convention. If a later planner controls a different
tool link, its additional fixed transform must come from the authoritative
robot model.

## 4. Proven frame behavior

GraspGenX centers the input object point cloud internally and restores its
mean to every output pose. In our deterministic translation test, adding
`[0.173, -0.081, 0.249]` m to the input changed the 60 output translations by
the same amount within 1.28 micrometres; maximum rotation difference was
0.0396 degrees and confidence difference was 0.000149.

This establishes the input/output frame behavior. It does not establish
contact or physical success.

## 5. Open, close, and approach are independent

The descriptor supplies fixed open and close dictionaries only for downstream
hand use. The halfway pose is a 50% joint-space interpolation used to describe
conditioning and inspect motion; it is not returned by the network.

A pregrasp is also not part of GraspGenX. Later scene-aware planning may offset
along canonical `-Z`, but its distance must be named and checked against the
actual object, table, robot, and approach corridor.

## 6. Candidate qualification contract

For the intrinsic hand/object checkpoint:

```text
all GraspGenX proposals
        ↓ copied unchanged
Isaac/PhysX close-and-five-tug qualification
        ↓
physically retained proposal set
```

There is no render-based closure heuristic, hand-authored pose correction, or
manual grasp selection before intrinsic physics. The qualifier uses the exact
current hand model and close trajectory, disables only intra-hand
self-collision to match upstream, and requires object contact in at least two
of the three finger chains after every tug.

For the later tabletop task, the retained set must additionally pass a
different layer:

1. transform to the selected right or left palm using the descriptor URDF;
2. plan a scene-aware pregrasp, approach, pickup, and retract;
3. reject table, robot, and non-target collisions;
4. reject arm IK/joint-limit failures; and
5. execute only the candidate selected by that planner.

These scene checks do not belong inside the intrinsic hand-only validator, and
intrinsic physics does not replace scene-aware arm planning.

## 7. Current proven state and open work

Proven for the exact current hands and 45 mm cube:

- the normal named descriptor resolves the intended 12-vector;
- all 120 GraspGenX transforms enter physics unchanged; and
- 118/120 survive on the right hand and 116/120 survive on the left hand when
  the identical canonical proposal set is replayed.

Still open:

- assembly-specific T/U selection by grasp region and connector clearance;
- exact G1 planner tool-link equivalence;
- tabletop approach and arm reachability;
- hardware force/retention calibration; and
- magnet-assisted attachment execution.
