# HERO cuRobo/Dex3 implementation audit

Date read: 2026-07-29

Paper: Runpei Dong, Ziyan Li, Arjun Gupta, Xialin He, and Saurabh Gupta,
“HERO: Learning Humanoid End-Effector Control for Visual Whole-Body
Open-Vocabulary Object Grasping,” arXiv:2602.16705v3, 3 June 2026.

Primary sources:

- project page: <https://hero-humanoid.github.io/>
- arXiv: <https://arxiv.org/abs/2602.16705v3>
- local v3 PDF: `docs/references/papers/hero_humanoid_2026.pdf`
- local machine-readable paper:
  `docs/references/papers/hero_humanoid_2026.md`
- PDF SHA-256:
  `4a6213d5694f11e6385d479d8c372c277b1b010d8131cc06940976299e42369f`

The full 27-page v3 paper and appendices were read. The official project page
still labels the code “Coming Soon,” so several operational details cannot yet
be verified against an implementation.

## Bottom line

HERO strongly confirms that cuRobo should not take minutes in a deployed G1
pipeline. The paper reports approximately 20 ms for a warmed cuRobo replan on
a laptop RTX 5070 Ti and runs such replanning every six seconds during
hardware execution.

It achieves this because cuRobo is not its grasp-search engine. HERO selects
one grasp before invoking cuRobo:

```text
RGB-D + language
  -> Grounding DINO detection
  -> SAM-3 segmentation
  -> AnyGrasp parallel-jaw proposals
  -> geometric proposal filters
  -> one highest-confidence, near-horizontal jaw grasp
  -> fixed Dex3 retargeting
  -> one 6-DoF end-effector target
  -> cuRobo 17-DoF arms+waist trajectory
  -> learned whole-body tracker
```

This directly exposes the problem in our current runtime. We have been using
the expensive motion generator to discover useful values from an already
physics-qualified offline grasp atlas. HERO first reduces perception output
to one end-effector target and uses cuRobo only for reaching that target.

HERO does **not** give us an assembly planner, a GraspGenX-to-Dex3 transform,
or code that can currently be copied. It gives us a clear latency and
architecture reference.

## 1. What HERO is solving

HERO performs open-vocabulary, single-object grasping with an unmodified
Unitree G1, Dex3 hands, the head-mounted D435i, proprioception, and the base
IMU. The robot starts 10–20 cm from a surface, reaches one visually selected
object, closes one hand, and must lift the object for more than two seconds.

The paper evaluates:

- objects on surfaces from 43 cm to 92 cm high;
- one operating hand, primarily the right hand;
- whole-body standing control, including the waist and legs;
- novel objects, scenes, clutter, and language queries; and
- limited moving-object and door-opening extensions.

It is not a seated, bimanual, multi-payload assembly system.

## 2. Grasp generation and Dex3 retargeting

HERO does not use a native articulated Dex3 grasp generator.

### 2.1 Proposal generation

At runtime:

1. Grounding DINO detects the object named in the query.
2. SAM-3 produces its segmentation mask.
3. AnyGrasp produces parallel-jaw grasp proposals from RGB-D.
4. Proposals outside the target mask are removed.
5. Grasps approaching from the side opposite the available hand are removed.
6. Grasps that are too high or low under a gravity-aligned depth estimate are
   removed.
7. The remaining grasp most parallel to the ground with the highest AnyGrasp
   confidence is selected.

The paper explicitly says AnyDexGrasp could be used, but the authors found
limited benefit because they consider Dex3 insufficiently dexterous for the
tested task.

### 2.2 Retargeting

The selected parallel-jaw grasp is converted into a Dex3 end-effector target
using two simple operations:

- rotate the gripper pose by 45 degrees around its z axis so the thumb opposes
  the other two fingers; and
- clip end-effector yaw to within 70 degrees to avoid twisted IK postures.

This is a task heuristic around an AnyGrasp jaw frame. It is not equivalent to
our established GraspGenX `object_T_G` contract and must not be applied on top
of our current Dex3 descriptor transform.

### 2.3 Finger closure

cuRobo does not plan the Dex3 finger trajectory. HERO monitors end-effector
pose error and closes immediately when translation error reaches 1.5 cm. At
that point it repeatedly feeds the same local reference waypoint to the
tracking policy to stabilize the arm during closure.

The paper does not publish:

- the seven Dex3 closing joint targets;
- closure duration or interpolation;
- effort, stiffness, or damping values;
- tactile conditions;
- per-object hand preshapes; or
- a physics grasp validator.

## 3. Exactly what cuRobo does

Given the selected 6-DoF target, HERO:

1. uses IK to derive a base-height command and a 17-DoF upper-body joint goal;
2. asks cuRobo for collision-free reference joint and end-effector
   trajectories; and
3. sends those references to a learned whole-body tracking policy.

The 17 DoFs are the two seven-DoF arms plus the three waist DoFs. A workspace
study uses cuRobo IK over a 2 cm grid and reports that adding the waist
increases combined two-arm reachable volume from 0.248 m³ to 0.523 m³.

During execution, the dynamically balancing base drifts relative to the
original target. Every 300 policy steps—six seconds at 50 Hz—HERO replans the
remaining reference from the current state to an odometry-corrected goal. The
paper reports approximately 20 ms per cuRobo replan.

Deployment uses CUDA graph acceleration on:

- NVIDIA RTX 5070 Ti laptop GPU;
- Intel Core Ultra 9 275HX CPU; and
- 32 GB system RAM.

The paper cites the 2023 cuRobo work, not cuRoboV2, and gives no repository
commit. The appendix states a “planning dt” of `7.25e-6`, but does not define
which cuRobo parameter this means. That value is not sufficient or safe to
copy into our configuration.

## 4. cuRobo is a reference generator, not the hardware controller

HERO does not directly replay cuRobo joint positions with ordinary arm PD
control. Direct PD tracking is one of its weak baselines.

Instead, a simulation-trained policy tracks:

- the cuRobo upper-body joint reference;
- the cuRobo end-effector reference;
- current proprioception and short histories;
- current end-effector pose residual;
- base-height and locomotion commands.

It outputs commands for all 29 G1 DoFs at 50 Hz. Separate upper- and
lower-body three-layer MLPs coordinate arms, waist, and balance. Training uses
Isaac Gym at 500 Hz with 4,096 environments and 20,000 PPO iterations, then
MuJoCo sim-to-sim validation before hardware.

This is how HERO tolerates whole-body dynamics while using a fast kinematic
reference planner.

## 5. Why analytical FK was not sufficient on their hardware

The authors measure significant systematic mismatch between nominal G1
kinematics and physical end-effector pose:

- analytical end-effector FK translation error: 1.30 cm;
- kinematic calibration: 0.58 cm;
- learned residual FK: 0.50 cm.

They collect three hours of OptiTrack data and train:

- a residual end-effector FK model; and
- a residual leg-odometry model for base drift.

They also use iterative translation-only goal adjustment, limited to 5 mm per
step, while error is under 15 cm and until it is below 1.5 cm. Replanning
reduces real-world tracking translation error from 5.17 cm to 2.44 cm.

This is a critical hardware warning for our project: even a correct cuRobo
trajectory and URDF do not guarantee that the real Dex3 grasp frame reaches
the AprilTag-derived target accurately.

## 6. Camera calibration finding

HERO uses the built-in head D435i at 640×480 and 60 Hz. The paper identifies
a mechanically movable, non-motorized neck-pitch degree of freedom that makes
manufacturer camera extrinsics unreliable.

Their final calibration uses OptiTrack markers on both the robot base and an
ArUco board, 60–70 board poses, and Tsai–Lenz eye-to-hand calibration. They
report 2.5 mm reprojection error.

This validates our concern that “the D435i is built in” does not remove the
need to calibrate `base_T_camera` on our actual G1. Their exact extrinsic
cannot be copied because it depends on the physical neck pose and individual
robot.

## 7. Reported results and boundaries

HERO reports:

- 90% success across ten everyday objects and two table heights;
- 73.3% across ten broader novel scenes;
- 80% in five cluttered layouts;
- 2.44 cm mean real-world end-effector translation error; and
- 8.22-degree mean real-world orientation error.

Common failures are object slip, knocking objects over, the hand becoming
stuck under a high table, and unstable grasps on plush or heavy objects.

These results do not establish millimetre-precision connector mating.
Our magnets are therefore still doing essential work: their capture region
must absorb centimetre-scale tracking and perception error.

## 8. What transfers directly to our project

### 8.1 Use cuRobo after grasp selection

Our offline atlas already replaces HERO's online AnyGrasp stage. Runtime
should not ask full motion generation to independently evaluate thousands of
atlas entries.

The correct adaptation is:

```text
OFFLINE
GraspGenX proposals
  -> exact Dex3 Isaac retention
  -> immutable object-relative atlas
  -> object-relative connector compatibility

RUNTIME
AprilTag object poses
  -> transform atlas candidates to world
  -> fast cuRobo IK/collision screening
  -> select one complete T/U/cube mode
  -> full cuRobo motion generation only for that mode
  -> backtrack to a few alternatives if necessary
```

### 8.2 Persist and prewarm cuRobo

HERO's 20 ms result is for a deployment process already running with CUDA
graphs. It supports our conclusion that constructing, warming, and destroying
a planner for every semantic stage is incorrect for online use.

### 8.3 Closed-loop execution matters

For hardware, our saved cuRobo trajectory should become a reference, not an
assumption that open-loop joint tracking places the palm exactly. At minimum
we will need:

- accurate current joint state;
- calibrated camera/base/object transforms;
- measured end-effector error;
- replanning from the realized state; and
- a bounded final correction/capture strategy.

HERO's learned whole-body tracker is one sophisticated solution. A seated
robot may allow a simpler arm/waist controller, but the real error must still
be measured rather than ignored.

### 8.4 Freeze during closure

Holding the final arm reference while closing Dex3 is a sensible explicit
execution state. It avoids moving the palm while contact is developing.

## 9. What does not transfer directly

| HERO | Our assembly |
|---|---|
| one visually selected object | known T, U, and cube |
| online AnyGrasp parallel-jaw proposals | offline articulated GraspGenX atlas |
| fixed 45-degree jaw-to-Dex3 retarget | exact current-Dex3 `object_T_G` |
| one hand and one carried object | bimanual holder/worker payload transitions |
| one reach, close, and lift | two pickups, two connector mates, and placement |
| standing 29-DoF learned tracker | seated fixed-base prototype initially |
| approximately 2.44 cm tracking error | magnet-assisted connector capture |

We should borrow the separation and lifecycle, not replace our qualified
grasps with AnyGrasp or copy the 45-degree retarget.

## 10. Missing details

Because code is not released, the paper does not let us verify:

- the exact cuRobo version, commit, or API;
- the G1/Dex3 URDF and collision spheres;
- whether the table and depth geometry enter the cuRobo collision world;
- warmup timing and planner object lifetime;
- IK/trajopt seeds and tolerances;
- whether graph fallback is enabled;
- the exact grasp-frame convention;
- the right Dex3 open/closed joint vectors;
- the exact scene updates used for visual replanning; or
- the relationship between `7.25e-6` and a named cuRobo parameter.

Any implementation claim beyond the published interfaces above must wait for
the code release or author clarification.

## 11. Resulting decision

HERO overturns the idea that our present multi-minute runtime is an acceptable
property of cuRobo or Dex3.

The next runtime rewrite should:

1. keep one or a small fixed set of cuRobo planner instances alive;
2. initialize and CUDA-warm them before the demonstration;
3. move atlas-wide screening to cuRobo IK/collision queries rather than
   `MotionPlanner.plan_grasp`;
4. precompute robot-independent grasp/connector compatibility offline;
5. motion-plan only complete modes selected by the fast screen;
6. preserve constraint-directed backtracking for the small surviving set; and
7. instrument cold start, warm IK screening, first-plan latency, replan
   latency, and end-to-end plan latency separately.

The target should be seconds for initial task planning and tens of
milliseconds for a warmed single-goal replan. HERO provides evidence that the
second target is realistic; our more complex bimanual assembly still needs to
measure the first.
