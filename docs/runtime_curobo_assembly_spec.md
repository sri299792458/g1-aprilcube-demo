# Runtime AprilCube assembly with upstream cuRobo

Status: implemented and acceptance-tested on 2026-07-22

Supersedes as architecture: the fixed-scene policy in
`g1_aprilcube_demo/planning/assembly_runner.py`

Retains as regression evidence: the previously rendered fixed T/U/cube run

## 1. Decision

The working planner will be rebuilt around the released cuRoboV2 manipulation
APIs already present in this repository. Project code will provide task data
and state transitions; it will not reproduce cuRobo's grasp planner, batched
IK, Cartesian approach planner, graph planner, or attachment geometry manager.

The fixed task remains:

1. one hand picks and retains the T root;
2. the other hand picks and mates the U;
3. the same worker hand picks and mates the cube;
4. the holder places the completed T+U+cube assembly.

The three loose parts are observations, not prepared fixtures. Their poses may
change between runs. A run is accepted when the observed arrangement has at
least one collision-free solution; an unreachable or overlapping arrangement
must fail with an explicit reason rather than silently replacing the
observation with nominal poses.

## 2. Existing upstream capabilities we will use

The pinned cuRobo checkout already supplies:

- `MotionPlanner.plan_grasp`: candidate goal-set selection followed by
  pregrasp, constrained final approach, and optional retract planning;
- `MotionPlanner.plan_pose` and `plan_cspace`: collision-aware Cartesian and
  joint-space transfers with graph fallback;
- `GoalToolPose`: one or more coupled tool frames and multiple candidate goals;
- `ToolPoseCriteria.linear_motion`: constrained connector approaches and
  retreats;
- dynamic scene replacement and obstacle enable/disable; and
- `AttachmentManager`: sphere fitting plus attachment updates at arbitrary
  robot attachment links.

The exact dual-Dex3 model has two reserved payload links,
`left_attached_object` and `right_attached_object`. A direct audit on
2026-07-22 also confirmed that a native coupled two-tool goal can hold one
grasp frame fixed while moving the other by 40 mm. The replacement must not
retain the old claim that multi-tool planning is intrinsically unavailable.

### 2.1 Implementation findings that changed this contract

The following facts were established against the pinned checkout while
implementing this replacement, rather than assumed from API names:

- A single `MotionPlanner.plan_pose` request with both G1 grasp frames is
  supported. A diagnostic held the left grasp frame fixed and displaced the
  right frame by 40 mm; cuRobo returned one collision-checked 14-joint plan.
- A native single-hand `plan_grasp` goal set succeeds with the actual T atlas
  and exact G1 model. Operational batches of 8, 16, and 32 candidates all
  selected a known reachable candidate.
- A 48-entry goal set returned no result even when that same reachable
  candidate was its first entry. Therefore 48 is a solver/configuration
  boundary in this checkout, not evidence that all 48 grasps are unreachable.
  The replacement uses a documented operational maximum of 32.
- Most importantly, cuRobo's IK result stores a goal-set index per tool link.
  `MotionPlanner.plan_grasp` subsequently reads the first tool's index and
  applies it to every tool. The batch implementation has the same extraction
  rule. A multi-tool goal set therefore does **not** preserve row-wise
  left/right pairing.
- `MotionPlanner.attachment_manager` currently raises `AttributeError`: the
  facade forwards to `TrajOptSolver.attachment_manager`, while the manager is
  actually owned by `TrajOptSolver.core`. The IK solver also owns a distinct
  kinematics instance and manager. Until the upstream facade is corrected and
  synchronizes attachments, the backend calls the two already-constructed
  upstream `SolverCore.attachment_manager` instances with the same immutable
  sphere tensor. It does not generate or alter the fitted sphere geometry.
- The generated cuRobo robot initially locked all 14 Dex3 finger joints at
  zero. That is correct for a pregrasp but wrong while collision-checking a
  carried payload or a connector approach. Every stage planner now locks each
  hand at the exact open/close interpolation recorded by the rev-1.0 atlas
  descriptor: open for pickup approach, closed after grasp, and open again
  after release. Finger actuation remains an explicit task command; these
  locked values make cuRobo's arm and payload collision model agree with that
  task state.

The last finding changes mate qualification: every holder/worker pair is sent
as its own two-tool problem with `num_goalset=1`. Independent pair problems
may be CUDA-batched, but multiple purported pairs must never be encoded as the
goal-set dimension of one multi-tool problem. This restriction can be removed
only after the upstream API explicitly returns and preserves a coupled index.

## 3. Project-owned responsibilities

Only the following behavior belongs in project code:

1. parse and validate a runtime scene observation;
2. load the physics-qualified GraspGenX atlas for each available hand;
3. compose object-relative grasps into the observed planning frame;
4. supply task connector transforms and generate bounded work/placement pose
   candidates;
5. ask cuRobo to evaluate and plan those candidates;
6. command the named Dex3 open/close profiles in the saved trajectory state;
7. apply explicit loose -> hand payload -> composite -> world state
   transitions and assert them against the compiled task snapshots; and
8. save a report that records every selected observation, grasp, work pose,
   attachment transition, and cuRobo result.

Project code must not:

- generate a grasp pose by hand;
- infer grasp validity from a render;
- choose a candidate from Cartesian distance alone;
- interpolate Cartesian waypoints itself;
- implement its own IK solver;
- project one arm out of a whole-body trajectory after planning;
- replace failed runtime observations with fixed object poses;
- require a tray, cradle, peg, or locating fixture; or
- modify the cuRobo source checkout to make the demo pass.

## 4. Frames and immutable transform contract

All transforms use `A_T_B`, which maps coordinates expressed in frame `B`
into frame `A`.

```text
planning_T_object       runtime observation
object_T_G              unchanged GraspGenX atlas entry
planning_T_G            planning_T_object * object_T_G

T_T_child               fixed connector transform from the task file
planning_T_child        planning_T_T * T_T_child
planning_T_child_G      planning_T_child * child_T_G
```

`G` is the real virtual grasp frame built into the current Dex3 robot model.
No extra palm or wrist offset is introduced by this planner.

The simulator supplies exact observations. Hardware will later provide the
same contract from table registration, camera extrinsics, AprilTag detection,
and AprilCube CAD. Perception must not leak into the motion planner API.

## 5. Runtime observation

A scene observation is a versioned YAML document:

```yaml
schema_version: 1
observation_id: shuffled_scene_001
stamp: simulation
planning_frame: world
table:
  center: [0.55, 0.0, 0.68]
  dimensions: [0.80, 0.80, 0.04]
objects:
  t_body:
    translation: [0.38, 0.24, 0.79]
    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
  u_legs:
    translation: [0.43, -0.18, 0.7675]
    quaternion_xyzw: [0.0, 0.0, 0.3826834, 0.9238795]
  cube_head:
    translation: [0.31, -0.30, 0.7225]
    quaternion_xyzw: [0.0, 0.0, 0.7071068, 0.7071068]
```

Validation checks finite normalized transforms, known object IDs, vertical
support, full support inside the tabletop XY bounds, and object-object
separation. Initial robot/world collision and reachability remain cuRobo
questions and are checked by the native pickup planning requests. Validation
does not demand exact XY positions or yaw values.

The observation owns the loose-part poses. The planner configuration owns the
robot model, safe initial arm state, motion parameters, work-pose search
region, and atlas paths. Those concerns must remain in separate files.

## 6. Candidate policy

The input grasp candidates are exactly the ordinary Isaac/PhysX
VIRAL-profile passes in the existing arm-grasp-pool files. Atlas transforms
are immutable.

Candidate ordering preserves the atlas's family-balanced ordering. Batching is
allowed for memory bounds, but batching must not change a candidate transform
or create a new pose. Neural confidence may be a tie-breaker after planning
feasibility; it is not a reachability or contact test.

The old approach-ray-to-cuboid classifier is not an admission gate in the
replacement planner. Connector exposure is tested by exact hand, carried
object, and composite collision geometry at the future mate stage. This lets
cuRobo reject obstructing grasps using the geometry that matters instead of a
project-authored contact heuristic.

Only hand/object pairs backed by a physics-qualified atlas are eligible. The
current complete task evidence supports left holder + right worker. The data
model permits other assignments, but it must not pretend an unqualified
left-U or left-cube atlas exists.

## 7. Planning model

### 7.1 Work-pose candidates

The robot must assemble in a comfortable shared workspace, but no single
world pose is privileged. The planner configuration defines bounded samples:

```text
x range, y range, z range, yaw choices
minimum clearance above the table
maximum candidate count
```

The samples describe an assembly workspace, not the loose-object layout.
They are converted into T-root poses. U and cube poses follow exactly from
`T_T_child`.

The placement region is described similarly by table-relative XY/yaw samples.
The vertical coordinate is derived from the complete assembly mesh support
height, not a hand-tuned center value.

### 7.2 Mode qualification

Before committing to a grasp, the planner establishes a complete connector
mode:

```text
holder hand and T grasp
U worker grasp and U-stage T work pose
cube worker grasp and head-stage T work pose
```

Mode generation is deterministic Cartesian composition. cuRobo is the sole
geometric/reachability evaluator.

The qualification stages are:

1. goal-set-test atlas candidates at each observed pickup pose;
2. test singleton coupled holder/worker goals for U mating across work-pose
   samples;
3. test singleton coupled holder/worker goals for cube mating across work-pose
   samples;
4. retain only a T grasp that participates in both a U and a cube mode; and
5. preserve deterministic atlas/workspace discovery order.

Placement is intentionally not prequalified from a detached endpoint. It
depends on the actual post-head-mate joint branch and live T+U+cube payload.
Each full planning attempt therefore tests bounded placement samples after
executing both connector plans in memory. A placement failure backtracks to
the next connector mode while the physical robot is still untouched.

Mode qualification uses cuRobo's own collision-aware multi-link `IKSolver` on
the two-tool singleton endpoints. It intentionally does not ask TrajOpt for a
path from the seated ready state while pretending both objects are already
attached there: an object-relative grasp can place those hypothetical
payloads through the torso at that unrelated start configuration and cause a
false `Start state in collision`. After endpoint qualification, the complete
runtime performs the actual pickup, attachment, and sequential trajectory
plans from their real states; only those trajectory results are executable.

The paired precontact joint configuration returned by that upstream IK solve
is retained as part of the mode. Qualification then creates the real
right-arm-only planner with the left arm locked at those seven joint values
and requires the constrained connector approach to succeed with both payloads
live. Execution first asks each arm to reach those exact qualified joint
values. Implementation showed that a paired joint vector can be collision-free
yet have no path from the realized post-pick state. In that case only, the
runtime asks upstream `plan_pose` for another collision-free joint branch at
the identical cached grasp-frame target. The subsequent linear connector
approach is always replanned and collision-checked from the branch actually
reached. Thus the cached joint vector is a preferred witness, not a blindly
replayed trajectory; the object pose, grasp pose, and connector corridor never
change in the fallback.

Complete qualified modes are cached by a SHA-256 digest over a versioned mode
qualification contract, the backend/goal/workspace source modules,
configuration, observation, task, robot configuration and URDF, both Dex3
profiles, all three atlas pools, and every part mesh/geometry file. The cache
stores only candidate identities, work poses, and cuRobo-returned precontact
joint solutions; it is invalidated automatically when any geometric or
planning input changes. This avoids repeating minutes of deterministic CUDA
qualification during rendering/debugging without allowing a stale mode to
survive a scene, hand-state, robot, part, or atlas edit. Execution still
replans and collision-checks every trajectory.

The execution-orchestration source is deliberately not hashed wholesale:
changing report text or a post-release retreat would otherwise force minutes
of unchanged connector endpoint qualification. Any edit to the qualification
logic in that file must bump `MODE_CACHE_CONTRACT`; its directly imported
backend, goal construction, and workspace modules are hashed automatically.

The first implementation capped the depth-first Cartesian product of
`T grasp × U grasp × head grasp`. Its first four modes all reused one T and
one head grasp and changed only the U grasp, so later execution failures did
not exercise meaningfully different holder geometry. The bounded list now
takes one complete connector mode per distinct T grasp first; secondary
U/head alternatives for the same T do not consume the bounded list. This is
deterministic diversity, not a new geometric heuristic—every retained member
has already passed the same cuRobo pickup and coupled connector qualification.

Each coupled mate hypothesis is one two-tool `GoalToolPose` with
`num_goalset=1`. This ensures the solver evaluates the actual two-arm geometry
rather than independently selecting the holder from one hypothesis and the
worker from another. Multiple singleton hypotheses may be represented along
the batch dimension; they may not be represented along the goal-set
dimension.

The complete task is planned before any hardware command is sent. If a later
stage fails, the planner backtracks to another qualified mode while the
physical robot is still at the start state.

### 7.3 Execution planning

For a selected mode:

#### Pick

1. build `planning_T_G` for all retained candidates;
2. call `MotionPlanner.plan_grasp(..., plan_grasp_to_lift=False)`;
3. use the returned `goalset_index` as the selected atlas identity;
4. append the returned pregrasp and constrained contact trajectories;
5. append the versioned Dex3 closing profile;
6. remove only the selected loose object from the collision world;
7. add it to that hand's attachment link using `AttachmentManager`; and
8. plan a collision-aware retract with the payload live.

The target remains collision-live during the transfer to pregrasp. Only the
terminal contact links named for the moving Dex3 may be disabled during exact
grasp selection/contact. The palm, proximal fingers, table, other loose
objects, other arm, and robot self-collision remain active.

The payload is initially in intentional support contact with the table. The
current collision API has no object-pair allowed-collision entry, so the table
alone is disabled during the first constrained world-vertical separation and
immediately re-enabled afterward. The payload, robot self-collision, other
objects, and linear-motion constraint remain live. Without this bounded
exception cuRobo correctly reports the lift's start state as colliding; leaving
the table disabled for a free-space transfer would be unsafe and is forbidden.

Placement has the symmetric condition. The free-space transfer terminates at
a configured precontact height with the table live. Only the final constrained
world-vertical descent disables the table, because its terminal assembly pose
is intentionally supported by it (and the sphere approximation plus collision
activation margin is more conservative than the 3 mm visual clearance). The
table is restored before release and empty-hand retreat.

Release creates a second intentional-contact boundary: the composite changes
from a holder attachment into three world obstacles while the newly opened
hand is still at the release pose. Because collision permissions cannot be
scoped to a hand/object pair, only those exact T/U/cube world obstacle names
are disabled during the constrained upward empty-hand separation. The table,
robot self-collision, and all unrelated obstacles stay active; the part
obstacles are restored automatically after that one planning call. Disabling
whole fingertip links was rejected because it would also hide fingertip/table
and fingertip/robot collisions.

The first implementation added a fixed 60 mm and then 150 mm upward offset to
the final support pose. Both were wrong for the selected holder branch: the
assembly was already at a collision-free `z=0.95 m` root pose and the offsets
asked a near-limit arm to move farther upward before descending. The runtime
now retains its actual current root height for the horizontal placement
transfer (or the minimum support clearance, whichever is higher), then makes
only the required downward support descent. The placement XY/yaw region
includes the shared-workspace center (`x=0.42 m`, `y=0`) but is still sampled,
and final support height remains derived from mesh bounds.

#### Mate

1. move the holder and worker sequentially to the selected coupled-mode
   configuration while both payloads remain live;
2. call the cuRobo grasp/linear-approach machinery for precontact -> connector
   contact;
3. append the worker open command;
4. clear the worker attachment;
5. replace the holder attachment with the exact T-relative composite; and
6. plan a constrained worker retreat with the composite live.

cuRobo currently has link-level collision enable/disable rather than a
pairwise allowed-collision matrix. During the final connector-contact segment
only, the holder payload link may be disabled because child/composite contact
is intentional. The holder is stationary and its configuration must already
have passed full collision checking at precontact. This exception is named,
bounded to one segment, and recorded in the report.

#### Place

1. plan the payload-aware holder transfer to the selected placement pose;
2. plan the final table-normal descent;
3. append the holder open command;
4. clear the holder attachment and publish all members at their exact world
   assembly poses; and
5. plan the empty hand's retreat.

## 8. State machine

The project task compiler remains the authority for legal state transitions.
The runtime planner consumes its commands and may not invent or omit an
attachment transition.

```text
loose T,U,cube
  -> left:T | loose U,cube
  -> left:T | right:U | loose cube
  -> left:T+U | loose cube
  -> left:T+U | right:cube
  -> left:T+U+cube
  -> placed:T+U+cube | empty hands
```

Every rendered segment stores the live object and hand state associated with
its trajectory frames. A failed planning attempt must leave no partial state
in a returned run.

After each of the six high-level steps, the runtime reconstructs its loose,
holder, worker, and placed sets from the live collision payloads and compares
them with the task compiler's `after` snapshot. A mismatch aborts planning;
the report records every successful assertion. The compiler is therefore an
executable state contract, not merely documentation.

## 9. Code structure

The replacement implementation uses these modules:

```text
g1_aprilcube_demo/runtime/observation.py
    observation schema, validation, and transforms

g1_aprilcube_demo/planning/grasp_goalset.py
    atlas loading and GoalToolPose construction only

g1_aprilcube_demo/planning/workspace.py
    deterministic work/placement pose samples

g1_aprilcube_demo/planning/curobo_backend.py
    narrow wrapper around public MotionPlanner, ToolPoseCriteria, scene
    updates, and AttachmentManager

g1_aprilcube_demo/planning/runtime_assembly.py
    mode qualification, task-state transitions, reporting

tools/run_runtime_assembly.py
    CLI: planner config + observation + output directory
```

The old `assembly_runner.py` is not imported by this path. It remains in the
repository until the replacement passes visual review; deletion requires
explicit approval.

## 10. Configuration boundary

The new planner configuration contains:

- exact robot/URDF paths and base transform;
- safe start arm configuration;
- task file and per-hand atlas paths;
- named terminal contact links per hand;
- measured goal-set capacity and solver seed counts;
- named approach, retract, precontact, and retreat distances;
- bounded work-pose and placement sampling regions; and
- rendering/report settings.

`candidate_goalset_size` is 32. This is not a quality heuristic: it is a
measured safe capacity of the pinned upstream solver. All atlas candidates are
still eligible and are offered in consecutive deterministic batches until a
plan succeeds or the pool is exhausted.

One stage planner is warmed once and reused across consecutive 32-candidate
batches for a given pickup state. Rebuilding and re-warming an identical CUDA
planner for every atlas slice changed no planning inputs and dominated runtime;
reuse preserves the exact upstream candidate decisions while avoiding that
pure lifecycle cost.

The upstream planner seed is reset before each independent atlas slice and
each singleton coupled hypothesis. Without that reset, reusing a planner makes
the candidate result depend on how many earlier slices consumed its random
seed stream; rebuilding happened to reset it implicitly. Explicit reset keeps
planner reuse fast while making every candidate batch reproducible and
independent of preceding failures.

Placement samples likewise share one warmed planner because robot state,
payload, and world are identical until a sample succeeds. Each sample still
receives an explicit upstream seed reset. Reconstructing and warming the same
planner for every XY/yaw sample added only lifecycle latency and no geometric
information.

It does not contain loose-object poses, a cube support peg, a selected grasp
ID, or one exact assembly/placement pose.

## 11. Acceptance tests

The replacement is complete only when all of the following pass:

1. **Schema:** malformed transforms, missing objects, unsupported placements,
   and object overlaps fail before CUDA planning; robot/world collisions fail
   in the upstream planning request.
2. **Transform:** every world grasp equals the observed object transform times
   the unchanged atlas transform.
3. **Upstream use:** picks invoke cuRobo `plan_grasp`; no project Cartesian
   interpolation or standalone project IK implementation exists in the new
   path.
4. **Coupling:** mate qualification uses a two-tool `GoalToolPose` and returns
   the selected work-pose and child-grasp identities.
5. **Payload:** loose, single payload, dual payload, composite, and placed
   collision states match the task compiler after every transition.
6. **No fixture:** the cube rests directly on the table and the scene contains
   no head support, tray, cradle, or peg.
7. **Runtime variation:** the same executable plans two observations with
   different XY positions and yaw angles without changing planner code or
   configuration.
8. **Full plan:** one observation completes all six high-level task stages
   with every cuRobo result successful.
9. **Visual evidence:** an MP4 reconstructed from the saved trajectories shows
   the observed loose poses, actual selected grasps, both mates, placement,
   release, and reveal.
10. **Regression:** existing atlas/task tests and the new runtime tests pass.

## 12. Evidence boundary

Passing this specification proves runtime-pose-conditioned collision-aware
kinematic planning with qualified grasps and explicit payload geometry. It
does not prove magnet capture, grasp survival under real arm acceleration,
tag accuracy, camera calibration, seated-base repeatability, ROS 2 execution,
or hardware controller behavior.

Those later boundaries must consume the saved selected transforms and
trajectories; they must not cause the planning architecture to be rewritten.

## 13. Acceptance evidence (2026-07-22)

The final source completed both versioned observations with the same planner
configuration and executable:

- nominal selected T `...0079...089`, U `...0119...043`, and cube
  `...0119...201`; it recorded 33 successful events and 138 state-bearing
  segments with no endpoint fallback;
- varied XY/yaw selected T `...0139...125`, U `...0119...043`, and cube
  `...0089...161`; it recorded 34 successful events and 138 segments. Its
  cached head joint target had no path from the realized cube-pick state, so
  upstream `plan_pose` found another joint branch at the unchanged precontact
  pose and the constrained connector approach passed from that branch; and
- both reports contain successful compiler-state assertions for `pick_t`,
  `pick_u`, `mate_u_to_t`, `pick_head`, `mate_head_to_t`, and
  `place_complete`, followed by `complete`.

The complete root suite reports `25 passed`. The committed nominal visual
evidence is `docs/assets/t_u_cube_runtime_curobo_v2.mp4`: 408 frames,
960×720, 24 fps, and 17.0 seconds. Its contact sheet and terminal frame were
inspected after encoding; the completed object is supported on the U legs and
both open hands have retreated.
