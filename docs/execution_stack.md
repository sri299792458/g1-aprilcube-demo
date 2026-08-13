# Execution stack for the fixed assembly sequence

Decision date: 2026-07-20

## Scope

The demo sequence is specified. Our problem is to execute it reliably while
maintaining correct robot, object, attachment, and collision state.

```text
GraspGenX      proposes object-relative grasps
Isaac/PhysX    qualifies grasp/contact behavior offline with the VIRAL profile
cuRoboV2       plans collision-aware G1 motion
sequence layer applies fixed stages and explicit scene-state transitions
ROS 2 bridge   executes approved trajectories on hardware later
```

## Required state transitions

1. The T, U, and cube begin as separate world objects.
2. One hand picks the T and the other hand may pick the U, producing two
   simultaneous attachments.
3. The U snaps to the T. The U is removed from the worker hand and the holder's
   carried collision model becomes T+U.
4. The worker hand picks the cube while the holder carries T+U.
5. The cube snaps to the composite; the holder now carries T+U+cube.
6. The complete object is placed, detached from the holder, and restored to
   the world collision model.

The readiness check now exposes both complete assignments: left holder/right
worker and right holder/left worker. Each assignment uses separately
VIRAL-profile ordinary-retention pools for the exact physical hand. The
runtime plans complete sequences for all arm-reachable assignments and chooses
the one with the lowest 14-arm-joint arc length. Raw object distance does not
decide the arm because it cannot test approach, collision, connector, or
placement feasibility.

## Verified cuRoboV2 contract

The audit was pinned to official cuRobo tag `v0.8.0`, commit
`4ea77366ca48ee453e7df139e39fa6532af49f3b`. Its upstream tests passed locally:

```text
test_attachment_manager.py   18 passed
test_motion_planner.py       72 passed
```

An additional contract probe loaded the shipped
`g1_29dof_with_hand_rev_1_0.urdf`, used both Dex3 palm frames as tool frames,
and added independent collision-sphere slots for objects carried by the two
hands. It verified that the underlying v0.8 mechanisms can:

- solve a two-palm goal;
- preserve two independent carried-object representations;
- plan while both attachment slots are populated;
- remove one hand's attachment without removing the other's;
- replace the holder's T geometry with T+U and then T+U+cube; and
- disable one named target world object while leaving the table enabled.

The probe used conservative synthetic spheres and small palm motions. It proves
the API contract, not seated reachability, grasp success, final part geometry,
connector feasibility, or hardware execution.

## Adapter constraints

- `MotionPlanner.attachment_manager` is a broken convenience property in the
  tagged release; the underlying manager is available through the trajectory
  solver core.
- The manager's convenience bookkeeping assumes one attachment. Every update
  and detach must explicitly identify the hand attachment link and world
  object.
- Its world-pose-offset helper uses `tool_frames[0]`, so it is unsafe for the
  second hand. Each object's collision spheres must be transformed into the
  correct hand-specific attachment-link frame before updating the manager.
- Collision enable/disable operates on an entire world object or robot link,
  not a pair-specific Allowed Collision Matrix.
- The generic grasp helper can disable a contact link globally. We must not do
  that near the table. During final intentional contact, disable only the
  target object's world copy while keeping the hand, table, other parts, and
  self-collision active.

If the real connector geometry later requires pair-specific collision
permissions that object-specific masking cannot express, reconsider the
collision backend at that concrete checkpoint.

## Fixed-sequence executor

Each operation has explicit preconditions, planner inputs, and scene effects:

```text
plan_to_pregrasp(candidate)
plan_contact_approach(candidate, target_object)
close_hand(profile)
attach_to_hand(object, hand)
plan_transfer_or_retract(goal)
snap_and_transfer(worker_object, holder_composite)
place_and_detach(composite)
```

For example, `snap_and_transfer(U, T)` succeeds only after the connector poses
meet their declared tolerances. It then removes U from the worker attachment
slot and replaces the holder's carried model with T+U. No operation invents an
offset or changes collision state outside its declared target.

The stage primitives and declarative attachment state are reusable. The
research-prototype `execute()` choreography is intentionally specific to the
T/U/cube reveal and can be rewritten when the demo changes; it is not a
backward-compatible generic task language.

## Implemented planning checkpoint

The complete fixed sequence now plans successfully with clean cuRobo v0.8.0:

1. Pick the T with a qualified left-Dex3 central-stem grasp.
2. Pick the U with the right hand, mate from below, transfer U into the left
   composite attachment, release, and retreat the right hand.
3. Pick the cube with the right hand, mate from above, transfer it into the
   composite attachment, release, and retreat.
4. Place the completed 360 mm figure, open the holder, restore all three parts
   as world objects, and retreat for the final reveal.

The planner keeps the table, non-target parts, robot self-collision, and the
nonmoving arm collision-live. It hides only the intentional-contact target (or
the holder attachment at exact connector contact) for the stage where that
contact is required. Magnet attraction is not simulated: a snap is an explicit
scene-state transfer at the declared connector transform.

This is a collision-aware kinematic planning result and exact-trajectory
visualization. It is not yet a robot execution result, a magnetic connector
physics test, or proof that the seated hardware bridge and perception chain
are complete.
