# Execution stack for the fixed assembly sequence

Decision date: 2026-07-20

## Scope

The demo sequence is specified. Our problem is to execute it reliably while
maintaining correct robot, object, attachment, and collision state.

```text
GraspGenX      proposes object-relative grasps
Newton         qualifies grasp/contact behavior offline
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

Arm assignment is selected from reachability. The executor uses symmetric
`holder_hand` and `worker_hand` roles rather than hard-coded left/right stages.

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

The executor remains object-agnostic. A task file supplies part IDs, connector
frames, qualified grasp candidates, assembly order, holder/worker roles, and
the final placement.

## Next visual checkpoint

Before implementing the complete sequence:

1. Load the actual table and one 45 mm AprilCube part with the shipped G1.
2. Show the seated/start state and both palm tool frames.
3. Convert qualified GraspGenX candidates using the verified frame contract.
4. Plan to pregrasp and contact approach while the table remains collidable.
5. Attach the part to one named hand slot and plan a lift.
6. Export the planned result for visual review before adding the second hand.

This checkpoint tests geometry and reachability; the API audit has already
established the required execution and attachment mechanisms.
