# Verified GraspGenX integration contract

This document is deliberately narrow. It records only what the pinned upstream
code and a reproduced run establish. It does not inherit assumptions from an
earlier implementation.

## 1. What goes into GraspGenX

For the released checkpoint used here:

```text
segmented object point cloud in frame F
          +
12-number gripper sweep descriptor
```

The 12 gripper numbers are:

```text
open box extents xyz
open box offset xyz
half-open box extents xyz
half-open box offset xyz
```

The stock `unitree_g1` values are:

```text
open: extents [0.10, 0.06, 0.04], offset [0.000, 0.000, 0.070]
half: extents [0.04, 0.06, 0.04], offset [0.007, 0.000, 0.060]
```

The released generator and discriminator select the `sweep_volume_v2`
conditioning path. The seven Dex3 joint values and URDF are therefore not read
by inference. They remain necessary for rendering, closure qualification, and
eventual execution.

## 2. What comes out

```text
candidate transforms: K x 4 x 4
confidence scores:     K
```

For point-cloud frame `F`, candidate `i` is:

```text
F_T_G[i]
```

`G` is the GraspGenX canonical gripper root. The transform maps a point written
in canonical gripper coordinates into `F`.

The output does **not** contain:

- a Dex3 palm or wrist pose;
- a seven-joint finger configuration;
- a pregrasp or retract pose;
- an arm IK solution or trajectory;
- table/scene collision clearance;
- an assertion that the robot can reach the candidate.

The confidence is model ranking evidence. It is not a complete execution
validity certificate.

## 3. Why the returned frame is not the palm

The stock Unitree descriptor URDF has a root link named `world`. In this file,
`world` means the canonical GraspGenX frame `G`; it is not our MoveIt world or
robot base.

The descriptor fixes `right_palm_link` below that root:

```text
G_T_right_palm_link ≈

[[-1, 0, 0, 0.07],
 [ 0, 0, 1, 0.00],
 [ 0, 1, 0, 0.02],
 [ 0, 0, 0, 1.00]]
```

Upstream renders the hand by multiplying every URDF geometry transform as:

```text
F_T_geometry = F_T_G @ G_T_geometry
```

The fixed palm transform is already part of `G_T_geometry`. This both proves
the output-frame meaning and gives the correct palm conversion.

## 4. The only valid palm conversion

```text
F_T_right_palm_link
    =
F_T_G
    @
G_T_right_palm_link
```

No inverse and no hand-tuned offset is introduced.

If MoveIt later controls a frame other than the exact same `right_palm_link`, a
second fixed transform must be obtained from the authoritative robot model and
verified by visual overlap:

```text
F_T_moveit_tool
    =
F_T_G
    @ G_T_right_palm_link
    @ right_palm_link_T_moveit_tool
```

That last transform does not yet exist in this repository.

## 5. Open, close, and approach are separate things

The stock descriptor also supplies dictionaries of seven right-hand joint
values named `open` and `close`. They are fixed endpoints, not a grasp-specific
finger solution returned by the network. The halfway image is only a 50%
linear interpolation for inspection.

Likewise, a pregrasp is not part of GraspGenX output. Once a candidate has been
geometrically qualified, the manipulation layer may define a named approach
transform along canonical `-Z`, because upstream defines canonical `+Z` along
the fingers toward the object. The approach distance must come from object and
scene clearance—not an unexplained global constant.

## 6. Required candidate qualification

Candidate generation is automatic. Candidate acceptance is a downstream,
deterministic filter:

1. Render the actual open and closing hand at `F_T_G`.
2. Reject palm or non-contact-link penetration.
3. Check that closing provides useful opposing contact/enclosure on the object.
4. Check the approach corridor against the object, table, and other parts.
5. Convert to the exact MoveIt tool frame using verified fixed transforms.
6. Reject candidates that fail arm IK, joint limits, or scene collision.
7. Rank the remaining candidates using model confidence plus deterministic
   execution margins.

This does not replace GraspGenX with manual grasps. GraspGenX proposes the
poses; the robot stack decides which proposals are executable.

## 7. Current open questions

- The upstream asset is right-hand-only. We still need an authoritative left
  Dex3 model/descriptor and a visual overlap test.
- We have not yet verified that the future MoveIt `right_palm_link` is identical
  to the upstream descriptor link.
- We have not generated candidates on the AprilCube T, U, or cube.
- We have not selected grasp-dependent approach distances.
- We have not introduced OMPL, MTC, robot IK, or PlanningScene objects.

Those are subsequent checkpoints, not assumptions to hide in the first one.
