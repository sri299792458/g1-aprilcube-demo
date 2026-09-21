# U support-conditioned right-Dex3 proposal audit

This is a pre-physics geometric audit. It does **not** claim that any
candidate closes on, lifts, or retains the U.

## Scope

- Raw immutable GraspGenX candidates: **4,096**
- Stable tabletop orientations: **2**
- Candidate/support pairs checked: **8,192**
- Geometric survivors: **98**
- Pre-physics proposal buckets: **24**

![Selected U tabletop supports](assets/u_legs_58p5mm_broad_supports.png)

Absolute tabletop height, XY translation, and in-plane yaw are absent
because they do not change object–support clearance. They belong to arm
reachability, not this object-relative support audit.

## Stable supports and gates

| Table-up object axis | Label | Equivalence class | Final table-clear | Pregrasp table-clear | Final object-clear | Full corridor-clear |
|---|---|---|---:|---:|---:|---:|
| +Y | broad_minus_y_face_down | broad_face | 139 | 137 | 91 | 90 |
| -Y | broad_plus_y_face_down | broad_face | 69 | 69 | 8 | 8 |

Only the support conditions selected by this experiment are evaluated.
Geometric symmetries remain recorded as equivalence classes rather
than silently deleting tag-distinguishable poses.

## Survivor distribution by semantic U component

| Support | Hip bridge | Left leg | Right leg | Unresolved | Total |
|---|---:|---:|---:|---:|---:|
| broad_minus_y_face_down | 32 | 37 | 20 | 1 | 90 |
| broad_plus_y_face_down | 2 | 1 | 3 | 2 | 8 |

The JSON artifact retains all 24 exact buckets, including surface,
cavity/exterior relation, support relation, approach direction, and
every concrete member ID. Nothing is selected or discarded by the
bucketing stage.

These are proposal buckets, not final grasp families. Final families
must be constructed only after physical closure and retention succeed.

## Required next gate

Run every geometric survivor in the corrected table-supported Isaac
test: collision-free pregrasp, complete approach, finger closure,
vertical lift, and hold under gravity. Only those successes may enter
the final object-centric family library.
