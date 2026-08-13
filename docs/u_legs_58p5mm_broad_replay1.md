# Right-Dex3 58.5 mm U broad-face discovered-pass replay

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **1**
- PASS: **1**
- FAIL: **0**
- Retained by at least two digit chains: **1**
- Hand/table collision: **0**
- U/table contact in final hold: **0**

Video: [`docs/assets/dex3_u_58p5mm_broad_pass_replay1.mp4`](assets/dex3_u_58p5mm_broad_pass_replay1.mp4)

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| broad_plus_y_face_down | 1 | 1 | 0 |

## Physical pass contract

A trial passes only when all three statements are true:

1. At least two Dex3 digit chains contact the U after the 20 cm lift
   and one-second final hold.
2. No Dex3 hand link contacted the table during settle, approach,
   closure, lift, or hold.
3. The U did not contact the table during the final elevated hold.

## Verdict combinations

| Verdict/reasons | Trials |
|---|---:|
| pass | 1 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

1 of 1 geometry-clear proposals passed the complete physical
contract for the sole PASS discovered in the 98-trial 58.5 mm run. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.
