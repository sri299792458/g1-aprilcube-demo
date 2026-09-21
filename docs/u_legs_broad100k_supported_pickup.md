# Right-Dex3 broad-face U 100K supported-pickup qualification

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **983**
- PASS: **0**
- FAIL: **983**
- Retained by at least two digit chains: **1**
- Hand/table collision: **340**
- U/table contact in final hold: **982**

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| broad_minus_y_face_down | 794 | 0 | 794 |
| broad_plus_y_face_down | 189 | 0 | 189 |

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
| hand_table_contact | 1 |
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 339 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 643 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

None of the 983 geometry-clear proposals from the 100,000-candidate pool
passed the complete physical contract. One trial retained the U after lift,
but only while violating the no-hand/table-contact gate; the other 982 still
contacted the table in the final hold. There is therefore no PASS set to
replay.

This closes the bounded larger-sample experiment. More unconditioned
GraspGenX sampling is not justified for a broad-face U pickup under this
descriptor-local straight-approach and controller contract. The runtime
library remains empty for both broad-face supports; use the already-qualified
upright-on-leg-ends support instead.

A ten-trial visual diagnostic, including the one retention/table-collision
near-miss, is summarized in
[`u_legs_broad100k_supported_pickup_review.md`](u_legs_broad100k_supported_pickup_review.md).
