# Right-Dex3 upright U supported-pickup qualification

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **1837**
- PASS: **405**
- FAIL: **1432**
- Retained by at least two digit chains: **428**
- Hand/table collision: **33**
- U/table contact in final hold: **1408**

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| upright_on_leg_ends | 1837 | 405 | 1432 |

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
| hand_table_contact | 23 |
| insufficient_final_digit_contacts | 1 |
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 10 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 1398 |
| pass | 405 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

405 of 1837 geometry-clear proposals passed the complete physical
contract for the U standing upright on both leg ends. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.

## Reproducibility replay

All 405 discovery passes were replayed under a different 64-environment batch
layout. **365 passed again** and 40 returned to the table. The twice-passing
set contains 149 hip-bridge, 147 left-leg, and 69 right-leg candidates across
five approach sectors and 14 proposal buckets.

Only these 365 twice-passing records enter the upright right-Dex3 physics
library. The replay ledger and summary are
`artifacts/grasp_support/u_legs_right_upright_replay1_isaac_v1/report.json` and
`docs/u_legs_upright_supported_pickup_replay1.md`.

One twice-passing member per bucket was then camera-tested a third time. The
complete review retained its single failure, while the readable sequence of
the 13 measured passes is
[`dex3_u_upright_supported_pickup_review13_passes.mp4`](assets/dex3_u_upright_supported_pickup_review13_passes.mp4).
