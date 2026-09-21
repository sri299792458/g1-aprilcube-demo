# Right-Dex3 upright U twice-passing bucket visual review

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **14**
- PASS: **13**
- FAIL: **1**
- Retained by at least two digit chains: **13**
- Hand/table collision: **0**
- U/table contact in final hold: **1**

Video: [`docs/assets/dex3_u_upright_supported_pickup_review14.mp4`](assets/dex3_u_upright_supported_pickup_review14.mp4)

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| upright_on_leg_ends | 14 | 13 | 1 |

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
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 1 |
| pass | 13 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

13 of 14 geometry-clear proposals passed the complete physical
contract for one twice-passing upright grasp from each robust proposal bucket. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.

The complete 14-trial video above preserves the failed `proposal_bfc42f5cbc01`
segment. A second video contains only the thirteen candidates that passed this
camera-enabled execution:
[`dex3_u_upright_supported_pickup_review13_passes.mp4`](assets/dex3_u_upright_supported_pickup_review13_passes.mp4).
