# Right-Dex3 upright U successful-bucket visual review

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **17**
- PASS: **11**
- FAIL: **6**
- Retained by at least two digit chains: **11**
- Hand/table collision: **0**
- U/table contact in final hold: **6**

Video: [`docs/assets/dex3_u_upright_supported_pickup_review17.mp4`](assets/dex3_u_upright_supported_pickup_review17.mp4)

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| upright_on_leg_ends | 17 | 11 | 6 |

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
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 6 |
| pass | 11 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

11 of 17 geometry-clear proposals passed the complete physical
contract for one measured PASS from each successful upright proposal bucket. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.
