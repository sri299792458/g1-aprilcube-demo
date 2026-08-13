# Left-Dex3 upright U supported-pickup qualification

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **1837**
- PASS: **447**
- FAIL: **1390**
- Retained by at least two digit chains: **470**
- Hand/table collision: **34**
- U/table contact in final hold: **1363**

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| upright_on_leg_ends | 1837 | 447 | 1390 |

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
| insufficient_final_digit_contacts | 4 |
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 11 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 1352 |
| pass | 447 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

447 of 1837 geometry-clear proposals passed the complete physical
contract for the U standing upright on both leg ends. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.
