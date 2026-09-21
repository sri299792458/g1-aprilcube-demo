# Left-Dex3 upright U exhaustive PASS replay 1

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **447**
- PASS: **406**
- FAIL: **41**
- Retained by at least two digit chains: **406**
- Hand/table collision: **0**
- U/table contact in final hold: **40**

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| upright_on_leg_ends | 447 | 406 | 41 |

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
| insufficient_final_digit_contacts | 1 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 40 |
| pass | 406 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

406 of 447 geometry-clear proposals passed the complete physical
contract for all 447 discovery PASS records for the upright U. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.
