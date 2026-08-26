# Pre-fix ledgers (2026-08-25, before the resolution floor and the plateau stop)

Produced by the optimizer as it stood before two changes:

- `local_model(eps_floor_rel=...)`, the floor that stops the eps-active cluster being lost at
  small trust radius. Without it, 8 of these 10 runs ended `trust radius collapsed to 5.68e-15`,
  including every run that found the best answer in its cell.
- the criticality-plateau stop, which replaced an absolute `gain_tol` that certified or not
  according to where each shape's noise floor happened to fall.

Kept as the "before" half of the comparison and NOT to be merged with post-fix results. The
`resolution` field is absent from these rows. Re-running the same cells regenerates them under
the current optimizer in the parent directory.
