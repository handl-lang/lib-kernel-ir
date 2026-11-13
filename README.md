# lib-kernel-ir

The kernel IR as data (ADR-0165, extended by ADR-0166): ten ops,
each with a signature, a grade, a cost and a **torch denotation**,
plus `Term` (op + args) so a kernel body is a value the search can
range over.

`split` is one op — `n_parts` is a parameter, not a flag: all four
variants share a composed denotation, so "exact" is a test rather
than a hope. (`pipeline`, the tenth op, was added by ADR-0166 — a
schedule-free IR can only express naive schedules.)

```bash
pytest -q          # needs torch; no GPU
```
