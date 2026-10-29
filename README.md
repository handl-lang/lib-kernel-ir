# lib-kernel-ir

The kernel IR as data (ADR-0160): nine ops, each with a signature, a grade,
a cost and a **torch denotation**, plus `Term` (op + args) so a kernel body
is a value the search can range over.

`split_k` is one op, not a flag: all four variants share a composed
denotation, so "exact" is a test rather than a hope.

```bash
pytest -q          # needs torch; no GPU
```
