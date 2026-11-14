# lib-kernel-ir

The kernel IR as data (ADR-0160/0165, extended by ADR-0166): **ten
ops**, each with a signature, a grade, a cost and a **torch
denotation**, plus `Term` (op + args) so a kernel body is a value the
search can range over.

`split` is one op — `n_parts` is a parameter, not a flag: all four
variants share a composed denotation, so "exact" is a test rather
than a hope. (`pipeline`, the tenth op, was added by ADR-0166 — a
schedule-free IR can only express naive schedules.)

## Install

```bash
uv pip install "lib-kernel-ir @ git+https://github.com/handl-lang/lib-kernel-ir.git@v0.1.0"
# or, from a checkout:
uv pip install -e .
```

## Use

```python
from lib_kernel_ir import Cost, Term, ops

# OPS maps each op name to its Op record (signature, grade, cost,
# torch denotation); Term(op, *args) builds a kernel body.
```

## Test

```bash
pytest -q          # needs torch; no GPU
```

Dual-licensed MIT OR Apache-2.0 — see `LICENSE-MIT` / `LICENSE-APACHE`.
Consumed by `handl-lang/kernelopt` (`mlopt_kernel.search`) and the
kit-kernel-ir toolchain.
