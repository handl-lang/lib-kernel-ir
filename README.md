# lib-kernel-ir

The kernel IR as data: **twenty-one ops**, each with a signature, a grade,
a cost and a **torch denotation**, plus `Term` (op + args) so a kernel
body is a value the search can range over.

`split` is one op — `n_parts` is a parameter, not a flag: all four
variants share a composed denotation, so "exact" is a test rather than a
hope. `pipeline` is the schedule op — a schedule-free IR can only express
naive schedules. The pointwise set (`mul`/`sub`/`neg`/`relu`/`exp`/
`sigmoid`/`tanh`/`gelu`/`silu`) and the layout views (`transpose`/
`permute`) round out the surface.

## Packages

- `lib_kernel_ir` (`src/lib_kernel_ir`) — the IR: `OPS` maps each op name
  to its `Op` record (signature, grade, cost, torch denotation);
  `Term(op, *args)` builds a kernel body.

## Develop

```sh
uv sync --group dev
```

or as a git dependency:

```toml
lib-kernel-ir = { git = "https://github.com/handl-lang/lib-kernel-ir.git", tag = "v0.1.0" }
```

## Verify

```sh
uv run scripts/gate.py      # pytest → ruff → ty → lint-imports → coverage floor
```

`scripts/gate.py` is a stdlib runner over the dev toolchain; `workflows/gate.yaml` is the
gate entry. Keep only the steps the repo configures (check `pyproject.toml`).

## License

MIT.
