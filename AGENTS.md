# lib-kernel-ir

The kernel IR **as data** (Python package `lib_kernel_ir`): twenty-one
ops, each with a signature, a grade, a cost and a torch denotation,
plus `Term(op, *args)` so a kernel body is a value the search can range
over. Transitional package — no `.hnl` module surface yet
(`pkg.hnl` `modules` is empty).

## Layout

| path | role |
| --- | --- |
| `src/lib_kernel_ir/term.py` | `Term` — op + args kernel body |
| `src/lib_kernel_ir/ops.py` | `OPS` — op name → `Op` record (signature, grade, cost, torch denotation) |
| `src/lib_kernel_ir/__init__.py` | package exports |
| `tests/test_ops.py`, `tests/test_json.py` | pytest suite |
| `pyproject.toml`, `uv.lock` | uv project + lock |
| `scripts/gate.py` | stdlib gate runner |
| `workflows/gate.yaml` | CI — shared `handl-gates` campaign |

## Contract

- `split` is one op (`n_parts` is a parameter); `pipeline` is the
  schedule op; the pointwise set (`mul`/`sub`/`neg`/`relu`/`exp`/
  `sigmoid`/`tanh`/`gelu`/`silu`) and layout views (`transpose`/
  `permute`) round out the surface.
- Python-only repo — no `.hnl`, no Rust.

## Verification

```bash
uv sync --group dev
uv run scripts/gate.py   # pytest → ruff → ty → lint-imports → coverage floor
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
