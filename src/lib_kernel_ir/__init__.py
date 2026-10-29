"""A kernel IR as data — the nine ops, their terms, and their denotations.

ADR-0165: the kernel's combinators stay closed; the IR enters as a
`Language` and is interpreted by an engine. Every op carries a signature, a
grade, a cost and a torch denotation, so two terms being equal is a test
rather than a claim.

    from lib_kernel_ir import qgemv, shifts
    term = qgemv(k=4864, split=4)
    out = term.denote({"w": packed, "z": zeros, "s": scales,
                       "x": act, "shifts": shifts(), "out": buf})
"""

from __future__ import annotations

from lib_kernel_ir.ops import EXACT, OPS, ROUNDS, Cost, Op
from lib_kernel_ir.term import Leaf, Term, from_json, qgemv, shifts, to_json

__all__ = [
    "EXACT",
    "OPS",
    "ROUNDS",
    "Cost",
    "Leaf",
    "Op",
    "Term",
    "from_json",
    "qgemv",
    "shifts",
    "to_json",
]
