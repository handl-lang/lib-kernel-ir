"""Terms: the kernel body as data, with meaning and cost.

A `Term` is an op applied to arguments; arguments are terms, leaves (names
bound in an environment), or literals. Two terms are equal iff their
denotations are equal — checked with an external incumbent
(`bench.benchkit.equality`), never term-against-term.

`split_k` is where this earns its keep. All four variants are built by
`qgemv(split=p)`, they differ *only* in the `parts` literal, and their
composed denotations are therefore provably identical — the claim
`p14_correct_splitk.md` could only assert because its reference shared the
kernel's assumption (P22).

    >>> t1, t4 = qgemv(1), qgemv(4)
    >>> t1.op, t4.op
    ('store', 'store')
    >>> t4.cost(env).launches > t1.cost(env).launches
    True

The cost is a **bound** (ADR-0160 rule 6): it prices bytes and launches so
a candidate can be pruned against the roofline floor, never so a winner
can be chosen by model instead of by measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .ops import OPS, Cost

__all__ = ["Leaf", "Term", "qgemv", "shifts"]


@dataclass(frozen=True)
class Leaf:
    """A name bound in the environment.

    ``kind`` says how a k-partition slices it: ``"k"`` rows are k-indexed,
    ``"g"`` rows are group-indexed (``div`` groups per k element), ``"out"``
    is not sliced at all.
    """

    name: str
    kind: str = "out"
    div: int = 1


@dataclass(frozen=True)
class Term:
    """An op applied to arguments."""

    op: str
    args: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"{self.op!r} is not one of the nine IR ops")

    # -- meaning ---------------------------------------------------
    def denote(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The torch value of this term under ``env``."""
        return _denote(self, env, None)

    # -- price -----------------------------------------------------
    def cost(self, env: Mapping[str, torch.Tensor]) -> Cost:
        """Bytes, ops and launches — computed from shapes, not memory."""
        meta = {
            k: torch.empty(tuple(v.shape), dtype=v.dtype, device="meta")
            for k, v in env.items()
            if isinstance(v, torch.Tensor)
        }
        return _cost(self, meta, None)


def shifts(dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """The eight nibble shifts, as a leaf the IR can carry."""
    return torch.arange(0, 32, 4, dtype=dtype)


def _leaf_value(leaf: Leaf, env, kr):
    value = env[leaf.name]
    if kr is None or leaf.kind == "out":
        return value
    lo, hi = kr
    if leaf.kind == "k":
        return value[lo:hi]
    if leaf.kind == "g":
        # group-indexed rows expanded to per-k rows, so a k-partition may
        # start mid-group (the kernel indexes ``g = rk // GS`` per element,
        # it does not require alignment).
        return value.repeat_interleave(leaf.div, dim=0)[lo:hi]
    raise ValueError(f"leaf {leaf.name!r}: unknown kind {leaf.kind!r}")


def qgemv(k: int, split: int, group: int = 128) -> Term:
    """The GEMV term with ``split`` k-partitions.

    ``split=1`` is a one-partition split, not a separate shape, so all four
    variants share one composed denotation by construction.
    """
    if k % group:
        raise ValueError(f"k={k} is not a multiple of group={group}")
    body = Term(
        "dot",
        (
            Term(
                "dequant",
                (
                    Term(
                        "unpack_i4",
                        (Leaf("w", "k"), Leaf("shifts")),
                    ),
                    Leaf("z", "g", group),
                    Leaf("s", "g", group),
                    group,
                ),
            ),
            Leaf("x", "k"),
        ),
    )
    return Term(
        "store",
        (
            Term("split", (k, split, body)),
            Leaf("out"),
        ),
    )


def _denote(term: Term, env, kr):
    if isinstance(term, Leaf):
        return _leaf_value(term, env, kr)
    op = term.op
    if op == "split":
        k_len, parts, body = term.args
        ranges = OPS["split"].denote(int(k_len), int(parts))
        partials = [
            OPS["stage"].denote(_denote(body, env, r)) for r in ranges
        ]
        return OPS["reduce"].denote(partials)
    values = [
        _denote(a, env, kr) if isinstance(a, (Term, Leaf)) else a
        for a in term.args
    ]
    return OPS[op].denote(*values)


def _cost(term: Term, env, kr) -> Cost:
    if isinstance(term, Term) and term.op == "split":
        k_len, parts, body = term.args
        ranges = OPS["split"].denote(int(k_len), int(parts))
        total = Cost()
        for r in ranges:
            part = _cost(body, env, r)
            staged = _denote(body, env, r)  # meta: shapes only
            total = total + part + OPS["stage"].cost(staged)
        last = _denote(body, env, ranges[-1])
        return total + OPS["reduce"].cost([last] * len(ranges))
    if isinstance(term, Leaf):
        return OPS["tile_load"].cost(_leaf_value(term, env, kr))
    values = [
        _cost(a, env, kr) if isinstance(a, (Term, Leaf)) else a
        for a in term.args
    ]
    op_cost = OPS[term.op].cost
    tensors = [
        _denote(a, env, kr) if isinstance(a, (Term, Leaf)) else a
        for a in term.args
    ]
    return sum(
        (v for v in values if isinstance(v, Cost)),
        OPS[term.op].cost(*tensors),
    )
