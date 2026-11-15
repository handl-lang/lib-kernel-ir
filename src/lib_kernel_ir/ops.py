"""The twenty-one ops: signature, grade, cost, and a torch denotation each.

Per ADR-0165 the set is closed; ADR-0171 (proposed) extended it with the
layout ops (`transpose`, `permute`) and the pointwise ops (`mul`, `sub`,
`neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`, `silu`) because the
mlopt-IR bridge measured them as the blocking gap. An op that is not here
is not in the kernel IR; if a kernel needs one, that is a finding about
the platform, not a commit.

The **denotation** is the meaning of one application of the op: a plain
torch function, written once, so that "two terms are equal" is a
computable question (`cosine_gate`) rather than a slogan. The **grade**
says what the op does to exactness: `EXACT` preserves bit-identity of
its inputs, `ROUNDS` may round (accumulate/convert). The **cost** is a
model — bytes, arithmetic, launches — used only as a hard bound, never to
rank candidates (ADR-0165 rule 6).

    >>> from lib_kernel_ir.ops import OPS
    >>> sorted(OPS)
    ['bias', 'dequant', 'dot', 'exp', 'gelu', 'mul', 'neg', 'permute', \
'pipeline', 'reduce', 'relu', 'sigmoid', 'silu', 'split', 'stage', \
'store', 'sub', 'tanh', 'tile_load', 'transpose', 'unpack_i4']
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F

__all__ = ["EXACT", "ROUNDS", "Cost", "OPS", "Op"]

EXACT = "exact"
ROUNDS = "rounds"


@dataclass(frozen=True)
class Cost:
    """The enrichment: what one term costs, in machine units."""

    bytes_moved: int = 0
    ops: int = 0
    launches: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            self.bytes_moved + other.bytes_moved,
            self.ops + other.ops,
            self.launches + other.launches,
        )

    def scale(self, factor: int) -> Cost:
        """A body executed on `factor` disjoint k-partitions."""
        return Cost(
            self.bytes_moved * factor,
            self.ops * factor,
            self.launches * factor,
        )

    def to_dict(self) -> dict:
        return {
            "bytes_moved": self.bytes_moved,
            "ops": self.ops,
            "launches": self.launches,
        }


@dataclass(frozen=True)
class Op:
    """One operation of the IR."""

    name: str
    signature: str
    grade: str
    #: bytes/ops/launches for one application, from the element counts of
    #: its inputs — the documented model behind the roofline bound.
    cost: Callable[..., Cost]
    #: the torch reference for one application
    denote: Callable[..., torch.Tensor]

    def __post_init__(self) -> None:
        if self.grade not in (EXACT, ROUNDS):
            raise ValueError(f"{self.name}: unknown grade {self.grade!r}")


def _n(t: torch.Tensor) -> int:
    return int(t.numel())


def _load_cost(src: torch.Tensor) -> Cost:
    return Cost(bytes_moved=2 * _n(src), launches=1)


def _unpack_cost(packed: torch.Tensor, _shifts) -> Cost:
    # read the packed words once, write one value per nibble
    return Cost(bytes_moved=4 * _n(packed) + _n(packed) * 8, ops=_n(packed))


def _dequant_cost(
    q: torch.Tensor, z: torch.Tensor, s: torch.Tensor, _group: int
) -> Cost:
    return Cost(bytes_moved=2 * _n(z) + 2 * _n(s), ops=3 * _n(q))


def _dot_cost(a: torch.Tensor, b: torch.Tensor) -> Cost:
    return Cost(ops=2 * _n(a), bytes_moved=2 * _n(b))


def _split_cost(_k_len: int, _parts: int) -> Cost:
    return Cost(launches=1)


def _stage_cost(partial: torch.Tensor) -> Cost:
    return Cost(bytes_moved=4 * _n(partial), launches=1)


def _reduce_cost(partials: list[torch.Tensor]) -> Cost:
    total = sum(_n(p) for p in partials)
    return Cost(bytes_moved=4 * total, ops=total, launches=1)


def _store_cost(tile: torch.Tensor, _out: torch.Tensor) -> Cost:
    return Cost(bytes_moved=2 * _n(tile), launches=1)


def _bias_cost(tile: torch.Tensor, bias: torch.Tensor) -> Cost:
    return Cost(bytes_moved=2 * _n(bias), ops=_n(tile))


def _unpack_i4(packed: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """``packed`` [k, n/8] int32 -> [k, n] values in plain nibble order."""
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    return nibbles.reshape(packed.shape[0], -1)


def _dequant(
    q: torch.Tensor,
    zero: torch.Tensor,
    scale: torch.Tensor,
    group: int,
) -> torch.Tensor:
    """``(q - zero) * scale`` on per-k rows (grouping is the loader's job)."""
    if q.shape != zero.shape or q.shape != scale.shape:
        raise ValueError(
            f"dequant wants per-k rows sharing a shape, got q{q.shape} "
            f"zero{zero.shape} scale{scale.shape} (group={group})"
        )
    return (q.to(torch.float32) - zero.to(torch.float32)) * scale.to(
        torch.float32
    )


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Contract the k axis: ``[k, n]`` against ``[k]`` -> ``[n]``."""
    return torch.einsum("kn,k->n", a.to(torch.float32), b.to(torch.float32))


def _split(k_len: int, parts: int) -> list[tuple[int, int]]:
    """Partition ``[0, k_len)`` into ``parts`` contiguous ranges."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    step = (k_len + parts - 1) // parts
    return [(lo, min(lo + step, k_len)) for lo in range(0, k_len, step)]


def _stage(partial: torch.Tensor) -> torch.Tensor:
    """Residency annotation: the partial is materialised as fp32."""
    return partial.to(torch.float32)


def _reduce(partials: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(partials)).sum(dim=0)


def _store(tile: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_(tile.to(out.dtype))
    return out


def _bias(tile: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return tile + bias


def _pipeline(depth: int, body: torch.Tensor) -> torch.Tensor:
    """A scheduling annotation: prefetch ``depth`` stages of the k-loop.

    Semantically the identity — that is the point. A schedule changes when
    bytes arrive, never which bytes, so the denotation is unchanged and the
    four `split_k` variants stay provably equal while their costs move.
    """
    if depth < 1:
        raise ValueError(f"pipeline depth must be >= 1, got {depth}")
    return body


def _pipeline_cost(depth: int, body: torch.Tensor) -> Cost:
    # No extra traffic and no extra launch: the depth buys overlap inside
    # the existing dispatch. Understating it is deliberate — cost is a lower
    # bound, and the referee decides whether the overlap materialised.
    return Cost()


# -- ADR-0171: layout and pointwise ops ------------------------------------
#
# The layout ops denote torch *views*: permute/transpose move no bytes in
# the denotation, so they are priced at zero — the `pipeline` precedent.
# A backend that materialises the relayout pays for it at lowering, which
# is measured, not modelled.


def _view_cost(*_args) -> Cost:
    return Cost()


def _broadcast_numel(*ts: torch.Tensor) -> int:
    """Element count of the broadcast output of ``ts``."""
    shape = torch.broadcast_shapes(*[tuple(t.shape) for t in ts])
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _bin_cost(a: torch.Tensor, b: torch.Tensor) -> Cost:
    """Read both operands, write the broadcast output — 2 B/elem."""
    n_out = _broadcast_numel(a, b)
    return Cost(bytes_moved=2 * (_n(a) + _n(b) + n_out), ops=n_out)


def _un_cost(a: torch.Tensor) -> Cost:
    """Read the operand, write the result — 2 B/elem each way."""
    return Cost(bytes_moved=4 * _n(a), ops=_n(a))


def _transpose(t: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Swap axes ``dim0``/``dim1`` — a view, exact."""
    return t.transpose(int(dim0), int(dim1))


def _permute(t: torch.Tensor, *dims: int) -> torch.Tensor:
    """Reorder the axes by ``dims`` — a view, exact."""
    if not dims:
        raise ValueError("permute needs at least one dim")
    if len(dims) != t.dim():
        raise ValueError(
            f"permute wants one dim per axis ({t.dim()}), got {len(dims)}"
        )
    return t.permute(*[int(d) for d in dims])


def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def _sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b


def _neg(a: torch.Tensor) -> torch.Tensor:
    return -a


def _relu(a: torch.Tensor) -> torch.Tensor:
    return torch.relu(a)


def _exp(a: torch.Tensor) -> torch.Tensor:
    return torch.exp(a)


def _sigmoid(a: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(a)


def _tanh(a: torch.Tensor) -> torch.Tensor:
    return torch.tanh(a)


def _gelu(a: torch.Tensor) -> torch.Tensor:
    return F.gelu(a)


def _silu(a: torch.Tensor) -> torch.Tensor:
    return F.silu(a)


#: The IR. Frozen; extension requires an ADR (0165 → 0166 → 0171).
OPS: Mapping[str, Op] = {
    "tile_load": Op(
        "tile_load", "tensor, idx -> tile", EXACT, _load_cost, lambda t: t
    ),
    "unpack_i4": Op(
        "unpack_i4",
        "packed[k, n/8] int32, shifts[8] -> tile[k, n]",
        EXACT,
        _unpack_cost,
        _unpack_i4,
    ),
    "dequant": Op(
        "dequant",
        "q, zero[g, n], scale[g, n], group -> fp tile",
        ROUNDS,
        _dequant_cost,
        _dequant,
    ),
    "dot": Op(
        "dot", "tile[k, n], act[k] -> tile[n]", ROUNDS, _dot_cost, _dot
    ),
    "split": Op(
        "split", "k_len, n_parts -> [k_range]", EXACT, _split_cost, _split
    ),
    "stage": Op(
        "stage",
        "partial[n] -> partial[n] fp32",
        EXACT,
        _stage_cost,
        _stage,
    ),
    "reduce": Op(
        "reduce",
        "[partial[n]] -> tile[n]",
        ROUNDS,
        _reduce_cost,
        _reduce,
    ),
    "store": Op(
        "store",
        "tile[n], out[n] -> out",
        ROUNDS,
        _store_cost,
        _store,
    ),
    "bias": Op(
        "bias",
        "tile[n], bias[n] -> tile[n]",
        ROUNDS,
        _bias_cost,
        _bias,
    ),
    "pipeline": Op(
        "pipeline",
        "depth, body -> body (scheduled)",
        EXACT,
        _pipeline_cost,
        _pipeline,
    ),
    # -- ADR-0171 ---------------------------------------------------------
    "transpose": Op(
        "transpose",
        "tensor, dim0, dim1 -> tensor (axes swapped)",
        EXACT,
        _view_cost,
        _transpose,
    ),
    "permute": Op(
        "permute",
        "tensor, dims... -> tensor (axes reordered)",
        EXACT,
        _view_cost,
        _permute,
    ),
    "mul": Op("mul", "a, b -> a * b (broadcast)", ROUNDS, _bin_cost, _mul),
    "sub": Op("sub", "a, b -> a - b (broadcast)", ROUNDS, _bin_cost, _sub),
    "neg": Op("neg", "a -> -a", EXACT, _un_cost, _neg),
    "relu": Op("relu", "a -> max(a, 0)", EXACT, _un_cost, _relu),
    "exp": Op("exp", "a -> exp(a)", ROUNDS, _un_cost, _exp),
    "sigmoid": Op(
        "sigmoid", "a -> sigmoid(a)", ROUNDS, _un_cost, _sigmoid
    ),
    "tanh": Op("tanh", "a -> tanh(a)", ROUNDS, _un_cost, _tanh),
    "gelu": Op("gelu", "a -> gelu(a)", ROUNDS, _un_cost, _gelu),
    "silu": Op(
        "silu", "a -> silu(a) = a * sigmoid(a)", ROUNDS, _un_cost, _silu
    ),
}
#: Twenty-one: ten through ADR-0165/0166, eleven more by ADR-0171
#: (layout + pointwise — the measured mlopt-bridge gap).
assert len(OPS) == 21, "the IR is twenty-one ops (ADR-0165/0166/0171)"
