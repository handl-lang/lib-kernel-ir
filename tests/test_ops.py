"""The four `split_k` variants denote the same function, and it is AWQ's.

The claim `p14_correct_splitk.md` made — "split-k is exact" — could not be
checked, because its reference shared the kernel's nibble order (P22).
Here the denotation of every variant is compared against a wide
dequantise-then-matmul written independently of the IR, on the *same*
packed weights.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from lib_kernel_ir import (
    OPS,
    Cost,
    Leaf,
    Term,
    from_json,
    qgemv,
    shifts,
    to_json,
)
from lib_kernel_ir.ops import EXACT, ROUNDS


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def awq_env(n: int = 64, k: int = 512, group: int = 128, seed: int = 0):
    """Packed int4 weights, group scales/zeros, one activation."""
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(0, 2**31 - 1, (k, n // 8), generator=g, dtype=torch.int32)
    zeros = torch.randint(0, 16, (k // group, n), generator=g).to(torch.int32)
    scales = torch.rand((k // group, n), generator=g) * 0.1 + 0.01
    return {
        "w": words,
        "z": zeros,
        "s": scales,
        "x": torch.randn(k, generator=g),
        "shifts": shifts(),
        "out": torch.zeros(n),
    }, group


def wide_reference(env, group: int) -> torch.Tensor:
    """Dequantise-then-matmul, written without the IR."""
    q = OPS["unpack_i4"].denote(env["w"], env["shifts"]).float()
    z = env["z"].repeat_interleave(group, dim=0).float()
    s = env["s"].repeat_interleave(group, dim=0).float()
    w = (q - z) * s
    return (w * env["x"][:, None]).sum(dim=0)


# -- the vocabulary (M0 gate) ----------------------------------------------


def test_the_ir_is_twenty_one_ops() -> None:
    assert len(OPS) == 21
    assert set(OPS) == {
        "tile_load",
        "unpack_i4",
        "dequant",
        "dot",
        "split",
        "stage",
        "reduce",
        "store",
        "bias",
        "pipeline",
        # ADR-0171
        "transpose",
        "permute",
        "mul",
        "sub",
        "neg",
        "relu",
        "exp",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
    }


def test_every_op_declares_signature_grade_cost_and_denotation() -> None:
    for name, op in OPS.items():
        assert op.name == name
        assert op.signature, name
        assert op.grade in (EXACT, ROUNDS), name
        assert callable(op.cost), name
        assert callable(op.denote), name


def test_grades_say_what_rounds() -> None:
    assert OPS["unpack_i4"].grade == EXACT
    assert OPS["tile_load"].grade == EXACT
    assert OPS["dot"].grade == ROUNDS
    assert OPS["dequant"].grade == ROUNDS


def test_each_op_denotation_on_small_tensors() -> None:
    w = torch.randint(0, 2**31 - 1, (16, 2), dtype=torch.int32)
    assert OPS["tile_load"].denote(w).shape == (16, 2)
    unpacked = OPS["unpack_i4"].denote(w, shifts())
    assert unpacked.shape == (16, 16)
    assert int(unpacked.min()) >= 0 and int(unpacked.max()) <= 15

    z = torch.full_like(unpacked, 8, dtype=torch.int32)
    s = torch.full(unpacked.shape, 0.5)
    dq = OPS["dequant"].denote(unpacked, z, s, 8)
    torch.testing.assert_close(dq, (unpacked.float() - 8) * 0.5)

    act = torch.ones(16)
    assert OPS["dot"].denote(dq, act).shape == (16,)
    assert OPS["split"].denote(16, 4) == [(0, 4), (4, 8), (8, 12), (12, 16)]
    part = OPS["stage"].denote(torch.ones(4))
    assert part.dtype == torch.float32
    assert torch.allclose(OPS["reduce"].denote([part, part]), torch.full((4,), 2.0))
    out = torch.zeros(4)
    assert OPS["store"].denote(torch.ones(4), out) is out
    torch.testing.assert_close(OPS["bias"].denote(torch.ones(4), torch.ones(4)), torch.full((4,), 2.0))


def test_split_rejects_zero_parts() -> None:
    with pytest.raises(ValueError, match="parts"):
        OPS["split"].denote(16, 0)


def test_group_broadcast_is_the_loader_not_the_kernel() -> None:
    q = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="per-k rows"):
        OPS["dequant"].denote(q, torch.zeros(2, 4), torch.ones(2, 4), 4)


# -- the M1 gate: four split_k variants, one meaning -----------------------


@pytest.mark.parametrize("split", [1, 2, 4, 8])
def test_every_split_denotes_the_wide_reference(split: int) -> None:
    env, group = awq_env(k=512, n=64)
    got = qgemv(k=512, split=split).denote(env)
    want = wide_reference(env, group)
    assert cosine(got, want) >= 0.99999, (split, cosine(got, want))


def test_all_four_splits_agree_pairwise() -> None:
    env, _ = awq_env(k=512, n=64)
    outs = {p: qgemv(k=512, split=p).denote(env) for p in (1, 2, 4, 8)}
    for a in (1, 2, 4, 8):
        for b in (1, 2, 4, 8):
            assert cosine(outs[a], outs[b]) >= 0.99999, (a, b)


def test_split_is_not_aligned_to_the_quant_group() -> None:
    """k=512, group=128: a 3-way split cuts mid-group and must still be exact.

    The kernel indexes ``g = rk // GS`` per element rather than requiring
    aligned ranges, and a denotation that assumed alignment would call a
    legal kernel wrong.
    """
    env, group = awq_env(k=512, n=64)
    want = wide_reference(env, group)
    for split in (3, 5, 6, 7):
        got = qgemv(k=512, split=split).denote(env)
        assert cosine(got, want) >= 0.99999, (split, cosine(got, want))


def test_cost_grows_with_launches_and_keeps_weight_traffic() -> None:
    env, _ = awq_env(k=512, n=64)
    costs = {p: qgemv(k=512, split=p).cost(env) for p in (1, 2, 4, 8)}
    launches = [costs[p].launches for p in (1, 2, 4, 8)]
    assert launches == sorted(launches), launches
    assert launches[0] < launches[-1]
    weight_bytes = 512 * 64 * 0.5
    for p, cost in costs.items():
        assert cost.bytes_moved >= weight_bytes, (p, cost)
        assert cost.ops > 0, p


def test_cost_is_a_bound_not_a_ranking() -> None:
    """split=8 costs more but is not thereby worse — no comparison API."""
    env, _ = awq_env(k=512, n=64)
    cheap, dear = qgemv(k=512, split=1).cost(env), qgemv(k=512, split=8).cost(env)
    assert isinstance(cheap, Cost) and isinstance(dear, Cost)
    assert not hasattr(cheap, "better_than")
    assert dear.bytes_moved >= cheap.bytes_moved


def test_unknown_op_is_refused() -> None:
    with pytest.raises(ValueError, match="not a kernel-ir op"):
        Term("matmul", ())


def test_qgemv_requires_group_divisible_k() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        qgemv(k=500, split=2, group=128)


# -- ADR-0166: a schedule annotation that means the identity -------------


def test_pipeline_denotes_the_identity() -> None:
    env, _ = awq_env(k=512, n=64)
    plain = qgemv(k=512, split=2).denote(env)
    deep = qgemv(k=512, split=2, depth=4).denote(env)
    torch.testing.assert_close(plain, deep)


def test_pipeline_does_not_change_the_cost_bound() -> None:
    env, _ = awq_env(k=512, n=64)
    a = qgemv(k=512, split=2).cost(env)
    b = qgemv(k=512, split=2, depth=4).cost(env)
    assert a.to_dict() == b.to_dict(), (a, b)


def test_pipeline_rejects_zero_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        qgemv(k=512, split=2, depth=0)


# -- ADR-0171: layout and pointwise ops ------------------------------------


def test_layout_ops_are_views_priced_at_zero() -> None:
    """transpose/permute denote torch views: the op itself moves nothing.

    The leaf operand still pays its ``tile_load`` — that is the load,
    not the relayout.
    """
    env, _ = awq_env(k=64, n=32)
    t = Term("transpose", (Leaf("w"), 0, 1))
    assert t.denote(env).shape == (4, 64)  # w is packed [k, n/8]
    assert OPS["transpose"].cost(env["w"], 0, 1) == Cost()
    assert OPS["permute"].cost(env["w"], 0, 1) == Cost()
    # The composed term still pays for reading the leaf once.
    load = OPS["tile_load"].cost(env["w"])
    assert t.cost(env) == load


def test_transpose_denotes_the_axis_swap() -> None:
    w = torch.arange(6).reshape(2, 3)
    torch.testing.assert_close(
        OPS["transpose"].denote(w, 0, 1), w.t()
    )
    torch.testing.assert_close(
        OPS["transpose"].denote(w, -2, -1), w.t()
    )


def test_permute_denotes_the_reorder() -> None:
    x = torch.arange(24).reshape(2, 3, 4)
    torch.testing.assert_close(
        OPS["permute"].denote(x, 0, 2, 1), x.permute(0, 2, 1)
    )
    # permute(2,0,1) equals two transposes composed — same denotation.
    torch.testing.assert_close(
        OPS["permute"].denote(x, 2, 0, 1),
        OPS["transpose"].denote(OPS["transpose"].denote(x, 0, 1), 0, 2),
    )


def test_permute_rejects_wrong_dim_count() -> None:
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="one dim per axis"):
        OPS["permute"].denote(x, 0)
    with pytest.raises(ValueError, match="at least one dim"):
        OPS["permute"].denote(x)


def test_pointwise_denotations() -> None:
    a = torch.tensor([-1.0, 0.0, 2.0])
    b = torch.tensor([2.0, 3.0, 4.0])
    torch.testing.assert_close(OPS["mul"].denote(a, b), a * b)
    torch.testing.assert_close(OPS["sub"].denote(a, b), a - b)
    torch.testing.assert_close(OPS["neg"].denote(a), -a)
    torch.testing.assert_close(OPS["relu"].denote(a), torch.relu(a))
    torch.testing.assert_close(OPS["exp"].denote(a), torch.exp(a))
    torch.testing.assert_close(OPS["sigmoid"].denote(a), torch.sigmoid(a))
    torch.testing.assert_close(OPS["tanh"].denote(a), torch.tanh(a))
    torch.testing.assert_close(OPS["gelu"].denote(a), F.gelu(a))
    torch.testing.assert_close(OPS["silu"].denote(a), F.silu(a))


def test_pointwise_broadcasts_like_torch() -> None:
    a = torch.ones(4, 1)
    b = torch.ones(3)
    out = OPS["mul"].denote(a, b)
    assert out.shape == (4, 3)
    torch.testing.assert_close(out, a * b)


def test_pointwise_costs_count_elements() -> None:
    a = torch.ones(4, 8)
    b = torch.ones(8)
    c = OPS["mul"].cost(a, b)
    assert c.ops == 32
    assert c.bytes_moved == 2 * (32 + 8 + 32)
    u = OPS["relu"].cost(a)
    assert u.ops == 32 and u.bytes_moved == 4 * 32
    assert OPS["neg"].grade == EXACT and OPS["relu"].grade == EXACT
    assert OPS["mul"].grade == ROUNDS and OPS["silu"].grade == ROUNDS


def test_weight_first_gemv_spelling_denotes() -> None:
    """dot(transpose(W,0,1), v) is matmul(W[m,k], v[k]) -> [m]."""
    env, _ = awq_env(k=64, n=32)
    e = dict(env)
    e["w"] = torch.randn(48, 64, generator=torch.Generator().manual_seed(1))
    term = Term(
        "dot",
        (Term("transpose", (Leaf("w"), 0, 1)), Leaf("x")),
    )
    want = torch.matmul(e["w"], e["x"])
    torch.testing.assert_close(term.denote(e), want, rtol=1e-5, atol=1e-5)


def test_new_ops_round_trip_json() -> None:
    term = Term(
        "bias",
        (
            Term(
                "silu",
                (Term("mul", (Term("transpose", (Leaf("w"), 0, 1)), Leaf("v"))),),
            ),
            Leaf("b"),
        ),
    )
    assert from_json(to_json(term)) == term
