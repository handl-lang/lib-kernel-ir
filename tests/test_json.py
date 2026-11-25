"""The term crosses a process boundary as data, and comes back."""

from __future__ import annotations

import json

import pytest

from lib_kernel_ir import Leaf, Term, from_json, qgemv, to_json


def test_round_trip_preserves_the_term() -> None:
    term = qgemv(k=4864, split=4)
    assert from_json(to_json(term)) == term


def test_wire_form_is_what_the_rust_lowering_reads() -> None:
    """Shape asserted here so a change cannot silently break `tool-compile-triton`."""
    wire = json.loads(to_json(qgemv(k=4864, split=4)))
    assert wire["op"] == "store"
    split = wire["args"][0]
    assert split["op"] == "split"
    assert split["args"][0] == 4864
    assert split["args"][1] == 4
    assert wire["args"][1]["leaf"]["name"] == "out"
    body = split["args"][2]
    assert body["op"] == "dot"
    assert body["args"][0]["op"] == "dequant"
    assert body["args"][0]["args"][3] == 128
    assert body["args"][0]["args"][0]["op"] == "unpack_i4"
    assert body["args"][0]["args"][0]["args"][0]["leaf"]["name"] == "w"


def test_leaf_div_survives() -> None:
    assert from_json(to_json(Leaf("z", "g", 128))) == Leaf("z", "g", 128)


def test_bad_payload_is_refused() -> None:
    with pytest.raises(ValueError, match="unrecognised node"):
        from_json('{"nonsense": 1}')
    with pytest.raises(ValueError, match="not a kernel-ir op"):
        from_json('{"op": "matmul", "args": []}')


def test_unserialisable_argument_is_an_error() -> None:
    with pytest.raises(TypeError, match="cannot serialise"):
        to_json(Term("bias", (object(),)))
