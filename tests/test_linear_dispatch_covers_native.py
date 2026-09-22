# SPDX-License-Identifier: BSD-3-Clause
"""`superl8.linear`'s k-quant dispatch table must cover every natively-fused type.

This table replaced a hand-written `if qt.codebook == ...` chain (#328) precisely because
a chain silently rejects a type whose kernel HAS shipped. It then went stale anyway when
#330 added four kernels and the wiring pass updated `_NATIVE_FUSED`, the `linear_*`
helpers, `__init__` exports, the bindings and the launchers -- but not this table. The
result was `ValueError: gguf_kquant type 'iq1_s' has no fused kernel` for a weight whose
kernel was merged and tested.

A table you cannot introspect is an if-chain with nicer syntax. Pin it.
"""
import pytest

from superl8.gguf import _NATIVE_FUSED
from superl8.ops import _FUSED_KQUANT_OPS

# tq3_4s is special-cased inside `linear()`: it derives in_features from TQ3_TYPE_SIZE.
_SPECIAL = {"tq3_4s"}


def test_every_native_fused_type_has_a_dispatch_entry():
    want = set(_NATIVE_FUSED.values()) - _SPECIAL
    missing = sorted(want - set(_FUSED_KQUANT_OPS))
    assert not missing, (
        f"_NATIVE_FUSED claims {missing} are natively fused, but superl8.linear has no "
        f"dispatch entry, so any resident weight of that type raises at serving time"
    )


def test_no_dispatch_entry_without_a_claim():
    """The reverse: a dispatch entry for a type the loader never keeps resident is dead."""
    extra = sorted(set(_FUSED_KQUANT_OPS) - (set(_NATIVE_FUSED.values()) - _SPECIAL))
    assert not extra, f"dispatch entries with no _NATIVE_FUSED claim: {extra}"


@pytest.mark.parametrize("codebook", sorted(set(_NATIVE_FUSED.values()) - _SPECIAL))
def test_each_entry_is_callable(codebook):
    assert callable(_FUSED_KQUANT_OPS[codebook])
