# SPDX-License-Identifier: BSD-3-Clause
"""`gguf_kquant` must be able to CARRY i-quant bytes (IQ1_S..IQ4_XS/IQ4_NL) so the loader
can keep them resident once their fused kernels land (superl8#317). Carrying them is a format
question only — `superl8.gguf._NATIVE_FUSED` stays the kernel gate, so a type without a kernel
still loads through dequant -> per_row_i8."""
import pytest
import torch

from superl8.format import _GGUF_KQUANT_BLOCK, _GGUF_KQUANT_GROUP, QTensor

# ggml block sizes (ggml-common.h): bytes per block, values per block.
GGML_GEOMETRY = {
    "q2_k": (84, 256), "q3_k": (110, 256), "q4_k": (144, 256), "q5_k": (176, 256),
    "q6_k": (210, 256), "tq3_4s": (16, 32),
    "iq1_s": (50, 256), "iq2_xxs": (66, 256), "iq2_xs": (74, 256), "iq2_s": (82, 256),
    "iq3_xxs": (98, 256), "iq3_s": (110, 256), "iq4_xs": (136, 256), "iq4_nl": (18, 32),
}


def _qt(codebook, n_blocks=4, out=2):
    ts = _GGUF_KQUANT_BLOCK[codebook]
    return QTensor(torch.zeros(out, n_blocks * ts, dtype=torch.uint8), None,
                   scheme="gguf_kquant", group_size=_GGUF_KQUANT_GROUP.get(codebook, 256),
                   codebook=codebook)


@pytest.mark.parametrize("codebook", sorted(GGML_GEOMETRY))
def test_block_geometry_matches_ggml(codebook):
    nb, vals = GGML_GEOMETRY[codebook]
    assert _GGUF_KQUANT_BLOCK[codebook] == nb
    assert _GGUF_KQUANT_GROUP.get(codebook, 256) == vals


@pytest.mark.parametrize("codebook", sorted(GGML_GEOMETRY))
def test_validate_accepts_every_carried_type(codebook):
    _qt(codebook).validate()


@pytest.mark.parametrize("codebook", ["iq3_s", "iq4_xs", "iq2_s"])
def test_validate_rejects_a_ragged_byte_count(codebook):
    ts = _GGUF_KQUANT_BLOCK[codebook]
    qt = QTensor(torch.zeros(2, 3 * ts + 1, dtype=torch.uint8), None, scheme="gguf_kquant",
                 group_size=256, codebook=codebook)
    with pytest.raises(AssertionError, match="n_superblocks"):
        qt.validate()


def test_validate_rejects_the_wrong_group_size():
    qt = QTensor(torch.zeros(2, 4 * 136, dtype=torch.uint8), None, scheme="gguf_kquant",
                 group_size=32, codebook="iq4_xs")  # IQ4_XS is a 256-value super-block
    with pytest.raises(AssertionError, match="group_size"):
        qt.validate()


def test_validate_still_rejects_an_unknown_tag():
    qt = QTensor(torch.zeros(2, 128, dtype=torch.uint8), None, scheme="gguf_kquant",
                 group_size=256, codebook="iq5_made_up")
    with pytest.raises(AssertionError, match="type tag"):
        qt.validate()


def test_only_kernel_backed_iquants_are_fused():
    """Format support must NOT silently promote a type to the fused path: a tag is in
    _NATIVE_FUSED only once its kernel ships, and never in both maps (coverage fills from
    _NATIVE_FUSED first, then _REQUANT_I8, so being in both demotes it back to requant)."""
    from superl8 import gguf as fgguf

    fused = {"IQ3_S": "iq3_s", "IQ4_XS": "iq4_xs", "IQ3_XXS": "iq3_xxs"}  # work items 2b-2d
    pending = ("IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ4_NL")

    for tag, code in fused.items():
        assert fgguf._NATIVE_FUSED.get(tag) == code, f"{tag} lost its fused kernel"
        assert tag not in fgguf._REQUANT_I8, f"{tag} is in both maps — coverage would demote it"
        assert fgguf.gguf_type_coverage()[tag] == "native_fused"
    for tag in pending:
        assert tag not in fgguf._NATIVE_FUSED, f"{tag} claims a fused kernel it does not have"
        assert tag in fgguf._REQUANT_I8, f"{tag} must still load via requant-i8"
        assert fgguf.gguf_type_coverage()[tag] == "requant_i8"
