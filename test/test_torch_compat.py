from __future__ import annotations

import pytest
import torch

from layers._mha import MixedScore_MultiHeadAttention, _MHA_V1, _MHA_V2
from layers._transformer import TransformerEncoderLayer
from layers.Mymodel_layers import GraphEncoderLayer


@pytest.mark.parametrize("attention_kind", ["v1", "v2", "mixed"])
def test_attention_layers_accept_legacy_byte_padding_masks(attention_kind: str) -> None:
    torch.manual_seed(7)
    queries = torch.randn(2, 3, 8)
    keys = torch.randn(2, 5, 8)
    values = torch.randn(2, 5, 8)
    cost_mat = torch.randn(2, 3, 5)
    mask = torch.tensor(
        [[False, True, False, False, False], [False, False, False, True, False]],
        dtype=torch.bool,
    )

    if attention_kind == "v1":
        attention = _MHA_V1(head_count=2, query_size=8)
    elif attention_kind == "v2":
        attention = _MHA_V2(head_count=2, query_size=8)
    else:
        attention = MixedScore_MultiHeadAttention(head_count=2, query_size=8)
    attention.eval()

    def apply(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_kind == "mixed":
            return attention(queries, keys, values, attention_mask, cost_mat)
        return attention(queries, keys, values, attention_mask)

    with torch.no_grad():
        expected = apply(mask)
        actual = apply(mask.to(torch.uint8))

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_transformer_encoder_accepts_legacy_byte_padding_masks() -> None:
    torch.manual_seed(11)
    encoder = TransformerEncoderLayer(head_count=2, model_size=8, ff_size=16)
    encoder.eval()
    inputs = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [[False, True, False, False, False], [False, False, True, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        expected = encoder(inputs, mask=mask)
        actual = encoder(inputs, mask=mask.to(torch.uint8))

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_graph_encoder_accepts_legacy_byte_padding_masks() -> None:
    torch.manual_seed(13)
    encoder = GraphEncoderLayer(head_count=2, model_size=8, ff_size=16, dropout=0.0)
    encoder.eval()
    inputs = torch.randn(2, 5, 8)
    cost_mat = torch.rand(2, 5, 5)
    mask = torch.tensor(
        [[False, True, False, False, False], [False, False, True, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        expected = encoder(inputs, cost_mat, mask=mask)
        actual = encoder(inputs, cost_mat, mask=mask.to(torch.uint8))

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
