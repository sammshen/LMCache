# SPDX-License-Identifier: Apache-2.0
"""Layout-agnostic KV extraction in the vLLM kvcache-check API.

vLLM moved the FlashAttention K/V axis from dim 0 (K/V-major) to dim 1
(num-blocks-major) in https://github.com/vllm-project/vllm/pull/42095, aligning
it with FlashInfer. These tests pin that ``cache_api`` extracts identical data
from both physical layouts.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.internal_api_server.vllm.cache_api import (
    _extract_kv_at_slots,
    _mha_kv_dim,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
NUM_HEADS = 2
HEAD_SIZE = 16
NUM_SLOTS = NUM_BLOCKS * BLOCK_SIZE


def _kv_major() -> torch.Tensor:
    """[2, num_blocks, block_size, num_heads, head_size] (legacy FlashAttention)."""
    torch.manual_seed(0)
    return torch.randn(2, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE)


def _num_blocks_major(kv_major: torch.Tensor) -> torch.Tensor:
    """Same logical data as ``kv_major`` in [num_blocks, 2, ...] layout."""
    return kv_major.permute(1, 0, 2, 3, 4).contiguous()


class TestMhaKvDim:
    def test_kv_major(self):
        assert _mha_kv_dim(_kv_major()) == 0

    def test_num_blocks_major(self):
        assert _mha_kv_dim(_num_blocks_major(_kv_major())) == 1

    def test_degenerate_num_blocks_2_resolves_num_blocks_major(self):
        # When num_blocks == 2 both dims are size 2; dim 1 wins (forward layout).
        assert _mha_kv_dim(torch.randn(2, 2, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE)) == 1

    def test_no_kv_axis_raises(self):
        with pytest.raises(ValueError):
            _mha_kv_dim(torch.randn(4, 4, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE))


class TestExtractKvAtSlots:
    @pytest.mark.parametrize("slots", [[0, 5, 13, 31], [0], list(range(NUM_SLOTS))])
    def test_both_layouts_extract_identical_data(self, slots):
        kv_major = _kv_major()
        nb_major = _num_blocks_major(kv_major)
        slot_tensor = torch.tensor(slots, dtype=torch.long)

        out_major = _extract_kv_at_slots(kv_major, slot_tensor)
        out_nb = _extract_kv_at_slots(nb_major, slot_tensor)

        # Canonical output is [2, num_slots, num_heads, head_size] regardless of
        # input layout.
        assert out_major.shape == (2, len(slots), NUM_HEADS, HEAD_SIZE)
        assert out_nb.shape == out_major.shape
        assert torch.equal(out_major, out_nb)

    def test_extracted_values_match_manual_gather(self):
        kv_major = _kv_major()
        slots = [0, 7, 8, 31]
        slot_tensor = torch.tensor(slots, dtype=torch.long)

        # Reference: flatten (num_blocks, block_size) -> slot, gather.
        expected = kv_major.reshape(2, NUM_SLOTS, NUM_HEADS, HEAD_SIZE)[:, slot_tensor]

        assert torch.equal(_extract_kv_at_slots(kv_major, slot_tensor), expected)
        assert torch.equal(
            _extract_kv_at_slots(_num_blocks_major(kv_major), slot_tensor), expected
        )
