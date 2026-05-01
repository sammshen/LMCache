# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the SGLang in-process adapter.

Runs the same store-then-retrieve flow as ``test_sglang_mp_e2e`` but
through :class:`~lmcache.integration.sglang.sglang_adapter.LMCacheConnector`
— a single-process LMCache engine living inside this test process.

Establishes a baseline against which the multi-process path is compared.
"""

# Standard
import sys
import types

# Third Party
import pytest
import torch


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _stub_sglang_module() -> None:
    """Same minimal stub as the MP test — ``ModelConfig`` is type-only."""
    if "sglang" in sys.modules:
        return

    pkg = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    configs = types.ModuleType("sglang.srt.configs")
    model_config = types.ModuleType("sglang.srt.configs.model_config")

    class _ModelConfig:
        pass

    model_config.ModelConfig = _ModelConfig  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "sglang": pkg,
            "sglang.srt": srt,
            "sglang.srt.configs": configs,
            "sglang.srt.configs.model_config": model_config,
        }
    )


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for E2E test")
def test_sglang_inprocess_store_then_retrieve_mla(tmp_path) -> None:
    """In-process LMCacheConnector (MLA): STORE one chunk, RETRIEVE it back.

    Using the MLA shape (head dim absorbs the per-token V) — the
    in-process adapter concatenates ``k_pool + v_pool`` into a flat
    per-layer list, which the format detector recognizes as MLA when
    the second tensor dimension is 1. MHA support requires the
    layerwise connector, which is exercised separately.
    """
    _stub_sglang_module()

    # First Party
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheConnector,
        LoadMetadata,
        StoreMetadata,
    )
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    NL = 4
    PBS = 1024
    HS = 128

    device = torch.device("cuda:0")
    # MLA shape: [PBS, 1, HS] per layer.
    k_pool = [
        torch.randn(PBS, 1, HS, dtype=torch.bfloat16, device=device) for _ in range(NL)
    ]
    # SGLang MLA shares K/V buffers; mirror that here so the engine
    # sees the same per-layer tensor for both halves.
    v_pool = list(k_pool)

    class _FakeModelConfig:
        num_hidden_layers = NL
        head_dim = HS
        model_path = "test-sglang-inprocess"
        is_mla = True

        def get_num_kv_heads(self, _tp_size: int) -> int:
            return 1

    config_yaml = tmp_path / "lmcache.yaml"
    config_yaml.write_text("chunk_size: 128\nlocal_cpu: True\nmax_local_cpu_size: 1\n")
    # Standard
    import os

    os.environ["LMCACHE_CONFIG_FILE"] = str(config_yaml)

    try:
        connector = LMCacheConnector(
            sgl_config=_FakeModelConfig(),
            tp_size=1,
            rank=0,
            k_pool=k_pool,
            v_pool=v_pool,
        )

        chunk_size = connector.chunk_size()
        assert chunk_size == 128

        token_ids = list(range(chunk_size))
        slot_mapping = torch.arange(chunk_size, dtype=torch.int64)
        kv_indices = torch.arange(chunk_size, dtype=torch.int64)

        original_k = [t[:chunk_size].clone() for t in k_pool]
        original_v = [t[:chunk_size].clone() for t in v_pool]

        connector.store_kv(
            StoreMetadata(
                last_node=None,
                token_ids=token_ids,
                kv_indices=kv_indices,
                offset=0,
            )
        )
        torch.cuda.synchronize()

        for t in k_pool:
            t[:chunk_size].zero_()
        for t in v_pool:
            t[:chunk_size].zero_()
        torch.cuda.synchronize()

        num_loaded = connector.load_kv(
            LoadMetadata(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                offset=0,
            )
        )
        assert num_loaded == chunk_size, (
            f"expected {chunk_size} tokens loaded, got {num_loaded}"
        )

        for layer_idx, (k_now, k_orig) in enumerate(
            zip(k_pool, original_k, strict=True)
        ):
            assert torch.allclose(k_now[:chunk_size], k_orig), (
                f"K layer {layer_idx} not restored"
            )
        for layer_idx, (v_now, v_orig) in enumerate(
            zip(v_pool, original_v, strict=True)
        ):
            assert torch.allclose(v_now[:chunk_size], v_orig), (
                f"V layer {layer_idx} not restored"
            )
    finally:
        # Clean up the singleton so the test is rerunnable.
        # First Party
        from lmcache.integration.sglang.utils import ENGINE_NAME

        if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
