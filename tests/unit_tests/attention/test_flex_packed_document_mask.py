# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.attention.flex_attention import FlexAttention
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.layers import GptOssAttention
from nemo_automodel.components.models.gpt_oss.model import GptOssForCausalLM

# Over the default 5s budget on purpose: CUDA FlexAttention compilation takes longer on a cold worker.
pytestmark = pytest.mark.timeout(120)
COLD_FLEX = pytest.mark.runtime_budget(
    60, hard_timeout=120, reason="first FlexAttention call compiles on a cold worker"
)

# Packed layout: real lengths [5, 3, 6] in physical slots [6, 4, 6] (one pad token after each of the
# first two documents), then 2 trailing pad tokens past the last boundary. T = 18.
REAL_LENS = [5, 3, 6]
SLOT_LENS = [6, 4, 6]
TOTAL = 18


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.fixture
def flex_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple[torch.device, torch.dtype]:
    """CPU runs eager flex in fp32; CUDA keeps the compiled kernel the training run uses, in bf16."""
    device = torch.device(request.param)
    monkeypatch.setattr(FlexAttention, "block_masks", {})
    monkeypatch.setattr(FlexAttention, "document_block_masks", {})
    monkeypatch.setattr(FlexAttention, "document_boundaries", None)
    if device.type == "cpu":
        monkeypatch.setattr(FlexAttention, "flex_attn", flex_attention)
        return device, torch.float32
    return device, torch.bfloat16


def _tolerances(dtype: torch.dtype) -> dict[str, float]:
    return {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 1e-5, "rtol": 1e-5}


def _boundaries(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32, device=device)


def _real_positions() -> list[torch.Tensor]:
    starts = [0] + torch.tensor(SLOT_LENS).cumsum(0).tolist()[:-1]
    return [torch.arange(start, start + length) for start, length in zip(starts, REAL_LENS)]


def _dense_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: torch.Tensor, scale: float, window: int
) -> torch.Tensor:
    """Single-document attention with sinks: softmax over [scores, sink] then drop the sink column."""
    _, num_q_heads, seq_len, _ = q.shape
    groups = num_q_heads // k.shape[1]
    k = k.repeat_interleave(groups, dim=1).float()
    v = v.repeat_interleave(groups, dim=1).float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k) * scale
    q_idx = torch.arange(seq_len)[:, None]
    kv_idx = torch.arange(seq_len)[None, :]
    allowed = kv_idx <= q_idx
    if window > 0:
        allowed &= q_idx - kv_idx < window
    scores = scores.masked_fill(~allowed.to(scores.device), float("-inf"))
    sink_column = sinks.float().view(1, -1, 1, 1).expand(1, num_q_heads, seq_len, 1)
    weights = torch.softmax(torch.cat([scores, sink_column], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bhqk,bhkd->bhqd", weights, v)


@COLD_FLEX
@pytest.mark.parametrize("flex_backend", _devices(), indirect=True)
@pytest.mark.parametrize("window", [0, 3])
def test_packed_documents_match_per_document_attention(flex_backend, window: int) -> None:
    device, dtype = flex_backend
    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim = 4, 2, 16
    scale = head_dim**-0.5
    q = torch.randn(1, num_q_heads, TOTAL, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, num_kv_heads, TOTAL, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, num_kv_heads, TOTAL, head_dim, device=device, dtype=dtype, requires_grad=True)
    sinks = torch.randn(num_q_heads, device=device, dtype=torch.float32)

    out = FlexAttention()(
        q,
        k,
        v,
        scale=scale,
        sink_weights=sinks,
        sliding_window=window,
        enable_gqa=True,
        cu_seqlens=_boundaries(SLOT_LENS, device),
    )

    real = torch.cat(_real_positions()).to(device)
    out[:, :, real].float().square().sum().backward()

    for positions in _real_positions():
        positions = positions.to(device)
        q_doc = q.detach()[:, :, positions].clone().requires_grad_(True)
        k_doc = k.detach()[:, :, positions].clone().requires_grad_(True)
        v_doc = v.detach()[:, :, positions].clone().requires_grad_(True)
        expected = _dense_reference(q_doc, k_doc, v_doc, sinks, scale, window)
        expected.square().sum().backward()

        torch.testing.assert_close(out[:, :, positions].float(), expected, **_tolerances(dtype))
        torch.testing.assert_close(q.grad[:, :, positions].float(), q_doc.grad.float(), **_tolerances(dtype))
        torch.testing.assert_close(k.grad[:, :, positions].float(), k_doc.grad.float(), **_tolerances(dtype))
        torch.testing.assert_close(v.grad[:, :, positions].float(), v_doc.grad.float(), **_tolerances(dtype))

    # Pad tokens are never attended to by real tokens, so no gradient reaches their keys/values.
    pad = torch.tensor([i for i in range(TOTAL) if i not in set(real.tolist())], device=device)
    assert torch.count_nonzero(k.grad[:, :, pad]) == 0
    assert torch.count_nonzero(v.grad[:, :, pad]) == 0


@COLD_FLEX
@pytest.mark.parametrize("flex_backend", _devices(), indirect=True)
def test_document_mask_is_rebuilt_for_a_new_layout(flex_backend) -> None:
    device, dtype = flex_backend
    torch.manual_seed(1)
    q = torch.randn(1, 2, TOTAL, 16, device=device, dtype=dtype)
    k = torch.randn(1, 2, TOTAL, 16, device=device, dtype=dtype)
    v = torch.randn(1, 2, TOTAL, 16, device=device, dtype=dtype)
    sinks = torch.zeros(2, device=device)
    attention = FlexAttention()

    attention(q, k, v, sink_weights=sinks, cu_seqlens=_boundaries([9, 9], device))
    attention(q, k, v, sink_weights=sinks, sliding_window=3, cu_seqlens=_boundaries([9, 9], device))
    assert len(FlexAttention.document_block_masks) == 2

    # Same shapes, different boundaries: a stale mask would silently mix documents.
    out = attention(q, k, v, sink_weights=sinks, cu_seqlens=_boundaries([4, 14], device))
    assert FlexAttention.document_boundaries == (0, 4, 18)
    assert len(FlexAttention.document_block_masks) == 1

    for positions in (torch.arange(0, 4), torch.arange(4, 18)):
        positions = positions.to(device)
        expected = _dense_reference(
            q[:, :, positions], k[:, :, positions], v[:, :, positions], sinks, 16**-0.5, window=0
        )
        torch.testing.assert_close(out[:, :, positions].float(), expected, **_tolerances(dtype))


@COLD_FLEX
@pytest.mark.parametrize("flex_backend", _devices(), indirect=True)
@pytest.mark.parametrize("use_sliding_attention", [False, True])
def test_gpt_oss_thd_flex_matches_unpacked_documents(flex_backend, use_sliding_attention: bool) -> None:
    device, dtype = flex_backend
    torch.manual_seed(2)
    config = GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=64,
        sliding_window=3,
        layer_types=["sliding_attention", "full_attention"],
        num_local_experts=2,
        num_experts_per_tok=1,
    )
    backend = BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )
    attention = GptOssAttention(config, backend, use_sliding_attention=use_sliding_attention).to(device, dtype)
    with torch.no_grad():
        attention.sinks.normal_()

    x = torch.randn(TOTAL, config.hidden_size, device=device, dtype=dtype)
    # Per-token [cos, sin] as built from position_ids that restart at 0 in every slot.
    position_ids = torch.cat([torch.arange(length) for length in SLOT_LENS] + [torch.arange(TOTAL - sum(SLOT_LENS))])
    inv_freq = 1.0 / (10000 ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
    angles = position_ids[:, None].float() * inv_freq[None, :]
    freqs_cis = torch.cat([angles.cos(), angles.sin()], dim=-1).to(device)

    with torch.no_grad():
        packed = attention(
            x,
            freqs_cis,
            qkv_format="thd",
            cu_seqlens=_boundaries(REAL_LENS, device),
            cu_seqlens_padded=_boundaries(SLOT_LENS, device),
        )
        for positions in _real_positions():
            positions = positions.to(device)
            expected = attention(x[positions].unsqueeze(0), freqs_cis[positions].unsqueeze(0)).squeeze(0)
            torch.testing.assert_close(packed[positions].float(), expected.float(), **_tolerances(dtype))


class _PackedModel(torch.nn.Module):
    def __init__(self, attn: str, consumes_thd: bool) -> None:
        super().__init__()
        self.backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", rope_fusion=False)
        self._flex_attention_consumes_thd = consumes_thd


def _loader_batch() -> dict:
    # Row 0: documents of 3 and 4 tokens, one pack pad in the last slot. Row 1: one 8-token document.
    return {
        "input_ids": torch.arange(16).reshape(2, 8),
        "labels": torch.arange(16).reshape(2, 8),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4], list(range(8))]),
        "seq_lens": torch.tensor([[3, 4], [8, -1000]]),
        "seq_lens_padded": torch.tensor([[3, 5], [8, -1000]]),
        "qkv_format": "thd",
    }


def test_gpt_oss_opts_into_thd_flattening_for_flex() -> None:
    assert GptOssForCausalLM._flex_attention_consumes_thd is True


@pytest.mark.parametrize(("attn", "consumes_thd", "flattened"), [("flex", True, True), ("flex", False, False)])
def test_packed_batch_is_flattened_only_for_flex_models_that_consume_thd(
    attn: str, consumes_thd: bool, flattened: bool
) -> None:
    batch = _loader_batch()
    sharder = ContextParallelSharder(model=_PackedModel(attn, consumes_thd), batch=batch)
    _, prepared = sharder.shard(batch)

    if not flattened:
        assert prepared["input_ids"].shape == (2, 8)
        return
    assert prepared["input_ids"].shape == (16,)
    assert prepared["labels"].shape == (16,)
    # Row 0's pad is internal to the flattened stream, so the physical boundaries differ from the real ones.
    assert prepared["cu_seqlens_padded"].tolist() == [0, 3, 8, 16]
    assert prepared["cu_seqlens"].tolist() == [0, 3, 7, 15]
    assert prepared["pad_between_seqs"] is True


@pytest.mark.parametrize("window", [0, 37, 128, 300])
@pytest.mark.parametrize(
    "lengths",
    [
        [1000],  # one document over many blocks
        [130, 1, 127, 256, 300, 50],  # boundaries on, before and after block edges
        [600, 250],  # 150 trailing tokens past the last boundary
    ],
)
def test_document_block_mask_matches_dense_construction(
    monkeypatch: pytest.MonkeyPatch, lengths: list[int], window: int
) -> None:
    """The block-level construction keeps exactly create_block_mask's partial and full blocks."""
    monkeypatch.setattr(FlexAttention, "document_block_masks", {})
    monkeypatch.setattr(FlexAttention, "document_boundaries", None)
    seq_len = 1000
    cu_seqlens = _boundaries(lengths, torch.device("cpu"))
    mask = FlexAttention._get_document_block_mask(cu_seqlens, window, seq_len, torch.device("cpu"))
    reference = create_block_mask(mask.mask_mod, 1, None, seq_len, seq_len, device="cpu", _compile=False)

    def block_sets(num_blocks: torch.Tensor, indices: torch.Tensor) -> list[set[int]]:
        return [set(row[:n].tolist()) for n, row in zip(num_blocks[0, 0], indices[0, 0])]

    assert block_sets(mask.kv_num_blocks, mask.kv_indices) == block_sets(reference.kv_num_blocks, reference.kv_indices)
    assert block_sets(mask.full_kv_num_blocks, mask.full_kv_indices) == block_sets(
        reference.full_kv_num_blocks, reference.full_kv_indices
    )
    assert torch.equal(mask.to_dense(), reference.to_dense())
