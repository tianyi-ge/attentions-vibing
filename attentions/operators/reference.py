from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _apply_sliding_window_mask(scores: torch.Tensor, window_size: int, causal: bool) -> torch.Tensor:
    q_positions = torch.arange(scores.size(-2), device=scores.device)
    k_positions = torch.arange(scores.size(-1), device=scores.device)
    distance = q_positions[:, None] - k_positions[None, :]
    mask = distance.abs() > window_size
    if causal:
        mask = mask | (distance < 0)
    return scores.masked_fill(mask, float("-inf"))


def torch_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def naive_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None = None,
    window_size: int = 0,
) -> torch.Tensor:
    scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        q_pos = torch.arange(scores.size(-2), device=scores.device)
        k_pos = torch.arange(scores.size(-1), device=scores.device)
        causal_mask = k_pos[None, :] > q_pos[:, None]
        scores = scores.masked_fill(causal_mask, float("-inf"))
    if window_size > 0:
        scores = _apply_sliding_window_mask(scores, window_size=window_size, causal=causal)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
