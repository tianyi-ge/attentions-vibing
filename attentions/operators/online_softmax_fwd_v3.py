from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_attn_fwd_kernel_v3(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_QM: tl.constexpr,
    BLOCK_KN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    Sq, Sk,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // H
    hid = pid_bh % H

    # stride_qm is divided to chunks of size BLOCK_QM
    # q_m_ids is the current row ids
    q_m_ids = pid_m * BLOCK_QM + tl.arange(0, BLOCK_QM)
    q_d_ids = tl.arange(0, BLOCK_D)
    k_n_offsets = tl.arange(0, BLOCK_KN)

    # calculate q ptrs for each position [BLOCK_QM, BLOCK_D]
    q_ptrs = (
        q_ptr
        + bid * stride_qb
        + hid * stride_qh
        + q_m_ids[:, None] * stride_qm
        + q_d_ids[None, :] * stride_qd
    )
    q_mask = (q_m_ids[:, None] < Sq) & (q_d_ids[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = q * SCALE

    max_so_far = tl.full((BLOCK_QM,), -float("inf"), dtype=tl.float32)
    denom = tl.zeros((BLOCK_QM,), dtype=tl.float32)
    numerator = tl.zeros((BLOCK_QM, BLOCK_D), dtype=tl.float32)

    for start_kn in range(0, Sk, BLOCK_KN):
        k_n_ids = start_kn + k_n_offsets
        # [BLOCK_KN, BLOCK_D]
        k_ptrs = (
            k_ptr
            + bid * stride_kb
            + hid * stride_kh
            + k_n_ids[:, None] * stride_kn
            + q_d_ids[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + bid * stride_vb
            + hid * stride_vh
            + k_n_ids[:, None] * stride_vn
            + q_d_ids[None, :] * stride_vd
        )
        k_mask = (k_n_ids[:, None] < Sk) & (q_d_ids[None, :] < D)
        v_mask = k_mask
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        # [BLOCK_QM, BLOCK_KN]
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)  # fp16/bf16 input, fp32 accumulator
        # remove elements out of boundary
        valid_scores = (q_m_ids[:, None] < Sq) & (k_n_ids[None, :] < Sk)
        if IS_CAUSAL:
            valid_scores = valid_scores & (k_n_ids[None, :] <= q_m_ids[:, None])
        if WINDOW_SIZE > 0:
            distance = q_m_ids[:, None] - k_n_ids[None, :]
            valid_scores = valid_scores & (distance <= WINDOW_SIZE) & (distance >= -WINDOW_SIZE)
        scores = tl.where(valid_scores, scores, -float("inf"))

        max_block = tl.max(scores, axis=1)
        has_valid_scores = max_block != -float("inf")
        max_new = tl.maximum(max_so_far, max_block)
        rescale_coeff = tl.where(has_valid_scores, tl.exp(max_so_far - max_new), 1.0)  # [BLOCK_QM]
        # exp value of the current block
        new_items = tl.exp(scores - max_new[:, None])
        new_items = tl.where(valid_scores, new_items, 0.0)

        # sum(exp(zi - max_so_far)) * exp(max_so_far - max_new) = sum(exp(zi - max_new))
        denom = denom * rescale_coeff + tl.sum(new_items, axis=1)
        # O_i = softmax * V_j = sum_j[ exp(score_ij)/denom ] * V_j = sum_j[ exp(score_ij) * V_j / denom ]
        p = new_items.to(v.dtype)
        numerator = numerator * rescale_coeff[:, None] + tl.dot(p, v)
        max_so_far = tl.where(has_valid_scores, max_new, max_so_far)

    o = numerator / denom[:, None]
    o_ptrs = (
        o_ptr
        + bid * stride_ob
        + hid * stride_oh
        + q_m_ids[:, None] * stride_om
        + q_d_ids[None, :] * stride_od
    )
    o_mask = (q_m_ids[:, None] < Sq) & (q_d_ids[None, :] < D)
    tl.store(o_ptrs, o, mask=o_mask)


def online_softmax_attention_forward_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None = None,
    window_size: int = 0,
    block_qm: int = 32,
    block_kn: int = 32,
    output_dtype: torch.dtype | None = None,
    num_warps: int | None = 4,
    num_stages: int | None = 2,
) -> torch.Tensor:
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise NotImplementedError("online_softmax_fwd_v3 Triton kernel requires CUDA tensors.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to have shape [B, H, S, D].")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Batch sizes of q, k, v must match.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("This Triton kernel only supports Hq == Hkv.")
    if k.shape[2] != v.shape[2] or k.shape[3] != v.shape[3]:
        raise ValueError("k and v must agree on sequence length and head dimension.")
    if q.shape[3] != k.shape[3]:
        raise ValueError("q and k must have the same head dimension.")
    if window_size < 0:
        raise ValueError("window_size must be non-negative.")

    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    if D > 128:
        raise NotImplementedError("This Triton kernel currently supports head_dim <= 128.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o_dtype = output_dtype if output_dtype is not None else q.dtype
    o = torch.empty(q.shape, device=q.device, dtype=o_dtype)
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    block_d = triton.next_power_of_2(D)  # D is per-head dim, usually 64/128/256
    grid = (triton.cdiv(Sq, block_qm), B * H)
    launch_kwargs = {}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    online_softmax_attn_fwd_kernel_v3[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H=H,
        SCALE=scale,
        D=D,
        BLOCK_QM=block_qm,
        BLOCK_KN=block_kn,
        BLOCK_D=block_d,
        IS_CAUSAL=causal,
        WINDOW_SIZE=window_size,
        Sq=Sq,
        Sk=Sk,
        **launch_kwargs,
    )
    return o
