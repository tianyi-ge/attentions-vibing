from __future__ import annotations

import os
import platform

from attentions.config import BenchmarkCase


RTX_5090_PEAK_TFLOPS = {
    "fp32": 104.8,
    # Dense low-precision Tensor Core estimate. Do not use marketing AI TOPS as MFU denominator.
    "fp16": 419.0,
    "bf16": 419.0,
}


def _detect_cuda_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except Exception:
        return ""
    return ""


def default_gpu_peak_tflops_by_dtype() -> tuple[dict[str, float], str]:
    device_name = _detect_cuda_device_name()
    normalized = device_name.lower().replace("geforce", "").replace("nvidia", "")
    if "rtx 5090" in normalized:
        return dict(RTX_5090_PEAK_TFLOPS), "rtx5090_default"
    return {}, "unconfigured_peak"


def _estimate_cpu_vector_bits() -> int:
    machine = platform.machine().lower()
    capability = ""
    try:
        capability = getattr(__import__("torch").backends.cpu, "get_cpu_capability", lambda: "")().upper()
    except Exception:
        capability = ""
    if "AVX512" in capability:
        return 512
    if "AVX2" in capability:
        return 256
    if machine in {"arm64", "aarch64"}:
        return 128
    return 128


def _estimate_cpu_default_freq_ghz() -> float:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return 3.2
    return 3.5


def _estimate_cpu_fma_units_per_core() -> int:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return 2
    return 2


def estimate_cpu_peak_tflops_by_dtype(
    *,
    core_count: int | None = None,
    freq_ghz: float | None = None,
) -> dict[str, float]:
    cores = core_count or os.cpu_count() or 1
    freq = freq_ghz or _estimate_cpu_default_freq_ghz()
    vector_bits = _estimate_cpu_vector_bits()
    fma_units = _estimate_cpu_fma_units_per_core()

    fp32_lanes = max(vector_bits // 32, 1)
    fp16_lanes = max(vector_bits // 16, 1)

    fp32_flops_per_cycle = fp32_lanes * 2 * fma_units
    fp16_flops_per_cycle = fp16_lanes * 2 * fma_units

    return {
        "fp32": cores * freq * fp32_flops_per_cycle / 1e3,
        "fp16": cores * freq * fp16_flops_per_cycle / 1e3,
        # Keep bf16 conservative by tying it to fp32 unless the user overrides it.
        "bf16": cores * freq * fp32_flops_per_cycle / 1e3,
    }


def resolve_peak_tflops_by_dtype(
    *,
    device_type: str,
    overrides: dict[str, float] | None = None,
    estimate_cpu_peak: bool = False,
    cpu_freq_ghz: float | None = None,
    cpu_core_count: int | None = None,
) -> tuple[dict[str, float], str]:
    peaks = dict(overrides or {})
    if device_type == "cpu":
        if peaks:
            return peaks, "manual_override"
        if not estimate_cpu_peak:
            return {}, "peak_na(cpu_default)"
        estimated = estimate_cpu_peak_tflops_by_dtype(
            core_count=cpu_core_count,
            freq_ghz=cpu_freq_ghz,
        )
        estimated.update(peaks)
        freq = cpu_freq_ghz or _estimate_cpu_default_freq_ghz()
        cores = cpu_core_count or os.cpu_count() or 1
        source = (
            f"cpu_estimator(cores={cores},freq_ghz={freq},vector_bits={_estimate_cpu_vector_bits()},"
            f"fma_units={_estimate_cpu_fma_units_per_core()})"
        )
        return estimated, source
    defaults, source = default_gpu_peak_tflops_by_dtype()
    has_device_defaults = bool(defaults)
    defaults.update(peaks)
    if peaks and has_device_defaults:
        return defaults, f"{source}+manual_override"
    if peaks:
        return defaults, "manual_override"
    return defaults, source


def estimate_attention_flops(case: BenchmarkCase) -> float:
    head_groups = case.h_q
    dense_factor = 0.5 if case.causal else 1.0
    if case.mask == "sliding_window" and case.window_size > 0:
        effective_sk = min(case.s_k, 2 * case.window_size + 1)
    else:
        effective_sk = case.s_k
    qk_flops = 2.0 * case.batch_size * head_groups * case.s_q * effective_sk * case.head_dim
    pv_flops = 2.0 * case.batch_size * head_groups * case.s_q * effective_sk * case.head_dim
    softmax_flops = 5.0 * case.batch_size * head_groups * case.s_q * effective_sk
    total = (qk_flops + pv_flops + softmax_flops) * dense_factor
    return total


def estimate_attention_hbm_bytes(case: BenchmarkCase, dtype_size_bytes: int, materializes_scores: bool) -> float:
    q_bytes = case.batch_size * case.h_q * case.s_q * case.head_dim * dtype_size_bytes
    k_bytes = case.batch_size * case.h_kv * case.s_k * case.head_dim * dtype_size_bytes
    v_bytes = case.batch_size * case.h_kv * case.s_k * case.head_dim * dtype_size_bytes
    o_bytes = case.batch_size * case.h_q * case.s_q * case.head_dim * dtype_size_bytes
    score_bytes = case.batch_size * case.h_q * case.s_q * case.s_k * dtype_size_bytes
    total = q_bytes + k_bytes + v_bytes + o_bytes
    if materializes_scores:
        total += 3 * score_bytes
    return float(total)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def throughput_tflops(flops: float, runtime_ms: float) -> float:
    if runtime_ms <= 0:
        return 0.0
    return flops / (runtime_ms * 1e-3) / 1e12


def estimate_math_mfu(tflops: float, dtype: str, peak_tflops_by_dtype: dict[str, float] | None = None) -> float:
    peaks = peak_tflops_by_dtype or {}
    peak = peaks.get(dtype, 0.0)
    if peak <= 0:
        return 0.0
    return tflops / peak


def format_mfu_value(tflops: float, dtype: str, peak_tflops_by_dtype: dict[str, float] | None = None) -> float | str:
    peaks = peak_tflops_by_dtype or {}
    peak = peaks.get(dtype, 0.0)
    if peak <= 0:
        return "na"
    return estimate_math_mfu(tflops, dtype, peaks)
