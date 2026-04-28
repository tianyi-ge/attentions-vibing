from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import uuid
import time

import torch

from attentions.bench.metrics import (
    arithmetic_intensity,
    estimate_attention_flops,
    estimate_attention_hbm_bytes,
    format_mfu_value,
    resolve_peak_tflops_by_dtype,
    throughput_tflops,
)
from attentions.config import BenchmarkCase
from attentions.registry import RuntimeOperator


TORCH_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class BenchmarkTensors:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    grad_out: torch.Tensor


@dataclass
class BenchmarkResult:
    run_id: str
    timestamp: str
    device_name: str
    operator_id: str
    case_id: str
    backend: str
    mode: str
    dtype: str
    is_correct: int
    max_abs_err: float
    max_rel_err: float
    runtime_ms: float
    throughput_tflops: float
    estimated_flops: float
    estimated_hbm_bytes: float
    arithmetic_intensity: float
    math_mfu: float | str
    tensorcore_mfu: float | str
    max_memory_mb: float
    occupancy_estimate: str
    bound_guess: str
    main_bottleneck: str
    next_optimization: str
    notes: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def make_tensors(case: BenchmarkCase, device: torch.device) -> BenchmarkTensors:
    dtype = TORCH_DTYPES[case.dtype]
    q = torch.randn(
        case.batch_size,
        case.h_q,
        case.s_q,
        case.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=case.mode in {"bwd", "fwd_bwd"},
    )
    k = torch.randn(
        case.batch_size,
        case.h_kv,
        case.s_k,
        case.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=case.mode in {"bwd", "fwd_bwd"},
    )
    v = torch.randn(
        case.batch_size,
        case.h_kv,
        case.s_k,
        case.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=case.mode in {"bwd", "fwd_bwd"},
    )
    if case.h_q != case.h_kv:
        repeat_factor = case.h_q // case.h_kv
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
    grad_out = torch.randn(
        case.batch_size,
        case.h_q,
        case.s_q,
        case.head_dim,
        device=device,
        dtype=dtype,
    )
    return BenchmarkTensors(q=q, k=k, v=v, grad_out=grad_out)


def _run_once(operator: RuntimeOperator, tensors: BenchmarkTensors, case: BenchmarkCase) -> torch.Tensor:
    kwargs = {
        "causal": case.causal,
        "window_size": case.window_size,
    }
    if operator.config.operator_id == "torch_sdpa_ref":
        kwargs.pop("window_size", None)
    return operator.fn(tensors.q, tensors.k, tensors.v, **kwargs)


def run_benchmark_case(
    operator: RuntimeOperator,
    case: BenchmarkCase,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    peak_tflops_by_dtype: dict[str, float] | None = None,
    estimate_cpu_peak: bool = False,
    cpu_freq_ghz: float | None = None,
    cpu_core_count: int | None = None,
) -> BenchmarkResult:
    can_run, reason = operator.supports_case(case)
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = uuid.uuid4().hex[:12]
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)

    if not can_run:
        return BenchmarkResult(
            run_id=run_id,
            timestamp=timestamp,
            device_name=device_name,
            operator_id=operator.config.operator_id,
            case_id=case.case_id,
            backend=operator.config.backend,
            mode=case.mode,
            dtype=case.dtype,
            is_correct=0,
            max_abs_err=0.0,
            max_rel_err=0.0,
            runtime_ms=0.0,
            throughput_tflops=0.0,
            estimated_flops=0.0,
            estimated_hbm_bytes=0.0,
            arithmetic_intensity=0.0,
            math_mfu="na",
            tensorcore_mfu="na",
            max_memory_mb=0.0,
            occupancy_estimate="",
            bound_guess="unsupported",
            main_bottleneck="",
            next_optimization="",
            notes=reason,
        )

    tensors = make_tensors(case, device=device)
    with torch.no_grad():
        baseline = _run_reference_for_case(tensors, case)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for _ in range(warmup):
        output = _run_once(operator, tensors, case)
        if case.mode in {"bwd", "fwd_bwd"}:
            output.backward(tensors.grad_out, retain_graph=True)
            tensors.q.grad = None
            tensors.k.grad = None
            tensors.v.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    last_output = None
    for _ in range(repeats):
        last_output = _run_once(operator, tensors, case)
        if case.mode in {"bwd", "fwd_bwd"}:
            last_output.backward(tensors.grad_out, retain_graph=True)
            tensors.q.grad = None
            tensors.k.grad = None
            tensors.v.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0 / max(repeats, 1)

    max_abs_err, max_rel_err = compare_outputs(last_output, baseline)
    estimated_flops = estimate_attention_flops(case)
    materializes_scores = operator.config.operator_id == "naive_sdpa"
    dtype_size = torch.tensor([], dtype=TORCH_DTYPES[case.dtype]).element_size()
    estimated_bytes = estimate_attention_hbm_bytes(case, dtype_size, materializes_scores=materializes_scores)
    achieved_tflops = throughput_tflops(estimated_flops, runtime_ms)
    resolved_peaks, peak_source = resolve_peak_tflops_by_dtype(
        device_type=device.type,
        overrides=peak_tflops_by_dtype,
        estimate_cpu_peak=estimate_cpu_peak,
        cpu_freq_ghz=cpu_freq_ghz,
        cpu_core_count=cpu_core_count,
    )
    math_mfu = format_mfu_value(achieved_tflops, case.dtype, resolved_peaks)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    bound_guess = "memory" if materializes_scores or case.mode == "decode" else "mixed"
    return BenchmarkResult(
        run_id=run_id,
        timestamp=timestamp,
        device_name=device_name,
        operator_id=operator.config.operator_id,
        case_id=case.case_id,
        backend=operator.config.backend,
        mode=case.mode,
        dtype=case.dtype,
        is_correct=int(torch.isfinite(last_output).all().item() and max_abs_err < 5e-2),
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        runtime_ms=runtime_ms,
        throughput_tflops=achieved_tflops,
        estimated_flops=estimated_flops,
        estimated_hbm_bytes=estimated_bytes,
        arithmetic_intensity=arithmetic_intensity(estimated_flops, estimated_bytes),
        math_mfu=math_mfu,
        tensorcore_mfu=math_mfu,
        max_memory_mb=peak_memory_mb,
        occupancy_estimate="",
        bound_guess=bound_guess,
        main_bottleneck="score_materialization" if materializes_scores else "",
        next_optimization="move_to_triton" if operator.config.operator_id == "naive_sdpa" else "",
        notes=_merge_notes(operator.availability_note, peak_source),
    )


def _run_reference_for_case(tensors: BenchmarkTensors, case: BenchmarkCase) -> torch.Tensor:
    from attentions.operators.reference import naive_sdpa_reference, torch_sdpa_reference

    if case.mask == "sliding_window":
        return naive_sdpa_reference(
            tensors.q.detach(),
            tensors.k.detach(),
            tensors.v.detach(),
            causal=case.causal,
            window_size=case.window_size,
        )
    return torch_sdpa_reference(
        tensors.q.detach(),
        tensors.k.detach(),
        tensors.v.detach(),
        causal=case.causal,
    )


def compare_outputs(output: torch.Tensor | None, reference: torch.Tensor | None) -> tuple[float, float]:
    if output is None or reference is None:
        return 0.0, 0.0
    diff = (output.detach().float() - reference.detach().float()).abs()
    ref_abs = reference.detach().float().abs().clamp_min(1e-6)
    rel = diff / ref_abs
    return float(diff.max().item()), float(rel.max().item())


def _merge_notes(*parts: str) -> str:
    return " | ".join(part for part in parts if part)
