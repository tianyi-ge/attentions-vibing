from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from attentions.operators.reference import torch_sdpa_reference


DEFAULT_OPERATOR = "online_softmax_fwd_v3"
SUPPORTED_OPERATORS = ("online_softmax_fwd_v1", "online_softmax_fwd_v2", "online_softmax_fwd_v3")
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _fp8_dtype(name: str) -> torch.dtype:
    if name in {"fp8", "fp8_e4m3", "fp8_e4m3fn"}:
        return torch.float8_e4m3fn
    if name in {"fp8_e5m2"}:
        return torch.float8_e5m2
    raise ValueError(f"Unsupported dtype: {name}")


@dataclass
class AutotuneResult:
    operator: str
    dtype: str
    batch_size: int
    heads: int
    seq_q: int
    seq_k: int
    head_dim: int
    block_qm: int
    block_kn: int
    num_warps: int
    num_stages: int
    is_correct: int
    max_abs_err: float
    max_rel_err: float
    runtime_ms: float
    max_memory_mb: float
    status: str


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep online-softmax attention launch parameters.")
    parser.add_argument(
        "--operator",
        choices=SUPPORTED_OPERATORS,
        default=DEFAULT_OPERATOR,
        help=f"Operator implementation to tune. Defaults to latest: {DEFAULT_OPERATOR}.",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp8", "fp8_e4m3", "fp8_e4m3fn", "fp8_e5m2"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-q", type=int, default=512)
    parser.add_argument("--seq-k", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-qm", default="16,32", help="Comma-separated BLOCK_QM candidates.")
    parser.add_argument("--block-kn", default="16,32", help="Comma-separated BLOCK_KN candidates.")
    parser.add_argument("--num-warps", default="4,8", help="Comma-separated num_warps candidates.")
    parser.add_argument("--num-stages", default="2,3", help="Comma-separated num_stages candidates.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "autotune_online_softmax.csv",
    )
    return parser.parse_args()


def make_tensors(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape_q = (args.batch_size, args.heads, args.seq_q, args.head_dim)
    shape_kv = (args.batch_size, args.heads, args.seq_k, args.head_dim)
    if args.dtype.startswith("fp8"):
        fp8_dtype = _fp8_dtype(args.dtype)
        q_ref = torch.randn(shape_q, device="cuda", dtype=torch.float16)
        k_ref = torch.randn(shape_kv, device="cuda", dtype=torch.float16)
        v_ref = torch.randn(shape_kv, device="cuda", dtype=torch.float16)
        q = q_ref.to(fp8_dtype)
        k = k_ref.to(fp8_dtype)
        v = v_ref.to(fp8_dtype)
        baseline = torch_sdpa_reference(q.float().half(), k.float().half(), v.float().half(), causal=False)
        return q, k, v, baseline

    dtype = DTYPES[args.dtype]
    q = torch.randn(shape_q, device="cuda", dtype=dtype)
    k = torch.randn(shape_kv, device="cuda", dtype=dtype)
    v = torch.randn(shape_kv, device="cuda", dtype=dtype)
    baseline = torch_sdpa_reference(q, k, v, causal=False)
    return q, k, v, baseline


def compare_outputs(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = (output.detach().float() - reference.detach().float()).abs()
    ref_abs = reference.detach().float().abs().clamp_min(1e-6)
    rel = diff / ref_abs
    return float(diff.max().item()), float(rel.max().item())


def resolve_operator(operator: str):
    if operator == "online_softmax_fwd_v1":
        from attentions.operators.online_softmax_fwd_v1 import online_softmax_attention_forward_v1

        return online_softmax_attention_forward_v1
    if operator == "online_softmax_fwd_v2":
        from attentions.operators.online_softmax_fwd_v2 import online_softmax_attention_forward_v2

        return online_softmax_attention_forward_v2
    if operator == "online_softmax_fwd_v3":
        from attentions.operators.online_softmax_fwd_v3 import online_softmax_attention_forward_v3

        return online_softmax_attention_forward_v3
    raise ValueError(f"Unsupported operator: {operator}")


def run_candidate(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    baseline: torch.Tensor,
    *,
    block_qm: int,
    block_kn: int,
    num_warps: int,
    num_stages: int,
) -> AutotuneResult:
    output_dtype = torch.float16 if args.dtype.startswith("fp8") else None
    atol = args.atol if args.atol is not None else (2e-1 if args.dtype.startswith("fp8") else 5e-2)
    rtol = args.rtol if args.rtol is not None else (2e-1 if args.dtype.startswith("fp8") else 1e-1)
    try:
        operator_fn = resolve_operator(args.operator)

        for _ in range(args.warmup):
            out = operator_fn(
                q,
                k,
                v,
                causal=False,
                block_qm=block_qm,
                block_kn=block_kn,
                output_dtype=output_dtype,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            del out
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        out = None
        for _ in range(args.repeats):
            out = operator_fn(
                q,
                k,
                v,
                causal=False,
                block_qm=block_qm,
                block_kn=block_kn,
                output_dtype=output_dtype,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - start) * 1000.0 / max(args.repeats, 1)
        assert out is not None
        max_abs_err, max_rel_err = compare_outputs(out, baseline)
        is_correct = int(max_abs_err <= atol or max_rel_err <= rtol)
        status = "ok"
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception as exc:
        runtime_ms = 0.0
        max_abs_err = 0.0
        max_rel_err = 0.0
        is_correct = 0
        max_memory_mb = 0.0
        status = f"error: {type(exc).__name__}: {exc}"

    return AutotuneResult(
        operator=args.operator,
        dtype=args.dtype,
        batch_size=args.batch_size,
        heads=args.heads,
        seq_q=args.seq_q,
        seq_k=args.seq_k,
        head_dim=args.head_dim,
        block_qm=block_qm,
        block_kn=block_kn,
        num_warps=num_warps,
        num_stages=num_stages,
        is_correct=is_correct,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        runtime_ms=runtime_ms,
        max_memory_mb=max_memory_mb,
        status=status,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to autotune the Triton kernel.")

    q, k, v, baseline = make_tensors(args)
    rows = []
    for block_qm in parse_int_list(args.block_qm):
        for block_kn in parse_int_list(args.block_kn):
            for num_warps in parse_int_list(args.num_warps):
                for num_stages in parse_int_list(args.num_stages):
                    result = run_candidate(
                        args,
                        q,
                        k,
                        v,
                        baseline,
                        block_qm=block_qm,
                        block_kn=block_kn,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    rows.append(result)
                    print(
                        f"BM={block_qm:>2} BN={block_kn:>2} warps={num_warps} stages={num_stages} | "
                        f"correct={result.is_correct} | runtime_ms={result.runtime_ms:.3f} | {result.status}"
                    )

    rows.sort(key=lambda row: (row.is_correct == 0, row.runtime_ms if row.runtime_ms > 0 else float("inf")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    best = rows[0]
    print(
        "\nBest candidate: "
        f"BM={best.block_qm} BN={best.block_kn} warps={best.num_warps} stages={best.num_stages} "
        f"runtime_ms={best.runtime_ms:.3f} correct={best.is_correct}"
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
