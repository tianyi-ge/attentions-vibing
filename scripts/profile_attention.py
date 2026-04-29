from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from attentions.operators.online_softmax_fwd_v1 import online_softmax_attention_forward_v1
from attentions.operators.online_softmax_fwd_v2 import online_softmax_attention_forward_v2
from attentions.operators.reference import torch_sdpa_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile torch SDPA vs Triton online softmax attention.")
    parser.add_argument("--operator", choices=["torch_sdpa_ref", "online_softmax_fwd_v1", "online_softmax_fwd_v2"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-q", type=int, default=128)
    parser.add_argument("--seq-k", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sort-by", default="self_cuda_time_total")
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks" / "profiles",
        help="Directory where profiler artifacts will be written.",
    )
    return parser.parse_args()


def make_tensors(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    q = torch.randn(args.batch_size, args.heads, args.seq_q, args.head_dim, device="cuda", dtype=dtype)
    k = torch.randn(args.batch_size, args.heads, args.seq_k, args.head_dim, device="cuda", dtype=dtype)
    v = torch.randn(args.batch_size, args.heads, args.seq_k, args.head_dim, device="cuda", dtype=dtype)
    return q, k, v


def run_operator(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if name == "torch_sdpa_ref":
        return torch_sdpa_reference(q, k, v, causal=False)
    if name == "online_softmax_fwd_v1":
        return online_softmax_attention_forward_v1(q, k, v, causal=False)
    if name == "online_softmax_fwd_v2":
        return online_softmax_attention_forward_v2(q, k, v, causal=False)
    raise ValueError(f"Unsupported operator: {name}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this profiler script.")

    q, k, v = make_tensors(args)

    for _ in range(args.warmup):
        out = run_operator(args.operator, q, k, v)
        del out
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            out = run_operator(args.operator, q, k, v)
            del out
        torch.cuda.synchronize()

    summary = prof.key_averages().table(
        sort_by=args.sort_by,
        row_limit=args.row_limit,
    )
    print(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.operator}_b{args.batch_size}_h{args.heads}_"
        f"sq{args.seq_q}_sk{args.seq_k}_d{args.head_dim}_{args.dtype}"
    )
    table_path = args.output_dir / f"{stem}.txt"
    trace_path = args.output_dir / f"{stem}.json"

    table_path.write_text(summary, encoding="utf-8")
    prof.export_chrome_trace(str(trace_path))

    print(f"\nSaved profiler table to {table_path}")
    print(f"Saved chrome trace to {trace_path}")


if __name__ == "__main__":
    main()
