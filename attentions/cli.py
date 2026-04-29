from __future__ import annotations

import argparse
from pathlib import Path

import torch

from attentions.bench.io import write_results_csv
from attentions.bench.runner import run_benchmark_case
from attentions.config import BENCHMARKS_DIR, load_benchmark_cases, load_operator_configs
from attentions.operators import build_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attention benchmark cases.")
    parser.add_argument("--operators", nargs="*", default=None, help="Operator ids to run.")
    parser.add_argument("--cases", nargs="*", default=None, help="Case ids to run.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--estimate-cpu-peak", action="store_true", help="Opt in to the lightweight CPU peak-TFLOPS estimator for MFU.")
    parser.add_argument("--cpu-freq-ghz", type=float, default=None, help="Override CPU frequency used by the peak-TFLOPS estimator.")
    parser.add_argument("--cpu-core-count", type=int, default=None, help="Override CPU core count used by the peak-TFLOPS estimator.")
    parser.add_argument("--peak-tflops-fp16", type=float, default=None, help="Manual peak TFLOPS override for fp16 MFU.")
    parser.add_argument("--peak-tflops-bf16", type=float, default=None, help="Manual peak TFLOPS override for bf16 MFU.")
    parser.add_argument("--peak-tflops-fp32", type=float, default=None, help="Manual peak TFLOPS override for fp32 MFU.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_DIR / "results" / "latest.csv",
        help="Where to write the normalized results CSV.",
    )
    parser.add_argument(
        "--implemented-only",
        action="store_true",
        help="Only run operators that already have a callable implementation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    peak_overrides = {
        dtype: value
        for dtype, value in {
            "fp16": args.peak_tflops_fp16,
            "bf16": args.peak_tflops_bf16,
            "fp32": args.peak_tflops_fp32,
        }.items()
        if value is not None
    }
    operator_configs = load_operator_configs()
    case_configs = load_benchmark_cases()
    case_configs = [case for case in case_configs if case.enabled]
    if args.operators:
        requested = set(args.operators)
        operator_configs = [config for config in operator_configs if config.operator_id in requested]
    if args.cases:
        requested_cases = set(args.cases)
        case_configs = [case for case in case_configs if case.case_id in requested_cases]

    registry = build_registry(operator_configs)
    rows = []
    operators = registry.all()
    if args.implemented_only:
        operators = [operator for operator in operators if operator.is_available]
    for operator in operators:
        for case in case_configs:
            result = run_benchmark_case(
                operator,
                case,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                peak_tflops_by_dtype=peak_overrides,
                estimate_cpu_peak=args.estimate_cpu_peak,
                cpu_freq_ghz=args.cpu_freq_ghz,
                cpu_core_count=args.cpu_core_count,
            )
            rows.append(result.to_row())
            print(
                f"{result.operator_id:>24} | {result.case_id:>18} | "
                f"correct={result.is_correct} | runtime_ms={result.runtime_ms:.3f}"
            )
    write_results_csv(args.output, rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
