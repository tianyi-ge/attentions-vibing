# attentions-vibing

Practice-first repository for building, analyzing, and benchmarking the
attention kernel family on RTX 5090.

## What is in this repo

- [docs/attention-roadmap.md](/Users/getianyi/Documents/attentions-vibing/docs/attention-roadmap.md)
  End-to-end learning and implementation plan, including milestone history,
  prioritized operators, optimization stages, MFU and arithmetic-intensity
  analysis, and a suggested Triton/CUDA split.
- [benchmarks/README.md](/Users/getianyi/Documents/attentions-vibing/benchmarks/README.md)
  Benchmark methodology, measurement dimensions, and rules for fair
  comparisons.
- [benchmarks/operators.csv](/Users/getianyi/Documents/attentions-vibing/benchmarks/operators.csv)
  Registry of operators to implement and compare.
- [benchmarks/cases.csv](/Users/getianyi/Documents/attentions-vibing/benchmarks/cases.csv)
  Benchmark case matrix. Add or remove rows to scale the experiment surface.
- [benchmarks/results-template.csv](/Users/getianyi/Documents/attentions-vibing/benchmarks/results-template.csv)
  A normalized results table for performance, correctness, and roofline-style
  analysis.

## Suggested workflow

1. Start with the operators in `docs/attention-roadmap.md`.
2. Implement the first 5 to 8 kernels in Triton.
3. Pick 1 to 2 deep kernels for handwritten CUDA.
4. Register every implementation in `benchmarks/operators.csv`.
5. Run the shared case matrix in `benchmarks/cases.csv`.
6. Fill `benchmarks/results-template.csv` and compare throughput, MFU,
   arithmetic intensity, and memory behavior.

## Initial priorities

- Triton:
  `naive_sdpa`, `online_softmax_fwd`, `flash_v1_fwd`, `flash_v2_fwd_bwd`,
  `grouped_query_attention`, `sliding_window_attention`, `paged_attention`
- CUDA:
  `flash_v2_full`, `paged_attention_decode`

## Scaffold Status

The repository now includes a runnable Python benchmark scaffold:

- `attentions/`
  Core package with CSV config loading, operator registry, reference kernels,
  metrics, and benchmark runner.
- `scripts/run_benchmarks.py`
  Thin entrypoint for local runs.
- `pyproject.toml`
  Package metadata and CLI registration for `attn-bench`.

Current runnable operators:

- `torch_sdpa_ref`
- `naive_sdpa`

The remaining operator ids are registered as placeholders so you can add a new
kernel without changing the surrounding harness.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

## Running Benchmarks

The benchmark runner supports filtering by operator id, case id, device, and
output path.

Show the current options:

```bash
attn-bench --help
```

Run a small CPU smoke benchmark on the implemented operators:

```bash
python scripts/run_benchmarks.py \
  --operators torch_sdpa_ref naive_sdpa \
  --cases cpu_smoke_fp32 \
  --implemented-only \
  --include-disabled \
  --device cpu \
  --warmup 1 \
  --repeats 3
```

Run one enabled training case with the installed CLI:

```bash
attn-bench \
  --operators torch_sdpa_ref naive_sdpa \
  --cases train_s128_fp16 \
  --implemented-only
```

Run the first Triton forward-only kernel against the forward-only benchmark case:

```bash
attn-bench \
  --operators online_softmax_fwd torch_sdpa_ref \
  --cases train_fwd_s128_fp16 \
  --implemented-only \
  --device cuda
```

Run a forward-only scaling sweep that is more useful for performance analysis:

```bash
attn-bench \
  --operators torch_sdpa_ref online_softmax_fwd \
  --cases train_fwd_s128_fp16 train_fwd_s512_fp16 train_fwd_s2k_fp16 train_fwd_s8k_bf16 \
  --implemented-only \
  --device cuda
```

Run all enabled implemented operators against all enabled cases:

```bash
attn-bench --implemented-only
```

Write results to a custom file:

```bash
attn-bench \
  --operators torch_sdpa_ref naive_sdpa \
  --cases train_s128_fp16 \
  --implemented-only \
  --output benchmarks/results/train_s128_fp16.csv
```

By default, results are written to
`benchmarks/results/latest.csv`.

Useful flags:

- `--operators ...`
  Only run the listed operator ids from `benchmarks/operators.csv`.
- `--cases ...`
  Only run the listed case ids from `benchmarks/cases.csv`.
- `--implemented-only`
  Skip placeholder operators that are registered but not implemented yet.
- `--include-disabled`
  Allow rows that are marked disabled in the CSV registries.
- `--device cpu|cuda`
  Override automatic device selection.
- `--warmup N`
  Number of warmup iterations before timing.
- `--repeats N`
  Number of timed iterations used to compute average runtime.
- `--output PATH`
  Output CSV path for the normalized benchmark results.
- `--cpu-freq-ghz F`
  Override the simple CPU peak-throughput estimator frequency.
- `--cpu-core-count N`
  Override the simple CPU peak-throughput estimator core count.
- `--estimate-cpu-peak`
  Opt in to the lightweight CPU peak estimator for MFU on CPU runs.
- `--peak-tflops-fp16/--peak-tflops-bf16/--peak-tflops-fp32`
  Manually override the theoretical peak used for MFU.

## Profiling

Use the profiler helper to compare PyTorch SDPA and the Triton kernel on the
same shape:

```bash
python scripts/profile_attention.py \
  --operator torch_sdpa_ref \
  --seq-q 512 \
  --seq-k 512 \
  --head-dim 128 \
  --dtype fp16
```

```bash
python scripts/profile_attention.py \
  --operator online_softmax_fwd \
  --seq-q 512 \
  --seq-k 512 \
  --head-dim 128 \
  --dtype fp16
```

By default, the script also writes:

- a profiler summary table to `benchmarks/profiles/*.txt`
- a Chrome trace to `benchmarks/profiles/*.json`

## Scoreboard

This section tracks milestone attention kernels against the same PyTorch SDPA
baseline on the same GPU.

Rules for this table:

- Baseline: `torch_sdpa_ref`
- Device: `RTX 5090`
- Mode: `fwd`
- Metrics:
  - `runtime_ms`: average latency per forward call
  - `speedup_vs_torch`: `torch_runtime / operator_runtime`
- Current scope: dense forward attention only

### Benchmark Cases

| Case | B | H | S_q | S_k | D | Dtype |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `train_fwd_s128_fp16` | 4 | 32 | 128 | 128 | 128 | fp16 |
| `train_fwd_s512_fp16` | 4 | 32 | 512 | 512 | 128 | fp16 |
| `train_fwd_s2k_fp16` | 4 | 32 | 2048 | 2048 | 128 | fp16 |
| `train_fwd_s8k_bf16` | 2 | 32 | 8192 | 8192 | 128 | bf16 |

### Runtime Scoreboard

| Operator | S=128 fp16 | S=512 fp16 | S=2048 fp16 | S=8192 bf16 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `torch_sdpa_ref` | `0.014 ms` | `0.117 ms` | `1.364 ms` | `10.302 ms` | Baseline |
| `online_softmax_fwd` | `0.025 ms (0.56x)` | `0.176 ms (0.66x)` | `2.332 ms (0.58x)` | `18.001 ms (0.57x)` | First Triton online-softmax kernel |

### Correctness

| Operator | S=128 | S=512 | S=2048 | S=8192 |
| --- | --- | --- | --- | --- |
| `online_softmax_fwd` | pass | pass | pass | pass |

## MFU Notes

For CPU runs, the default behavior is conservative:

- `math_mfu = na`
- `tensorcore_mfu = na`

This avoids pretending we know the CPU peak accurately during smoke tests.

If you explicitly pass `--estimate-cpu-peak`, MFU uses a lightweight estimator
based on:

- core count
- a default frequency heuristic, or `--cpu-freq-ghz`
- SIMD width heuristic from the local architecture
- a simple FMA-per-cycle assumption

This is intentionally approximate. It is useful for rough comparison, but it
is not a substitute for a carefully validated hardware peak model.

For non-CPU runs, MFU is also `na` unless you pass explicit peak TFLOPS overrides.
