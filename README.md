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
