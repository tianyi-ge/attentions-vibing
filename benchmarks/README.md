# Benchmark Design

This directory holds the experiment registry and result tables for comparing
attention kernels fairly.

## Files

- [operators.csv](/Users/getianyi/Documents/attentions-vibing/benchmarks/operators.csv)
  One row per implementation target.
- [cases.csv](/Users/getianyi/Documents/attentions-vibing/benchmarks/cases.csv)
  One row per benchmark case.
- [results-template.csv](/Users/getianyi/Documents/attentions-vibing/benchmarks/results-template.csv)
  One row per `(operator, case)` measurement.

## Workflow

1. Add a kernel to `operators.csv`.
2. Add or trim workloads in `cases.csv`.
3. Run your benchmark harness for every enabled `(operator, case)` pair.
4. Write normalized results into `results-template.csv` or a generated copy.

## Current scaffold

The code scaffold lives in `attentions/` and currently provides:

- CSV-driven operator and case loading
- runtime operator registry
- PyTorch SDPA reference path
- eager naive SDPA baseline
- normalized result writing
- one CLI for filtered benchmark runs

As you add Triton or CUDA kernels, the intended flow is:

1. Keep the metadata row in `operators.csv`.
2. Add the implementation function under `attentions/operators/`.
3. Register it in `attentions/operators/registry_builder.py`.
4. Reuse the same cases and result schema.

## Registry principles

- Keep operator metadata in `operators.csv`, not hard-coded in benchmark loops.
- Keep workloads in `cases.csv`, not inside kernel-specific files.
- Store derived metrics like TFLOP/s, arithmetic intensity, and MFU in the
  results table so you can compare kernels with one pivot.

## Minimum correctness checks

Each measurement row should also carry:

- `max_abs_err`
- `max_rel_err`
- `is_correct`

Incorrect rows should never be compared as performance wins.
