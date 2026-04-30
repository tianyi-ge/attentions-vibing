# Scoreboard

This document tracks milestone attention kernels against the same PyTorch SDPA
baseline on the same GPU, and keeps short tuning notes for each version.

## Rules

- Baseline: `torch_sdpa_ref`
- Device: `RTX 5090`
- Mode: `fwd`
- Current scope: dense forward attention only
- Main metrics:
  - `runtime_ms`: average latency per forward call
  - `speedup_vs_torch`: `torch_runtime / operator_runtime`

## Benchmark Cases

| Case | B | H | S_q | S_k | D | Dtype |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `train_fwd_s128_fp16` | 4 | 32 | 128 | 128 | 128 | fp16 |
| `train_fwd_s512_fp16` | 4 | 32 | 512 | 512 | 128 | fp16 |
| `train_fwd_s2k_fp16` | 4 | 32 | 2048 | 2048 | 128 | fp16 |
| `train_fwd_s8k_bf16` | 2 | 32 | 8192 | 8192 | 128 | bf16 |

## Runtime Scoreboard

| Operator | S=128 fp16 | S=512 fp16 | S=2048 fp16 | S=8192 bf16 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `torch_sdpa_ref` | `0.014 ms` | `0.117 ms` | `1.364 ms` | `10.302 ms` | Baseline |
| `online_softmax_fwd_v1` | `0.025 ms (0.56x)` | `0.176 ms (0.66x)` | `2.332 ms (0.58x)` | `18.001 ms (0.57x)` | Archived Triton v1 baseline |

## Correctness

| Operator | S=128 | S=512 | S=2048 | S=8192 |
| --- | --- | --- | --- | --- |
| `online_softmax_fwd_v1` | pass | pass | pass | pass |

## Version Notes

### `online_softmax_fwd_v1`

- Status: archived baseline
- Summary:
  - First Triton online-softmax forward kernel
  - Correctness passed on the forward-only dense cases above
  - Stable but slower than PyTorch fused flash backend by about `1.5x-1.8x`
- Key observations:
  - PyTorch baseline routes to a fused flash kernel
  - Main gap is kernel quality, not benchmark unfairness

### `online_softmax_fwd_v2`

- Status: active optimization branch
- Goal:
  - Reduce per-block resource usage
  - Raise occupancy
  - Close the performance gap vs `torch_sdpa_ref`

#### Block Sweep Notes

Workload used for the first sweep:

- `train_fwd_s512_fp16`
- `train_fwd_s2k_fp16`

Results:

| Block `(BLOCK_QM, BLOCK_KN)` | S=512 fp16 | S=2048 fp16 | Notes |
| --- | ---: | ---: | --- |
| `64 x 64` | `0.178 ms` | `2.334 ms` | Initial heavy configuration |
| `32 x 64` | `0.175 ms` | `2.358 ms` | Slight change, still heavy overall |
| `64 x 32` | `0.191 ms` | `2.530 ms` | Worst of the four |
| `32 x 32` | `0.154 ms` | `2.143 ms` | Best current configuration |

#### NCU Notes

For the original heavier configuration:

- `Achieved Occupancy`: `8.32%`
- `Registers / Thread`: `255`
- `Dynamic Shared Memory / Block`: `99.33 KB`

For the lighter `32 x 32` configuration:

- `Achieved Occupancy`: `16%`
- `Registers / Thread`: `128`
- `Dynamic Shared Memory / Block`: `45.57 KB`

Interpretation:

- The kernel is resource-limited before it is DRAM-bandwidth-limited
- Shrinking the block reduced both register pressure and shared-memory usage
- `BLOCK_QM` is more sensitive than `BLOCK_KN` in this kernel
- `32 x 32` is the current best default starting point for `v2`

#### Next Questions

- Can `32 x 32` be improved further without losing too much reuse?
- Is `P @ V` in the current fp32-heavy path the next major bottleneck?
- How much more can occupancy rise before performance stops improving?
