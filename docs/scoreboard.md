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
| `torch_sdpa_ref` | `0.012 ms` | `0.115 ms` | `1.376 ms` | `10.399 ms` | Baseline |
| `online_softmax_fwd_v1` | `0.019 ms (0.66x)` | `0.173 ms (0.66x)` | `2.366 ms (0.58x)` | `18.173 ms (0.57x)` | Online-softmax baseline |
| `online_softmax_fwd_v2` | `0.016 ms (0.80x)` | `0.111 ms (1.03x)` | `1.678 ms (0.82x)` | `13.157 ms (0.79x)` | Autotuned + low-precision `P @ V` |

## Correctness

| Operator | S=128 | S=512 | S=2048 | S=8192 |
| --- | --- | --- | --- | --- |
| `online_softmax_fwd_v1` | pass | pass | pass | pass |
| `online_softmax_fwd_v2` | pass | pass | pass | pass |

## Version Notes

### `online_softmax_fwd_v1`

- Goal:
  - Build the first exact dense attention Triton baseline without materializing `S_q x S_k`
- Main optimizations:
  - Online softmax recurrence keeps running row max, denominator, and numerator in fp32
  - Streams over K/V blocks instead of storing the full score matrix
- Notes:
  - Correctness passed on the forward-only dense cases above
  - Stable but slower than PyTorch fused flash backend
- Key observations:
  - PyTorch baseline routes to a fused flash kernel
  - Main gap is kernel quality, not benchmark unfairness

### `online_softmax_fwd_v2`

- Goal:
  - Reduce per-block resource usage
  - Raise occupancy
  - Close the performance gap vs `torch_sdpa_ref`
- Main optimizations:
  - Autotuned `BLOCK_QM`, `BLOCK_KN`, `num_warps`, and `num_stages`
  - Set the current default to `32 x 32`, `4` warps, `2` stages
  - Cast `P` to `V.dtype` before `P @ V` so the matmul uses low-precision tensor-core-friendly operands

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

Autotune sweep with `num_warps in {4, 8}` and `num_stages in {2, 3}`:

| Config `(BLOCK_QM, BLOCK_KN, warps, stages)` | S=512 fp16 | S=2048 fp16 | S=8192 fp16 | Notes |
| --- | ---: | ---: | ---: | --- |
| `32 x 32, 4, 2` | `0.143 ms` | `2.117 ms` | `35.149 ms` | Best across tested sequence lengths |
| `32 x 32, 4, 3` | `0.148 ms` | `2.191 ms` | `36.215 ms` | Slightly slower; extra stage does not pay off |
| `16 x 32, 4, 2` | `0.173 ms` | `2.660 ms` | `42.901 ms` | Smaller `BLOCK_QM` loses too much work per CTA |
| `32 x 16, 4, 2` | `0.182 ms` | `2.742 ms` | `45.203 ms` | Smaller `BLOCK_KN` increases loop overhead |

#### Dot Dtype Optimization

The initial v2 path kept `P @ V` in a fp32-heavy form:

```python
tl.dot(new_items, v.to(tl.float32))
```

The optimized v2 path keeps the online softmax state in fp32, but casts
`new_items` to `v.dtype` before the `P @ V` matmul:

```python
p = new_items.to(v.dtype)
tl.dot(p, v)
```

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| S=512 fp16 | `0.143 ms` | `0.110 ms` | `1.30x` |
| S=2048 fp16 | `2.117 ms` | `1.742 ms` | `1.22x` |
| S=8192 fp16 | `35.149 ms` | `27.248 ms` | `1.29x` |
| S=512 bf16 | n/a | `0.108 ms` | n/a |

#### NCU Notes

S=8192 fp16 NCU comparison between v1 and v2:

Command:

```bash
bash scripts/profile_ncu.sh online_softmax_fwd_v2 8192 8192 128 fp16 4 32 5 20 online_softmax_attn_fwd_kernel_v2
```

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Duration | `51.31 ms` | `32.45 ms` |
| Compute throughput | `36.64%` | `51.61%` |
| Memory throughput | `24.23%` | `76.29%` |
| DRAM throughput | `1.23%` | `1.90%` |
| L2 hit rate | `98.38%` | `99.20%` |
| Tensor pipeline | `36.9%` | `39.0%` |
| Issue slots busy | `16.60%` | `26.46%` |
| Achieved occupancy | `8.33%` | `24.87%` |
| Active warps / SM | `4.00` | `11.94` |
| Active warps / scheduler | `1.00` | `2.99` |
| Eligible warps / scheduler | `0.17` | `0.35` |
| No eligible | `83.40%` | `73.52%` |
| Registers / thread | `255` | `122` |
| Dynamic shared memory / block | `99.33 KB` | `26.62 KB` |
| Block limit by registers | `2` | `4` |
| Block limit by shared memory | `1` | `3` |
| Theoretical occupancy | `8.33%` | `25.00%` |
| Shared store bank-conflict wavefronts | n/a | `14.64%` |
| Executed instructions | `11.60B` | `11.67B` |
| Non-fused FP32 instructions | `2.26B` | `3.17B` |
| Grid size | `16,384` | `32,768` |
| Waves / SM | `96.38` | `64.25` |

Interpretation:

- v2 is faster mainly because resource usage dropped: shared memory per block fell from `99.33 KB` to `26.62 KB`, raising the shared-memory block limit from `1` to `3`
- Higher occupancy gives the schedulers more active and eligible warps, improving issue slots busy from `16.60%` to `26.46%`
- DRAM throughput remains tiny in both versions, so the kernel is not HBM-bandwidth bound
- v2 drives the L2/on-chip memory path much harder; Memory Throughput rises from `24.23%` to `76.29%`
- The `P @ V` dtype optimization improves runtime even though total executed instructions are similar at S=8192; the important change is better instruction mix, resource usage, and scheduling
- v2 still has low eligible warps (`0.35` per scheduler), so latency hiding and the online-softmax dependency chain remain major limits

#### Next Questions

- Can `32 x 32` be improved further without losing too much reuse?
- Can a different work partition, such as split-K for long sequences, expose more parallelism without excessive partial-state reduction overhead?
- Can the online-softmax fp32 dependency chain be shortened or better overlapped?
- Can shared-memory bank conflicts be reduced with a different tile/layout?
- How much more can occupancy rise before performance stops improving?
