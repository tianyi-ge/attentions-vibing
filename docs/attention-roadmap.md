# Attention Kernel Roadmap

This document turns the attention-kernel space into a practical build order for
manual implementation, profiling, and optimization on RTX 5090.

## 1. Goal

You want to:

1. Understand the historical evolution of attention kernels.
2. Estimate arithmetic intensity and MFU from first principles.
3. Rebuild the important kernels yourself.
4. Replace them into PyTorch and compare performance fairly.
5. Learn when a kernel is compute-bound, memory-bound, or launch-bound.

The best way to do this is not to start from "the fastest possible kernel", but
from a sequence where each step teaches one new systems idea.

## 2. Milestones in the Attention Family

The list below is organized by systems ideas rather than paper prestige.

| Era | Milestone | Core idea | Why it matters |
| --- | --- | --- | --- |
| 2017 | Vanilla scaled dot-product attention | Materialize `S = QK^T`, softmax, then `PV` | Baseline math and baseline inefficiency |
| 2018 | Multi-head attention optimization | Head parallelism and batched GEMM layout tuning | Teaches tensor layout and launch strategy |
| 2019 | Sparse attention | Restrict score matrix connectivity | Reduces asymptotic memory/compute for long contexts |
| 2019 | Local/sliding-window attention | Windowed receptive field | Simple structured sparsity with predictable memory access |
| 2020 | Linear/IO-aware reformulations | Avoid or reduce score materialization | Helps reason about memory traffic vs FLOPs |
| 2021 | Online softmax fusion | Streaming max/sum accumulation | Foundation for FlashAttention |
| 2022 | FlashAttention v1 | IO-aware tiled exact attention | First major practical breakthrough for exact attention |
| 2023 | FlashAttention v2 | Better work partitioning and parallelism | Strong reference point for custom kernels |
| 2023 | MQA / GQA | Fewer KV heads than Q heads | Critical for inference efficiency in modern LLMs |
| 2023 | Paged attention | KV cache split into pages/blocks | Core systems primitive for long-context serving |
| 2023 | Sliding-window + causal hybrids | Exact local attention at long sequence lengths | Important for modern long-context models |
| 2024 | Flash decoding / split-K decode attention | Decode-stage specialization for tiny `Sq` and large cache | Essential for inference kernels |
| 2024+ | FlashAttention v3 style Hopper-era kernels | WGMMA/TMA/pipelining-centric design | Great reference for advanced CUDA craft, even if 5090 differs |

## 3. First-Principles Performance Model

### 3.1 Forward FLOPs of dense exact attention

For batch `B`, query heads `Hq`, KV heads `Hkv`, query length `Sq`, key length
`Sk`, head dimension `D`, and assuming `Hq == Hkv == H` for standard MHA:

- `QK^T`: about `2 * B * H * Sq * Sk * D`
- `softmax`: about `~ 5 * B * H * Sq * Sk`
  This is smaller than the GEMM terms when `D` is moderate.
- `PV`: about `2 * B * H * Sq * Sk * D`

Dense forward attention FLOPs are therefore roughly:

`F_fwd ~= 4 * B * H * Sq * Sk * D`

For causal attention, arithmetic work is about half the full matrix on average:

`F_fwd_causal ~= 2 * B * H * Sq * Sk * D`

### 3.2 Naive memory traffic intuition

The naive implementation often:

1. Reads `Q`, `K`, `V`
2. Writes scores `S`
3. Reads scores `S` back for softmax
4. Writes probabilities `P`
5. Reads `P` back for `PV`
6. Writes output `O`

This makes the operator strongly memory-inefficient because the `Sq x Sk` score
matrix is materialized at least once, sometimes twice.

### 3.3 Flash-style memory traffic intuition

Flash-style kernels avoid writing the full score/probability matrices to HBM.
They stream tiles of `K/V`, maintain online softmax statistics, and write only
the final output plus a small amount of saved state for backward.

That shifts the kernel upward on a roofline plot by increasing arithmetic
intensity:

`Arithmetic Intensity = FLOPs / Bytes moved to and from HBM`

### 3.4 MFU definition for your experiments

For a measured runtime `t`:

`Achieved FLOP/s = Estimated FLOPs / t`

`MFU = Achieved FLOP/s / Theoretical peak FLOP/s`

Use two MFU views:

1. `math_mfu`
   Based on algorithmic FLOPs.
2. `tensorcore_mfu`
   Based on the tensor-core-relevant FLOP count for the actual precision path.

Keep them separate because a kernel can have a high end-to-end speedup while
still not using hardware math units especially well.

## 4. High-Priority Operator List

This is the recommended implementation order.

| Priority | Operator id | Kind | Why it is high value | Recommended impl |
| --- | --- | --- | --- | --- |
| P0 | `naive_sdpa` | Dense exact | Ground-truth reference and roofline baseline | Triton |
| P0 | `online_softmax_fwd` | Primitive | The key building block before FlashAttention | Triton |
| P0 | `flash_v1_fwd` | Dense exact | First serious IO-aware win; teaches tiling | Triton |
| P0 | `flash_v2_fwd_bwd` | Dense exact | Best all-around kernel study target | Triton then CUDA |
| P0 | `grouped_query_attention` | Dense exact variant | Matches modern LLM inference/training layouts | Triton |
| P1 | `sliding_window_attention` | Structured sparse | Clean sparse pattern with real-world relevance | Triton |
| P1 | `paged_attention_decode` | Decode/inference | Teaches KV-cache systems design | Triton then CUDA |
| P1 | `splitk_decode_attention` | Decode/inference | Important when `Sq=1` and `Sk` is very large | Triton |
| P2 | `block_sparse_attention` | Structured sparse | Good deeper sparse project after windowed attention | Triton |
| P2 | `flash_v3_style_pipeline` | Dense exact | Advanced CUDA study topic | CUDA only, later |

## 5. Recommended Scope for This Repo

### 5.1 Triton set: 5 to 8 kernels

This is the sweet spot for a first serious round:

1. `naive_sdpa`
2. `online_softmax_fwd`
3. `flash_v1_fwd`
4. `flash_v2_fwd_bwd`
5. `grouped_query_attention`
6. `sliding_window_attention`
7. `paged_attention_decode`

Optional 8th:

8. `splitk_decode_attention`

This set covers:

- dense exact attention
- online normalization
- training-oriented flash kernels
- modern head-sharing layouts
- local structured sparsity
- serving-side decode kernels and KV cache behavior

### 5.2 CUDA set: 1 to 2 classic deep dives

Pick the kernels with the best "systems depth per hour spent":

1. `flash_v2_full`
   Forward and backward, exact dense attention, tile scheduling, warp/block
   partitioning, online softmax state handling, and numerics.
2. `paged_attention_decode`
   Real serving kernel with page tables, unaligned cache blocks, low arithmetic
   intensity, and strong memory-system sensitivity.

If you want one training kernel and one inference kernel, this pair is ideal.

## 6. Build Order

### Stage 0: Reference and measurement harness

Implement before any optimization:

- a PyTorch reference path using `torch.nn.functional.scaled_dot_product_attention`
- correctness checks against eager reference
- benchmark harness with warmup, repeats, and synchronized timing
- a compact operator registry so adding a new kernel is one row, not a refactor

### Stage 1: Math baseline

Implement:

- `naive_sdpa`

Learn:

- tensor layout cost
- score materialization penalty
- memory explosion with large `Sq x Sk`
- the gap between algorithmic FLOPs and useful throughput

### Stage 2: Streaming softmax

Implement:

- `online_softmax_fwd`

Learn:

- blockwise max/sum updates
- numerical stability
- why full score materialization is not required

### Stage 3: Flash exact attention

Implement:

- `flash_v1_fwd`
- `flash_v2_fwd_bwd`

Learn:

- SRAM/shared-memory tiling
- occupancy vs tile size
- register pressure
- causal masking without writing full masks
- work partitioning for forward and backward

### Stage 4: Model-driven variants

Implement:

- `grouped_query_attention`
- `sliding_window_attention`

Learn:

- head mapping when `Hq != Hkv`
- structured sparsity masks
- window-size dependent arithmetic intensity

### Stage 5: Serving kernels

Implement:

- `paged_attention_decode`
- optionally `splitk_decode_attention`

Learn:

- `Sq = 1` decode behavior
- memory-bound kernels
- KV cache addressing
- page/block table overhead
- split-K tradeoffs for long decode contexts

## 7. Optimization Checklist by Kernel

Use the same progression for each kernel so comparisons stay fair.

| Step | What to optimize | Typical indicators |
| --- | --- | --- |
| 1 | Correctness first | max error, mean error, reproducibility |
| 2 | Layout and strides | contiguous loads, coalescing, fewer transposes |
| 3 | Tile sizes | runtime, occupancy, register usage |
| 4 | Memory hierarchy | shared-memory hit rate, HBM bytes, L2 behavior |
| 5 | Parallel decomposition | warp/block balance, persistent scheduling, split-K |
| 6 | Numerics | fp16/bf16 accumulation path, online rescaling |
| 7 | Fusion | fewer round trips to HBM |
| 8 | Backward | saved state size, re-materialization tradeoffs |

## 8. PyTorch Integration Strategy

Start simple and keep replacement points explicit.

### Option A: Registry-driven wrapper

Create one wrapper:

`attention_impl(name, q, k, v, ...)`

Then dispatch to:

- PyTorch SDPA
- Triton kernels
- CUDA extensions

This is the cleanest way to keep benchmarks and model integration aligned.

### Option B: Module-level drop-in

Wrap `nn.Module` blocks such as:

- `CausalSelfAttention`
- `MultiheadAttention`

Swap only the attention call while keeping projections unchanged.

Do this after the standalone benchmark harness is stable.

## 9. Fair Benchmark Design

All kernels should be tested under the same axes.

### 9.1 Core dimensions

- mode: `fwd`, `bwd`, `fwd_bwd`, `decode`
- dtype: `fp16`, `bf16`, optionally `fp32` reference only
- mask: `none`, `causal`, `sliding_window`, `block_sparse`
- batch size `B`
- query length `Sq`
- key length `Sk`
- query heads `Hq`
- KV heads `Hkv`
- head dimension `D`
- page size or block size for paged/block-sparse kernels

### 9.2 Measurement outputs

- runtime in ms
- achieved TFLOP/s
- estimated HBM bytes
- arithmetic intensity
- math MFU
- max memory allocated
- max abs error
- max rel error

### 9.3 Case families to include

1. Training short-medium context
   `Sq=Sk in {128, 512, 2048, 8192}`
2. Long-context training
   `Sq=Sk in {8192, 16384, 32768}` where hardware allows
3. Decode
   `Sq=1`, `Sk in {512, 2048, 8192, 32768, 131072}`
4. GQA
   `Hq/Hkv in {(32, 32), (32, 8), (32, 4)}`
5. Sliding window
   `window in {128, 256, 512, 1024}`

## 10. Suggested Analysis Table

Use the following questions when writing up each kernel:

| Operator | Scenario | Time ms | TFLOP/s | Est. Bytes | AI | Math MFU | Bound guess | Main bottleneck | Next fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `naive_sdpa` | `B=4,H=32,S=2048,D=128` |  |  |  |  |  | Memory | score materialization | stream softmax |
| `flash_v1_fwd` | `B=4,H=32,S=2048,D=128` |  |  |  |  |  | Mixed | tile size | retune blocks |
| `paged_attention_decode` | `B=32,Sq=1,Sk=32768,D=128` |  |  |  |  |  | Memory | KV indirection | page/block tuning |

## 11. Concrete Recommendation

If you want the highest return path:

1. Build the benchmark harness first.
2. Implement `naive_sdpa` and `online_softmax_fwd`.
3. Move immediately to `flash_v1_fwd`.
4. Spend the largest Triton effort on `flash_v2_fwd_bwd`.
5. Add `grouped_query_attention` and `sliding_window_attention`.
6. Finish Triton with `paged_attention_decode`.
7. Re-implement `flash_v2_full` and `paged_attention_decode` in CUDA.

That path gives you one coherent story from math baseline to real serving
kernel, instead of a collection of disconnected experiments.
