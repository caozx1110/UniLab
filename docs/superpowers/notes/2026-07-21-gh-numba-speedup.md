# GH numba update_state acceleration — speedup measurement

**Date:** 2026-07-21 · **Machine:** 5090 GPU + 32-core CPU · **num_envs=4096, horizon=32, train phase** · numba 0.66.0 (OMP threading layer, 32 threads)

## Headline

| metric | numba OFF | numba ON | speedup |
|---|---|---|---|
| update_state | 1.712 s | 1.410 s | **1.21x** |
| collect | 5.111 s | 4.825 s | 1.06x |
| **iter (total)** | **5.935 s** | **5.659 s** | **1.05x** |
| CPU util (system) | 18% | 24% | — |
| busy cores /32 | 1.0 | 1.1 | — |

Parity: full 10-step all-outputs test passes at rtol=1e-4/atol=1e-5, **max abs diff 6.6e-6**.

## Why only 1.05x end-to-end (honest breakdown)

**1. update_state is only ~29% of the iter.** The iter is dominated by segments this task did NOT touch: physics `mj_step` ~1.26s, `set_ctrl` per-substep loop ~0.79s, astype fp64→fp32 ~0.44s, train (GPU) ~0.82s. Even a perfect update_state → 0 would cap iter speedup at ~1.4x.

**2. Within the numba update_state (1.40s), the accelerated part is a minority:**

| phase | s/iter | status |
|---|---|---|
| `before_update` (motion slice gather + force/admittance) | **0.786 (56%)** | **numpy, OUT OF SCOPE — now the ceiling** |
| reward prep + kernel | 0.222 | accelerated (was ~0.24) |
| obs prep + kernel | 0.393 | accelerated (was ~0.66, **~1.7x**) |

The obs kernel delivered a real ~1.7x on its slice; reward barely moved because its arithmetic was already cheap and the **numpy input-prep (fp32 conversion + array gathering) dominates the reward path**, partially offsetting the parallel kernel.

**3. `before_update` (0.786s) was explicitly deferred** and is now 56% of update_state — the single largest remaining lever inside update_state.

## Verified: numba prange genuinely parallelizes

Isolated compute-bound kernel (4096×700, sin·cos): parallel 0.559 ms/call vs sequential 9.831 ms/call = **10.9x**, OMP layer, 32 threads. The infrastructure is not the bottleneck — the GH kernel's compute is simply a small slice of update_state, and its numpy input-prep runs single-core.

## Conclusion

The "safe part" (reward + obs + termination) is correct, parity-locked, and parallelizes where it computes — but the end-to-end win is modest (1.05x iter) because update_state is a minority of the iter and `before_update` + numpy input-prep dominate what's left. Bigger levers, in order:

1. **`before_update`** (0.786s, deferred) — motion slice gather + force/admittance. Largest single block inside update_state.
2. **Reduce input-prep overhead** — the fp32 conversion + array gathering feeding the kernels runs single-core numpy and offsets gains (esp. reward). Could gather directly into pre-alloc'd fp32 buffers or widen the kernel to consume raw backend arrays.
3. **Out-of-scope iter dominators** — `set_ctrl` per-substep loop (0.79s), astype (0.44s), and the structural collect serialization remain the biggest whole-iter opportunities.
