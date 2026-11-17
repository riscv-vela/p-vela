# DistilRoBERTa Power Backend Flow

This document describes the power-only DistilRoBERTa guest binaries.

## Scope

The power-only CPU and Gemmini binaries share the same runner:

```c
distilroberta_cpu_power_infer.c
  -> DISTILROBERTA_USE_GEMMINI_BACKEND 0
  -> include "distilroberta_gemmini_infer.c"

distilroberta_gemmini_power_infer.c
  -> DISTILROBERTA_USE_GEMMINI_BACKEND 1
  -> include "distilroberta_gemmini_infer.c"
```

So the sample loop, forward graph, quantization, postprocess, metrics, output, and timeline locations are common. The backend changes only inside `linear_forward()`.

The non-power `distilroberta_int8_infer.c` remains the original standalone CPU reference path.

## Common Linear Flow

Both backends run the same high-level linear flow:

```text
input fp32
-> quantize input to int8
-> backend A_int8 * W_int8 accumulation
-> acc_scale
-> bias
-> requantize with output_scale
-> dequantized fp32 output for following layers
```

Equivalent formula:

```text
acc = backend_sum_k(A_int8 * W_int8)
out = dequantize(requantize(acc * acc_scale + bias, output_scale))
```

## Backend Difference

CPU backend:

```c
cpu_matmul_acc_chunk()
```

- Runs the int8 accumulation with a C loop on the guest vCPU.
- Computes `acc = sum_k A_int8 * W_int8`.
- Does not use Gemmini flush/fence/executor submit/wait.

Gemmini backend:

```c
tiled_matmul_auto()
```

- Sends only the int8 accumulation step to the Gemmini API.
- In split mode, the executor thread is pinned to `--gemmini-vcpu`.
- Includes the required Gemmini runtime overhead: `gemmini_flush()`, `gemmini_fence()`, memory fence, chunk submit/wait, and executor scheduling.

## Intentional Differences

The only intended runtime differences are:

- `backend=cpu` vs. `backend=gemmini`
- `cpu_matmul_acc_chunk()` vs. `tiled_matmul_auto()` / executor
- Gemmini-only flush/fence/executor overhead
- `gemmini_vcpu=-1` for CPU vs. configured Gemmini executor vCPU for Gemmini

Backend-internal debug logs are off by default. Use `--log-backend-calls` only when debugging backend calls, not for power comparison runs.
