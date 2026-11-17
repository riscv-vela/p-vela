# Energy Evaluation — Quantitative Evaluation of 20% Energy Savings

**Denominator = CPU + NPU (memory excluded)**. Synthetic decision logs (frequency, gating) are
overlaid on existing run outputs to reconstruct the energy savings of low-power techniques. Because this is
a reconstruction rather than a re-measurement, both scenarios carry the same measurement noise, so the
comparison is fair.

## Files

| File | Purpose |
|---|---|
| `energy_model.py` | **Main tool.** Computes energy on a piecewise time axis + frequency sweep |
| `energy_compare.py` | Initial version (based on scaled CSV, alpha approximation). Kept for regression checks |

```bash
RUN=.../dual_app_power_modeling/logs/distilbert_polybench_gemm/20260812_004450_...
python3 energy_model.py "$RUN" --app distilbert --npus 4
python3 energy_model.py "$RUN" --app polybench_gemm --npus 8 --freqs 2.512,2.0,1.5
```

⚠️ **Pin the run directory by name.** `ls -td` (sorted by mtime) picks up an old, invalid run
if that run gets touched (this actually happened).

## Why recompute the time axis — the attribution error in the SSU model

The SSU pipeline compresses a gemmini interval into **a single blended scale** and applies it uniformly
across the interval (`time_scale_gemmini_logs.py` L1215-1216):

```
scale = (rtl_time + embedded_cpu_raw × 0.235) / gemmini_raw_time
```

However, the interval consists of two parts whose QEMU-to-hardware ratios differ by 100x:

| Component | raw | Hardware | Compression | Source |
|---|---|---|---|---|
| NPU compute (tiled_matmul) | 70.4 s | 0.146 s | ~480x | RTL cycle model (validated by measurement) |
| CPU work (embedded CPU) | 3.6 s | 0.840 s | ~4.3x | Fixed coefficient `cpu_time_scale=0.235` |

Result of uniform compression — **the total matches because the errors cancel, but the internal breakdown is inverted**:

| | Applied value | Intended value | Error |
|---|---|---|---|
| NPU part | 0.938 s | 0.146 s | 6.4x too long |
| CPU part | 0.048 s | 0.840 s | 17.6x too short |
| Total | 0.986 s | 0.986 s | ✓ |

It shows up in energy in two directions:
- **NPU**: power is a busy-based step function (0 / 0.862 W), so it is overestimated in proportion to the longer active time → **about 4–7x**
- **CPU**: power differs by 3x between the vCPU waiting on the NPU (0.13 W) and actually working (0.38 W),
  and the waiting portion occupies 95% of the interval → **underestimated**

There is a single cause: **inside a gemmini interval, NPU compute time and CPU work time are not distinguished.**

## Fix

Every raw sample is classified into one of 3 classes, and each class gets its own scale. **The model's total time budget
is preserved**, so the runtime does not change; only the attribution is corrected.

```
npu_compute   (inside gemmini interval & worker busy≥90%)  → total set to rtl_time_s
embedded_cpu  (inside gemmini interval & worker idle)      → total set to embedded_cpu_scaled_time_s
cpu_outside   (outside gemmini interval)                    → each sample × cpu_time_scale(0.235)
```

The DVFS slowdown applies only to `embedded_cpu` and `cpu_outside` (the NPU has an independent clock).

For **workloads without a backend profile** (polybench, etc., `raw_tiled_matmul_profile_s = null`),
the model counts the entire gemmini interval as NPU time, so **there is no component to split and no attribution
error.** The tool detects this automatically and skips the split; it self-verifies by producing identical results
in both modes. In other words, this error is **limited to transformer-family workloads that report a backend profile**.

**Note**: the raw total of `embedded_cpu` is larger under turbostat (5.15 s) than under the app profile (3.57 s)
— due to turbostat's 100 ms/90% threshold. Since the total is fitted to the model budget, the totals are
consistent, but the class boundaries carry an error of this magnitude.

## Model

```
CPU  P = 0.5625 × (a·util·f³ + b)      a,b = SSU coefficients (regression check error 0.00%)
NPU  active 0.862 W / powered-idle 0.013 W / gated 0 W
```
Integration is trapezoidal (same as `summarize_power_csv.py`). Do not use left-endpoint rectangles; they
overestimate by 7% on a binary NPU signal.

## Known approximations

- This tool computes on the raw `gemmini_power.csv` grid (about 104 ms) and joins vCPU util with ZOH.
  Compared with SSU's scaled CSV (interpolated fine grid), CPU energy differs by about 10%
  (0.529 vs 0.585 J). This does not affect conclusions about savings rates (ratios), but take care when quoting absolute values.
- If no interval is found / there are 0 NPU-active samples, the tool **aborts immediately** (to avoid silently wrong answers).
- Pin the run directory by name. `ls -td` can pick up an invalid run.

## Results (one DistilBERT inference, 1 of 4 NPUs used)

```
BASELINE   current 1.394 J (CPU 0.529 + NPU 0.865)
           fixed   0.837 J (CPU 0.627 + NPU 0.210)   runtime 1.653 s (unchanged)
           → corrects NPU 4.1x overestimate / CPU 1.2x underestimate

Savings    gating only      10.1%
           DVFS+gating      20.1% @2.2GHz / 25.2% @2.0GHz / 32.5% @1.5GHz (optimal)
           by NPU count     1-NPU 19.0% → 2-NPU 21.2% → 4-NPU 25.2% → 8-NPU 32.2%

PolyBench (NPU-dominated, no attribution error): gating 3.8%, almost no DVFS effect
→ the 20% claim must be made with transformer workloads that have a large CPU share
```

An optimal frequency exists (theoretical f* = (β/(2·α·util))^⅓ ≈ 1.5 GHz). Going lower, the static power
penalty eats up the f² gain — **"lower the frequency as much as possible" is not the answer.**

## Phase 2 and beyond

Replace the synthetic logs with actual governor logs (`epoch_ns, cpu_id, freq_mhz` / `epoch_ns, npu_id, powered`,
recording change points only + zero-order hold). Replace slowdown(f), currently `f_ref/f`, with the low-power team's
instruction-complexity regression model.

## Rationale for the operating point (frequency) — updated 2026-09-07

The SSU workshop materials confirmed the origin and **validation range** of the power model.

| Item | Details |
|---|---|
| CPU power model construction | AMD Ryzen 5 7600 + QEMU 9.2.0, workloads FFT/INT64/Float/JMP/OpenSSL/7z |
| RISC-V port | scaling factor θ applied (`P' = θ·P`) — `sys_scale = 0.5625` in the code |
| **Validation target/range** | **Real board HiFive Premier P550, 1.2 / 1.3 / 1.4 GHz** |
| Validation error | Mean 0.0277 W (2.30%) |
| CPU performance model | Ratio of board execution time / QEMU execution time, 4 SPEC CPU2017 benchmarks (mcf, lbm, x264, deepsjeng) — `cpu_time_scale = 0.235` in the code |
| NPU power | ASAP7 7nm synthesis at 200 MHz, Fmax 258 MHz, Total 0.862 W (Internal 3.27e-1 + Switching 5.33e-1 + Leakage 9.93e-5) |
| NPU cycle model | `35 + 16·M_t·N_t·K_t − pad_M + max(0,4−M) + 16·max(0,P−257)`, exact match on all 3,503 Verilator cases |

**Therefore the evaluation operating point is fixed at 1.4 GHz** (`--fref 1.4`). The
`FIXED_FREQ_GHZ=2.512` used by the existing experiment runner is outside the validation range, and also conflicts with the
default (1.4) of SSU's own `power_sim_gemmini.py` / `power_sim_cpu_only.py`.

### Result: DVFS strategy dropped, target met with power gating alone

The power model's energy-optimal frequency is `f* = (β / 2αu)^(1/3) ≈ 1.44 GHz` (measured util 0.357),
**which effectively coincides with the board's 1.4 GHz ceiling**. That is, the entire P550 operating range (1.2–1.4 GHz) is already
at or below the optimum, so lowering the frequency actually increases energy (+1.4% at 1.2 GHz).

Savings at fixed 1.4 GHz with idle-NPU power gating alone:

| NPU count | 1 | 2 | **4** | 8 | 16 |
|---|---|---|---|---|---|
| Savings | 5.0% | 9.8% | **18.3%** | 31.1% | 47.6% |

18.3% with 4 NPUs, 31.1% with 8 NPUs — the 20% target is met by gating alone with 5 or more NPUs.
Setting the DVFS contribution to 0 is the defensible statement.

Reproduce:
```bash
python3 energy_model.py <run_dir> --app distilbert --npus 4 \
        --fref 1.4 --freqs 1.4,1.3,1.2 --dvfs-freq 1.3
python3 make_figures.py          # sweep_1p4.json → energy_summary.png
```

## Parameter update — reflecting the SSU final delivery (2026-10-07)

Constants were aligned with `power_model_defaults.env` in `ssu_final.zip` (final contract deliverable, 26-09-30).

| Item | Old | New | Notes |
|---|---:|---:|---|
| `CPU_TIME_SCALE` | 0.235 | **0.3035** | Has a real impact |
| `GEMMINI_UTIL_THRESHOLD_PCT` | 90 | **80** | No effect on old runs (see below) |
| NPU idle power | 0 (we assumed 0.013) | **0.013** | Adopted by the delivery — matches our assumption |

**Why the threshold change is meaningless for old runs**: the `gemmini_power.csv` files collected in 26-08 contain
only values in the `busy_core_<N>` columns that were **already binarized to 0/100 at collection time**
(checking unique values shows only two: `0.000000` / `100.000000`). The `raw_busy_core_<N>` columns holding
the raw pre-threshold Busy% only exist starting with the new delivery. So re-applying 80% does not
change the results, and `load()` prefers `raw_busy_core_*` when present.

**embedded_cpu rescaling**: `embedded_cpu_scaled_time_s` in `summary.json` is a value **already compressed** with the
`cpu_time_scale` in effect when the run was collected. `load()` reads the `cpu_time_scale` field from the same JSON,
undoes it with `× (current / at collection)`, and then applies the scale again.

### Updated results (DistilBERT, 1.4 GHz, 1 of 4 NPUs used)

```
              runtime      CPU        NPU       baseline
Old   0.235   1.653 s   0.2503 J   0.2098 J   0.4601 J
New   0.3035  2.093 s   0.3187 J   0.2327 J   0.5513 J
```

| NPUs installed | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Old (0.235) | 5.0% | 9.8% | 18.3% | 31.1% | 47.6% |
| **New (0.3035)** | **5.4%** | **10.6%** | **19.4%** | **32.7%** | **49.4%** |

CPU time grows by 29%, so the runtime goes from 1.653 → 2.093 s, and the idle leakage grows along with it,
so **the savings rate actually rises slightly** (0.0841 → 0.1069 J). With 4 NPUs it reaches 19.4%, close to the 20% target
but still short; the conclusion that the target is met with 5 or more NPUs still holds.

### ⚠ Values where the report and the code disagree

The delivered code and the body of the SSU contract report differ. The current tool follows **the code**.

| | Report | Code |
|---|---|---|
| Correction factor γ | 0.235 | 0.3035 |
| scaling factor θ | 0.54 | 0.5625 |
| NPU active threshold | 90% | 80% |
| Static power | "measured board idle value, θ correction on dynamic only" | `sys_scale × (a·u·f³ + k)` — θ is also multiplied into the static term |

The last item determines the DVFS conclusion (location of f\*), so it needs to be confirmed.

Reproduce:
```bash
python3 energy_model.py <run_dir> --app distilbert --npus 4 --fref 1.4 \
        --freqs 1.4,1.3,1.2 --cpu-time-scale 0.3035 --busy-th 80
```

## NPU idle power update — 0.013 → 0.014 W (2026-11-09)

Based on re-measurement results, the power of an NPU that is powered on but not computing was raised to 0.014 W.
The active power of 0.862 W and the gated 0 W are unchanged. The changes were made in `energy_model.py`,
`summarize_multi_npu.py`, `energy_compare.py`, and on the instrumentation side `power_model_defaults.env`
(ssu tree, program package). The original delivery `ssu_final/` was not modified.

```
              runtime      CPU        NPU       baseline (4-NPU, 1.4 GHz)
0.013 W       2.093 s   0.3187 J   0.2327 J   0.5513 J
0.014 W       2.093 s   0.3187 J   0.2409 J   0.5596 J
```

| NPUs installed | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Idle 0.013 W | 5.4% | 10.6% | 19.4% | 32.7% | 49.4% |
| **Idle 0.014 W** | **5.8%** | **11.3%** | **20.6%** | **34.3%** | **51.2%** |

The post-gating energy (0.4444 J) does not depend on idle power, so the higher idle power raises only
the baseline, and the savings rate goes up. **With 4 NPUs it reaches 20.6%, exceeding the 20% target.**
The DVFS conclusion is unchanged (1.3 GHz −1.8%, 1.2 GHz −4.4%, minimum ≈1.44 GHz).

`sweep_1p4.json` was regenerated with a script that reproduces the previous file with zero error
(`params.npu_idle_w` field added). Reproduce:
```bash
python3 energy_model.py <run_dir> --app distilbert --npus 4 --fref 1.4 \
        --freqs 1.4,1.3,1.2 --dvfs-freq 1.4
python3 make_figures.py
```
