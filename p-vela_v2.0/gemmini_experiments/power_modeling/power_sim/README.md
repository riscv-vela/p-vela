# Power Simulation

`power_modeling/power_sim` handles raw CPU/Gemmini power calculation, Gemmini RTL cycle
calculation, and time-scaled log generation.

## Key Files

- [power_model_defaults.env](./power_model_defaults.env)
  - Default power, period, frequency, time-scale, and scaled-plot values shared by single/dual and Python
- [power_model_config.py](./power_model_config.py)
  - Common Python loader that reads the `KEY=value` entries of the file above
- [power_sim_vcpu_procstat.py](./power_sim_vcpu_procstat.py)
  - In single-app, reads the guest `/proc/stat` together with the shared host `turbostat`
    and produces `raw_power.csv` in real time
- [power_sim_gemmini.py](./power_sim_gemmini.py)
  - Provides the common power equations and the turbostat parser; also used as the host collector for dual-app
- [merge_split_power_csv.py](./merge_split_power_csv.py)
  - Merges, by timestamp, the CPU/Gemmini CSVs collected separately in dual-app
- [gemmini_hw_cycle_model.py](./gemmini_hw_cycle_model.py)
  - Computes the validated no-DMA `LOOP_WS` and application workload cycles
- [time_scale_gemmini_logs.py](./time_scale_gemmini_logs.py)
  - Converts raw logs to the target CPU/Gemmini time axis

Only the code used by the current single/dual execution paths is kept.
Validation tests not used for running experiments have been removed and can be restored from Git history.
The former 7z/OpenSSL, CPU-only, and mpstat-based simulators and unused post-processing tools
have been removed and can be restored from Git history. The CPU power equation
is kept in `power_sim_gemmini.py`.

## Changing Common Settings

Edit the following values in a single place, [power_model_defaults.env](./power_model_defaults.env).

- Gemmini full/idle power and utilization threshold
- CPU frequency, platform scale, and AMD power-equation coefficients
- Host turbostat, power polling, and guest vCPU sampling periods
- Time mode, Gemmini frequency, and CPU scale factor
- Scaled plot display margin, width, and time ticks

Shell runners read the same file via `source`, and Python reads it via `power_model_config.py`.
Write values as `KEY=value` literals, without shell commands or variable references.
Existing command-line overrides take precedence over the defaults, and the dual runner's environment-variable
overrides are also kept. Setting changes apply to new processes only; existing run logs are not modified.
Core placement, SSH, and application execution settings, as well as RTL structure/profile constants, remain in their respective code.

The defaults for running Python directly have also been aligned with the current experiments. The host collector's frequency and
polling change from the former `1.4 GHz / 1 s` to `2.512 GHz / 0.1 s`, and time conversion changes from
`manual / CPU 1.0` to `rtl-cycle / CPU 0.3035`. The default values that the existing single/dual
runners passed explicitly are unchanged. Override via the CLI if needed.

## Raw Power Calculation

In the canonical single-app runner, the integrated collector in the guest reads both sources in real time
and writes a single power CSV.

```text
Guest /proc/stat + shared raw_turbostat.log -> raw_power.csv
```

`raw_power.csv` records vCPU, Gemmini, the totals of both components, and `p_total_w` in the same
row. The dual runner writes split CPU/Gemmini CSVs and then merges them.

`raw_busy_core_<Gemmini core>` is the pre-threshold host Busy% read by the collector.
`busy_core_<Gemmini core>` is the 0/100% model value, and `busy_core_<vCPU core>` is the guest
CPU utilization. The single raw/scaled graph values use only the power CSV;
the bench log is read separately for the time reference and interval markers. New columns are preserved in dual CSV
merging and time conversion as well. Rows with the new schema are not appended to old CSVs.

The Gemmini worker is binarized with the default 90% threshold before the real-time collector writes it
to the CSV.

```text
Busy% >= threshold  -> util=1.0, power=full_util_power_w
Busy% <  threshold  -> util=0.0, power=0.013 W
```

The default `full_util_power_w` uses the current setting in `power_model_defaults.env`,
`0.862 W`. The threshold and power can be changed with
`--gemmini-util-threshold-pct` and `--gemmini-full-util-power-w`, respectively.

The Gemmini utilization in single/dual plots is displayed by multiplying `util_core_<N>` in the power CSV by 100,
and uses the same `sample_tod_s` as power. The graphs do not
reapply the threshold. Separate dual host/guest diagnostics use the original turbostat and
`/proc/stat` CSVs. Single plots do not read these logs directly.
No separate threshold log is generated.
The guest collector also records `busy_core_<N>` and `util_core_<N>` of raw-only vCPUs in the power CSV.
These columns are for the guest executor/housekeeping graphs and are not
used in the power sum. Runs without these columns must be collected with a new experiment.
The original turbostat log is passed to time conversion, and scaled plots also use the
utilization of the converted power CSV as is.

An example of direct execution is as follows.

```bash
python3 gemmini_experiments/power_modeling/power_sim/power_sim_gemmini.py \
  --turbostat /path/to/turbostat.log \
  --out-log /path/to/gemmini_power.csv \
  --gemmini-cores 2 \
  --fixed-freq-ghz 2.512 \
  --sys-scale 0.5625 \
  --gemmini-full-util-power-w 0.862 \
  --gemmini-util-threshold-pct 90 \
  --poll-interval 0.1
```

## Gemmini RTL cycle model

[gemmini_hw_cycle_model.py](./gemmini_hw_cycle_model.py) implements the `LOOP_WS accept -> idle` cycle model
validated on the production Gemmini Top TB. The current validated scope is
DMA excluded, `DIM=16`, `M_TILES=1`.

```text
M          = 16*M_TILES - PAD_M
P          = N_TILES*K_TILES
compute    = 16*M_TILES*P - PAD_M
drain      = max(0, 4-M)
backpressure = 16*max(0, P-257)

LOOP_WS cycles = 35 + compute + drain + backpressure
```

When multiple `LOOP_WS` are executed back-to-back within a single API call, they share the fixed overhead
between commands.

```text
workload cycles
  = sum of each LOOP_WS cycle
  - 35*(number of LOOP_WS calls - number of API calls)
```

The current application profiles support `distilbert`, `distilroberta`, `distilgpt2`, and
`polybench_gemm`. PolyBench is based on `MEDIUM_DATASET`.

```bash
python3 gemmini_experiments/power_modeling/power_sim/gemmini_hw_cycle_model.py \
  distilbert --executions 1 --frequency-mhz 200
```

## Time scaling

[time_scale_gemmini_logs.py](./time_scale_gemmini_logs.py) replaces QEMU raw intervals with
the target CPU and Gemmini RTL times.

Required inputs:

```text
bench.log      sample window, Gemmini event, backend profile
turbostat.log  host core utilization
power.csv      per-component power and total power
```

If the optional input `--guest-util-csv` is given, the guest `/proc/stat` CSV is also converted to the same
time axis.

### Measurement Intervals

```text
inference CPU interval = sample-start/done window - Gemmini active interval
Gemmini interval       = actor=gemmini event -> matching *_end event
```

- `--cpu-time-scale` is applied to CPU intervals.
- The selected Gemmini time mode is applied to Gemmini intervals.
- Timestamps after a converted interval are shifted earlier by the amount of time removed.
- The same interval conversion is applied to bench, turbostat, power, and guest-util.

### Time mode

| Mode | Description |
|---|---|
| `rtl-cycle` | Current runner default. Converts workload cycles to time at the specified frequency |
| `cpu-ref-auto` | Legacy-compatible mode that uses the CPU reference bench ratio |

Both direct script execution and the single/dual runners use the mode and CPU time scale from the common settings.
The current defaults are `rtl-cycle` and `0.3035`. The single/dual runners support only RTL,
and the `cpu-ref-auto` compatibility path in the Python utility is kept separately.
The `manual` mode, `--time-scale`, `--model-time-scale`, and per-application fixed scale factors
have been removed. The CPU `--cpu-time-scale` is kept as is.

The Gemmini target time for `rtl-cycle` is as follows.

```text
RTL time = workload cycle total / (frequency MHz * 1,000,000)

embedded CPU raw time
  = raw Gemmini active time - backend-profile tiled_matmul time

Gemmini target time
  = RTL time + embedded CPU raw time * cpu_time_scale

Gemmini time scale
  = Gemmini target time / raw Gemmini active time
```

If the number of Gemmini intervals expected by the profile differs from the number of intervals observed in the bench,
execution is aborted to prevent applying the wrong workload.

For a fixed workload profile, execution count, and frequency, `RTL time` is always the same.
`WorkloadProfile.cycles_per_execution` sums the LOOP_WS cycles on first use
and reuses the result within the same process. Changing the execution count or frequency does not
repeat the profile summation. Starting a new process recomputes it once,
and no separate disk cache or duplicated numeric table is kept.
Post-processing multiplies this cycle count by the execution count and divides by the frequency. The Transformer execution count is
determined by the number of `sample-start` events, and for PolyBench by the number of `gemm_begin` events.
However, the embedded CPU correction and the denominator (raw Gemmini time) are taken from the execution logs, so
the final target time and the actual applied scale factor may vary from run to run.
The current conversion matches the total application time and does not assign a different RTL
target time to each individual GEMM. The single/dual target time and scale-factor calculation and profile validation
share a single `resolve_rtl_workload_time_scale()`. Measured times and the embedded CPU correction are
not cached, and the existing RTL cycle equation and embedded CPU correction equation are kept unchanged.

### Single-app Example

```bash
python3 gemmini_experiments/power_modeling/power_sim/time_scale_gemmini_logs.py \
  --bench /path/to/raw_bench.log \
  --turbostat /path/to/raw_turbostat.log \
  --power-csv /path/to/raw_power.csv \
  --guest-util-csv /path/to/raw_guest_proc_stat.csv \
  --time-scale-mode rtl-cycle \
  --rtl-workload distilbert \
  --gemmini-frequency-mhz 200 \
  --cpu-time-scale 0.3035 \
  --out-prefix /path/to/time_scaled_rtl_200mhz_cpu_0p3035
```

In single-app, only the CPU/Gemmini intervals within the inference sample window are converted,
and the collector preparation intervals outside it keep their original spacing.

### Dual-app Independent Time Axis Example

```bash
python3 gemmini_experiments/power_modeling/power_sim/time_scale_gemmini_logs.py \
  --bench /path/to/raw_bench.log \
  --turbostat /path/to/raw_turbostat.log \
  --power-csv /path/to/raw_power.csv \
  --guest-util-csv /path/to/raw_guest_proc_stat.csv \
  --time-scale-mode rtl-cycle \
  --model-rtl-workload distilbert=distilbert \
  --model-rtl-workload distilgpt2=distilgpt2 \
  --model-core distilbert=3,9,11 \
  --model-core distilgpt2=2,10,12 \
  --gemmini-frequency-mhz 200 \
  --cpu-time-scale 0.3035 \
  --out-prefix /path/to/time_scaled_rtl_200mhz_cpu_0p3035
```

Dual-app uses the model prefix and `--model-core` to independently convert each app's
`remote_launch -> sample_start`, inference CPU, and Gemmini intervals. The scaled power CSV merges all
sample timestamps converted from both apps, interpolates
each core's values, and then recomputes the component totals and `p_total_w`.

## Output

The following files are generated based on `--out-prefix PREFIX`.

```text
PREFIX_bench.log
PREFIX_turbostat.log
PREFIX_guest_util.csv  # when --guest-util-csv is specified
PREFIX_power.csv
PREFIX_summary.json
```

If `--result-prefix RESULT_PREFIX` is specified, only the power CSV and summary are saved to
`RESULT_PREFIX_power.csv` and `RESULT_PREFIX_summary.json`.
The bench/turbostat/guest-util intermediate files still follow `--out-prefix`.
The single runner uses this to separate results and runs, while the existing dual paths are kept.

`raw_sample_tod_s` is added to the scaled CSV for checking the original timestamps.
The main fields of `summary.json` are as follows.

```text
time_scale_mode
cpu_time_scale
cpu_raw_time_s, cpu_scaled_time_s
gemmini_raw_time_s, gemmini_scaled_time_s
inference_raw_time_s, inference_scaled_time_s
energy_raw_j, energy_scaled_j
rtl_cycle_total, rtl_time_s                  # single rtl-cycle
model_rtl, model_time_scales, per_model_time # dual independent
sample_windows, cpu_intervals, gemmini_intervals
```

The current single-app and dual-app runners automatically call this utility after raw measurement finishes
and generate the scaled CSV and plots. Both runners pass the CPU time scale default `0.3035`
and the Gemmini frequency default `200 MHz`.
