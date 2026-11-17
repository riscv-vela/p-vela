# Single-app Power Modeling

## 0. Prerequisites

| Item | Requirement |
|---|---|
| Application | Prepare the Gemmini power binaries and the model/input data as described in [Applications](../../applications/README.md) |
| QEMU | Boot with `1 3` as described in [QEMU Boot](../../../qemu_boot/README.md) and apply `pin.sh 8 2` |
| Host | `sudo turbostat`, Python 3 and `matplotlib` available |
| Guest | Python 3, `timedatectl`, and a `sudo` environment that can run the required commands without a password prompt |
| SSH | Default `ubuntu@127.0.0.1:2200`, key `q-vela/tmp-cloudinit/id_ed25519` |
| Shared folder | Host `gemmini_experiments/` is mounted at guest `/mnt/hostshare` and is writable for results |

The single runner uses binaries that are already built; it does not build them automatically.
PolyBench uses the `MEDIUM_DATASET` binary.
Currently only `--threads 1` is supported for the number of application threads.

## 1. How to Run

Run all commands from the **workspace root on the host**, in a terminal separate from QEMU.
The entry script is [run_single_app_power.sh](./experiment_scripts/run_single_app_power.sh).

### Usage and Arguments

```text
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh APP [OPTIONS]
```

The first argument `APP` is one of `distilbert`, `distilroberta`, `distilgpt2`, or `polybench_gemm`.
You may also specify it as `--app APP` instead of `APP`.

| Option | Default | Purpose |
|---|---:|---|
| `--sample-limit N` | `1` | Number of Transformer input samples |
| `--reps N` | `50` | Number of PolyBench GEMM repetitions |
| `--threads N` | `1` | Number of application threads; currently only `1` is supported |
| `--run-id ID` | Auto-generated | Run ID that identifies results and logs. Must not duplicate an existing ID |
| `--fixed-freq-ghz FLOAT` | `2.512` | Frequency used by the CPU power model |
| `--sys-scale FLOAT` | `0.5625` | CPU power model calibration factor |
| `--gemmini-full-util-power-w FLOAT` | `0.862` | Gemmini active-state power (W) |
| `--gemmini-util-threshold-pct FLOAT` | `80` | Busy% threshold for judging Gemmini as active |
| `--time-scale-mode rtl-cycle` | `rtl-cycle` | Currently supported time scaling mode |
| `--gemmini-frequency-mhz FLOAT` | `200` | Frequency used to convert RTL cycles to time |
| `--cpu-time-scale FLOAT` | `0.3035` | Time scaling factor for CPU phases |

Common defaults for the power and time models are managed in
[power_model_defaults.env](../power_sim/power_model_defaults.env).
Command-line options take precedence over the defaults for that run, and configuration changes apply from the next experiment.

Use the options below only when changing the core placement. They must match the QEMU pinning.

| Option | Default | Purpose |
|---|---:|---|
| `--main-vcpu N` | `1` | Guest vCPU for the application main thread |
| `--housekeeping-vcpu N` | `0` | Housekeeping guest vCPU |
| `--executor-vcpu N` | `2` | Gemmini executor guest vCPU |
| `--vcpu-base-core N` | `8` | Host CPU number corresponding to guest vCPU 0 |
| `--gemmini-host-core N` | `2` | Host CPU number of the Gemmini worker |

The CPU for the host control process is set with the `HOST_HOUSEKEEPING_CORE` environment variable; the default is 8.

### Examples

Run DistilBERT with 1 sample:

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    distilbert --sample-limit 1
```

Replace `distilbert` with `distilroberta` or `distilgpt2` to run them the same way.

Run PolyBench GEMM 50 times:

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    polybench_gemm --reps 50
```

Run with explicit model settings:

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    distilbert --sample-limit 1 \
    --gemmini-util-threshold-pct 80 \
    --gemmini-frequency-mhz 200 \
    --cpu-time-scale 0.3035
```

You may be prompted for the host `sudo` password. The runner manages both host measurement and
guest execution, so you do not need to enter any separate run command in the guest.
When the run finishes, check the CSVs, graphs and summary at the printed `model results:` path.

## 2. Checking Results

The paths below are relative to `gemmini_experiments/power_modeling/single_app_power_modeling/`.

### Primary Result: `raw_power.csv`

The first file to check is `results/<run-id>/raw_power.csv`.
It stores CPU/Gemmini utilization, modeled power and total power in a single CSV,
and is updated every 100 ms by default during the experiment. Use this file for real-time power monitoring.

To view it from a separate host terminal during the run, replace the placeholder with the actual run ID.

```bash
tail -f "gemmini_experiments/power_modeling/single_app_power_modeling/results/<run-id>/raw_power.csv"
```

### Auto-generated Graphs and Time-scaled Results

After the experiment finishes, the script **automatically generates the raw graphs as well as the
time-scaled CSV, graphs and summary**. There is no need to run a graph script separately; everything is saved in `results/<run-id>/`.

With the current default settings, CPU phase time is post-scaled by **0.3035x** and Gemmini phase time
is post-scaled **based on the RTL cycle model and a 200 MHz frequency**. You can evaluate performance by checking
the estimated execution time and energy in the time-scaled graphs and
`time_scaled_<tag>_summary.json`. These are model-based estimates computed after the experiment finishes and differ from real-time performance measurement.

**If real-time performance measurement is needed, install the corresponding measurement tool
separately by following the user manual in the SSU contract report.**

| File | Contents |
|---|---|
| `raw_util_power.png` | Raw CPU/Gemmini utilization and CPU/Gemmini/total power |
| `raw_host_gemmini_util.png` | Pre-threshold Busy% of the host Gemmini worker |
| `raw_guest_vcpu_util.png` | Guest application main utilization |
| `raw_guest_executor_vcpu_util.png` | Guest Gemmini executor utilization |
| `raw_guest_housekeeping_util.png` | Guest housekeeping utilization |
| `time_scaled_<tag>_power.csv` | Time-scaled power CSV |
| `time_scaled_<tag>_summary.json` | Raw/scaled time and energy, and cycle model information |
| `time_scaled_<tag>_util_power.png` | Time-scaled utilization and CPU/Gemmini/total power |

With the default settings, `<tag>` is `rtl_200mhz_cpu_0p3035`. There are 6 graphs in total:
5 raw and 1 scaled.

### Run Records: `runs/<run-id>/`

Stores the run configuration and logs for error checking.

| File | Contents |
|---|---|
| `request.env` | Configuration of the run |
| `raw_bench.log` | Application execution log |
| `raw_turbostat.log` | Raw host utilization log |
| `raw_guest_proc_stat.csv` | Detailed guest vCPU measurements |
| `raw_power_sim.log` | Power collection log |
| `app_output/` | Application output |

`runs` and `results` are linked by the same run ID. Do not delete files in these folders while a run is in progress.
Reusing a run ID conflicts with the existing folder and aborts the run.

On failure, check the terminal error and `raw_bench.log`, `raw_power_sim.log` and
`raw_turbostat.log` for the same run ID.
