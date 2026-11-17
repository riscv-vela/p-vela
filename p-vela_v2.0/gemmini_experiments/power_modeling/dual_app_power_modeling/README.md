# Dual-app Power Modeling

## 0. Prerequisites

| Item | Requirement |
|---|---|
| Application | Prepare the model/input data and build environment for the two applications as described in [Applications](../../applications/README.md) |
| QEMU | Boot with `2 Gemmini / 5 vCPU` and the `modulo` policy as described in [QEMU Boot](../../../qemu_boot/README.md) |
| Pinning | QEMU thread pinning completed on CPUs available on the host |
| Host | `sudo turbostat`, Python 3 and `matplotlib` available |
| Guest | Python 3, `timedatectl`, and a `sudo` environment that can run the required commands without a password prompt |
| SSH | Default `ubuntu@127.0.0.1:2200`, key `q-vela/tmp-cloudinit/id_ed25519` |
| Shared folder | Host `gemmini_experiments/` is mounted at guest `/mnt/hostshare` and is writable for results |

By default, the dual runner builds the binaries for both applications before running.
Model/input data is not generated automatically, so it must be prepared in advance.
To use binaries that are already built, specify `--build 0`.

The default pinning example `pin.sh 8 2` uses host CPUs 8-12 and 2-3.
On a host that only has CPUs 0-11, lower the starting number after QEMU boots, as shown below.
Confirm `[OK]` for each thread; by default the runner reads and uses the actual pinning result.

```bash
bash qemu_boot/pin.sh 7 2
```

## 1. How to Run

Run all commands from the **workspace root on the host**, in a terminal separate from QEMU.
The run script is [run_dual_app_power_exp.sh](./experiment_scripts/run_dual_app_power_exp.sh).

### Usage and Arguments

```text
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh --apps APP0,APP1 [OPTIONS]
```

For `APP0` and `APP1`, choose two different apps from `distilbert`, `distilroberta`, `distilgpt2`, and `polybench_gemm`.
The `--app0 APP0 --app1 APP1` form can also be used.

| Option | Default | Purpose |
|---|---:|---|
| `--apps APP0,APP1` | Required | The two applications to run concurrently |
| `--sample-limit N` | `1` | Number of input samples for each Transformer |
| `--threads N` | `1` | Number of threads per application; currently only `1` is supported |
| `--dataset MEDIUM` | `MEDIUM` | PolyBench dataset; time scaling currently supports only `MEDIUM` |
| `--reps N` | `50` | Number of PolyBench GEMM repetitions |
| `--build 0\|1` | `1` | `1`: build before running, `0`: use existing binaries |
| `--sys-scale FLOAT` | `0.5625` | CPU power model calibration factor |
| `--gemmini-full-util-power-w FLOAT` | `0.862` | Active-state power of one Gemmini (W) |
| `--gemmini-util-threshold-pct FLOAT` | `80` | Busy% threshold for judging Gemmini as active |

Common defaults for the power and time models are managed in
[power_model_defaults.env](../power_sim/power_model_defaults.env).
Command-line options, when specified, take precedence for that run, and configuration changes apply from the next experiment.

Frequency and time scaling factors are changed with the environment variables below. Unlike the single runner,
the `--gemmini-frequency-mhz` and `--cpu-time-scale` command-line options are not used.

| Environment variable | Default | Purpose |
|---|---:|---|
| `GEMMINI_FREQUENCY_MHZ` | `200` | Reference frequency (MHz) for Gemmini time scaling |
| `CPU_TIME_SCALE` | `0.3035` | Time scaling factor for CPU phases |
| `FIXED_FREQ_GHZ` | `2.512` | Frequency (GHz) used by the CPU power model |

### Examples

Run DistilBERT and DistilRoBERTa with 1 sample each:

```bash
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh \
    --apps distilbert,distilroberta --sample-limit 1 --threads 1
```

Run DistilGPT2 with 1 sample and PolyBench GEMM 50 times:

```bash
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh \
    --apps distilgpt2,polybench_gemm --sample-limit 1 --threads 1 \
    --dataset MEDIUM --reps 50
```

Run with explicit time scaling settings using existing binaries:

```bash
GEMMINI_FREQUENCY_MHZ=200 CPU_TIME_SCALE=0.3035 \
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh \
    --apps distilbert,distilroberta --sample-limit 1 --build 0
```

You may be prompted for the host `sudo` password. You do not need to enter any separate run command in the guest.
When the run finishes, check the results at the printed `Run dir` path.

## 2. Checking Results

Results and run logs are saved together in the folder below. A subfolder is created automatically for each run,
and this log directory is excluded from Git tracking.

```text
gemmini_experiments/power_modeling/dual_app_power_modeling/logs/<app0>_<app1>/<run-id>/
```

### Primary Result: Power CSV

Unlike the single experiment, the dual experiment currently **does not save results to a single `raw_power.csv`.**
Use the following two files for real-time power monitoring. The default recording interval is 100 ms.

| File | Contents |
|---|---|
| `vcpu_power.csv` | Main vCPU utilization of the two applications and modeled CPU power |
| `gemmini_power.csv` | Utilization of the two Gemmini workers and modeled Gemmini power |

To view them from a separate host terminal during the run, replace the path below with the actual `Run dir`.

```bash
tail -f \
    "gemmini_experiments/power_modeling/dual_app_power_modeling/logs/<app0>_<app1>/<run-id>/vcpu_power.csv" \
    "gemmini_experiments/power_modeling/dual_app_power_modeling/logs/<app0>_<app1>/<run-id>/gemmini_power.csv"
```

The total in each CSV is the sum of that component. The overall power combining CPU and Gemmini can be checked
in the auto-generated combined graphs and the time-scaled power CSV.

### Auto-generated Graphs and Time-scaled Results

After the experiment finishes, the script automatically generates the raw graphs and the time-scaled CSV, graphs and summary.
There is no need to run a graph script separately.

With the current default settings, for each application, CPU phase time is post-scaled by **0.3035x** and Gemmini phase time
is post-scaled **based on the RTL cycle model and a 200 MHz frequency**. You can evaluate performance by checking
the estimated per-application execution time and energy in the time-scaled graphs and
`time_scaled_<tag>_summary.json`. These are model-based estimates computed after the experiment finishes.

If real-time performance measurement is needed, **install the corresponding measurement tool
separately by following the user manual in the SSU contract report.**

| File | Contents |
|---|---|
| `raw_combined_power.png` | Raw combined utilization/power graph for the two applications |
| `raw_combined_power_<app>.png` | Per-application raw utilization/power graph |
| `time_scaled_<tag>_power.csv` | Time-scaled power CSV for the two applications, with total power |
| `time_scaled_<tag>_summary.json` | Per-application raw/scaled time and energy results |
| `time_scaled_<tag>_combined_power.png` | Time-scaled combined utilization/power graph |
| `time_scaled_<tag>_combined_power_<app>.png` | Per-application time-scaled graph |

The default value of `<tag>` is `rtl_200mhz_cpu_0p3035`.
For each combined graph, utilization graphs with the same file name prefix and the suffixes below are also generated.

| Suffix | Contents |
|---|---|
| `_guest_vcpu_util.png`, `_host_vcpu_util.png` | Guest/host utilization of the application main thread |
| `_guest_executor_vcpu_util.png`, `_host_executor_vcpu_util.png` | Guest/host utilization of the Gemmini executor |
| `_guest_housekeeping_util.png`, `_host_housekeeping_util.png` | Guest/host utilization of housekeeping |

There are 18 graphs in total: 9 raw and 9 scaled.

### Run Logs

| File | Contents |
|---|---|
| `raw_bench.log` | Run configuration and application execution log |
| `raw_turbostat.log` | Raw host utilization log |
| `raw_guest_proc_stat.csv` | Detailed guest vCPU measurements |
| `raw_gemmini_power_sim.log` | Gemmini power collection log |
| `qemu_thread_affinity_summary.csv` | CPU placement of QEMU threads during the run |
| `qemu_gemmini_mapping_summary.txt` | Mapping status between applications and Gemmini |

On failure, check the terminal error and the logs in the corresponding run folder.
Do not delete files in the result folder while a run is in progress.
