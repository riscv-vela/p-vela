# Power Modeling

Run applications in QEMU while monitoring CPU and Gemmini power,
then check the graphs and time-scaling results after the experiment ends.
This document walks through preparation, configuration, and execution based on the single-application experiment.

## Documentation Guide

| Topic | Document |
|---|---|
| Power model and time-scaling settings | [Power Simulation](./power_sim/README.md) |
| Application run commands, arguments, and checking results | [Single-app Power Modeling](./single_app_power_modeling/README.md) |

## 0. Experiment Preparation

1. In [Applications](../applications/README.md), prepare the model and input data for the application to run,
   along with the Gemmini power binary.
2. Following [QEMU Boot](../../qemu_boot/README.md), boot QEMU with `1 Gemmini / 3 vCPU` and
   verify CPU pinning, SSH, and the shared folder connection.

Supported applications are `distilbert`, `distilroberta`, `distilgpt2`, and `polybench_gemm`.

## 1. Power Simulation Configuration

Common defaults for the power and time models are edited in
[power_sim/power_model_defaults.env](./power_sim/power_model_defaults.env).

- CPU frequency and power calibration coefficients
- Gemmini active/idle power and utilization threshold
- Sampling interval
- Gemmini time-scaling reference frequency and CPU time scale factor
- Graph display settings

Values specified as experiment script options take precedence over the defaults, and configuration changes
apply from the next experiment. For per-option usage, see the single-app README; for descriptions of the common settings, see the
[power_sim README](./power_sim/README.md).
For normal experiments, there is no need to run the power_sim Python files separately.

## 2. Running a Single Application and Checking Results

Following the instructions in the [Single-app README](./single_app_power_modeling/README.md),
run `run_single_app_power.sh` on the host.
This script handles everything from measurement to generating the graphs and time-scaling results.

The first result to check is **`raw_power.csv`**. CPU/Gemmini utilization,
modeled power, and total power are recorded to a single file during the experiment.

| Purpose | Result |
|---|---|
| Real-time power monitoring | `raw_power.csv` |
| Checking power changes after the experiment | Automatically generated raw graphs |
| Checking estimated execution time and energy | Time-scaled CSV, graphs, and summary |

Model results are saved to `single_app_power_modeling/results/<run-id>/`, and run logs to
`single_app_power_modeling/runs/<run-id>/`.
For per-file descriptions and commands to inspect them, see the single-app README.

Time-scaling results are performance estimates checked after the experiment ends. If real-time performance
measurement is needed, install the corresponding measurement tool separately following the user manual in the SSU contract report.

## Additional Support: Dual Application

Experiments that run two different applications simultaneously are also supported.
Since the QEMU setup, run commands, and result file layout differ from single-app,
see the [Dual-app README](./dual_app_power_modeling/README.md).
