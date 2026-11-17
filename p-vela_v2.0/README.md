# Real-Time Performance and Power Modeling in a QEMU Environment

## Project Overview

To quantitatively evaluate the system SW of the project "Development of RISC-V-based System
Software Technology for Building an Open SDR Ecosystem", this project develops a performance
and power measurement model for a system that includes a RISC-V CPU and an AI execution engine
(Gemmini) in a QEMU emulation environment. The goal is to extend the existing CPU power model
to the AI execution engine and to support application execution-time estimation and real-time
power consumption monitoring, so that the performance and power improvements of the system SW
can be evaluated.

## Documentation Guide

| Topic | Document |
|---|---|
| QEMU boot, shared folder, SSH, thread pinning | [QEMU Boot](./qemu_boot/README.md) |
| Application setup and preparation | [Applications](./gemmini_experiments/applications/README.md) |
| Power model configuration, running experiments, and checking results | [Power Modeling](./gemmini_experiments/power_modeling/README.md) |
| Power equations, RTL cycle model, time scaling | [Power Simulation](./gemmini_experiments/power_modeling/power_sim/README.md) |

## Supported Applications

| Application | Purpose |
|---|---|
| [DistilBERT](./gemmini_experiments/applications/distilbert/) | SST-2 text classification |
| [DistilRoBERTa](./gemmini_experiments/applications/distilroberta/) | SST-2 text classification |
| [DistilGPT2](./gemmini_experiments/applications/distilgpt2/) | SST-2 text classification, WikiText language modeling |
| [PolyBench GEMM](./gemmini_experiments/applications/polybench_gemm/) | Matrix multiplication benchmark |

The DistilGPT2 execution path in the current workspace is based on SST-2; the WikiText pipeline is not included.

## Experiment Workflow

0. In [Applications](./gemmini_experiments/applications/), prepare the application to run along with the required model and input data.
1. Boot QEMU following the instructions in [QEMU Boot](./qemu_boot/).
2. Pin the QEMU threads to host CPUs following the pinning instructions in [QEMU Boot](./qemu_boot/).
3. Run the experiment and check the results following the instructions in [Power Modeling](./gemmini_experiments/power_modeling/README.md).

For detailed preparation steps, run commands, and descriptions of the results, see the README in each folder.

### Quick Start: Single Application Example

#### 0. Prerequisites

Prepare the DistilBERT model and input data, the power binary, the QEMU executable and guest image,
and an environment for SSH and the shared folder. Run all commands below from the **workspace root on the host**.

#### 1. Boot QEMU — host terminal 1

```bash
bash qemu_boot/run_qemu.sh 1 3
```

#### 2. Pinning — host terminal 2, after the guest has finished booting

```bash
bash qemu_boot/pin.sh 8 2
```

#### 3. Run DistilBERT — host terminal 2

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    distilbert --sample-limit 1
```

When the run finishes, check `raw_power.csv` and the automatically generated graphs and summary
under the `model results:` path printed at the end. For detailed options and descriptions of the results, see [Power Modeling](./gemmini_experiments/power_modeling/README.md).
