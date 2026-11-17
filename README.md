# P-Vela

P-Vela is a power and performance modeling platform for QEMU virtualization
environments, developed as part of the [Vela](https://github.com/riscv-vela/vela)
project. It estimates the **power consumption and execution time of a RISC-V
system — CPU cores and AI execution engines (Gemmini) — on the target board**,
without physical power sensors, by combining the Guest's (QEMU) resource
utilization with component power models and an RTL cycle model.

Power is modeled per component:

```
P_CPU    = θ × (α × CPU_util × f³ + β)          # per vCPU
P_AI     = P_active   while the engine computes # per AI execution engine
           P_idle     while idle but powered
           0          while power gated
P_System = Σ P_CPU + Σ P_AI
```

where `f` is the target CPU frequency, `CPU_util` is the guest CPU utilization,
`α`/`β` are regression coefficients (separate values for Boost ON/OFF), and `θ`
maps the host-calibrated model to the RISC-V board. `P_active`/`P_idle` come
from RTL synthesis of the AI execution engine. Execution time on the target
board is estimated with an RTL cycle model for the AI execution engine and a
time scaling factor for the CPU.

## Repository Structure

```
p-vela/
├── p-vela_v1.0/                    # v1.0: CPU power model (7zip workload)
│   ├── README.md                   # Full experiment guide (setup → measurement → logs)
│   ├── power_sim.py / power_sim_mem.py
│   └── log_powersim.sh / run_7z_top_powersim*.sh
└── p-vela_v2.0/                    # v2.0: CPU + AI execution engine power/performance model
    ├── README.md                   # Overview and quick start (Korean)
    ├── qemu_boot/                  # QEMU launch and thread pinning scripts
    └── gemmini_experiments/
        ├── applications/           # DistilBERT, DistilRoBERTa, DistilGPT2, PolyBench GEMM
        └── power_modeling/
            ├── power_sim/          # Power models, RTL cycle model, time scaling, unit tests
            ├── single_app_power_modeling/  # Single-application experiment runner
            ├── dual_app_power_modeling/    # Two-application experiment runner
            └── energy_eval/        # Energy and power-saving (power gating / DVFS) analysis
```

### What's new in v2.0

| | v1.0 | v2.0 |
|---|---|---|
| Measured components | CPU (+ memory) | CPU + AI execution engines |
| AI engine power | – | Active / idle / power-gated, from host Busy% and the q-vela power gating log |
| CPU utilization | Guest `mpstat` | Guest `/proc/stat` |
| Execution time | – | Target-board time via RTL cycle model and CPU time scaling |
| Workloads | 7zip benchmark | DistilBERT, DistilRoBERTa, DistilGPT2, PolyBench GEMM |
| Experiment flow | Manual host/guest scripts | Automated single-app / dual-app runners with plots and summaries |

## Getting Started

The measurement flow consists of three stages. See
[p-vela_v2.0/README.md](p-vela_v2.0/README.md) and the README in each folder for
the complete step-by-step guide. The v1.0 guide is in
[p-vela_v1.0/README.md](p-vela_v1.0/README.md).

**1. Set up the environment**
- Build QEMU with the AI execution engine device model and prepare an Ubuntu
  RISC-V guest image by following the
  **[q-vela Getting Started](https://github.com/riscv-vela/q-vela#getting-started)**
  — P-Vela runs on top of the q-vela emulation platform.
- Prepare the application models and input data
  ([applications/README.md](p-vela_v2.0/gemmini_experiments/applications/README.md)).

**2. Boot and pin the guest** (from `p-vela_v2.0/`)
- Boot QEMU with the AI execution engines: `bash qemu_boot/run_qemu.sh 1 3`
- Pin the vCPU and AI engine worker threads to host cores: `bash qemu_boot/pin.sh 8 2`

**3. Run the measurement**
- Host: run the single-application runner, which starts the host collector
  (turbostat) and the guest-side power meter and workload:
  ```bash
  bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
      distilbert --sample-limit 1
  ```

After the run, the following files are produced:

| File | Location | Description |
|------|----------|-------------|
| `raw_power.csv` | `results/<run-id>/` | **Real-time CPU / AI engine / total power (main result)** |
| `time_scaled_<tag>_power.csv` | `results/<run-id>/` | Power trace on the target-board time axis |
| `time_scaled_<tag>_summary.json` | `results/<run-id>/` | Estimated execution time and energy, cycle model details |
| `raw_util_power.png`, `time_scaled_<tag>_util_power.png` | `results/<run-id>/` | Utilization and power plots |
| `raw_bench.log`, `raw_turbostat.log`, `raw_guest_proc_stat.csv` | `runs/<run-id>/` | Application log, host and guest utilization logs |
