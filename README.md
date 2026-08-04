# P-Vela

P-Vela is a power and performance modeling platform for QEMU virtualization
environments, developed as part of the [Vela](https://github.com/riscv-vela/vela)
project. It estimates the **real-time power consumption of a RISC-V guest**
without physical power sensors, by combining the Host PC's CPU frequency and
power characteristics with the Guest's (QEMU) resource utilization.

Dynamic power per core is modeled as:

```
P_Dynamic = α × CPU_util × f³ + β
P_System  = P_Dynamic(Core) + P_mem
```

where `f` is the host core frequency, `CPU_util` is the guest CPU utilization,
and `α`/`β` are regression coefficients calibrated per platform (separate
values for Boost ON/OFF).

## Repository Structure

```
p-vela/
└── Power Model_v1/            # Power simulation model v1.0
    ├── README.md              # Full experiment guide (setup → measurement → logs)
    ├── power_sim.py           # Real-time power simulator (CPU model)
    ├── power_sim_mem.py       # Power simulator with memory model
    ├── log_powersim.sh        # Host-side monitoring script (turbostat logging)
    ├── run_7z_top_powersim.sh # Guest-side integrated workload script (7zip benchmark)
    └── run_7z_top_powersim_mem.sh
```

## Getting Started

The measurement flow consists of three stages. See
[Power Model_v1/README.md](Power%20Model_v1/README.md) for the complete
step-by-step guide.

**1. Set up the environment**
- Build QEMU and prepare an Ubuntu RISC-V guest image by following the
  **[q-vela Getting Started](https://github.com/riscv-vela/q-vela#getting-started)**
  — P-Vela runs on top of the q-vela emulation platform.
- Launch the guest with a 9p shared folder, then mount it inside the guest
  (`mount -t 9p ... hostshare /mnt/host`) — scripts and logs are exchanged
  through this folder.

**2. Prepare the host for accurate measurement**
- Isolate host cores from the scheduler (`isolcpus`) and fix their frequency
  (`cpupower frequency-set`).
- Pin each QEMU vCPU thread to an isolated host core 1:1 (`taskset`).

**3. Run the measurement**
- Host: run `log_powersim.sh` to log per-core frequency/utilization/power via turbostat.
- Guest: run `run_7z_top_powersim.sh`, which executes the 7zip benchmark and
  automatically starts `power_sim.py` to produce real-time power estimates.

After the run, the following logs are produced in the shared folder:

| File | Produced by | Description |
|------|-------------|-------------|
| `turbostat.csv` | Host | Measured CPU utilization, frequency, CorWatt/PkgWatt |
| `sim_power.csv` | Guest | **Simulated real-time power estimates (main result)** |
| `*_meta.log`, `*_mpstat.log`, `*_top.log` | Guest | Benchmark timestamps, per-core utilization, memory status |
