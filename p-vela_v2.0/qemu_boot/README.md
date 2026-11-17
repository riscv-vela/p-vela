# QEMU Boot

## Quick Start

In an environment where the QEMU executable and guest image are ready, run from the **workspace root on the host**.
Choose only one of single or dual.

**Host terminal 1 — boot**

Single-app:

```bash
bash qemu_boot/run_qemu.sh 1 3
```

Dual-app:

```bash
QEMU_GEMMINI_POLICY=modulo bash qemu_boot/run_qemu.sh 2 5
```

**Host terminal 2 — pinning after the guest has finished booting (same for single/dual)**

```bash
bash qemu_boot/pin.sh 8 2
```

After confirming `[OK]` for each thread,
proceed with the experiment following the instructions in [Power Modeling](../gemmini_experiments/power_modeling/README.md).
Preparation for a first-time setup and how to verify SSH and the shared folder are described below.

## Detailed Guide

This section covers booting the QEMU RISC-V guest and pinning host CPUs as required for power modeling.
[run_qemu.sh](./run_qemu.sh) configures the Gemmini devices and guest vCPUs, and
[pin.sh](./pin.sh) places the QEMU threads on the specified host CPUs.

Except where marked as guest commands, run all commands from the **workspace root on the host**.

## 0. Boot Preparation

Clone the q-vela repository into `./q-vela/` under the workspace root.

Packages required for booting on Ubuntu 24.04:

```bash
sudo apt update
sudo apt install -y opensbi u-boot-qemu qemu-system-misc
```

The files used by `run_qemu.sh` are as follows.

| File | Role |
|---|---|
| `q-vela/qemu/build/qemu-system-riscv64` | QEMU executable with Gemmini device support |
| `ubuntu-24.04.3-preinstalled-server-riscv64.img` | Guest disk image in the workspace root |
| `/usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.bin` | OpenSBI firmware |
| `/usr/lib/u-boot/qemu-riscv64_smode/uboot.elf` | U-Boot |
| `q-vela/tmp-cloudinit/id_ed25519` | Guest SSH key used by the power runner |

Installing the packages alone does not provide the Gemmini-enabled QEMU or the guest image.
If `q-vela` and the image are symbolic links, their actual targets must exist.
The guest must have an SSH server and an account that can be accessed with that key.
The current default guest account for the power runner is `ubuntu`.

## 1. Boot for the Experiment Configuration

Choose **only one** of the two configurations below, matching the experiment you will run.
Do not run QEMU instances that use the same disk image and SSH port at the same time.

### Single-app

Boot with 1 Gemmini and 3 vCPUs.

```bash
bash qemu_boot/run_qemu.sh 1 3
```

### Dual-app

Boot with 2 Gemmini and 5 vCPUs, and use `modulo` to assign a Gemmini per executor.

```bash
QEMU_GEMMINI_POLICY=modulo bash qemu_boot/run_qemu.sh 2 5
```

General usage is `bash qemu_boot/run_qemu.sh [num_gemmini] [num_vcpu]`.
The arguments are in the order **number of Gemmini → number of vCPUs**, and default to `2 2` if omitted.
The current script allows 1–4 Gemmini and 1–12 vCPUs.
For power experiments, do not omit the arguments; use the configurations above.

The default binding policy is `idle_first`; with `modulo`, guest `vCPU i` uses
`Gemmini (i % num_gemmini)`. In dual, executor vCPUs 3 and 4 are connected to
Gemmini 1 and 0, respectively.

QEMU runs in the current terminal. Run subsequent host commands in another terminal.
The QEMU log is saved to `q-vela/qemu/qemu-gemmini.log`, and the previous log is cleared on each new boot.

## 2. Host CPU Pinning

Run this in another host terminal after the guest has booted. The current defaults are the same for single and dual.

```bash
bash qemu_boot/pin.sh 8 2
```

The first argument is the starting host CPU number for the vCPUs, and the second is the starting number for the Gemmini workers.
Here, CPU numbers are the host logical CPU IDs used by `taskset`.

| Target | Placement rule | `8 2` example |
|---|---|---|
| QEMU main and auxiliary threads | vCPU start CPU | CPU 8 |
| Guest vCPU i | vCPU start CPU + i | Single: 8~10, dual: 8~12 |
| Gemmini worker i | Gemmini start CPU + i | Single: 2, dual: 2~3 |

Check for `[OK]` for each thread in the output. The specified CPUs must be available on the host,
and when changing the CPU placement, the power runner's measurement targets must be updated accordingly.
Thread IDs change when QEMU is rebooted, so run pinning again.

To specify a separate QEMU log, use the following form.

```bash
bash qemu_boot/pin.sh q-vela/qemu/qemu-gemmini.log 8 2
```

## 3. Verify the Shared Folder and SSH

The host's `gemmini_experiments/` must be accessible in the guest at `/mnt/hostshare`.
The boot script provides a 9p shared device, but it must be mounted separately inside the guest.

Verify in a guest terminal.

```bash
mountpoint /mnt/hostshare
ls /mnt/hostshare/applications /mnt/hostshare/power_modeling
```

If it is not mounted yet, run the following in the guest.

```bash
sudo mkdir -p /mnt/hostshare
sudo mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/hostshare
```

Host TCP port 2200 is forwarded to the guest's SSH port 22.
From a host terminal, verify access using the same key as the power runner.

```bash
ssh -i q-vela/tmp-cloudinit/id_ed25519 -p 2200 ubuntu@127.0.0.1
```

It must be possible to read application files from and write experiment files to the shared folder.
Once boot, pinning, SSH, and shared-folder preparation are complete, move on to the experiment.

- [Power Modeling](../gemmini_experiments/power_modeling/README.md)
