# p-vela
# RISC-V QEMU Power Simulation System

This system estimates real-time power consumption by combining the Host PC's CPU frequency with the Guest (QEMU) resource utilization.

---

## Directory Structure

```
~/qemu/               # QEMU source tree
~/qemu/build/         # Build directory
~/qemu_workspace/qemu_share/   # Host–Guest shared folder
```

---

## Prerequisites

### 1. Download RISC-V Ubuntu Image

Download the RISC-V OS image to the working directory (`~/qemu_workspace`):

```bash
# Download Ubuntu 24.04 RISC-V 64-bit server image
wget http://cdimage.ubuntu.com/releases/24.04/release/ubuntu-24.04.3-preinstalled-server-riscv64.img.xz

# Decompress the image
unxz ubuntu-24.04.3-preinstalled-server-riscv64.img.xz
```

### 2. Build and Install QEMU

```bash
# Install required dependencies
sudo apt install python3-ninja python3-pip libglib2.0-dev libpixman-1-dev -y

# Clone QEMU source
git clone https://github.com/qemu/qemu.git

# Create build directory and configure
cd ~/qemu
mkdir build && cd build
../configure --target-list=riscv64-softmmu

# Compile
ninja -j$(nproc)

# Verify installation (should print version)
qemu-system-riscv64 --version
```

### 3. Create Host Shared Folder

```bash
mkdir -p ~/qemu_workspace/qemu_share
```

---

## Step 1. Launch QEMU (RISC-V Environment)

```bash
qemu-system-riscv64 \
  -M virt,accel=tcg \
  -cpu rv64 \
  -smp 4 \
  -m 4G \
  -nographic \
  -kernel /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf \
  -drive file=ubuntu-24.04.3-preinstalled-server-riscv64.img,format=raw,if=virtio \
  -netdev user,id=net0,hostfwd=tcp::2200-:22 \
  -device virtio-net-device,netdev=net0 \
  -fsdev local,id=myfs,path=~/qemu_workspace/qemu_share,security_model=none \
  -device virtio-9p-device,fsdev=myfs,mount_tag=hostshare
```

> **Note:** The `-smp` value sets the number of virtual CPU cores.  
> If you change `-smp`, you must also update `GUEST_CORES` in `power_sim.py` and `CPU_ID` in `log_powersim.sh` to match.  
> Example: `-smp 2` → `GUEST_CORES = [0, 1]`

---

## Step 2. Mount Shared Folder (Inside Guest)

After the Guest OS boots, run the following inside the Guest:

```bash
# Create mount point
mkdir /mnt/host

# Mount via 9p protocol
mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host

# Verify (files from the Host should appear)
ls /mnt/host
```

---

## Step 3. Host CPU Core Isolation & Frequency Pinning

To ensure accurate power measurement, isolate Host CPU cores from the OS scheduler and fix their frequency.

### 3-1. Core Isolation (GRUB)

```bash
# Edit GRUB config
sudo vim /etc/default/grub

# Add isolcpus to GRUB_CMDLINE_LINUX_DEFAULT
# Example: isolating cores 2, 3, 4, 5
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash isolcpus=2,3,4,5"

# Apply and reboot
sudo update-grub
sudo reboot

# Verify
cat /proc/cmdline
```

### 3-2. Change CPU Driver & Fix Frequency

```bash
# Edit GRUB config again to add amd_pstate=passive
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash isolcpus=2,3,4,5 amd_pstate=passive"

sudo update-grub
sudo reboot

# Verify driver
cpupower frequency-info | grep "driver"

# Set governor to userspace
sudo cpupower -c 2,3,4,5 frequency-set --governor userspace

# Check available frequencies
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies

# Fix frequency (example: 3.0 GHz)
sudo cpupower -c 2,3,4,5 frequency-set -d 3000000 -u 3000000

# Verify all target cores are fixed at 3000 MHz
sudo turbostat --cpu 2,3,4,5 --interval 0.1
```

---

## Step 4. Host–Guest CPU Core Pinning (1:1 Mapping)

Pin each QEMU vCPU thread to a specific Host physical core.

```bash
# Find QEMU thread list
ps -eLo pid,tid,comm | grep qemu

# Check thread roles (threads with high TIME are vCPU threads)
ps -T -p [MAIN_PID]

# Pin each vCPU thread to a Host core
sudo taskset -cp 2 [TID_1]   # vCPU 0 → Host Core 2
sudo taskset -cp 3 [TID_2]   # vCPU 1 → Host Core 3
sudo taskset -cp 4 [TID_3]   # vCPU 2 → Host Core 4
sudo taskset -cp 5 [TID_4]   # vCPU 3 → Host Core 5

# Verify pinning (PSR values should match assigned core numbers)
ps -eLo pid,tid,psr,comm | grep qemu
```

---

## Step 5. Run Host Monitoring Script (`log_powersim.sh`)

```bash
# Navigate to shared directory
cd ~/qemu_workspace/qemu_share

# Verify CPU_ID setting in the script matches your isolated cores
# CPU_ID=2,3,4,5

# Grant execute permission
chmod +x log_powersim.sh

# Run on core 1 to avoid interfering with measurement cores
sudo taskset -c 1 ./log_powersim.sh
```

Script configuration (`log_powersim.sh`):

```bash
#setup
CPU_ID=2,3,4,5
INTERVAL=1    #second
OUTPUT_FILE="turbostat.csv"
```

---

## Step 6. Run Integrated Workload Script (Inside Guest)

```bash
# Navigate to mount path
cd /mnt/host

# Grant execute permission
sudo chmod +x /mnt/host/run_7z_top_powersim.sh

# Install benchmark tools (first time only)
sudo apt install 7zip sysstat -y

# Run workload
sudo ./run_7z_top_powersim.sh
```

> When the script runs, `power_sim.py` starts automatically in the background and logs real-time power estimates to `sim_power.csv`.

> To measure power with a different application instead of 7zip, replace the `7z b` portion in the script with the desired command.

---

## Step 7. Power Simulator Configuration (`power_sim.py`)

Key parameters to configure before running:

```python
# =====================================================
# System Power Model Parameters (AMD, FINAL)
# =====================================================
BOOST_OFF_TH = 2.6       # Boost OFF threshold in GHz (AMD: 2.6 / Intel: 3.5)

# CPU power model coefficients
a_boost = 0.039747769    # Boost ON: alpha
k_boost = 2.80057        # Boost ON: beta

a_nom   = 0.086391591    # Boost OFF: alpha
k_nom   = 0.184415       # Boost OFF: beta

# Guest core list (must match -smp count)
GUEST_CORES = [0, 1, 2, 3]

INTERVAL = 1.0           # Sampling interval in seconds

# Platform-level scaling factor
# AMD baseline = 1.0 / Intel example = 1.2947
SETA_SYS = 1.0
```

| Parameter | Description |
|---|---|
| `SHARED_DIR` | Path to the shared log directory |
| `TURBO_LOG` | Path to read `turbostat_100ms.csv` generated by the Host |
| `BOOST_OFF_TH` | Boost mode threshold (AMD: 2.6 GHz / Intel: 3.5 GHz) |
| `GUEST_CORES` | List of virtual cores to simulate (must match `-smp`) |
| `SETA_SYS` | Power scaling factor per platform (AMD=1.0, Intel=1.2947) |

---

## Step 8. Terminate Experiment

- **Guest:** Background processes (`power_sim.py`, `mpstat`, `top`) are automatically terminated by the `kill` commands inside `run_7z_top_powersim.sh`.
- **Host:** Press `Ctrl+C` in the Host terminal to stop `log_powersim.sh` manually.

After termination, verify that log files are saved in `~/qemu_workspace/qemu_share/`.

---

## Output Log Files

| Location | File | Description |
|---|---|---|
| Host | `turbostat.csv` | CPU utilization, frequency, CorWatt, PkgWatt |
| Guest | `sim_power.csv` | Real-time simulated transient power log |
| Guest | `*_meta.log` | Benchmark start (`[START]`) and end (`[END]`) timestamps |
| Guest | `*_mpstat.log` | Per-core CPU utilization inside QEMU |
| Guest | `*_top.log` | Memory utilization and process status inside QEMU |

### Key Fields per File

**`turbostat.csv`**
- `Avg_MHz`, `Busy%`, `Bzy_MHz`, `CorWatt`, `PkgWatt`

**`sim_power.csv`**
- `epoch_ns`, `freq_ghz`, `mem`, `util_0~3`, `power_0~3`, `p_system_w`

**`*_meta.log`**
- `[START]` / `[END]` timestamps

**`*_mpstat.log`**
- `%usr`, `%sys`, `%idle` (per core)

**`*_top.log`**
- `%CPU`, `%MEM`, `VIRT`, `RES`

---

## Power Model Overview

Dynamic power per core is modeled as:

```
P_Dynamic = α × CPU_util × f³ + β
```

Where:
- `f` = CPU operating frequency (GHz)
- `CPU_util` = Guest CPU utilization (0–1)
- `α`, `β` = regression coefficients (separate values for Boost ON / OFF)

System-level total power:

```
P_System = P_Dynamic(Core) + P_mem
P_mem    = P_min + (P_max - P_min) × MEM_util
```
