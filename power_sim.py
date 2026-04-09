#!/usr/bin/env python3
import time
import os

# ====================================================
# Shared directory (host <-> guest)
# ====================================================
SHARED_DIR = "/mnt/host/4_core/7z"
TURBO_LOG  = os.path.join(SHARED_DIR, "turbostat.csv")
OUT_LOG    = os.path.join(SHARED_DIR, "sim_power.csv")

# ====================================================
# System Power Model Parameters (AMD, FINAL)
# ====================================================
BOOST_OFF_TH = 2.6   # GHz_boost OFF

# CPU coefficients
a_boost = 0.039747769
k_boost = 2.80057

a_nom   = 0.086391591
k_nom   = 0.184415

# Guest cores
GUEST_CORES = [0, 1, 2, 3]

INTERVAL = 1.0   # seconds

# AMD scaling factor (platform-level)
SETA_SYS = 1.0  # AMD platform scaling

# Memory coefficient
BETA_MEM = 7.811760

# ====================================================
# Boost state 판단
# ====================================================
def cpu_power(freq_ghz, util):
    """
    freq_ghz : float (GHz)
    util     : 0~1
    """
    f3 = freq_ghz ** 3
    if freq_ghz <= BOOST_OFF_TH:
        return a_nom * util * f3 + k_nom
    else:
        return a_boost * util * f3 + k_boost


# ====================================================
# Read latest frequency from turbostat log (host)
# ====================================================
def read_latest_freq():
    try:
        with open(TURBO_LOG, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    for line in reversed(lines):
        if line.startswith("Timestamp"):
            continue
        parts = line.strip().split(",")
        try:
            mhz = float(parts[4])      # Avg_MHz
            return mhz / 1000.0        # MHz → GHz
        except:
            continue
    return None


# ====================================================
# CPU utilization from /proc/stat
# ====================================================
def read_proc_stat():
    stats = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                parts = line.split()
                stats[parts[0]] = list(map(int, parts[1:]))
    return stats


def calc_utils(prev, cur):
    utils = []
    for c in GUEST_CORES:
        key = f"cpu{c}"
        p = prev[key]
        n = cur[key]

        prev_total = sum(p)
        cur_total  = sum(n)

        prev_idle = p[3] + p[4]
        cur_idle  = n[3] + n[4]

        dt = cur_total - prev_total
        didle = cur_idle - prev_idle

        util = 0.0 if dt <= 0 else (dt - didle) / dt
        utils.append(max(0.0, min(1.0, util)))

    return utils


# ====================================================
# Memory utilization from /proc/meminfo
# ====================================================
def read_mem_util():
    mem_total = mem_avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                mem_total = int(line.split()[1])
            elif line.startswith("MemAvailable"):
                mem_avail = int(line.split()[1])

    if not mem_total or mem_avail is None:
        return 0.0

    return max(0.0, min(1.0, 1.0 - mem_avail / mem_total))


# ====================================================
# MAIN LOOP
# ====================================================
def main():
    print("[power_sim] System power simulator started (AMD)")

    prev = read_proc_stat()

    # Write header once
    if not os.path.exists(OUT_LOG):
        with open(OUT_LOG, "w") as f:
            f.write(
                "epoch_ns,freq_ghz,mem," +
                ",".join([f"util_{c}" for c in GUEST_CORES]) + "," +
                ",".join([f"power_{c}" for c in GUEST_CORES]) +
                ",p_system_w\n"
            )

    while True:
        time.sleep(INTERVAL)

        freq = read_latest_freq()
        if freq is None:
            continue

        cur = read_proc_stat()
        utils = calc_utils(prev, cur)
        prev = cur

        mem = read_mem_util()


        # raw per-core power 
        per_core_p_raw = [cpu_power(freq, u) for u in utils]
        per_core_p = [SETA_SYS * p for p in per_core_p_raw]
        p_mem = SETA_SYS * (BETA_MEM * mem)

        # final system power
        p_total = sum(per_core_p) + p_mem


        with open(OUT_LOG, "a") as f:
            f.write(
                f"{time.time_ns()},{freq:.4f},{mem:.4f}," +
                ",".join(f"{u:.4f}" for u in utils) + "," +
                ",".join(f"{p:.4f}" for p in per_core_p) +
                f",{p_total:.4f}\n"
            )


if __name__ == "__main__":
    main()
