#!/usr/bin/env python3

"""Sample guest vCPU utilization and optionally model combined system power."""

import argparse
import csv
import os
import signal
import threading
import time

from power_model_config import number

from power_sim_gemmini import (
    DEFAULT_GEMMINI_FULL_UTIL_POWER_W,
    DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT,
    append_sample,
    parse_latest_complete_block,
    write_header_once,
)


RAW_HEADER = [
    "epoch_ns",
    "sample_tod_s",
    "monotonic_ns",
    "interval_s",
    "model_core",
    "guest_vcpu",
    "delta_total",
    "delta_idle",
    "busy_percent",
]
STOP_REQUESTED = threading.Event()


def parse_vcpu_map(value):
    try:
        model_core_raw, guest_vcpu_raw = value.split(":", 1)
        model_core = int(model_core_raw)
        guest_vcpu = int(guest_vcpu_raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected MODEL_CORE:GUEST_VCPU, got {value!r}"
        ) from exc
    if model_core < 0 or guest_vcpu < 0:
        raise argparse.ArgumentTypeError("core and vCPU IDs must be non-negative")
    return model_core, guest_vcpu


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Guest vCPU power simulator using /proc/stat deltas. When a "
            "shared host turbostat log is provided, write combined vCPU and "
            "Gemmini power to one real-time CSV."
        )
    )
    parser.add_argument("--out-log", required=True, help="power CSV output")
    parser.add_argument("--raw-log", required=True, help="raw /proc/stat delta CSV output")
    parser.add_argument(
        "--turbostat",
        help=(
            "shared host turbostat log; enables combined vCPU/Gemmini power "
            "output"
        ),
    )
    parser.add_argument(
        "--gemmini-cores",
        type=int,
        nargs="+",
        default=[],
        help="host cores used by Gemmini worker threads",
    )
    parser.add_argument(
        "--vcpu-map",
        type=parse_vcpu_map,
        action="append",
        required=True,
        metavar="MODEL_CORE:GUEST_VCPU",
        help="map the plot/model core slot to a guest vCPU",
    )
    parser.add_argument(
        "--raw-only-vcpu-map",
        type=parse_vcpu_map,
        action="append",
        default=[],
        metavar="MODEL_CORE:GUEST_VCPU",
        help=(
            "sample this guest vCPU into --raw-log without adding it to "
            "the vCPU power output"
        ),
    )
    parser.add_argument("--fixed-freq-ghz", type=float, default=number("DEFAULT_FIXED_FREQ_GHZ"))
    parser.add_argument("--sys-scale", type=float, default=number("DEFAULT_SYS_SCALE"))
    parser.add_argument(
        "--gemmini-full-util-power-w",
        type=float,
        default=DEFAULT_GEMMINI_FULL_UTIL_POWER_W,
    )
    parser.add_argument(
        "--gemmini-util-threshold-pct",
        type=float,
        default=DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT,
    )
    parser.add_argument("--interval", type=float, default=number("DEFAULT_VCPU_SAMPLE_INTERVAL_S"))
    parser.add_argument(
        "--proc-stat",
        default="/proc/stat",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if (
        args.fixed_freq_ghz <= 0
        or args.sys_scale <= 0
        or args.gemmini_full_util_power_w <= 0
        or args.interval <= 0
    ):
        parser.error("frequency, scales, power, and interval must be greater than zero")
    if not 0 <= args.gemmini_util_threshold_pct <= 100:
        parser.error("Gemmini utilization threshold must be between 0 and 100")
    if bool(args.turbostat) != bool(args.gemmini_cores):
        parser.error("--turbostat and --gemmini-cores must be provided together")
    if len(set(args.gemmini_cores)) != len(args.gemmini_cores):
        parser.error("--gemmini-cores contains a duplicate core")

    vcpu_map = dict(args.vcpu_map)
    raw_only_vcpu_map = dict(args.raw_only_vcpu_map)
    if len(vcpu_map) != len(args.vcpu_map):
        parser.error("--vcpu-map contains a duplicate model core")
    if len(set(vcpu_map.values())) != len(vcpu_map):
        parser.error("--vcpu-map contains a duplicate guest vCPU")
    if len(raw_only_vcpu_map) != len(args.raw_only_vcpu_map):
        parser.error("--raw-only-vcpu-map contains a duplicate model core")
    if len(set(raw_only_vcpu_map.values())) != len(raw_only_vcpu_map):
        parser.error("--raw-only-vcpu-map contains a duplicate guest vCPU")
    if set(vcpu_map) & set(raw_only_vcpu_map):
        parser.error("modeled and raw-only maps contain a duplicate model core")
    if set(vcpu_map.values()) & set(raw_only_vcpu_map.values()):
        parser.error("modeled and raw-only maps contain a duplicate guest vCPU")
    if set(args.gemmini_cores) & (set(vcpu_map) | set(raw_only_vcpu_map)):
        parser.error("Gemmini and modeled vCPU core IDs must differ")
    return args


def read_cpu_counters(path, target_vcpus):
    """Read the first eight /proc/stat CPU fields for the requested vCPUs."""

    counters = {}
    with open(path, "r", encoding="utf-8", errors="strict") as stream:
        for raw in stream:
            tokens = raw.split()
            if not tokens or not tokens[0].startswith("cpu"):
                continue
            cpu_suffix = tokens[0][3:]
            if not cpu_suffix.isdigit():
                continue
            guest_vcpu = int(cpu_suffix)
            if guest_vcpu not in target_vcpus:
                continue
            if len(tokens) < 9:
                raise RuntimeError(f"incomplete /proc/stat row for cpu{guest_vcpu}")
            try:
                # user nice system idle iowait irq softirq steal
                # guest/guest_nice are excluded because Linux already includes
                # them in user/nice time.
                counters[guest_vcpu] = tuple(int(value) for value in tokens[1:9])
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid /proc/stat counters for cpu{guest_vcpu}"
                ) from exc

    missing = sorted(target_vcpus - counters.keys())
    if missing:
        missing_text = ", ".join(f"cpu{cpu}" for cpu in missing)
        raise RuntimeError(f"missing requested vCPU counters: {missing_text}")
    return counters


def utilization_delta(previous, current):
    if any(cur < prev for prev, cur in zip(previous, current)):
        raise RuntimeError("/proc/stat CPU counter decreased")

    delta_total = sum(current) - sum(previous)
    delta_idle = (current[3] - previous[3]) + (current[4] - previous[4])
    if delta_total <= 0:
        busy_percent = 0.0
    else:
        busy_percent = 100.0 * (delta_total - delta_idle) / delta_total
    busy_percent = max(0.0, min(100.0, busy_percent))
    return delta_total, delta_idle, busy_percent


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_raw_header_once(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(RAW_HEADER)


def request_stop(_signum, _frame):
    STOP_REQUESTED.set()


def main():
    args = parse_args()
    os.umask(0)

    vcpu_map = dict(args.vcpu_map)
    raw_only_vcpu_map = dict(args.raw_only_vcpu_map)
    raw_vcpu_map = {**vcpu_map, **raw_only_vcpu_map}
    gemmini_cores = sorted(args.gemmini_cores)
    model_cores = sorted(vcpu_map)
    raw_cores = sorted(raw_vcpu_map)
    target_vcpus = set(raw_vcpu_map.values())
    interval_ns = max(1, round(args.interval * 1_000_000_000))

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    write_header_once(args.out_log, gemmini_cores, model_cores, raw_only_vcpu_map)
    write_raw_header_once(args.raw_log)

    previous = read_cpu_counters(args.proc_stat, target_vcpus)
    previous_monotonic_ns = time.monotonic_ns()
    next_deadline_ns = previous_monotonic_ns + interval_ns
    sample_count = 0
    power_sample_count = 0

    print("[power_sim_vcpu_procstat] started", flush=True)
    print(f"[power_sim_vcpu_procstat] proc stat: {args.proc_stat}", flush=True)
    print(f"[power_sim_vcpu_procstat] raw csv  : {args.raw_log}", flush=True)
    print(f"[power_sim_vcpu_procstat] power csv: {args.out_log}", flush=True)
    if args.turbostat:
        print(f"[power_sim_vcpu_procstat] turbostat: {args.turbostat}", flush=True)
        print(
            f"[power_sim_vcpu_procstat] Gemmini cores: {gemmini_cores}",
            flush=True,
        )
    print(f"[power_sim_vcpu_procstat] modeled map : {vcpu_map}", flush=True)
    print(
        f"[power_sim_vcpu_procstat] raw-only map: {raw_only_vcpu_map}",
        flush=True,
    )
    print(f"[power_sim_vcpu_procstat] interval : {args.interval:.6f}s", flush=True)

    with open(args.raw_log, "a", newline="", encoding="utf-8") as raw_stream:
        raw_writer = csv.writer(raw_stream)
        while not STOP_REQUESTED.is_set():
            now_ns = time.monotonic_ns()
            remaining_s = max(0, next_deadline_ns - now_ns) / 1_000_000_000
            if STOP_REQUESTED.wait(remaining_s):
                break

            current = read_cpu_counters(args.proc_stat, target_vcpus)
            current_monotonic_ns = time.monotonic_ns()
            sample_epoch_ns = time.time_ns()
            sample_tod_s = sample_epoch_ns / 1_000_000_000
            actual_interval_s = (
                current_monotonic_ns - previous_monotonic_ns
            ) / 1_000_000_000

            raw_busy_by_core = {}
            for raw_core in raw_cores:
                guest_vcpu = raw_vcpu_map[raw_core]
                delta_total, delta_idle, busy_percent = utilization_delta(
                    previous[guest_vcpu], current[guest_vcpu]
                )
                raw_busy_by_core[raw_core] = busy_percent
                raw_writer.writerow(
                    [
                        sample_epoch_ns,
                        f"{sample_tod_s:.9f}",
                        current_monotonic_ns,
                        f"{actual_interval_s:.9f}",
                        raw_core,
                        guest_vcpu,
                        delta_total,
                        delta_idle,
                        f"{busy_percent:.6f}",
                    ]
                )
            raw_stream.flush()

            busy_by_core = {
                model_core: raw_busy_by_core[model_core]
                for model_core in model_cores
            }
            write_power_sample = True
            if args.turbostat:
                latest_gemmini = parse_latest_complete_block(
                    args.turbostat, gemmini_cores
                )
                if latest_gemmini is None:
                    write_power_sample = False
                else:
                    busy_by_core.update(latest_gemmini["busy_by_core"])

            if write_power_sample:
                append_sample(
                    args.out_log,
                    sample_tod_s,
                    gemmini_cores,
                    model_cores,
                    args.fixed_freq_ghz,
                    args.sys_scale,
                    args.gemmini_full_util_power_w if gemmini_cores else 0.0,
                    busy_by_core,
                    epoch_ns=sample_epoch_ns,
                    gemmini_util_threshold_pct=args.gemmini_util_threshold_pct,
                    observed_vcpu_busy={
                        core: raw_busy_by_core[core] for core in raw_only_vcpu_map
                    },
                )
                power_sample_count += 1
            sample_count += 1

            previous = current
            previous_monotonic_ns = current_monotonic_ns
            next_deadline_ns += interval_ns
            if next_deadline_ns <= current_monotonic_ns:
                missed_intervals = (
                    (current_monotonic_ns - next_deadline_ns) // interval_ns
                ) + 1
                next_deadline_ns += missed_intervals * interval_ns

    print(
        "[power_sim_vcpu_procstat] stopped after "
        f"{sample_count} utilization samples and {power_sample_count} power samples",
        flush=True,
    )


if __name__ == "__main__":
    main()
