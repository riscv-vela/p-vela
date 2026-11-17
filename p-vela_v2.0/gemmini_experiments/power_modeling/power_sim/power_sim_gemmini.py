#!/usr/bin/env python3

import argparse
import csv
import os
import re
import time

from power_model_config import number

# ====================================================
# System Power Model Parameters (AMD, FINAL)
# ----------------------------------------------------
# This part is intentionally kept the same as the
# existing OpenSSL / 7z power simulators.
# Only the utilization input source changes in this
# Gemmini version; the core CPU power formula does not.
# ====================================================
BOOST_OFF_TH = number("CPU_BOOST_OFF_THRESHOLD_GHZ")

a_boost = number("CPU_A_BOOST")
k_boost = number("CPU_K_BOOST")

a_nom = number("CPU_A_NOM")
k_nom = number("CPU_K_NOM")

# Default platform-level scaling factor.
DEFAULT_SETA_SYS = number("DEFAULT_SYS_SCALE")

# Default full-util Gemmini power scale used by the single-app model.
DEFAULT_GEMMINI_FULL_UTIL_POWER_W = number("DEFAULT_GEMMINI_FULL_UTIL_POWER_W")
DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT = number("DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT")
# Powered-idle power. Was 0.013 W (0.0127 W clock internal + leakage, from
# synthesis); re-measured as 0.014 W (26-11-09).
DEFAULT_GEMMINI_IDLE_POWER_W = number("DEFAULT_GEMMINI_IDLE_POWER_W")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Real-time Gemmini power simulator. "
            "Reads the latest gemmini-worker / vCPU Busy%% samples from a "
            "growing turbostat log and appends per-core / breakdown / total "
            "power to a CSV log."
        )
    )
    # This collector does not read guest /proc/stat or memory usage. It watches a
    # host-side turbostat log and uses host Busy% for the pinned
    # gemmini-worker and vCPU cores.
    parser.add_argument(
        "--turbostat",
        required=True,
        help="Path to the growing turbostat log file",
    )
    parser.add_argument(
        "--out-log",
        required=True,
        help="CSV path to append simulated power samples",
    )
    parser.add_argument(
        "--gemmini-cores",
        type=int,
        nargs="+",
        required=True,
        help="Host cores used by gemmini-worker threads, e.g. --gemmini-cores 2 3 4 5",
    )
    parser.add_argument(
        "--vcpu-cores",
        type=int,
        nargs="*",
        default=[],
        help="Host cores used by vCPU threads, e.g. --vcpu-cores 8 9 10 11",
    )
    parser.add_argument(
        "--fixed-freq-ghz",
        type=float,
        default=number("DEFAULT_FIXED_FREQ_GHZ"),
        help="Fixed frequency used by the power model (GHz; from shared defaults)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=number("DEFAULT_POWER_POLL_INTERVAL_S"),
        help="Seconds between log polling iterations (from shared defaults)",
    )
    parser.add_argument(
        "--sys-scale",
        type=float,
        default=DEFAULT_SETA_SYS,
        help="Platform-level scaling factor applied to vCPU CPU-model power",
    )
    parser.add_argument(
        "--gemmini-full-util-power-w",
        "--gemmini-theta",
        type=float,
        default=DEFAULT_GEMMINI_FULL_UTIL_POWER_W,
        dest="gemmini_full_util_power_w",
        help=(
            "Gemmini power in watts at util=1.0 "
            f"(default: {DEFAULT_GEMMINI_FULL_UTIL_POWER_W:.6f}, "
            "from the configured single-app Gemmini calibration)"
        ),
    )
    parser.add_argument(
        "--gemmini-gate-log",
        default=None,
        help="QEMU log carrying [POWER_GATE] transitions; gated intervals are "
             "recorded as 0 W instead of idle power (TC.00.007)",
    )
    parser.add_argument(
        "--gemmini-util-threshold-pct",
        type=float,
        default=DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT,
        help=(
            "Binary Gemmini activity threshold in Busy%%: samples at or above "
            "the threshold use 100%% utilization/full power; lower samples use "
            f"0%%/{DEFAULT_GEMMINI_IDLE_POWER_W:.3f} W "
            f"(default: {DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT:g})"
        ),
    )
    return parser.parse_args()


def cpu_power(freq_ghz, util):
    # Same CPU power model used by the existing OpenSSL-based script.
    # We keep these coefficients / equations identical so that the
    # only experimental change is the utilization source.
    f3 = freq_ghz ** 3
    if freq_ghz <= BOOST_OFF_TH:
        return a_nom * util * f3 + k_nom
    return a_boost * util * f3 + k_boost


GATE_LINE = re.compile(
    r"\[POWER_GATE\]\[Gemmini (\d+)\]\s+state=(gated|powered)\s+host_ts_ns=(\d+)")


class GateTimeline:
    """QEMU 로그의 [POWER_GATE] 전이를 **증분으로** 따라 읽는다.

    전이는 측정 도중에 발생하므로 시작 시 한 번만 읽으면 놓친다. 파일 오프셋을
    유지하며 매 샘플마다 새 줄만 파싱한다 (로그가 커도 비용이 일정하다).
    QEMU 가 로그를 truncate 하면 오프셋을 되감는다.
    """

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.by_dev = {}

    def poll(self):
        if not self.path:
            return self.by_dev
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return self.by_dev
        if size < self.pos:          # truncate 되었으면 처음부터
            self.pos = 0
        if size == self.pos:
            return self.by_dev
        with open(self.path, "r", errors="replace") as f:
            f.seek(self.pos)
            for line in f:
                m = GATE_LINE.search(line)
                if m:
                    dev, state, ts = int(m.group(1)), m.group(2), int(m.group(3))
                    self.by_dev.setdefault(dev, []).append((ts / 1e9, state == "gated"))
            self.pos = f.tell()
        for v in self.by_dev.values():
            v.sort()
        return self.by_dev


def gated_at(timeline, t):
    """시각 t 에서의 게이팅 여부 (마지막 전이 기준)."""
    state = False
    for ts, g in timeline:
        if ts > t:
            break
        state = g
    return state


def threshold_gemmini_busy(busy_percent, threshold_pct):
    return 100.0 if busy_percent >= threshold_pct else 0.0


def gemmini_power(util, full_util_power_w, gated=False):
    """활성 / 전원 인가 유휴 / power gating 의 3상태.

    turbostat Busy% 만으로는 "유휴지만 전원 인가"와 "게이팅"을 구분할 수 없다.
    게이팅 여부는 QEMU 가 남기는 [POWER_GATE] 전이 로그에서 가져온다
    (--gemmini-gate-log).  SFR-POW-06 / TC.00.007.
    """
    if gated:
        return 0.0
    return full_util_power_w if util >= 1.0 else DEFAULT_GEMMINI_IDLE_POWER_W


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_header_once(path, gemmini_cores, vcpu_cores, observed_vcpu_cores=()):
    ensure_parent_dir(path)

    all_cores = gemmini_cores + vcpu_cores
    header = [
        "epoch_ns",
        "sample_tod_s",
        "model_freq_ghz",
        "model_sys_scale",
        "model_gemmini_full_util_power_w",
    ]
    header += [f"raw_busy_core_{core}" for core in gemmini_cores]
    header += [f"busy_core_{core}" for core in all_cores]
    header += [f"util_core_{core}" for core in all_cores]
    header += [f"busy_core_{core}" for core in sorted(observed_vcpu_cores)]
    header += [f"util_core_{core}" for core in sorted(observed_vcpu_cores)]
    header += [f"power_gemmini_core_{core}" for core in gemmini_cores]
    header += [f"power_vcpu_core_{core}" for core in vcpu_cores]
    header += ["p_gemmini_total_w", "p_vcpu_total_w", "p_total_w"]

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != header:
                raise SystemExit(f"Power CSV schema differs; use a new output file: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_sample(
    path,
    sample_tod_s,
    gemmini_cores,
    vcpu_cores,
    fixed_freq_ghz,
    sys_scale,
    gemmini_full_util_power_w,
    busy_by_core,
    epoch_ns=None,
    gemmini_util_threshold_pct=DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT,
    observed_vcpu_busy=None,
    gate_timelines=None,
):
    # Gemmini Busy% is converted to a binary 0/100% activity signal using the
    # configured threshold. Gemmini power is therefore either idle power or
    # full-util power. vCPU power keeps the existing CPU/frequency-based model.
    all_cores = gemmini_cores + vcpu_cores
    modeled_busy_by_core = dict(busy_by_core)
    for core in gemmini_cores:
        modeled_busy_by_core[core] = threshold_gemmini_busy(
            busy_by_core[core], gemmini_util_threshold_pct
        )
    busy_values = [modeled_busy_by_core[core] for core in all_cores]
    utils = [busy / 100.0 for busy in busy_values]
    util_by_core = dict(zip(all_cores, utils))

    # gemmini_cores 의 i 번째가 장치 i 에 대응한다 (러너가 그 순서로 핀한다).
    per_gemmini_power = [
        gemmini_power(
            util_by_core[core],
            gemmini_full_util_power_w,
            gated_at((gate_timelines or {}).get(i, []), sample_tod_s),
        )
        for i, core in enumerate(gemmini_cores)
    ]
    per_vcpu_power = [
        sys_scale * cpu_power(fixed_freq_ghz, util_by_core[core]) for core in vcpu_cores
    ]
    gemmini_total_power = sum(per_gemmini_power)
    vcpu_total_power = sum(per_vcpu_power)
    total_power = gemmini_total_power + vcpu_total_power

    if epoch_ns is None:
        epoch_ns = time.time_ns()

    row = [
        epoch_ns,
        f"{sample_tod_s:.6f}",
        f"{fixed_freq_ghz:.6f}",
        f"{sys_scale:.6f}",
        f"{gemmini_full_util_power_w:.6f}",
    ]
    # Preserve the host observation before binary activity modeling.
    row += [f"{busy_by_core[core]:.6f}" for core in gemmini_cores]
    row += [f"{busy:.6f}" for busy in busy_values]
    row += [f"{util:.6f}" for util in utils]
    observed = observed_vcpu_busy or {}
    row += [f"{observed[core]:.6f}" for core in sorted(observed)]
    row += [f"{observed[core] / 100.0:.6f}" for core in sorted(observed)]
    row += [f"{power:.6f}" for power in per_gemmini_power]
    row += [f"{power:.6f}" for power in per_vcpu_power]
    row += [
        f"{gemmini_total_power:.6f}",
        f"{vcpu_total_power:.6f}",
        f"{total_power:.6f}",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return per_gemmini_power, per_vcpu_power, gemmini_total_power, vcpu_total_power, total_power


def parse_latest_complete_block(path, target_cores):
    # turbostat periodically re-prints the header and then emits one
    # sample block. We scan the growing log and keep the latest block
    # that contains every requested gemmini worker core.
    # Host Busy% from turbostat is used as Gemmini worker utilization.
    header = None
    current_block = {}
    latest_complete = None

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                tokens = line.split()

                if (
                    "Time_Of_Day_Seconds" in tokens
                    and "CPU" in tokens
                    and "Busy%" in tokens
                ):
                    if all(core in current_block for core in target_cores):
                        latest_complete = dict(current_block)
                    header = tokens
                    current_block = {}
                    continue

                if header is None or len(tokens) < len(header):
                    continue

                try:
                    cpu_token = tokens[header.index("CPU")]
                    tod = float(tokens[header.index("Time_Of_Day_Seconds")])
                    busy = float(tokens[header.index("Busy%")])
                except (ValueError, IndexError):
                    continue

                if cpu_token == "-":
                    continue

                cpu = int(cpu_token)
                if cpu in target_cores:
                    current_block[cpu] = {
                        "sample_tod_s": tod,
                        "busy_percent": busy,
                    }
    except FileNotFoundError:
        return None

    if all(core in current_block for core in target_cores):
        latest_complete = dict(current_block)

    if latest_complete is None:
        return None

    sample_tod_s = max(latest_complete[core]["sample_tod_s"] for core in target_cores)
    busy_by_core = {
        core: latest_complete[core]["busy_percent"] for core in target_cores
    }

    return {
        "sample_tod_s": sample_tod_s,
        "busy_by_core": busy_by_core,
    }


def main():
    args = parse_args()

    # QEMU 가 남긴 [POWER_GATE] 전이를 읽어 장치별 게이팅 타임라인을 만든다.
    # 로그를 주지 않으면 빈 dict 가 되어 기존 2상태 동작과 동일하다.
    gate_tracker = GateTimeline(args.gemmini_gate_log)
    if args.gemmini_gate_log:
        print(f"[power_sim_gemmini] gate log: {args.gemmini_gate_log}", flush=True)

    gemmini_cores = sorted(args.gemmini_cores)
    vcpu_cores = sorted(args.vcpu_cores)
    target_cores = gemmini_cores + vcpu_cores
    write_header_once(args.out_log, gemmini_cores, vcpu_cores)

    # Each real-time iteration reads the latest host Busy% sample block.
    print("[power_sim_gemmini] real-time Gemmini power simulator started")
    print(f"[power_sim_gemmini] turbostat log : {args.turbostat}")
    print(f"[power_sim_gemmini] out csv       : {args.out_log}")
    print(f"[power_sim_gemmini] gemmini cores : {gemmini_cores}")
    print(f"[power_sim_gemmini] vcpu cores    : {vcpu_cores}")
    print(f"[power_sim_gemmini] fixed freq    : {args.fixed_freq_ghz:.3f} GHz")
    print(f"[power_sim_gemmini] sys scale     : {args.sys_scale:.3f}")
    print(
        "[power_sim_gemmini] gemmini full-util power : "
        f"{args.gemmini_full_util_power_w:.6f} W"
    )
    print(
        "[power_sim_gemmini] gemmini util threshold : "
        f"{args.gemmini_util_threshold_pct:g}% (binary off/full)"
    )
    print(f"[power_sim_gemmini] poll interval : {args.poll_interval:.3f} s")

    last_sample_tod_s = None

    try:
        while True:
            latest = parse_latest_complete_block(args.turbostat, target_cores)
            if latest is None:
                time.sleep(args.poll_interval)
                continue

            sample_tod_s = latest["sample_tod_s"]
            if last_sample_tod_s is not None and sample_tod_s <= last_sample_tod_s:
                time.sleep(args.poll_interval)
                continue

            # Use a fixed frequency so that this experiment isolates the
            # effect of Gemmini worker utilization on power.
            (
                per_gemmini_power,
                per_vcpu_power,
                gemmini_total_power,
                vcpu_total_power,
                total_power,
            ) = append_sample(
                args.out_log,
                sample_tod_s,
                gemmini_cores,
                vcpu_cores,
                args.fixed_freq_ghz,
                args.sys_scale,
                args.gemmini_full_util_power_w,
                latest["busy_by_core"],
                gemmini_util_threshold_pct=args.gemmini_util_threshold_pct,
                gate_timelines=gate_tracker.poll(),
            )

            gemmini_busy_str = ", ".join(
                f"g{core}={latest['busy_by_core'][core]:.2f}%"
                for core in gemmini_cores
            )
            vcpu_busy_str = ", ".join(
                f"v{core}={latest['busy_by_core'][core]:.2f}%"
                for core in vcpu_cores
            )
            gemmini_power_str = ", ".join(
                f"g{core}={power:.3f}W"
                for core, power in zip(gemmini_cores, per_gemmini_power)
            )
            vcpu_power_str = ", ".join(
                f"v{core}={power:.3f}W"
                for core, power in zip(vcpu_cores, per_vcpu_power)
            )

            status_parts = [f"[power_sim_gemmini] tod={sample_tod_s:.6f}"]
            if gemmini_busy_str:
                status_parts.append(f"gemmini_busy[{gemmini_busy_str}]")
            if vcpu_busy_str:
                status_parts.append(f"vcpu_busy[{vcpu_busy_str}]")
            if gemmini_power_str:
                status_parts.append(
                    f"gemmini_power[{gemmini_power_str}] sum={gemmini_total_power:.3f}W"
                )
            if vcpu_power_str:
                status_parts.append(
                    f"vcpu_power[{vcpu_power_str}] sum={vcpu_total_power:.3f}W"
                )
            status_parts.append(f"total={total_power:.3f}W")
            print(
                " | ".join(status_parts)
            )

            last_sample_tod_s = sample_tod_s
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[power_sim_gemmini] stopped")


if __name__ == "__main__":
    main()
