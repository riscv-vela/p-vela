#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
from bisect import bisect_right

from gemmini_hw_cycle_model import WORKLOAD_PROFILES, estimate_workload
from power_model_config import DEFAULTS, number


HOST_TS_PATTERN = re.compile(r"^(\[HOST_TS_REALTIME\]\s+\S+\s+)([0-9]+\.[0-9]+)\s*$")
HOST_TS_LINE_PATTERN = re.compile(r"^(\[HOST_TS_LINE\]\s+)([0-9]+\.[0-9]+)(\s+.*)$")
TIMELINE_PATTERN = re.compile(
    r"^\[timeline\]\s+actor=(\S+)\s+label=(\S+)\s+sample=(\d+)\s+thread=(\d+)\s+cpu=(-?\d+)\s*$"
)
GEMM_TIMELINE_PATTERN = re.compile(
    r"^\[timeline\]\s+t_ms=\S+\s+event=(gemm_begin|gemm_end)\s+rep=(\d+)\s*$"
)
SAMPLE_START_PATTERN = re.compile(
    r"^\[sample-start\]\s+order=(\d+)\s+idx=(\d+)\s+thread=(\d+)\s+cpu=(-?\d+)\s*$"
)
SAMPLE_DONE_PATTERN = re.compile(
    r"^\[sample-done\]\s+order=(\d+)\s+idx=(\d+)\s+thread=(\d+)\s+cpu=(-?\d+)\s+.*$"
)
BENCH_START_PATTERN = re.compile(r"^\[HOST_TS_REALTIME\]\s+BENCH_HOST_START\s+([0-9]+\.[0-9]+)\s*$")
BENCH_END_PATTERN = re.compile(r"^\[HOST_TS_REALTIME\]\s+BENCH_HOST_END\s+([0-9]+\.[0-9]+)\s*$")
MODEL_PREFIX_PATTERN = re.compile(
    r"^\[(?P<model>[^\]]+)\]\s+(?P<payload>\[(?:timeline|sample-start|sample-done|backend-profile)\].*)$"
)
CPU_GEMM_BREAKDOWN_PATTERN = re.compile(
    r"\[cpu-gemm-breakdown\].*\btotal_ms=([0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
)
CPU_BACKEND_PROFILE_PATTERN = re.compile(
    r"\[backend-profile\].*\bbackend=cpu\b.*\blinear_total_ms=([0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
)
GEMMINI_BACKEND_PROFILE_PATTERN = re.compile(
    r"\[backend-profile\].*\bbackend=gemmini\b.*\btiled_matmul_ms=([0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
)


DEFAULT_TARGET_RATIO = number("DEFAULT_CPU_REF_TARGET_RATIO")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply CPU/Gemmini time scaling to bench/turbostat/power logs and "
            "emit scaled logs plus a summary JSON."
        )
    )
    parser.add_argument("--bench", required=True, help="input bench log path")
    parser.add_argument("--turbostat", required=True, help="input turbostat log path")
    parser.add_argument("--power-csv", required=True, help="input power CSV path")
    parser.add_argument(
        "--guest-util-csv",
        help="optional guest /proc/stat utilization CSV to place on the scaled time axis",
    )
    parser.add_argument(
        "--time-scale-mode",
        choices=("cpu-ref-auto", "rtl-cycle"),
        default=DEFAULTS["DEFAULT_TIME_SCALE_MODE"],
        help="RTL cycle-based scaling or CPU-reference compatibility mode",
    )
    parser.add_argument(
        "--rtl-workload",
        choices=sorted(WORKLOAD_PROFILES),
        help="application command profile used by --time-scale-mode rtl-cycle",
    )
    parser.add_argument(
        "--gemmini-frequency-mhz",
        type=float,
        default=number("DEFAULT_GEMMINI_FREQUENCY_MHZ"),
        help="Gemmini RTL clock frequency in MHz (from shared defaults)",
    )
    parser.add_argument(
        "--cpu-time-scale",
        type=float,
        default=number("DEFAULT_CPU_TIME_SCALE"),
        help=(
            "scale factor applied to non-Gemmini portions of inference sample "
            "windows (from shared defaults)"
        ),
    )
    parser.add_argument(
        "--model-rtl-workload",
        action="append",
        default=[],
        metavar="MODEL=WORKLOAD",
        help=(
            "model-specific RTL workload profile for independent dual-app "
            "rtl-cycle scaling; may be repeated"
        ),
    )
    parser.add_argument(
        "--model-core",
        action="append",
        default=[],
        metavar="MODEL=CORE[,CORE...]",
        help=(
            "cores owned by each model for independent dual-app scaling; may be repeated, "
            "e.g. --model-core distilbert=3,9,11"
        ),
    )
    parser.add_argument(
        "--cpu-ref-bench",
        help="CPU-only bench log used by --time-scale-mode cpu-ref-auto",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=DEFAULT_TARGET_RATIO,
        help="target ratio applied to CPU-only/raw-Gemmini active-time ratio",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="output prefix; script writes *_bench.log, *_turbostat.log, *_power.csv, *_summary.json",
    )
    parser.add_argument(
        "--result-prefix",
        help="optional separate prefix for power CSV and summary; intermediate logs use --out-prefix",
    )
    args = parser.parse_args()
    if args.time_scale_mode not in ("rtl-cycle", "cpu-ref-auto"):
        parser.error("time scale mode must be rtl-cycle or cpu-ref-auto")
    return args


def parse_model_rtl_workloads(values):
    model_workloads = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(
                f"--model-rtl-workload expects MODEL=WORKLOAD, got: {item}"
            )
        model, workload = (value.strip() for value in item.split("=", 1))
        if not model:
            raise SystemExit(f"--model-rtl-workload has an empty MODEL: {item}")
        if workload not in WORKLOAD_PROFILES:
            raise SystemExit(
                f"unsupported workload in --model-rtl-workload {item}"
            )
        model_workloads[model] = workload
    return model_workloads


def parse_model_cores(values):
    model_cores = {}
    used_cores = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--model-core expects MODEL=CORE[,CORE...], got: {item}")
        model, raw_cores = item.split("=", 1)
        model = model.strip()
        if not model:
            raise SystemExit(f"--model-core has an empty MODEL: {item}")
        try:
            cores = {int(value.strip()) for value in raw_cores.split(",") if value.strip()}
        except ValueError as exc:
            raise SystemExit(f"invalid core in --model-core {item}") from exc
        if not cores:
            raise SystemExit(f"--model-core has no cores: {item}")
        for core in cores:
            previous = used_cores.get(core)
            if previous is not None and previous != model:
                raise SystemExit(f"core {core} is assigned to both {previous} and {model}")
            used_cores[core] = model
        model_cores[model] = cores
    return model_cores


def split_model_prefix(payload):
    stripped = payload.strip()
    match = MODEL_PREFIX_PATTERN.match(stripped)
    if match:
        return match.group("model"), match.group("payload")
    return None, stripped


def parse_bench_entries(path):
    entries = []
    bench_host_start = None
    bench_host_end = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f):
            stripped = raw.strip()

            start_match = BENCH_START_PATTERN.match(stripped)
            if start_match:
                bench_host_start = float(start_match.group(1))

            end_match = BENCH_END_PATTERN.match(stripped)
            if end_match:
                bench_host_end = float(end_match.group(1))

            host_line_match = HOST_TS_LINE_PATTERN.match(raw.rstrip("\n"))
            if not host_line_match:
                continue

            host_ts = float(host_line_match.group(2))
            model_key, payload = split_model_prefix(host_line_match.group(3).strip())
            entry = {
                "line_no": line_no,
                "host_ts": host_ts,
                "model_key": model_key,
                "payload": payload,
                "kind": "other",
                "sample": None,
                "thread": None,
                "actor": None,
                "label": None,
            }

            timeline_match = TIMELINE_PATTERN.match(payload)
            if timeline_match:
                entry["kind"] = "timeline"
                entry["actor"] = timeline_match.group(1)
                entry["label"] = timeline_match.group(2)
                entry["sample"] = int(timeline_match.group(3))
                entry["thread"] = int(timeline_match.group(4))
            else:
                gemm_timeline_match = GEMM_TIMELINE_PATTERN.match(payload)
                if gemm_timeline_match:
                    entry["kind"] = "timeline"
                    entry["actor"] = "gemmini"
                    entry["label"] = gemm_timeline_match.group(1)
                    entry["sample"] = int(gemm_timeline_match.group(2))
                    entry["thread"] = -1
                else:
                    start_match = SAMPLE_START_PATTERN.match(payload)
                    if start_match:
                        entry["kind"] = "sample-start"
                        entry["sample"] = int(start_match.group(2))
                        entry["thread"] = int(start_match.group(3))
                    else:
                        done_match = SAMPLE_DONE_PATTERN.match(payload)
                        if done_match:
                            entry["kind"] = "sample-done"
                            entry["sample"] = int(done_match.group(2))
                            entry["thread"] = int(done_match.group(3))

            entries.append(entry)

    if bench_host_start is None or bench_host_end is None:
        raise SystemExit("bench log is missing BENCH_HOST_START or BENCH_HOST_END markers")

    return entries, bench_host_start, bench_host_end


def build_model_start_anchors(entries):
    anchors = {}
    for entry in entries:
        model_key = entry["model_key"]
        if (
            model_key is None
            or entry["kind"] != "timeline"
            or entry["label"] != "remote_launch"
        ):
            continue
        anchors[model_key] = min(entry["host_ts"], anchors.get(model_key, math.inf))
    return anchors


def build_sample_windows(entries):
    starts = {}
    ends = {}
    fallback_first = {}

    for entry in entries:
        sample = entry["sample"]
        thread = entry["thread"]
        if sample is None or thread is None:
            continue

        key = (entry["model_key"], sample, thread)
        fallback_first.setdefault(key, entry["host_ts"])

        if entry["kind"] == "sample-start":
            starts[key] = entry["host_ts"]
        elif entry["kind"] == "sample-done":
            ends[key] = entry["host_ts"]
        elif entry["kind"] == "timeline" and entry["label"] == "gemm_begin":
            starts[key] = entry["host_ts"]
        elif entry["kind"] == "timeline" and entry["label"] == "gemm_end":
            ends[key] = entry["host_ts"]

    windows = []
    for key in sorted(set(starts) | set(ends) | set(fallback_first)):
        start_ts = starts.get(key, fallback_first.get(key))
        end_ts = ends.get(key)
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            continue
        windows.append(
            {
                "key": key,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        )

    if not windows:
        raise SystemExit("no valid sample windows were found in the bench log")

    return windows


def build_labeled_gemmini_intervals(entries, sample_windows):
    grouped = {}
    for entry in entries:
        sample = entry["sample"]
        thread = entry["thread"]
        if sample is None or thread is None:
            continue
        grouped.setdefault((entry["model_key"], sample, thread), []).append(entry)

    intervals = []
    for window in sample_windows:
        key = window["key"]
        local_entries = sorted(
            grouped.get(key, []),
            key=lambda item: (item["host_ts"], item["line_no"]),
        )
        if not local_entries:
            continue

        for idx, entry in enumerate(local_entries):
            if entry["kind"] != "timeline" or entry["actor"] != "gemmini":
                continue
            if entry["label"].endswith("_end"):
                continue

            start_ts = entry["host_ts"]
            end_ts = window["end_ts"]
            for next_entry in local_entries[idx + 1 :]:
                if next_entry["host_ts"] > start_ts:
                    end_ts = next_entry["host_ts"]
                    break

            if end_ts > start_ts:
                intervals.append(
                    {
                        "model_key": key[0],
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    }
                )

    if not intervals:
        raise SystemExit("no Gemmini-active intervals were found in the bench log")

    return merge_labeled_intervals(intervals)


def merge_labeled_intervals(intervals):
    merged_by_model = {}
    for interval in sorted(
        intervals,
        key=lambda item: ("" if item["model_key"] is None else item["model_key"], item["start_ts"], item["end_ts"]),
    ):
        model_key = interval["model_key"]
        merged = merged_by_model.setdefault(model_key, [])
        start_ts = interval["start_ts"]
        end_ts = interval["end_ts"]
        if not merged or start_ts > merged[-1]["end_ts"]:
            merged.append({"model_key": model_key, "start_ts": start_ts, "end_ts": end_ts})
            continue
        if end_ts > merged[-1]["end_ts"]:
            merged[-1]["end_ts"] = end_ts

    out = []
    for merged in merged_by_model.values():
        out.extend(merged)
    return sorted(out, key=lambda item: (item["start_ts"], item["end_ts"], "" if item["model_key"] is None else item["model_key"]))


def build_gemmini_intervals(entries, sample_windows):
    labeled_intervals = build_labeled_gemmini_intervals(entries, sample_windows)
    return merge_intervals(
        [(interval["start_ts"], interval["end_ts"]) for interval in labeled_intervals]
    )

def merge_intervals(intervals):
    merged = []
    for start_ts, end_ts in sorted(intervals):
        if not merged or start_ts > merged[-1][1]:
            merged.append([start_ts, end_ts])
            continue

        if end_ts > merged[-1][1]:
            merged[-1][1] = end_ts

    return [(start_ts, end_ts) for start_ts, end_ts in merged]


def subtract_intervals(intervals, excluded):
    remaining = []
    excluded = merge_intervals(excluded)
    for start_ts, end_ts in merge_intervals(intervals):
        cursor = start_ts
        for excluded_start, excluded_end in excluded:
            if excluded_end <= cursor:
                continue
            if excluded_start >= end_ts:
                break
            if excluded_start > cursor:
                remaining.append((cursor, min(excluded_start, end_ts)))
            cursor = max(cursor, excluded_end)
            if cursor >= end_ts:
                break
        if cursor < end_ts:
            remaining.append((cursor, end_ts))
    return remaining


def interval_total_duration(intervals):
    return sum(end_ts - start_ts for start_ts, end_ts in intervals)


def parse_cpu_gemm_breakdown_duration(path):
    total_ms_values = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            match = CPU_GEMM_BREAKDOWN_PATTERN.search(raw)
            if match:
                total_ms_values.append(float(match.group(1)))

    if not total_ms_values:
        return None

    return {
        "cpu_ref_target_region_s": sum(total_ms_values) / 1000.0,
        "cpu_ref_target_region_method": "cpu_gemm_breakdown_total_ms",
        "cpu_ref_target_region_count": len(total_ms_values),
    }


def parse_cpu_backend_profile_duration(path):
    total_ms_values = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            match = CPU_BACKEND_PROFILE_PATTERN.search(raw)
            if match:
                total_ms_values.append(float(match.group(1)))

    if not total_ms_values:
        return None

    return {
        "cpu_ref_target_region_s": sum(total_ms_values) / 1000.0,
        "cpu_ref_target_region_method": "backend_profile_linear_total_ms",
        "cpu_ref_target_region_count": len(total_ms_values),
    }


def parse_cpu_timeline_duration(path):
    entries, _, _ = parse_bench_entries(path)
    sample_windows = build_sample_windows(entries)
    intervals = build_gemmini_intervals(entries, sample_windows)
    return {
        "cpu_ref_target_region_s": interval_total_duration(intervals),
        "cpu_ref_target_region_method": "timeline_gemm_intervals",
        "cpu_ref_target_region_count": len(intervals),
    }


def parse_cpu_ref_target_duration(path):
    breakdown = parse_cpu_gemm_breakdown_duration(path)
    if breakdown is not None:
        return breakdown

    backend_profile = parse_cpu_backend_profile_duration(path)
    if backend_profile is not None:
        return backend_profile

    try:
        return parse_cpu_timeline_duration(path)
    except SystemExit as exc:
        raise SystemExit(
            "CPU reference bench log does not contain [cpu-gemm-breakdown], "
            "[backend-profile] backend=cpu linear_total_ms, or usable "
            "gemm_begin/gemm_end timeline intervals: "
            f"{path}"
        ) from exc


class PiecewiseTimeScaler:
    def __init__(self, intervals, scale, cpu_intervals=None, cpu_scale=1.0):
        if scale <= 0.0 or cpu_scale <= 0.0:
            raise SystemExit("time scales must be > 0")

        self.scale = scale
        self.intervals = intervals
        self.cpu_scale = cpu_scale
        self.cpu_intervals = cpu_intervals or []
        self.segments = sorted(
            [(start_ts, end_ts, scale) for start_ts, end_ts in intervals]
            + [
                (start_ts, end_ts, cpu_scale)
                for start_ts, end_ts in self.cpu_intervals
            ]
        )
        self.starts = [start_ts for start_ts, _, _ in self.segments]
        self.ends = [end_ts for _, end_ts, _ in self.segments]
        self.scales = [segment_scale for _, _, segment_scale in self.segments]
        self.reduction_before = []

        total_reduction = 0.0
        for start_ts, end_ts, segment_scale in self.segments:
            self.reduction_before.append(total_reduction)
            total_reduction += (end_ts - start_ts) * (1.0 - segment_scale)

        self.total_reduction = total_reduction

    def map_time(self, raw_ts):
        idx = bisect_right(self.starts, raw_ts) - 1
        if idx < 0:
            return raw_ts

        start_ts = self.starts[idx]
        end_ts = self.ends[idx]
        scale = self.scales[idx]
        reduction_before = self.reduction_before[idx]

        if raw_ts < end_ts:
            scaled_start = start_ts - reduction_before
            return scaled_start + (raw_ts - start_ts) * scale

        return raw_ts - (
            reduction_before + (end_ts - start_ts) * (1.0 - scale)
        )


    def scaled_duration(self, start_ts, end_ts):
        return self.map_time(end_ts) - self.map_time(start_ts)


class IndependentModelTimeScaler:
    def __init__(
        self,
        labeled_intervals,
        model_time_scales,
        sample_windows=None,
        model_start_anchors=None,
        cpu_scale=1.0,
    ):
        self.scale = None
        self.cpu_scale = cpu_scale
        self.model_time_scales = dict(model_time_scales)
        self.model_start_anchors = dict(model_start_anchors or {})
        intervals_by_model = {}
        for interval in labeled_intervals:
            model_key = interval["model_key"]
            intervals_by_model.setdefault(model_key, []).append(
                (interval["start_ts"], interval["end_ts"])
            )

        sample_intervals_by_model = {}
        for window in sample_windows or []:
            model_key = window["key"][0]
            sample_intervals_by_model.setdefault(model_key, []).append(
                (window["start_ts"], window["end_ts"])
            )

        self.sample_intervals_by_model = {}
        self.cpu_intervals_by_model = {}
        self.startup_intervals_by_model = {}
        self.scalers = {}
        for model_key, scale in self.model_time_scales.items():
            intervals = merge_intervals(intervals_by_model.get(model_key, []))
            if not intervals:
                raise SystemExit(f"no Gemmini-active intervals found for model: {model_key}")
            sample_intervals = merge_intervals(
                sample_intervals_by_model.get(model_key, [])
            )
            if cpu_scale != 1.0 and not sample_intervals:
                raise SystemExit(f"no sample windows found for model: {model_key}")
            startup_intervals = []
            start_anchor = self.model_start_anchors.get(model_key)
            if (
                start_anchor is not None
                and sample_intervals
                and start_anchor < sample_intervals[0][0]
            ):
                startup_intervals = [(start_anchor, sample_intervals[0][0])]
            cpu_intervals = subtract_intervals(sample_intervals, intervals)
            self.sample_intervals_by_model[model_key] = sample_intervals
            self.cpu_intervals_by_model[model_key] = cpu_intervals
            self.startup_intervals_by_model[model_key] = startup_intervals
            self.scalers[model_key] = PiecewiseTimeScaler(
                intervals,
                scale,
                cpu_intervals=startup_intervals + cpu_intervals,
                cpu_scale=cpu_scale,
            )

    def map_model_time(self, model_key, raw_ts):
        scaler = self.scalers.get(model_key)
        if scaler is None:
            model_name = "<unlabeled>" if model_key is None else model_key
            raise SystemExit(f"missing independent time scaler for model: {model_name}")
        return scaler.map_time(raw_ts)

    def map_time(self, raw_ts):
        return max(scaler.map_time(raw_ts) for scaler in self.scalers.values())

    def scaled_duration(self, start_ts, end_ts):
        return self.map_time(end_ts) - self.map_time(start_ts)

    def scaled_model_duration(self, model_key, start_ts, end_ts):
        return self.map_model_time(model_key, end_ts) - self.map_model_time(model_key, start_ts)

    def segment_summaries(self):
        summaries = []
        for model_key, scaler in sorted(self.scalers.items()):
            gemmini_intervals = set(scaler.intervals)
            for start_ts, end_ts, scale in scaler.segments:
                summaries.append(
                    {
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "scale": scale,
                        "region": (
                            "gemmini"
                            if (start_ts, end_ts) in gemmini_intervals
                            else "cpu"
                        ),
                        "active_models": [model_key],
                    }
                )
        return summaries


def map_time_for_model(scaler, model_key, raw_ts):
    if isinstance(scaler, IndependentModelTimeScaler) and model_key is not None:
        return scaler.map_model_time(model_key, raw_ts)
    return scaler.map_time(raw_ts)



def build_per_model_time_summary(scaler):
    summaries = {}
    cpu_interval_rows = []
    for model_key, model_scaler in sorted(scaler.scalers.items()):
        sample_intervals = scaler.sample_intervals_by_model[model_key]
        gemmini_intervals = model_scaler.intervals
        cpu_intervals = scaler.cpu_intervals_by_model[model_key]
        startup_intervals = scaler.startup_intervals_by_model[model_key]
        summaries[model_key] = {
            "cpu_time_scale": model_scaler.cpu_scale,
            "gemmini_time_scale": model_scaler.scale,
            "inference_raw_time_s": interval_total_duration(sample_intervals),
            "inference_scaled_time_s": sum(
                model_scaler.scaled_duration(start_ts, end_ts)
                for start_ts, end_ts in sample_intervals
            ),
            "cpu_raw_time_s": interval_total_duration(cpu_intervals),
            "cpu_scaled_time_s": sum(
                model_scaler.scaled_duration(start_ts, end_ts)
                for start_ts, end_ts in cpu_intervals
            ),
            "startup_raw_time_s": interval_total_duration(startup_intervals),
            "startup_scaled_time_s": sum(
                model_scaler.scaled_duration(start_ts, end_ts)
                for start_ts, end_ts in startup_intervals
            ),
            "gemmini_raw_time_s": interval_total_duration(gemmini_intervals),
            "gemmini_scaled_time_s": sum(
                model_scaler.scaled_duration(start_ts, end_ts)
                for start_ts, end_ts in gemmini_intervals
            ),
            "sample_interval_count": len(sample_intervals),
            "cpu_interval_count": len(cpu_intervals),
            "startup_interval_count": len(startup_intervals),
            "gemmini_interval_count": len(gemmini_intervals),
        }
        for start_ts, end_ts in cpu_intervals:
            cpu_interval_rows.append(
                {
                    "model_key": model_key,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "scaled_start_ts": model_scaler.map_time(start_ts),
                    "scaled_end_ts": model_scaler.map_time(end_ts),
                    "raw_duration_s": end_ts - start_ts,
                    "scaled_duration_s": model_scaler.scaled_duration(
                        start_ts, end_ts
                    ),
                }
            )
    return summaries, cpu_interval_rows


def integrate_energy(samples):
    if len(samples) < 2:
        return 0.0

    energy_j = 0.0
    for prev_row, next_row in zip(samples, samples[1:]):
        dt = next_row["sample_tod_s"] - prev_row["sample_tod_s"]
        if dt <= 0.0:
            continue
        energy_j += 0.5 * (prev_row["p_total_w"] + next_row["p_total_w"]) * dt
    return energy_j


def rewrite_power_csv(in_path, out_path, scaler):
    rewritten_rows = []

    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "sample_tod_s" not in fieldnames:
            raise SystemExit("power CSV is missing sample_tod_s")
        if "p_total_w" not in fieldnames:
            raise SystemExit("power CSV is missing p_total_w")

        output_fieldnames = list(fieldnames)
        if "raw_sample_tod_s" not in output_fieldnames:
            sample_idx = output_fieldnames.index("sample_tod_s")
            output_fieldnames.insert(sample_idx, "raw_sample_tod_s")

        for row in reader:
            try:
                raw_sample_tod_s = float(row["sample_tod_s"])
                total_power = float(row["p_total_w"])
            except ValueError:
                continue

            scaled_sample_tod_s = scaler.map_time(raw_sample_tod_s)
            rewritten = dict(row)
            rewritten["raw_sample_tod_s"] = f"{raw_sample_tod_s:.6f}"
            rewritten["sample_tod_s"] = f"{scaled_sample_tod_s:.6f}"
            rewritten_rows.append(
                {
                    "sample_tod_s": raw_sample_tod_s,
                    "p_total_w": total_power,
                    "out_row": rewritten,
                    "scaled_sample_tod_s": scaled_sample_tod_s,
                }
            )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for item in rewritten_rows:
            writer.writerow(item["out_row"])

    raw_energy_j = integrate_energy(rewritten_rows)
    scaled_energy_j = integrate_energy(
        [
            {
                "sample_tod_s": item["scaled_sample_tod_s"],
                "p_total_w": item["p_total_w"],
            }
            for item in rewritten_rows
        ]
    )

    return raw_energy_j, scaled_energy_j, len(rewritten_rows)


def rewrite_guest_util_csv(in_path, out_path, scaler, core_to_model=None):
    core_to_model = core_to_model or {}
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "sample_tod_s" not in fieldnames:
            raise SystemExit("guest utilization CSV is missing sample_tod_s")
        rows = list(reader)

    output_fieldnames = list(fieldnames)
    if "raw_sample_tod_s" not in output_fieldnames:
        sample_idx = output_fieldnames.index("sample_tod_s")
        output_fieldnames.insert(sample_idx, "raw_sample_tod_s")

    rewritten_rows = []
    for row in rows:
        try:
            raw_sample_tod_s = float(row["sample_tod_s"])
        except (TypeError, ValueError):
            continue

        map_time = scaler.map_time
        if isinstance(scaler, IndependentModelTimeScaler):
            try:
                model_core = int(row.get("model_core", ""))
            except (TypeError, ValueError):
                model_core = None
            model_key = core_to_model.get(model_core)
            if model_key is not None:
                map_time = lambda raw_ts, key=model_key: scaler.map_model_time(
                    key, raw_ts
                )

        scaled_sample_tod_s = map_time(raw_sample_tod_s)
        rewritten = dict(row)
        rewritten["raw_sample_tod_s"] = f"{raw_sample_tod_s:.9f}"
        rewritten["sample_tod_s"] = f"{scaled_sample_tod_s:.9f}"

        if "epoch_ns" in output_fieldnames:
            rewritten["epoch_ns"] = str(int(round(scaled_sample_tod_s * 1_000_000_000)))
        if "interval_s" in output_fieldnames:
            try:
                raw_interval_s = float(row["interval_s"])
            except (TypeError, ValueError):
                pass
            else:
                scaled_interval_s = scaled_sample_tod_s - map_time(
                    raw_sample_tod_s - raw_interval_s
                )
                rewritten["interval_s"] = f"{max(0.0, scaled_interval_s):.9f}"

        rewritten_rows.append(rewritten)

    if not rewritten_rows:
        raise SystemExit("guest utilization CSV has no valid data rows")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rewritten_rows)

    return len(rewritten_rows)


def interpolate_series(times, values, target_ts):
    idx = bisect_right(times, target_ts)
    if idx <= 0:
        return values[0]
    if idx >= len(times):
        return values[-1]
    left_ts = times[idx - 1]
    right_ts = times[idx]
    left_value = values[idx - 1]
    right_value = values[idx]
    if right_ts <= left_ts:
        return right_value
    ratio = (target_ts - left_ts) / (right_ts - left_ts)
    return left_value + (right_value - left_value) * ratio


def rewrite_power_csv_independent(in_path, out_path, scaler, model_cores):
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or "sample_tod_s" not in fieldnames or "p_total_w" not in fieldnames:
        raise SystemExit("power CSV is missing rows, sample_tod_s, or p_total_w")

    raw_times = [float(row["sample_tod_s"]) for row in rows]
    raw_power_samples = [
        {"sample_tod_s": raw_ts, "p_total_w": float(row["p_total_w"])}
        for raw_ts, row in zip(raw_times, rows)
    ]
    # Preserve every independently mapped model sample.  A raw 100 ms grid
    # must not be reused after time compression, because doing so drops most
    # of the samples and turns the plotted power trace into coarse blocks.
    mapped_times = {
        model: [scaler.map_model_time(model, raw_ts) for raw_ts in raw_times]
        for model in model_cores
    }
    output_times = sorted(
        {round(mapped_ts, 9) for times in mapped_times.values() for mapped_ts in times}
    )
    core_to_model = {
        core: model
        for model, cores in model_cores.items()
        for core in cores
    }

    core_fields = {}
    for field in fieldnames:
        match = re.search(r"_core_(\d+)$", field)
        if not match:
            continue
        core = int(match.group(1))
        model = core_to_model.get(core)
        if model is None:
            continue
        core_fields[field] = (model, [float(row[field]) for row in rows])

    output_fieldnames = list(fieldnames)
    if "raw_sample_tod_s" not in output_fieldnames:
        output_fieldnames.insert(output_fieldnames.index("sample_tod_s"), "raw_sample_tod_s")

    output_rows = []
    scaled_power_samples = []
    for output_ts in output_times:
        output_row = dict(rows[0])
        output_row["epoch_ns"] = str(int(round(output_ts * 1_000_000_000)))
        output_row["raw_sample_tod_s"] = ""
        output_row["sample_tod_s"] = f"{output_ts:.9f}"
        for field, (model, values) in core_fields.items():
            value = interpolate_series(mapped_times[model], values, output_ts)
            output_row[field] = f"{value:.6f}"

        gemmini_total = sum(
            float(output_row[field])
            for field in fieldnames
            if field.startswith("power_gemmini_core_")
        )
        vcpu_total = sum(
            float(output_row[field])
            for field in fieldnames
            if field.startswith("power_vcpu_core_")
        )
        total_power = gemmini_total + vcpu_total
        output_row["p_gemmini_total_w"] = f"{gemmini_total:.6f}"
        output_row["p_vcpu_total_w"] = f"{vcpu_total:.6f}"
        output_row["p_total_w"] = f"{total_power:.6f}"
        output_rows.append(output_row)
        scaled_power_samples.append(
            {"sample_tod_s": output_ts, "p_total_w": total_power}
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return (
        integrate_energy(raw_power_samples),
        integrate_energy(scaled_power_samples),
        len(output_rows),
    )



def rewrite_bench_log(in_path, out_path, scaler):
    with open(in_path, "r", encoding="utf-8", errors="replace") as src, open(
        out_path, "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            match = HOST_TS_PATTERN.match(raw.rstrip("\n"))
            if match:
                scaled_ts = scaler.map_time(float(match.group(2)))
                dst.write(f"{match.group(1)}{scaled_ts:.9f}\n")
                continue

            match = HOST_TS_LINE_PATTERN.match(raw.rstrip("\n"))
            if match:
                model_key, _ = split_model_prefix(match.group(3).strip())
                scaled_ts = map_time_for_model(scaler, model_key, float(match.group(2)))
                dst.write(f"{match.group(1)}{scaled_ts:.9f}{match.group(3)}\n")
                continue

            dst.write(raw)


def rewrite_turbostat(in_path, out_path, scaler, core_to_model=None):
    header = None
    time_idx = None
    cpu_idx = None

    with open(in_path, "r", encoding="utf-8", errors="replace") as src, open(
        out_path, "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            stripped = raw.strip()
            if not stripped:
                dst.write(raw)
                continue

            tokens = stripped.split()
            if "Time_Of_Day_Seconds" in tokens and "CPU" in tokens:
                header = tokens
                time_idx = header.index("Time_Of_Day_Seconds")
                cpu_idx = header.index("CPU")
                dst.write(raw)
                continue

            if header is None or time_idx is None or cpu_idx is None or len(tokens) < len(header):
                dst.write(raw)
                continue

            try:
                raw_ts = float(tokens[time_idx])
                core = int(tokens[cpu_idx])
            except ValueError:
                dst.write(raw)
                continue

            model_key = (core_to_model or {}).get(core)
            scaled_ts = map_time_for_model(scaler, model_key, raw_ts)
            tokens[time_idx] = f"{scaled_ts:.6f}"
            dst.write("\t".join(tokens) + "\n")


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_gemmini_tiled_matmul_duration(path, model_key=None):
    total_ms_values = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            payload = raw
            if model_key is not None:
                host_line_match = HOST_TS_LINE_PATTERN.match(raw.rstrip("\n"))
                if not host_line_match:
                    continue
                line_model, payload = split_model_prefix(
                    host_line_match.group(3).strip()
                )
                if line_model != model_key:
                    continue
            match = GEMMINI_BACKEND_PROFILE_PATTERN.search(payload)
            if match:
                total_ms_values.append(float(match.group(1)))
    return None if not total_ms_values else sum(total_ms_values) / 1000.0


def count_rtl_workload_executions(entries, workload, model_key=None):
    scoped_entries = (
        entry
        for entry in entries
        if model_key is None or entry["model_key"] == model_key
    )
    if workload == "polybench_gemm":
        return sum(
            entry["kind"] == "timeline" and entry["label"] == "gemm_begin"
            for entry in scoped_entries
        )
    return sum(entry["kind"] == "sample-start" for entry in scoped_entries)


def resolve_rtl_workload_time_scale(
    args, entries, workload, gemmini_raw_time_s, model_key=None
):
    """Shared single/dual RTL target calculation; raw measurements are never cached."""
    if args.gemmini_frequency_mhz <= 0.0:
        raise SystemExit("--gemmini-frequency-mhz must be > 0")
    if gemmini_raw_time_s <= 0.0:
        scope = "rtl-cycle" if model_key is None else f"model: {model_key}"
        raise SystemExit(f"Gemmini raw active time must be > 0 for {scope}")

    executions = count_rtl_workload_executions(entries, workload, model_key=model_key)
    if executions <= 0:
        scope = "in the bench log" if model_key is None else f"for model: {model_key}"
        raise SystemExit(f"no executions of {workload} were found {scope}")
    estimate = estimate_workload(workload, executions, args.gemmini_frequency_mhz)
    observed_intervals = sum(
        (model_key is None or entry["model_key"] == model_key)
        and entry["kind"] == "timeline"
        and entry["actor"] == "gemmini"
        and not entry["label"].endswith("_end")
        for entry in entries
    )
    if observed_intervals != estimate.gemmini_intervals:
        scope = "" if model_key is None else f" for {model_key}"
        raise SystemExit(
            f"RTL workload profile{scope} does not match bench log: "
            f"expected {estimate.gemmini_intervals} Gemmini intervals, "
            f"found {observed_intervals}"
        )

    raw_tiled_matmul_s = parse_gemmini_tiled_matmul_duration(
        args.bench, model_key=model_key
    )
    if raw_tiled_matmul_s is None:
        embedded_cpu_raw_time_s = 0.0
        embedded_cpu_method = "unavailable_assumed_zero"
    else:
        embedded_cpu_raw_time_s = max(
            0.0, gemmini_raw_time_s - min(raw_tiled_matmul_s, gemmini_raw_time_s)
        )
        embedded_cpu_method = "raw_interval_minus_backend_tiled_matmul"
    embedded_cpu_scaled_time_s = embedded_cpu_raw_time_s * args.cpu_time_scale
    gemmini_target_time_s = estimate.rtl_time_s + embedded_cpu_scaled_time_s
    time_scale = gemmini_target_time_s / gemmini_raw_time_s
    return time_scale, {
        "rtl_workload": estimate.workload,
        "rtl_frequency_mhz": estimate.frequency_mhz,
        "rtl_execution_count": estimate.executions,
        "rtl_cycles_per_execution": estimate.cycles_per_execution,
        "rtl_cycle_total": estimate.total_cycles,
        "rtl_time_s": estimate.rtl_time_s,
        "rtl_api_call_count": estimate.api_calls,
        "rtl_loop_ws_count": estimate.loop_ws_calls,
        "rtl_expected_gemmini_intervals": estimate.gemmini_intervals,
        "rtl_observed_gemmini_intervals": observed_intervals,
        "raw_tiled_matmul_profile_s": raw_tiled_matmul_s,
        "embedded_cpu_method": embedded_cpu_method,
        "embedded_cpu_raw_time_s": embedded_cpu_raw_time_s,
        "embedded_cpu_scaled_time_s": embedded_cpu_scaled_time_s,
        "gemmini_target_time_s": gemmini_target_time_s,
    }


def resolve_model_rtl_time_scales(
    args, entries, labeled_gemmini_intervals, model_workloads
):
    if args.gemmini_frequency_mhz <= 0.0:
        raise SystemExit("--gemmini-frequency-mhz must be > 0")

    model_time_scales = {}
    model_rtl = {}
    for model_key, workload in sorted(model_workloads.items()):
        model_intervals = merge_intervals(
            [
                (interval["start_ts"], interval["end_ts"])
                for interval in labeled_gemmini_intervals
                if interval["model_key"] == model_key
            ]
        )
        gemmini_raw_time_s = interval_total_duration(model_intervals)
        time_scale, rtl_meta = resolve_rtl_workload_time_scale(
            args, entries, workload, gemmini_raw_time_s, model_key=model_key
        )
        model_time_scales[model_key] = time_scale
        model_rtl[model_key] = {
            **rtl_meta,
            "gemmini_raw_time_s": gemmini_raw_time_s,
            "gemmini_time_scale": time_scale,
        }

    return model_time_scales, model_rtl


def resolve_time_scale(args, gemmini_raw_time_s, entries):
    if args.time_scale_mode == "rtl-cycle":
        if args.rtl_workload is None:
            raise SystemExit("--rtl-workload is required in rtl-cycle mode")
        time_scale, rtl_meta = resolve_rtl_workload_time_scale(
            args, entries, args.rtl_workload, gemmini_raw_time_s
        )
        return time_scale, {
            "time_scale_mode": "rtl-cycle",
            "target_ratio": None,
            "cpu_ref_bench_log": None,
            "cpu_ref_target_region_s": None,
            "cpu_ref_target_region_method": None,
            "cpu_ref_target_region_count": None,
            "cpu_to_gemmini_raw_time_ratio": None,
            "gemmini_to_cpu_ref_time_ratio": None,
            "model_time_scales": None,
            "effective_time_scale_segments": None,
            **rtl_meta,
            "gemmini_time_semantics": "rtl_compute_plus_scaled_embedded_cpu",
        }

    if args.cpu_ref_bench is None:
        raise SystemExit("--cpu-ref-bench is required when --time-scale-mode cpu-ref-auto")
    if args.target_ratio <= 0.0:
        raise SystemExit("--target-ratio must be > 0")
    if gemmini_raw_time_s <= 0.0:
        raise SystemExit("Gemmini raw active time must be > 0 for cpu-ref-auto")

    cpu_ref = parse_cpu_ref_target_duration(args.cpu_ref_bench)
    cpu_ref_target_region_s = cpu_ref["cpu_ref_target_region_s"]
    if cpu_ref_target_region_s <= 0.0:
        raise SystemExit("CPU reference target region time must be > 0")

    cpu_to_gemmini_raw_time_ratio = cpu_ref_target_region_s / gemmini_raw_time_s
    gemmini_to_cpu_ref_time_ratio = gemmini_raw_time_s / cpu_ref_target_region_s
    time_scale = cpu_to_gemmini_raw_time_ratio * args.target_ratio
    if time_scale <= 0.0:
        raise SystemExit("computed time scale must be > 0")

    return time_scale, {
        "time_scale_mode": "cpu-ref-auto",
        "target_ratio": args.target_ratio,
        "cpu_ref_bench_log": os.path.abspath(args.cpu_ref_bench),
        "cpu_ref_target_region_s": cpu_ref_target_region_s,
        "cpu_ref_target_region_method": cpu_ref["cpu_ref_target_region_method"],
        "cpu_ref_target_region_count": cpu_ref["cpu_ref_target_region_count"],
        "cpu_to_gemmini_raw_time_ratio": cpu_to_gemmini_raw_time_ratio,
        "gemmini_to_cpu_ref_time_ratio": gemmini_to_cpu_ref_time_ratio,
        "model_time_scales": None,
        "effective_time_scale_segments": None,
    }


def build_summary(
    bench_path,
    turbostat_path,
    power_csv_path,
    bench_host_start,
    bench_host_end,
    sample_windows,
    sample_union,
    cpu_intervals,
    gemmini_intervals,
    labeled_gemmini_intervals,
    scaler,
    time_scale_meta,
    raw_energy_j,
    scaled_energy_j,
    power_row_count,
):
    sample_union_scaled = merge_intervals(
        [
            (
                map_time_for_model(scaler, window["key"][0], window["start_ts"]),
                map_time_for_model(scaler, window["key"][0], window["end_ts"]),
            )
            for window in sample_windows
        ]
    )
    gemmini_intervals_scaled = merge_intervals(
        [
            (
                map_time_for_model(scaler, interval["model_key"], interval["start_ts"]),
                map_time_for_model(scaler, interval["model_key"], interval["end_ts"]),
            )
            for interval in labeled_gemmini_intervals
        ]
    )

    inference_raw_time_s = interval_total_duration(sample_union)
    gemmini_raw_time_s = interval_total_duration(gemmini_intervals)
    gemmini_scaled_time_s = interval_total_duration(gemmini_intervals_scaled)
    inference_scaled_time_s = interval_total_duration(sample_union_scaled)

    per_model_time = None
    cpu_time_semantics = "inference_cpu_intervals"
    if isinstance(scaler, IndependentModelTimeScaler):
        per_model_time, cpu_interval_rows = build_per_model_time_summary(scaler)
        cpu_raw_time_s = interval_total_duration(
            merge_intervals(
                [(row["start_ts"], row["end_ts"]) for row in cpu_interval_rows]
            )
        )
        cpu_scaled_time_s = interval_total_duration(
            merge_intervals(
                [
                    (row["scaled_start_ts"], row["scaled_end_ts"])
                    for row in cpu_interval_rows
                ]
            )
        )
        cpu_time_semantics = "wallclock_union_of_per_model_cpu_intervals"
    else:
        cpu_raw_time_s = interval_total_duration(cpu_intervals)
        cpu_scaled_time_s = sum(
            scaler.scaled_duration(start_ts, end_ts)
            for start_ts, end_ts in cpu_intervals
        )
        cpu_interval_rows = [
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "scaled_start_ts": scaler.map_time(start_ts),
                "scaled_end_ts": scaler.map_time(end_ts),
                "raw_duration_s": end_ts - start_ts,
                "scaled_duration_s": scaler.scaled_duration(start_ts, end_ts),
            }
            for start_ts, end_ts in cpu_intervals
        ]

    if isinstance(scaler, IndependentModelTimeScaler):
        gemmini_time_semantics = "wallclock_union_of_per_model_gemmini_intervals"
        gemmini_interval_rows = []
        for interval in labeled_gemmini_intervals:
            model_key = interval["model_key"]
            model_scaler = scaler.scalers[model_key]
            start_ts = interval["start_ts"]
            end_ts = interval["end_ts"]
            gemmini_interval_rows.append(
                {
                    "model_key": model_key,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "scaled_start_ts": model_scaler.map_time(start_ts),
                    "scaled_end_ts": model_scaler.map_time(end_ts),
                    "raw_duration_s": end_ts - start_ts,
                    "scaled_duration_s": model_scaler.scaled_duration(
                        start_ts, end_ts
                    ),
                }
            )
    else:
        gemmini_time_semantics = time_scale_meta.get(
            "gemmini_time_semantics", "gemmini_active_intervals"
        )
        gemmini_interval_rows = [
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "scaled_start_ts": scaler.map_time(start_ts),
                "scaled_end_ts": scaler.map_time(end_ts),
                "raw_duration_s": end_ts - start_ts,
                "scaled_duration_s": scaler.scaled_duration(start_ts, end_ts),
            }
            for start_ts, end_ts in gemmini_intervals
        ]

    summary = {
        "summary_kind": "gemmini_time_scale_summary",
        "source_bench_log": os.path.abspath(bench_path),
        "source_turbostat_log": os.path.abspath(turbostat_path),
        "source_power_csv": os.path.abspath(power_csv_path),
        "time_scale_mode": time_scale_meta["time_scale_mode"],
        "rtl_workload": time_scale_meta.get("rtl_workload"),
        "rtl_frequency_mhz": time_scale_meta.get("rtl_frequency_mhz"),
        "rtl_execution_count": time_scale_meta.get("rtl_execution_count"),
        "rtl_cycles_per_execution": time_scale_meta.get("rtl_cycles_per_execution"),
        "rtl_cycle_total": time_scale_meta.get("rtl_cycle_total"),
        "rtl_time_s": time_scale_meta.get("rtl_time_s"),
        "rtl_api_call_count": time_scale_meta.get("rtl_api_call_count"),
        "rtl_loop_ws_count": time_scale_meta.get("rtl_loop_ws_count"),
        "rtl_expected_gemmini_intervals": time_scale_meta.get("rtl_expected_gemmini_intervals"),
        "rtl_observed_gemmini_intervals": time_scale_meta.get("rtl_observed_gemmini_intervals"),
        "raw_tiled_matmul_profile_s": time_scale_meta.get("raw_tiled_matmul_profile_s"),
        "embedded_cpu_method": time_scale_meta.get("embedded_cpu_method"),
        "embedded_cpu_raw_time_s": time_scale_meta.get("embedded_cpu_raw_time_s"),
        "embedded_cpu_scaled_time_s": time_scale_meta.get("embedded_cpu_scaled_time_s"),
        "gemmini_target_time_s": time_scale_meta.get("gemmini_target_time_s"),
        "gemmini_time_scale": getattr(scaler, "scale", None),
        "computed_time_scale": getattr(scaler, "scale", None),
        "model_time_scales": time_scale_meta.get("model_time_scales"),
        "model_rtl": time_scale_meta.get("model_rtl"),
        "effective_time_scale_segments": time_scale_meta.get("effective_time_scale_segments"),
        "target_ratio": time_scale_meta["target_ratio"],
        "cpu_ref_bench_log": time_scale_meta["cpu_ref_bench_log"],
        "cpu_ref_target_region_s": time_scale_meta["cpu_ref_target_region_s"],
        "cpu_ref_target_region_method": time_scale_meta["cpu_ref_target_region_method"],
        "cpu_ref_target_region_count": time_scale_meta["cpu_ref_target_region_count"],
        "cpu_to_gemmini_raw_time_ratio": time_scale_meta["cpu_to_gemmini_raw_time_ratio"],
        "gemmini_to_cpu_ref_time_ratio": time_scale_meta["gemmini_to_cpu_ref_time_ratio"],
        "cpu_time_scale": getattr(scaler, "cpu_scale", 1.0),
        "cpu_time_semantics": cpu_time_semantics,
        "cpu_raw_time_s": cpu_raw_time_s,
        "cpu_scaled_time_s": cpu_scaled_time_s,
        "per_model_time": per_model_time,
        "gemmini_time_semantics": gemmini_time_semantics,
        "gemmini_raw_active_time_s": gemmini_raw_time_s,
        "host_wallclock_elapsed_s": bench_host_end - bench_host_start,
        "inference_raw_time_s": inference_raw_time_s,
        "gemmini_raw_time_s": gemmini_raw_time_s,
        "gemmini_scaled_time_s": gemmini_scaled_time_s,
        "inference_scaled_time_s": inference_scaled_time_s,
        "time_reduction_s": inference_raw_time_s - inference_scaled_time_s,
        "time_reduction_ratio": (
            0.0
            if inference_raw_time_s <= 0.0
            else (inference_raw_time_s - inference_scaled_time_s) / inference_raw_time_s
        ),
        "energy_raw_j": raw_energy_j,
        "energy_scaled_j": scaled_energy_j,
        "sample_count": len(sample_windows),
        "sample_count_semantics": "sample_window_count",
        "sample_window_count": len(sample_windows),
        "power_row_count": power_row_count,
        "gemmini_interval_count": len(gemmini_interval_rows),
        "cpu_interval_count": len(cpu_interval_rows),
        "sample_windows": [
            {
                "model_key": model_key,
                "sample": sample,
                "thread": thread,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "scaled_start_ts": map_time_for_model(scaler, model_key, start_ts),
                "scaled_end_ts": map_time_for_model(scaler, model_key, end_ts),
                "raw_duration_s": end_ts - start_ts,
                "scaled_duration_s": map_time_for_model(scaler, model_key, end_ts) - map_time_for_model(scaler, model_key, start_ts),
            }
            for (model_key, sample, thread), start_ts, end_ts in (
                (window["key"], window["start_ts"], window["end_ts"]) for window in sample_windows
            )
        ],
        "gemmini_intervals": gemmini_interval_rows,
        "cpu_intervals": cpu_interval_rows,
    }
    return summary


def summary_paths(out_prefix, include_guest_util=False):
    paths = {
        "bench": f"{out_prefix}_bench.log",
        "turbostat": f"{out_prefix}_turbostat.log",
        "power_csv": f"{out_prefix}_power.csv",
        "summary": f"{out_prefix}_summary.json",
    }
    if include_guest_util:
        paths["guest_util_csv"] = f"{out_prefix}_guest_util.csv"
    return paths


def main():
    args = parse_args()

    paths = summary_paths(args.out_prefix, include_guest_util=bool(args.guest_util_csv))
    if args.result_prefix:
        paths["power_csv"] = f"{args.result_prefix}_power.csv"
        paths["summary"] = f"{args.result_prefix}_summary.json"
    for path in paths.values():
        ensure_parent_dir(path)

    entries, bench_host_start, bench_host_end = parse_bench_entries(args.bench)
    model_start_anchors = build_model_start_anchors(entries)
    sample_windows = build_sample_windows(entries)
    sample_union = merge_intervals(
        [(window["start_ts"], window["end_ts"]) for window in sample_windows]
    )
    if args.cpu_time_scale <= 0.0:
        raise SystemExit("--cpu-time-scale must be > 0")
    model_rtl_workloads = parse_model_rtl_workloads(args.model_rtl_workload)
    model_cores = parse_model_cores(args.model_core)
    labeled_gemmini_intervals = build_labeled_gemmini_intervals(entries, sample_windows)
    gemmini_intervals = merge_intervals(
        [(interval["start_ts"], interval["end_ts"]) for interval in labeled_gemmini_intervals]
    )
    cpu_intervals = subtract_intervals(sample_union, gemmini_intervals)
    gemmini_raw_time_s = interval_total_duration(gemmini_intervals)
    if model_rtl_workloads:
        if args.time_scale_mode != "rtl-cycle":
            raise SystemExit(
                "--model-rtl-workload requires --time-scale-mode rtl-cycle"
            )
        if args.rtl_workload is not None:
            raise SystemExit(
                "--rtl-workload and --model-rtl-workload cannot be combined"
            )
        if not model_cores:
            raise SystemExit("--model-rtl-workload requires --model-core")
        if set(model_cores) != set(model_rtl_workloads):
            raise SystemExit(
                "--model-core and --model-rtl-workload must name the same models"
            )
        model_time_scales, model_rtl = resolve_model_rtl_time_scales(
            args,
            entries,
            labeled_gemmini_intervals,
            model_rtl_workloads,
        )
        scaler = IndependentModelTimeScaler(
            labeled_gemmini_intervals,
            model_time_scales,
            sample_windows=sample_windows,
            model_start_anchors=model_start_anchors,
            cpu_scale=args.cpu_time_scale,
        )
        time_scale_meta = {
            "time_scale_mode": "rtl-cycle-per-model-independent",
            "target_ratio": None,
            "cpu_ref_bench_log": None,
            "cpu_ref_target_region_s": None,
            "cpu_ref_target_region_method": None,
            "cpu_ref_target_region_count": None,
            "cpu_to_gemmini_raw_time_ratio": None,
            "gemmini_to_cpu_ref_time_ratio": None,
            "model_time_scales": model_time_scales,
            "model_rtl": model_rtl,
            "effective_time_scale_segments": scaler.segment_summaries(),
        }
    else:
        time_scale, time_scale_meta = resolve_time_scale(args, gemmini_raw_time_s, entries)
        scaler = PiecewiseTimeScaler(
            gemmini_intervals,
            time_scale,
            cpu_intervals=cpu_intervals,
            cpu_scale=args.cpu_time_scale,
        )

    rewrite_bench_log(args.bench, paths["bench"], scaler)
    core_to_model = None
    if isinstance(scaler, IndependentModelTimeScaler):
        core_to_model = {
            core: model
            for model, cores in model_cores.items()
            for core in cores
        }
        rewrite_turbostat(
            args.turbostat, paths["turbostat"], scaler, core_to_model=core_to_model
        )
        raw_energy_j, scaled_energy_j, power_row_count = rewrite_power_csv_independent(
            args.power_csv, paths["power_csv"], scaler, model_cores
        )
    else:
        rewrite_turbostat(args.turbostat, paths["turbostat"], scaler)
        raw_energy_j, scaled_energy_j, power_row_count = rewrite_power_csv(
            args.power_csv,
            paths["power_csv"],
            scaler,
        )

    if args.guest_util_csv:
        rewrite_guest_util_csv(
            args.guest_util_csv,
            paths["guest_util_csv"],
            scaler,
            core_to_model=core_to_model,
        )

    summary = build_summary(
        args.bench,
        args.turbostat,
        args.power_csv,
        bench_host_start,
        bench_host_end,
        sample_windows,
        sample_union,
        cpu_intervals,
        gemmini_intervals,
        labeled_gemmini_intervals,
        scaler,
        time_scale_meta,
        raw_energy_j,
        scaled_energy_j,
        power_row_count,
    )
    if args.guest_util_csv:
        summary["source_guest_util_csv"] = os.path.abspath(args.guest_util_csv)
        summary["scaled_guest_util_csv"] = os.path.abspath(paths["guest_util_csv"])

    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"[time_scale_gemmini_logs] bench   : {paths['bench']}")
    print(f"[time_scale_gemmini_logs] turbostat: {paths['turbostat']}")
    print(f"[time_scale_gemmini_logs] power   : {paths['power_csv']}")
    if args.guest_util_csv:
        print(f"[time_scale_gemmini_logs] guest util: {paths['guest_util_csv']}")
    print(f"[time_scale_gemmini_logs] summary : {paths['summary']}")
    print(f"[time_scale_gemmini_logs] mode    : {summary['time_scale_mode']}")
    if summary["gemmini_time_scale"] is None:
        model_scale_text = ", ".join(
            f"{model}={scale:.12g}"
            for model, scale in sorted((summary.get("model_time_scales") or {}).items())
        )
        print(f"[time_scale_gemmini_logs] scales  : {model_scale_text}")
    else:
        print(f"[time_scale_gemmini_logs] scale   : {summary['gemmini_time_scale']:.12g}")
    if summary["rtl_cycle_total"] is not None:
        print(
            "[time_scale_gemmini_logs] RTL     : "
            f"{summary['rtl_cycle_total']} cycles, "
            f"{summary['rtl_time_s']:.9f} s at {summary['rtl_frequency_mhz']:g} MHz"
        )
    if summary['cpu_ref_target_region_s'] is not None:
        print(
            "[time_scale_gemmini_logs] ratio   : "
            f"cpu/gemmini={summary['cpu_to_gemmini_raw_time_ratio']:.6f}, "
            f"target-ratio={summary['target_ratio']:.6f}"
        )
    print(
        "[time_scale_gemmini_logs] scaled inference time : "
        f"{summary['inference_scaled_time_s']:.6f} s"
    )
    print(
        "[time_scale_gemmini_logs] scaled energy : "
        f"{summary['energy_scaled_j']:.6f} J"
    )


if __name__ == "__main__":
    main()
