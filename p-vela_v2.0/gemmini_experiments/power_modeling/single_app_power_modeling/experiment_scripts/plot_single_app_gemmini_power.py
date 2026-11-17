#!/usr/bin/env python3

"""Plot a single application's Gemmini/vCPU power and utilization traces."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.patches import Patch


HOST_TS_PATTERN = re.compile(r"^\[HOST_TS_REALTIME\]\s+(\S+)\s+([0-9]+\.[0-9]+)\s*$")
HOST_TS_LINE_PATTERN = re.compile(r"^\[HOST_TS_LINE\]\s+([0-9]+\.[0-9]+)\s+(.*)$")
TIMELINE_PATTERN = re.compile(r"^\[timeline\]\s+actor=(\S+)\s+label=(\S+)\s+sample=(\d+)\s+thread=(\d+)\s+cpu=(-?\d+)\s*$")
GEMM_TIMELINE_PATTERN = re.compile(r"^\[timeline\]\s+t_ms=\S+\s+event=(gemm_begin|gemm_end)\s+rep=\d+\s*")
SAMPLE_START_PATTERN = re.compile(r"^\[sample-start\]\s+")
SAMPLE_DONE_PATTERN = re.compile(r"^\[sample-done\]\s+")
CONFIG_PATTERN = re.compile(r"^\[config\]\s+")
GEMMINI_EXECUTOR_START_PATTERN = re.compile(r"^\[gemmini-executor-start\]\s+")
LAYER_OP_PATTERN = re.compile(r"^L(\d+)_(\S+)$")
UTIL_COLOR_HOUSEKEEPING = "#9467BD"
UTIL_COLOR_EXECUTOR = "#7F7F7F"

UTIL_STYLE_VCPU = {
    "color": "#8FD0F2",
    "linewidth": 0.9,
    "marker": "o",
    "markersize": 1.4,
}
UTIL_STYLE_GEMMINI = {
    "color": "#08519C",
    "linewidth": 0.9,
    "alpha": 0.92,
    "marker": "o",
    "markersize": 1.4,
}
POWER_STYLE_VCPU = {
    "color": "#8FD0F2",
    "linewidth": 0.8,
}
POWER_STYLE_GEMMINI = {
    "color": "#08519C",
    "linewidth": 0.8,
}
POWER_STYLE_TOTAL = {
    "color": "#D62728",
    "linewidth": 0.9,
}
TIMELINE_STYLE_GEMMINI = {
    "color": "#D62728",
    "linestyle": ":",
    "linewidth": 0.55,
    "alpha": 0.65,
}
TIMELINE_STYLE_VCPU = {
    "color": "#D62728",
    "linestyle": ":",
    "linewidth": 0.55,
    "alpha": 0.65,
}
TIMELINE_STYLE_PREPROCESS = {
    "color": "#8C8C8C",
    "linestyle": ":",
    "linewidth": 0.55,
    "alpha": 0.62,
}
TIMELINE_STYLE_BOUNDARY = {
    "color": "#222222",
    "linestyle": "--",
    "linewidth": 0.65,
    "alpha": 0.78,
}
REFERENCE_STYLE_BENCH = {
    "color": "green",
    "linestyle": "--",
    "linewidth": 0.9,
    "alpha": 0.92,
}
REFERENCE_STYLE_MAIN_START = {
    "color": "#D62728",
    "linestyle": "--",
    "linewidth": 0.9,
    "alpha": 0.9,
}
REFERENCE_STYLE_INFERENCE_START = {
    "color": "#222222",
    "linestyle": "--",
    "linewidth": 0.9,
    "alpha": 0.9,
}
COMBINED_TIMELINE_TEXT_GEMMINI = "#D62728"
COMBINED_TIMELINE_TEXT_VCPU = "#D62728"
TIMELINE_TEXT_PREPROCESS = "#6A6A6A"
TIMELINE_TEXT_BOUNDARY = "#222222"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot single-application Gemmini/vCPU utilization and modeled power"
    )
    parser.add_argument(
        "--host-gemmini-util-only", action="store_true",
        help="output only raw_busy_core_<N> from the power CSV",
    )
    parser.add_argument(
        "--raw-util-plots", action="store_true",
        help="write raw host Gemmini and guest main/executor/housekeeping utilization plots",
    )
    parser.add_argument(
        "--power-csv", required=True,
        help="raw or time-scaled power CSV, including modeled Gemmini utilization",
    )
    parser.add_argument("--bench", required=True, help="benchmark log path")
    parser.add_argument("--out", required=True, help="output figure path")
    parser.add_argument(
        "--gemmini-core",
        type=int,
        default=2,
        help="host core used by the Gemmini worker thread (default: 2)",
    )
    parser.add_argument(
        "--vcpu-core",
        type=int,
        default=8,
        help="host core backing the guest application vCPU (default: 8)",
    )
    parser.add_argument(
        "--vcpu-cores",
        type=int,
        nargs="+",
        default=None,
        help="host cores backing guest application vCPUs; defaults to --vcpu-core",
    )
    parser.add_argument(
        "--guest-vcpus",
        type=int,
        nargs="+",
        required=True,
        help="guest vCPU IDs corresponding one-to-one with --vcpu-cores",
    )
    parser.add_argument(
        "--housekeeping-core",
        type=int,
        default=None,
        help="optional host core backing the observed housekeeping vCPU",
    )
    parser.add_argument(
        "--guest-housekeeping-vcpu",
        type=int,
        default=None,
        help="optional observed guest housekeeping vCPU ID",
    )
    parser.add_argument(
        "--executor-core",
        type=int,
        default=None,
        help="optional host core backing the observed executor vCPU",
    )
    parser.add_argument(
        "--guest-executor-vcpu",
        type=int,
        default=None,
        help="optional observed guest executor vCPU ID",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="optional plot title override",
    )
    parser.add_argument(
        "--time-zero-event",
        default=None,
        help="timeline label or bench marker to use as x=0, for example process_start or sample_start",
    )
    parser.add_argument(
        "--crop-before-time-zero",
        action="store_true",
        help="hide samples before --time-zero-event",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=10.5,
        help="figure width in inches",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="use compact presentation labels for the combined util/power plot",
    )
    parser.add_argument("--ai-color", default=None, help="optional AI Engine color override")
    parser.add_argument("--riscv-color", default=None, help="optional RISC-V color override")
    parser.add_argument("--app-name", default=None, help="presentation title")
    parser.add_argument(
        "--x-display-scale",
        type=float,
        default=1.0,
        help="visual-only multiplier for plotted x offsets; does not change input timestamps",
    )
    parser.add_argument(
        "--x-tick-interval-s",
        type=float,
        default=None,
        help="optional X-axis major tick interval in seconds",
    )
    parser.add_argument(
        "--plot-window",
        choices=("full", "inference"),
        default="full",
        help="crop plotted x-axis to the full run or inference window",
    )
    parser.add_argument(
        "--plot-window-pre-s",
        type=float,
        default=0.5,
        help="seconds to show before the inference window when --plot-window inference",
    )
    parser.add_argument(
        "--plot-window-post-s",
        type=float,
        default=0.5,
        help="seconds to show after the inference window when --plot-window inference",
    )
    args = parser.parse_args()
    if args.fig_width <= 0.0:
        parser.error("--fig-width must be > 0")
    if args.x_tick_interval_s is not None and args.x_tick_interval_s <= 0.0:
        parser.error("--x-tick-interval-s must be > 0")
    if args.x_display_scale <= 0.0:
        parser.error("--x-display-scale must be > 0")
    if args.plot_window_pre_s < 0.0 or args.plot_window_post_s < 0.0:
        parser.error("plot window padding must be >= 0")
    if args.vcpu_cores is None:
        args.vcpu_cores = [args.vcpu_core]
    else:
        args.vcpu_core = args.vcpu_cores[0]
    if len(args.guest_vcpus) != len(args.vcpu_cores):
        parser.error("--guest-vcpus must have the same length as --vcpu-cores")
    if (args.housekeeping_core is None) != (
        args.guest_housekeeping_vcpu is None
    ):
        parser.error(
            "--housekeeping-core and --guest-housekeeping-vcpu must be used together"
        )
    if args.housekeeping_core is not None:
        if args.housekeeping_core < 0 or args.guest_housekeeping_vcpu < 0:
            parser.error("housekeeping core and vCPU IDs must be non-negative")
        if args.housekeeping_core == args.gemmini_core:
            parser.error("--housekeeping-core must differ from --gemmini-core")
        if args.housekeeping_core in args.vcpu_cores:
            parser.error("--housekeeping-core must differ from --vcpu-cores")
        if args.guest_housekeeping_vcpu in args.guest_vcpus:
            parser.error(
                "--guest-housekeeping-vcpu must differ from --guest-vcpus"
            )
    if (args.executor_core is None) != (args.guest_executor_vcpu is None):
        parser.error(
            "--executor-core and --guest-executor-vcpu must be used together"
        )
    if args.executor_core is not None:
        if args.executor_core < 0 or args.guest_executor_vcpu < 0:
            parser.error("executor core and vCPU IDs must be non-negative")
        if args.executor_core == args.gemmini_core:
            parser.error("--executor-core must differ from --gemmini-core")
        if args.executor_core in args.vcpu_cores:
            parser.error("--executor-core must differ from --vcpu-cores")
        if args.housekeeping_core is not None and (
            args.executor_core == args.housekeeping_core
        ):
            parser.error("--executor-core must differ from --housekeeping-core")
        if args.guest_executor_vcpu in args.guest_vcpus:
            parser.error("--guest-executor-vcpu must differ from --guest-vcpus")
        if args.guest_housekeeping_vcpu is not None and (
            args.guest_executor_vcpu == args.guest_housekeeping_vcpu
        ):
            parser.error("executor and housekeeping guest vCPUs must differ")
    return args


def parse_bench_markers(path):
    markers = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            match = HOST_TS_PATTERN.match(raw.strip())
            if match:
                markers[match.group(1)] = float(match.group(2))
    return markers


def parse_timeline_events(path):
    events = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            host_match = HOST_TS_LINE_PATTERN.match(raw.strip())
            if not host_match:
                continue

            host_ts = float(host_match.group(1))
            payload = host_match.group(2)
            timeline_match = TIMELINE_PATTERN.match(payload)
            if timeline_match:
                events.append(
                    {
                        "host_ts": host_ts,
                        "actor": timeline_match.group(1),
                        "label": timeline_match.group(2),
                    }
                )
                continue

            gemm_match = GEMM_TIMELINE_PATTERN.match(payload)
            if gemm_match:
                events.append({"host_ts": host_ts, "actor": "gemmini", "label": gemm_match.group(1)})
                continue

            events.append(
                {
                    "host_ts": host_ts,
                    "actor": "sample",
                    "label": "sample_start",
                }
            ) if SAMPLE_START_PATTERN.match(payload) else None

            events.append(
                {
                    "host_ts": host_ts,
                    "actor": "sample",
                    "label": "sample_done",
                }
            ) if SAMPLE_DONE_PATTERN.match(payload) else None

            events.append(
                {
                    "host_ts": host_ts,
                    "actor": "setup",
                    "label": "config",
                }
            ) if CONFIG_PATTERN.match(payload) else None

            events.append(
                {
                    "host_ts": host_ts,
                    "actor": "setup",
                    "label": "gemmini_executor_start",
                }
            ) if GEMMINI_EXECUTOR_START_PATTERN.match(payload) else None
    events.sort(key=lambda event: event["host_ts"])
    return events


def parse_turbostat(path, target_cores):
    header = None
    rows = defaultdict(list)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            tokens = line.split()
            if "Time_Of_Day_Seconds" in tokens and "CPU" in tokens and "Busy%" in tokens:
                header = tokens
                continue

            if header is None or len(tokens) < len(header):
                continue

            try:
                tod = float(tokens[header.index("Time_Of_Day_Seconds")])
                cpu = int(tokens[header.index("CPU")])
                busy = float(tokens[header.index("Busy%")])
            except (ValueError, IndexError):
                continue

            if cpu in target_cores:
                rows[cpu].append((tod, busy))

    return rows


def parse_guest_util_csv(path, vcpu_cores, guest_vcpus):
    rows = defaultdict(list)
    expected_guest_by_core = dict(zip(vcpu_cores, guest_vcpus))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch_s = int(row["epoch_ns"]) / 1_000_000_000.0
                model_core = int(row["model_core"])
                guest_vcpu = int(row["guest_vcpu"])
                busy = float(row["busy_percent"])
            except (KeyError, ValueError):
                continue

            expected_guest = expected_guest_by_core.get(model_core)
            if expected_guest is None:
                continue
            if guest_vcpu != expected_guest:
                raise SystemExit(
                    "guest utilization mapping mismatch: "
                    f"model core {model_core} expected guest vCPU {expected_guest}, "
                    f"got {guest_vcpu}"
                )
            rows[model_core].append((epoch_s, busy))

    missing = [core for core in vcpu_cores if not rows[core]]
    if missing:
        raise SystemExit(f"No guest Busy% samples for model cores: {missing}")
    for samples in rows.values():
        samples.sort(key=lambda sample: sample[0])
    return rows


def parse_power_csv(path, gemmini_core, vcpu_cores, require_raw_busy=False, observed_cores=()):
    xs, gemmini_power, total_power, gemmini_busy, raw_busy = [], [], [], [], []
    vcpu_power = {core: [] for core in vcpu_cores}
    vcpu_busy = {core: [] for core in [*vcpu_cores, *observed_cores]}
    gemmini_key = f"power_gemmini_core_{gemmini_core}"
    util_key = f"util_core_{gemmini_core}"
    raw_key = f"raw_busy_core_{gemmini_core}"
    required = ["sample_tod_s", gemmini_key, util_key, "p_total_w"]
    required += [f"power_vcpu_core_{core}" for core in vcpu_cores]
    required += [f"busy_core_{core}" for core in vcpu_busy]
    if require_raw_busy:
        required.append(raw_key)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Power CSV missing columns: {', '.join(sorted(missing))}. "
                "For raw host utilization, collect a new run with raw_busy_core_<N>."
            )
        for row in reader:
            try:
                values = {key: float(row[key]) for key in required}
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid power CSV data at line {reader.line_num}") from exc
            xs.append(values["sample_tod_s"])
            gemmini_power.append(values[gemmini_key])
            total_power.append(values["p_total_w"])
            gemmini_busy.append(100.0 * values[util_key])
            if require_raw_busy:
                raw_busy.append(values[raw_key])
            for core in vcpu_cores:
                vcpu_power[core].append(values[f"power_vcpu_core_{core}"])
            for core in vcpu_busy:
                vcpu_busy[core].append(values[f"busy_core_{core}"])
    return xs, gemmini_power, vcpu_power, total_power, gemmini_busy, vcpu_busy, raw_busy


def build_extra_output_path(base_path, suffix):
    if "." in base_path.rsplit("/", 1)[-1]:
        stem, ext = base_path.rsplit(".", 1)
        return f"{stem}_{suffix}.{ext}"
    return f"{base_path}_{suffix}.png"


def resolve_time_zero(args, rows, power_times, markers, events):
    default_first_ts = min(
        min(samples[0][0] for samples in rows.values() if samples),
        power_times[0],
    )
    if args.time_zero_event is None:
        return default_first_ts

    marker_ts = markers.get(args.time_zero_event)
    if marker_ts is not None:
        return marker_ts

    matches = [
        event["host_ts"]
        for event in events
        if event["label"] == args.time_zero_event
    ]
    if not matches:
        raise SystemExit(f"No timeline event or bench marker matched: {args.time_zero_event}")
    return min(matches)


def maybe_crop_axis(ax, args):
    if args.crop_before_time_zero:
        ax.set_xlim(left=0)


def add_right_margin(ax, fraction=0.04):
    left, right = ax.get_xlim()
    width = right - left
    if width > 0:
        ax.set_xlim(left, right + width * fraction)


def x_offset(ts, first_ts, x_scale):
    return (ts - first_ts) * x_scale


def x_axis_label(x_scale):
    if abs(x_scale - 1.0) < 1e-9:
        return "Host realtime offset (s)"
    return f"Display offset (s, x{x_scale:g})"


def apply_x_tick_interval(ax, args):
    if args.x_tick_interval_s is None:
        return
    interval = args.x_tick_interval_s * args.x_display_scale
    ax.xaxis.set_major_locator(MultipleLocator(interval))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", labelsize=8)


def add_compact_legend(ax):
    ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.78,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=2.0,
    )


def add_presentation_legend(ax, metric):
    if metric == "utilization":
        handles = [
            Patch(facecolor=UTIL_STYLE_GEMMINI["color"], label="AI Engine Utilization"),
            Patch(facecolor=UTIL_STYLE_VCPU["color"], label="RISC-V Utilization"),
        ]
    else:
        handles = [
            Patch(facecolor=POWER_STYLE_GEMMINI["color"], label="AI Engine Power"),
            Patch(facecolor=POWER_STYLE_VCPU["color"], label="RISC-V Power"),
            Patch(facecolor=POWER_STYLE_TOTAL["color"], label="Total Power"),
        ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=11,
        prop={"size": 11, "weight": "bold"},
        framealpha=0.9,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=1.5,
        handleheight=0.55,
    )


def add_presentation_reference_lines(ax, args, events, first_ts, x_scale):
    if args.app_name == "PolyBench GEMM":
        starts = [event["host_ts"] for event in events if event["label"] == "gemm_begin"]
        ends = [event["host_ts"] for event in events if event["label"] == "gemm_end"]
    else:
        starts = [event["host_ts"] for event in events if event["label"] == "sample_start"]
        ends = [event["host_ts"] for event in events if event["label"] == "sample_done"]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    if start is not None:
        ax.axvline(x_offset(start, first_ts, x_scale), label="_nolegend_", **REFERENCE_STYLE_BENCH)
    if end is not None:
        ax.axvline(x_offset(end, first_ts, x_scale), label="_nolegend_", color="#D62728", linestyle="--", linewidth=0.9, alpha=0.92)


def first_event_ts(events, label):
    matches = [event["host_ts"] for event in events if event["label"] == label]
    if not matches:
        return None
    return min(matches)


def add_reference_lines(ax, markers, events, first_ts, x_scale):
    start = markers.get("BENCH_HOST_START")
    end = markers.get("BENCH_HOST_END")
    main_start = first_event_ts(events, "process_start")
    inference_start = first_vcpu_inference_ts(events)

    if start is not None:
        ax.axvline(x_offset(start, first_ts, x_scale), label="bench start/end", **REFERENCE_STYLE_BENCH)
    if end is not None:
        ax.axvline(x_offset(end, first_ts, x_scale), label="_nolegend_", **REFERENCE_STYLE_BENCH)
    if main_start is not None:
        ax.axvline(x_offset(main_start, first_ts, x_scale), label="main start", **REFERENCE_STYLE_MAIN_START)
    if inference_start is not None:
        ax.axvline(x_offset(inference_start, first_ts, x_scale), label="inference start", **REFERENCE_STYLE_INFERENCE_START)


def relative_busy_xy(rows, core, first_ts, x_scale):
    samples = rows.get(core, [])
    xs = [x_offset(ts, first_ts, x_scale) for ts, _ in samples]
    ys = [busy for _, busy in samples]
    return xs, ys


def build_title(args):
    if args.title:
        return args.title
    return (
        f"{args.app_name or 'Single application'} util/power "
        f"(vCPU cores {args.vcpu_cores}, Gemmini core {args.gemmini_core})"
    )


def vcpu_role_label(_idx, core):
    return f"application vCPU core {core}"


def vcpu_power_style(idx):
    if idx == 0:
        return POWER_STYLE_VCPU
    return {
        "linewidth": 0.7,
        "linestyle": "--",
    }


def vcpu_util_style(idx):
    if idx == 0:
        return UTIL_STYLE_VCPU
    return {
        "linewidth": 0.7,
        "linestyle": "--",
        "marker": "o",
        "markersize": 1.4,
    }


def host_vcpu_util_style(idx):
    style = dict(vcpu_util_style(idx))
    style.update(
        {
            "linestyle": "--",
            "linewidth": 1.0,
            "marker": None,
            "alpha": 0.9,
        }
    )
    return style


def guest_vcpu_util_label(_idx, guest_vcpu):
    return f"guest application vCPU {guest_vcpu}"


def host_vcpu_util_label(_idx, core, guest_vcpu):
    return f"host core {core} backing application vCPU {guest_vcpu} (turbostat)"

def guest_housekeeping_util_label(guest_vcpu):
    return f"guest housekeeping vCPU {guest_vcpu}"


def host_housekeeping_util_label(core, guest_vcpu):
    return (
        f"host core {core}: housekeeping vCPU {guest_vcpu} + "
        "QEMU auxiliary (turbostat)"
    )

def guest_executor_util_label(guest_vcpu):
    return f"guest Gemmini executor vCPU {guest_vcpu}"


def host_executor_util_label(core, guest_vcpu):
    return f"host core {core} backing Gemmini executor vCPU {guest_vcpu} (turbostat)"



def prettify_event_label(label):
    return label.replace("_", " ")


def setup_phase_label(label):
    if label == "model_encoder":
        return "model load"
    return None


def inference_phase_label(label):
    if label == "embeddings":
        return "embeddings"
    if label == "pre_cls":
        return "pre cls"
    if label == "cls_relu":
        return "cls relu"
    if label == "cls":
        return "cls"

    layer_match = LAYER_OP_PATTERN.match(label)
    if not layer_match:
        return prettify_event_label(label)

    layer_idx = layer_match.group(1)
    op = layer_match.group(2)
    if op == "qkv":
        return f"L{layer_idx} qkv"
    if op == "attn":
        return f"L{layer_idx} attn"
    if op == "out":
        return f"L{layer_idx} out"
    if op == "sa_norm":
        return f"L{layer_idx} sa norm"
    if op == "ffn1":
        return f"L{layer_idx} ffn1"
    if op == "gelu":
        return f"L{layer_idx} gelu"
    if op == "ffn2":
        return f"L{layer_idx} ffn2"
    if op == "out_norm":
        return f"L{layer_idx} out norm"
    return None


def display_event_label(event, inference_start):
    if event["label"] == "sample_start":
        return None
    if event["label"] == "sample_done":
        return None
    if inference_start is not None and event["host_ts"] < inference_start:
        return setup_phase_label(event["label"])
    return inference_phase_label(event["label"])


def compact_events_for_labels(events, inference_start):
    compacted = []
    seen_setup_labels = set()
    for event in events:
        if event["label"] == "process_start":
            continue
        label = display_event_label(event, inference_start)
        if label is None:
            continue
        is_setup = inference_start is not None and event["host_ts"] < inference_start
        if is_setup:
            if label in seen_setup_labels:
                continue
            seen_setup_labels.add(label)
        compacted.append({**event, "display_label": label})
    return compacted


def first_vcpu_inference_ts(events):
    matches = [
        event["host_ts"]
        for event in events
        if event["label"] in ("sample_start", "embeddings")
    ]
    if not matches:
        return None
    return min(matches)


def last_vcpu_inference_ts(events, markers):
    matches = [
        event["host_ts"]
        for event in events
        if event["label"] == "sample_done"
    ]
    if matches:
        return max(matches)
    return markers.get("BENCH_HOST_END")


def apply_plot_window(ax, args, markers, events, first_ts):
    if args.plot_window != "inference":
        maybe_crop_axis(ax, args)
        return

    if args.app_name == "PolyBench GEMM":
        starts = [event["host_ts"] for event in events if event["label"] == "gemm_begin"]
        ends = [event["host_ts"] for event in events if event["label"] == "gemm_end"]
        start = min(starts) if starts else None
        end = max(ends) if ends else None
    else:
        start = first_vcpu_inference_ts(events)
        end = last_vcpu_inference_ts(events, markers)
    if start is None or end is None:
        maybe_crop_axis(ax, args)
        return

    left = x_offset(start - args.plot_window_pre_s, first_ts, args.x_display_scale)
    right = x_offset(end + args.plot_window_post_s, first_ts, args.x_display_scale)
    if args.crop_before_time_zero:
        left = max(0.0, left)
    if right > left:
        ax.set_xlim(left, right)


def event_style(event, inference_start):
    if event["label"] in ("sample_start", "sample_done"):
        return TIMELINE_STYLE_BOUNDARY, TIMELINE_TEXT_BOUNDARY
    if inference_start is not None and event["host_ts"] < inference_start:
        return TIMELINE_STYLE_PREPROCESS, TIMELINE_TEXT_PREPROCESS
    if event["actor"] == "gemmini":
        return TIMELINE_STYLE_GEMMINI, COMBINED_TIMELINE_TEXT_GEMMINI
    return TIMELINE_STYLE_VCPU, COMBINED_TIMELINE_TEXT_VCPU


def annotate_timeline(ax, events, actor, first_ts, x_scale, show_text=True):
    inference_start = first_vcpu_inference_ts(events)
    events = compact_events_for_labels(events, inference_start)
    if actor == "vcpu":
        actor_events = [
            event for event in events
            if event["actor"] in ("vcpu", "setup", "sample")
        ]
    else:
        actor_events = [
            event for event in events
            if event["actor"] in (actor, "setup", "sample")
        ]

    y_positions = [102, 95, 88, 81]

    for idx, event in enumerate(actor_events):
        x = x_offset(event["host_ts"], first_ts, x_scale)
        style, text_color = event_style(event, inference_start)
        ax.axvline(x, **style)
        if not show_text:
            continue
        ax.text(
            x,
            y_positions[idx % len(y_positions)],
            event["display_label"],
            rotation=90,
            va="top",
            ha="right",
            fontsize=6.5,
            color=text_color,
            alpha=0.9,
            clip_on=True,
        )


def annotate_combined_util_timeline(ax, events, first_ts, x_scale, show_text=True):
    inference_start = first_vcpu_inference_ts(events)
    events = compact_events_for_labels(events, inference_start)
    y_positions = {
        "pre": [102, 95, 88, 81, 74, 67],
        "gemmini": [102, 95, 88, 81],
        "vcpu": [74, 67, 60, 53],
    }
    y_counts = defaultdict(int)

    for event in events:
        x = x_offset(event["host_ts"], first_ts, x_scale)
        style, text_color = event_style(event, inference_start)
        ax.axvline(x, **style)
        if not show_text:
            continue

        if inference_start is not None and event["host_ts"] < inference_start:
            lane = "pre"
        elif event["actor"] == "gemmini":
            lane = "gemmini"
        else:
            lane = "vcpu"
        y = y_positions[lane][y_counts[lane] % len(y_positions[lane])]
        y_counts[lane] += 1
        ax.text(
            x,
            y,
            event["display_label"],
            rotation=90,
            va="top",
            ha="right",
            fontsize=6.0,
            color=text_color,
            alpha=0.9,
            clip_on=True,
        )


def save_util_series_plot(
    out_path,
    series,
    title,
    markers,
    first_ts,
    events,
    actor,
    args,
    show_timeline_text=True,
    bench_window=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(args.fig_width, 3.6))
    for xs, ys, style, label in series:
        ax.plot(xs, ys, label=label, **style)
    if args.presentation:
        # Presentation plots show only the overall workload boundaries.  Drawing
        # every PolyBench GEMM begin/end event produces hundreds of red lines.
        add_presentation_reference_lines(
            ax, args, events, first_ts, args.x_display_scale
        )
    else:
        add_reference_lines(ax, markers, events, first_ts, args.x_display_scale)
        annotate_timeline(
            ax,
            events,
            actor,
            first_ts,
            args.x_display_scale,
            show_text=show_timeline_text,
        )
    ax.set_xlabel(x_axis_label(args.x_display_scale))
    ax.set_ylabel("Busy%")
    ax.set_ylim(0, 105)
    bench_start = markers.get("BENCH_HOST_START")
    bench_end = markers.get("BENCH_HOST_END")
    if (
        bench_window
        and args.plot_window != "inference"
        and bench_start is not None
        and bench_end is not None
    ):
        ax.set_xlim(
            x_offset(bench_start, first_ts, args.x_display_scale),
            x_offset(bench_end, first_ts, args.x_display_scale),
        )
    else:
        apply_plot_window(ax, args, markers, events, first_ts)
        if args.plot_window != "inference":
            add_right_margin(ax)
    apply_x_tick_interval(ax, args)
    ax.grid(True, alpha=0.3)
    add_compact_legend(ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.ai_color:
        UTIL_STYLE_GEMMINI["color"] = args.ai_color
        POWER_STYLE_GEMMINI["color"] = args.ai_color
    if args.riscv_color:
        UTIL_STYLE_VCPU["color"] = args.riscv_color
        POWER_STYLE_VCPU["color"] = args.riscv_color
    if args.presentation:
        POWER_STYLE_GEMMINI["color"] = "#000000"
        POWER_STYLE_VCPU["color"] = "#7F7F7F"

    observed_cores = [core for core in (args.executor_core, args.housekeeping_core)
                      if core is not None] if args.raw_util_plots else []
    (power_times, gemmini_power, vcpu_power_by_core, total_power,
     gemmini_busy_ys, vcpu_busy_by_core, raw_busy_ys) = parse_power_csv(
        args.power_csv, args.gemmini_core, args.vcpu_cores,
        require_raw_busy=args.host_gemmini_util_only or args.raw_util_plots,
        observed_cores=observed_cores,
    )
    if not power_times:
        raise SystemExit("No power samples were parsed")

    markers = parse_bench_markers(args.bench)
    timeline_events = parse_timeline_events(args.bench)
    rows = {args.gemmini_core: list(zip(power_times, gemmini_busy_ys))}
    first_ts = resolve_time_zero(args, rows, power_times, markers, timeline_events)
    power_xs = [x_offset(ts, first_ts, args.x_display_scale) for ts in power_times]
    if args.host_gemmini_util_only or args.raw_util_plots:
        save_util_series_plot(
            args.out,
            [(power_xs, raw_busy_ys, UTIL_STYLE_GEMMINI, f"Host Gemmini core {args.gemmini_core}")],
            f"Host Gemmini worker utilization (core {args.gemmini_core})",
            markers, first_ts, timeline_events, "gemmini", args,
        )
        print(f"[single_app_util_power] plot : {args.out}")
        if args.raw_util_plots:
            guest_plots = [(
                "raw_guest_vcpu_util.png", "Guest application vCPU utilization",
                [(power_xs, vcpu_busy_by_core[core], vcpu_util_style(idx),
                  guest_vcpu_util_label(idx, guest_vcpu))
                 for idx, (core, guest_vcpu) in enumerate(zip(args.vcpu_cores, args.guest_vcpus))],
            )]
            for core, guest, role, color in (
                (args.executor_core, args.guest_executor_vcpu, "executor", UTIL_COLOR_EXECUTOR),
                (args.housekeeping_core, args.guest_housekeeping_vcpu, "housekeeping", UTIL_COLOR_HOUSEKEEPING),
            ):
                if core is not None:
                    suffix = "executor_vcpu" if role == "executor" else role
                    guest_plots.append((
                        f"raw_guest_{suffix}_util.png", f"Guest {role} vCPU utilization",
                        [(power_xs, vcpu_busy_by_core[core],
                          {**vcpu_util_style(0), "color": color, "alpha": 1.0},
                          f"guest {role} vCPU {guest}")],
                    ))
            for filename, title, series in guest_plots:
                out = str(Path(args.out).with_name(filename))
                save_util_series_plot(out, series, title, markers, first_ts,
                                      timeline_events, "vcpu", args, bench_window=True)
                print(f"[single_app_util_power] plot : {out}")
        return

    figure_height = 5.4 if args.presentation else 6.8
    fig, axes = plt.subplots(2, 1, figsize=(args.fig_width, figure_height), sharex=True)

    # The collector already modeled Gemmini utilization; use the same CSV
    # samples as power, including any time-scaled/interpolated values.
    gemmini_busy_xs = power_xs
    for idx, (core, guest_vcpu) in enumerate(zip(args.vcpu_cores, args.guest_vcpus)):
        axes[0].plot(
            power_xs,
            vcpu_busy_by_core[core],
            label=guest_vcpu_util_label(idx, guest_vcpu),
            **vcpu_util_style(idx),
        )
    gemmini_util_plot_style = dict(UTIL_STYLE_GEMMINI)
    if args.presentation:
        gemmini_util_plot_style.pop("marker", None)
        gemmini_util_plot_style.pop("markersize", None)
    axes[0].plot(
        gemmini_busy_xs,
        gemmini_busy_ys,
        label=f"Gemmini core {args.gemmini_core}",
        **gemmini_util_plot_style,
    )
    if args.presentation:
        add_presentation_reference_lines(axes[0], args, timeline_events, first_ts, args.x_display_scale)
    else:
        add_reference_lines(axes[0], markers, timeline_events, first_ts, args.x_display_scale)
    axes[0].set_ylabel("Utilization (%)")
    axes[0].set_ylim(-5 if args.presentation else 0, 105)
    apply_plot_window(axes[0], args, markers, timeline_events, first_ts)
    if args.plot_window != "inference":
        add_right_margin(axes[0])
    apply_x_tick_interval(axes[0], args)
    axes[0].grid(True, alpha=0.3)
    if args.presentation:
        add_presentation_legend(axes[0], "utilization")
    else:
        add_compact_legend(axes[0])

    for idx, core in enumerate(args.vcpu_cores):
        axes[1].plot(
            power_xs,
            vcpu_power_by_core.get(core, []),
            label=f"{vcpu_role_label(idx, core)} power",
            **vcpu_power_style(idx),
        )
    axes[1].plot(
        power_xs,
        gemmini_power,
        label=f"Gemmini core {args.gemmini_core} power",
        **POWER_STYLE_GEMMINI,
    )
    axes[1].plot(
        power_xs,
        total_power,
        label="total power",
        **POWER_STYLE_TOTAL,
    )
    if args.presentation:
        add_presentation_reference_lines(axes[1], args, timeline_events, first_ts, args.x_display_scale)
    else:
        add_reference_lines(axes[1], markers, timeline_events, first_ts, args.x_display_scale)
    axes[1].set_xlabel("Time (s)" if args.presentation else x_axis_label(args.x_display_scale))
    axes[1].set_ylabel("Power (W)")
    apply_plot_window(axes[1], args, markers, timeline_events, first_ts)
    if args.plot_window != "inference":
        add_right_margin(axes[1])
    apply_x_tick_interval(axes[1], args)
    axes[1].grid(True, alpha=0.3)
    if args.presentation:
        add_presentation_legend(axes[1], "power")
    else:
        add_compact_legend(axes[1])

    if args.presentation:
        fig.suptitle(args.app_name or "Application", fontsize=18, fontweight="bold", y=0.995)
    else:
        fig.suptitle(build_title(args))
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"[single_app_util_power] plot : {args.out}")


if __name__ == "__main__":
    main()
