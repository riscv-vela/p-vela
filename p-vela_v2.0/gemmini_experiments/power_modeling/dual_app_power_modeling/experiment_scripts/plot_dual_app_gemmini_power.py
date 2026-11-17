#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


HOST_TS_PATTERN = re.compile(r"^\[HOST_TS_REALTIME\]\s+(\S+)\s+([0-9]+\.[0-9]+)\s*$")
HOST_TS_LINE_PATTERN = re.compile(
    r"^\[HOST_TS_LINE\]\s+([0-9]+\.[0-9]+)\s+\[([^\]]+)\]\s+(.*)$"
)
TIMELINE_PATTERN = re.compile(
    r"^\[timeline\]\s+actor=(\S+)\s+label=(\S+)\s+sample=(\d+)\s+thread=(\d+)\s+cpu=(-?\d+)\s*$"
)
POLYBENCH_TIMELINE_PATTERN = re.compile(
    r"^\[timeline\]\s+t_ms=\S+\s+event=(\S+)(?:\s+rep=(\d+))?\s*$"
)
SAMPLE_START_PATTERN = re.compile(r"^\[sample-start\]\s+")
SAMPLE_DONE_PATTERN = re.compile(r"^\[sample-done\]\s+")
POLYBENCH_DONE_PATTERN = re.compile(r"^\[done\]\s+mode=gemmini_power\b")
LAYER_OP_PATTERN = re.compile(r"^L(\d+)_(\S+)$")

APP_STYLES = {
    "distilbert": {
        "main_color": "#8FD0F2",
        "executor_color": "#7F7F7F",
        "gemmini_color": "#08519C",
        "total_color": "#1F77B4",
    },
    "distilroberta": {
        "main_color": "#9AD29A",
        "executor_color": "#7F7F7F",
        "gemmini_color": "#1B7F2A",
        "total_color": "#2CA02C",
    },
    "distilgpt2": {
        "main_color": "#C9A0DC",
        "executor_color": "#7F7F7F",
        "gemmini_color": "#6F3FA0",
        "total_color": "#9467BD",
    },
    "polybench_gemm": {
        "main_color": "#C49A6C",
        "executor_color": "#7F7F7F",
        "gemmini_color": "#8B5A2B",
        "total_color": "#8C564B",
    },
}
FALLBACK_STYLE = {
    "main_color": "#8FD0F2",
    "executor_color": "#7F7F7F",
    "gemmini_color": "#08519C",
    "total_color": "#1F77B4",
}
OVERALL_TOTAL_COLOR = "#D62728"
HOUSEKEEPING_COLOR = "#9467BD"
BENCH_COLOR = "#2CA02C"
PROCESS_COLOR = "#7A7A7A"
SAMPLE_COLOR = "#D62728"
TIMELINE_STYLE_GEMMINI = {"color": "#D62728", "linestyle": ":", "linewidth": 0.55, "alpha": 0.65}
TIMELINE_STYLE_VCPU = {"color": "#D62728", "linestyle": ":", "linewidth": 0.55, "alpha": 0.65}
TIMELINE_STYLE_PREPROCESS = {"color": "#8C8C8C", "linestyle": ":", "linewidth": 0.55, "alpha": 0.62}
TIMELINE_TEXT_GEMMINI = "#D62728"
TIMELINE_TEXT_VCPU = "#D62728"
TIMELINE_TEXT_PREPROCESS = "#6A6A6A"
TIME_MAJOR_TICK_S = 10
UTIL_LINEWIDTH = 0.9
POWER_LINEWIDTH = 1.05
POWER_ALPHA = 0.72
PAIR_TOTAL_POWER_ALPHA = 0.96
MARKERSIZE = 1.2
MODEL_FIGSIZE = (13.0, 8.4)
COMBINED_FIGSIZE = (13.8, 8.8)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot generic dual-app Gemmini utilization and power")
    parser.add_argument("--turbostat", required=True)
    parser.add_argument("--guest-util-csv", required=True)
    parser.add_argument("--power-csv", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--app0-id", required=True)
    parser.add_argument("--app0-name", required=True)
    parser.add_argument("--app0-gemmini-core", type=int, required=True)
    parser.add_argument("--app0-main-vcpu-core", type=int, required=True)
    parser.add_argument("--app0-main-guest-vcpu", type=int, required=True)
    parser.add_argument("--app0-gemmini-vcpu-core", type=int, required=True)
    parser.add_argument("--app0-executor-guest-vcpu", type=int, required=True)
    parser.add_argument("--app1-id", required=True)
    parser.add_argument("--app1-name", required=True)
    parser.add_argument("--app1-gemmini-core", type=int, required=True)
    parser.add_argument("--app1-main-vcpu-core", type=int, required=True)
    parser.add_argument("--title", default="dual Gemmini power")
    parser.add_argument("--app1-main-guest-vcpu", type=int, required=True)
    parser.add_argument("--app1-gemmini-vcpu-core", type=int, required=True)
    parser.add_argument("--app1-executor-guest-vcpu", type=int, required=True)
    parser.add_argument("--time-zero-event", default=None)
    parser.add_argument("--housekeeping-vcpu-core", type=int, required=True)
    parser.add_argument("--guest-housekeeping-vcpu", type=int, required=True)
    parser.add_argument("--crop-before-time-zero", action="store_true")
    parser.add_argument("--time-window-before", type=float, default=None)
    parser.add_argument("--time-window-after", type=float, default=None)
    parser.add_argument("--sample-window-margin", type=float, default=None)
    parser.add_argument("--time-major-tick-s", type=float, default=None)
    parser.add_argument("--fig-width", type=float, default=None)
    parser.add_argument("--presentation-util", action="store_true")
    return parser.parse_args()


def model_args(args):
    return [
        {
            "slot": 1,
            "key": args.app0_id,
            "name": args.app0_name,
            "gemmini_core": args.app0_gemmini_core,
            "main_core": args.app0_main_vcpu_core,
            "main_guest_vcpu": args.app0_main_guest_vcpu,
            "executor_core": args.app0_gemmini_vcpu_core,
            "executor_guest_vcpu": args.app0_executor_guest_vcpu,
        },
        {
            "slot": 2,
            "key": args.app1_id,
            "name": args.app1_name,
            "gemmini_core": args.app1_gemmini_core,
            "main_core": args.app1_main_vcpu_core,
            "main_guest_vcpu": args.app1_main_guest_vcpu,
            "executor_core": args.app1_gemmini_vcpu_core,
            "executor_guest_vcpu": args.app1_executor_guest_vcpu,
        },
    ]


def parse_bench_markers(path):
    markers = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            match = HOST_TS_PATTERN.match(raw.strip())
            if match:
                markers[match.group(1)] = float(match.group(2))
    return markers


def parse_events(path, allowed_models):
    events_by_model = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            match = HOST_TS_LINE_PATTERN.match(raw.strip())
            if not match:
                continue

            host_ts = float(match.group(1))
            model_key = match.group(2)
            if model_key not in allowed_models:
                continue
            payload = match.group(3)
            timeline = TIMELINE_PATTERN.match(payload)
            polybench_timeline = POLYBENCH_TIMELINE_PATTERN.match(payload)
            if timeline:
                events_by_model[model_key].append(
                    {
                        "host_ts": host_ts,
                        "actor": timeline.group(1),
                        "label": timeline.group(2),
                        "sample": int(timeline.group(3)),
                    }
                )
            elif polybench_timeline:
                event = polybench_timeline.group(1)
                rep = int(polybench_timeline.group(2) or 0)
                if event in ("init_begin", "init_end"):
                    events_by_model[model_key].append(
                        {"host_ts": host_ts, "actor": "vcpu", "label": event, "sample": rep}
                    )
                elif event == "gemm_begin":
                    events_by_model[model_key].append(
                        {"host_ts": host_ts, "actor": "sample", "label": "sample_start", "sample": rep}
                    )
                    events_by_model[model_key].append(
                        {"host_ts": host_ts, "actor": "gemmini", "label": "gemm_begin", "sample": rep}
                    )
                elif event == "gemm_end":
                    events_by_model[model_key].append(
                        {"host_ts": host_ts, "actor": "gemmini", "label": "gemm_end", "sample": rep}
                    )
                elif event == "done":
                    events_by_model[model_key].append(
                        {"host_ts": host_ts, "actor": "sample", "label": "sample_done", "sample": rep}
                    )
            elif SAMPLE_START_PATTERN.match(payload):
                events_by_model[model_key].append(
                    {"host_ts": host_ts, "actor": "sample", "label": "sample_start", "sample": 0}
                )
            elif SAMPLE_DONE_PATTERN.match(payload):
                events_by_model[model_key].append(
                    {"host_ts": host_ts, "actor": "sample", "label": "sample_done", "sample": 0}
                )
            elif POLYBENCH_DONE_PATTERN.match(payload):
                events_by_model[model_key].append(
                    {"host_ts": host_ts, "actor": "sample", "label": "sample_done", "sample": 0}
                )

    for events in events_by_model.values():
        events.sort(key=lambda event: event["host_ts"])
    return events_by_model


def parse_turbostat(path, target_cores):
    header = None
    rows = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            tokens = raw.strip().split()
            if not tokens:
                continue
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


def parse_guest_util(path, target_vcpus):
    rows = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        required = {"sample_tod_s", "guest_vcpu", "busy_percent"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                "guest utilization CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            try:
                tod = float(row["sample_tod_s"])
                guest_vcpu = int(row["guest_vcpu"])
                busy = float(row["busy_percent"])
            except (TypeError, ValueError):
                continue
            if guest_vcpu in target_vcpus:
                rows[guest_vcpu].append((tod, busy))
    return rows

def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_power_csv(path, gemmini_cores, vcpu_cores):
    power_times = []
    gemmini_per_core = {core: [] for core in gemmini_cores}
    gemmini_util_rows = {core: [] for core in gemmini_cores}
    vcpu_per_core = {core: [] for core in vcpu_cores}
    total_power = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        required_util = {f"util_core_{core}" for core in gemmini_cores}
        missing = required_util - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                "Power CSV is missing Gemmini utilization: " + ", ".join(sorted(missing))
            )
        for row in reader:
            try:
                sample_tod_s = float(row["sample_tod_s"])
                total_p = float(row["p_total_w"])
                gemmini_busy = {
                    core: 100.0 * float(row[f"util_core_{core}"])
                    for core in gemmini_cores
                }
            except (KeyError, TypeError, ValueError):
                continue
            power_times.append(sample_tod_s)
            total_power.append(total_p)
            for core in gemmini_cores:
                gemmini_per_core[core].append(parse_float(row.get(f"power_gemmini_core_{core}")))
                gemmini_util_rows[core].append((sample_tod_s, gemmini_busy[core]))
            for core in vcpu_cores:
                vcpu_per_core[core].append(parse_float(row.get(f"power_vcpu_core_{core}")))

    return power_times, gemmini_per_core, vcpu_per_core, total_power, gemmini_util_rows


def safe_add(*series):
    if not series:
        return []
    out = []
    for values in zip(*series):
        total = 0.0
        saw_value = False
        for value in values:
            if not math.isnan(value):
                total += value
                saw_value = True
        out.append(total if saw_value else float("nan"))
    return out


def build_extra_output_path(base_path, suffix):
    if "." in base_path.rsplit("/", 1)[-1]:
        stem, ext = base_path.rsplit(".", 1)
        return f"{stem}_{suffix}.{ext}"
    return f"{base_path}_{suffix}.png"


def first_event_ts(events, label):
    timestamps = [event["host_ts"] for event in events if event["label"] == label]
    return min(timestamps) if timestamps else None


def prettify_event_label(label):
    return label.replace("_", " ")


def phase_label(label):
    if label in ("process_start", "sample_start", "sample_done", "gemm_end"):
        return None
    if label == "model_encoder":
        return "model load"
    if label == "gemm_begin":
        return "gemm begin"
    if label in ("init_begin", "init_end", "embeddings", "pre_cls", "cls_relu", "cls_tanh", "cls"):
        return prettify_event_label(label)

    layer_match = LAYER_OP_PATTERN.match(label)
    if not layer_match:
        return prettify_event_label(label)

    layer_idx = layer_match.group(1)
    op = layer_match.group(2)
    known = {
        "qkv": "qkv",
        "attn": "attn",
        "out": "out",
        "sa_norm": "sa norm",
        "ffn1": "ffn1",
        "gelu": "gelu",
        "ffn2": "ffn2",
        "out_norm": "out norm",
    }
    return f"L{layer_idx} {known.get(op, op)}"


def first_inference_ts(events):
    timestamps = [
        event["host_ts"]
        for event in events
        if event["label"] in ("sample_start", "embeddings", "gemm_begin")
    ]
    return min(timestamps) if timestamps else None


def event_style(event, inference_start):
    if inference_start is not None and event["host_ts"] < inference_start:
        return TIMELINE_STYLE_PREPROCESS, TIMELINE_TEXT_PREPROCESS
    if event["actor"] == "gemmini":
        return TIMELINE_STYLE_GEMMINI, TIMELINE_TEXT_GEMMINI
    return TIMELINE_STYLE_VCPU, TIMELINE_TEXT_VCPU


def compact_events_for_labels(events):
    inference_start = first_inference_ts(events)
    compacted = []
    seen_setup_labels = set()
    seen_repeated_labels = set()
    for event in events:
        label = phase_label(event["label"])
        if label is None:
            continue
        is_setup = inference_start is not None and event["host_ts"] < inference_start
        if is_setup:
            if label in seen_setup_labels:
                continue
            seen_setup_labels.add(label)
        repeated_key = (event["actor"], label)
        if label == "gemm begin" and repeated_key in seen_repeated_labels:
            continue
        seen_repeated_labels.add(repeated_key)
        compacted.append({**event, "display_label": label})
    return compacted, inference_start


def annotate_timeline(ax, events, actor, first_ts, show_text=True):
    labeled_events, inference_start = compact_events_for_labels(events)
    if actor == "vcpu":
        actor_events = [event for event in labeled_events if event["actor"] in ("vcpu", "sample")]
    else:
        actor_events = [event for event in labeled_events if event["actor"] in (actor, "sample")]

    y_positions = [102, 95, 88, 81]
    for idx, event in enumerate(actor_events):
        x = event["host_ts"] - first_ts
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
            fontsize=6.4,
            color=text_color,
            alpha=0.9,
            clip_on=True,
        )


def resolve_time_zero(args, rows, power_times, markers, events_by_model):
    first_candidates = [power_times[0]] if power_times else []
    first_candidates.extend(samples[0][0] for samples in rows.values() if samples)
    if not first_candidates:
        raise SystemExit("no turbostat or power samples found")
    default_first_ts = min(first_candidates)
    if args.time_zero_event is None:
        return default_first_ts

    marker_ts = markers.get(args.time_zero_event)
    if marker_ts is not None:
        return marker_ts

    matches = [
        event["host_ts"]
        for events in events_by_model.values()
        for event in events
        if event["label"] == args.time_zero_event
    ]
    if not matches:
        raise SystemExit(f"No timeline event or bench marker matched: {args.time_zero_event}")
    return min(matches)


def add_bench_lines(ax, markers, first_ts):
    start = markers.get("BENCH_HOST_START")
    end = markers.get("BENCH_HOST_END")
    if start is not None:
        ax.axvline(start - first_ts, color=BENCH_COLOR, linestyle="--", linewidth=0.9)
    if end is not None:
        ax.axvline(end - first_ts, color=BENCH_COLOR, linestyle="--", linewidth=0.9)


def add_pair_boundaries(ax, events, first_ts):
    process_start = first_event_ts(events, "process_start")
    sample_start = first_event_ts(events, "sample_start")
    sample_done = first_event_ts(events, "sample_done")
    if process_start is not None:
        ax.axvline(process_start - first_ts, color=PROCESS_COLOR, linestyle=":", linewidth=0.75)
    if sample_start is not None:
        ax.axvline(sample_start - first_ts, color=SAMPLE_COLOR, linestyle="--", linewidth=0.8)
    if sample_done is not None:
        ax.axvline(sample_done - first_ts, color=SAMPLE_COLOR, linestyle="--", linewidth=0.8)


def boundary_handles():
    return [
        Line2D([0], [0], color=BENCH_COLOR, linestyle="--", linewidth=0.9, label="bench start/end"),
        Line2D([0], [0], color=PROCESS_COLOR, linestyle=":", linewidth=0.75, label="process start"),
        Line2D([0], [0], color=SAMPLE_COLOR, linestyle="--", linewidth=0.8, label="sample start/end"),
    ]


def compact_legend(ax, include_boundaries=True, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    if include_boundaries:
        for handle in boundary_handles():
            handles.append(handle)
            labels.append(handle.get_label())

    unique = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        unique.append((handle, label))
        seen.add(label)
    if unique:
        ax.legend(
            [handle for handle, _ in unique],
            [label for _, label in unique],
            fontsize=8,
            ncol=ncol,
            framealpha=0.82,
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=2.0,
        )


def presentation_util_legend(ax):
    ax.legend(
        handles=[
            Patch(facecolor="#8FD0F2", label="RISC-V Utilization"),
            Patch(facecolor="#08519C", label="AI Engine Utilization"),
        ],
        loc="upper right",
        prop={"size": 11, "weight": "bold"},
        framealpha=0.9,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=1.5,
        handleheight=0.55,
    )


def style_axis(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.xaxis.set_major_locator(MultipleLocator(TIME_MAJOR_TICK_S))


def choose_time_tick(left, right):
    if left is None or right is None:
        return None
    span = right - left
    if span <= 4.0:
        return 0.5
    if span <= 10.0:
        return 1.0
    if span <= 30.0:
        return 5.0
    return None


def sample_window_bounds(args, events_by_model, first_ts):
    if args.sample_window_margin is None:
        return None
    if args.sample_window_margin < 0.0:
        raise SystemExit("--sample-window-margin must be non-negative")

    sample_starts = [
        event["host_ts"]
        for events in events_by_model.values()
        for event in events
        if event["label"] == "sample_start"
    ]
    sample_dones = [
        event["host_ts"]
        for events in events_by_model.values()
        for event in events
        if event["label"] == "sample_done"
    ]
    if not sample_starts:
        raise SystemExit("--sample-window-margin requires at least one sample_start event")

    window_start = min(sample_starts) - args.sample_window_margin
    window_end = (max(sample_dones) if sample_dones else max(sample_starts)) + args.sample_window_margin
    return window_start - first_ts, window_end - first_ts


def maybe_crop_axes(axes, args):
    resolved_window = getattr(args, "resolved_time_window", None)
    if resolved_window is not None:
        left, right = resolved_window
        for ax in axes:
            ax.set_xlim(left=left, right=right)
        tick_s = args.time_major_tick_s or choose_time_tick(left, right)
        if tick_s is not None:
            if tick_s <= 0.0:
                raise SystemExit("--time-major-tick-s must be > 0")
            for ax in axes:
                ax.xaxis.set_major_locator(MultipleLocator(tick_s))
        return

    window_before = args.time_window_before
    window_after = args.time_window_after
    if window_before is not None or window_after is not None:
        if window_before is not None and window_before < 0.0:
            raise SystemExit("--time-window-before must be non-negative")
        if window_after is not None and window_after < 0.0:
            raise SystemExit("--time-window-after must be non-negative")
        left = -window_before if window_before is not None else None
        right = window_after if window_after is not None else None
        for ax in axes:
            ax.set_xlim(left=left, right=right)
        tick_s = args.time_major_tick_s or choose_time_tick(left, right)
        if tick_s is not None:
            if tick_s <= 0.0:
                raise SystemExit("--time-major-tick-s must be > 0")
            for ax in axes:
                ax.xaxis.set_major_locator(MultipleLocator(tick_s))
        return

    if args.crop_before_time_zero:
        for ax in axes:
            ax.set_xlim(left=0)

    if args.time_major_tick_s is not None:
        if args.time_major_tick_s <= 0.0:
            raise SystemExit("--time-major-tick-s must be > 0")
        for ax in axes:
            ax.xaxis.set_major_locator(MultipleLocator(args.time_major_tick_s))


def relative_busy_xy(rows, core, first_ts):
    samples = rows.get(core, [])
    return [ts - first_ts for ts, _ in samples], [busy for _, busy in samples]


def relative_busy_sum_xy(rows, cores, first_ts):
    summed = {}
    for core in cores:
        for ts, busy in rows.get(core, []):
            summed[ts] = summed.get(ts, 0.0) + busy
    timestamps = sorted(summed)
    return [ts - first_ts for ts in timestamps], [summed[ts] for ts in timestamps]


def relative_xs(times, first_ts):
    return [ts - first_ts for ts in times]


def plot_util(ax, rows, core, first_ts, color, label, linestyle="-"):
    xs, ys = relative_busy_xy(rows, core, first_ts)
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=UTIL_LINEWIDTH,
        marker="o",
        markersize=MARKERSIZE,
        label=label,
    )


def plot_power(ax, xs, series, color, label, linewidth=POWER_LINEWIDTH, alpha=POWER_ALPHA):
    ax.plot(xs, series, color=color, linewidth=linewidth, alpha=alpha, label=label)


def model_power_series(model, gemmini_power_per_core, vcpu_power_per_core):
    main_power = vcpu_power_per_core[model["main_core"]]
    gemmini_power = gemmini_power_per_core[model["gemmini_core"]]
    total_power = safe_add(main_power, gemmini_power)
    return {
        "main": main_power,
        "gemmini": gemmini_power,
        "vcpu": main_power,
        "total": total_power,
    }


def save_model_plot(out_path, model, gemmini_rows, guest_rows, power_xs, power_series, markers, events, first_ts, args):
    style = APP_STYLES.get(model["key"], FALLBACK_STYLE)
    figsize = (args.fig_width or MODEL_FIGSIZE[0], MODEL_FIGSIZE[1])
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    plot_util(axes[0], guest_rows, model["main_guest_vcpu"], first_ts, style["main_color"], f"guest main vCPU {model['main_guest_vcpu']}")
    axes[0].set_ylim(0, 105)
    style_axis(axes[0], "Guest vCPU Busy%")
    compact_legend(axes[0], ncol=2)

    plot_util(axes[1], gemmini_rows, model["gemmini_core"], first_ts, style["gemmini_color"], f"Gemmini core {model['gemmini_core']}")
    axes[1].set_ylim(0, 105)
    style_axis(axes[1], "Gemmini Busy%")
    compact_legend(axes[1], ncol=3)

    plot_power(axes[2], power_xs, power_series["main"], style["main_color"], "main vCPU power")
    plot_power(axes[2], power_xs, power_series["gemmini"], style["gemmini_color"], "Gemmini power")
    plot_power(
        axes[2],
        power_xs,
        power_series["total"],
        style["total_color"],
        "app total power",
        1.65,
        PAIR_TOTAL_POWER_ALPHA,
    )
    style_axis(axes[2], "Power (W)")
    axes[2].set_xlabel("Time (s)" if args.presentation_util else "Host realtime offset (s)")
    compact_legend(axes[2], ncol=3)

    for ax in axes:
        add_bench_lines(ax, markers, first_ts)
        add_pair_boundaries(ax, events, first_ts)
    annotate_timeline(axes[0], events, "vcpu", first_ts, show_text=True)
    annotate_timeline(axes[1], events, "gemmini", first_ts, show_text=True)
    maybe_crop_axes(axes, args)

    fig.suptitle(f"{model['name']} utilization and power")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_combined_plot(out_path, models, gemmini_rows, guest_rows, power_xs, power_series_by_model, total_power, markers, events_by_model, first_ts, args):
    figsize = (
        args.fig_width or COMBINED_FIGSIZE[0], 8.1 if args.presentation_util else COMBINED_FIGSIZE[1]
    )
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for ax, model in zip(axes[:2], models):
        style = APP_STYLES.get(model["key"], FALLBACK_STYLE)
        riscv_color = "#8FD0F2" if args.presentation_util else style["main_color"]
        ai_color = "#08519C" if args.presentation_util else style["gemmini_color"]
        riscv_label = "RISC-V Utilization" if args.presentation_util else f'guest main vCPU {model["main_guest_vcpu"]}'
        ai_label = "AI Engine Utilization" if args.presentation_util else f'Gemmini {model["gemmini_core"]}'
        plot_util(
            ax,
            guest_rows,
            model["main_guest_vcpu"],
            first_ts,
            riscv_color,
            riscv_label,
        )
        plot_util(
            ax,
            gemmini_rows,
            model["gemmini_core"],
            first_ts,
            ai_color,
            ai_label,
        )
        ax.set_ylim(0, 105)
        if args.presentation_util:
            ax.set_title("")
            ax.text(0.02, 0.94, model["name"], transform=ax.transAxes, ha="left", va="top", fontsize=13, color="#707070", fontweight="bold", bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#A0A0A0", "alpha": 0.9})
        else:
            ax.set_title(f'Application {model["slot"]}: {model["name"]} pair utilization', fontsize=11)
        style_axis(ax, "Utilization (%)" if args.presentation_util else "Busy%")
        if args.presentation_util:
            presentation_util_legend(ax)
        else:
            compact_legend(ax, ncol=3)

    if args.presentation_util:
        app_power_colors = ("#E88C8C", "#F4B6B6")
        for model, color in zip(models, app_power_colors):
            axes[2].plot(
                power_xs,
                power_series_by_model[model["key"]]["total"],
                label=f'{model["name"]} Power',
                color=color,
                linestyle="-",
                linewidth=UTIL_LINEWIDTH,
                alpha=0.98,
            )
        axes[2].plot(power_xs, total_power, color="#D62728", linewidth=UTIL_LINEWIDTH, label="Total Power", zorder=5)
        axes[2].set_title("")
    else:
        for model in models:
            style = APP_STYLES.get(model["key"], FALLBACK_STYLE)
            plot_power(axes[2], power_xs, power_series_by_model[model["key"]]["total"], style["total_color"], f'{model["name"]} total', 1.45, PAIR_TOTAL_POWER_ALPHA)
        plot_power(axes[2], power_xs, total_power, OVERALL_TOTAL_COLOR, "dual total", 1.9, 0.98)
    style_axis(axes[2], "Power (W)")
    axes[2].set_xlabel("Time (s)" if args.presentation_util else "Host realtime offset (s)")
    if args.presentation_util:
        axes[2].legend(loc="upper right", prop={"size": 10, "weight": "bold"}, framealpha=0.9)
    else:
        compact_legend(axes[2], ncol=3)

    for ax, model in zip(axes[:2], models):
        add_bench_lines(ax, markers, first_ts)
        add_pair_boundaries(ax, events_by_model.get(model["key"], []), first_ts)
    add_bench_lines(axes[2], markers, first_ts)
    for model in models:
        add_pair_boundaries(axes[2], events_by_model.get(model["key"], []), first_ts)
    maybe_crop_axes(axes, args)

    presentation_title = f'({models[0]["name"]}, {models[1]["name"]})'
    fig.suptitle(presentation_title if args.presentation_util else args.title, fontsize=18 if args.presentation_util else None, fontweight="bold" if args.presentation_util else None)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95 if args.presentation_util else 0.97])
    fig.savefig(out_path, dpi=150 if args.presentation_util else 160)
    plt.close(fig)

def save_util_plot(
    out_path,
    series_specs,
    markers,
    events_by_model,
    first_ts,
    args,
    title,
    ylabel,
):
    figsize = (args.fig_width or MODEL_FIGSIZE[0], 4.6)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for spec in series_specs:
        plot_util(
            ax,
            spec["rows"],
            spec["core"],
            first_ts,
            spec["color"],
            spec["label"],
            linestyle=spec.get("linestyle", "-"),
        )
    ax.set_ylim(0, 105)
    style_axis(ax, ylabel)
    ax.set_xlabel("Host realtime offset (s)")
    add_bench_lines(ax, markers, first_ts)
    for events in events_by_model.values():
        add_pair_boundaries(ax, events, first_ts)
    compact_legend(ax, ncol=2)
    maybe_crop_axes([ax], args)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    args = parse_args()
    if args.fig_width is not None and args.fig_width <= 0.0:
        raise SystemExit("--fig-width must be > 0")
    models = model_args(args)
    allowed_models = {model["key"] for model in models}
    gemmini_cores = [model["gemmini_core"] for model in models]
    main_vcpu_cores = [model["main_core"] for model in models]
    executor_cores = [model["executor_core"] for model in models]
    target_cores = set(
        gemmini_cores + main_vcpu_cores + executor_cores + [args.housekeeping_vcpu_core]
    )
    target_guest_vcpus = {args.guest_housekeeping_vcpu}
    for model in models:
        target_guest_vcpus.add(model["main_guest_vcpu"])
        target_guest_vcpus.add(model["executor_guest_vcpu"])

    markers = parse_bench_markers(args.bench)
    events_by_model = parse_events(args.bench, allowed_models)
    host_rows = parse_turbostat(args.turbostat, target_cores)
    guest_rows = parse_guest_util(args.guest_util_csv, target_guest_vcpus)
    power_times, gemmini_power_per_core, vcpu_power_per_core, total_power, gemmini_rows = parse_power_csv(
        args.power_csv, gemmini_cores, main_vcpu_cores
    )
    if not power_times:
        raise SystemExit("No power samples were parsed")
    first_ts = resolve_time_zero(args, host_rows, power_times, markers, events_by_model)
    args.resolved_time_window = sample_window_bounds(args, events_by_model, first_ts)
    power_xs = relative_xs(power_times, first_ts)

    power_series_by_model = {}
    for model in models:
        power_series_by_model[model["key"]] = model_power_series(
            model, gemmini_power_per_core, vcpu_power_per_core
        )

    save_combined_plot(
        args.out,
        models,
        gemmini_rows,
        guest_rows,
        power_xs,
        power_series_by_model,
        total_power,
        markers,
        events_by_model,
        first_ts,
        args,
    )
    for model in models:
        save_model_plot(
            build_extra_output_path(args.out, model["key"]),
            model,
            gemmini_rows,
            guest_rows,
            power_xs,
            power_series_by_model[model["key"]],
            markers,
            events_by_model.get(model["key"], []),
            first_ts,
            args,
        )
    line_styles = ("-", "--")

    def model_specs(rows, core_key, color_key, label_prefix):
        specs = []
        for index, model in enumerate(models):
            style = APP_STYLES.get(model["key"], FALLBACK_STYLE)
            specs.append(
                {
                    "rows": rows,
                    "core": model[core_key],
                    "color": style[color_key],
                    "label": f'{model["name"]} {label_prefix} {model[core_key]}',
                    "linestyle": line_styles[index],
                }
            )
        return specs

    diagnostic_plots = [
        (
            "guest_vcpu_util",
            model_specs(guest_rows, "main_guest_vcpu", "main_color", "guest main vCPU"),
            "Guest main vCPU utilization",
        ),
        (
            "host_vcpu_util",
            model_specs(host_rows, "main_core", "main_color", "host backing core"),
            "Host backing-core utilization for main vCPUs",
        ),
        (
            "guest_executor_vcpu_util",
            model_specs(guest_rows, "executor_guest_vcpu", "executor_color", "guest executor vCPU"),
            "Guest executor vCPU utilization",
        ),
        (
            "host_executor_vcpu_util",
            model_specs(host_rows, "executor_core", "executor_color", "host executor backing core"),
            "Host backing-core utilization for executor vCPUs",
        ),
        (
            "guest_housekeeping_util",
            [
                {
                    "rows": guest_rows,
                    "core": args.guest_housekeeping_vcpu,
                    "color": HOUSEKEEPING_COLOR,
                    "label": f"guest housekeeping vCPU {args.guest_housekeeping_vcpu}",
                }
            ],
            "Guest housekeeping vCPU utilization",
        ),
        (
            "host_housekeeping_util",
            [
                {
                    "rows": host_rows,
                    "core": args.housekeeping_vcpu_core,
                    "color": HOUSEKEEPING_COLOR,
                    "label": f"host housekeeping backing core {args.housekeeping_vcpu_core}",
                }
            ],
            "Host backing-core utilization for housekeeping vCPU",
        ),
    ]
    for suffix, specs, title in diagnostic_plots:
        save_util_plot(
            build_extra_output_path(args.out, suffix),
            specs,
            markers,
            events_by_model,
            first_ts,
            args,
            title,
            "Busy%",
        )


if __name__ == "__main__":
    main()
