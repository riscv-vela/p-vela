#!/usr/bin/env python3
"""Postprocess a time-scaled power CSV with a hard Gemmini-util threshold."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

GEMMINI_IDLE_POWER_W = 0.014


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-csv", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[70, 80, 90])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="time_scaled_0p061_step")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.power_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("power CSV has no data rows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_ts = float(rows[0]["sample_tod_s"])
    times = [float(row["sample_tod_s"]) - first_ts for row in rows]

    for threshold_pct in args.thresholds:
        threshold = threshold_pct / 100.0
        gemmini = []
        vcpu = []
        total = []
        adjusted_rows = []

        for row in rows:
            util = float(row["util_core_2"])
            full_power = float(row["model_gemmini_full_util_power_w"])
            # Binary activity model: utilization only selects OFF or full power.
            # Do not scale Gemmini power proportionally below the threshold.
            gemmini_power = full_power if util >= threshold else GEMMINI_IDLE_POWER_W
            vcpu_power = float(row["p_vcpu_total_w"])

            updated = dict(row)
            updated["util_core_2"] = f"{(1.0 if util >= threshold else 0.0):.6f}"
            updated["busy_core_2"] = f"{(100.0 if util >= threshold else 0.0):.6f}"
            updated["power_gemmini_core_2"] = f"{gemmini_power:.6f}"
            updated["p_gemmini_total_w"] = f"{gemmini_power:.6f}"
            updated["p_total_w"] = f"{gemmini_power + vcpu_power:.6f}"
            adjusted_rows.append(updated)
            gemmini.append(gemmini_power)
            vcpu.append(vcpu_power)
            total.append(gemmini_power + vcpu_power)

        tag = f"{threshold_pct:g}".replace(".", "p")
        csv_path = out_dir / f"{args.prefix}_threshold_{tag}pct_power.csv"
        png_path = out_dir / f"{args.prefix}_threshold_{tag}pct_total_power.png"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=adjusted_rows[0].keys())
            writer.writeheader()
            writer.writerows(adjusted_rows)

        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.plot(times, total, color="#d62728", linewidth=1.6, label="Total power")
        ax.plot(times, vcpu, color="#1f77b4", linewidth=1.0, alpha=0.85, label="vCPU power")
        ax.plot(times, gemmini, color="#2ca02c", linewidth=1.0, alpha=0.9, label="Gemmini power")
        ax.set_title(
            f"DistilBERT time-scaled power (Gemmini util >= {threshold_pct:g}% -> 100%)"
        )
        ax.set_xlabel("Time-scaled elapsed time (s)")
        ax.set_ylabel("Modeled power (W)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(png_path)


if __name__ == "__main__":
    main()
