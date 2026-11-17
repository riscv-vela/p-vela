#!/usr/bin/env python3

"""Build transient plot input from separately modeled vCPU/Gemmini power."""

import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align guest vCPU and host Gemmini power by epoch_ns using "
            "zero-order hold on the vCPU trace."
        )
    )
    parser.add_argument("--vcpu-power", required=True)
    parser.add_argument("--gemmini-power", required=True)
    parser.add_argument("--out-power", required=True)
    parser.add_argument("--out-util", required=True)
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"power CSV has no samples: {path}")
    try:
        rows.sort(key=lambda row: int(row["epoch_ns"]))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"power CSV is missing a valid epoch_ns: {path}") from exc
    return rows


def core_ids(row, prefix):
    cores = []
    for key in row:
        if not key.startswith(prefix):
            continue
        try:
            cores.append(int(key[len(prefix):]))
        except ValueError:
            continue
    return sorted(cores)


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def combine_rows(gemmini_rows, vcpu_rows, gemmini_cores, vcpu_cores):
    combined = []
    vcpu_index = 0
    latest_vcpu = None

    for gemmini_row in gemmini_rows:
        gemmini_epoch_ns = int(gemmini_row["epoch_ns"])
        while (
            vcpu_index < len(vcpu_rows)
            and int(vcpu_rows[vcpu_index]["epoch_ns"]) <= gemmini_epoch_ns
        ):
            latest_vcpu = vcpu_rows[vcpu_index]
            vcpu_index += 1
        if latest_vcpu is None:
            continue

        row = {
            "epoch_ns": gemmini_row["epoch_ns"],
            "sample_tod_s": gemmini_row["sample_tod_s"],
            "model_freq_ghz": latest_vcpu.get("model_freq_ghz", "0"),
            "model_sys_scale": latest_vcpu.get("model_sys_scale", "0"),
            "model_gemmini_full_util_power_w": gemmini_row.get(
                "model_gemmini_full_util_power_w", "0"
            ),
        }
        for core in gemmini_cores:
            raw_key = f"raw_busy_core_{core}"
            if raw_key in gemmini_row:
                row[raw_key] = gemmini_row[raw_key]
            row[f"busy_core_{core}"] = gemmini_row[f"busy_core_{core}"]
            row[f"util_core_{core}"] = gemmini_row[f"util_core_{core}"]
            row[f"power_gemmini_core_{core}"] = gemmini_row[
                f"power_gemmini_core_{core}"
            ]
        for core in vcpu_cores:
            row[f"busy_core_{core}"] = latest_vcpu[f"busy_core_{core}"]
            row[f"util_core_{core}"] = latest_vcpu[f"util_core_{core}"]
            row[f"power_vcpu_core_{core}"] = latest_vcpu[
                f"power_vcpu_core_{core}"
            ]

        gemmini_total = float(gemmini_row["p_gemmini_total_w"])
        vcpu_total = float(latest_vcpu["p_vcpu_total_w"])
        row["p_gemmini_total_w"] = f"{gemmini_total:.6f}"
        row["p_vcpu_total_w"] = f"{vcpu_total:.6f}"
        row["p_total_w"] = f"{gemmini_total + vcpu_total:.6f}"
        combined.append(row)

    if not combined:
        raise SystemExit("no epoch-overlapping vCPU and Gemmini power samples")
    return combined


def write_power(path, rows, gemmini_cores, vcpu_cores):
    ensure_parent(path)
    fieldnames = [
        "epoch_ns",
        "sample_tod_s",
        "model_freq_ghz",
        "model_sys_scale",
        "model_gemmini_full_util_power_w",
    ]
    fieldnames += [f"raw_busy_core_{core}" for core in gemmini_cores
                   if f"raw_busy_core_{core}" in rows[0]]
    fieldnames += [f"busy_core_{core}" for core in gemmini_cores + vcpu_cores]
    fieldnames += [f"util_core_{core}" for core in gemmini_cores + vcpu_cores]
    fieldnames += [f"power_gemmini_core_{core}" for core in gemmini_cores]
    fieldnames += [f"power_vcpu_core_{core}" for core in vcpu_cores]
    fieldnames += ["p_gemmini_total_w", "p_vcpu_total_w", "p_total_w"]

    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_util(path, rows, gemmini_cores, vcpu_cores):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("# plot utilization input derived from split power CSVs\n")
        for core in gemmini_cores:
            stream.write(f"# core={core} source=host_turbostat\n")
        for core in vcpu_cores:
            stream.write(f"# core={core} source=guest_proc_stat_100ms\n")

        for row in rows:
            stream.write("Time_Of_Day_Seconds CPU Busy%\n")
            for core in gemmini_cores + vcpu_cores:
                stream.write(
                    f"{row['sample_tod_s']} {core} {row[f'busy_core_{core}']}\n"
                )
            stream.write("\n")


def main():
    args = parse_args()
    vcpu_rows = read_rows(args.vcpu_power)
    gemmini_rows = read_rows(args.gemmini_power)
    vcpu_cores = core_ids(vcpu_rows[0], "power_vcpu_core_")
    gemmini_cores = core_ids(gemmini_rows[0], "power_gemmini_core_")
    if not vcpu_cores or not gemmini_cores:
        raise SystemExit("split power CSV is missing per-core power columns")

    combined = combine_rows(
        gemmini_rows,
        vcpu_rows,
        gemmini_cores,
        vcpu_cores,
    )
    write_power(args.out_power, combined, gemmini_cores, vcpu_cores)
    write_util(args.out_util, combined, gemmini_cores, vcpu_cores)
    print(f"[merge_split_power_csv] rows      : {len(combined)}")
    print(f"[merge_split_power_csv] plot power: {args.out_power}")
    print(f"[merge_split_power_csv] plot util : {args.out_util}")


if __name__ == "__main__":
    main()
