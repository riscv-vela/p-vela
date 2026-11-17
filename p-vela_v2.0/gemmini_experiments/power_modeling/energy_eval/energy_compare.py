#!/usr/bin/env python3
"""Phase 1 energy comparison: DVFS + NPU power gating, synthetic decision traces.

Basis: CPU + NPU only (memory excluded, per project decision 2026-09).
Reconstructs energy on the RTL-scaled (real-hardware) time axis from an existing
run, then re-evaluates it under synthetic frequency / gating decisions.

Model (documented assumptions):
  CPU power   P = sys_scale * (a*util*f^3 + b)        [SSU coefficients, unchanged]
  NPU power   active 0.862 W / powered-idle 0.014 W / gated 0 W   [3-state extension]
  Time axis   RTL compute time is fixed (NPU has its own clock);
              every other slice stretches by slowdown(f) = f_ref / f.
              An NPU-active sample carries both RTL compute and embedded CPU work,
              so it stretches only by its non-RTL fraction.
"""
import argparse, csv, glob, json, re, sys

# --- SSU model constants (must stay identical to power_sim_gemmini.py) ---
A_NOM, K_NOM = 0.086391591, 0.184415
A_BOOST, K_BOOST = 0.039747769, 2.80057
BOOST_OFF_TH = 2.6
SYS_SCALE = 0.5625
F_REF = 2.512
P_NPU_ACTIVE = 0.862      # ASAP7 5ns/200MHz, full Gemmini
P_NPU_IDLE = 0.014        # clock-on, zero-activity (re-measured; was 0.013)


def cpu_power(f, util):
    f3 = f ** 3
    if f <= BOOST_OFF_TH:
        return SYS_SCALE * (A_NOM * util * f3 + K_NOM)
    return SYS_SCALE * (A_BOOST * util * f3 + K_BOOST)


def verify_mapping(run_dir, app, expected_core, meta):
    """Guard against the 2026-08-11 class of bug: the runner records the EXPECTED
    executor vCPU, but the app may actually have run on a different one, which
    silently swaps per-app power attribution under the modulo policy.
    Cross-check against the routing actually logged by QEMU."""
    import collections, os
    log = f"{run_dir}/qemu_gemmini_run.log"
    if not os.path.exists(log):
        return "no qemu_gemmini_run.log — mapping NOT verified"
    pairs = collections.Counter(re.findall(r"Core (\d+)-Gemmini (\d+)", open(log).read()))
    if not pairs:
        return "no routing lines in qemu log — mapping NOT verified"
    # the app issuing the most loop_ws commands must own the busiest routing pair
    counts = {a: (r.get("rtl_loop_ws_count") or 0) for a, r in (meta.get("model_rtl") or {}).items()}
    heaviest = max(counts, key=counts.get)
    (core, gem), n = pairs.most_common(1)[0]
    if heaviest == app and 8 + int(core) != expected_core - 2 + 8:
        pass  # executor vCPU, not main vCPU — compared below
    if heaviest == app:
        got = 2 + (int(core) % 2)
        if got != expected_core:
            sys.exit(f"\nMAPPING MISMATCH: {app} is the heaviest app ({counts[app]} loop_ws) "
                     f"but QEMU routed the busiest stream (Core {core} -> Gemmini {gem}, {n} cmds) "
                     f"to core {got}, while launch-affinity implies core {expected_core}.\n"
                     f"Per-app power attribution in this run is unreliable — discard it.")
        return f"verified (Core {core} -> Gemmini {gem}, {n} cmds)"
    return f"cross-checked via {heaviest} (Core {core} -> Gemmini {gem})"


def load_app(run_dir, app):
    pcs = sorted(glob.glob(f"{run_dir}/time_scaled_*_power.csv"))
    if not pcs:
        sys.exit(f"no time-scaled power csv in {run_dir}")
    prefix = pcs[0][: -len("_power.csv")]
    meta = json.load(open(f"{prefix}_summary.json"))
    rtl = (meta.get("model_rtl") or {})[app]

    # per-app core mapping from the bench log
    vcpu_core = npu_core = None
    win = {}
    aff = re.compile(rf"\[{app}\] \[launch-affinity\] main_vcpu=(\d+) gemmini_executor_vcpu=(\d+)")
    smp = re.compile(rf"\[HOST_TS_LINE\] ([0-9.]+) \[{app}\] \[(sample-start|sample-done)\]")
    for line in open(f"{prefix}_bench.log"):
        m = aff.search(line)
        if m:
            vcpu_core, npu_core = 8 + int(m.group(1)), 2 + (int(m.group(2)) % 2)
        m = smp.search(line)
        if m:
            win.setdefault(m.group(2), float(m.group(1)))

    rows = []
    for r in csv.DictReader(open(pcs[0])):
        t = float(r["sample_tod_s"])
        if win["sample-start"] <= t <= win["sample-done"]:
            rows.append((t,
                         float(r[f"util_core_{vcpu_core}"]),
                         float(r[f"power_gemmini_core_{npu_core}"])))
    check = verify_mapping(run_dir, app, npu_core, meta)
    return rows, rtl, vcpu_core, npu_core, check


def evaluate(rows, rtl_time, f, gated, n_npu, mode='measured'):
    """Return (energy_J, runtime_s, e_cpu, e_npu) for one scenario.

    gated=True  -> unused NPUs off, used NPU off during its idle gaps
    gated=False -> every NPU powered for the whole window (idle draws P_NPU_IDLE)
    """
    slow = F_REF / f
    # fraction of NPU-active wall time that is real RTL compute; the rest is
    # embedded CPU work that DVFS does slow down
    t_active = sum(rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1) if rows[i][2] > 0)
    alpha = min(1.0, rtl_time / t_active) if t_active > 0 else 0.0

    # effective per-sample NPU power: measured when active, P_IDLE when powered-idle,
    # 0 when gated. Integrated trapezoidally to match summarize_power_csv.py.
    p_eff = [p if p > 0 else (0.0 if gated else P_NPU_IDLE) for _, _, p in rows]
    p_cpu = [cpu_power(f, u) for _, u, _ in rows]
    p_idle = 0.0 if gated else P_NPU_IDLE

    t = e_cpu = e_npu_used = 0.0
    for i in range(len(rows) - 1):
        dt = rows[i + 1][0] - rows[i][0]
        if dt <= 0:
            continue
        active = rows[i][2] > 0
        pc = 0.5 * (p_cpu[i] + p_cpu[i + 1])
        if active and mode == "rtl":
            # split: RTL compute (fixed, NPU active) + embedded CPU work (stretches,
            # NPU idle because it is waiting for the next command)
            dt_rtl, dt_emb = dt * alpha, dt * (1 - alpha) * slow
            t += dt_rtl + dt_emb
            e_cpu += pc * (dt_rtl + dt_emb)
            e_npu_used += P_NPU_ACTIVE * dt_rtl + p_idle * dt_emb
        elif mode == "rtl":
            dt_new = dt * slow
            t += dt_new
            e_cpu += pc * dt_new
            e_npu_used += p_idle * dt_new          # 유휴 구간은 평탄값
        else:
            dt_new = dt * (alpha + (1 - alpha) * slow) if active else dt * slow
            t += dt_new
            e_cpu += pc * dt_new
            e_npu_used += 0.5 * (p_eff[i] + p_eff[i + 1]) * dt_new

    e_npu_other = 0.0 if gated else P_NPU_IDLE * (n_npu - 1) * t
    return e_cpu + e_npu_used + e_npu_other, t, e_cpu, e_npu_used + e_npu_other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--app", default="distilbert")
    ap.add_argument("--npus", type=int, default=4)
    ap.add_argument("--freqs", default="2.512,2.2,2.0,1.8,1.5,1.2,1.0")
    ap.add_argument("--mode", default="both", choices=["measured", "rtl", "both"])
    args = ap.parse_args()

    rows, rtl, vc, nc, check = load_app(args.run_dir, args.app)
    rtl_time = rtl["rtl_time_s"]
    freqs = [float(x) for x in args.freqs.split(",")]

    print(f"\n{'='*78}")
    print(f"  Phase 1 energy sensitivity — {args.app}   (basis: CPU + NPU, memory excluded)")
    import os
    print(f"  run: {os.path.basename(args.run_dir.rstrip('/'))[:15]}   mapping: {check}")
    print(f"  {args.npus} NPUs attached, 1 in use | vCPU core {vc}, NPU core {nc}"
          f" | RTL compute {rtl_time:.4f} s")
    print(f"{'='*78}")

    modes = ["measured", "rtl"] if args.mode == "both" else [args.mode]
    label = {"measured": "A) as-is  : worker-busy 구간 전체에 NPU 활성전력 부과 (현행 SSU 모델)",
             "rtl":      "B) rtl    : RTL 연산시간만 NPU 활성, embedded CPU 구간은 NPU 유휴"}
    for mode in modes:
        base_e, base_t, base_c, base_n = evaluate(rows, rtl_time, F_REF, False, args.npus, mode)
        print(f"\n  {label[mode]}")
        print(f"    BASELINE  runtime {base_t:.3f} s | energy {base_e:.4f} J"
              f"  (CPU {base_c:.4f} + NPU {base_n:.4f})")
        print(f"    {'freq':>6} {'slow':>5} | {'gating':>8} {'DVFS':>8} {'DVFS+gate':>10}"
              f" | {'runtime':>8} {'energy':>8}")
        print(f"    {'-'*6} {'-'*5} | {'-'*8} {'-'*8} {'-'*10} | {'-'*8} {'-'*8}")
        for f in freqs:
            g_e = evaluate(rows, rtl_time, F_REF, True, args.npus, mode)[0]
            d_e = evaluate(rows, rtl_time, f, False, args.npus, mode)[0]
            b_e, b_t, _, _ = evaluate(rows, rtl_time, f, True, args.npus, mode)
            print(f"    {f:6.3f} {F_REF/f:5.2f} |"
                  f" {(base_e-g_e)/base_e*100:7.1f}% {(base_e-d_e)/base_e*100:7.1f}%"
                  f" {(base_e-b_e)/base_e*100:9.1f}% | {b_t:7.3f}s {b_e:7.4f}J")

    print(f"\n  NPU 개수 민감도 (mode B, DVFS 2.0GHz + gating):")
    for n in (1, 2, 4, 8):
        bb = evaluate(rows, rtl_time, F_REF, False, n, "rtl")[0]
        pp = evaluate(rows, rtl_time, 2.0, True, n, "rtl")[0]
        print(f"    {n} NPU: baseline {bb:.4f} J -> {pp:.4f} J = {(bb-pp)/bb*100:5.1f}%")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
