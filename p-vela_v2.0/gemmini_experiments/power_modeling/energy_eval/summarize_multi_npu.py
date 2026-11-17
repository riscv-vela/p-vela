#!/usr/bin/env python3
"""Compare "1 NPU busy + N-1 idle" against "N NPUs busy" at equal total work.

Reads the serial/parallel power CSVs produced by run_multi_npu_power.sh,
integrates energy over each benchmark window (trapezoid — the NPU signal is a
binary step function, so left-rectangle overestimates by ~7%), and reports what
idle-NPU power gating would save in each case.

The point of the comparison: gating and parallelisation both consume the same
resource — idle NPU time. Parallelising removes the idle time that gating was
going to harvest, so the two are alternatives, not addends.
"""
import argparse, csv, glob, os, re, sys

P_NPU_IDLE = 0.014          # powered but not computing (re-measured 26-11-09; was 0.013)


def window(bench_log):
    """Benchmark start/end as host epoch seconds."""
    lo = hi = None
    for line in open(bench_log, encoding="utf-8", errors="replace"):
        m = re.match(r"\[HOST_TS\] (START|END) ([0-9.]+)", line.strip())
        if m:
            if m.group(1) == "START":
                lo = float(m.group(2))
            else:
                hi = float(m.group(2))
    if lo is None or hi is None:
        sys.exit(f"no [HOST_TS] markers in {bench_log}")
    return lo, hi


def integrate(csv_path, lo, hi):
    """Trapezoid-integrate CPU and NPU power over [lo, hi]."""
    ts, pn, pc = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["sample_tod_s"])
            if lo <= t <= hi:
                ts.append(t)
                pn.append(float(row["p_gemmini_total_w"]))
                pc.append(float(row["p_vcpu_total_w"]))
    if len(ts) < 2:
        sys.exit(f"{os.path.basename(csv_path)}: only {len(ts)} samples in window")
    e_npu = e_cpu = 0.0
    npu_active_s = 0.0
    n_active = sum(1 for v in pn if v > 0)
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        e_npu += (pn[i] + pn[i - 1]) / 2 * dt
        e_cpu += (pc[i] + pc[i - 1]) / 2 * dt
        # device-seconds of NPU activity, for the gating estimate
        npu_active_s += (pn[i] + pn[i - 1]) / 2 * dt
    return e_cpu, e_npu, ts[-1] - ts[0], len(ts), npu_active_s, n_active


def route(path, n_npu):
    """Per-device command counts, to show how evenly work was distributed."""
    if not os.path.exists(path):
        return {}
    c = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"Gemmini (\d+)\]", line)
        if m:
            c[int(m.group(1))] = c.get(int(m.group(1)), 0) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--npus", type=int, default=4, help="devices present in the system")
    ap.add_argument("--threads", type=int, default=4, help="threads in the parallel case")
    a = ap.parse_args()

    print("=" * 78)
    print(f"  다중 NPU 병렬 분배 vs 유휴 게이팅  ({a.npus}-NPU 시스템)")
    print("=" * 78)

    res = {}
    for tag, label in (("serial", f"순차 (1 NPU 사용)"), ("parallel", f"병렬 ({a.threads} NPU 사용)")):
        bl = os.path.join(a.run_dir, f"{tag}_bench.log")
        cs = os.path.join(a.run_dir, f"{tag}_power.csv")
        if not (os.path.exists(bl) and os.path.exists(cs)):
            sys.exit(f"missing {tag} artifacts in {a.run_dir}")
        lo, hi = window(bl)
        e_cpu, e_npu, dur, n, act, n_act = integrate(cs, lo, hi)

        # 게이팅 절감: 전력모델은 유휴 NPU를 이미 0 W로 기록하므로 CSV에서 직접
        # 읽을 수 없다.  장치-초 기준으로 "전원은 켜져 있으나 연산하지 않은" 시간을
        # 되살려 P_idle을 부여한다 (energy_model.py와 동일한 가정).
        busy_dev_s = act / 0.862 if act else 0.0
        idle_dev_s = max(a.npus * dur - busy_dev_s, 0.0)
        e_idle = P_NPU_IDLE * idle_dev_s

        base = e_cpu + e_npu + e_idle          # 게이팅 없음 (유휴 누설 포함)
        gated = e_cpu + e_npu                  # 게이팅 적용
        res[tag] = dict(label=label, dur=dur, n=n, n_act=n_act, cpu=e_cpu, npu=e_npu,
                        idle=e_idle, base=base, gated=gated,
                        busy=busy_dev_s, idle_s=idle_dev_s)

        print(f"\n  {label}   [{n} samples · NPU 활성 {n_act}개 · {dur:.3f} s]")
        print(f"    CPU                 {e_cpu:8.4f} J")
        print(f"    NPU 활성 연산       {e_npu:8.4f} J   (장치-초 {busy_dev_s:.3f})")
        print(f"    NPU 유휴 누설       {e_idle:8.4f} J   (장치-초 {idle_dev_s:.3f})")
        print(f"    합계 (게이팅 없음)  {base:8.4f} J")
        print(f"    합계 (게이팅 적용)  {gated:8.4f} J    절감 {(base-gated)/base*100:5.1f}%")
        r = route(os.path.join(a.run_dir, f"{tag}_route.txt"), a.npus)
        if r:
            tot = sum(r.values())
            print("    부하 분포          " +
                  "  ".join(f"g{k}={v}({v/tot*100:.0f}%)" for k, v in sorted(r.items())))

    MIN_ACTIVE = 20
    thin = {t: r["n_act"] for t, r in res.items() if r["n_act"] < MIN_ACTIVE}
    if thin:
        print("\n" + "!" * 78)
        print("  부족표본 — 정량 비교를 생략한다.")
        for t, k in thin.items():
            print(f"    {t}: NPU 활성 샘플 {k}개 (최소 {MIN_ACTIVE}개 필요)")
        print("  turbostat 간격 0.1 s 대비 연산 구간이 너무 짧다. reps를 늘려")
        print("  연산 구간이 10 s 이상 되도록 할 것 (예: reps=1000).")
        print("  또한 창에 ssh/마운트 오버헤드가 포함되어 순차×N 외삽이 왜곡된다.")
        print("!" * 78)
        return

    s, p = res["serial"], res["parallel"]
    # 동일 작업량 기준: 순차는 병렬과 같은 일을 하려면 threads 번 반복해야 한다.
    k = a.threads
    print("\n" + "-" * 78)
    print(f"  동일 작업량 비교 — 순차 {k}회 반복  vs  병렬 1회")
    print("-" * 78)
    print(f"    {'':22}{'순차 ×'+str(k):>14}{'병렬':>14}{'차이':>12}")
    for key, name in (("base", "게이팅 없음"), ("gated", "게이팅 적용")):
        sv, pv = s[key] * k, p[key]
        print(f"    {name:22}{sv:12.4f} J{pv:12.4f} J{(sv-pv)/sv*100:11.1f}%")
    print(f"    {'수행시간':22}{s['dur']*k:12.3f} s{p['dur']:12.3f} s"
          f"{(s['dur']*k-p['dur'])/(s['dur']*k)*100:11.1f}%")
    print(f"\n    게이팅 절감 여지:  순차 {(s['base']-s['gated'])/s['base']*100:.1f}%"
          f"  →  병렬 {(p['base']-p['gated'])/p['base']*100:.1f}%")
    print("    (병렬화가 유휴시간을 소진하므로 게이팅으로 아낄 몫이 줄어든다)")
    print("=" * 78)


if __name__ == "__main__":
    main()
