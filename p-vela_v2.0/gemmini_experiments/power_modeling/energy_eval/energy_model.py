#!/usr/bin/env python3
"""Energy model with piecewise time reconstruction (fixes the NPU/CPU attribution).

Basis: CPU + NPU (memory excluded).

Why this exists
---------------
The SSU pipeline compresses each gemmini interval with ONE blended factor
    scale = (rtl_time + embedded_cpu_raw * 0.235) / gemmini_raw_time
and applies it uniformly inside the interval.  But the interval is made of two
parts whose QEMU-to-hardware ratios differ by ~100x:
    NPU emulation (tiled_matmul) : 70.4 s -> 0.146 s   (~480x)
    CPU work (embedded CPU)      :  3.6 s -> 0.840 s   (~4.3x)
Uniform compression therefore leaves the NPU part ~6.4x too long and squeezes
the CPU part ~17.6x too short.  Because NPU power is a step function tied to
the worker-core busy signal, the NPU energy comes out ~7x too high; and because
the vCPU draws much less power while it waits for the NPU (0.13 W) than while
it works (0.38 W), the CPU energy comes out ~2.6x too low.

This tool reclassifies every raw sample and gives each class its own factor,
preserving the model's own aggregate budget:
    npu_compute  (inside gemmini interval, worker busy)  -> rtl_time_s
    embedded_cpu (inside gemmini interval, worker idle)  -> embedded_cpu_scaled_time_s
    cpu_outside  (outside gemmini intervals)             -> raw * cpu_time_scale
Total runtime is unchanged; only the attribution inside the interval changes.
"""
import argparse, bisect, csv, json, os, re, sys

A_NOM, K_NOM = 0.086391591, 0.184415
A_BOOST, K_BOOST = 0.039747769, 2.80057
BOOST_OFF_TH, SYS_SCALE, F_REF = 2.6, 0.5625, 2.512
P_NPU_ACTIVE, P_NPU_IDLE = 0.862, 0.014   # 유휴: 0.013 → 0.014 재측정 (26-11-09)
# 숭실대 최종 납품본(ssu_final, 26-09-30)의 power_model_defaults.env 기준값.
# 이전 실험(26-08)은 CPU_TIME_SCALE=0.235로 수집되었으므로, load()가 run의
# summary.json에 기록된 당시 배율을 읽어 embedded_cpu 성분을 재환산한다.
CPU_TIME_SCALE = 0.3035          # 구: 0.235
# 임계값 90→80. 다만 구 CSV의 busy_core_* 는 수집 시점에 이미 0/100으로
# 이진화되어 저장되므로 재적용해도 결과가 바뀌지 않는다. 신규본은 threshold
# 적용 전 raw_busy_core_* 를 함께 기록하므로, 있으면 그쪽을 쓴다.
BUSY_TH = 80.0                   # 구: 90.0


def cpu_power(f, util):
    f3 = f ** 3
    a, k = (A_NOM, K_NOM) if f <= BOOST_OFF_TH else (A_BOOST, K_BOOST)
    return SYS_SCALE * (a * util * f3 + k)


def load(run_dir, app):
    meta_path = next(iter(sorted(
        p for p in os.listdir(run_dir) if p.startswith("time_scaled_") and p.endswith("_summary.json"))))
    meta = json.load(open(f"{run_dir}/{meta_path}"))
    rtl = dict((meta.get("model_rtl") or {})[app])

    # embedded_cpu_scaled_time_s 는 run 수집 당시의 cpu_time_scale 로 이미 압축된
    # 값이다.  현재 설정과 다르면 비율로 되돌려 다시 적용한다.
    run_cpu_scale = meta.get("cpu_time_scale")
    if rtl.get("embedded_cpu_scaled_time_s") and run_cpu_scale:
        if abs(run_cpu_scale - CPU_TIME_SCALE) > 1e-9:
            rtl["embedded_cpu_scaled_time_s"] *= CPU_TIME_SCALE / run_cpu_scale
            rtl["_rescaled_from"] = run_cpu_scale

    # core mapping + sample window + gemmini intervals, all from the raw bench log
    vcpu_core = npu_core = None
    win, ev = {}, []
    for line in open(f"{run_dir}/raw_bench.log"):
        m = re.search(rf"\[{app}\] \[launch-affinity\] main_vcpu=(\d+) gemmini_executor_vcpu=(\d+)", line)
        if m:
            vcpu_core, npu_core = 8 + int(m.group(1)), 2 + (int(m.group(2)) % 2)
        m = re.search(rf"\[HOST_TS_LINE\] ([0-9.]+) \[{app}\] \[(sample-start|sample-done)\]", line)
        if m:
            win.setdefault(m.group(2), float(m.group(1)))
        # 두 가지 이벤트 형식을 모두 지원:
        #   transformer : [timeline] actor=gemmini label=X  → 다음 timeline 이벤트까지가 구간
        #   polybench   : [timeline] ... event=gemm_begin / gemm_end  → 쌍이 구간
        m = re.search(rf"\[HOST_TS_LINE\] ([0-9.]+) \[{app}\] \[timeline\] actor=(\w+)", line)
        if m:
            ev.append((float(m.group(1)), "gemmini" if m.group(2) == "gemmini" else "other"))
        m = re.search(rf"\[HOST_TS_LINE\] ([0-9.]+) \[{app}\] \[timeline\].*event=(gemm_begin|gemm_end)", line)
        if m:
            ev.append((float(m.group(1)), "begin" if m.group(2) == "gemm_begin" else "end"))
    ev.sort()
    intervals = []
    for i, (t, kind) in enumerate(ev):
        if kind == "gemmini" and i + 1 < len(ev):
            intervals.append((t, ev[i + 1][0]))          # actor=gemmini → 다음 이벤트
        elif kind == "begin":
            nxt = next((tt for tt, kk in ev[i + 1:] if kk == "end"), None)
            if nxt:
                intervals.append((t, nxt))               # gemm_begin → gemm_end
    intervals.sort()
    if not intervals:
        sys.exit(f"ERROR: {app}의 gemmini 구간 이벤트를 찾지 못했습니다 (raw_bench.log 형식 확인 필요).")

    # join vCPU util onto the gemmini-power time grid (zero-order hold)
    vc = [(float(r["sample_tod_s"]), float(r[f"util_core_{vcpu_core}"]))
          for r in csv.DictReader(open(f"{run_dir}/vcpu_power.csv"))]
    vt = [x[0] for x in vc]
    rows = []
    gp = list(csv.DictReader(open(f"{run_dir}/gemmini_power.csv")))
    busy_col = (f"raw_busy_core_{npu_core}"
                if gp and f"raw_busy_core_{npu_core}" in gp[0] else f"busy_core_{npu_core}")
    meta["_busy_col"] = busy_col
    for r in gp:
        t = float(r["sample_tod_s"])
        if not (win["sample-start"] <= t <= win["sample-done"]):
            continue
        i = max(0, min(bisect.bisect_right(vt, t) - 1, len(vc) - 1))
        rows.append((t, float(r[busy_col]) >= BUSY_TH, vc[i][1]))
    return rows, rtl, intervals, vcpu_core, npu_core, meta


def classify(rows, intervals, split=True):
    """Return [(dt_raw, cls, util)] where cls in {npu_compute, embedded_cpu, cpu_outside}."""
    starts = [a for a, _ in intervals]
    out = []
    for (t0, busy, util), (t1, _, _) in zip(rows, rows[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        i = bisect.bisect_right(starts, t0) - 1
        inside = i >= 0 and t0 < intervals[i][1]
        if not inside:
            cls = "cpu_outside"
        elif not split:
            # backend profile이 없는 워크로드: 모델이 구간 전체를 NPU 시간으로 계상하므로
            # 분할할 embedded CPU 성분이 없다 (귀속 오류도 없음)
            cls = "npu_compute"
        else:
            cls = "npu_compute" if busy else "embedded_cpu"
        out.append((dt, cls, util))
    return out


def evaluate(samples, rtl, f, gated, n_npu, mode):
    slow = F_REF / f
    tot = {}
    for dt, cls, _ in samples:
        tot[cls] = tot.get(cls, 0.0) + dt

    if mode == "piecewise":
        # each class gets its own factor; the model's aggregate budget is preserved
        fac = {
            "npu_compute": rtl["rtl_time_s"] / tot.get("npu_compute", 1e-12),
            "embedded_cpu": (rtl.get("embedded_cpu_scaled_time_s") or 0.0) / max(tot.get("embedded_cpu", 0.0), 1e-12),
            "cpu_outside": CPU_TIME_SCALE,
        }
    else:  # "blended" — reproduces the current SSU behaviour
        b = (rtl["rtl_time_s"] + (rtl.get("embedded_cpu_scaled_time_s") or 0.0)) / rtl["gemmini_raw_time_s"]
        fac = {"npu_compute": b, "embedded_cpu": b, "cpu_outside": CPU_TIME_SCALE}

    t = e_cpu = e_npu = 0.0
    for dt, cls, util in samples:
        d = dt * fac[cls]
        if cls != "npu_compute":          # CPU time stretches under DVFS
            d *= slow
        t += d
        e_cpu += cpu_power(f, util) * d
        if cls == "npu_compute":
            e_npu += P_NPU_ACTIVE * d
        elif not gated:
            e_npu += P_NPU_IDLE * d
    if not gated:
        e_npu += P_NPU_IDLE * (n_npu - 1) * t
    return e_cpu + e_npu, t, e_cpu, e_npu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--app", default="distilbert")
    ap.add_argument("--npus", type=int, default=4)
    ap.add_argument("--freqs", default="2.512,2.2,2.0,1.8,1.5,1.2")
    # 참조(baseline) 주파수. SSU 전력모델이 HiFive Premier P550으로 검증된 구간은
    # 1.2~1.4 GHz 이므로, 보드 기준 평가에는 --fref 1.4 --freqs 1.4,1.3,1.2 를 쓴다.
    ap.add_argument("--fref", type=float, default=F_REF)  # noqa: module default
    ap.add_argument("--cpu-time-scale", type=float, default=CPU_TIME_SCALE,
                    help="QEMU→보드 CPU 시간 배율 (신규 납품본 0.3035 / 구 0.235)")
    ap.add_argument("--busy-th", type=float, default=BUSY_TH,
                    help="NPU 활성 판정 Busy%% 임계값 (신규 80 / 구 90)")
    ap.add_argument("--dvfs-freq", type=float, default=2.0,
                    help="frequency used for the NPU-count sensitivity sweep")
    a = ap.parse_args()
    globals()['F_REF'] = a.fref   # 참조 주파수 재설정 (evaluate가 참조)
    globals()['CPU_TIME_SCALE'] = a.cpu_time_scale
    globals()['BUSY_TH'] = a.busy_th

    rows, rtl, intervals, vc, nc, meta = load(a.run_dir, a.app)
    split = bool(rtl.get("embedded_cpu_scaled_time_s") or 0.0)
    samples = classify(rows, intervals, split)
    tot = {}
    for dt, cls, _ in samples:
        tot[cls] = tot.get(cls, 0.0) + dt
    if tot.get("npu_compute", 0.0) <= 0:
        sys.exit(f"ERROR: {a.app}의 NPU 활성 샘플이 0개입니다 "
                 f"(코어 {nc} busy>=90%인 구간이 gemmini 인터벌 내에 없음). 결과 신뢰 불가.")

    print(f"\n{'='*80}")
    print(f"  {a.app} — piecewise energy model   (basis: CPU + NPU, memory excluded)")
    print(f"  run {os.path.basename(a.run_dir.rstrip('/'))[:15]} | {a.npus} NPUs, 1 in use"
          f" | vCPU core {vc}, NPU core {nc}")
    print(f"{'='*80}")
    rs = rtl.get("_rescaled_from")
    print(f"  파라미터: f_ref {F_REF} GHz · cpu_time_scale {CPU_TIME_SCALE}"
          + (f" (run은 {rs}로 수집 → 재환산)" if rs else "")
          + f" · busy_th {BUSY_TH}% ({meta.get('_busy_col','?')})")
    print(f"            sys_scale {SYS_SCALE} · NPU {P_NPU_ACTIVE}/{P_NPU_IDLE} W")
    print("  raw 시간 분류와 적용 배율:")
    if not split:
        print(f"  ※ 이 워크로드는 backend profile(raw_tiled_matmul_profile_s)이 없어")
        print(f"     모델이 gemmini 구간 전체를 NPU 시간으로 계상합니다 → 귀속 오류 없음")
    for cls, target in (("npu_compute", rtl["rtl_time_s"]),
                        ("embedded_cpu", rtl.get("embedded_cpu_scaled_time_s") or 0.0),
                        ("cpu_outside", tot.get("cpu_outside", 0) * CPU_TIME_SCALE)):
        raw = tot.get(cls, 0.0)
        fmt = f"1/{raw/target:.0f}" if target > 0 else "—"
        print(f"    {cls:14} raw {raw:7.2f} s → {target:7.4f} s   (배율 {fmt})")

    # 현행 방식은 gemmini 구간에 내부 구조가 없어 DVFS를 표현할 수 없다.
    # 따라서 baseline 에너지만 참조값으로 제시한다.
    ae, at, ac, an = evaluate(samples, rtl, F_REF, False, a.npus, "blended")
    be, bt, bc, bn = evaluate(samples, rtl, F_REF, False, a.npus, "piecewise")
    print(f"\n  BASELINE 비교 (DVFS·gating 없음, {a.npus}-NPU)")
    print(f"    현행 (혼합 배율) : {ae:.4f} J   CPU {ac:.4f} + NPU {an:.4f}   runtime {at:.3f} s")
    print(f"    수정 (구간 분할) : {be:.4f} J   CPU {bc:.4f} + NPU {bn:.4f}   runtime {bt:.3f} s")
    print(f"    → NPU {an/bn:.1f}배 과대, CPU {bc/ac:.1f}배 과소가 정정됨 (runtime 동일)")

    print(f"\n  저전력 기법 절감률 (수정 모델 기준)")
    print(f"    {'freq':>6} {'slow':>5} | {'gating':>8} {'DVFS':>8} {'DVFS+gate':>10} | {'runtime':>8} {'energy':>8}")
    print(f"    {'-'*6} {'-'*5} | {'-'*8} {'-'*8} {'-'*10} | {'-'*8} {'-'*8}")
    for fr in [float(x) for x in a.freqs.split(",")]:
        g = evaluate(samples, rtl, F_REF, True, a.npus, "piecewise")[0]
        d = evaluate(samples, rtl, fr, False, a.npus, "piecewise")[0]
        bg, bgt, _, _ = evaluate(samples, rtl, fr, True, a.npus, "piecewise")
        print(f"    {fr:6.3f} {F_REF/fr:5.2f} | {(be-g)/be*100:7.1f}% {(be-d)/be*100:7.1f}%"
              f" {(be-bg)/be*100:9.1f}% | {bgt:7.3f}s {bg:7.4f}J")

    print(f"\n  NPU 개수 민감도 (DVFS {a.dvfs_freq}GHz + gating)")
    for n in (1, 2, 4, 8):
        b0 = evaluate(samples, rtl, F_REF, False, n, "piecewise")[0]
        b1 = evaluate(samples, rtl, a.dvfs_freq, True, n, "piecewise")[0]
        print(f"    {n} NPU: {b0:.4f} J → {b1:.4f} J = {(b0-b1)/b0*100:5.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
