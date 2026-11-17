#!/usr/bin/env python3
"""에너지 20% 절감 평가 결과 그래프 (DistilBERT, 4-NPU 중 1개 사용).

기준 동작점 = 1.4 GHz. 근거: SSU 전력모델은 AMD Ryzen 5 7600에서 구축 후
scaling factor θ=0.5625로 RISC-V에 이식되었고, 실보드 HiFive Premier P550
1.2~1.4 GHz 구간에서만 검증됨(평균오차 0.0277 W, 2.30%).
에너지 수치는 모델 기반 추정치이며 실측 전력계 값이 아니다.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({
    "font.family": "Noto Sans CJK KR", "font.size": 11,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # 검증된 categorical slot 1·2·3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
BAND = "#dfe9f7"

D = json.load(open("sweep_1p4.json"))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


fig = plt.figure(figsize=(15.5, 4.9))
gs = gridspec.GridSpec(1, 3, width_ratios=[0.85, 1.35, 1.0], wspace=0.30,
                       left=0.05, right=0.985, top=0.79, bottom=0.22)

# ── ① BASELINE 귀속 정정 (주파수와 무관) ────────────────────────────────
axA = fig.add_subplot(gs[0]); style(axA)
cpu, npu = [0.2968, 0.3187], [1.0889, 0.2409]
x = [0, 1]
axA.bar(x, cpu, width=0.5, color=S1, label="CPU", zorder=3)
axA.bar(x, npu, width=0.5, bottom=[c + 0.004 for c in cpu], color=S2,
        label="NPU", zorder=3)                      # 2px surface gap
for i in x:
    axA.text(i, cpu[i]/2, f"{cpu[i]:.3f}", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    axA.text(i, cpu[i]+npu[i]/2, f"{npu[i]:.3f}", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    axA.text(i, cpu[i]+npu[i]+0.035, f"{cpu[i]+npu[i]:.3f} J", ha="center",
             color=INK, fontsize=11, fontweight="bold")
axA.annotate("", xy=(0.86, 0.66), xytext=(0, 1.50),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4,
                             connectionstyle="arc3,rad=-0.25"))
axA.text(0.52, 1.53, "NPU 에너지 4.5배 과대\n계상분 정정", ha="center",
         color=MUTED, fontsize=9.5, linespacing=1.4)
axA.set_xticks(x)
axA.set_xticklabels(["현행\n(혼합 배율)", "수정\n(구간 분할)"], fontsize=10, color=INK)
axA.set_ylim(0, 1.95); axA.set_ylabel("에너지 (J)")
axA.set_title("① BASELINE 에너지 — 귀속 정정", fontsize=12,
              fontweight="bold", color=INK, pad=12, loc="left")
axA.text(0, -0.31, "runtime 2.093 s 동일 (총 시간 예산 보존)",
         transform=axA.transAxes, fontsize=9, color=MUTED)
axA.legend(frameon=False, fontsize=10, loc="upper right",
           bbox_to_anchor=(1.03, 1.0), labelcolor=MUTED)

# ── ② DVFS — 하향 여지가 없음 ───────────────────────────────────────────
# 곡선의 모든 점은 같은 출처다: 한 번 측정한 가동률 트레이스를 주파수만 바꿔
# 전력 공식에 다시 넣은 값.  따라서 "검증 구간 / 외삽 구간" 구분은 성립하지
# 않으며(공식의 보드 대조는 곡선의 모양을 결정하는 α:β 비율을 검증하지 못한다),
# 음영과 실선/파선 구분을 두지 않는다.
axB = fig.add_subplot(gs[1]); style(axB)
f, e = D["freq"], D["e_gated"]
axB.plot(f, e, color=S1, lw=2.0, zorder=3)
i14 = f.index(1.4)
axB.plot([1.4], [e[i14]], "o", ms=9, color=S1, mec="#fcfcfb", mew=2, zorder=4)
axB.set_xlim(1.0, 2.5)
axB.set_xlabel("CPU 주파수 (GHz)"); axB.set_ylabel("총 에너지 (J, gating 적용)")
axB.set_title("② CPU DVFS — 하향 여지 없음", fontsize=12,
              fontweight="bold", color=INK, pad=12, loc="left")
axB.text(0, -0.31, "단일 측정 트레이스를 주파수별로 재평가한 모델 곡선 · "
                   "최소점 f* = (β / 2αu)^(1/3) ≈ 1.44 GHz",
         transform=axB.transAxes, fontsize=9, color=MUTED)

# ── ③ NPU power gating — 개수 민감도 (주된 절감 수단) ────────────────────
axC = fig.add_subplot(gs[2]); style(axC)
ns = [1, 2, 4, 8, 16]
pct = [(D["npu_gating"][str(n)][0] - D["npu_gating"][str(n)][1])
       / D["npu_gating"][str(n)][0] * 100 for n in ns]
xs = range(len(ns))
cols = [S3 if p >= 20 else "#a8ded0" for p in pct]
axC.bar(xs, pct, width=0.55, color=cols, zorder=3)
for i, p in enumerate(pct):
    axC.text(i, p + 1.2, f"{p:.1f}%", ha="center", color=INK,
             fontsize=10.5, fontweight="bold")
axC.axhline(20, color=S2, lw=1.6, ls=(0, (5, 3)), zorder=4)
axC.text(len(ns) - 0.4, 21.5, "목표 20%", ha="right", color=S2, fontsize=10,
         fontweight="bold")
axC.set_xticks(list(xs)); axC.set_xticklabels([f"{n}" for n in ns], color=INK)
axC.set_xlabel("시스템 내 VelaNPU 개수")
axC.set_ylabel("에너지 절감률 (%)")
axC.set_ylim(0, 56)
axC.set_title("③ 유휴 NPU power gating 단독 효과", fontsize=12,
              fontweight="bold", color=INK, pad=12, loc="left")
axC.text(0, -0.31, "1.4 GHz · cpu_time_scale 0.3035 · DistilBERT 1건이 NPU 1개만 사용",
         transform=axC.transAxes, fontsize=9, color=MUTED)

fig.text(0.05, 0.925,
         "DistilBERT (INT8, SST-2) · 숭실대 전력/성능 모델 정정본 · "
         "동작점 1.4 GHz (HiFive Premier P550 검증 구간 상한)",
         fontsize=11.5, color=INK, fontweight="bold")
fig.text(0.05, 0.878,
         "모델 기반 추정치 — 실측 전력계 값 아님. 파라미터는 숭실대 최종 납품본(26-09-30) 기준 "
         "cpu_time_scale 0.3035 · NPU 0.862 W / 유휴 0.014 W (ASAP7 7nm, 200 MHz).",
         fontsize=9.5, color=MUTED)

fig.savefig("energy_summary.png", dpi=200)
print("wrote energy_summary.png")
