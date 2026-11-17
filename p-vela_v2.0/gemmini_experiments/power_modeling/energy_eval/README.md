# Energy Evaluation — 에너지 20% 절감 정량 평가

**분모 = CPU + NPU (메모리 제외)**. 기존 run 산출물에 합성 결정 로그(주파수·gating)를
입혀 저전력 기법의 에너지 절감률을 재구성한다. 재측정이 아니라 재구성이므로 두
시나리오에 동일한 측정 노이즈가 들어가 비교가 공정하다.

## 파일

| 파일 | 용도 |
|---|---|
| `energy_model.py` | **주 도구.** 구간 분할(piecewise) 시간축으로 에너지 계산 + 주파수 스윕 |
| `energy_compare.py` | 초기 버전 (scaled CSV 기반, alpha 근사). 회귀 검증용으로 보존 |

```bash
RUN=.../dual_app_power_modeling/logs/distilbert_polybench_gemm/20260812_004450_...
python3 energy_model.py "$RUN" --app distilbert --npus 4
python3 energy_model.py "$RUN" --app polybench_gemm --npus 8 --freqs 2.512,2.0,1.5
```

⚠️ run 디렉터리는 **이름으로 고정**할 것. `ls -td`(mtime 정렬)는 오래된 무효 run이
갱신되면 그것을 집는다 (실제로 발생).

## 왜 시간축을 다시 계산하는가 — SSU 모델의 귀속 오류

SSU 파이프라인은 gemmini 구간을 **혼합 배율 하나**로 압축하고 구간 내부에 균일
적용한다 (`time_scale_gemmini_logs.py` L1215-1216):

```
scale = (rtl_time + embedded_cpu_raw × 0.235) / gemmini_raw_time
```

그런데 구간은 QEMU↔하드웨어 비율이 100배 다른 두 부분으로 구성된다:

| 구성요소 | raw | 하드웨어 | 압축률 | 출처 |
|---|---|---|---|---|
| NPU 연산 (tiled_matmul) | 70.4 s | 0.146 s | ~480배 | RTL 사이클 모델 (실측 검증) |
| CPU 작업 (embedded CPU) | 3.6 s | 0.840 s | ~4.3배 | 고정 계수 `cpu_time_scale=0.235` |

균일 압축의 결과 — **합계는 상쇄로 맞고 내부 구성만 뒤바뀐다**:

| | 적용된 값 | 의도한 값 | 오차 |
|---|---|---|---|
| NPU 부분 | 0.938 s | 0.146 s | 6.4배 길다 |
| CPU 부분 | 0.048 s | 0.840 s | 17.6배 짧다 |
| 합계 | 0.986 s | 0.986 s | ✓ |

에너지에는 두 방향으로 나타난다:
- **NPU**: 전력이 busy 기반 계단 함수(0 / 0.862 W)라 활성 시간이 길어진 만큼 과대 → **약 4~7배**
- **CPU**: vCPU가 NPU 대기 중일 때(0.13 W)와 실제 일할 때(0.38 W)의 전력이 3배 다른데,
  대기 구간이 구간의 95%를 점유 → **과소**

원인은 하나다: **gemmini 구간 내부에 NPU 연산 시간과 CPU 작업 시간의 구분이 없다.**

## 수정 방식

모든 raw 샘플을 3분류하고 각 분류에 자기 배율을 적용한다. **모델의 총 시간 예산은
그대로 보존**하므로 runtime은 변하지 않고 귀속만 정정된다.

```
npu_compute   (gemmini 구간 내 & worker busy≥90%)  → 합계를 rtl_time_s 로
embedded_cpu  (gemmini 구간 내 & worker idle)      → 합계를 embedded_cpu_scaled_time_s 로
cpu_outside   (gemmini 구간 밖)                     → 각 샘플 × cpu_time_scale(0.235)
```

DVFS의 slowdown은 `embedded_cpu`·`cpu_outside`에만 적용한다 (NPU는 독립 클럭).

**backend profile이 없는 워크로드**(polybench 등, `raw_tiled_matmul_profile_s = null`)는
모델이 gemmini 구간 전체를 NPU 시간으로 계상하므로 **분할할 성분이 없고 귀속 오류도
없다.** 도구가 이를 자동 판정해 분할을 건너뛰며, 두 모드의 결과가 동일하게 나오는 것으로
자기검증된다. 즉 이 오류는 **backend profile을 보고하는 transformer 계열에 한정**된다.

**주의**: `embedded_cpu`의 raw 총량은 turbostat 기준(5.15 s)이 앱 프로파일 기준(3.57 s)보다
크다 — turbostat의 100 ms/90% threshold 때문. 합계를 모델 예산에 맞추므로 총량은
정합하지만, 분류 경계에는 이 정도 오차가 있다.

## 모델

```
CPU  P = 0.5625 × (a·util·f³ + b)      a,b = SSU 계수 (회귀 검증 오차 0.00%)
NPU  active 0.862 W / powered-idle 0.013 W / gated 0 W
```
적분은 사다리꼴(`summarize_power_csv.py`와 동일). 좌단 직사각형은 이진 NPU 신호에서
7% 과대평가하므로 쓰지 말 것.

## 알려진 근사

- 이 도구는 raw `gemmini_power.csv` 그리드(약 104 ms)에서 계산하고 vCPU util을 ZOH로
  조인한다. SSU의 scaled CSV(보간된 미세 그리드)와 비교하면 CPU 에너지가 약 10%
  차이난다 (0.529 vs 0.585 J). 절감률(비율) 결론에는 영향 없으나 절대값 인용 시 유의.
- 구간 못 찾음 / NPU 활성 샘플 0개면 **즉시 중단**한다 (조용히 틀린 답 방지).
- run 디렉터리는 이름으로 고정. `ls -td`는 무효 run을 집을 수 있다.

## 결과 (DistilBERT 추론 1건, 4-NPU 중 1개 사용)

```
BASELINE   현행 1.394 J (CPU 0.529 + NPU 0.865)
           수정 0.837 J (CPU 0.627 + NPU 0.210)   runtime 1.653 s (동일)
           → NPU 4.1배 과대 / CPU 1.2배 과소 정정

절감률     gating 단독      10.1%
           DVFS+gating      20.1% @2.2GHz / 25.2% @2.0GHz / 32.5% @1.5GHz (최적)
           NPU 개수별       1-NPU 19.0% → 2-NPU 21.2% → 4-NPU 25.2% → 8-NPU 32.2%

PolyBench (NPU 지배적, 귀속 오류 없음): gating 3.8%, DVFS 효과 거의 없음
→ 20% 주장은 CPU 비중이 큰 transformer 워크로드로 해야 함
```

최적 주파수가 존재한다 (이론값 f* = (β/(2·α·util))^⅓ ≈ 1.5 GHz). 더 내리면 정적 전력
페널티가 f² 이득을 잡아먹는다 — **"주파수를 최대한 낮춘다"가 답이 아니다.**

## Phase 2 이후

합성 로그를 실제 governor 로그로 교체 (`epoch_ns, cpu_id, freq_mhz` / `epoch_ns, npu_id, powered`,
변화점만 기록 + zero-order hold). slowdown(f)를 현재 `f_ref/f`에서 저전력팀의 명령어
복잡도 회귀 모델로 교체.

## 동작점(주파수) 근거 — 2026-09-07 갱신

SSU 워크숍 자료로 전력모델의 출처와 **검증 범위**가 확인되었다.

| 항목 | 내용 |
|---|---|
| CPU 전력모델 구축 | AMD Ryzen 5 7600 + QEMU 9.2.0, 워크로드 FFT/INT64/Float/JMP/OpenSSL/7z |
| RISC-V 이식 | scaling factor θ 적용 (`P' = θ·P`) — 코드의 `sys_scale = 0.5625` |
| **검증 대상/범위** | **실보드 HiFive Premier P550, 1.2 / 1.3 / 1.4 GHz** |
| 검증 오차 | 평균 0.0277 W (2.30%) |
| CPU 성능모델 | 보드 수행시간 / QEMU 수행시간 비율, SPEC CPU2017 4종(mcf·lbm·x264·deepsjeng) — 코드의 `cpu_time_scale = 0.235` |
| NPU 전력 | ASAP7 7nm 200 MHz 합성, Fmax 258 MHz, Total 0.862 W (Internal 3.27e-1 + Switching 5.33e-1 + Leakage 9.93e-5) |
| NPU cycle 모델 | `35 + 16·M_t·N_t·K_t − pad_M + max(0,4−M) + 16·max(0,P−257)`, Verilator 3,503 케이스 전수 일치 |

**따라서 평가 동작점은 1.4 GHz로 고정한다** (`--fref 1.4`). 기존 실험 러너가 쓰던
`FIXED_FREQ_GHZ=2.512`는 검증 범위 밖이며, SSU 자신의 `power_sim_gemmini.py` /
`power_sim_cpu_only.py` 기본값(1.4)과도 어긋난다.

### 결과: DVFS 전략 폐기, power gating 단독으로 목표 달성

전력모델의 에너지 최적 주파수는 `f* = (β / 2αu)^(1/3) ≈ 1.44 GHz`(측정 util 0.357)로
**보드 상한 1.4 GHz와 사실상 일치**한다. 즉 P550 동작 범위(1.2–1.4 GHz) 전체가 이미
최적점 이하이므로 주파수를 낮추면 에너지가 오히려 늘어난다(1.2 GHz에서 +1.4%).

1.4 GHz 고정, 유휴 NPU power gating 단독 절감률:

| NPU 개수 | 1 | 2 | **4** | 8 | 16 |
|---|---|---|---|---|---|
| 절감률 | 5.0% | 9.8% | **18.3%** | 31.1% | 47.6% |

4-NPU에서 18.3%, 8-NPU에서 31.1% — 목표 20%는 NPU 5개 이상이면 gating만으로 달성된다.
DVFS 기여는 0으로 두는 것이 방어 가능한 서술이다.

재현:
```bash
python3 energy_model.py <run_dir> --app distilbert --npus 4 \
        --fref 1.4 --freqs 1.4,1.3,1.2 --dvfs-freq 1.3
python3 make_figures.py          # sweep_1p4.json → energy_summary.png
```

## 파라미터 갱신 — 숭실대 최종 납품본 반영 (2026-10-07)

`ssu_final.zip`(용역 최종본, 26-09-30)의 `power_model_defaults.env` 기준으로 상수를 맞췄다.

| 항목 | 구 | 신규 | 비고 |
|---|---:|---:|---|
| `CPU_TIME_SCALE` | 0.235 | **0.3035** | 실질 영향 있음 |
| `GEMMINI_UTIL_THRESHOLD_PCT` | 90 | **80** | 구 run에는 영향 없음(아래) |
| NPU 유휴 전력 | 0 (우리가 0.013 가정) | **0.013** | 납품본이 채택 — 우리 가정과 일치 |

**구 run의 임계값 변경이 무의미한 이유**: 26-08 수집분의 `gemmini_power.csv`는
`busy_core_<N>` 컬럼에 **수집 시점에 이미 0/100으로 이진화된 값**만 담고 있다
(고유값 확인 결과 `0.000000` / `100.000000` 두 가지뿐). 임계값 전 원시 Busy%를
담는 `raw_busy_core_<N>` 컬럼은 신규 납품본부터 생긴다. 따라서 80%를 재적용해도
결과가 바뀌지 않으며, `load()`는 `raw_busy_core_*`가 있으면 그쪽을 우선 사용한다.

**embedded_cpu 재환산**: `summary.json`의 `embedded_cpu_scaled_time_s`는 run 수집
당시의 `cpu_time_scale`로 **이미 압축된 값**이다. `load()`가 같은 JSON의
`cpu_time_scale` 필드를 읽어 `× (현재 / 당시)`로 되돌린 뒤 다시 적용한다.

### 갱신 결과 (DistilBERT, 1.4 GHz, 4-NPU 중 1개 사용)

```
              runtime      CPU        NPU       baseline
구    0.235   1.653 s   0.2503 J   0.2098 J   0.4601 J
신규  0.3035  2.093 s   0.3187 J   0.2327 J   0.5513 J
```

| 장착 NPU | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 구 (0.235) | 5.0% | 9.8% | 18.3% | 31.1% | 47.6% |
| **신규 (0.3035)** | **5.4%** | **10.6%** | **19.4%** | **32.7%** | **49.4%** |

CPU 시간이 29% 늘어 runtime이 1.653 → 2.093 s가 되고, 유휴 누설분이 함께 커져
**절감률이 오히려 소폭 상승**한다(0.0841 → 0.1069 J). 4-NPU는 19.4%로 목표 20%에
근접하나 여전히 미달이고, 5개 이상에서 달성된다는 결론은 유지된다.

### ⚠ 보고서와 코드가 어긋나는 값

납품본 코드와 용역결과보고서 본문이 다르다. 현재 도구는 **코드 쪽**을 따랐다.

| | 보고서 | 코드 |
|---|---|---|
| 보정계수 γ | 0.235 | 0.3035 |
| scaling factor θ | 0.54 | 0.5625 |
| NPU 활성 임계값 | 90% | 80% |
| 정적 전력 | "보드 idle 실측값, 동적만 θ 보정" | `sys_scale × (a·u·f³ + k)` — θ가 정적항에도 곱해짐 |

마지막 항목은 DVFS 결론(f\* 위치)을 좌우하므로 확인이 필요하다.

재현:
```bash
python3 energy_model.py <run_dir> --app distilbert --npus 4 --fref 1.4 \
        --freqs 1.4,1.3,1.2 --cpu-time-scale 0.3035 --busy-th 80
```

## NPU 유휴 전력 갱신 — 0.013 → 0.014 W (2026-11-09)

재측정 결과에 따라 전원은 들어와 있지만 연산하지 않는 NPU의 전력을 0.014 W로 올렸다.
활성 전력 0.862 W와 차단 0 W는 그대로다. 바꾼 곳은 `energy_model.py`,
`summarize_multi_npu.py`, `energy_compare.py`와 계측 쪽 `power_model_defaults.env`
(ssu 트리, program 패키지)다. 납품 원본 `ssu_final/`은 수정하지 않았다.

```
              runtime      CPU        NPU       baseline (4-NPU, 1.4 GHz)
0.013 W       2.093 s   0.3187 J   0.2327 J   0.5513 J
0.014 W       2.093 s   0.3187 J   0.2409 J   0.5596 J
```

| 장착 NPU | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 유휴 0.013 W | 5.4% | 10.6% | 19.4% | 32.7% | 49.4% |
| **유휴 0.014 W** | **5.8%** | **11.3%** | **20.6%** | **34.3%** | **51.2%** |

gating 후 에너지(0.4444 J)는 유휴 전력과 무관하므로, 유휴 전력이 커진 만큼
baseline만 커져 절감률이 올라간다. **4-NPU에서 20.6%로 목표 20%를 넘는다.**
DVFS 결론은 그대로다(1.3 GHz −1.8%, 1.2 GHz −4.4%, 최소점 ≈1.44 GHz).

`sweep_1p4.json`은 기존 파일을 오차 0으로 재현하는 스크립트로 다시 만들었다
(`params.npu_idle_w` 필드 추가). 재현:
```bash
python3 energy_model.py <run_dir> --app distilbert --npus 4 --fref 1.4 \
        --freqs 1.4,1.3,1.2 --dvfs-freq 1.4
python3 make_figures.py
```
