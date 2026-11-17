# Power Simulation

`power_modeling/power_sim`은 raw CPU/Gemmini power 계산, Gemmini RTL cycle
계산, time-scaled 로그 생성을 담당한다.

## 주요 파일

- [power_model_defaults.env](./power_model_defaults.env)
  - single/dual 및 Python에서 공유하는 전력·주기·주파수·시간 배율·scaled plot 기본값
- [power_model_config.py](./power_model_config.py)
  - 위 파일의 여러 `KEY=value` 항목을 읽는 Python 공통 로더
- [power_sim_vcpu_procstat.py](./power_sim_vcpu_procstat.py)
  - single-app에서 guest `/proc/stat`과 shared host `turbostat`을 함께 읽어
    실시간 `raw_power.csv`를 생성함
- [power_sim_gemmini.py](./power_sim_gemmini.py)
  - 공통 전력식과 turbostat parser를 제공하며 dual-app의 host collector로도 사용함
- [merge_split_power_csv.py](./merge_split_power_csv.py)
  - dual-app에서 분리 수집한 CPU/Gemmini CSV의 timestamp 병합용
- [gemmini_hw_cycle_model.py](./gemmini_hw_cycle_model.py)
  - 검증된 no-DMA `LOOP_WS` 및 application workload cycle을 계산함
- [time_scale_gemmini_logs.py](./time_scale_gemmini_logs.py)
  - raw 로그를 CPU/Gemmini 목표 시간축으로 변환함

현재 single/dual 실행 경로에서 사용하는 코드만 유지한다.
실험 실행에서 사용하지 않는 검증용 테스트는 제거했으며 Git 이력에서 복구할 수 있다.
이전 7z/OpenSSL·CPU-only·mpstat 기반 simulator와 미사용 후처리 도구는
제거했으며 Git 이력에서 복구할 수 있다. CPU 전력식은
`power_sim_gemmini.py`에 유지된다.

## 공통 설정 변경

[power_model_defaults.env](./power_model_defaults.env) 한 곳에서 다음 값을 수정한다.

- Gemmini full/idle power와 utilization threshold
- CPU 주파수·platform scale·AMD 전력식 계수
- Host turbostat·power polling·guest vCPU sampling 주기
- Time mode·Gemmini 주파수·CPU 배율
- Scaled plot 표시 여백·가로 크기·시간 눈금

Shell runner는 `source`로, Python은 `power_model_config.py`로 같은 파일을 읽는다.
값은 shell 명령이나 변수 참조 없이 `KEY=value` 리터럴로 작성한다.
기존 명령행 override가 기본값보다 우선하며, dual runner의 환경변수 override도
유지된다. 설정 변경은 새 프로세스부터 적용하며 기존 run 로그는 수정하지 않는다.
코어 배치·SSH·application 실행 설정과 RTL 구조/profile 상수는 각 코드에 유지한다.

Python 직접 실행의 기본값도 현재 실험과 통일했다. Host collector의 주파수와
polling은 기존 `1.4 GHz / 1 s`에서 `2.512 GHz / 0.1 s`로, 시간 변환은
`manual / CPU 1.0`에서 `rtl-cycle / CPU 0.3035`로 바뀐다. 기존 single/dual
runner가 명시적으로 전달하던 기본 수치는 그대로다. 필요한 경우 CLI로 override한다.

## Raw power 계산

Canonical single-app runner는 두 source를 Guest의 통합 collector가 실시간으로
읽고 하나의 power CSV를 기록한다.

```text
Guest /proc/stat + shared raw_turbostat.log -> raw_power.csv
```

`raw_power.csv`에는 vCPU, Gemmini, 두 component의 total과 `p_total_w`가 같은
행에 기록된다. Dual runner는 CPU/Gemmini split CSV를 기록한 뒤 병합한다.

`raw_busy_core_<Gemmini core>`는 collector가 읽은 threshold 전 host Busy%다.
`busy_core_<Gemmini core>`는 0/100% 모델 값, `busy_core_<vCPU core>`는 guest
CPU utilization이다. Single의 raw/scaled 그래프 수치는 power CSV만 사용하며,
bench 로그는 시간 기준과 구간 표시를 위해 별도로 읽는다. 새 column은 dual CSV
병합과 시간 변환에서도 보존된다. 이전 CSV에 새 schema 행을 이어 쓰지는 않는다.

Gemmini worker는 실시간 collector가 CSV에 기록하기 전에 기본 90% threshold로
이진화한다.

```text
Busy% >= threshold  -> util=1.0, power=full_util_power_w
Busy% <  threshold  -> util=0.0, power=0.013 W
```

기본 `full_util_power_w`는 `power_model_defaults.env`의 현재 설정값
`0.862 W`를 사용한다. threshold와 power는 각각
`--gemmini-util-threshold-pct`, `--gemmini-full-util-power-w`로 변경한다.

Single/dual plot의 Gemmini utilization은 power CSV의 `util_core_<N>`에 100을
곱해 표시하며 전력과 같은 `sample_tod_s`를 사용한다. 그래프에서 threshold를
다시 적용하지 않는다. Dual의 별도 host/guest 진단에는 원본 turbostat와
`/proc/stat` CSV를 사용한다. Single plot은 이 로그들을 직접 읽지 않는다.
별도의 threshold 로그는 생성하지 않는다.
Guest collector는 raw-only vCPU의 `busy_core_<N>`, `util_core_<N>`도 power CSV에
기록한다. 이 column은 guest executor/housekeeping 그래프용이며 전력 합산에는
사용하지 않는다. 기존 column이 없는 run은 새 실험으로 수집해야 한다.
시간 변환에는 원본 turbostat 로그를 전달하며, scaled plot도 변환된 power CSV의
utilization을 그대로 사용한다.

직접 실행 예시는 다음과 같다.

```bash
python3 gemmini_experiments/power_modeling/power_sim/power_sim_gemmini.py \
  --turbostat /path/to/turbostat.log \
  --out-log /path/to/gemmini_power.csv \
  --gemmini-cores 2 \
  --fixed-freq-ghz 2.512 \
  --sys-scale 0.5625 \
  --gemmini-full-util-power-w 0.862 \
  --gemmini-util-threshold-pct 90 \
  --poll-interval 0.1
```

## Gemmini RTL cycle model

[gemmini_hw_cycle_model.py](./gemmini_hw_cycle_model.py)는 production Gemmini Top
TB에서 검증한 `LOOP_WS accept -> idle` cycle model을 구현한다. 현재 검증 범위는
DMA 제외, `DIM=16`, `M_TILES=1`이다.

```text
M          = 16*M_TILES - PAD_M
P          = N_TILES*K_TILES
compute    = 16*M_TILES*P - PAD_M
drain      = max(0, 4-M)
backpressure = 16*max(0, P-257)

LOOP_WS cycles = 35 + compute + drain + backpressure
```

하나의 API 호출에서 여러 `LOOP_WS`를 연속 실행하면 명령 사이 고정 overhead를
공유한다.

```text
workload cycles
  = 각 LOOP_WS cycle 합
  - 35*(LOOP_WS 호출 수 - API 호출 수)
```

현재 application profile은 `distilbert`, `distilroberta`, `distilgpt2`,
`polybench_gemm`을 지원한다. PolyBench는 `MEDIUM_DATASET` 기준이다.

```bash
python3 gemmini_experiments/power_modeling/power_sim/gemmini_hw_cycle_model.py \
  distilbert --executions 1 --frequency-mhz 200
```

## Time scaling

[time_scale_gemmini_logs.py](./time_scale_gemmini_logs.py)는 QEMU raw 구간을
목표 CPU와 Gemmini RTL 시간으로 치환한다.

필수 입력:

```text
bench.log      sample window, Gemmini event, backend profile
turbostat.log  host core utilization
power.csv      component별 power와 total power
```

선택 입력 `--guest-util-csv`를 주면 guest `/proc/stat` CSV도 같은 시간축으로
변환한다.

### 측정 구간

```text
inference CPU 구간 = sample-start/done window - Gemmini active interval
Gemmini 구간       = actor=gemmini event -> 대응하는 *_end event
```

- CPU 구간에는 `--cpu-time-scale`을 적용한다.
- Gemmini 구간에는 선택한 Gemmini time mode를 적용한다.
- 변환된 구간 뒤 timestamp는 줄어든 시간만큼 앞으로 이동한다.
- bench, turbostat, power, guest-util에 동일한 구간 변환을 적용한다.

### Time mode

| Mode | 설명 |
|---|---|
| `rtl-cycle` | 현재 runner 기본값. workload cycle을 지정 주파수 시간으로 변환 |
| `cpu-ref-auto` | CPU reference bench 비율을 사용하는 과거 호환 mode |

스크립트 직접 실행과 single/dual runner 모두 공통 설정의 mode와 CPU time scale을
사용한다. 현재 기본값은 `rtl-cycle`, `0.3035`다. Single/dual runner는 RTL만
지원하며, Python 유틸의 `cpu-ref-auto` 호환 경로는 별도로 유지한다.
`manual` mode, `--time-scale`, `--model-time-scale`과 application별 고정 배율은
제거했다. CPU의 `--cpu-time-scale`은 그대로 유지한다.

`rtl-cycle`의 Gemmini 목표 시간은 다음과 같다.

```text
RTL time = workload cycle total / (frequency MHz * 1,000,000)

embedded CPU raw time
  = raw Gemmini active time - backend-profile tiled_matmul time

Gemmini target time
  = RTL time + embedded CPU raw time * cpu_time_scale

Gemmini time scale
  = Gemmini target time / raw Gemmini active time
```

profile의 예상 Gemmini interval 수와 bench에서 관측한 interval 수가 다르면
잘못된 workload 적용을 막기 위해 실행을 중단한다.

고정된 workload profile·실행 횟수·주파수에서 `RTL time`은 항상 같다.
`WorkloadProfile.cycles_per_execution`은 처음 사용할 때 LOOP_WS cycle을
합산한 뒤 같은 프로세스 안에서 재사용한다. 실행 횟수나 주파수를 바꿔도
profile 합산을 반복하지 않는다. 프로세스를 새로 시작하면 한 번 다시 계산하며,
별도 디스크 cache나 중복된 수치 테이블은 두지 않는다.
후처리는 이 cycle에 실행 횟수를 곱하고 주파수로 나눈다. Transformer 실행 횟수는
`sample-start` 수, PolyBench는 `gemm_begin` 수로 결정한다.
다만 내부 CPU 보정분과 분모인 raw Gemmini 시간은 실행 로그에서 가져오므로
최종 목표 시간과 실제 적용 배율은 실행마다 달라질 수 있다.
현재 변환 방식은 application 총시간을 맞추며 개별 GEMM마다 서로 다른 RTL
목표 시간을 배정하지 않는다. Single/dual의 목표 시간·배율 계산과 profile 검증은
`resolve_rtl_workload_time_scale()` 하나를 공유한다. 측정 시간·내부 CPU 보정분은
cache하지 않으며, 기존 RTL cycle 식과 내부 CPU 보정식은 그대로 유지한다.

### Single-app 예시

```bash
python3 gemmini_experiments/power_modeling/power_sim/time_scale_gemmini_logs.py \
  --bench /path/to/raw_bench.log \
  --turbostat /path/to/raw_turbostat.log \
  --power-csv /path/to/raw_power.csv \
  --guest-util-csv /path/to/raw_guest_proc_stat.csv \
  --time-scale-mode rtl-cycle \
  --rtl-workload distilbert \
  --gemmini-frequency-mhz 200 \
  --cpu-time-scale 0.3035 \
  --out-prefix /path/to/time_scaled_rtl_200mhz_cpu_0p3035
```

single-app에서는 inference sample window 안의 CPU/Gemmini 구간만 변환하고,
그 밖의 collector 준비 구간은 원래 간격을 유지한다.

### Dual-app 독립 시간축 예시

```bash
python3 gemmini_experiments/power_modeling/power_sim/time_scale_gemmini_logs.py \
  --bench /path/to/raw_bench.log \
  --turbostat /path/to/raw_turbostat.log \
  --power-csv /path/to/raw_power.csv \
  --guest-util-csv /path/to/raw_guest_proc_stat.csv \
  --time-scale-mode rtl-cycle \
  --model-rtl-workload distilbert=distilbert \
  --model-rtl-workload distilgpt2=distilgpt2 \
  --model-core distilbert=3,9,11 \
  --model-core distilgpt2=2,10,12 \
  --gemmini-frequency-mhz 200 \
  --cpu-time-scale 0.3035 \
  --out-prefix /path/to/time_scaled_rtl_200mhz_cpu_0p3035
```

dual-app은 model prefix와 `--model-core`를 이용해 각 app의
`remote_launch -> sample_start`, inference CPU, Gemmini 구간을 독립적으로
변환한다. scaled power CSV는 두 app에서 변환된 모든 sample timestamp를 합치고
각 core 값을 보간한 뒤 component total과 `p_total_w`를 다시 계산한다.

## 출력

`--out-prefix PREFIX`를 기준으로 다음 파일을 생성한다.

```text
PREFIX_bench.log
PREFIX_turbostat.log
PREFIX_guest_util.csv  # --guest-util-csv 지정 시
PREFIX_power.csv
PREFIX_summary.json
```

`--result-prefix RESULT_PREFIX`를 지정하면 power CSV와 summary만
`RESULT_PREFIX_power.csv`, `RESULT_PREFIX_summary.json`에 저장한다.
Bench/turbostat/guest-util 중간 파일은 계속 `--out-prefix`를 따른다.
Single runner는 이를 사용해 results와 runs를 분리하며, dual의 기존 경로는 유지한다.

scaled CSV에는 원래 timestamp 확인용 `raw_sample_tod_s`를 추가한다.
`summary.json`의 주요 항목은 다음과 같다.

```text
time_scale_mode
cpu_time_scale
cpu_raw_time_s, cpu_scaled_time_s
gemmini_raw_time_s, gemmini_scaled_time_s
inference_raw_time_s, inference_scaled_time_s
energy_raw_j, energy_scaled_j
rtl_cycle_total, rtl_time_s                  # single rtl-cycle
model_rtl, model_time_scales, per_model_time # dual independent
sample_windows, cpu_intervals, gemmini_intervals
```

현재 single-app과 dual-app runner는 raw 계측이 끝난 뒤 이 유틸을 자동 호출하고
scaled CSV와 plot을 생성한다. 두 runner 모두 CPU time scale 기본값 `0.3035`,
Gemmini frequency 기본값 `200 MHz`를 전달한다.
