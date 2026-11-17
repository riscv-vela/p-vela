# Dual-app Power Modeling

## 0. 실행 전 준비

| 준비 항목 | 기준 |
|---|---|
| Application | [Applications](../../applications/README.md)에 따라 사용할 두 application의 모델·입력 데이터와 빌드 환경 준비 |
| QEMU | [QEMU Boot](../../../qemu_boot/README.md)에 따라 `2 Gemmini / 5 vCPU`, `modulo` policy로 부팅 |
| Pinning | Host에서 사용 가능한 CPU를 선택해 QEMU thread pinning 완료 |
| Host | `sudo turbostat`, Python 3 및 `matplotlib` 사용 가능 |
| Guest | Python 3, `timedatectl`, 비밀번호 입력 없이 필요한 명령을 실행할 수 있는 `sudo` 환경 |
| SSH | 기본 `ubuntu@127.0.0.1:2200`, 키는 `q-vela/tmp-cloudinit/id_ed25519` |
| 공유 폴더 | Host `gemmini_experiments/`가 guest `/mnt/hostshare`에 mount되어 있고 결과 기록 가능 |

Dual runner는 기본적으로 실행 전에 두 application의 바이너리를 빌드한다.
모델·입력 데이터는 자동 생성하지 않으므로 미리 준비해야 한다.
이미 빌드된 바이너리를 사용하려면 `--build 0`을 지정한다.

기본 pinning 예시인 `pin.sh 8 2`는 host CPU 8~12와 2~3을 사용한다.
CPU가 0~11까지만 있는 host에서는 QEMU 부팅 후 아래처럼 시작 번호를 낮춘다.
각 thread의 `[OK]`를 확인해야 하며, runner는 기본적으로 실제 pinning 결과를 읽어 사용한다.

```bash
bash qemu_boot/pin.sh 7 2
```

## 1. 실행 방법

모든 명령은 **host의 workspace root**에서 실행하며, QEMU와 별도 터미널을 사용한다.
실행 스크립트는 [run_dual_app_power_exp.sh](./experiment_scripts/run_dual_app_power_exp.sh)다.

### 사용법과 인자

```text
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh --apps APP0,APP1 [옵션]
```

`APP0`, `APP1`은 `distilbert`, `distilroberta`, `distilgpt2`, `polybench_gemm` 중
서로 다른 두 개를 선택한다. `--app0 APP0 --app1 APP1` 형식도 사용할 수 있다.

| 옵션 | 기본값 | 용도 |
|---|---:|---|
| `--apps APP0,APP1` | 필수 | 동시에 실행할 두 application |
| `--sample-limit N` | `1` | 각 Transformer의 입력 sample 수 |
| `--threads N` | `1` | Application별 thread 수, 현재 `1`만 지원 |
| `--dataset MEDIUM` | `MEDIUM` | PolyBench dataset, 현재 시간 후보정은 `MEDIUM`만 지원 |
| `--reps N` | `50` | PolyBench GEMM 반복 수 |
| `--build 0\|1` | `1` | `1`: 실행 전 빌드, `0`: 기존 바이너리 사용 |
| `--sys-scale FLOAT` | `0.5625` | CPU 전력 모델 보정 계수 |
| `--gemmini-full-util-power-w FLOAT` | `0.862` | Gemmini 한 개의 활성 상태 전력(W) |
| `--gemmini-util-threshold-pct FLOAT` | `80` | Gemmini 활성 판정 Busy% |

전력·시간 모델의 공통 기본값은
[power_model_defaults.env](../power_sim/power_model_defaults.env)에서 관리한다.
명령행 옵션을 지정하면 해당 실행에 우선 적용되며, 설정 변경은 새 실험부터 적용된다.

주파수와 시간 배율은 아래 환경변수로 변경한다. Single runner와 달리
`--gemmini-frequency-mhz`, `--cpu-time-scale` 명령행 옵션은 사용하지 않는다.

| 환경변수 | 기본값 | 용도 |
|---|---:|---|
| `GEMMINI_FREQUENCY_MHZ` | `200` | Gemmini 시간 후보정 기준 주파수(MHz) |
| `CPU_TIME_SCALE` | `0.3035` | CPU 구간 시간 배율 |
| `FIXED_FREQ_GHZ` | `2.512` | CPU 전력 모델에 사용하는 주파수(GHz) |

### 실행 예시

DistilBERT와 DistilRoBERTa를 각각 sample 1개씩 실행:

```bash
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh \
    --apps distilbert,distilroberta --sample-limit 1 --threads 1
```

DistilGPT2 sample 1개와 PolyBench GEMM 50회 실행:

```bash
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh \
    --apps distilgpt2,polybench_gemm --sample-limit 1 --threads 1 \
    --dataset MEDIUM --reps 50
```

시간 후보정 설정을 명시하고 기존 바이너리로 실행:

```bash
GEMMINI_FREQUENCY_MHZ=200 CPU_TIME_SCALE=0.3035 \
bash gemmini_experiments/power_modeling/dual_app_power_modeling/experiment_scripts/run_dual_app_power_exp.sh \
    --apps distilbert,distilroberta --sample-limit 1 --build 0
```

Host의 `sudo` 비밀번호를 요청할 수 있다. Guest에서 별도 실행 명령을 입력할 필요는 없다.
종료 시 출력되는 `Run dir` 경로에서 결과를 확인한다.

## 2. 결과 확인

결과와 실행 로그는 아래 폴더에 함께 저장한다. 실행마다 하위 폴더가 자동 생성되며,
이 로그 디렉터리는 Git 추적 대상에서 제외한다.

```text
gemmini_experiments/power_modeling/dual_app_power_modeling/logs/<app0>_<app1>/<run-id>/
```

### 기본 결과: Power CSV

현재 dual 실험은 single과 달리 **`raw_power.csv` 하나로 저장하지 않는다.**
실시간 전력 확인에는 다음 두 파일을 사용한다. 기본 기록 주기는 100 ms다.

| 파일 | 내용 |
|---|---|
| `vcpu_power.csv` | 두 application의 main vCPU utilization과 모델링된 CPU 전력 |
| `gemmini_power.csv` | 두 Gemmini worker의 utilization과 모델링된 Gemmini 전력 |

실행 중 별도 host 터미널에서 확인하려면 아래 경로를 실제 `Run dir`로 바꿔 실행한다.

```bash
tail -f \
    "gemmini_experiments/power_modeling/dual_app_power_modeling/logs/<app0>_<app1>/<run-id>/vcpu_power.csv" \
    "gemmini_experiments/power_modeling/dual_app_power_modeling/logs/<app0>_<app1>/<run-id>/gemmini_power.csv"
```

각 CSV의 total은 해당 component의 합이다. CPU와 Gemmini를 합친 전체 전력은
자동 생성되는 통합 그래프와 시간 후보정된 power CSV에서 확인한다.

### 자동 생성 그래프와 시간 후보정 결과

스크립트는 실험 종료 후 raw 그래프와 시간 후보정된 CSV·그래프·summary를 자동 생성한다.
별도로 그래프 스크립트를 실행할 필요는 없다.

현재 기본 설정에서는 application별로 CPU 구간 시간을 **0.3035배**, Gemmini 구간 시간을
**RTL cycle 모델과 200 MHz 주파수 기준**으로 후보정한다. 후보정된 그래프와
`time_scaled_<tag>_summary.json`에서 application별 예상 실행 시간과 에너지를 확인하여
성능을 평가할 수 있다. 이는 실험 종료 후 산출하는 모델 기반 추정 결과다.

실시간 성능 계측이 필요한 경우에는 **용역보고서의 사용 설명서를 참고하여
해당 계측 도구를 별도로 설치한다.**

| 파일 | 내용 |
|---|---|
| `raw_combined_power.png` | 두 application의 raw 통합 utilization·power 그래프 |
| `raw_combined_power_<app>.png` | Application별 raw utilization·power 그래프 |
| `time_scaled_<tag>_power.csv` | 두 application의 시간 후보정된 power CSV와 total power |
| `time_scaled_<tag>_summary.json` | Application별 raw/scaled 시간 및 에너지 결과 |
| `time_scaled_<tag>_combined_power.png` | 시간 후보정된 통합 utilization·power 그래프 |
| `time_scaled_<tag>_combined_power_<app>.png` | Application별 시간 후보정된 그래프 |

`<tag>`의 기본값은 `rtl_200mhz_cpu_0p3035`다.
각 통합 그래프와 같은 파일명 접두사에 아래 suffix가 붙은 utilization 그래프도 생성된다.

| Suffix | 내용 |
|---|---|
| `_guest_vcpu_util.png`, `_host_vcpu_util.png` | Application main의 guest/host utilization |
| `_guest_executor_vcpu_util.png`, `_host_executor_vcpu_util.png` | Gemmini executor의 guest/host utilization |
| `_guest_housekeeping_util.png`, `_host_housekeeping_util.png` | Housekeeping의 guest/host utilization |

그래프는 raw 9장과 scaled 9장, 총 18장이다.

### 실행 로그

| 파일 | 내용 |
|---|---|
| `raw_bench.log` | 실행 설정과 application 실행 로그 |
| `raw_turbostat.log` | Host 원본 utilization 로그 |
| `raw_guest_proc_stat.csv` | Guest vCPU 상세 계측 |
| `raw_gemmini_power_sim.log` | Gemmini 전력 수집 로그 |
| `qemu_thread_affinity_summary.csv` | 실행 시 QEMU thread의 CPU 배치 |
| `qemu_gemmini_mapping_summary.txt` | Application과 Gemmini 연결 상태 |

실패 시 터미널 오류와 해당 실행 폴더의 로그를 확인한다.
실행 중에는 결과 폴더의 파일을 삭제하지 않는다.
