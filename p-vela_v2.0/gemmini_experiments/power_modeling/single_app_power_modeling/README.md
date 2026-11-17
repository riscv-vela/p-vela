# Single-app Power Modeling

## 0. 실행 전 준비

| 준비 항목 | 기준 |
|---|---|
| Application | [Applications](../../applications/README.md)에 따라 Gemmini power 바이너리와 모델·입력 데이터 준비 |
| QEMU | [QEMU Boot](../../../qemu_boot/README.md)에 따라 `1 3`으로 부팅하고 `pin.sh 8 2` 적용 |
| Host | `sudo turbostat`, Python 3 및 `matplotlib` 사용 가능 |
| Guest | Python 3, `timedatectl`, 비밀번호 입력 없이 필요한 명령을 실행할 수 있는 `sudo` 환경 |
| SSH | 기본 `ubuntu@127.0.0.1:2200`, 키는 `q-vela/tmp-cloudinit/id_ed25519` |
| 공유 폴더 | Host `gemmini_experiments/`가 guest `/mnt/hostshare`에 mount되어 있고 결과 기록 가능 |

Single runner는 이미 준비된 바이너리를 사용하며 자동 빌드하지 않는다.
PolyBench는 `MEDIUM_DATASET` 바이너리를 사용한다.
현재 application thread 수는 `--threads 1`만 지원한다.

## 1. 실행 방법

모든 명령은 **host의 workspace root**에서 실행하며, QEMU와 별도 터미널을 사용한다.
진입 스크립트는 [run_single_app_power.sh](./experiment_scripts/run_single_app_power.sh)다.

### 사용법과 인자

```text
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh APP [옵션]
```

첫 번째 인자 `APP`은 `distilbert`, `distilroberta`, `distilgpt2`, `polybench_gemm` 중
하나를 지정한다. `APP` 대신 `--app APP` 형식으로 지정해도 된다.

| 옵션 | 기본값 | 용도 |
|---|---:|---|
| `--sample-limit N` | `1` | Transformer 입력 sample 수 |
| `--reps N` | `50` | PolyBench GEMM 반복 수 |
| `--threads N` | `1` | Application thread 수, 현재 `1`만 지원 |
| `--run-id ID` | 자동 생성 | 결과·로그를 식별하는 실행 ID. 기존 ID와 중복되지 않아야 함 |
| `--fixed-freq-ghz FLOAT` | `2.512` | CPU 전력 모델에 사용하는 주파수 |
| `--sys-scale FLOAT` | `0.5625` | CPU 전력 모델 보정 계수 |
| `--gemmini-full-util-power-w FLOAT` | `0.862` | Gemmini 활성 상태 전력(W) |
| `--gemmini-util-threshold-pct FLOAT` | `80` | Gemmini 활성 판정 Busy% |
| `--time-scale-mode rtl-cycle` | `rtl-cycle` | 현재 지원하는 시간 보정 모드 |
| `--gemmini-frequency-mhz FLOAT` | `200` | RTL cycle을 시간으로 변환할 주파수 |
| `--cpu-time-scale FLOAT` | `0.3035` | CPU 구간 시간 배율 |

전력·시간 모델의 공통 기본값은
[power_model_defaults.env](../power_sim/power_model_defaults.env)에서 관리한다.
명령행 옵션은 해당 실행에서 기본값보다 우선하며, 설정 변경은 새 실험부터 적용된다.

코어 배치를 바꾸는 경우에만 아래 옵션을 사용한다. QEMU pinning과 일치하도록 설정해야 한다.

| 옵션 | 기본값 | 용도 |
|---|---:|---|
| `--main-vcpu N` | `1` | Application main guest vCPU |
| `--housekeeping-vcpu N` | `0` | Housekeeping guest vCPU |
| `--executor-vcpu N` | `2` | Gemmini executor guest vCPU |
| `--vcpu-base-core N` | `8` | Guest vCPU 0에 대응하는 host CPU 번호 |
| `--gemmini-host-core N` | `2` | Gemmini worker의 host CPU 번호 |

Host 제어 프로세스의 CPU는 `HOST_HOUSEKEEPING_CORE` 환경변수로 지정하며 기본값은 8이다.

### 실행 예시

DistilBERT sample 1개 실행:

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    distilbert --sample-limit 1
```

`distilbert`를 `distilroberta` 또는 `distilgpt2`로 바꾸면 같은 방식으로 실행한다.

PolyBench GEMM 50회 실행:

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    polybench_gemm --reps 50
```

모델 설정을 명시해서 실행:

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    distilbert --sample-limit 1 \
    --gemmini-util-threshold-pct 80 \
    --gemmini-frequency-mhz 200 \
    --cpu-time-scale 0.3035
```

Host의 `sudo` 비밀번호를 요청할 수 있다. Runner가 host 계측과 guest 실행을 함께
관리하므로 guest에서 별도 실행 명령을 입력할 필요는 없다.
종료 시 출력되는 `model results:` 경로에서 CSV·그래프·summary를 확인한다.

## 2. 결과 확인

아래 경로는 `gemmini_experiments/power_modeling/single_app_power_modeling/` 기준이다.

### 기본 결과: `raw_power.csv`

가장 먼저 확인할 파일은 `results/<run-id>/raw_power.csv`다.
CPU/Gemmini utilization과 모델링된 전력, total power를 하나의 CSV에 저장하며,
실험 중 기본 100 ms 주기로 갱신된다. 실시간 전력 모니터링은 이 파일을 사용한다.

실행 중 별도 host 터미널에서 확인하려면 실제 run ID로 바꿔 실행한다.

```bash
tail -f "gemmini_experiments/power_modeling/single_app_power_modeling/results/<run-id>/raw_power.csv"
```

### 자동 생성 그래프와 시간 후보정 결과

스크립트는 실험 종료 후 **raw 그래프와 시간 후보정된 CSV·그래프·summary까지 자동으로
생성**한다. 별도로 그래프 스크립트를 실행할 필요는 없으며, 모두 `results/<run-id>/`에 저장된다.

현재 기본 설정에서는 CPU 구간 시간을 **0.3035배**, Gemmini 구간 시간을
**RTL cycle 모델과 200 MHz 주파수 기준**으로 후보정한다. 후보정된 그래프와
`time_scaled_<tag>_summary.json`에서 예상 실행 시간과 에너지를 확인하여 성능을
평가할 수 있다. 이는 실험 종료 후 산출하는 모델 기반 추정 결과이며 실시간 성능 계측과는 다르다.

**실시간 성능 계측이 필요한 경우에는 용역보고서의 사용 설명서를 참고하여
해당 계측 도구를 별도로 설치한다.**

| 파일 | 내용 |
|---|---|
| `raw_util_power.png` | Raw CPU/Gemmini utilization 및 CPU/Gemmini/total power |
| `raw_host_gemmini_util.png` | Host Gemmini worker의 threshold 전 Busy% |
| `raw_guest_vcpu_util.png` | Guest application main utilization |
| `raw_guest_executor_vcpu_util.png` | Guest Gemmini executor utilization |
| `raw_guest_housekeeping_util.png` | Guest housekeeping utilization |
| `time_scaled_<tag>_power.csv` | 시간 보정된 power CSV |
| `time_scaled_<tag>_summary.json` | Raw/scaled 시간·에너지와 cycle 모델 정보 |
| `time_scaled_<tag>_util_power.png` | 시간 보정된 utilization 및 CPU/Gemmini/total power |

기본 설정의 `<tag>`는 `rtl_200mhz_cpu_0p3035`다. 그래프는 raw 5장과 scaled 1장,
총 6장이다.

### 실행 기록: `runs/<run-id>/`

실행 설정과 오류 확인용 로그를 저장한다.

| 파일 | 내용 |
|---|---|
| `request.env` | 해당 실행의 설정 |
| `raw_bench.log` | Application 실행 로그 |
| `raw_turbostat.log` | Host 원본 utilization 로그 |
| `raw_guest_proc_stat.csv` | Guest vCPU 상세 계측 |
| `raw_power_sim.log` | 전력 수집 로그 |
| `app_output/` | Application 실행 산출물 |

`runs`와 `results`는 같은 run ID로 연결된다. 실행 중에는 해당 폴더의 파일을 삭제하지 않는다.
같은 run ID를 재사용하면 기존 폴더와 충돌하여 실행이 중단된다.

실패 시 터미널 오류와 같은 run ID의 `raw_bench.log`, `raw_power_sim.log`,
`raw_turbostat.log`를 확인한다.
