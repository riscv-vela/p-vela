# Power Modeling

QEMU에서 application을 실행하며 CPU와 Gemmini의 전력을 모니터링하고,
실험 종료 후 그래프와 시간 후보정 결과를 확인한다.
이 문서는 single application 실험을 기준으로 준비·설정·실행 순서를 안내한다.

## 문서 안내

| 확인할 내용 | 문서 |
|---|---|
| 전력 모델 및 시간 후보정 설정 | [Power Simulation](./power_sim/README.md) |
| Application 실행 명령·인자·결과 확인 | [Single-app Power Modeling](./single_app_power_modeling/README.md) |

## 0. 실험 준비

1. [Applications](../applications/README.md)에서 실행할 application의 모델·입력 데이터와
   Gemmini power 바이너리를 준비한다.
2. [QEMU Boot](../../qemu_boot/README.md)에 따라 QEMU를 `1 Gemmini / 3 vCPU`로 부팅하고
   CPU pinning, SSH 및 공유 폴더 연결을 확인한다.

지원 application은 `distilbert`, `distilroberta`, `distilgpt2`, `polybench_gemm`이다.

## 1. Power Simulation 설정

전력·시간 모델의 공통 기본값은
[power_sim/power_model_defaults.env](./power_sim/power_model_defaults.env)에서 수정한다.

- CPU 주파수와 전력 보정 계수
- Gemmini 활성·유휴 전력과 utilization threshold
- 계측 주기
- Gemmini 시간 후보정 기준 주파수와 CPU 시간 배율
- 그래프 표시 설정

실험 스크립트의 옵션으로 지정한 값이 기본값보다 우선하며, 설정 변경은 새 실험부터
적용된다. 옵션별 사용법은 single-app README, 공통 설정 설명은
[power_sim README](./power_sim/README.md)를 참고한다.
일반 실험에서는 power_sim의 Python 파일을 따로 실행할 필요가 없다.

## 2. Single Application 실행 및 결과 확인

[Single-app README](./single_app_power_modeling/README.md)의 실행 방법에 따라
host에서 `run_single_app_power.sh`를 실행한다.
이 스크립트가 계측부터 그래프·시간 후보정 결과 생성까지 수행한다.

가장 먼저 확인할 결과는 **`raw_power.csv`**다. CPU/Gemmini utilization과
모델링된 전력, total power가 실험 중 하나의 파일에 기록된다.

| 확인 목적 | 결과 |
|---|---|
| 실시간 전력 모니터링 | `raw_power.csv` |
| 실험 종료 후 전력 변화 확인 | 자동 생성된 raw 그래프 |
| 예상 실행 시간·에너지 확인 | 시간 후보정된 CSV·그래프·summary |

모델 결과는 `single_app_power_modeling/results/<run-id>/`, 실행 로그는
`single_app_power_modeling/runs/<run-id>/`에 저장한다.
파일별 설명과 확인 명령은 single-app README를 참고한다.

시간 후보정 결과는 실험 종료 후 확인하는 성능 추정값이다. 실시간 성능 계측이 필요한
경우에는 용역보고서의 사용 설명서에 따라 해당 계측 도구를 별도로 설치한다.

## 추가 지원: Dual Application

서로 다른 두 application을 동시에 실행하는 실험도 지원한다.
QEMU 준비 구성과 실행 명령, 결과 파일 구성이 single-app과 다르므로
[Dual-app README](./dual_app_power_modeling/README.md)를 참고한다.
