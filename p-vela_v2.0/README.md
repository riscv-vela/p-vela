# QEMU 환경에서의 실시간 성능 및 전력 모델링

## 프로젝트 개요

본 프로젝트는 “SDR의 개방형 생태계 구축을 위한 RISC-V 기반 시스템 소프트웨어 기술개발”
과제의 시스템 SW를 정량적으로 평가하기 위해, QEMU 에뮬레이션 환경에서 RISC-V CPU와
AI 실행 엔진(Gemmini)을 포함한 시스템의 성능·전력 계측 모델을 개발한다. 기존 CPU 전력
모델을 AI 실행 엔진까지 확장하고, 애플리케이션 실행 시간 추정과 실시간 소모 전력
모니터링을 지원하여 시스템 SW의 성능 및 전력 개선 효과를 평가하는 것을 목표로 한다.

## 문서 안내

| 확인할 내용 | 문서 |
|---|---|
| QEMU 부팅, 공유 폴더, SSH, thread pinning | [QEMU Boot](./qemu_boot/README.md) |
| Application 구성 및 준비 | [Applications](./gemmini_experiments/applications/README.md) |
| 전력 모델 설정, 실험 실행 및 결과 확인 | [Power Modeling](./gemmini_experiments/power_modeling/README.md) |
| 전력 계산식, RTL cycle 모델, 시간 보정 | [Power Simulation](./gemmini_experiments/power_modeling/power_sim/README.md) |

## 지원 Application

| Application | 용도 |
|---|---|
| [DistilBERT](./gemmini_experiments/applications/distilbert/) | SST-2 텍스트 분류 |
| [DistilRoBERTa](./gemmini_experiments/applications/distilroberta/) | SST-2 텍스트 분류 |
| [DistilGPT2](./gemmini_experiments/applications/distilgpt2/) | SST-2 텍스트 분류, WikiText 언어 모델링 |
| [PolyBench GEMM](./gemmini_experiments/applications/polybench_gemm/) | 행렬 곱셈 benchmark |

현재 workspace의 DistilGPT2 실행 경로는 SST-2 기준이며, WikiText 파이프라인은 포함되어 있지 않다.

## 실험 진행 순서

0. [Applications](./gemmini_experiments/applications/)에서 실행할 application과 필요한 모델·입력 데이터를 준비한다.
1. [QEMU Boot](./qemu_boot/)의 안내에 따라 QEMU를 부팅한다.
2. [QEMU Boot](./qemu_boot/)의 pinning 안내에 따라 QEMU thread를 host CPU에 배치한다.
3. [Power Modeling](./gemmini_experiments/power_modeling/README.md)의 안내에 따라 실험을 실행하고 결과를 확인한다.

상세 준비 과정, 실행 명령과 결과 설명은 각 폴더의 README를 참고한다.

### Quick Start: Single Application 예시

#### 0. 사전 준비

DistilBERT 모델·입력 데이터와 power 바이너리, QEMU 실행 파일·guest 이미지,
SSH·공유 폴더 사용 환경을 준비한다. 아래 명령은 모두 **host의 workspace root**에서 실행한다.

#### 1. QEMU 부팅 — host 터미널 1

```bash
bash qemu_boot/run_qemu.sh 1 3
```

#### 2. Pinning — guest 부팅 완료 후 host 터미널 2

```bash
bash qemu_boot/pin.sh 8 2
```

#### 3. DistilBERT 실행 — host 터미널 2

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
    distilbert --sample-limit 1
```

종료 시 출력되는 `model results:` 경로에서 `raw_power.csv`와 자동 생성된 그래프·summary를
확인한다. 상세 옵션과 결과 설명은 [Power Modeling](./gemmini_experiments/power_modeling/README.md)을 참고한다.
