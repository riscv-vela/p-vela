# Applications

QEMU 전력 모델링 실험에 필요한 application의 모델·입력 데이터 준비와
Gemmini power 바이너리 빌드 방법을 안내한다. 모든 명령은 **host의 workspace root**에서
실행한다. 빌드할 때 QEMU를 부팅할 필요는 없다.

## 지원 Application

| Application | 용도 | Guest 소스·바이너리 위치 |
|---|---|---|
| DistilBERT | SST-2 텍스트 분류 | [distilbert/sst2_pipeline/guest](./distilbert/sst2_pipeline/guest/) |
| DistilRoBERTa | SST-2 텍스트 분류 | [distilroberta/sst2_pipeline/guest](./distilroberta/sst2_pipeline/guest/) |
| DistilGPT2 | SST-2 텍스트 분류 | [distilgpt2/sst2_pipeline/guest](./distilgpt2/sst2_pipeline/guest/) |
| PolyBench GEMM | 행렬 곱셈 benchmark | [polybench_gemm/guest](./polybench_gemm/guest/) |

## 0. 준비 환경

- `make`, `riscv64-linux-gnu-gcc` 및 RISC-V Linux용 static C 라이브러리
- `q-vela/qemu/test-code/gemmini_test_code/include/`의 Gemmini 헤더
- Transformer 모델 데이터를 생성할 경우 Python 3.10 이상과 `torch`, `numpy`,
  `transformers`, `datasets`, `peft`가 설치된 host Python 환경
- 모델·tokenizer·SST-2 데이터가 캐시에 없으면 다운로드 가능한 네트워크 환경

Ubuntu 계열 host에서 C 빌드 도구가 없다면 다음과 같이 설치한다.

```bash
sudo apt install make gcc-riscv64-linux-gnu libc6-dev-riscv64-cross
```

`q-vela`가 심볼릭 링크라면 실제 경로가 연결되어 있어야 한다.
Makefile은 위 Gemmini 헤더 경로를 기본으로 사용하며, 다른 위치를 사용할 때는
`GEMMINI_INCLUDE=/absolute/path/to/include`를 make 인자로 전달한다.

## 1. Transformer 모델·입력 데이터 준비

Transformer는 실행 바이너리 외에 `<app>/sst2_pipeline/guest/generated/`의 데이터가
필요하다. 기존 실험에 사용하는 데이터가 이미 준비되어 있으면 이 단계는 건너뛰고
바이너리 빌드부터 진행한다. PolyBench는 별도의 모델 데이터가 필요 없다.

처음 준비할 때는 아래 명령으로 세 application의 기존 데이터 생성 파이프라인을
순서대로 실행한다. 현재 `03_export_c_model.py`가 `00`, `01`, `02`의 산출물을
읽으므로 중간 단계를 생략하지 않는다.

```bash
(
    set -e
    for app in distilbert distilroberta distilgpt2; do
        pipeline="gemmini_experiments/applications/${app}/sst2_pipeline"
        python3 "${pipeline}/00_fp32_baseline.py" --allow-download
        python3 "${pipeline}/01_quantize_linear_int8.py" --allow-download
        python3 "${pipeline}/02_eval_host_int8.py" --allow-download
        python3 "${pipeline}/03_export_c_model.py" --allow-download
    done
)
```

한 application만 준비하려면 `for app in ...`의 목록에 해당 이름만 남긴다.
위 명령은 스크립트에 정의된 모델과 기본 입력 길이 128을 사용한다.
모델 다운로드와 host 연산이 포함되어 시간이 걸릴 수 있다.

중간 산출물은 `<app>/sst2_pipeline/outputs/`, guest용 데이터는 다음 위치에 생성된다.

```text
<app>/sst2_pipeline/guest/generated/
├── export_manifest.json
├── model_bin/
├── eval_bin/
└── debug_bin/
```

재실행하면 기존 산출물을 갱신하며, export 단계는 `model_bin/`, `eval_bin/`,
`debug_bin/`을 삭제 후 재생성한다. 보관할 데이터가 있으면 먼저 별도로 백업한다.
모델 구조나 입력 길이를 바꾸면 power 모델의 workload cycle profile도 함께 확인해야 한다.

## 2. 전체 Gemmini Power 바이너리 빌드

현재 power runner에서 사용하는 네 바이너리만 빌드한다.

```bash
(
    set -e
    for app in distilbert distilroberta distilgpt2; do
        make -B -C "gemmini_experiments/applications/${app}/sst2_pipeline/guest" \
            "${app}_gemmini_power_infer.riscv"
    done

    make -B -C gemmini_experiments/applications/polybench_gemm/guest \
        gemmini-power DATASET=MEDIUM_DATASET
)
```

`-B`는 기존 바이너리가 있어도 다시 빌드한다. 특히 PolyBench의 dataset 변경은
파일 수정 시간만으로 감지되지 않으므로 강제 빌드를 사용한다.
현재 Gemmini workload cycle profile에 맞춰 PolyBench는 `MEDIUM_DATASET`으로 빌드한다.

생성 파일은 다음과 같다.

```text
distilbert/sst2_pipeline/guest/distilbert_gemmini_power_infer.riscv
distilroberta/sst2_pipeline/guest/distilroberta_gemmini_power_infer.riscv
distilgpt2/sst2_pipeline/guest/distilgpt2_gemmini_power_infer.riscv
polybench_gemm/guest/polybench_gemm_gemmini_power.riscv
```

개별 빌드도 같은 target을 사용한다. 예를 들어 DistilBERT만 빌드하려면:

```bash
make -B -C gemmini_experiments/applications/distilbert/sst2_pipeline/guest \
    distilbert_gemmini_power_infer.riscv
```

## 3. Power 실험 실행

준비한 바이너리는 RISC-V guest에서 실행한다. QEMU 부팅과 계측·후처리는
아래 문서에 따라 진행한다.

- [QEMU 부팅 및 pinning](../../qemu_boot/README.md)
- [Power Modeling](../power_modeling/README.md)

## 부가 도구: Tiler

[gemmini_ws_tiler.c](./gemmini_tiler/gemmini_ws_tiler.c)와
[transformer_gemmini_tiler.c](./transformer_gemmini_tiler.c)는 **cycle 실험 분석을 위한
보조 도구**다. GEMM 크기의 tile 분해와 application별 Gemmini API·`LOOP_WS` 호출 구성을
분석하며, power 바이너리 빌드나 실험 실행의 필수 단계는 아니다.
