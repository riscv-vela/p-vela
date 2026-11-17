# PolyBench GEMM Power Flow

이 문서는 PolyBench GEMM power용 guest C 코드의 CPU-only/Gemmini backend
구조를 정리한다. 현재 single-app runner는 Gemmini backend를 실행한다.

## 목표

power 실험용 PolyBench GEMM은 CPU-only와 Gemmini가 같은 코드 흐름을 쓰고,
`A_int8 * B_int8 -> acc` backend만 바꾼다.

```text
common:
  C = beta * C + alpha * acc

CPU-only backend:
  acc = sum_k(A_int8 * B_int8)

Gemmini backend:
  acc = tiled_matmul_auto(A_int8, B_int8)
```

Gemmini 호출을 위한 fence와 executor thread는 Gemmini backend 안에 남겨둔다.
현재 power baseline은 기본 Gemmini backend에서 `GEMMINI_SAFE_MAX_I/J` 단위 chunking 경로를 사용한다.
direct `tiled_matmul_auto()` 경로는 power baseline에서 사용하지 않는다.

## 현재 실행 흐름

Workspace root에서 다음 명령으로 Gemmini power 실험을 실행한다.

```bash
bash gemmini_experiments/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power.sh \
  polybench_gemm --reps 50
```

Host collector와 guest runner가 계측한 뒤 raw/scaled 결과를 생성한다.
이전 CPU-only → Gemmini → compare 자동 실행 runner는 제거했다.
아래 CPU-only 설명은 backend 코드 비교를 위한 참고이며 현재 runner의 실행 단계는 아니다.

## CPU-only stage

실행 바이너리:

```text
polybench_gemm_cpu_power.riscv
```

코드 구조:

```text
polybench_gemm_cpu_power.c
  -> #define POLYBENCH_GEMM_POWER_ONLY 1
  -> #define POLYBENCH_GEMM_USE_GEMMINI_BACKEND 0
  -> #include "polybench_gemm_gemmini.c"

polybench_gemm_gemmini.c
  -> CPU용 power_backend_matmul() 선택
  -> polybench_gemm_power_common.h 의 공통 runner 호출
```

실행 로직:

```text
공통 runner:
  A_int8, B_int8, C 초기화
  for rep:
    gemm_begin
    power_backend_matmul()
      -> CPU loop로 acc = sum_k(A_int8 * B_int8)
    C = beta*C + alpha*acc
    gemm_end
  checksum, metrics 저장
```

## CPU Baseline 변경점

power용 CPU baseline은 PolyBench 원본 GEMM의 outer 수식은 유지하지만,
multiply input type을 Gemmini path와 맞추기 위해 바꿨다.

원본 PolyBench GEMM:

```text
C = beta * C + alpha * sum_k(A_float * B_float)
```

power용 CPU baseline:

```text
C = beta * C + alpha * sum_k(A_int8 * B_int8)
```

유지한 것:

```text
PolyBench dataset M/N/K
C 초기화식
alpha = 1.5
beta = 1.2
C = beta*C + alpha*A*B update 구조
rep 반복 구조
```

바꾼 것:

```text
A/B data type: float/double -> int8
A/B 생성: PolyBench 초기화값을 deterministic int8 값으로 변환
A*B backend: CPU int8 accumulation loop
```

이렇게 바꾼 이유는 Gemmini path가 float GEMM을 직접 수행하는 것이 아니라
int8 GEMM accumulation을 수행하기 때문이다. 즉 CPU baseline은 PolyBench GEMM의
수식 구조를 유지하되, Gemmini와 같은 int8 multiply input으로 맞춘 baseline이다.

## Gemmini stage

실행 바이너리:

```text
polybench_gemm_gemmini_power.riscv
```

코드 구조:

```text
polybench_gemm_gemmini_power.c
  -> #define POLYBENCH_GEMM_POWER_ONLY 1
  -> #define POLYBENCH_GEMM_USE_GEMMINI_BACKEND 1
  -> #include "polybench_gemm_gemmini.c"

polybench_gemm_gemmini.c
  -> Gemmini용 power_backend_matmul() 정의
  -> polybench_gemm_power_common.h 의 공통 runner 호출
```

실행 로직:

```text
공통 runner:
  A_int8, B_int8, C 초기화
  for rep:
    gemm_begin
    power_backend_matmul()
      -> run_gemmini_matmul()
      -> tiled_matmul_auto()
    C = beta*C + alpha*acc
    gemm_end
  checksum, metrics 저장
```

## 이전 Compare stage

구형 runner는 최신 CPU-only summary와 최신 Gemmini summary를 비교해
`cpu_vs_gemmini` report와 plot을 생성했다. 현재 runner에는 이 자동 비교 단계가 없다.

## 핵심 기준

power path에서 통일된 부분:

```text
입력 생성
C 초기화
reps loop
gemm_begin / gemm_end timeline
sample-start / sample-done log
C = beta*C + alpha*acc update
checksum / metrics / done log
```

교체되는 부분:

```text
A_int8 * B_int8 -> acc
```

즉 CPU-only와 Gemmini의 의도된 알고리즘 차이는 GEMM accumulation backend뿐이다.

## 동일성 범위

power 측정 control flow 기준으로 CPU-only와 Gemmini power 바이너리는 같은
`polybench_gemm_gemmini.c` + `polybench_gemm_power_common.h` runner를 사용한다.
wrapper macro만 다르다.

```text
CPU-only wrapper:
  POLYBENCH_GEMM_USE_GEMMINI_BACKEND=0

Gemmini wrapper:
  POLYBENCH_GEMM_USE_GEMMINI_BACKEND=1
```

따라서 아래 흐름은 동일하다.

```text
init
sample-start
for rep:
  gemm_begin
  power_backend_matmul()
  C = beta*C + alpha*acc
  gemm_end
sample-done
checksum / metrics / done
```

교체되는 함수는 `power_backend_matmul()` 하나다.

```text
CPU-only:
  power_backend_matmul()
    -> CPU loop

Gemmini:
  power_backend_matmul()
    -> run_gemmini_matmul()
    -> tiled_matmul_auto()
```

단, 프로세스 전체가 byte-for-byte로 같은 것은 아니다. Gemmini backend 내부에는
Gemmini 호출을 위해 필요한 구현 detail이 남아 있다.

```text
gemmini_flush / gemmini_fence
zero_bias
executor thread submit/wait
```

기본 Gemmini power 구현이 chunking 경로를 사용한다.

따라서 정확한 표현은 다음과 같다.

```text
Power experiment control flow is identical.
The only intended algorithmic substitution is the A_int8 * B_int8 -> acc backend:
CPU loop vs Gemmini tiled_matmul_auto path.
```
