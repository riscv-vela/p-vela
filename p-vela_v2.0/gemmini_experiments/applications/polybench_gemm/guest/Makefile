HOST_CC ?= gcc
RISCV_CC ?= riscv64-linux-gnu-gcc

COMMON_CFLAGS ?= -O2 -Wall -Wextra -std=gnu11
HOST_CFLAGS ?= $(COMMON_CFLAGS)
RISCV_CFLAGS ?= $(COMMON_CFLAGS) -static
GEMMINI_WARNING_CFLAGS ?= -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable
LDFLAGS ?= -lm
GEMMINI_LDFLAGS ?= -pthread

GEMMINI_INCLUDE ?= ../../../../q-vela/qemu/test-code/gemmini_test_code/include
GEMMINI_CFLAGS ?= -isystem $(GEMMINI_INCLUDE)
DATASET ?= SMALL_DATASET
POLYBENCH_CFLAGS ?= -D$(DATASET) -DPOLYBENCH_TIME -Ipolybench

HOST_ORIGINAL_TARGET := polybench_gemm_original.native
RISCV_ORIGINAL_TARGET := polybench_gemm_original.riscv
RISCV_CPU_POWER_TARGET := polybench_gemm_cpu_power.riscv
RISCV_GEMMINI_TARGET := polybench_gemm_gemmini.riscv
RISCV_GEMMINI_POWER_TARGET := polybench_gemm_gemmini_power.riscv
OBSOLETE_TARGETS := polybench_gemm_cpu_chunked_power.riscv polybench_gemm_gemmini_direct.riscv polybench_gemm_gemmini_direct_power.riscv polybench_gemm_gemmini_chunked_legacy.riscv polybench_gemm_gemmini_chunked_legacy_power.riscv

ORIGINAL_SRCS := polybench/gemm.c polybench/polybench.c
CPU_POWER_SRCS := polybench_gemm_cpu_power.c polybench_gemm_gemmini.c
GEMMINI_SRCS := polybench_gemm_gemmini.c
GEMMINI_POWER_SRCS := polybench_gemm_gemmini_power.c polybench_gemm_gemmini.c

all: original-riscv

original-native: $(HOST_ORIGINAL_TARGET)

original-riscv: $(RISCV_ORIGINAL_TARGET)

cpu-power: $(RISCV_CPU_POWER_TARGET)

gemmini: $(RISCV_GEMMINI_TARGET)

gemmini-power: $(RISCV_GEMMINI_POWER_TARGET)

$(HOST_ORIGINAL_TARGET): $(ORIGINAL_SRCS) polybench/gemm.h polybench/polybench.h
	$(HOST_CC) $(HOST_CFLAGS) $(POLYBENCH_CFLAGS) -o $@ $(ORIGINAL_SRCS) $(LDFLAGS)

$(RISCV_ORIGINAL_TARGET): $(ORIGINAL_SRCS) polybench/gemm.h polybench/polybench.h
	$(RISCV_CC) $(RISCV_CFLAGS) $(POLYBENCH_CFLAGS) -o $@ $(ORIGINAL_SRCS) $(LDFLAGS)

$(RISCV_CPU_POWER_TARGET): $(CPU_POWER_SRCS) polybench_gemm_common.h polybench_gemm_power_common.h polybench/gemm.h polybench/polybench.h
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_WARNING_CFLAGS) $(POLYBENCH_CFLAGS) $(GEMMINI_CFLAGS) -o $@ polybench_gemm_cpu_power.c $(LDFLAGS) $(GEMMINI_LDFLAGS)

$(RISCV_GEMMINI_TARGET): $(GEMMINI_SRCS) polybench_gemm_common.h polybench_gemm_power_common.h polybench/gemm.h
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_WARNING_CFLAGS) $(POLYBENCH_CFLAGS) $(GEMMINI_CFLAGS) -o $@ $(GEMMINI_SRCS) $(LDFLAGS) $(GEMMINI_LDFLAGS)

$(RISCV_GEMMINI_POWER_TARGET): $(GEMMINI_POWER_SRCS) polybench_gemm_common.h polybench_gemm_power_common.h polybench/gemm.h
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_WARNING_CFLAGS) $(POLYBENCH_CFLAGS) $(GEMMINI_CFLAGS) -o $@ polybench_gemm_gemmini_power.c $(LDFLAGS) $(GEMMINI_LDFLAGS)

clean:
	rm -f $(HOST_ORIGINAL_TARGET) $(RISCV_ORIGINAL_TARGET) $(RISCV_CPU_POWER_TARGET) $(RISCV_GEMMINI_TARGET) $(RISCV_GEMMINI_POWER_TARGET) $(OBSOLETE_TARGETS)

.PHONY: all original-native original-riscv cpu-power gemmini gemmini-power clean
