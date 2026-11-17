RISCV_CC ?= riscv64-linux-gnu-gcc
HOST_CC ?= gcc
GEMMINI_INCLUDE ?= ../../../../../q-vela/qemu/test-code/gemmini_test_code/include
GEMMINI_CFLAGS ?= -isystem $(GEMMINI_INCLUDE)
GEMMINI_CHUNK_CFLAGS ?=

COMMON_CFLAGS ?= -O2 -Wall -Wextra -std=gnu11 -pthread
RISCV_CFLAGS ?= $(COMMON_CFLAGS) -static
HOST_CFLAGS ?= $(COMMON_CFLAGS)
LDFLAGS ?= -lm -pthread

RISCV_TARGET := distilbert_int8_infer.riscv
HOST_TARGET := distilbert_int8_infer.native
RISCV_GEMMINI_TARGET := distilbert_gemmini_infer.riscv
RISCV_GEMMINI_DIRECT_TARGET := distilbert_gemmini_direct_infer.riscv
RISCV_GEMMINI_POWER_TARGET := distilbert_gemmini_power_infer.riscv
RISCV_GEMMINI_DIRECT_POWER_TARGET := distilbert_gemmini_direct_power_infer.riscv
RISCV_LAYER_GEMM_ACCURACY_TARGET := distilbert_layer_gemm_accuracy.riscv
RISCV_CPU_POWER_TARGET := distilbert_cpu_power_infer.riscv

all: $(RISCV_TARGET) $(RISCV_GEMMINI_TARGET) $(RISCV_GEMMINI_POWER_TARGET) $(RISCV_CPU_POWER_TARGET)

native: $(HOST_TARGET)

gemmini: $(RISCV_GEMMINI_TARGET)

gemmini-direct: $(RISCV_GEMMINI_DIRECT_TARGET)

gemmini-v2: gemmini

power: $(RISCV_GEMMINI_POWER_TARGET) $(RISCV_CPU_POWER_TARGET)

gemmini-direct-power: $(RISCV_GEMMINI_DIRECT_POWER_TARGET)

layer-gemm-accuracy: $(RISCV_LAYER_GEMM_ACCURACY_TARGET)

$(RISCV_TARGET): distilbert_int8_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) -o $@ $< $(LDFLAGS)

$(HOST_TARGET): distilbert_int8_infer.c
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_GEMMINI_TARGET): distilbert_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_CHUNK_CFLAGS) $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_GEMMINI_DIRECT_TARGET): distilbert_gemmini_direct_infer.c distilbert_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_CHUNK_CFLAGS) $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_GEMMINI_POWER_TARGET): distilbert_gemmini_power_infer.c distilbert_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_CHUNK_CFLAGS) -Wno-unused-function $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_GEMMINI_DIRECT_POWER_TARGET): distilbert_gemmini_direct_power_infer.c distilbert_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_CHUNK_CFLAGS) -Wno-unused-function $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_LAYER_GEMM_ACCURACY_TARGET): distilbert_layer_gemm_accuracy.c distilbert_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) -Wno-unused-function $(GEMMINI_CHUNK_CFLAGS) $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_CPU_POWER_TARGET): distilbert_cpu_power_infer.c distilbert_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) -Wno-unused-function $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(RISCV_TARGET) $(HOST_TARGET) $(RISCV_GEMMINI_TARGET) $(RISCV_GEMMINI_DIRECT_TARGET) $(RISCV_GEMMINI_POWER_TARGET) $(RISCV_GEMMINI_DIRECT_POWER_TARGET) $(RISCV_LAYER_GEMM_ACCURACY_TARGET) $(RISCV_CPU_POWER_TARGET)

.PHONY: all native gemmini gemmini-direct gemmini-v2 power gemmini-direct-power layer-gemm-accuracy clean
