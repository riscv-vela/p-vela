RISCV_CC ?= riscv64-linux-gnu-gcc
HOST_CC ?= gcc
GEMMINI_INCLUDE ?= ../../../../../q-vela/qemu/test-code/gemmini_test_code/include
GEMMINI_CFLAGS ?= -isystem $(GEMMINI_INCLUDE)
GEMMINI_CHUNK_CFLAGS ?=

COMMON_CFLAGS ?= -O2 -Wall -Wextra -std=gnu11 -pthread
RISCV_CFLAGS ?= $(COMMON_CFLAGS) -static
HOST_CFLAGS ?= $(COMMON_CFLAGS)
LDFLAGS ?= -lm -pthread

RISCV_TARGET := distilroberta_int8_infer.riscv
HOST_TARGET := distilroberta_int8_infer.native
RISCV_GEMMINI_TARGET := distilroberta_gemmini_infer.riscv
RISCV_GEMMINI_POWER_TARGET := distilroberta_gemmini_power_infer.riscv
RISCV_CPU_POWER_TARGET := distilroberta_cpu_power_infer.riscv

all: $(RISCV_TARGET) $(RISCV_GEMMINI_TARGET) $(RISCV_GEMMINI_POWER_TARGET) $(RISCV_CPU_POWER_TARGET)

native: $(HOST_TARGET)

gemmini: $(RISCV_GEMMINI_TARGET)

gemmini-v2: gemmini

power: $(RISCV_GEMMINI_POWER_TARGET) $(RISCV_CPU_POWER_TARGET)

$(RISCV_TARGET): distilroberta_int8_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) -o $@ $< $(LDFLAGS)

$(HOST_TARGET): distilroberta_int8_infer.c
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_GEMMINI_TARGET): distilroberta_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_CHUNK_CFLAGS) $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_GEMMINI_POWER_TARGET): distilroberta_gemmini_power_infer.c distilroberta_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) $(GEMMINI_CHUNK_CFLAGS) -Wno-unused-function $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

$(RISCV_CPU_POWER_TARGET): distilroberta_cpu_power_infer.c distilroberta_gemmini_infer.c
	$(RISCV_CC) $(RISCV_CFLAGS) -Wno-unused-function $(GEMMINI_CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(RISCV_TARGET) $(HOST_TARGET) $(RISCV_GEMMINI_TARGET) $(RISCV_GEMMINI_POWER_TARGET) $(RISCV_CPU_POWER_TARGET)

.PHONY: all native gemmini gemmini-v2 power clean
