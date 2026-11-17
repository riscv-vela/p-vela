#!/usr/bin/env bash

#
# Boot a RISC-V QEMU guest with configurable Gemmini / vCPU counts.
#
# Usage:
#   ./run_qemu.sh [num_gemmini] [num_vcpu]
#
# Default logfile: qemu-gemmini.log
# Shared folder:
#   host ${WORKSPACE_ROOT}/gemmini_experiments
#   guest /mnt/hostshare
#
# Optional runtime policy override:
#   QEMU_GEMMINI_POLICY=modulo ./run_qemu.sh 2 2
#   - modulo     : vCPU i -> Gemmini (i % num_gemmini)
#   - idle_first : existing default behavior
#
# Optional QEMU TCG plugin:
#   QEMU_PLUGIN_LIB=/path/to/libinsn.so QEMU_PLUGIN_OPTS="inline=true,match=rdcycle,trace=true" ./run_qemu.sh 1 2
#   - plugin output is written to ${QEMU_LOG}
#
# Layout:
#   - num_gemmini Gemmini devices are attached as gemmini0, gemmini1, ...
#   - num_vcpu guest vCPUs are created with -smp
#
# Example:
#   ./run_qemu.sh
#   ./run_qemu.sh 2 2
#   ./run_qemu.sh 4 4
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QEMU_ROOT="${WORKSPACE_ROOT}/q-vela/qemu"

QEMU_BIN="${QEMU_ROOT}/build/qemu-system-riscv64"
BIOS_BIN="/usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.bin"
KERNEL_BIN="/usr/lib/u-boot/qemu-riscv64_smode/uboot.elf"
DISK_IMAGE="${WORKSPACE_ROOT}/ubuntu-24.04.3-preinstalled-server-riscv64.img"
HOST_SHARE_DIR="${WORKSPACE_ROOT}/gemmini_experiments"
QEMU_LOG="${QEMU_ROOT}/qemu-gemmini.log"
SSH_PORT="2200"
NUM_GEMMINI="${1:-2}"
NUM_VCPU="${2:-2}"
MAX_NUM_GEMMINI=4
MAX_NUM_VCPU=12
QEMU_PLUGIN_LIB="${QEMU_PLUGIN_LIB:-}"
QEMU_PLUGIN_OPTS="${QEMU_PLUGIN_OPTS:-}"
QEMU_TRACE_FLAGS="${QEMU_TRACE_FLAGS:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [num_gemmini] [num_vcpu]

Boot QEMU with:
  - num_vcpu vCPUs (default: ${NUM_VCPU})
  - num_gemmini Gemmini devices (default: ${NUM_GEMMINI})
  - 9p shared folder mounted from:
    ${HOST_SHARE_DIR}

Examples:
  $(basename "$0")
  $(basename "$0") 2
  $(basename "$0") 2 2
  QEMU_GEMMINI_POLICY=modulo $(basename "$0") 2 2
  QEMU_PLUGIN_LIB=${QEMU_ROOT}/build/tests/tcg/plugins/libinsn.so QEMU_PLUGIN_OPTS="inline=true,match=rdcycle,trace=true" $(basename "$0") 1 2
  $(basename "$0") 3 3
  $(basename "$0") 4 4
  $(basename "$0") 1 12
EOF
}

main() {
    local gemmini_args=""
    local plugin_args=""
    local plugin_spec
    local idx

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    if ! [[ "${NUM_GEMMINI}" =~ ^[0-9]+$ ]] || (( NUM_GEMMINI < 1 || NUM_GEMMINI > MAX_NUM_GEMMINI )); then
        echo "num_gemmini must be between 1 and ${MAX_NUM_GEMMINI}" >&2
        exit 1
    fi

    if ! [[ "${NUM_VCPU}" =~ ^[0-9]+$ ]] || (( NUM_VCPU < 1 || NUM_VCPU > MAX_NUM_VCPU )); then
        echo "num_vcpu must be between 1 and ${MAX_NUM_VCPU}" >&2
        exit 1
    fi

    if [[ ! -x "${QEMU_BIN}" ]]; then
        echo "Missing QEMU binary: ${QEMU_BIN}" >&2
        exit 1
    fi

    if [[ ! -f "${DISK_IMAGE}" ]]; then
        echo "Missing disk image: ${DISK_IMAGE}" >&2
        exit 1
    fi

    if [[ ! -d "${HOST_SHARE_DIR}" ]]; then
        echo "Missing host share directory: ${HOST_SHARE_DIR}" >&2
        exit 1
    fi

    for idx in $(seq 0 $((NUM_GEMMINI - 1))); do
        gemmini_args+=" -device gemmini,id=gemmini${idx}"
    done

    if [[ -n "${QEMU_PLUGIN_LIB}" ]]; then
        if [[ ! -f "${QEMU_PLUGIN_LIB}" ]]; then
            echo "Missing QEMU plugin: ${QEMU_PLUGIN_LIB}" >&2
            exit 1
        fi
        plugin_spec="${QEMU_PLUGIN_LIB}"
        if [[ -n "${QEMU_PLUGIN_OPTS}" ]]; then
            plugin_spec+=",${QEMU_PLUGIN_OPTS}"
        fi
        plugin_args+=" -plugin ${plugin_spec}"
        plugin_args+=" -d ${QEMU_TRACE_FLAGS:-plugin}"
    fi

    cd "${QEMU_ROOT}"
    rm -f "${QEMU_LOG}"

    # shellcheck disable=SC2086
    exec "${QEMU_BIN}" \
      -machine virt \
      -accel tcg,thread=multi \
      -cpu rv64,v=true,vlen=512,vext_spec=v1.0,zba=true,zbb=true,zbs=true,zbc=true,zbkb=true,zcb=true,zicond=true \
      -smp "${NUM_VCPU}" \
      -m 4G \
      -nographic \
      -bios "${BIOS_BIN}" \
      -kernel "${KERNEL_BIN}" \
      -drive "file=${DISK_IMAGE},format=raw,if=virtio" \
      -netdev "user,id=net0,hostfwd=tcp::${SSH_PORT}-:22" \
      -device virtio-net-pci,netdev=net0,romfile="" \
      -fsdev "local,id=hostshare,path=${HOST_SHARE_DIR},security_model=none" \
      -device virtio-9p-device,fsdev=hostshare,mount_tag=hostshare \
      ${gemmini_args} \
      ${plugin_args} \
      -D "${QEMU_LOG}"
}

main "$@"
