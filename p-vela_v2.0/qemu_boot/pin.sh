#!/bin/bash
#
# pin_threads.sh — Pin QEMU vCPU and Gemmini worker threads to physical cores
#
# Usage:
#   ./pin.sh [vcpu_base_core] [gemmini_base_core]
#   ./pin.sh [logfile] [vcpu_base_core] [gemmini_base_core]
#
# Default logfile: qemu-gemmini.log
# Layout:
#   vCPU  0,1,2,3   → physical cores vcpu_base_core + 0,1,2,3
#   Worker 0,1,2,3  → physical cores gemmini_base_core + 0,1,2,3
#
# Example for single-vCPU/single-Gemmini experiment:
#   ./pin.sh 2 3
#   -> vCPU 0 goes to core 2
#   -> Gemmini 0 goes to core 3
#

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LOGFILE="${WORKSPACE_ROOT}/q-vela/qemu/qemu-gemmini.log"

is_positive_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ $# -ge 1 ]] && is_positive_int "$1"; then
    LOGFILE="${DEFAULT_LOGFILE}"
    VCPU_BASE_CORE="${1:-0}"
    GEMMINI_BASE_CORE="${2:-}"
else
    LOGFILE="${1:-${DEFAULT_LOGFILE}}"
    VCPU_BASE_CORE="${2:-0}"
    GEMMINI_BASE_CORE="${3:-}"
fi

if [[ ! -f "$LOGFILE" ]]; then
    echo "ERROR: log file '$LOGFILE' not found."
    echo "Start QEMU first with: -D qemu_thread.log"
    exit 1
fi

# ---- Parse TIDs from log ----
# Log lines:
#   qemu_thread_start called for CPU 0/TCG at TID 606793
#   qemu_thread_start called for gemmini-worker at TID 606797

declare -a VCPU_TIDS
declare -a GEMMINI_TIDS

while IFS= read -r line; do
    if [[ "$line" =~ "qemu_thread_start called for CPU "([0-9]+)"/TCG at TID "([0-9]+) ]]; then
        cpu_id="${BASH_REMATCH[1]}"
        tid="${BASH_REMATCH[2]}"
        VCPU_TIDS[$cpu_id]=$tid
    elif [[ "$line" =~ "qemu_thread_start called for gemmini-worker at TID "([0-9]+) ]]; then
        GEMMINI_TIDS+=("${BASH_REMATCH[1]}")
    fi
done < "$LOGFILE"

echo "=== Parsed TIDs from $LOGFILE ==="
for i in "${!VCPU_TIDS[@]}"; do
    echo "  vCPU $i   → TID ${VCPU_TIDS[$i]}"
done
for i in "${!GEMMINI_TIDS[@]}"; do
    echo "  Gemmini $i → TID ${GEMMINI_TIDS[$i]}"
done
echo ""

NUM_VCPUS=${#VCPU_TIDS[@]}
NUM_GEMMINIS=${#GEMMINI_TIDS[@]}

if [[ $NUM_VCPUS -eq 0 ]]; then
    echo "ERROR: No vCPU TIDs found in log. Check log format."
    exit 1
fi
if [[ $NUM_GEMMINIS -eq 0 ]]; then
    echo "ERROR: No gemmini-worker TIDs found in log. Check log format."
    exit 1
fi

QEMU_PID=""
for i in "${!VCPU_TIDS[@]}"; do
    QEMU_PID="$(sed -n 's/^Tgid:[[:space:]]*//p' "/proc/${VCPU_TIDS[$i]}/status")"
    break
done
if [[ ! "${QEMU_PID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Failed to resolve the QEMU process ID from vCPU TIDs."
    exit 1
fi

DEFAULT_GEMMINI_BASE_CORE=$((VCPU_BASE_CORE + NUM_VCPUS))
GEMMINI_BASE_CORE="${GEMMINI_BASE_CORE:-$DEFAULT_GEMMINI_BASE_CORE}"

echo "=== Core Mapping Policy ==="
echo "  QEMU housekeeping : host core ${VCPU_BASE_CORE}"
echo "  vCPU base core    : ${VCPU_BASE_CORE}"
echo "  Gemmini base core : ${GEMMINI_BASE_CORE}"
echo ""

pin_thread() {
    local tid=$1
    local core=$2
    local label=$3

    if ! kill -0 "$tid" 2>/dev/null; then
        echo "  [SKIP] $label TID $tid — thread not running"
        return
    fi

    taskset -cp "$core" "$tid" > /dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo "  [OK]   $label TID $tid → core $core"
    else
        echo "  [FAIL] $label TID $tid → core $core (taskset failed — core out of range?)"
    fi
}

echo "=== Pinning QEMU process to housekeeping core ==="
if taskset -apc "${VCPU_BASE_CORE}" "${QEMU_PID}" > /dev/null 2>&1; then
    echo "  [OK]   QEMU PID ${QEMU_PID} and current auxiliary threads → core ${VCPU_BASE_CORE}"
else
    echo "  [FAIL] QEMU PID ${QEMU_PID} → core ${VCPU_BASE_CORE}"
    exit 1
fi


echo "=== Pinning vCPU threads ==="
for i in "${!VCPU_TIDS[@]}"; do
    core=$((VCPU_BASE_CORE + i))
    pin_thread "${VCPU_TIDS[$i]}" "$core" "vCPU $i  "
done

echo ""
echo "=== Pinning Gemmini threads ==="
for i in "${!GEMMINI_TIDS[@]}"; do
    core=$((GEMMINI_BASE_CORE + i))
    pin_thread "${GEMMINI_TIDS[$i]}" "$core" "Gemmini $i"
done

echo ""
echo "Done. Verify with: taskset -cp <TID>"
