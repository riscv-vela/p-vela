#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_ROOT="$(cd -- "${APP_ROOT}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${EXPERIMENTS_ROOT}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
GUEST_SRC_DIR="${APP_ROOT}/guest"

SSH_KEY="${WORKSPACE_ROOT}/q-vela/tmp-cloudinit/id_ed25519"
SSH_USER="ubuntu"
SSH_HOST="127.0.0.1"
SSH_PORT="2200"

GUEST_WORKDIR="/mnt/hostshare/applications/polybench_gemm/guest"
BENCH_BIN="./polybench_gemm_original.riscv"

DATASET="${1:-SMALL}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [MINI|SMALL|MEDIUM|LARGE|EXTRALARGE]

Run upstream PolyBench GEMM in the QEMU guest.

Default:
  dataset  SMALL

Examples:
  $(basename "$0") SMALL
  $(basename "$0") MEDIUM
EOF
}

dataset_macro() {
    case "$1" in
        MINI|MINI_DATASET) echo "MINI_DATASET" ;;
        SMALL|SMALL_DATASET) echo "SMALL_DATASET" ;;
        MEDIUM|MEDIUM_DATASET) echo "MEDIUM_DATASET" ;;
        LARGE|LARGE_DATASET) echo "LARGE_DATASET" ;;
        EXTRALARGE|EXTRALARGE_DATASET) echo "EXTRALARGE_DATASET" ;;
        *)
            echo "Unknown dataset: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
}

build_bench() {
    local macro="$1"
    make -B -C "${GUEST_SRC_DIR}" original-riscv DATASET="${macro}"
}

run_remote_eval() {
    ssh \
        -i "${SSH_KEY}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o PreferredAuthentications=publickey \
        -o PasswordAuthentication=no \
        -o LogLevel=ERROR \
        -p "${SSH_PORT}" \
        "${SSH_USER}@${SSH_HOST}" \
        "bash -lc '
            set -euo pipefail
            sudo -n mkdir -p /mnt/hostshare
            if ! mountpoint -q /mnt/hostshare; then
                sudo -n mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/hostshare
            fi

            cd ${GUEST_WORKDIR}
            echo [remote] guest_workdir=${GUEST_WORKDIR}
            echo [remote] bench_bin=${BENCH_BIN}
            stdbuf -oL -eL ${BENCH_BIN}
        '" | while IFS= read -r line; do
            printf '[HOST_TS_LINE] %s %s\n' "$(date +%s.%N)" "${line}"
        done
}

main() {
    local macro ts bench_log

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    macro="$(dataset_macro "${DATASET}")"
    require_file "${SSH_KEY}"
    mkdir -p "${LOG_DIR}"

    build_bench "${macro}"

    ts="$(date +%Y%m%d_%H%M%S)"
    bench_log="${LOG_DIR}/${ts}_polybench_gemm_original_${macro}.log"

    {
        echo "[HOST_TS_REALTIME] BENCH_HOST_START $(date +%s.%N)"
        echo "[host] dataset=${macro}"
        run_remote_eval
        echo "[HOST_TS_REALTIME] BENCH_HOST_END $(date +%s.%N)"
    } | tee "${bench_log}"

    echo "[host] bench_log=${bench_log}"
}

main "$@"

