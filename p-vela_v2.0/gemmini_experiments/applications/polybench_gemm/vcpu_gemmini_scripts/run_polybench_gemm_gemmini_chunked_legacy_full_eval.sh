#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_ROOT="$(cd -- "${APP_ROOT}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${EXPERIMENTS_ROOT}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
GUEST_SRC_DIR="${APP_ROOT}/guest"
HOST_OUTPUT_ROOT="${APP_ROOT}/outputs/gemmini_chunked_legacy"

SSH_KEY="${WORKSPACE_ROOT}/q-vela/tmp-cloudinit/id_ed25519"
SSH_USER="ubuntu"
SSH_HOST="127.0.0.1"
SSH_PORT="2200"

GUEST_WORKDIR="/mnt/hostshare/applications/polybench_gemm/guest"
BENCH_BIN="./polybench_gemm_gemmini_chunked_legacy.riscv"
GUEST_OUTPUT_ROOT="/tmp/polybench_gemm_gemmini_chunked_legacy_eval"

DATASET="${1:-SMALL}"
REPS="${2:-1}"
GUEST_MAIN_VCPU="${3:-${GUEST_MAIN_VCPU:-1}}"
GUEST_GEMMINI_VCPU="${4:-${GUEST_GEMMINI_VCPU:-2}}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [MINI|SMALL|MEDIUM|LARGE|EXTRALARGE] [reps] [guest_main_vcpu] [guest_gemmini_vcpu]

Run PolyBench GEMM Gemmini chunked legacy evaluation from host via ssh.

Defaults:
  dataset  SMALL
  reps     1
  guest_main_vcpu     1
  guest_gemmini_vcpu  2

Examples:
  $(basename "$0") SMALL 1 1 2
  $(basename "$0") MEDIUM 50 1 2
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

dataset_dims() {
    case "$1" in
        MINI_DATASET) echo "20 25 30" ;;
        SMALL_DATASET) echo "60 70 80" ;;
        MEDIUM_DATASET) echo "200 220 240" ;;
        LARGE_DATASET) echo "1000 1100 1200" ;;
        EXTRALARGE_DATASET) echo "2000 2300 2600" ;;
        *) return 1 ;;
    esac
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "${name} must be a positive integer" >&2
        exit 1
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "${name} must be a non-negative integer" >&2
        exit 1
    fi
}

build_bench() {
    local macro="$1"
    make -B -C "${GUEST_SRC_DIR}" gemmini-chunked-legacy DATASET="${macro}"
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
        "taskset -c ${GUEST_MAIN_VCPU} bash -lc '
            set -euo pipefail
            sudo -n mkdir -p /mnt/hostshare
            if ! mountpoint -q /mnt/hostshare; then
                sudo -n mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/hostshare
            fi

            mkdir -p ${GUEST_OUTPUT_ROOT}
            RUN_NAME=run_\$(date +%s)_\$\$
            RUN_DIR=${GUEST_OUTPUT_ROOT}/\${RUN_NAME}

            cd ${GUEST_WORKDIR}

            echo [remote] guest_workdir=${GUEST_WORKDIR}
            echo [remote] run_dir=\${RUN_DIR}
            echo [launch-affinity] main_taskset_vcpu=${GUEST_MAIN_VCPU} gemmini_executor_vcpu=${GUEST_GEMMINI_VCPU}
            stdbuf -oL -eL ${BENCH_BIN} \
                --reps ${REPS} \
                --output-dir \${RUN_DIR} \
                --vcpu-offset ${GUEST_MAIN_VCPU} \
                --gemmini-vcpu ${GUEST_GEMMINI_VCPU}

            echo [remote] metrics_begin
            cat \"\${RUN_DIR}/metrics.json\"
            echo [remote] metrics_end
        '" | while IFS= read -r line; do
            printf '[HOST_TS_LINE] %s %s\n' "$(date +%s.%N)" "${line}"
        done
}

query_latest_remote_run() {
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
        "ls -dt ${GUEST_OUTPUT_ROOT}/run_* | head -n 1"
}

pull_remote_run_to_host() {
    local remote_run_dir="$1"
    local run_name

    run_name="$(basename "${remote_run_dir}")"
    mkdir -p "${HOST_OUTPUT_ROOT}"

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
        "tar -C ${GUEST_OUTPUT_ROOT} -cf - ${run_name}" \
        | tar -C "${HOST_OUTPUT_ROOT}" -xf -
}

main() {
    local macro dims ts bench_log latest_run

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    macro="$(dataset_macro "${DATASET}")"
    dims="$(dataset_dims "${macro}")"
    require_positive_int "reps" "${REPS}"
    require_nonnegative_int "guest_main_vcpu" "${GUEST_MAIN_VCPU}"
    require_nonnegative_int "guest_gemmini_vcpu" "${GUEST_GEMMINI_VCPU}"
    if [[ "${GUEST_MAIN_VCPU}" == "${GUEST_GEMMINI_VCPU}" ]]; then
        echo "guest_main_vcpu and guest_gemmini_vcpu must be different" >&2
        exit 1
    fi
    require_file "${SSH_KEY}"
    mkdir -p "${LOG_DIR}" "${HOST_OUTPUT_ROOT}"

    build_bench "${macro}"

    ts="$(date +%Y%m%d_%H%M%S)"
    bench_log="${LOG_DIR}/${ts}_polybench_gemm_gemmini_chunked_legacy_${macro}_reps${REPS}.log"

    {
        echo "[HOST_TS_REALTIME] BENCH_HOST_START $(date +%s.%N)"
        echo "[host] dataset=${macro} dims=${dims} reps=${REPS}"
        echo "[host] guest_main_vcpu=${GUEST_MAIN_VCPU} guest_gemmini_vcpu=${GUEST_GEMMINI_VCPU}"
        echo "[host] host_output_root=${HOST_OUTPUT_ROOT}"
        run_remote_eval
        echo "[HOST_TS_REALTIME] BENCH_HOST_END $(date +%s.%N)"
    } | tee "${bench_log}"

    latest_run="$(query_latest_remote_run)"
    if [[ -n "${latest_run}" ]]; then
        pull_remote_run_to_host "${latest_run}"
        echo "[host] pulled_remote_run=${latest_run}"
        echo "[host] host_run_dir=${HOST_OUTPUT_ROOT}/$(basename "${latest_run}")"
    fi

    echo "[host] bench_log=${bench_log}"
}

main "$@"
