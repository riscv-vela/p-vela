#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACC_ROOT="$(cd -- "${SCRIPT_DIR}" && pwd)"
DISTILROBERTA_ROOT="$(cd -- "${ACC_ROOT}/.." && pwd)"
EXPERIMENTS_ROOT="$(cd -- "${DISTILROBERTA_ROOT}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${EXPERIMENTS_ROOT}/../.." && pwd)"
LOG_DIR="${ACC_ROOT}/logs"
PIPELINE_OUTPUT_ROOT="${DISTILROBERTA_ROOT}/sst2_pipeline/outputs/c_vcpu"

SSH_KEY="${WORKSPACE_ROOT}/q-vela/tmp-cloudinit/id_ed25519"
SSH_USER="ubuntu"
SSH_HOST="127.0.0.1"
SSH_PORT="2200"

GUEST_WORKDIR="/mnt/hostshare/applications/distilroberta/sst2_pipeline/guest"
BENCH_BIN="./distilroberta_int8_infer.riscv"
GUEST_OUTPUT_ROOT="/tmp/distilroberta_c_vcpu_eval_full"

SAMPLE_LIMIT="${1:-all}"
NUM_THREADS="${2:-1}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [sample_limit|all] [threads]

Run DistilRoBERTa vCPU accuracy evaluation from host via ssh.
The guest binary writes completed run directories under:
  ${GUEST_OUTPUT_ROOT}
After completion, the newest run directory is copied back to:
  ${PIPELINE_OUTPUT_ROOT}

Defaults:
  sample_limit  ${SAMPLE_LIMIT}
  threads       ${NUM_THREADS}

Examples:
  $(basename "$0")
  $(basename "$0") all 1
  $(basename "$0") 32 1
  $(basename "$0") 1 1
EOF
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
}

run_remote_eval() {
    local limit_arg="$1"

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

            mkdir -p ${GUEST_OUTPUT_ROOT}

            cd ${GUEST_WORKDIR}

            echo [remote] guest_workdir=${GUEST_WORKDIR}
            echo [remote] guest_output_root=${GUEST_OUTPUT_ROOT}

            if [[ \"${limit_arg}\" == \"all\" ]]; then
                stdbuf -oL -eL ${BENCH_BIN} \
                    --threads ${NUM_THREADS} \
                    --output-dir ${GUEST_OUTPUT_ROOT}
            else
                stdbuf -oL -eL ${BENCH_BIN} \
                    --limit ${limit_arg} \
                    --threads ${NUM_THREADS} \
                    --output-dir ${GUEST_OUTPUT_ROOT}
            fi

            RUN_DIR=\$(ls -dt ${GUEST_OUTPUT_ROOT}/run_* | head -n 1)

            echo [remote] run_dir=\${RUN_DIR}
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
    mkdir -p "${PIPELINE_OUTPUT_ROOT}"

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
        | tar -C "${PIPELINE_OUTPUT_ROOT}" -xf -
}

main() {
    local ts bench_log latest_run latest_metrics remote_run

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    if [[ "${SAMPLE_LIMIT}" != "all" ]]; then
        if ! [[ "${SAMPLE_LIMIT}" =~ ^[0-9]+$ ]] || (( SAMPLE_LIMIT < 1 )); then
            echo "sample_limit must be 'all' or a positive integer" >&2
            exit 1
        fi
    fi

    if ! [[ "${NUM_THREADS}" =~ ^[0-9]+$ ]] || (( NUM_THREADS < 1 )); then
        echo "threads must be a positive integer" >&2
        exit 1
    fi

    require_file "${SSH_KEY}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${PIPELINE_OUTPUT_ROOT}"

    ts="$(date +%Y%m%d_%H%M%S)"
    bench_log="${LOG_DIR}/${ts}_distilroberta_c_vcpu_eval_limit${SAMPLE_LIMIT}_threads${NUM_THREADS}.log"

    {
        echo "[HOST_TS_REALTIME] BENCH_HOST_START $(date +%s.%N)"
        echo "[host] sample_limit=${SAMPLE_LIMIT} threads=${NUM_THREADS}"
        echo "[host] host_output_root=${PIPELINE_OUTPUT_ROOT}"
        run_remote_eval "${SAMPLE_LIMIT}"
        echo "[HOST_TS_REALTIME] BENCH_HOST_END $(date +%s.%N)"
    } | tee "${bench_log}"

    remote_run="$(query_latest_remote_run)"
    pull_remote_run_to_host "${remote_run}"
    latest_run="${PIPELINE_OUTPUT_ROOT}/$(basename "${remote_run}")"
    latest_metrics="${latest_run}/metrics.json"

    echo
    echo "Bench log   : ${bench_log}"
    echo "Host run dir: ${latest_run}"
    echo "Metrics     : ${latest_metrics}"
    echo
    cat "${latest_metrics}"
}

main "$@"
