#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${WORKSPACE_ROOT}/gemmini_experiments/power_modeling/single_app_power_modeling/runs"
RESULT_ROOT="${WORKSPACE_ROOT}/gemmini_experiments/power_modeling/single_app_power_modeling/results"
HOST_COLLECTOR="${SCRIPT_DIR}/run_single_app_power_host_collector.sh"
GUEST_SCRIPT="/mnt/hostshare/power_modeling/single_app_power_modeling/experiment_scripts/run_single_app_power_guest.sh"

SSH_KEY="${WORKSPACE_ROOT}/q-vela/tmp-cloudinit/id_ed25519"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_HOST="${SSH_HOST:-127.0.0.1}"
SSH_PORT="${SSH_PORT:-2200}"
HOST_HOUSEKEEPING_CORE="${HOST_HOUSEKEEPING_CORE:-8}"

HOST_COLLECTOR_PID=""

usage() {
    cat <<EOF
Usage:
  $(basename "$0") APP [guest options]
  $(basename "$0") --app APP [guest options]

Examples:
  $(basename "$0") polybench_gemm --reps 10
  $(basename "$0") distilbert --sample-limit 1

Run this on the host after QEMU boot and pinning.
EOF
}

cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "${HOST_COLLECTOR_PID}" ]] && kill -0 "${HOST_COLLECTOR_PID}" 2>/dev/null; then
        kill "${HOST_COLLECTOR_PID}" 2>/dev/null || true
        wait "${HOST_COLLECTOR_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}

ssh_guest() {
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
        "$@"
}

main() {
    local remote_command
    local guest_args=("$@")
    local app=""
    local run_id=""
    local has_run_id=0
    local index

    if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi
    [[ -f "${SSH_KEY}" ]] || {
        echo "ERROR: missing SSH key: ${SSH_KEY}" >&2
        exit 1
    }
    [[ "${HOST_HOUSEKEEPING_CORE}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: HOST_HOUSEKEEPING_CORE must be a non-negative integer" >&2
        exit 1
    }
    taskset -pc "${HOST_HOUSEKEEPING_CORE}" "$$" >/dev/null || exit 1
    echo "[single-power] host orchestration core=${HOST_HOUSEKEEPING_CORE}"

    if [[ "${guest_args[0]}" != -* ]]; then
        app="${guest_args[0]}"
    fi
    for ((index = 0; index < ${#guest_args[@]}; index++)); do
        case "${guest_args[index]}" in
            --app)
                app="${guest_args[index + 1]:-}"
                ;;
            --run-id)
                run_id="${guest_args[index + 1]:-}"
                has_run_id=1
                ;;
        esac
    done
    [[ -n "${app}" ]] || {
        echo "ERROR: application argument is required" >&2
        exit 1
    }
    run_id="${run_id:-run_$(date +%Y%m%d_%H%M%S)_${app}_$$}"
    [[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        echo "ERROR: invalid run ID" >&2
        exit 1
    }
    if [[ "${has_run_id}" != "1" ]]; then
        guest_args+=(--run-id "${run_id}")
    fi

    mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"
    chmod 0777 "${RUN_ROOT}" "${RESULT_ROOT}"
    echo "Refreshing sudo credential for turbostat..."
    sudo -v

    trap cleanup EXIT INT TERM
    echo "[single-power] run_id=${run_id}"
    bash "${HOST_COLLECTOR}" --once --run-id "${run_id}" &
    HOST_COLLECTOR_PID=$!

    printf -v remote_command '%q ' bash "${GUEST_SCRIPT}" "${guest_args[@]}"
    if ! ssh_guest "${remote_command}"; then
        echo "ERROR: guest power run failed" >&2
        exit 1
    fi

    wait "${HOST_COLLECTOR_PID}"
    HOST_COLLECTOR_PID=""
    echo "[single-power] host/guest measurement complete"
    echo "[single-power] model results: ${RESULT_ROOT}/${run_id}"
}

main "$@"
