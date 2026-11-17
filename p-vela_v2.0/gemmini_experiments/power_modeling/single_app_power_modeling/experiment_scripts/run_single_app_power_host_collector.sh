#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
POWER_MODELING_ROOT="$(cd -- "${SINGLE_ROOT}/.." && pwd)"
source "${POWER_MODELING_ROOT}/power_sim/power_model_defaults.env"
RUN_ROOT="${SINGLE_ROOT}/runs"
RESULT_ROOT="${SINGLE_ROOT}/results"
TIME_SCALE_SCRIPT="${POWER_MODELING_ROOT}/power_sim/time_scale_gemmini_logs.py"

HOST_HOUSEKEEPING_CORE="${HOST_HOUSEKEEPING_CORE:-8}"
POLL_INTERVAL=0.2
WATCH=0
TARGET_RUN_ID=""
ACTIVE_RUN_DIR=""
TURBOSTAT_PID=""
VCPU_CORES=()
GUEST_VCPU_IDS=()
SUCCESS=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--once|--watch] [--run-id ID]

Run this on the host before starting run_single_app_power_guest.sh.

  --once   process one guest request and exit (default)
  --watch  keep processing guest requests
  --run-id process only the matching guest request
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --once) WATCH=0; shift ;;
            --watch) WATCH=1; shift ;;
            --run-id) TARGET_RUN_ID="${2:-}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
}

require_uint() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
}

require_positive_number() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "${name} must be a positive number"
    awk -v value="${value}" 'BEGIN { exit !(value > 0) }' || die "${name} must be > 0"
}

stop_host_collectors() {
    if [[ -n "${TURBOSTAT_PID}" ]] && kill -0 "${TURBOSTAT_PID}" 2>/dev/null; then
        sudo -n kill "${TURBOSTAT_PID}" 2>/dev/null || true
        wait "${TURBOSTAT_PID}" 2>/dev/null || true
    fi
    TURBOSTAT_PID=""
}

cleanup() {
    local status=$?
    trap - EXIT
    stop_host_collectors
    if [[ "${SUCCESS}" != "1" && -n "${ACTIVE_RUN_DIR}" ]]; then
        echo "${status}" > "${ACTIVE_RUN_DIR}/host.failed"
    fi
    exit "${status}"
}

next_request_dir() {
    local marker
    local run_dir
    shopt -s nullglob
    for marker in "${RUN_ROOT}"/*/request.ready; do
        run_dir="$(dirname "${marker}")"
        if [[ -n "${TARGET_RUN_ID}" && "$(basename "${run_dir}")" != "${TARGET_RUN_ID}" ]]; then
            continue
        fi
        if [[ ! -e "${run_dir}/host.ready" && ! -e "${run_dir}/host.done" && ! -e "${run_dir}/host.failed" ]]; then
            printf '%s\n' "${run_dir}"
            return 0
        fi
    done
    return 1
}

read_request() {
    local request_file="$1"
    local key
    local value

    APP=""
    RUN_ID=""
    HOST_GEMMINI_CORE=""
    VCPU_MODEL_CORES=""
    GUEST_VCPUS=""
    EXECUTOR_MODEL_CORE=""
    GUEST_EXECUTOR_VCPU=""
    HOUSEKEEPING_MODEL_CORE=""
    GUEST_HOUSEKEEPING_VCPU=""
    TURBOSTAT_INTERVAL=""
    FIXED_FREQ_GHZ=""
    SYS_SCALE=""
    GEMMINI_FULL_UTIL_POWER_W=""
    GEMMINI_UTIL_THRESHOLD_PCT=""
    TIME_SCALE_MODE="${DEFAULT_TIME_SCALE_MODE}"
    GEMMINI_FREQUENCY_MHZ="${DEFAULT_GEMMINI_FREQUENCY_MHZ}"
    CPU_TIME_SCALE=""

    while IFS='=' read -r key value; do
        case "${key}" in
            APP) APP="${value}" ;;
            RUN_ID) RUN_ID="${value}" ;;
            HOST_GEMMINI_CORE) HOST_GEMMINI_CORE="${value}" ;;
            VCPU_MODEL_CORES) VCPU_MODEL_CORES="${value}" ;;
            GUEST_VCPUS) GUEST_VCPUS="${value}" ;;
            EXECUTOR_MODEL_CORE) EXECUTOR_MODEL_CORE="${value}" ;;
            GUEST_EXECUTOR_VCPU) GUEST_EXECUTOR_VCPU="${value}" ;;
            HOUSEKEEPING_MODEL_CORE) HOUSEKEEPING_MODEL_CORE="${value}" ;;
            GUEST_HOUSEKEEPING_VCPU) GUEST_HOUSEKEEPING_VCPU="${value}" ;;
            TURBOSTAT_INTERVAL) TURBOSTAT_INTERVAL="${value}" ;;
            FIXED_FREQ_GHZ) FIXED_FREQ_GHZ="${value}" ;;
            SYS_SCALE) SYS_SCALE="${value}" ;;
            GEMMINI_FULL_UTIL_POWER_W) GEMMINI_FULL_UTIL_POWER_W="${value}" ;;
            GEMMINI_UTIL_THRESHOLD_PCT) GEMMINI_UTIL_THRESHOLD_PCT="${value}" ;;
            TIME_SCALE_MODE) TIME_SCALE_MODE="${value}" ;;
            GEMMINI_FREQUENCY_MHZ) GEMMINI_FREQUENCY_MHZ="${value}" ;;
            CPU_TIME_SCALE) CPU_TIME_SCALE="${value}" ;;
        esac
    done < "${request_file}"

    [[ "${RUN_ID}" == "$(basename "${ACTIVE_RUN_DIR}")" ]] || die "request RUN_ID does not match its directory"
    [[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid RUN_ID"
    require_uint "HOST_GEMMINI_CORE" "${HOST_GEMMINI_CORE}"
    require_positive_number "TURBOSTAT_INTERVAL" "${TURBOSTAT_INTERVAL}"
    require_positive_number "FIXED_FREQ_GHZ" "${FIXED_FREQ_GHZ}"
    require_positive_number "SYS_SCALE" "${SYS_SCALE}"
    require_positive_number "GEMMINI_FULL_UTIL_POWER_W" "${GEMMINI_FULL_UTIL_POWER_W}"
    require_positive_number "GEMMINI_UTIL_THRESHOLD_PCT" "${GEMMINI_UTIL_THRESHOLD_PCT}"
    case "${TIME_SCALE_MODE}" in
        rtl-cycle) ;;
        *) die "TIME_SCALE_MODE must be rtl-cycle" ;;
    esac
    require_positive_number "GEMMINI_FREQUENCY_MHZ" "${GEMMINI_FREQUENCY_MHZ}"
    require_positive_number "CPU_TIME_SCALE" "${CPU_TIME_SCALE}"
    [[ "${VCPU_MODEL_CORES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid VCPU_MODEL_CORES"
    [[ "${GUEST_VCPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid GUEST_VCPUS"
    require_uint "EXECUTOR_MODEL_CORE" "${EXECUTOR_MODEL_CORE}"
    require_uint "GUEST_EXECUTOR_VCPU" "${GUEST_EXECUTOR_VCPU}"
    require_uint "HOUSEKEEPING_MODEL_CORE" "${HOUSEKEEPING_MODEL_CORE}"
    require_uint "GUEST_HOUSEKEEPING_VCPU" "${GUEST_HOUSEKEEPING_VCPU}"

    IFS=',' read -r -a VCPU_CORES <<< "${VCPU_MODEL_CORES}"
    IFS=',' read -r -a GUEST_VCPU_IDS <<< "${GUEST_VCPUS}"
    (( ${#VCPU_CORES[@]} == ${#GUEST_VCPU_IDS[@]} )) || \
        die "VCPU_MODEL_CORES and GUEST_VCPUS must contain the same number of entries"
    (( ${#VCPU_CORES[@]} == 1 )) || \
        die "single workload-vCPU mode requires exactly one vCPU mapping"
    [[ "${HOUSEKEEPING_MODEL_CORE}" != "${VCPU_CORES[0]}" ]] || \
        die "housekeeping and application host cores must differ"
    [[ "${EXECUTOR_MODEL_CORE}" != "${VCPU_CORES[0]}" ]] || \
        die "executor and application host cores must differ"
    [[ "${EXECUTOR_MODEL_CORE}" != "${HOUSEKEEPING_MODEL_CORE}" ]] || \
        die "executor and housekeeping host cores must differ"
    [[ "${GUEST_HOUSEKEEPING_VCPU}" != "${GUEST_VCPU_IDS[0]}" ]] || \
        die "housekeeping and application guest vCPUs must differ"
    [[ "${GUEST_EXECUTOR_VCPU}" != "${GUEST_VCPU_IDS[0]}" ]] || \
        die "executor and application guest vCPUs must differ"
    [[ "${GUEST_EXECUTOR_VCPU}" != "${GUEST_HOUSEKEEPING_VCPU}" ]] || \
        die "executor and housekeeping guest vCPUs must differ"
    configure_plot
}

configure_plot() {
    # All four applications share the single-app plotter next to this runner.
    PLOT_SCRIPT="${SCRIPT_DIR}/plot_single_app_gemmini_power.py"
    TIME_ZERO_EVENT="BENCH_HOST_START"
    case "${APP}" in
        distilbert)
            PRESENTATION_APP_NAME="DistilBERT"
            ;;
        distilroberta)
            PRESENTATION_APP_NAME="DistilRoBERTa"
            ;;
        distilgpt2)
            PRESENTATION_APP_NAME="DistilGPT2"
            ;;
        polybench_gemm)
            PRESENTATION_APP_NAME="PolyBench GEMM"
            ;;
        *) die "unsupported APP in request: ${APP}" ;;
    esac
    [[ -f "${PLOT_SCRIPT}" ]] || die "missing plot script: ${PLOT_SCRIPT}"
}

wait_for_guest() {
    while [[ ! -e "${ACTIVE_RUN_DIR}/guest.done" ]]; do
        if [[ -e "${ACTIVE_RUN_DIR}/guest.failed" ]]; then
            return 1
        fi
        if ! kill -0 "${TURBOSTAT_PID}" 2>/dev/null; then
            die "turbostat stopped unexpectedly"
        fi
        sleep "${POLL_INTERVAL}"
    done
}

run_plot() {
    local power_path="$1"
    local bench_path="$2"
    local plot_variant="${4:-raw}"
    local plot_time_zero_event="${TIME_ZERO_EVENT}"
    if [[ "${plot_variant}" == "scaled" ]]; then
        plot_time_zero_event="sample_start"
    fi
    local out_path="$3"
    local plot_args=(
        --guest-vcpus "${GUEST_VCPU_IDS[@]}"
        --power-csv "${power_path}"
        --bench "${bench_path}"
        --out "${out_path}"
        --gemmini-core "${HOST_GEMMINI_CORE}"
        --vcpu-cores "${VCPU_CORES[@]}"
        --time-zero-event "${plot_time_zero_event}"
        --presentation
        --app-name "${PRESENTATION_APP_NAME}"
    )
    if [[ "${plot_variant}" == "scaled" ]]; then
        plot_args+=(
            --plot-window inference
            --plot-window-pre-s "${DEFAULT_SCALED_PLOT_WINDOW_MARGIN_S}"
            --plot-window-post-s "${DEFAULT_SCALED_PLOT_WINDOW_MARGIN_S}"
            --x-tick-interval-s "${DEFAULT_PLOT_TIME_MAJOR_TICK_S}"
            --fig-width "${DEFAULT_SCALED_PLOT_FIG_WIDTH}"
        )
    elif [[ "${plot_variant}" == "raw-combined" ]]; then
        plot_args+=(--crop-before-time-zero)
    else
        plot_args+=(
            --crop-before-time-zero --raw-util-plots
            --executor-core "${EXECUTOR_MODEL_CORE}"
            --guest-executor-vcpu "${GUEST_EXECUTOR_VCPU}"
            --housekeeping-core "${HOUSEKEEPING_MODEL_CORE}"
            --guest-housekeeping-vcpu "${GUEST_HOUSEKEEPING_VCPU}"
        )
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 "${PLOT_SCRIPT}" "${plot_args[@]}"
}

verify_plot() {
    [[ -s "$1" ]] || die "missing plot: $1"
}

process_request() {
    local result_dir
    local result_prefix
    local turbostat_log
    local raw_power_csv
    local tracked_cores
    local time_scale_tag frequency_tag
    local cpu_time_scale_tag
    local scaled_prefix
    local scaled_bench_log
    local scaled_turbostat_log
    local scaled_guest_util_csv
    local scaled_power_csv
    local scaled_plot
    local required
    local time_scale_args=()

    read_request "${ACTIVE_RUN_DIR}/request.env"
    result_dir="${RESULT_ROOT}/${RUN_ID}"
    [[ -d "${result_dir}" && -w "${result_dir}" ]] || \
        die "guest must create writable result directory: ${result_dir}"
    mv "${ACTIVE_RUN_DIR}/request.ready" "${ACTIVE_RUN_DIR}/request.accepted"

    turbostat_log="${ACTIVE_RUN_DIR}/raw_turbostat.log"
    raw_power_csv="${result_dir}/raw_power.csv"
    tracked_cores="${HOST_GEMMINI_CORE},${VCPU_MODEL_CORES},${EXECUTOR_MODEL_CORE},${HOUSEKEEPING_MODEL_CORE}"
    cpu_time_scale_tag="${CPU_TIME_SCALE//./p}"
    frequency_tag="${GEMMINI_FREQUENCY_MHZ//./p}"
    time_scale_tag="rtl_${frequency_tag}mhz"
    time_scale_args=(
        --time-scale-mode rtl-cycle
        --rtl-workload "${APP}"
        --gemmini-frequency-mhz "${GEMMINI_FREQUENCY_MHZ}"
    )
    scaled_prefix="${ACTIVE_RUN_DIR}/time_scaled_${time_scale_tag}_cpu_${cpu_time_scale_tag}"
    result_prefix="${result_dir}/time_scaled_${time_scale_tag}_cpu_${cpu_time_scale_tag}"
    scaled_bench_log="${scaled_prefix}_bench.log"
    scaled_turbostat_log="${scaled_prefix}_turbostat.log"
    scaled_guest_util_csv="${scaled_prefix}_guest_util.csv"
    scaled_power_csv="${result_prefix}_power.csv"
    scaled_plot="${result_prefix}_util_power.png"

    echo "[host] request: app=${APP} run_id=${RUN_ID}"
    echo "[host] scales: sys=${SYS_SCALE} gemmini=${GEMMINI_FULL_UTIL_POWER_W} threshold=${GEMMINI_UTIL_THRESHOLD_PCT}% time_mode=${TIME_SCALE_MODE} frequency=${GEMMINI_FREQUENCY_MHZ}MHz cpu_time=${CPU_TIME_SCALE}"
    echo "[host] starting turbostat on host cores ${tracked_cores}"
    sudo -n turbostat \
        -q \
        -d \
        --show Time_Of_Day_Seconds,CPU,Busy%,Bzy_MHz,Avg_MHz \
        -c "${tracked_cores}" \
        -i "${TURBOSTAT_INTERVAL}" \
        > "${turbostat_log}" 2>&1 &
    TURBOSTAT_PID=$!

    sleep 1
    kill -0 "${TURBOSTAT_PID}" 2>/dev/null || die "turbostat failed to start"
    : > "${ACTIVE_RUN_DIR}/host.ready"

    wait_for_guest || die "guest benchmark failed"
    sleep 1
    stop_host_collectors

    [[ -s "${raw_power_csv}" ]] || die "missing combined power CSV"
    [[ -s "${ACTIVE_RUN_DIR}/raw_guest_proc_stat.csv" ]] || die "missing guest /proc/stat CSV"
    run_plot \
        "${raw_power_csv}" \
        "${ACTIVE_RUN_DIR}/raw_bench.log" \
        "${result_dir}/raw_host_gemmini_util.png"
    verify_plot "${result_dir}/raw_host_gemmini_util.png"
    verify_plot "${result_dir}/raw_guest_vcpu_util.png"
    verify_plot "${result_dir}/raw_guest_executor_vcpu_util.png"
    verify_plot "${result_dir}/raw_guest_housekeeping_util.png"

    run_plot \
        "${raw_power_csv}" \
        "${ACTIVE_RUN_DIR}/raw_bench.log" \
        "${result_dir}/raw_util_power.png" \
        raw-combined
    verify_plot "${result_dir}/raw_util_power.png"

    PYTHONDONTWRITEBYTECODE=1 python3 "${TIME_SCALE_SCRIPT}" \
        --bench "${ACTIVE_RUN_DIR}/raw_bench.log" \
        --turbostat "${turbostat_log}" \
        --power-csv "${raw_power_csv}" \
        --guest-util-csv "${ACTIVE_RUN_DIR}/raw_guest_proc_stat.csv" \
        "${time_scale_args[@]}" \
        --cpu-time-scale "${CPU_TIME_SCALE}" \
        --out-prefix "${scaled_prefix}" \
        --result-prefix "${result_prefix}"

    for required in \
        "${scaled_bench_log}" \
        "${scaled_turbostat_log}" \
        "${scaled_guest_util_csv}" \
        "${scaled_power_csv}" \
        "${result_prefix}_summary.json"; do
        [[ -s "${required}" ]] || die "missing time-scaled output: ${required}"
    done

    run_plot \
        "${scaled_power_csv}" \
        "${scaled_bench_log}" \
        "${scaled_plot}" \
        scaled
    verify_plot "${scaled_plot}"
    : > "${ACTIVE_RUN_DIR}/host.done"

    echo "[host] complete"
    echo "[host] run dir       : ${ACTIVE_RUN_DIR}"
    echo "[host] model results : ${result_dir}"
    echo "[host] raw host util : ${result_dir}/raw_host_gemmini_util.png"
    echo "[host] raw util/power: ${result_dir}/raw_util_power.png"
    echo "[host] guest procstat: ${ACTIVE_RUN_DIR}/raw_guest_proc_stat.csv"
    echo "[host] host turbostat: ${turbostat_log}"
    echo "[host] combined power: ${raw_power_csv}"
    echo "[host] scaled power  : ${scaled_power_csv}"
    echo "[host] scaled summary: ${result_prefix}_summary.json"
    echo "[host] scaled plot   : ${scaled_plot}"
}

main() {
    local request_dir
    local waiting_reported=0

    parse_args "$@"
    require_uint "HOST_HOUSEKEEPING_CORE" "${HOST_HOUSEKEEPING_CORE}"
    taskset -pc "${HOST_HOUSEKEEPING_CORE}" "$$" >/dev/null || die "failed to pin host collector"
    echo "[host] collector housekeeping core=${HOST_HOUSEKEEPING_CORE}"
    if [[ -n "${TARGET_RUN_ID}" && ! "${TARGET_RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        die "invalid --run-id"
    fi
    for required in "${TIME_SCALE_SCRIPT}"; do
        [[ -f "${required}" ]] || die "missing required file: ${required}"
    done
    mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"
    chmod 0777 "${RUN_ROOT}" "${RESULT_ROOT}"

    echo "Refreshing sudo credential for turbostat..."
    sudo -v
    trap cleanup EXIT INT TERM

    while true; do
        request_dir=""
        request_dir="$(next_request_dir || true)"
        if [[ -z "${request_dir}" ]]; then
            if [[ "${waiting_reported}" != "1" ]]; then
                echo "[host] waiting for guest request under ${RUN_ROOT}"
                waiting_reported=1
            fi
            sleep "${POLL_INTERVAL}"
            continue
        fi

        waiting_reported=0
        ACTIVE_RUN_DIR="${request_dir}"
        process_request
        SUCCESS=1
        ACTIVE_RUN_DIR=""
        if [[ "${WATCH}" != "1" ]]; then
            break
        fi
        SUCCESS=0
    done
}

main "$@"
