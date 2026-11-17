#!/usr/bin/env bash

set -euo pipefail
umask 000

SHARE_ROOT="/mnt/hostshare"
SINGLE_ROOT="${SHARE_ROOT}/power_modeling/single_app_power_modeling"
RUN_ROOT="${SINGLE_ROOT}/runs"
RESULT_ROOT="${SINGLE_ROOT}/results"
RESULT_DIR=""
COMBINED_POWER_SIM="${SHARE_ROOT}/power_modeling/power_sim/power_sim_vcpu_procstat.py"
source "${SHARE_ROOT}/power_modeling/power_sim/power_model_defaults.env"

APP=""
RUN_ID=""
SAMPLE_LIMIT=1
NUM_THREADS=1
REPS=50
GUEST_MAIN_VCPU=1
GUEST_HOUSEKEEPING_VCPU=0
GUEST_EXECUTOR_VCPU=2
VCPU_BASE_CORE=8
HOST_GEMMINI_CORE=2
VCPU_SAMPLE_INTERVAL="${DEFAULT_VCPU_SAMPLE_INTERVAL_S}"
TURBOSTAT_INTERVAL="${DEFAULT_TURBOSTAT_INTERVAL_S}"
FIXED_FREQ_GHZ="${DEFAULT_FIXED_FREQ_GHZ}"
SYS_SCALE=""
GEMMINI_FULL_UTIL_POWER_W="${GEMMINI_FULL_UTIL_POWER_W:-${DEFAULT_GEMMINI_FULL_UTIL_POWER_W}}"
GEMMINI_UTIL_THRESHOLD_PCT="${DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT}"
TIME_SCALE_MODE="${DEFAULT_TIME_SCALE_MODE}"
GEMMINI_FREQUENCY_MHZ="${DEFAULT_GEMMINI_FREQUENCY_MHZ}"
CPU_TIME_SCALE="${DEFAULT_CPU_TIME_SCALE}"
HOST_READY_TIMEOUT_S=120
HOST_DONE_TIMEOUT_S=180
EXPECTED_TIMEZONE="Asia/Seoul"
EXPECTED_GUEST_VCPU_COUNT=3

POWER_MONITOR_PID=""
RUN_DIR=""
HOUSEKEEPING_OBS_CORE=""
EXECUTOR_MODEL_CORE=""
SUCCESS=0

usage() {
    cat <<EOF
Usage:
  $(basename "$0") APP [options]
  $(basename "$0") --app APP [options]

Apps:
  distilbert | distilroberta | distilgpt2 | polybench_gemm

Options:
  --run-id ID
  --sample-limit N
  --threads N
  --reps N
  --main-vcpu N
  --housekeeping-vcpu N
  --executor-vcpu N
  --vcpu-base-core N
  --gemmini-host-core N
  --sys-scale FLOAT
  --fixed-freq-ghz FLOAT
  --gemmini-full-util-power-w FLOAT
  --gemmini-util-threshold-pct FLOAT
  --time-scale-mode rtl-cycle
  --gemmini-frequency-mhz FLOAT
  --cpu-time-scale FLOAT

The host collector must be running before this command.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
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

parse_args() {
    if [[ $# -gt 0 && "$1" != -* ]]; then
        APP="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --app) APP="${2:-}"; shift 2 ;;
            --run-id) RUN_ID="${2:-}"; shift 2 ;;
            --sample-limit) SAMPLE_LIMIT="${2:-}"; shift 2 ;;
            --threads) NUM_THREADS="${2:-}"; shift 2 ;;
            --reps) REPS="${2:-}"; shift 2 ;;
            --main-vcpu) GUEST_MAIN_VCPU="${2:-}"; shift 2 ;;
            --housekeeping-vcpu) GUEST_HOUSEKEEPING_VCPU="${2:-}"; shift 2 ;;
            --executor-vcpu) GUEST_EXECUTOR_VCPU="${2:-}"; shift 2 ;;
            --vcpu-base-core) VCPU_BASE_CORE="${2:-}"; shift 2 ;;
            --gemmini-host-core) HOST_GEMMINI_CORE="${2:-}"; shift 2 ;;
            --sys-scale) SYS_SCALE="${2:-}"; shift 2 ;;
            --fixed-freq-ghz) FIXED_FREQ_GHZ="${2:-}"; shift 2 ;;
            --gemmini-full-util-power-w) GEMMINI_FULL_UTIL_POWER_W="${2:-}"; shift 2 ;;
            --gemmini-util-threshold-pct) GEMMINI_UTIL_THRESHOLD_PCT="${2:-}"; shift 2 ;;
            --time-scale-mode) TIME_SCALE_MODE="${2:-}"; shift 2 ;;
            --gemmini-frequency-mhz) GEMMINI_FREQUENCY_MHZ="${2:-}"; shift 2 ;;
            --cpu-time-scale) CPU_TIME_SCALE="${2:-}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
}

configure_app() {
    case "${APP}" in
        distilbert)
            APP_KIND="transformer"
            WORKDIR="${SHARE_ROOT}/applications/distilbert/sst2_pipeline/guest"
            BENCH_BIN="./distilbert_gemmini_power_infer.riscv"
            EXTRA_ARGS=()
            ;;
        distilroberta)
            APP_KIND="transformer"
            WORKDIR="${SHARE_ROOT}/applications/distilroberta/sst2_pipeline/guest"
            BENCH_BIN="./distilroberta_gemmini_power_infer.riscv"
            EXTRA_ARGS=()
            ;;
        distilgpt2)
            APP_KIND="transformer"
            WORKDIR="${SHARE_ROOT}/applications/distilgpt2/sst2_pipeline/guest"
            BENCH_BIN="./distilgpt2_gemmini_power_infer.riscv"
            EXTRA_ARGS=(--no-host-compare)
            ;;
        polybench_gemm)
            APP_KIND="polybench"
            WORKDIR="${SHARE_ROOT}/applications/polybench_gemm/guest"
            BENCH_BIN="./polybench_gemm_gemmini_power.riscv"
            EXTRA_ARGS=()
            ;;
        *) die "unsupported app: ${APP}" ;;
    esac
    SYS_SCALE="${SYS_SCALE:-${DEFAULT_SYS_SCALE}}"
    CPU_TIME_SCALE="${CPU_TIME_SCALE:-${DEFAULT_CPU_TIME_SCALE}}"
}

validate_config() {
    local online_vcpus
    [[ -n "${APP}" ]] || die "APP is required"
    for item in \
        "sample_limit:${SAMPLE_LIMIT}" \
        "threads:${NUM_THREADS}" \
        "reps:${REPS}" \
        "main_vcpu:${GUEST_MAIN_VCPU}" \
        "housekeeping_vcpu:${GUEST_HOUSEKEEPING_VCPU}" \
        "executor_vcpu:${GUEST_EXECUTOR_VCPU}" \
        "vcpu_base_core:${VCPU_BASE_CORE}" \
        "gemmini_host_core:${HOST_GEMMINI_CORE}"; do
        require_uint "${item%%:*}" "${item#*:}"
    done
    require_positive_number "sys_scale" "${SYS_SCALE}"
    require_positive_number "fixed_freq_ghz" "${FIXED_FREQ_GHZ}"
    require_positive_number "gemmini_full_util_power_w" "${GEMMINI_FULL_UTIL_POWER_W}"
    require_positive_number "gemmini_util_threshold_pct" "${GEMMINI_UTIL_THRESHOLD_PCT}"
    case "${TIME_SCALE_MODE}" in
        rtl-cycle) ;;
        *) die "time_scale_mode must be rtl-cycle" ;;
    esac
    require_positive_number "gemmini_frequency_mhz" "${GEMMINI_FREQUENCY_MHZ}"
    require_positive_number "cpu_time_scale" "${CPU_TIME_SCALE}"
    [[ "${NUM_THREADS}" == "1" ]] || die "single workload-vCPU mode requires --threads 1"
    [[ "${GUEST_HOUSEKEEPING_VCPU}" != "${GUEST_MAIN_VCPU}" ]] || \
        die "housekeeping and application vCPU must differ"
    [[ "${GUEST_HOUSEKEEPING_VCPU}" != "${GUEST_EXECUTOR_VCPU}" ]] || \
        die "housekeeping and executor vCPU must differ"
    [[ "${GUEST_MAIN_VCPU}" != "${GUEST_EXECUTOR_VCPU}" ]] || \
        die "application and executor vCPU must differ"
    online_vcpus="$(getconf _NPROCESSORS_ONLN)" || die "failed to read guest vCPU count"
    [[ "${online_vcpus}" == "${EXPECTED_GUEST_VCPU_COUNT}" ]] || \
        die "expected ${EXPECTED_GUEST_VCPU_COUNT} guest vCPUs, found ${online_vcpus}; reboot with run_qemu.sh 1 3"
    [[ -x "${WORKDIR}/${BENCH_BIN#./}" ]] || die "missing benchmark binary: ${WORKDIR}/${BENCH_BIN#./}"
    command -v python3 >/dev/null 2>&1 || die "python3 is required in the guest"
    command -v timedatectl >/dev/null 2>&1 || die "timedatectl is required in the guest"
    [[ -f "${COMBINED_POWER_SIM}" ]] || die "missing combined power simulator: ${COMBINED_POWER_SIM}"
}

stop_guest_collectors() {
    if [[ -n "${POWER_MONITOR_PID}" ]] && kill -0 "${POWER_MONITOR_PID}" 2>/dev/null; then
        kill "${POWER_MONITOR_PID}" 2>/dev/null || true
        wait "${POWER_MONITOR_PID}" 2>/dev/null || true
    fi
    POWER_MONITOR_PID=""
}

csv_has_data_row() {
    local path="$1"
    [[ -f "${path}" ]] && awk 'END { exit !(NR >= 2) }' "${path}"
}

ensure_guest_timezone() {
    local timezone
    timezone="$(timedatectl show --property=Timezone --value)" || \
        die "failed to read guest timezone"
    if [[ "${timezone}" != "${EXPECTED_TIMEZONE}" ]]; then
        echo "[guest] setting timezone: ${timezone} -> ${EXPECTED_TIMEZONE}"
        sudo -n timedatectl set-timezone "${EXPECTED_TIMEZONE}" || \
            die "failed to set guest timezone to ${EXPECTED_TIMEZONE}"
        timezone="$(timedatectl show --property=Timezone --value)" || \
            die "failed to verify guest timezone"
    fi
    [[ "${timezone}" == "${EXPECTED_TIMEZONE}" ]] || \
        die "guest timezone is ${timezone}, expected ${EXPECTED_TIMEZONE}"
    echo "[guest] timezone: ${timezone}"
}

configure_guest_housekeeping() {
    local housekeeping_mask

    printf -v housekeeping_mask '%x' "$((1 << GUEST_HOUSEKEEPING_VCPU))"
    if ! sudo -n env HOUSEKEEPING_CPU="${GUEST_HOUSEKEEPING_VCPU}" HOUSEKEEPING_MASK="${housekeeping_mask}" bash -c '
            set -u
            irq_moved=0
            irq_failed=0
            workqueue_moved=0
            workqueue_failed=0
            process_moved=0
            process_failed=0
            userspace_violations=0

            if printf "%s\n" "${HOUSEKEEPING_MASK}" > /proc/irq/default_smp_affinity 2>/dev/null; then
                default_irq_status=ok
            else
                default_irq_status=failed
            fi

            for affinity in /proc/irq/[0-9]*/smp_affinity_list; do
                [[ -e "${affinity}" ]] || continue
                if printf "%s\n" "${HOUSEKEEPING_CPU}" > "${affinity}" 2>/dev/null; then
                    irq_moved=$((irq_moved + 1))
                else
                    irq_failed=$((irq_failed + 1))
                fi
            done

            for cpumask in /sys/devices/virtual/workqueue/cpumask /sys/devices/virtual/workqueue/*/cpumask; do
                [[ -e "${cpumask}" ]] || continue
                if printf "%s\n" "${HOUSEKEEPING_MASK}" > "${cpumask}" 2>/dev/null; then
                    workqueue_moved=$((workqueue_moved + 1))
                else
                    workqueue_failed=$((workqueue_failed + 1))
                fi
            done

            for proc_dir in /proc/[0-9]*; do
                pid="${proc_dir##*/}"
                if taskset -apc "${HOUSEKEEPING_CPU}" "${pid}" >/dev/null 2>&1; then
                    process_moved=$((process_moved + 1))
                else
                    process_failed=$((process_failed + 1))
                fi
            done

            for proc_dir in /proc/[0-9]*; do
                [[ -e "${proc_dir}/exe" ]] || continue
                allowed="$(sed -n "s/^Cpus_allowed_list:[[:space:]]*//p" "${proc_dir}/status" 2>/dev/null || true)"
                if [[ -n "${allowed}" && "${allowed}" != "${HOUSEKEEPING_CPU}" ]]; then
                    userspace_violations=$((userspace_violations + 1))
                    printf "[guest-housekeeping-warning] pid=%s allowed=%s comm=%s\n" "${proc_dir##*/}" "${allowed}" "$(cat "${proc_dir}/comm" 2>/dev/null || true)"
                fi
            done

            printf "[guest-housekeeping] cpu=%s mask=%s default_irq=%s irq_moved=%d irq_failed=%d workqueue_moved=%d workqueue_failed=%d process_moved=%d process_failed=%d userspace_violations=%d\n" "${HOUSEKEEPING_CPU}" "${HOUSEKEEPING_MASK}" "${default_irq_status}" "${irq_moved}" "${irq_failed}" "${workqueue_moved}" "${workqueue_failed}" "${process_moved}" "${process_failed}" "${userspace_violations}"
        ' 2>&1; then
        die "failed to configure guest housekeeping affinity"
    fi

}

cleanup() {
    local status=$?
    trap - EXIT
    stop_guest_collectors
    if [[ "${SUCCESS}" != "1" && -n "${RUN_DIR}" ]]; then
        echo "${status}" > "${RUN_DIR}/guest.failed"
    fi
    exit "${status}"
}

wait_for_file() {
    local path="$1"
    local timeout_s="$2"
    local failed_path="${3:-}"
    local start_s
    start_s="$(date +%s)"
    while [[ ! -e "${path}" ]]; do
        if [[ -n "${failed_path}" && -e "${failed_path}" ]]; then
            return 2
        fi
        if (( $(date +%s) - start_s >= timeout_s )); then
            return 1
        fi
        sleep 0.2
    done
}

write_request() {
    local request_tmp="${RUN_DIR}/request.env.tmp"
    local request_file="${RUN_DIR}/request.env"
    cat > "${request_tmp}" <<EOF
APP=${APP}
RUN_ID=${RUN_ID}
HOST_GEMMINI_CORE=${HOST_GEMMINI_CORE}
VCPU_MODEL_CORES=${MAIN_MODEL_CORE}
GUEST_VCPUS=${GUEST_MAIN_VCPU}
EXECUTOR_MODEL_CORE=${EXECUTOR_MODEL_CORE}
GUEST_EXECUTOR_VCPU=${GUEST_EXECUTOR_VCPU}
HOUSEKEEPING_MODEL_CORE=${HOUSEKEEPING_OBS_CORE}
GUEST_HOUSEKEEPING_VCPU=${GUEST_HOUSEKEEPING_VCPU}
TURBOSTAT_INTERVAL=${TURBOSTAT_INTERVAL}
FIXED_FREQ_GHZ=${FIXED_FREQ_GHZ}
SYS_SCALE=${SYS_SCALE}
GEMMINI_FULL_UTIL_POWER_W=${GEMMINI_FULL_UTIL_POWER_W}
GEMMINI_UTIL_THRESHOLD_PCT=${GEMMINI_UTIL_THRESHOLD_PCT}
TIME_SCALE_MODE=${TIME_SCALE_MODE}
GEMMINI_FREQUENCY_MHZ=${GEMMINI_FREQUENCY_MHZ}
CPU_TIME_SCALE=${CPU_TIME_SCALE}
EOF
    mv "${request_tmp}" "${request_file}"
    : > "${RUN_DIR}/request.ready"
}

run_benchmark() {
    mkdir -p "${RUN_DIR}/app_output"
    cd "${WORKDIR}"

    if [[ "${APP_KIND}" == "transformer" ]]; then
        sudo -n taskset -c "${GUEST_MAIN_VCPU}" \
            stdbuf -oL -eL "${BENCH_BIN}" \
            --limit "${SAMPLE_LIMIT}" \
            --threads "${NUM_THREADS}" \
            --vcpu-offset "${GUEST_MAIN_VCPU}" \
            --gemmini-vcpu "${GUEST_EXECUTOR_VCPU}" \
            --output-dir "${RUN_DIR}/app_output" \
            "${EXTRA_ARGS[@]}" \
            --timeline
    else
        taskset -c "${GUEST_MAIN_VCPU}" \
            stdbuf -oL -eL "${BENCH_BIN}" \
            --reps "${REPS}" \
            --output-dir "${RUN_DIR}/app_output" \
            --vcpu-offset "${GUEST_MAIN_VCPU}" \
            --gemmini-vcpu "${GUEST_EXECUTOR_VCPU}"
    fi
}

main() {
    local bench_status
    local model_args

    parse_args "$@"
    configure_app
    validate_config
    ensure_guest_timezone

    [[ -d "${RUN_ROOT}" && -w "${RUN_ROOT}" ]] || \
        die "host collector must create writable run root first: ${RUN_ROOT}"
    [[ -d "${RESULT_ROOT}" && -w "${RESULT_ROOT}" ]] || \
        die "host collector must create writable result root first: ${RESULT_ROOT}"

    RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)_${APP}_$$}"
    [[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "run ID contains unsupported characters"
    RUN_DIR="${RUN_ROOT}/${RUN_ID}"
    RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
    [[ ! -e "${RESULT_DIR}" ]] || die "result directory already exists: ${RESULT_DIR}"
    mkdir "${RUN_DIR}"
    trap cleanup EXIT INT TERM
    mkdir "${RESULT_DIR}"
    configure_guest_housekeeping

    MAIN_MODEL_CORE=$((VCPU_BASE_CORE + GUEST_MAIN_VCPU))
    HOUSEKEEPING_OBS_CORE=$((VCPU_BASE_CORE + GUEST_HOUSEKEEPING_VCPU))
    EXECUTOR_MODEL_CORE=$((VCPU_BASE_CORE + GUEST_EXECUTOR_VCPU))
    write_request

    echo "[guest] waiting for host collector: ${RUN_ID}"
    wait_for_file "${RUN_DIR}/host.ready" "${HOST_READY_TIMEOUT_S}" "${RUN_DIR}/host.failed" || \
        die "host collector did not become ready"

    model_args=(
        --out-log "${RESULT_DIR}/raw_power.csv"
        --turbostat "${RUN_DIR}/raw_turbostat.log"
        --gemmini-cores "${HOST_GEMMINI_CORE}"
        --gemmini-full-util-power-w "${GEMMINI_FULL_UTIL_POWER_W}"
        --gemmini-util-threshold-pct "${GEMMINI_UTIL_THRESHOLD_PCT}"
        --raw-log "${RUN_DIR}/raw_guest_proc_stat.csv"
        --vcpu-map "${MAIN_MODEL_CORE}:${GUEST_MAIN_VCPU}"
        --raw-only-vcpu-map "${HOUSEKEEPING_OBS_CORE}:${GUEST_HOUSEKEEPING_VCPU}"
        --raw-only-vcpu-map "${EXECUTOR_MODEL_CORE}:${GUEST_EXECUTOR_VCPU}"
        --fixed-freq-ghz "${FIXED_FREQ_GHZ}"
        --sys-scale "${SYS_SCALE}"
        --interval "${VCPU_SAMPLE_INTERVAL}"
    )
    PYTHONDONTWRITEBYTECODE=1 taskset -c "${GUEST_HOUSEKEEPING_VCPU}" \
        python3 "${COMBINED_POWER_SIM}" "${model_args[@]}" \
        > "${RUN_DIR}/raw_power_sim.log" 2>&1 &
    POWER_MONITOR_PID=$!

    sleep 2
    kill -0 "${POWER_MONITOR_PID}" 2>/dev/null || \
        die "guest combined power collector failed to start"
    {
        echo "[HOST_TS_REALTIME] BENCH_HOST_START $(date +%s.%N)"
        echo "[KST_TS] BENCH_HOST_START $(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
        echo "[HOST_TS_LINE] $(date +%s.%N) [remote] app=${APP} mode=gemmini"
        echo "[HOST_TS_LINE] $(date +%s.%N) [launch-affinity] application_vcpu=${GUEST_MAIN_VCPU} executor_vcpu=${GUEST_EXECUTOR_VCPU} housekeeping_vcpu=${GUEST_HOUSEKEEPING_VCPU} guest_housekeeping_isolated=1 gemmini_call_mode=executor"
    } > "${RUN_DIR}/raw_bench.log"

    set +e
    run_benchmark 2>&1 | while IFS= read -r line; do
        printf '[HOST_TS_LINE] %s %s\n' "$(date +%s.%N)" "${line}"
    done >> "${RUN_DIR}/raw_bench.log"
    bench_status=${PIPESTATUS[0]}
    set -e
    echo "[HOST_TS_REALTIME] BENCH_HOST_END $(date +%s.%N)" >> "${RUN_DIR}/raw_bench.log"
    echo "[KST_TS] BENCH_HOST_END $(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" >> "${RUN_DIR}/raw_bench.log"
    (( bench_status == 0 )) || die "benchmark exited with status ${bench_status}"

    sleep 2
    kill -0 "${POWER_MONITOR_PID}" 2>/dev/null || \
        die "guest combined power collector stopped unexpectedly"
    stop_guest_collectors
    csv_has_data_row "${RUN_DIR}/raw_guest_proc_stat.csv" || \
        die "guest /proc/stat CSV has no data rows"
    csv_has_data_row "${RESULT_DIR}/raw_power.csv" || \
        die "combined power CSV has no data rows"
    : > "${RUN_DIR}/guest.done"

    echo "[guest] waiting for host plot..."
    if ! wait_for_file "${RUN_DIR}/host.done" "${HOST_DONE_TIMEOUT_S}" "${RUN_DIR}/host.failed"; then
        die "host collector failed or timed out"
    fi

    SUCCESS=1
    echo "[guest] complete"
    echo "[guest] final timezone: $(timedatectl show --property=Timezone --value)"
    echo "[guest] shared run dir: ${RUN_DIR}"
    echo "[guest] model results: ${RESULT_DIR}"
}

main "$@"
