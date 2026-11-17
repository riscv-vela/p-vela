#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DUAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
POWER_MODELING_ROOT="$(cd -- "${DUAL_ROOT}/.." && pwd)"
source "${POWER_MODELING_ROOT}/power_sim/power_model_defaults.env"
GEMMINI_EXPERIMENTS_ROOT="$(cd -- "${POWER_MODELING_ROOT}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${GEMMINI_EXPERIMENTS_ROOT}/.." && pwd)"
QEMU_BOOT_DIR="${WORKSPACE_ROOT}/qemu_boot"
QEMU_LOG="${WORKSPACE_ROOT}/q-vela/qemu/qemu-gemmini.log"

PLOT_SCRIPT="${SCRIPT_DIR}/plot_dual_app_gemmini_power.py"
POWER_SIM="${POWER_MODELING_ROOT}/power_sim/power_sim_gemmini.py"
MERGE_POWER="${POWER_MODELING_ROOT}/power_sim/merge_split_power_csv.py"
TIME_SCALE_UTIL="${POWER_MODELING_ROOT}/power_sim/time_scale_gemmini_logs.py"
GUEST_VCPU_POWER_SIM="/mnt/hostshare/power_modeling/power_sim/power_sim_vcpu_procstat.py"
GUEST_DUAL_ROOT="/mnt/hostshare/power_modeling/dual_app_power_modeling"

SSH_KEY="${WORKSPACE_ROOT}/q-vela/tmp-cloudinit/id_ed25519"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_HOST="${SSH_HOST:-127.0.0.1}"
SSH_PORT="${SSH_PORT:-2200}"

TURBOSTAT_INTERVAL="${TURBOSTAT_INTERVAL:-${DEFAULT_TURBOSTAT_INTERVAL_S}}"
POWER_POLL_INTERVAL="${POWER_POLL_INTERVAL:-${DEFAULT_POWER_POLL_INTERVAL_S}}"
VCPU_SAMPLE_INTERVAL="${VCPU_SAMPLE_INTERVAL:-${DEFAULT_VCPU_SAMPLE_INTERVAL_S}}"
SCALED_PLOT_WINDOW_MARGIN_S="${SCALED_PLOT_WINDOW_MARGIN_S:-${DEFAULT_SCALED_PLOT_WINDOW_MARGIN_S}}"
SCALED_PLOT_FIG_WIDTH="${SCALED_PLOT_FIG_WIDTH:-${DEFAULT_SCALED_PLOT_FIG_WIDTH}}"
PLOT_TIME_MAJOR_TICK_S="${PLOT_TIME_MAJOR_TICK_S:-${DEFAULT_PLOT_TIME_MAJOR_TICK_S}}"
TIME_SCALE_MODE="${TIME_SCALE_MODE:-${DEFAULT_TIME_SCALE_MODE}}"
GEMMINI_FREQUENCY_MHZ="${GEMMINI_FREQUENCY_MHZ:-${DEFAULT_GEMMINI_FREQUENCY_MHZ}}"
FIXED_FREQ_GHZ="${FIXED_FREQ_GHZ:-${DEFAULT_FIXED_FREQ_GHZ}}"
CPU_TIME_SCALE="${CPU_TIME_SCALE:-${DEFAULT_CPU_TIME_SCALE}}"
SYS_SCALE="${SYS_SCALE:-${DEFAULT_SYS_SCALE}}"
GEMMINI_FULL_UTIL_POWER_W="${GEMMINI_FULL_UTIL_POWER_W:-${DEFAULT_GEMMINI_FULL_UTIL_POWER_W}}"
GEMMINI_UTIL_THRESHOLD_PCT="${GEMMINI_UTIL_THRESHOLD_PCT:-${DEFAULT_GEMMINI_UTIL_THRESHOLD_PCT}}"
AUTO_RESOLVE_HOST_CORES="${AUTO_RESOLVE_HOST_CORES:-1}"
GEMMINI_DISPATCH_POLICY="${GEMMINI_DISPATCH_POLICY:-modulo}"
PIN_QEMU_BEFORE_RUN="${PIN_QEMU_BEFORE_RUN:-0}"
QEMU_VCPU_BASE_CORE="${QEMU_VCPU_BASE_CORE:-8}"
QEMU_GEMMINI_BASE_CORE="${QEMU_GEMMINI_BASE_CORE:-2}"
BUILD_BEFORE_RUN="${BUILD_BEFORE_RUN:-1}"

APP0=""
APP1=""
SAMPLE_LIMIT="1"
NUM_THREADS="1"
POLYBENCH_DATASET="${POLYBENCH_DATASET:-MEDIUM}"
POLYBENCH_REPS="${POLYBENCH_REPS:-50}"
POLYBENCH_DATASET_MACRO=""

GUEST_HOUSEKEEPING_VCPU="${GUEST_HOUSEKEEPING_VCPU:-0}"
GUEST_APP0_MAIN_VCPU="${GUEST_APP0_MAIN_VCPU:-1}"
GUEST_APP1_MAIN_VCPU="${GUEST_APP1_MAIN_VCPU:-2}"
GUEST_APP0_GEMMINI_VCPU="${GUEST_APP0_GEMMINI_VCPU:-3}"
GUEST_APP1_GEMMINI_VCPU="${GUEST_APP1_GEMMINI_VCPU:-4}"

HOUSEKEEPING_VCPU_CORE="${HOUSEKEEPING_VCPU_CORE:-8}"
APP0_MAIN_VCPU_CORE="${APP0_MAIN_VCPU_CORE:-9}"
APP1_MAIN_VCPU_CORE="${APP1_MAIN_VCPU_CORE:-10}"
APP0_GEMMINI_VCPU_CORE="${APP0_GEMMINI_VCPU_CORE:-11}"
APP1_GEMMINI_VCPU_CORE="${APP1_GEMMINI_VCPU_CORE:-12}"
APP0_GEMMINI_CORE="${APP0_GEMMINI_CORE:-3}"
APP1_GEMMINI_CORE="${APP1_GEMMINI_CORE:-2}"
HOST_LOGGER_CORE="${HOST_LOGGER_CORE:-7}"
PIN_HOST_LOGGER_TO_HOUSEKEEPING="${PIN_HOST_LOGGER_TO_HOUSEKEEPING:-1}"
PIN_REMOTE_WRAPPER_TO_HOUSEKEEPING="${PIN_REMOTE_WRAPPER_TO_HOUSEKEEPING:-1}"

APP0_GEMMINI_WORKER_INDEX="${APP0_GEMMINI_WORKER_INDEX:-}"
APP1_GEMMINI_WORKER_INDEX="${APP1_GEMMINI_WORKER_INDEX:-}"
QEMU_VCPU_COUNT=""
QEMU_GEMMINI_COUNT=""
QEMU_PIN_OUTPUT=""

APP0_ID=""
APP0_NAME=""
APP0_KIND=""
APP0_SRC_DIR=""
APP0_GUEST_WORKDIR=""
APP0_BENCH_BIN=""
APP0_GUEST_OUTPUT=""
APP0_EXTRA_ARGS=""

APP1_ID=""
APP1_NAME=""
APP1_KIND=""
APP1_SRC_DIR=""
APP1_GUEST_WORKDIR=""
APP1_BENCH_BIN=""
APP1_GUEST_OUTPUT=""
APP1_EXTRA_ARGS=""

CURRENT_TURBOSTAT_PID=""
CURRENT_POWER_PID=""
PLOT_POWER_TMP=""
PLOT_UTIL_TMP=""
GUEST_RUN_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") --app0 APP --app1 APP [options]
       $(basename "$0") --apps APP0,APP1 [options]

Run one dual Gemmini power experiment from any two single apps.

Apps:
  distilbert
  distilroberta
  distilgpt2
  polybench_gemm

Required QEMU setup:
  QEMU_GEMMINI_POLICY=modulo ${QEMU_BOOT_DIR}/run_qemu.sh 2 5
  ${QEMU_BOOT_DIR}/pin.sh 8 2

Common options:
  --sample-limit N
  --threads N
  --app0-main-vcpu-core N
  --app1-main-vcpu-core N
  --app0-gemmini-vcpu-core N
  --app1-gemmini-vcpu-core N
  --app0-gemmini-core N
  --app1-gemmini-core N
  --sys-scale FLOAT
  --gemmini-full-util-power-w FLOAT
  --gemmini-util-threshold-pct FLOAT
  --build 0|1

PolyBench options:
  --dataset MINI|SMALL|MEDIUM|LARGE|EXTRALARGE
  --reps N

Examples:
  $(basename "$0") --apps distilbert,distilroberta --sample-limit 1
  $(basename "$0") --app0 distilgpt2 --app1 polybench_gemm --dataset MEDIUM --reps 50
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        die "${name} must be a positive integer"
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        die "${name} must be a non-negative integer"
    fi
}

require_number() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
        die "${name} must be a number"
    fi
}

dataset_macro() {
    case "$1" in
        MINI|MINI_DATASET) echo "MINI_DATASET" ;;
        SMALL|SMALL_DATASET) echo "SMALL_DATASET" ;;
        MEDIUM|MEDIUM_DATASET) echo "MEDIUM_DATASET" ;;
        LARGE|LARGE_DATASET) echo "LARGE_DATASET" ;;
        EXTRALARGE|EXTRALARGE_DATASET) echo "EXTRALARGE_DATASET" ;;
        *) die "unknown PolyBench dataset: $1" ;;
    esac
}

normalize_app() {
    case "$1" in
        distilbert|bert) echo "distilbert" ;;
        distilroberta|roberta) echo "distilroberta" ;;
        distilgpt2|gpt2) echo "distilgpt2" ;;
        polybench_gemm|polybench|gemm|polybgemm|polybenchgemm) echo "polybench_gemm" ;;
        *) die "unsupported app: $1" ;;
    esac
}

set_pair_from_value() {
    local value="$1"
    case "${value}" in
        *,*)
            APP0="${value%%,*}"
            APP1="${value#*,}"
            ;;
        distilbert_distilroberta|distilbert-distilroberta)
            APP0="distilbert"
            APP1="distilroberta"
            ;;
        distilbert_distilgpt2|distilbert-distilgpt2)
            APP0="distilbert"
            APP1="distilgpt2"
            ;;
        distilbert_polybench_gemm|distilbert-polybench-gemm|distilbert_polybgemm)
            APP0="distilbert"
            APP1="polybench_gemm"
            ;;
        distilroberta_distilgpt2|distilroberta-distilgpt2)
            APP0="distilroberta"
            APP1="distilgpt2"
            ;;
        distilroberta_polybench_gemm|distilroberta-polybench-gemm|distilroberta_polybgemm)
            APP0="distilroberta"
            APP1="polybench_gemm"
            ;;
        distilgpt2_polybench_gemm|distilgpt2-polybench-gemm|distilgpt2_polybgemm)
            APP0="distilgpt2"
            APP1="polybench_gemm"
            ;;
        *)
            die "--apps expects APP0,APP1 or a known pair id"
            ;;
    esac
}

configure_slot() {
    local slot="$1"
    local raw_app="$2"
    local id name kind src_dir guest_workdir bench_bin guest_output extra_args

    id="$(normalize_app "${raw_app}")"
    case "${id}" in
        distilbert)
            name="DistilBERT"
            kind="transformer"
            src_dir="${GEMMINI_EXPERIMENTS_ROOT}/applications/distilbert/sst2_pipeline/guest"
            guest_workdir="/mnt/hostshare/applications/distilbert/sst2_pipeline/guest"
            bench_bin="./distilbert_gemmini_power_infer.riscv"
            guest_output="/mnt/hostshare/applications/distilbert/sst2_pipeline/outputs/gemmini"
            extra_args=""
            ;;
        distilroberta)
            name="DistilRoBERTa"
            kind="transformer"
            src_dir="${GEMMINI_EXPERIMENTS_ROOT}/applications/distilroberta/sst2_pipeline/guest"
            guest_workdir="/mnt/hostshare/applications/distilroberta/sst2_pipeline/guest"
            bench_bin="./distilroberta_gemmini_power_infer.riscv"
            guest_output="/mnt/hostshare/applications/distilroberta/sst2_pipeline/outputs/gemmini"
            extra_args=""
            ;;
        distilgpt2)
            name="DistilGPT2"
            kind="transformer"
            src_dir="${GEMMINI_EXPERIMENTS_ROOT}/applications/distilgpt2/sst2_pipeline/guest"
            guest_workdir="/mnt/hostshare/applications/distilgpt2/sst2_pipeline/guest"
            bench_bin="./distilgpt2_gemmini_power_infer.riscv"
            guest_output="/mnt/hostshare/applications/distilgpt2/sst2_pipeline/outputs/gemmini"
            extra_args="--no-host-compare"
            ;;
        polybench_gemm)
            name="PolyBench GEMM"
            kind="polybench_gemm"
            src_dir="${GEMMINI_EXPERIMENTS_ROOT}/applications/polybench_gemm/guest"
            guest_workdir="/mnt/hostshare/applications/polybench_gemm/guest"
            bench_bin="./polybench_gemm_gemmini_power.riscv"
            guest_output="/mnt/hostshare/applications/polybench_gemm/outputs/gemmini"
            extra_args=""
            ;;
    esac

    printf -v "${slot}_ID" '%s' "${id}"
    printf -v "${slot}_NAME" '%s' "${name}"
    printf -v "${slot}_KIND" '%s' "${kind}"
    printf -v "${slot}_SRC_DIR" '%s' "${src_dir}"
    printf -v "${slot}_GUEST_WORKDIR" '%s' "${guest_workdir}"
    printf -v "${slot}_BENCH_BIN" '%s' "${bench_bin}"
    printf -v "${slot}_GUEST_OUTPUT" '%s' "${guest_output}"
    printf -v "${slot}_EXTRA_ARGS" '%s' "${extra_args}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --apps|--app)
                [[ $# -ge 2 ]] || die "missing value for $1"
                set_pair_from_value "$2"
                shift 2
                ;;
            --app0|--first-app)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP0="$2"
                shift 2
                ;;
            --app1|--second-app)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP1="$2"
                shift 2
                ;;
            --sample-limit|--limit)
                [[ $# -ge 2 ]] || die "missing value for $1"
                SAMPLE_LIMIT="$2"
                shift 2
                ;;
            --threads)
                [[ $# -ge 2 ]] || die "missing value for --threads"
                NUM_THREADS="$2"
                shift 2
                ;;
            --app0-main-vcpu-core|--main0-vcpu-core)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP0_MAIN_VCPU_CORE="$2"
                shift 2
                ;;
            --app1-main-vcpu-core|--main1-vcpu-core)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP1_MAIN_VCPU_CORE="$2"
                shift 2
                ;;
            --app0-gemmini-vcpu-core|--gemmini0-vcpu-core)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP0_GEMMINI_VCPU_CORE="$2"
                shift 2
                ;;
            --app1-gemmini-vcpu-core|--gemmini1-vcpu-core)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP1_GEMMINI_VCPU_CORE="$2"
                shift 2
                ;;
            --app0-gemmini-core|--app0-gemmini-worker-core|--gemmini0-core)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP0_GEMMINI_CORE="$2"
                shift 2
                ;;
            --app1-gemmini-core|--app1-gemmini-worker-core|--gemmini1-core)
                [[ $# -ge 2 ]] || die "missing value for $1"
                APP1_GEMMINI_CORE="$2"
                shift 2
                ;;
            --housekeeping-vcpu-core)
                [[ $# -ge 2 ]] || die "missing value for --housekeeping-vcpu-core"
                HOUSEKEEPING_VCPU_CORE="$2"
                shift 2
                ;;
            --sys-scale)
                [[ $# -ge 2 ]] || die "missing value for --sys-scale"
                SYS_SCALE="$2"
                shift 2
                ;;
            --gemmini-full-util-power-w)
                [[ $# -ge 2 ]] || die "missing value for --gemmini-full-util-power-w"
                GEMMINI_FULL_UTIL_POWER_W="$2"
                shift 2
                ;;
            --gemmini-util-threshold-pct)
                [[ $# -ge 2 ]] || die "missing value for --gemmini-util-threshold-pct"
                GEMMINI_UTIL_THRESHOLD_PCT="$2"
                shift 2
                ;;
            --dataset|--polybench-dataset)
                [[ $# -ge 2 ]] || die "missing value for $1"
                POLYBENCH_DATASET="$2"
                shift 2
                ;;
            --reps|--polybench-reps)
                [[ $# -ge 2 ]] || die "missing value for $1"
                POLYBENCH_REPS="$2"
                shift 2
                ;;
            --build)
                [[ $# -ge 2 ]] || die "missing value for --build"
                BUILD_BEFORE_RUN="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            -*)
                die "unknown option: $1"
                ;;
            *)
                if [[ -z "${APP0}" ]]; then
                    APP0="$1"
                elif [[ -z "${APP1}" ]]; then
                    APP1="$1"
                else
                    die "unexpected positional argument: $1"
                fi
                shift
                ;;
        esac
    done
}

cleanup() {
    if [[ -n "${CURRENT_POWER_PID}" ]] && kill -0 "${CURRENT_POWER_PID}" 2>/dev/null; then
        kill "${CURRENT_POWER_PID}" 2>/dev/null || true
        wait "${CURRENT_POWER_PID}" 2>/dev/null || true
    fi

    if [[ -n "${CURRENT_TURBOSTAT_PID}" ]] && kill -0 "${CURRENT_TURBOSTAT_PID}" 2>/dev/null; then
        sudo -n kill "${CURRENT_TURBOSTAT_PID}" 2>/dev/null || true
        wait "${CURRENT_TURBOSTAT_PID}" 2>/dev/null || true
    fi

    if [[ -n "${PLOT_POWER_TMP}" ]]; then
        rm -f "${PLOT_POWER_TMP}"
    fi
    if [[ -n "${PLOT_UTIL_TMP}" ]]; then
        rm -f "${PLOT_UTIL_TMP}"
    fi
}

require_unique_cores() {
    local seen=" "
    local core
    for core in "$@"; do
        if [[ "${seen}" == *" ${core} "* ]]; then
            die "core/vCPU values must be unique, but ${core} was repeated"
        fi
        seen+="${core} "
    done
}

qemu_vcpu_tid() {
    local guest_vcpu="$1"
    sed -n "s/.*qemu_thread_start called for CPU ${guest_vcpu}\/TCG at TID \([0-9][0-9]*\).*/\1/p" "${QEMU_LOG}" |
        tail -1
}

qemu_gemmini_tid() {
    local worker_index="$1"
    sed -n 's/.*qemu_thread_start called for gemmini-worker at TID \([0-9][0-9]*\).*/\1/p' "${QEMU_LOG}" |
        sed -n "$((worker_index + 1))p"
}

tid_affinity_list() {
    local tid="$1"
    taskset -pc "${tid}" 2>/dev/null |
        sed -n "s/.*current affinity list: //p" |
        tail -1
}

tid_single_host_core() {
    local tid="$1"
    local affinity
    affinity="$(tid_affinity_list "${tid}")"
    if [[ "${affinity}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${affinity}"
    fi
}

resolve_vcpu_host_core() {
    local guest_vcpu="$1"
    local role="$2"
    local tid core
    tid="$(qemu_vcpu_tid "${guest_vcpu}")"
    [[ -n "${tid}" ]] || die "could not find QEMU TID for guest vCPU ${guest_vcpu} (${role})"
    core="$(tid_single_host_core "${tid}")"
    if [[ -z "${core}" ]]; then
        echo "QEMU vCPU ${guest_vcpu} (${role}) TID ${tid} is not pinned to one host core." >&2
        echo "Current affinity: $(tid_affinity_list "${tid}")" >&2
        die "run ${QEMU_BOOT_DIR}/pin.sh ${QEMU_VCPU_BASE_CORE} ${QEMU_GEMMINI_BASE_CORE}, or set AUTO_RESOLVE_HOST_CORES=0"
    fi
    printf '%s\n' "${core}"
}

resolve_gemmini_host_core() {
    local worker_index="$1"
    local role="$2"
    local tid core
    tid="$(qemu_gemmini_tid "${worker_index}")"
    [[ -n "${tid}" ]] || die "could not find QEMU Gemmini worker ${worker_index} TID (${role})"
    core="$(tid_single_host_core "${tid}")"
    if [[ -z "${core}" ]]; then
        echo "QEMU Gemmini worker ${worker_index} (${role}) TID ${tid} is not pinned to one host core." >&2
        echo "Current affinity: $(tid_affinity_list "${tid}")" >&2
        die "run ${QEMU_BOOT_DIR}/pin.sh ${QEMU_VCPU_BASE_CORE} ${QEMU_GEMMINI_BASE_CORE}, or set AUTO_RESOLVE_HOST_CORES=0"
    fi
    printf '%s\n' "${core}"
}

require_qemu_topology() {
    local vcpu_count gemmini_count
    [[ -f "${QEMU_LOG}" ]] || die "missing QEMU log: ${QEMU_LOG}; boot with QEMU_GEMMINI_POLICY=modulo ${QEMU_BOOT_DIR}/run_qemu.sh 2 5"

    vcpu_count="$(
        sed -n 's/.*qemu_thread_start called for CPU \([0-9][0-9]*\)\/TCG at TID.*/\1/p' "${QEMU_LOG}" |
            sort -n -u |
            wc -l
    )"
    gemmini_count="$(
        sed -n 's/.*qemu_thread_start called for gemmini-worker at TID.*/gemmini/p' "${QEMU_LOG}" |
            wc -l
    )"
    QEMU_VCPU_COUNT="${vcpu_count}"
    QEMU_GEMMINI_COUNT="${gemmini_count}"

    if (( vcpu_count < 5 || gemmini_count < 2 )); then
        echo "parsed vCPU threads    : ${vcpu_count}" >&2
        echo "parsed Gemmini workers : ${gemmini_count}" >&2
        die "current QEMU topology is not dual-workload ready; reboot with QEMU_GEMMINI_POLICY=modulo ${QEMU_BOOT_DIR}/run_qemu.sh 2 5"
    fi
}

pin_qemu_threads_if_requested() {
    if [[ "${PIN_QEMU_BEFORE_RUN}" != "1" ]]; then
        QEMU_PIN_OUTPUT="pin_qemu_before_run=${PIN_QEMU_BEFORE_RUN}; skipped"
        return
    fi
    [[ -f "${QEMU_BOOT_DIR}/pin.sh" ]] || die "missing QEMU pin script: ${QEMU_BOOT_DIR}/pin.sh"
    echo "Pinning QEMU threads with: ${QEMU_BOOT_DIR}/pin.sh ${QEMU_VCPU_BASE_CORE} ${QEMU_GEMMINI_BASE_CORE}"
    if ! QEMU_PIN_OUTPUT="$(bash "${QEMU_BOOT_DIR}/pin.sh" "${QEMU_VCPU_BASE_CORE}" "${QEMU_GEMMINI_BASE_CORE}" 2>&1)"; then
        printf '%s\n' "${QEMU_PIN_OUTPUT}" >&2
        exit 1
    fi
    printf '%s\n' "${QEMU_PIN_OUTPUT}"
}

resolve_model_gemmini_workers() {
    [[ -n "${QEMU_GEMMINI_COUNT}" && "${QEMU_GEMMINI_COUNT}" != "0" ]] || die "QEMU_GEMMINI_COUNT was not resolved"
    case "${GEMMINI_DISPATCH_POLICY}" in
        modulo)
            APP0_GEMMINI_WORKER_INDEX="$((GUEST_APP0_GEMMINI_VCPU % QEMU_GEMMINI_COUNT))"
            APP1_GEMMINI_WORKER_INDEX="$((GUEST_APP1_GEMMINI_VCPU % QEMU_GEMMINI_COUNT))"
            ;;
        manual)
            [[ -n "${APP0_GEMMINI_WORKER_INDEX}" && -n "${APP1_GEMMINI_WORKER_INDEX}" ]] ||
                die "manual Gemmini dispatch requires APP0_GEMMINI_WORKER_INDEX and APP1_GEMMINI_WORKER_INDEX"
            ;;
        *)
            die "unsupported GEMMINI_DISPATCH_POLICY=${GEMMINI_DISPATCH_POLICY}; use modulo or manual"
            ;;
    esac

    if (( APP0_GEMMINI_WORKER_INDEX < 0 || APP0_GEMMINI_WORKER_INDEX >= QEMU_GEMMINI_COUNT )); then
        die "${APP0_ID} Gemmini worker index ${APP0_GEMMINI_WORKER_INDEX} is out of range"
    fi
    if (( APP1_GEMMINI_WORKER_INDEX < 0 || APP1_GEMMINI_WORKER_INDEX >= QEMU_GEMMINI_COUNT )); then
        die "${APP1_ID} Gemmini worker index ${APP1_GEMMINI_WORKER_INDEX} is out of range"
    fi
    [[ "${APP0_GEMMINI_WORKER_INDEX}" != "${APP1_GEMMINI_WORKER_INDEX}" ]] ||
        die "${APP0_ID} and ${APP1_ID} resolved to the same Gemmini worker ${APP0_GEMMINI_WORKER_INDEX}"
}

resolve_host_core_mapping() {
    resolve_model_gemmini_workers
    if [[ "${AUTO_RESOLVE_HOST_CORES}" != "1" ]]; then
        return
    fi

    HOUSEKEEPING_VCPU_CORE="$(resolve_vcpu_host_core "${GUEST_HOUSEKEEPING_VCPU}" "housekeeping")"
    APP0_MAIN_VCPU_CORE="$(resolve_vcpu_host_core "${GUEST_APP0_MAIN_VCPU}" "${APP0_ID} main")"
    APP1_MAIN_VCPU_CORE="$(resolve_vcpu_host_core "${GUEST_APP1_MAIN_VCPU}" "${APP1_ID} main")"
    APP0_GEMMINI_VCPU_CORE="$(resolve_vcpu_host_core "${GUEST_APP0_GEMMINI_VCPU}" "${APP0_ID} Gemmini executor")"
    APP1_GEMMINI_VCPU_CORE="$(resolve_vcpu_host_core "${GUEST_APP1_GEMMINI_VCPU}" "${APP1_ID} Gemmini executor")"
    APP0_GEMMINI_CORE="$(resolve_gemmini_host_core "${APP0_GEMMINI_WORKER_INDEX}" "${APP0_ID} Gemmini")"
    APP1_GEMMINI_CORE="$(resolve_gemmini_host_core "${APP1_GEMMINI_WORKER_INDEX}" "${APP1_ID} Gemmini")"
}

build_slot_if_needed() {
    local slot="$1"
    local kind_var="${slot}_KIND"
    local src_var="${slot}_SRC_DIR"
    local id_var="${slot}_ID"
    local kind="${!kind_var}"
    local src_dir="${!src_var}"
    local id="${!id_var}"

    if [[ "${BUILD_BEFORE_RUN}" != "1" ]]; then
        return 0
    fi
    [[ -d "${src_dir}" ]] || die "missing guest source dir for ${id}: ${src_dir}"

    if [[ "${kind}" == "polybench_gemm" ]]; then
        make -B -C "${src_dir}" gemmini-power DATASET="${POLYBENCH_DATASET_MACRO}"
    else
        make -C "${src_dir}" power
    fi
}

pin_guest_launch_services() {
    local required_max_vcpu vcpu
    required_max_vcpu="${GUEST_HOUSEKEEPING_VCPU}"
    for vcpu in "${GUEST_APP0_MAIN_VCPU}" "${GUEST_APP1_MAIN_VCPU}" "${GUEST_APP0_GEMMINI_VCPU}" "${GUEST_APP1_GEMMINI_VCPU}"; do
        if (( vcpu > required_max_vcpu )); then
            required_max_vcpu="${vcpu}"
        fi
    done

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
        bash -s -- \
        "${GUEST_HOUSEKEEPING_VCPU}" \
        "${required_max_vcpu}" \
        "${APP0_ID}" \
        "${APP1_ID}" \
        "${GUEST_APP0_MAIN_VCPU}" \
        "${GUEST_APP1_MAIN_VCPU}" \
        "${GUEST_APP0_GEMMINI_VCPU}" \
        "${GUEST_APP1_GEMMINI_VCPU}" <<\REMOTE_AFFINITY
set -euo pipefail

housekeeping_vcpu="$1"
required_max_vcpu="$2"
app0_id="$3"
app1_id="$4"
app0_main_vcpu="$5"
app1_main_vcpu="$6"
app0_gemmini_vcpu="$7"
app1_gemmini_vcpu="$8"

ncpu=$(getconf _NPROCESSORS_ONLN)
if [ "${required_max_vcpu}" -ge "${ncpu}" ]; then
    echo "[guest-launch-affinity-error] required_max_vcpu=${required_max_vcpu} ncpu=${ncpu}"
    exit 2
fi

housekeeping_mask=$(printf "%x" $((1 << housekeeping_vcpu)))
sudo -n env \
    HOUSEKEEPING_CPU="${housekeeping_vcpu}" \
    HOUSEKEEPING_MASK="${housekeeping_mask}" \
    bash -c '
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
            [ -e "${affinity}" ] || continue
            if printf "%s\n" "${HOUSEKEEPING_CPU}" > "${affinity}" 2>/dev/null; then
                irq_moved=$((irq_moved + 1))
            else
                irq_failed=$((irq_failed + 1))
            fi
        done

        for cpumask in /sys/devices/virtual/workqueue/cpumask /sys/devices/virtual/workqueue/*/cpumask; do
            [ -e "${cpumask}" ] || continue
            if printf "%s\n" "${HOUSEKEEPING_MASK}" > "${cpumask}" 2>/dev/null; then
                workqueue_moved=$((workqueue_moved + 1))
            else
                workqueue_failed=$((workqueue_failed + 1))
            fi
        done

        for proc_dir in /proc/[0-9]*; do
            pid=${proc_dir##*/}
            if taskset -apc "${HOUSEKEEPING_CPU}" "${pid}" >/dev/null 2>&1; then
                process_moved=$((process_moved + 1))
            else
                process_failed=$((process_failed + 1))
            fi
        done

        for proc_dir in /proc/[0-9]*; do
            [ -e "${proc_dir}/exe" ] || continue
            allowed=$(sed -n "s/^Cpus_allowed_list:[[:space:]]*//p" "${proc_dir}/status" 2>/dev/null || true)
            if [ -n "${allowed}" ] && [ "${allowed}" != "${HOUSEKEEPING_CPU}" ]; then
                userspace_violations=$((userspace_violations + 1))
                printf "[guest-housekeeping-warning] pid=%s allowed=%s comm=%s\n" \
                    "${proc_dir##*/}" "${allowed}" "$(cat "${proc_dir}/comm" 2>/dev/null || true)"
            fi
        done

        printf "[guest-housekeeping] cpu=%s mask=%s default_irq=%s irq_moved=%d irq_failed=%d workqueue_moved=%d workqueue_failed=%d process_moved=%d process_failed=%d userspace_violations=%d\n" \
            "${HOUSEKEEPING_CPU}" "${HOUSEKEEPING_MASK}" "${default_irq_status}" \
            "${irq_moved}" "${irq_failed}" "${workqueue_moved}" "${workqueue_failed}" \
            "${process_moved}" "${process_failed}" "${userspace_violations}"
    '

echo "[guest-launch-affinity] app0=${app0_id} app1=${app1_id} app0_main=${app0_main_vcpu} app1_main=${app1_main_vcpu} app0_gemmini=${app0_gemmini_vcpu} app1_gemmini=${app1_gemmini_vcpu} housekeeping_vcpu=${housekeeping_vcpu}"
REMOTE_AFFINITY
}

run_remote_bench_raw() {
    {
        printf "[%s] [timeline] actor=vcpu label=ssh_invoke sample=0 thread=0 cpu=-1\n" "${APP0_ID}"
        printf "[%s] [timeline] actor=vcpu label=ssh_invoke sample=0 thread=0 cpu=-1\n" "${APP1_ID}"
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
            bash -s -- \
            "${SAMPLE_LIMIT}" \
            "${NUM_THREADS}" \
            "${POLYBENCH_REPS}" \
            "${GUEST_HOUSEKEEPING_VCPU}" \
            "${PIN_REMOTE_WRAPPER_TO_HOUSEKEEPING}" \
            "${APP0_ID}" \
            "${APP0_KIND}" \
            "${APP0_GUEST_WORKDIR}" \
            "${APP0_BENCH_BIN}" \
            "${APP0_GUEST_OUTPUT}" \
            "${GUEST_APP0_MAIN_VCPU}" \
            "${GUEST_APP0_GEMMINI_VCPU}" \
            "${APP0_EXTRA_ARGS:-__NO_EXTRA_ARGS__}" \
            "${APP1_ID}" \
            "${APP1_KIND}" \
            "${APP1_GUEST_WORKDIR}" \
            "${APP1_BENCH_BIN}" \
            "${APP1_GUEST_OUTPUT}" \
            "${GUEST_APP1_MAIN_VCPU}" \
            "${GUEST_APP1_GEMMINI_VCPU}" \
            "${APP1_EXTRA_ARGS:-__NO_EXTRA_ARGS__}" \
            "${GUEST_RUN_DIR}" \
            "${GUEST_VCPU_POWER_SIM}" \
            "${FIXED_FREQ_GHZ}" \
            "${SYS_SCALE}" \
            "${VCPU_SAMPLE_INTERVAL}" \
            "${HOUSEKEEPING_VCPU_CORE}" \
            "${APP0_MAIN_VCPU_CORE}" \
            "${APP1_MAIN_VCPU_CORE}" \
            "${APP0_GEMMINI_VCPU_CORE}" \
            "${APP1_GEMMINI_VCPU_CORE}" <<\REMOTE_EOF
set -euo pipefail

umask 000

sample_limit="$1"
num_threads="$2"
polybench_reps="$3"
housekeeping_vcpu="$4"
pin_remote_wrapper_to_housekeeping="$5"
app0_id="$6"
app0_kind="$7"
app0_workdir="$8"
app0_bin="$9"
app0_output="${10}"
app0_main_vcpu="${11}"
app0_gemmini_vcpu="${12}"
app0_extra_args="${13}"
app1_id="${14}"
app1_kind="${15}"
app1_workdir="${16}"
app1_bin="${17}"
app1_output="${18}"
app1_main_vcpu="${19}"
app1_gemmini_vcpu="${20}"
app1_extra_args="${21}"
guest_run_dir="${22}"
vcpu_power_sim="${23}"
fixed_freq_ghz="${24}"
sys_scale="${25}"
vcpu_sample_interval="${26}"
housekeeping_host_core="${27}"
app0_main_host_core="${28}"
app1_main_host_core="${29}"
app0_executor_host_core="${30}"
app1_executor_host_core="${31}"

if [ "${app0_extra_args}" = "__NO_EXTRA_ARGS__" ]; then
    app0_extra_args=""
fi
if [ "${app1_extra_args}" = "__NO_EXTRA_ARGS__" ]; then
    app1_extra_args=""
fi

printf "[%s] [timeline] actor=vcpu label=ssh_ready sample=0 thread=0 cpu=-1\n" "${app0_id}"
printf "[%s] [timeline] actor=vcpu label=ssh_ready sample=0 thread=0 cpu=-1\n" "${app1_id}"

sudo -n mkdir -p /mnt/hostshare
if ! mountpoint -q /mnt/hostshare; then
    sudo -n mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/hostshare
fi

online_vcpus="$(getconf _NPROCESSORS_ONLN)"
if [ "${online_vcpus}" -ne 5 ]; then
    echo "[guest-collector-error] expected 5 vCPUs, found ${online_vcpus}"
    exit 2
fi
if [ "${num_threads}" -ne 1 ]; then
    echo "[guest-collector-error] dual split mode requires --threads 1"
    exit 2
fi
if [ ! -f "${vcpu_power_sim}" ]; then
    echo "[guest-collector-error] missing ${vcpu_power_sim}"
    exit 2
fi
if [ ! -d "${guest_run_dir}" ] || [ ! -w "${guest_run_dir}" ]; then
    echo "[guest-collector-error] shared run directory is not writable: ${guest_run_dir}"
    exit 2
fi

guest_timezone="$(timedatectl show --property=Timezone --value)"
if [ "${guest_timezone}" != "Asia/Seoul" ]; then
    sudo -n timedatectl set-timezone Asia/Seoul
    guest_timezone="$(timedatectl show --property=Timezone --value)"
fi
if [ "${guest_timezone}" != "Asia/Seoul" ]; then
    echo "[guest-collector-error] timezone=${guest_timezone}, expected Asia/Seoul"
    exit 2
fi
printf "[guest-collector] timezone=%s\n" "${guest_timezone}"

vcpu_power_pid=""
stop_vcpu_collector() {
    if [ -n "${vcpu_power_pid}" ] && kill -0 "${vcpu_power_pid}" 2>/dev/null; then
        kill "${vcpu_power_pid}" 2>/dev/null || true
        wait "${vcpu_power_pid}" 2>/dev/null || true
    fi
    vcpu_power_pid=""
}
trap stop_vcpu_collector EXIT INT TERM

PYTHONDONTWRITEBYTECODE=1 taskset -c "${housekeeping_vcpu}" \
    python3 "${vcpu_power_sim}" \
    --out-log "${guest_run_dir}/vcpu_power.csv" \
    --raw-log "${guest_run_dir}/raw_guest_proc_stat.csv" \
    --vcpu-map "${app0_main_host_core}:${app0_main_vcpu}" \
    --vcpu-map "${app1_main_host_core}:${app1_main_vcpu}" \
    --raw-only-vcpu-map "${housekeeping_host_core}:${housekeeping_vcpu}" \
    --raw-only-vcpu-map "${app0_executor_host_core}:${app0_gemmini_vcpu}" \
    --raw-only-vcpu-map "${app1_executor_host_core}:${app1_gemmini_vcpu}" \
    --fixed-freq-ghz "${fixed_freq_ghz}" \
    --sys-scale "${sys_scale}" \
    --interval "${vcpu_sample_interval}" \
    > "${guest_run_dir}/raw_vcpu_power_sim.log" 2>&1 &
vcpu_power_pid=$!
sleep 2
if ! kill -0 "${vcpu_power_pid}" 2>/dev/null; then
    cat "${guest_run_dir}/raw_vcpu_power_sim.log" >&2 || true
    echo "[guest-collector-error] /proc/stat collector failed to start"
    exit 2
fi
printf "[guest-collector] procstat_started interval=%s modeled_vcpus=%s,%s raw_only_vcpus=%s,%s,%s\n" \
    "${vcpu_sample_interval}" "${app0_main_vcpu}" "${app1_main_vcpu}" \
    "${housekeeping_vcpu}" "${app0_gemmini_vcpu}" "${app1_gemmini_vcpu}"

current_guest_cpu() {
    awk '{print $39}' /proc/self/stat
}

pin_current_shell_to_vcpu() {
    local target_vcpu="$1"
    taskset -pc "${target_vcpu}" "${BASHPID}" >/dev/null 2>&1 || return 1
}

run_one_app() {
    local app_id="$1"
    local app_kind="$2"
    local workdir="$3"
    local bench_bin="$4"
    local output_path="$5"
    local main_vcpu="$6"
    local gemmini_vcpu="$7"
    local extra_args="$8"
    local wrapper_vcpu run_name run_dir status

    wrapper_vcpu="${main_vcpu}"
    if [ "${pin_remote_wrapper_to_housekeeping}" = "1" ]; then
        wrapper_vcpu="${housekeeping_vcpu}"
    fi
    pin_current_shell_to_vcpu "${wrapper_vcpu}" || true

    printf "[%s] [timeline] actor=vcpu label=output_dir_prepare sample=0 thread=0 cpu=-1\n" "${app_id}"
    if [ "${app_kind}" = "polybench_gemm" ]; then
        sudo -n mkdir -p "${output_path}"
    else
        mkdir -p "${output_path}"
    fi

    cd "${workdir}"
    printf "[%s] [launch-wrapper-pin] target_vcpu=%s cpu=%s main_vcpu=%s\n" "${app_id}" "${wrapper_vcpu}" "$(current_guest_cpu)" "${main_vcpu}"
    printf "[%s] [launch-affinity] main_vcpu=%s gemmini_executor_vcpu=%s housekeeping_vcpu=%s\n" "${app_id}" "${main_vcpu}" "${gemmini_vcpu}" "${housekeeping_vcpu}"
    printf "[%s] [timeline] actor=vcpu label=remote_launch sample=0 thread=0 cpu=-1\n" "${app_id}"

    set +e
    if [ "${app_kind}" = "polybench_gemm" ]; then
        run_name="run_$(date +%s)_${app_id}_$$"
        run_dir="${output_path}/${run_name}"
        sudo -n mkdir -p "${run_dir}"
        printf "[%s] [remote] run_dir=%s reps=%s\n" "${app_id}" "${run_dir}" "${polybench_reps}"
        sudo -n taskset -c "${main_vcpu}" "${bench_bin}" \
            --reps "${polybench_reps}" \
            --threads "${num_threads}" \
            --vcpu-offset "${main_vcpu}" \
            --gemmini-vcpu "${gemmini_vcpu}" \
            --housekeeping-vcpu "${housekeeping_vcpu}" \
            --output-dir "${run_dir}" \
            --timeline 2>&1 | while IFS= read -r inner_line; do
                printf "[%s] %s\n" "${app_id}" "${inner_line}"
            done
        status=${PIPESTATUS[0]}
        if [ -f "${run_dir}/metrics.json" ]; then
            printf "[%s] [remote] metrics_begin\n" "${app_id}"
            while IFS= read -r metrics_line; do
                printf "[%s] %s\n" "${app_id}" "${metrics_line}"
            done < "${run_dir}/metrics.json"
            printf "[%s] [remote] metrics_end\n" "${app_id}"
        fi
    else
        sudo -n taskset -c "${main_vcpu}" "${bench_bin}" \
            --limit "${sample_limit}" \
            --threads "${num_threads}" \
            --vcpu-offset "${main_vcpu}" \
            --gemmini-vcpu "${gemmini_vcpu}" \
            --output-dir "${output_path}" \
            ${extra_args} \
            --timeline 2>&1 | while IFS= read -r inner_line; do
                printf "[%s] %s\n" "${app_id}" "${inner_line}"
            done
        status=${PIPESTATUS[0]}
    fi
    set -e

    printf "[%s] [remote] exit_status=%d\n" "${app_id}" "${status}"
    exit "${status}"
}

(
    run_one_app "${app0_id}" "${app0_kind}" "${app0_workdir}" "${app0_bin}" "${app0_output}" "${app0_main_vcpu}" "${app0_gemmini_vcpu}" "${app0_extra_args}"
) &
app0_pid=$!

(
    run_one_app "${app1_id}" "${app1_kind}" "${app1_workdir}" "${app1_bin}" "${app1_output}" "${app1_main_vcpu}" "${app1_gemmini_vcpu}" "${app1_extra_args}"
) &
app1_pid=$!

set +e
wait "${app0_pid}"
app0_status=$?
wait "${app1_pid}"
app1_status=$?
set -e

sleep 2
if ! kill -0 "${vcpu_power_pid}" 2>/dev/null; then
    cat "${guest_run_dir}/raw_vcpu_power_sim.log" >&2 || true
    echo "[guest-collector-error] /proc/stat collector stopped unexpectedly"
    exit 2
fi
stop_vcpu_collector
for csv_path in "${guest_run_dir}/vcpu_power.csv" "${guest_run_dir}/raw_guest_proc_stat.csv"; do
    if ! awk 'END { exit !(NR >= 2) }' "${csv_path}"; then
        echo "[guest-collector-error] CSV has no data rows: ${csv_path}"
        exit 2
    fi
done
printf "[guest-collector] procstat_stopped\n"

printf "[remote] %s_status=%d %s_status=%d\n" "${app0_id}" "${app0_status}" "${app1_id}" "${app1_status}"
if [[ "${app0_status}" -ne 0 || "${app1_status}" -ne 0 ]]; then
    exit 1
fi
REMOTE_EOF
    }
}

run_remote_bench() {
    if [[ "${PIN_HOST_LOGGER_TO_HOUSEKEEPING}" == "1" ]]; then
        run_remote_bench_raw | taskset -c "${HOST_LOGGER_CORE}" bash -c '
while IFS= read -r line; do
    printf "[HOST_TS_LINE] %s %s\n" "$(date +%s.%N)" "${line}"
done
'
        return
    fi

    run_remote_bench_raw | while IFS= read -r line; do
        printf "[HOST_TS_LINE] %s %s\n" "$(date +%s.%N)" "${line}"
    done
}

emit_run_config() {
    local qemu_cmd
    qemu_cmd="$(pgrep -af qemu-system-riscv64 || true)"
    echo "[HOST_CONFIG] app0=${APP0_ID}"
    echo "[HOST_CONFIG] app1=${APP1_ID}"
    echo "[HOST_CONFIG] sample_limit=${SAMPLE_LIMIT}"
    echo "[HOST_CONFIG] threads=${NUM_THREADS}"
    echo "[HOST_CONFIG] polybench_dataset=${POLYBENCH_DATASET_MACRO}"
    echo "[HOST_CONFIG] polybench_reps=${POLYBENCH_REPS}"
    echo "[HOST_CONFIG] app0_main_vcpu_core=${APP0_MAIN_VCPU_CORE}"
    echo "[HOST_CONFIG] app1_main_vcpu_core=${APP1_MAIN_VCPU_CORE}"
    echo "[HOST_CONFIG] app0_gemmini_vcpu_core=${APP0_GEMMINI_VCPU_CORE}"
    echo "[HOST_CONFIG] app1_gemmini_vcpu_core=${APP1_GEMMINI_VCPU_CORE}"
    echo "[HOST_CONFIG] app0_gemmini_core=${APP0_GEMMINI_CORE}"
    echo "[HOST_CONFIG] app1_gemmini_core=${APP1_GEMMINI_CORE}"
    echo "[HOST_CONFIG] housekeeping_vcpu_core=${HOUSEKEEPING_VCPU_CORE}"
    echo "[HOST_CONFIG] guest_housekeeping_vcpu=${GUEST_HOUSEKEEPING_VCPU}"
    echo "[HOST_CONFIG] guest_app0_main_vcpu=${GUEST_APP0_MAIN_VCPU}"
    echo "[HOST_CONFIG] guest_app1_main_vcpu=${GUEST_APP1_MAIN_VCPU}"
    echo "[HOST_CONFIG] guest_app0_gemmini_vcpu=${GUEST_APP0_GEMMINI_VCPU}"
    echo "[HOST_CONFIG] guest_app1_gemmini_vcpu=${GUEST_APP1_GEMMINI_VCPU}"
    echo "[HOST_CONFIG] sys_scale=${SYS_SCALE}"
    echo "[HOST_CONFIG] gemmini_full_util_power_w=${GEMMINI_FULL_UTIL_POWER_W}"
    echo "[HOST_CONFIG] gemmini_util_threshold_pct=${GEMMINI_UTIL_THRESHOLD_PCT}"
    echo "[HOST_CONFIG] vcpu_util_source=guest_proc_stat_main_only"
    echo "[HOST_CONFIG] gemmini_util_source=host_turbostat"
    echo "[HOST_CONFIG] vcpu_sample_interval=${VCPU_SAMPLE_INTERVAL}"
    echo "[HOST_CONFIG] raw_plot_time_zero_event=BENCH_HOST_START"
    echo "[HOST_CONFIG] scaled_plot_time_zero_event=sample_start"
    echo "[HOST_CONFIG] scaled_plot_window=sample_start_minus_margin_to_sample_done_plus_margin"
    echo "[HOST_CONFIG] scaled_plot_window_margin_s=${SCALED_PLOT_WINDOW_MARGIN_S}"
    echo "[HOST_CONFIG] scaled_plot_fig_width=${SCALED_PLOT_FIG_WIDTH}"
    echo "[HOST_CONFIG] plot_time_major_tick_s=${PLOT_TIME_MAJOR_TICK_S}"
    echo "[HOST_CONFIG] time_scale_mode=${TIME_SCALE_MODE}"
    echo "[HOST_CONFIG] gemmini_frequency_mhz=${GEMMINI_FREQUENCY_MHZ}"
    echo "[HOST_CONFIG] cpu_time_scale=${CPU_TIME_SCALE}"
    echo "[HOST_CONFIG] fixed_freq_ghz=${FIXED_FREQ_GHZ}"
    echo "[HOST_CONFIG] auto_resolve_host_cores=${AUTO_RESOLVE_HOST_CORES}"
    echo "[HOST_CONFIG] pin_qemu_before_run=${PIN_QEMU_BEFORE_RUN}"
    echo "[HOST_CONFIG] qemu_vcpu_base_core=${QEMU_VCPU_BASE_CORE}"
    echo "[HOST_CONFIG] qemu_gemmini_base_core=${QEMU_GEMMINI_BASE_CORE}"
    echo "[HOST_CONFIG] host_logger_core=${HOST_LOGGER_CORE}"
    echo "[HOST_CONFIG] pin_host_logger_to_housekeeping=${PIN_HOST_LOGGER_TO_HOUSEKEEPING}"
    echo "[HOST_CONFIG] pin_remote_wrapper_to_housekeeping=${PIN_REMOTE_WRAPPER_TO_HOUSEKEEPING}"
    echo "[HOST_CONFIG] gemmini_dispatch_policy=${GEMMINI_DISPATCH_POLICY}"
    echo "[HOST_CONFIG] qemu_vcpu_count=${QEMU_VCPU_COUNT}"
    echo "[HOST_CONFIG] qemu_gemmini_count=${QEMU_GEMMINI_COUNT}"
    echo "[HOST_CONFIG] app0_gemmini_worker_index=${APP0_GEMMINI_WORKER_INDEX}"
    echo "[HOST_CONFIG] app1_gemmini_worker_index=${APP1_GEMMINI_WORKER_INDEX}"
    echo "[HOST_CONFIG] build_before_run=${BUILD_BEFORE_RUN}"
    echo "[HOST_CONFIG] qemu_log=${QEMU_LOG}"
    echo "[HOST_CONFIG] qemu_cmd=${qemu_cmd}"
}

write_qemu_mapping_summary() {
    local start_line="$1"
    local out_log="$2"
    local out_summary="$3"
    local mapping_rows

    if [[ -f "${QEMU_LOG}" ]]; then
        sed -n "$((start_line + 1)),\$p" "${QEMU_LOG}" > "${out_log}"
    else
        : > "${out_log}"
    fi

    mapping_rows="$(
        sed -n \
            -e 's/.*\[IDLE_FIRST\] CPU \([0-9][0-9]*\) bound to Gemmini \([0-9][0-9]*\).*/bind core_\1 gemmini_\2/p' \
            -e 's/.*\[IDLE_FIRST FALLBACK\] CPU \([0-9][0-9]*\).*Gemmini \([0-9][0-9]*\).*/fallback core_\1 gemmini_\2/p' \
            -e 's/.*\[MODULO\] CPU \([0-9][0-9]*\) bound to Gemmini \([0-9][0-9]*\).*/modulo_bind core_\1 gemmini_\2/p' \
            -e 's/.*\[RANDOM\] CPU \([0-9][0-9]*\) bound to Gemmini \([0-9][0-9]*\).*/random_bind core_\1 gemmini_\2/p' \
            -e 's/.*\[WORK ENQUEUED\]\[Core \([0-9][0-9]*\)-Gemmini \([0-9][0-9]*\)\].*/work_enqueued core_\1 gemmini_\2/p' \
            "${out_log}" |
            awk \
                -v app0_core="core_${GUEST_APP0_GEMMINI_VCPU}" \
                -v app1_core="core_${GUEST_APP1_GEMMINI_VCPU}" \
                '$2 == app0_core || $2 == app1_core { print }' |
            sort |
            uniq -c
    )"

    {
        echo "# QEMU Gemmini mapping summary"
        echo "app0=${APP0_ID}"
        echo "app1=${APP1_ID}"
        echo "gemmini_dispatch_policy=${GEMMINI_DISPATCH_POLICY}"
        echo "app0_executor_guest_vcpu=${GUEST_APP0_GEMMINI_VCPU}"
        echo "app1_executor_guest_vcpu=${GUEST_APP1_GEMMINI_VCPU}"
        echo "app0_expected_gemmini_worker=${APP0_GEMMINI_WORKER_INDEX}"
        echo "app1_expected_gemmini_worker=${APP1_GEMMINI_WORKER_INDEX}"
        echo "app0_gemmini_host_core=${APP0_GEMMINI_CORE}"
        echo "app1_gemmini_host_core=${APP1_GEMMINI_CORE}"
        echo
        if [[ -n "${mapping_rows}" ]]; then
            printf '%s\n' "${mapping_rows}"
        else
            echo "no_app_executor_gemmini_mapping_rows_found"
        fi
    } > "${out_summary}"

    echo "QEMU mapping log     : ${out_log}"
    echo "QEMU mapping summary : ${out_summary}"
}

write_qemu_thread_affinity_summary() {
    local out_summary="$1"
    local role guest_vcpu worker_index tid affinity host_core

    {
        echo "# QEMU thread affinity summary"
        echo "auto_resolve_host_cores=${AUTO_RESOLVE_HOST_CORES}"
        echo "gemmini_dispatch_policy=${GEMMINI_DISPATCH_POLICY}"
        echo "qemu_vcpu_count=${QEMU_VCPU_COUNT}"
        echo "qemu_gemmini_count=${QEMU_GEMMINI_COUNT}"
        echo
        echo "role,guest_vcpu,gemmini_worker_index,tid,host_affinity,tracked_host_core"

        for role in housekeeping app0_main app1_main app0_gemmini_executor app1_gemmini_executor; do
            case "${role}" in
                housekeeping)
                    guest_vcpu="${GUEST_HOUSEKEEPING_VCPU}"
                    host_core="${HOUSEKEEPING_VCPU_CORE}"
                    ;;
                app0_main)
                    guest_vcpu="${GUEST_APP0_MAIN_VCPU}"
                    host_core="${APP0_MAIN_VCPU_CORE}"
                    ;;
                app1_main)
                    guest_vcpu="${GUEST_APP1_MAIN_VCPU}"
                    host_core="${APP1_MAIN_VCPU_CORE}"
                    ;;
                app0_gemmini_executor)
                    guest_vcpu="${GUEST_APP0_GEMMINI_VCPU}"
                    host_core="${APP0_GEMMINI_VCPU_CORE}"
                    ;;
                app1_gemmini_executor)
                    guest_vcpu="${GUEST_APP1_GEMMINI_VCPU}"
                    host_core="${APP1_GEMMINI_VCPU_CORE}"
                    ;;
            esac
            tid="$(qemu_vcpu_tid "${guest_vcpu}")"
            affinity="$(tid_affinity_list "${tid}")"
            echo "${role},${guest_vcpu},,${tid},${affinity},${host_core}"
        done

        worker_index="${APP0_GEMMINI_WORKER_INDEX}"
        tid="$(qemu_gemmini_tid "${worker_index}")"
        affinity="$(tid_affinity_list "${tid}")"
        echo "app0_gemmini_worker,,${worker_index},${tid},${affinity},${APP0_GEMMINI_CORE}"

        worker_index="${APP1_GEMMINI_WORKER_INDEX}"
        tid="$(qemu_gemmini_tid "${worker_index}")"
        affinity="$(tid_affinity_list "${tid}")"
        echo "app1_gemmini_worker,,${worker_index},${tid},${affinity},${APP1_GEMMINI_CORE}"
    } > "${out_summary}"

    echo "QEMU affinity summary: ${out_summary}"
}

validate_and_configure() {
    [[ -n "${APP0}" && -n "${APP1}" ]] || die "--app0/--app1 or --apps is required"
    configure_slot APP0 "${APP0}"
    configure_slot APP1 "${APP1}"
    [[ "${APP0_ID}" != "${APP1_ID}" ]] || die "dual experiment requires two different apps"

    POLYBENCH_DATASET_MACRO="$(dataset_macro "${POLYBENCH_DATASET}")"
    require_positive_int "sample_limit" "${SAMPLE_LIMIT}"
    require_positive_int "threads" "${NUM_THREADS}"
    [[ "${NUM_THREADS}" == "1" ]] || die "dual main/executor split mode requires --threads 1"
    require_positive_int "polybench_reps" "${POLYBENCH_REPS}"
    require_number "sys_scale" "${SYS_SCALE}"
    require_number "gemmini_full_util_power_w" "${GEMMINI_FULL_UTIL_POWER_W}"
    require_number "gemmini_util_threshold_pct" "${GEMMINI_UTIL_THRESHOLD_PCT}"
    require_number "fixed_freq_ghz" "${FIXED_FREQ_GHZ}"
    require_number "scaled_plot_window_margin_s" "${SCALED_PLOT_WINDOW_MARGIN_S}"
    require_number "scaled_plot_fig_width" "${SCALED_PLOT_FIG_WIDTH}"
    require_number "plot_time_major_tick_s" "${PLOT_TIME_MAJOR_TICK_S}"
    require_number "gemmini_frequency_mhz" "${GEMMINI_FREQUENCY_MHZ}"
    case "${TIME_SCALE_MODE}" in
        rtl-cycle) ;;
        *) die "TIME_SCALE_MODE must be rtl-cycle" ;;
    esac
    if [[ "${TIME_SCALE_MODE}" == "rtl-cycle" \
        && ( "${APP0_KIND}" == "polybench_gemm" || "${APP1_KIND}" == "polybench_gemm" ) \
        && "${POLYBENCH_DATASET_MACRO}" != "MEDIUM_DATASET" ]]; then
        die "the PolyBench RTL cycle profile is validated only for MEDIUM_DATASET"
    fi
    require_number "cpu_time_scale" "${CPU_TIME_SCALE}"
    require_number "turbostat_interval" "${TURBOSTAT_INTERVAL}"
    require_number "power_poll_interval" "${POWER_POLL_INTERVAL}"
    require_number "vcpu_sample_interval" "${VCPU_SAMPLE_INTERVAL}"

    case "${BUILD_BEFORE_RUN}" in
        0|1) ;;
        *) die "--build must be 0 or 1" ;;
    esac

    for core in \
        "${APP0_MAIN_VCPU_CORE}" \
        "${APP1_MAIN_VCPU_CORE}" \
        "${APP0_GEMMINI_VCPU_CORE}" \
        "${APP1_GEMMINI_VCPU_CORE}" \
        "${APP0_GEMMINI_CORE}" \
        "${APP1_GEMMINI_CORE}" \
        "${HOUSEKEEPING_VCPU_CORE}" \
        "${QEMU_VCPU_BASE_CORE}" \
        "${QEMU_GEMMINI_BASE_CORE}" \
        "${HOST_LOGGER_CORE}"; do
        require_nonnegative_int "host core" "${core}"
    done

    for vcpu in \
        "${GUEST_HOUSEKEEPING_VCPU}" \
        "${GUEST_APP0_MAIN_VCPU}" \
        "${GUEST_APP1_MAIN_VCPU}" \
        "${GUEST_APP0_GEMMINI_VCPU}" \
        "${GUEST_APP1_GEMMINI_VCPU}"; do
        require_nonnegative_int "guest vCPU" "${vcpu}"
    done

    require_unique_cores \
        "${APP0_MAIN_VCPU_CORE}" \
        "${APP1_MAIN_VCPU_CORE}" \
        "${APP0_GEMMINI_VCPU_CORE}" \
        "${APP1_GEMMINI_VCPU_CORE}" \
        "${APP0_GEMMINI_CORE}" \
        "${APP1_GEMMINI_CORE}" \
        "${HOUSEKEEPING_VCPU_CORE}"
    require_unique_cores \
        "${GUEST_HOUSEKEEPING_VCPU}" \
        "${GUEST_APP0_MAIN_VCPU}" \
        "${GUEST_APP1_MAIN_VCPU}" \
        "${GUEST_APP0_GEMMINI_VCPU}" \
        "${GUEST_APP1_GEMMINI_VCPU}"

    for required in \
        "${PLOT_SCRIPT}" \
        "${POWER_SIM}" \
        "${MERGE_POWER}" \
        "${POWER_MODELING_ROOT}/power_sim/power_sim_vcpu_procstat.py" \
        "${TIME_SCALE_UTIL}" \
        "${SSH_KEY}"; do
        [[ -f "${required}" ]] || die "missing required file: ${required}"
    done
}

run_dual_plot() {
    local turbostat_path="$1"
    local guest_util_path="$2"
    local power_path="$3"
    local bench_path="$4"
    local out_path="$5"
    local variant="${6:-raw}"
    local plot_title="${APP0_NAME} + ${APP1_NAME} dual power (CPU=${SYS_SCALE}, Gemmini=${gemmini_power_tag})"
    local plot_args=(
        --turbostat "${turbostat_path}"
        --guest-util-csv "${guest_util_path}"
        --power-csv "${power_path}"
        --bench "${bench_path}"
        --out "${out_path}"
        --app0-id "${APP0_ID}"
        --app0-name "${APP0_NAME}"
        --app0-gemmini-core "${APP0_GEMMINI_CORE}"
        --app0-main-vcpu-core "${APP0_MAIN_VCPU_CORE}"
        --app0-main-guest-vcpu "${GUEST_APP0_MAIN_VCPU}"
        --app0-gemmini-vcpu-core "${APP0_GEMMINI_VCPU_CORE}"
        --app0-executor-guest-vcpu "${GUEST_APP0_GEMMINI_VCPU}"
        --app1-id "${APP1_ID}"
        --app1-name "${APP1_NAME}"
        --app1-gemmini-core "${APP1_GEMMINI_CORE}"
        --app1-main-vcpu-core "${APP1_MAIN_VCPU_CORE}"
        --app1-main-guest-vcpu "${GUEST_APP1_MAIN_VCPU}"
        --app1-gemmini-vcpu-core "${APP1_GEMMINI_VCPU_CORE}"
        --app1-executor-guest-vcpu "${GUEST_APP1_GEMMINI_VCPU}"
        --housekeeping-vcpu-core "${HOUSEKEEPING_VCPU_CORE}"
        --guest-housekeeping-vcpu "${GUEST_HOUSEKEEPING_VCPU}"
        --presentation-util
    )

    if [[ "${variant}" == "scaled" ]]; then
        plot_title="${APP0_NAME} + ${APP1_NAME} dual scaled power (time ${time_scale_label})"
        plot_args+=(
            --title "${plot_title}"
            --time-zero-event sample_start
            --sample-window-margin "${SCALED_PLOT_WINDOW_MARGIN_S}"
            --time-major-tick-s "${PLOT_TIME_MAJOR_TICK_S}"
            --fig-width "${SCALED_PLOT_FIG_WIDTH}"
        )
    else
        plot_args+=(
            --title "${plot_title}"
            --time-zero-event BENCH_HOST_START
            --crop-before-time-zero
        )
    fi

    PYTHONDONTWRITEBYTECODE=1 python3 "${PLOT_SCRIPT}" "${plot_args[@]}"
}

verify_plot_family() {
    local out_path="$1"
    local stem="${out_path%.png}"
    local expected
    for expected in \
        "${out_path}" \
        "${stem}_${APP0_ID}.png" \
        "${stem}_${APP1_ID}.png" \
        "${stem}_guest_vcpu_util.png" \
        "${stem}_host_vcpu_util.png" \
        "${stem}_guest_executor_vcpu_util.png" \
        "${stem}_host_executor_vcpu_util.png" \
        "${stem}_guest_housekeeping_util.png" \
        "${stem}_host_housekeeping_util.png"; do
        [[ -s "${expected}" ]] || die "missing plot: ${expected}"
    done
}

main() {
    local pair_id ts run_name run_dir tracked_cores scale_tag gemmini_power_tag time_scale_tag
    local cpu_time_scale_tag frequency_tag time_scale_label
    local bench_log turbostat_log vcpu_power_csv guest_util_csv
    local gemmini_power_csv gemmini_power_log combined_power_png
    local qemu_log_start_line qemu_run_log qemu_mapping_summary qemu_affinity_summary qemu_pin_log
    local scaled_prefix scaled_bench_log scaled_turbostat_log scaled_guest_util_csv
    local scaled_power_csv scaled_summary scaled_combined_power_png
    local required_output
    local -a time_scale_args

    parse_args "$@"
    validate_and_configure

    build_slot_if_needed APP0
    build_slot_if_needed APP1

    require_qemu_topology
    pin_qemu_threads_if_requested
    resolve_host_core_mapping
    require_unique_cores \
        "${APP0_MAIN_VCPU_CORE}" \
        "${APP1_MAIN_VCPU_CORE}" \
        "${APP0_GEMMINI_VCPU_CORE}" \
        "${APP1_GEMMINI_VCPU_CORE}" \
        "${APP0_GEMMINI_CORE}" \
        "${APP1_GEMMINI_CORE}" \
        "${HOUSEKEEPING_VCPU_CORE}"

    echo "Refreshing sudo credential for turbostat..."
    sudo -v

    pair_id="${APP0_ID}_${APP1_ID}"
    ts="$(date +%Y%m%d_%H%M%S)"
    scale_tag="${SYS_SCALE//./p}"
    cpu_time_scale_tag="${CPU_TIME_SCALE//./p}"
    frequency_tag="${GEMMINI_FREQUENCY_MHZ//./p}"
    time_scale_tag="rtl_${frequency_tag}mhz_cpu_${cpu_time_scale_tag}"
    time_scale_label="RTL ${GEMMINI_FREQUENCY_MHZ} MHz, CPU=${CPU_TIME_SCALE}"
    time_scale_args=(
        --time-scale-mode rtl-cycle
        --gemmini-frequency-mhz "${GEMMINI_FREQUENCY_MHZ}"
        --model-rtl-workload "${APP0_ID}=${APP0_ID}"
        --model-rtl-workload "${APP1_ID}=${APP1_ID}"
    )
    printf -v gemmini_power_tag "%.6f" "${GEMMINI_FULL_UTIL_POWER_W}"
    tracked_cores="${APP0_GEMMINI_CORE},${APP1_GEMMINI_CORE},${HOUSEKEEPING_VCPU_CORE},${APP0_MAIN_VCPU_CORE},${APP1_MAIN_VCPU_CORE},${APP0_GEMMINI_VCPU_CORE},${APP1_GEMMINI_VCPU_CORE}"
    run_name="${ts}_${pair_id}_limit${SAMPLE_LIMIT}_vcpu${HOUSEKEEPING_VCPU_CORE}-${APP0_MAIN_VCPU_CORE}-${APP1_MAIN_VCPU_CORE}-${APP0_GEMMINI_VCPU_CORE}-${APP1_GEMMINI_VCPU_CORE}_gemmini${APP0_GEMMINI_CORE}-${APP1_GEMMINI_CORE}_scale${scale_tag}"
    if [[ "${APP0_KIND}" == "polybench_gemm" || "${APP1_KIND}" == "polybench_gemm" ]]; then
        run_name="${run_name}_${POLYBENCH_DATASET_MACRO}_reps${POLYBENCH_REPS}"
    fi

    run_dir="${DUAL_ROOT}/logs/${pair_id}/${run_name}"
    GUEST_RUN_DIR="${GUEST_DUAL_ROOT}/logs/${pair_id}/${run_name}"
    mkdir -p "${run_dir}"
    chmod 0777 "${run_dir}"
    trap cleanup EXIT INT TERM

    bench_log="${run_dir}/raw_bench.log"
    turbostat_log="${run_dir}/raw_turbostat.log"
    vcpu_power_csv="${run_dir}/vcpu_power.csv"
    guest_util_csv="${run_dir}/raw_guest_proc_stat.csv"
    gemmini_power_csv="${run_dir}/gemmini_power.csv"
    gemmini_power_log="${run_dir}/raw_gemmini_power_sim.log"
    combined_power_png="${run_dir}/raw_combined_power.png"
    qemu_run_log="${run_dir}/qemu_gemmini_run.log"
    qemu_mapping_summary="${run_dir}/qemu_gemmini_mapping_summary.txt"
    qemu_affinity_summary="${run_dir}/qemu_thread_affinity_summary.csv"
    qemu_pin_log="${run_dir}/qemu_pin.log"
    scaled_prefix="${run_dir}/time_scaled_${time_scale_tag}"
    scaled_bench_log="${scaled_prefix}_bench.log"
    scaled_turbostat_log="${scaled_prefix}_turbostat.log"
    scaled_guest_util_csv="${scaled_prefix}_guest_util.csv"
    scaled_power_csv="${scaled_prefix}_power.csv"
    scaled_summary="${scaled_prefix}_summary.json"
    scaled_combined_power_png="${scaled_prefix}_combined_power.png"

    printf '%s\n' "${QEMU_PIN_OUTPUT}" > "${qemu_pin_log}"
    write_qemu_thread_affinity_summary "${qemu_affinity_summary}"
    pin_guest_launch_services |& tee "${run_dir}/raw_guest_housekeeping.log"

    echo "Starting turbostat on host cores ${tracked_cores} at ${TURBOSTAT_INTERVAL}s interval..."
    sudo -n turbostat \
        -q \
        -d \
        --show Time_Of_Day_Seconds,CPU,Busy%,Bzy_MHz,Avg_MHz \
        -c "${tracked_cores}" \
        -i "${TURBOSTAT_INTERVAL}" \
        > "${turbostat_log}" 2>&1 &
    CURRENT_TURBOSTAT_PID="$!"

    sleep 1

    echo "Starting real-time Gemmini power simulator..."
    PYTHONDONTWRITEBYTECODE=1 python3 "${POWER_SIM}" \
        --turbostat "${turbostat_log}" \
        --out-log "${gemmini_power_csv}" \
        --gemmini-cores "${APP0_GEMMINI_CORE}" "${APP1_GEMMINI_CORE}" \
        --fixed-freq-ghz "${FIXED_FREQ_GHZ}" \
        --sys-scale "${SYS_SCALE}" \
        --gemmini-full-util-power-w "${GEMMINI_FULL_UTIL_POWER_W}" \
        --gemmini-util-threshold-pct "${GEMMINI_UTIL_THRESHOLD_PCT}" \
        --poll-interval "${POWER_POLL_INTERVAL}" \
        > "${gemmini_power_log}" 2>&1 &
    CURRENT_POWER_PID="$!"

    sleep 1
    if ! kill -0 "${CURRENT_TURBOSTAT_PID}" 2>/dev/null; then
        tail -40 "${turbostat_log}" >&2 || true
        die "turbostat failed to start"
    fi
    if ! kill -0 "${CURRENT_POWER_PID}" 2>/dev/null; then
        tail -40 "${gemmini_power_log}" >&2 || true
        die "Gemmini power simulator failed to start"
    fi
    qemu_log_start_line="$(wc -l < "${QEMU_LOG}")"

    {
        emit_run_config
        echo "[HOST_TS_REALTIME] BENCH_HOST_START $(date +%s.%N)"
        run_remote_bench
        echo "[HOST_TS_REALTIME] BENCH_HOST_END $(date +%s.%N)"
    } | tee "${bench_log}"

    sleep 1
    cleanup
    CURRENT_TURBOSTAT_PID=""
    CURRENT_POWER_PID=""

    write_qemu_mapping_summary "${qemu_log_start_line}" "${qemu_run_log}" "${qemu_mapping_summary}"

    for required_output in \
        "${bench_log}" \
        "${turbostat_log}" \
        "${vcpu_power_csv}" \
        "${guest_util_csv}" \
        "${gemmini_power_csv}" \
        "${run_dir}/raw_guest_housekeeping.log"; do
        [[ -s "${required_output}" ]] || die "missing measurement output: ${required_output}"
    done

    PLOT_POWER_TMP="$(mktemp /tmp/qvela_dual_plot_power.XXXXXX.csv)"
    PLOT_UTIL_TMP="$(mktemp /tmp/qvela_dual_plot_util.XXXXXX.log)"
    PYTHONDONTWRITEBYTECODE=1 python3 "${MERGE_POWER}" \
        --vcpu-power "${vcpu_power_csv}" \
        --gemmini-power "${gemmini_power_csv}" \
        --out-power "${PLOT_POWER_TMP}" \
        --out-util "${PLOT_UTIL_TMP}"

    run_dual_plot \
        "${turbostat_log}" \
        "${guest_util_csv}" \
        "${PLOT_POWER_TMP}" \
        "${bench_log}" \
        "${combined_power_png}"
    verify_plot_family "${combined_power_png}"

    PYTHONDONTWRITEBYTECODE=1 python3 "${TIME_SCALE_UTIL}" \
        --bench "${bench_log}" \
        --turbostat "${turbostat_log}" \
        --power-csv "${PLOT_POWER_TMP}" \
        --guest-util-csv "${guest_util_csv}" \
        "${time_scale_args[@]}" \
        --cpu-time-scale "${CPU_TIME_SCALE}" \
        --model-core "${APP0_ID}=${APP0_MAIN_VCPU_CORE},${APP0_GEMMINI_VCPU_CORE},${APP0_GEMMINI_CORE}" \
        --model-core "${APP1_ID}=${APP1_MAIN_VCPU_CORE},${APP1_GEMMINI_VCPU_CORE},${APP1_GEMMINI_CORE}" \
        --out-prefix "${scaled_prefix}"

    for required_output in \
        "${scaled_bench_log}" \
        "${scaled_turbostat_log}" \
        "${scaled_guest_util_csv}" \
        "${scaled_power_csv}" \
        "${scaled_summary}"; do
        [[ -s "${required_output}" ]] || die "missing time-scaled output: ${required_output}"
    done

    run_dual_plot \
        "${scaled_turbostat_log}" \
        "${scaled_guest_util_csv}" \
        "${scaled_power_csv}" \
        "${scaled_bench_log}" \
        "${scaled_combined_power_png}" \
        scaled
    verify_plot_family "${scaled_combined_power_png}"
    rm -f "${PLOT_POWER_TMP}"
    PLOT_POWER_TMP=""
    rm -f "${PLOT_UTIL_TMP}"
    PLOT_UTIL_TMP=""
    echo
    echo "Run dir              : ${run_dir}"
    echo "Bench log            : ${bench_log}"
    echo "Guest affinity log   : ${run_dir}/raw_guest_housekeeping.log"
    echo "Guest procstat       : ${guest_util_csv}"
    echo "Host turbostat       : ${turbostat_log}"
    echo "vCPU power csv       : ${vcpu_power_csv}"
    echo "Gemmini power csv    : ${gemmini_power_csv}"
    echo "Gemmini power log    : ${gemmini_power_log}"
    echo "QEMU run log         : ${qemu_run_log}"
    echo "QEMU mapping summary : ${qemu_mapping_summary}"
    echo "QEMU affinity summary: ${qemu_affinity_summary}"
    echo "QEMU pin log         : ${qemu_pin_log}"
    echo "Combined power plot  : ${combined_power_png}"
    echo "${APP0_NAME} plot       : ${combined_power_png%.png}_${APP0_ID}.png"
    echo "${APP1_NAME} plot       : ${combined_power_png%.png}_${APP1_ID}.png"
    echo "Guest main plot      : ${combined_power_png%.png}_guest_vcpu_util.png"
    echo "Host main plot       : ${combined_power_png%.png}_host_vcpu_util.png"
    echo "Guest executor plot  : ${combined_power_png%.png}_guest_executor_vcpu_util.png"
    echo "Host executor plot   : ${combined_power_png%.png}_host_executor_vcpu_util.png"
    echo "Guest housekeeping   : ${combined_power_png%.png}_guest_housekeeping_util.png"
    echo "Host housekeeping    : ${combined_power_png%.png}_host_housekeeping_util.png"
    echo "Scaled guest util    : ${scaled_guest_util_csv}"
    echo "Scaled power csv     : ${scaled_power_csv}"
    echo "Scaled summary       : ${scaled_summary}"
    echo "Scaled combined plot : ${scaled_combined_power_png}"
}

main "$@"
