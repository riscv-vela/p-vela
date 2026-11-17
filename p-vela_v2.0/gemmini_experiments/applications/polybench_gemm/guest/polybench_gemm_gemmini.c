#define _GNU_SOURCE

#include <math.h>
#include <pthread.h>
#include <sched.h>

#include "polybench/gemm.h"
#include "gemmini.h"
#include "polybench_gemm_common.h"

#define GEMMINI_SAFE_MAX_I 16
#define GEMMINI_SAFE_MAX_J 128

#ifndef POLYBENCH_GEMM_POWER_ONLY
#define POLYBENCH_GEMM_POWER_ONLY 0
#endif

#ifndef POLYBENCH_GEMM_USE_GEMMINI_BACKEND
#define POLYBENCH_GEMM_USE_GEMMINI_BACKEND 1
#endif

#ifndef POLYBENCH_GEMM_CPU_CHUNKED_BACKEND
#define POLYBENCH_GEMM_CPU_CHUNKED_BACKEND 0
#endif

#ifndef POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
#define POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND 0
#endif

#ifndef POLYBENCH_GEMM_CHUNKED_LEGACY_BACKEND
#define POLYBENCH_GEMM_CHUNKED_LEGACY_BACKEND 0
#endif

#if POLYBENCH_GEMM_USE_GEMMINI_BACKEND && POLYBENCH_GEMM_CPU_CHUNKED_BACKEND
#error "CPU chunked backend and Gemmini backend are mutually exclusive."
#endif

#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND && !POLYBENCH_GEMM_USE_GEMMINI_BACKEND
#error "The direct Gemmini backend requires POLYBENCH_GEMM_USE_GEMMINI_BACKEND."
#endif

#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND && POLYBENCH_GEMM_CHUNKED_LEGACY_BACKEND
#error "The direct and chunked legacy Gemmini backends are mutually exclusive."
#endif

#if !POLYBENCH_GEMM_USE_GEMMINI_BACKEND && !POLYBENCH_GEMM_POWER_ONLY
#error "The CPU backend shares this runner for power-only builds."
#endif

#if POLYBENCH_GEMM_USE_GEMMINI_BACKEND
#if POLYBENCH_GEMM_POWER_ONLY
#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
#define POLYBENCH_GEMM_BACKEND_MODE "gemmini_direct_power"
#elif POLYBENCH_GEMM_CHUNKED_LEGACY_BACKEND
#define POLYBENCH_GEMM_BACKEND_MODE "gemmini_chunked_legacy_power"
#else
#define POLYBENCH_GEMM_BACKEND_MODE "gemmini_power"
#endif
#elif POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
#define POLYBENCH_GEMM_BACKEND_MODE "gemmini_direct"
#elif POLYBENCH_GEMM_CHUNKED_LEGACY_BACKEND
#define POLYBENCH_GEMM_BACKEND_MODE "gemmini_chunked_legacy"
#else
#define POLYBENCH_GEMM_BACKEND_MODE "gemmini"
#endif
#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
#define POLYBENCH_GEMM_BACKEND_NAME "tiled_matmul_auto_direct"
#elif POLYBENCH_GEMM_CHUNKED_LEGACY_BACKEND
#define POLYBENCH_GEMM_BACKEND_NAME "tiled_matmul_auto_chunked_legacy"
#else
#define POLYBENCH_GEMM_BACKEND_NAME "tiled_matmul_auto"
#endif
#elif POLYBENCH_GEMM_CPU_CHUNKED_BACKEND
#define POLYBENCH_GEMM_BACKEND_MODE "cpu_int8_chunked_polybench_power"
#define POLYBENCH_GEMM_BACKEND_NAME "cpu_loop_chunked"
#else
#define POLYBENCH_GEMM_BACKEND_MODE "cpu_int8_polybench_power"
#define POLYBENCH_GEMM_BACKEND_NAME "cpu_loop"
#endif

static bool g_log_backend_calls = false;

typedef struct {
    long long chunks_total;
    double backend_matmul_ms;
    double executor_submit_wait_ms;
    double run_chunk_total_ms;
    double full_acc_memset_ms;
    double chunk_acc_memset_ms;
    double zero_bias_memset_ms;
    double pre_fence_flush_ms;
    double tiled_matmul_ms;
    double post_fence_ms;
    double copy_chunk_acc_ms;
    double polybench_update_ms;
} GemminiLatencyProfile;

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int target_vcpu;
    int pinned_cpu;
    bool stop;
    bool has_job;
    bool job_done;
    const GemmOptions *opts;
    const elem_t *a_chunk;
    const elem_t *b_chunk;
    acc_t *chunk_acc;
    acc_t *zero_bias;
    int chunk_rows;
    int chunk_cols;
    int row_base;
    int col_base;
    GemminiLatencyProfile *profile;
} GemminiExecutor;

static elem_t quantize_unit_interval(double value)
{
    long q = lround((value * 2.0 - 1.0) * 63.0);

    if (q > 127) {
        q = 127;
    } else if (q < -128) {
        q = -128;
    }
    return (elem_t)q;
}

static void *aligned_alloc_or_die(size_t alignment, size_t bytes, const char *name)
{
    void *ptr = NULL;
    const int rc = posix_memalign(&ptr, alignment, bytes);

    if (rc != 0 || ptr == NULL) {
        fatalf("failed to allocate %s: %s", name, strerror(rc));
    }
    memset(ptr, 0, bytes);
    return ptr;
}

static elem_t *alloc_elem_matrix(size_t rows, size_t cols, const char *name)
{
    size_t count;

    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols) {
        fatalf("invalid elem matrix size for %s", name);
    }
    count = rows * cols;
    if (count > SIZE_MAX / sizeof(elem_t)) {
        fatalf("elem matrix allocation is too large for %s", name);
    }
    return (elem_t *)aligned_alloc_or_die(64, count * sizeof(elem_t), name);
}

static acc_t *alloc_acc_matrix(size_t rows, size_t cols, const char *name)
{
    size_t count;

    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols) {
        fatalf("invalid acc matrix size for %s", name);
    }
    count = rows * cols;
    if (count > SIZE_MAX / sizeof(acc_t)) {
        fatalf("acc matrix allocation is too large for %s", name);
    }
    return (acc_t *)aligned_alloc_or_die(64, count * sizeof(acc_t), name);
}

static void init_gemmini_inputs(int m, int n, int k, elem_t *a, elem_t *b, double *c)
{
    if (c != NULL) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[(size_t)i * (size_t)n + (size_t)j] =
                    (double)((i * j + 1) % m) / (double)m;
            }
        }
    }

    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            const double value = (double)(i * (kk + 1) % k) / (double)k;
            a[(size_t)i * (size_t)k + (size_t)kk] = quantize_unit_interval(value);
        }
    }

    for (int kk = 0; kk < k; kk++) {
        for (int j = 0; j < n; j++) {
            const double value = (double)(kk * (j + 2) % n) / (double)n;
            b[(size_t)kk * (size_t)n + (size_t)j] = quantize_unit_interval(value);
        }
    }
}

static void run_cpu_int8_matmul(const GemmOptions *opts, const elem_t *a, const elem_t *b, acc_t *acc)
{
    memset(acc, 0, (size_t)opts->m * (size_t)opts->n * sizeof(acc_t));
    for (int i = 0; i < opts->m; i++) {
        for (int j = 0; j < opts->n; j++) {
            int32_t sum = 0;
            for (int kk = 0; kk < opts->k; kk++) {
                sum +=
                    (int32_t)a[(size_t)i * (size_t)opts->k + (size_t)kk] *
                    (int32_t)b[(size_t)kk * (size_t)opts->n + (size_t)j];
            }
            acc[(size_t)i * (size_t)opts->n + (size_t)j] = (acc_t)sum;
        }
    }
}

static void run_cpu_int8_chunk(
    const GemmOptions *opts,
    const elem_t *a_chunk,
    const elem_t *b_chunk,
    int chunk_rows,
    int chunk_cols,
    acc_t *chunk_acc)
{
    memset(
        chunk_acc,
        0,
        (size_t)GEMMINI_SAFE_MAX_I * (size_t)GEMMINI_SAFE_MAX_J * sizeof(acc_t)
    );
    for (int row = 0; row < chunk_rows; row++) {
        for (int col = 0; col < chunk_cols; col++) {
            int32_t sum = 0;
            for (int kk = 0; kk < opts->k; kk++) {
                sum +=
                    (int32_t)a_chunk[(size_t)row * (size_t)opts->k + (size_t)kk] *
                    (int32_t)b_chunk[(size_t)kk * (size_t)opts->n + (size_t)col];
            }
            chunk_acc[(size_t)row * (size_t)GEMMINI_SAFE_MAX_J + (size_t)col] =
                (acc_t)sum;
        }
    }
}

static void apply_polybench_update(
    const GemmOptions *opts,
    double alpha,
    double beta,
    double *c,
    const acc_t *gemm_acc)
{
    const size_t count = (size_t)opts->m * (size_t)opts->n;

    for (size_t idx = 0; idx < count; idx++) {
        c[idx] = beta * c[idx] + alpha * (double)gemm_acc[idx];
    }
}

static void compare_outputs(
    const GemmOptions *opts,
    const double *actual,
    const double *reference,
    double *max_abs_diff,
    int *mismatches)
{
    const size_t count = (size_t)opts->m * (size_t)opts->n;

    *max_abs_diff = 0.0;
    *mismatches = 0;
    for (size_t idx = 0; idx < count; idx++) {
        const double diff = fabs(actual[idx] - reference[idx]);
        if (diff > *max_abs_diff) {
            *max_abs_diff = diff;
        }
        if (diff > 1.0e-6) {
            (*mismatches)++;
        }
    }
}

static int count_acc_mismatches(
    const GemmOptions *opts,
    const acc_t *actual,
    const acc_t *reference,
    double *max_abs_diff)
{
    const size_t count = (size_t)opts->m * (size_t)opts->n;
    int mismatches = 0;

    for (size_t idx = 0; idx < count; idx++) {
        const double diff = fabs((double)actual[idx] - (double)reference[idx]);
        if (diff > *max_abs_diff) {
            *max_abs_diff = diff;
        }
        if (diff != 0.0) {
            mismatches++;
        }
    }
    return mismatches;
}

static void prepare_gemmini_state(void)
{
    gemmini_flush(0);
    gemmini_fence();
}

static int pin_thread_to_vcpu(int guest_cpu)
{
    cpu_set_t cpuset;
    int rc;

    CPU_ZERO(&cpuset);
    CPU_SET(guest_cpu, &cpuset);
    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fatalf("failed to pin thread to guest vCPU %d: %s", guest_cpu, strerror(rc));
    }
    return sched_getcpu();
}

static inline void cpu_memory_fence(void)
{
    asm volatile("fence rw, rw" : : : "memory");
}

static int min_int(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

static void profile_add_ms(GemminiLatencyProfile *profile, double *field, double ms)
{
    if (profile != NULL) {
        *field += ms;
    }
}

static void copy_chunk_acc_to_full(
    const GemmOptions *opts,
    int row_base,
    int col_base,
    int chunk_rows,
    int chunk_cols,
    int chunk_stride,
    acc_t *acc,
    const acc_t *chunk_acc)
{
    for (int row = 0; row < chunk_rows; row++) {
        acc_t *dst =
            acc + (size_t)(row_base + row) * (size_t)opts->n + (size_t)col_base;
        const acc_t *src = chunk_acc + (size_t)row * (size_t)chunk_stride;
        memcpy(dst, src, (size_t)chunk_cols * sizeof(acc_t));
    }
}

static void run_gemmini_chunk(
    const GemmOptions *opts,
    const elem_t *a_chunk,
    const elem_t *b_chunk,
    int chunk_rows,
    int chunk_cols,
    acc_t *chunk_acc,
    acc_t *zero_bias,
    GemminiLatencyProfile *profile)
{
    const double chunk_start_ms = now_ms();
    double t0_ms;
    double t1_ms;

    if (profile != NULL) {
        profile->chunks_total++;
    }

    t0_ms = now_ms();
    memset(
        chunk_acc,
        0,
        (size_t)GEMMINI_SAFE_MAX_I * (size_t)GEMMINI_SAFE_MAX_J * sizeof(acc_t)
    );
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->chunk_acc_memset_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    memset(
        zero_bias,
        0,
        (size_t)GEMMINI_SAFE_MAX_I * (size_t)GEMMINI_SAFE_MAX_J * sizeof(acc_t)
    );
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->zero_bias_memset_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    cpu_memory_fence();
    gemmini_flush(0);
    gemmini_fence();
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->pre_fence_flush_ms += t1_ms - t0_ms;
    }

    /* Keep WS accumulator initialization deterministic; D=NULL was unstable on MEDIUM. */
    t0_ms = now_ms();
    tiled_matmul_auto(
        (size_t)chunk_rows,
        (size_t)chunk_cols,
        (size_t)opts->k,
        a_chunk,
        b_chunk,
        zero_bias,
        chunk_acc,
        (size_t)opts->k,
        (size_t)opts->n,
        (size_t)GEMMINI_SAFE_MAX_J,
        (size_t)GEMMINI_SAFE_MAX_J,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        NO_ACTIVATION,
        ACC_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        false,
        false,
        false,
        true,
        false,
        0,
        WS
    );
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->tiled_matmul_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    gemmini_fence();
    cpu_memory_fence();
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->post_fence_ms += t1_ms - t0_ms;
        profile->run_chunk_total_ms += now_ms() - chunk_start_ms;
    }
}

static void run_gemmini_direct_call(
    const GemmOptions *opts,
    const elem_t *a,
    const elem_t *b,
    acc_t *acc,
    acc_t *zero_bias,
    GemminiLatencyProfile *profile)
{
    const double call_start_ms = now_ms();
    const size_t acc_count = (size_t)opts->m * (size_t)opts->n;
    double t0_ms;
    double t1_ms;

    if (profile != NULL) {
        profile->chunks_total++;
    }

    t0_ms = now_ms();
    memset(acc, 0, acc_count * sizeof(acc_t));
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->full_acc_memset_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    memset(zero_bias, 0, acc_count * sizeof(acc_t));
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->zero_bias_memset_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    cpu_memory_fence();
    gemmini_flush(0);
    gemmini_fence();
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->pre_fence_flush_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    tiled_matmul_auto(
        (size_t)opts->m,
        (size_t)opts->n,
        (size_t)opts->k,
        a,
        b,
        zero_bias,
        acc,
        (size_t)opts->k,
        (size_t)opts->n,
        (size_t)opts->n,
        (size_t)opts->n,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        NO_ACTIVATION,
        ACC_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        false,
        false,
        false,
        true,
        false,
        0,
        WS
    );
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->tiled_matmul_ms += t1_ms - t0_ms;
    }

    t0_ms = now_ms();
    gemmini_fence();
    cpu_memory_fence();
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->post_fence_ms += t1_ms - t0_ms;
        profile->run_chunk_total_ms += now_ms() - call_start_ms;
    }
}

static void *gemmini_executor_main(void *arg)
{
    GemminiExecutor *executor = (GemminiExecutor *)arg;

    executor->pinned_cpu = pin_thread_to_vcpu(executor->target_vcpu);
    if (g_log_backend_calls) {
        printf("[backend-executor-start] backend=%s target_vcpu=%d cpu=%d\n",
               POLYBENCH_GEMM_BACKEND_NAME,
               executor->target_vcpu,
               executor->pinned_cpu);
        fflush(stdout);
    }
    prepare_gemmini_state();

    pthread_mutex_lock(&executor->mutex);
    for (;;) {
        while (!executor->has_job && !executor->stop) {
            pthread_cond_wait(&executor->cond, &executor->mutex);
        }
        if (executor->stop && !executor->has_job) {
            break;
        }

        const GemmOptions *opts = executor->opts;
        const elem_t *a_chunk = executor->a_chunk;
        const elem_t *b_chunk = executor->b_chunk;
        acc_t *chunk_acc = executor->chunk_acc;
        acc_t *zero_bias = executor->zero_bias;
        const int chunk_rows = executor->chunk_rows;
        const int chunk_cols = executor->chunk_cols;
        const int row_base = executor->row_base;
        const int col_base = executor->col_base;
        GemminiLatencyProfile *profile = executor->profile;
        pthread_mutex_unlock(&executor->mutex);

        if (opts->verbose_chunks) {
            printf("[backend-call] backend=%s row_base=%d col_base=%d rows=%d cols=%d cpu=%d\n",
                   POLYBENCH_GEMM_BACKEND_NAME,
                   row_base,
                   col_base,
                   chunk_rows,
                   chunk_cols,
                   sched_getcpu());
            fflush(stdout);
        }
#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
        run_gemmini_direct_call(opts, a_chunk, b_chunk, chunk_acc, zero_bias, profile);
#else
        run_gemmini_chunk(
            opts,
            a_chunk,
            b_chunk,
            chunk_rows,
            chunk_cols,
            chunk_acc,
            zero_bias,
            profile
        );
#endif

        pthread_mutex_lock(&executor->mutex);
        executor->has_job = false;
        executor->job_done = true;
        pthread_cond_broadcast(&executor->cond);
    }
    pthread_mutex_unlock(&executor->mutex);
    return NULL;
}

static void gemmini_executor_start(GemminiExecutor *executor, int target_vcpu)
{
    memset(executor, 0, sizeof(*executor));
    executor->target_vcpu = target_vcpu;
    executor->pinned_cpu = -1;
    pthread_mutex_init(&executor->mutex, NULL);
    pthread_cond_init(&executor->cond, NULL);
    if (pthread_create(&executor->thread, NULL, gemmini_executor_main, executor) != 0) {
        fatalf("failed to create Gemmini executor thread");
    }
}

static void gemmini_executor_stop(GemminiExecutor *executor)
{
    pthread_mutex_lock(&executor->mutex);
    executor->stop = true;
    pthread_cond_broadcast(&executor->cond);
    pthread_mutex_unlock(&executor->mutex);

    pthread_join(executor->thread, NULL);
    pthread_cond_destroy(&executor->cond);
    pthread_mutex_destroy(&executor->mutex);
}

static void gemmini_executor_run_chunk(
    GemminiExecutor *executor,
    const GemmOptions *opts,
    const elem_t *a_chunk,
    const elem_t *b_chunk,
    int chunk_rows,
    int chunk_cols,
    acc_t *chunk_acc,
    acc_t *zero_bias,
    int row_base,
    int col_base,
    GemminiLatencyProfile *profile)
{
    const double submit_start_ms = now_ms();

    pthread_mutex_lock(&executor->mutex);
    while (executor->has_job) {
        pthread_cond_wait(&executor->cond, &executor->mutex);
    }
    executor->opts = opts;
    executor->a_chunk = a_chunk;
    executor->b_chunk = b_chunk;
    executor->chunk_acc = chunk_acc;
    executor->zero_bias = zero_bias;
    executor->chunk_rows = chunk_rows;
    executor->chunk_cols = chunk_cols;
    executor->row_base = row_base;
    executor->col_base = col_base;
    executor->profile = profile;
    executor->job_done = false;
    executor->has_job = true;
    pthread_cond_broadcast(&executor->cond);

    while (!executor->job_done) {
        pthread_cond_wait(&executor->cond, &executor->mutex);
    }
    pthread_mutex_unlock(&executor->mutex);

    if (profile != NULL) {
        profile->executor_submit_wait_ms += now_ms() - submit_start_ms;
    }
}

static void run_gemmini_matmul(
    const GemmOptions *opts,
    const elem_t *a,
    const elem_t *b,
    acc_t *acc,
    acc_t *chunk_acc,
    acc_t *zero_bias,
    GemminiExecutor *executor,
    GemminiLatencyProfile *profile)
{
    double t0_ms;
    double t1_ms;

#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
    (void)chunk_acc;
    if (executor != NULL) {
        gemmini_executor_run_chunk(
            executor,
            opts,
            a,
            b,
            opts->m,
            opts->n,
            acc,
            zero_bias,
            0,
            0,
            profile
        );
    } else {
        run_gemmini_direct_call(opts, a, b, acc, zero_bias, profile);
    }
    return;
#endif

    t0_ms = now_ms();
    memset(acc, 0, (size_t)opts->m * (size_t)opts->n * sizeof(acc_t));
    t1_ms = now_ms();
    if (profile != NULL) {
        profile->full_acc_memset_ms += t1_ms - t0_ms;
    }

    for (int row_base = 0; row_base < opts->m; row_base += GEMMINI_SAFE_MAX_I) {
        const int chunk_rows = min_int(GEMMINI_SAFE_MAX_I, opts->m - row_base);
        const elem_t *a_chunk = a + (size_t)row_base * (size_t)opts->k;

        for (int col_base = 0; col_base < opts->n; col_base += GEMMINI_SAFE_MAX_J) {
            const int chunk_cols = min_int(GEMMINI_SAFE_MAX_J, opts->n - col_base);
            const elem_t *b_chunk = b + (size_t)col_base;

            if (executor != NULL) {
                gemmini_executor_run_chunk(
                    executor,
                    opts,
                    a_chunk,
                    b_chunk,
                    chunk_rows,
                    chunk_cols,
                    chunk_acc,
                    zero_bias,
                    row_base,
                    col_base,
                    profile
                );
            } else {
                run_gemmini_chunk(
                    opts,
                    a_chunk,
                    b_chunk,
                    chunk_rows,
                    chunk_cols,
                    chunk_acc,
                    zero_bias,
                    profile
                );
            }
            t0_ms = now_ms();
            copy_chunk_acc_to_full(
                opts,
                row_base,
                col_base,
                chunk_rows,
                chunk_cols,
                GEMMINI_SAFE_MAX_J,
                acc,
                chunk_acc
            );
            t1_ms = now_ms();
            if (profile != NULL) {
                profile->copy_chunk_acc_ms += t1_ms - t0_ms;
            }
        }
    }
}

static void run_cpu_int8_chunked_matmul(
    const GemmOptions *opts,
    const elem_t *a,
    const elem_t *b,
    acc_t *acc,
    acc_t *chunk_acc)
{
    memset(acc, 0, (size_t)opts->m * (size_t)opts->n * sizeof(acc_t));

    for (int row_base = 0; row_base < opts->m; row_base += GEMMINI_SAFE_MAX_I) {
        const int chunk_rows = min_int(GEMMINI_SAFE_MAX_I, opts->m - row_base);
        const elem_t *a_chunk = a + (size_t)row_base * (size_t)opts->k;

        for (int col_base = 0; col_base < opts->n; col_base += GEMMINI_SAFE_MAX_J) {
            const int chunk_cols = min_int(GEMMINI_SAFE_MAX_J, opts->n - col_base);
            const elem_t *b_chunk = b + (size_t)col_base;

            run_cpu_int8_chunk(
                opts,
                a_chunk,
                b_chunk,
                chunk_rows,
                chunk_cols,
                chunk_acc
            );
            copy_chunk_acc_to_full(
                opts,
                row_base,
                col_base,
                chunk_rows,
                chunk_cols,
                GEMMINI_SAFE_MAX_J,
                acc,
                chunk_acc
            );
        }
    }
}

static void write_gemmini_metrics(
    const GemmOptions *opts,
    double elapsed_ms,
    double checksum,
    double reference_checksum,
    double checksum_abs_diff,
    int checksum_match,
    double acc_max_abs_diff,
    long long acc_mismatches,
    int acc_mismatched_reps,
    double output_max_abs_diff,
    int output_mismatches)
{
    FILE *fp = open_output_file(opts, "metrics.json");

    fprintf(fp, "{\n");
    fprintf(fp, "  \"workload\": \"polybench_gemm\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", POLYBENCH_GEMM_BACKEND_MODE);
    fprintf(fp, "  \"backend\": \"%s\",\n", POLYBENCH_GEMM_BACKEND_NAME);
    fprintf(fp, "  \"operation\": \"C=beta*C+alpha*sum_k(A_int8*B_int8)\",\n");
    fprintf(fp, "  \"dataset\": \"%s\",\n", POLYBENCH_GEMM_DATASET_NAME);
    fprintf(fp, "  \"m\": %d,\n", opts->m);
    fprintf(fp, "  \"n\": %d,\n", opts->n);
    fprintf(fp, "  \"k\": %d,\n", opts->k);
    fprintf(fp, "  \"reps\": %d,\n", opts->reps);
    fprintf(fp, "  \"checked_reps\": %d,\n", opts->reps);
    fprintf(fp, "  \"elapsed_ms\": %.6f,\n", elapsed_ms);
    fprintf(fp, "  \"avg_ms_per_rep\": %.6f,\n", elapsed_ms / (double)opts->reps);
    fprintf(fp, "  \"checksum\": %.17g,\n", checksum);
    fprintf(fp, "  \"reference_checksum\": %.17g,\n", reference_checksum);
    fprintf(fp, "  \"checksum_abs_diff\": %.17g,\n", checksum_abs_diff);
    fprintf(fp, "  \"checksum_match\": %s,\n", checksum_match ? "true" : "false");
    fprintf(fp, "  \"acc_checked_reps\": %d,\n", opts->reps);
    fprintf(fp, "  \"acc_mismatched_reps\": %d,\n", acc_mismatched_reps);
    fprintf(fp, "  \"acc_max_abs_diff\": %.17g,\n", acc_max_abs_diff);
    fprintf(fp, "  \"acc_mismatches\": %lld,\n", acc_mismatches);
    fprintf(fp, "  \"output_max_abs_diff\": %.17g,\n", output_max_abs_diff);
    fprintf(fp, "  \"output_mismatches\": %d,\n", output_mismatches);
    fprintf(fp, "  \"threads\": %d,\n", opts->threads);
    fprintf(fp, "  \"vcpu_offset\": %d,\n", opts->vcpu_offset);
    fprintf(fp, "  \"serialize_gemmini\": %s,\n", opts->serialize_gemmini ? "true" : "false");
    fprintf(fp, "  \"verbose_chunks\": %s,\n", opts->verbose_chunks ? "true" : "false");
    fprintf(fp, "  \"gemmini_vcpu\": %d,\n", opts->gemmini_vcpu);
    fprintf(fp, "  \"housekeeping_vcpu\": %d\n", opts->housekeeping_vcpu);
    fprintf(fp, "}\n");
    fclose(fp);
}

static void write_gemmini_power_metrics(
    const GemmOptions *opts,
    double elapsed_ms,
    double checksum,
    long long guard_acc)
{
    FILE *fp = open_output_file(opts, "metrics.json");

    fprintf(fp, "{\n");
    fprintf(fp, "  \"workload\": \"polybench_gemm\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", POLYBENCH_GEMM_BACKEND_MODE);
    fprintf(fp, "  \"backend\": \"%s\",\n", POLYBENCH_GEMM_BACKEND_NAME);
    fprintf(fp, "  \"operation\": \"C=beta*C+alpha*sum_k(A_int8*B_int8)\",\n");
    fprintf(fp, "  \"dataset\": \"%s\",\n", POLYBENCH_GEMM_DATASET_NAME);
    fprintf(fp, "  \"m\": %d,\n", opts->m);
    fprintf(fp, "  \"n\": %d,\n", opts->n);
    fprintf(fp, "  \"k\": %d,\n", opts->k);
    fprintf(fp, "  \"reps\": %d,\n", opts->reps);
    fprintf(fp, "  \"checked_reps\": 0,\n");
    fprintf(fp, "  \"correctness_check_enabled\": false,\n");
    fprintf(fp, "  \"elapsed_ms\": %.6f,\n", elapsed_ms);
    fprintf(fp, "  \"avg_ms_per_rep\": %.6f,\n", elapsed_ms / (double)opts->reps);
    fprintf(fp, "  \"checksum\": %.17g,\n", checksum);
    fprintf(fp, "  \"guard_acc\": %lld,\n", guard_acc);
    fprintf(fp, "  \"threads\": %d,\n", opts->threads);
    fprintf(fp, "  \"vcpu_offset\": %d,\n", opts->vcpu_offset);
    fprintf(fp, "  \"serialize_gemmini\": %s,\n", opts->serialize_gemmini ? "true" : "false");
    fprintf(fp, "  \"verbose_chunks\": %s,\n", opts->verbose_chunks ? "true" : "false");
    fprintf(fp, "  \"gemmini_vcpu\": %d,\n", opts->gemmini_vcpu);
    fprintf(fp, "  \"housekeeping_vcpu\": %d\n", opts->housekeeping_vcpu);
    fprintf(fp, "}\n");
    fclose(fp);
}

#if POLYBENCH_GEMM_POWER_ONLY && !POLYBENCH_GEMM_USE_GEMMINI_BACKEND
typedef int8_t PowerElem;
typedef int32_t PowerAcc;

typedef struct {
    PowerAcc *chunk_acc;
} PowerBackendContext;

static void power_backend_init(PowerBackendContext *ctx, const GemmOptions *opts)
{
    (void)opts;
    memset(ctx, 0, sizeof(*ctx));
#if POLYBENCH_GEMM_CPU_CHUNKED_BACKEND
    ctx->chunk_acc = alloc_acc_matrix(
        GEMMINI_SAFE_MAX_I,
        GEMMINI_SAFE_MAX_J,
        "CPU chunk accumulator"
    );
#endif
}

static void power_backend_destroy(PowerBackendContext *ctx)
{
    free(ctx->chunk_acc);
}

static void power_backend_matmul(
    const GemmOptions *opts,
    const PowerElem *a,
    const PowerElem *b,
    PowerAcc *acc,
    PowerBackendContext *ctx)
{
#if POLYBENCH_GEMM_CPU_CHUNKED_BACKEND
    run_cpu_int8_chunked_matmul(opts, a, b, acc, ctx->chunk_acc);
#else
    (void)ctx;
    memset(acc, 0, (size_t)opts->m * (size_t)opts->n * sizeof(PowerAcc));
    for (int i = 0; i < opts->m; i++) {
        for (int j = 0; j < opts->n; j++) {
            int32_t sum = 0;
            for (int kk = 0; kk < opts->k; kk++) {
                sum +=
                    (int32_t)a[(size_t)i * (size_t)opts->k + (size_t)kk] *
                    (int32_t)b[(size_t)kk * (size_t)opts->n + (size_t)j];
            }
            acc[(size_t)i * (size_t)opts->n + (size_t)j] = (PowerAcc)sum;
        }
    }
#endif
}

static void power_backend_note_update(PowerBackendContext *ctx, double update_ms)
{
    (void)ctx;
    (void)update_ms;
}

static void power_backend_write_profile(
    FILE *fp,
    const GemmOptions *opts,
    const PowerBackendContext *ctx,
    double elapsed_ms)
{
    (void)opts;
    (void)ctx;
    (void)elapsed_ms;
    fprintf(fp, "  \"profile_supported\": false");
}
#endif

#if POLYBENCH_GEMM_POWER_ONLY && POLYBENCH_GEMM_USE_GEMMINI_BACKEND
typedef elem_t PowerElem;
typedef acc_t PowerAcc;

typedef struct {
    GemminiExecutor executor;
    GemminiExecutor *executor_ptr;
    acc_t *chunk_acc;
    acc_t *zero_bias;
    GemminiLatencyProfile profile;
    GemminiLatencyProfile *profile_ptr;
} PowerBackendContext;

static void power_backend_init(PowerBackendContext *ctx, const GemmOptions *opts)
{
    memset(ctx, 0, sizeof(*ctx));
    g_log_backend_calls = opts->verbose_chunks;
#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
    ctx->zero_bias = alloc_acc_matrix(
        (size_t)opts->m,
        (size_t)opts->n,
        "Gemmini direct zero bias"
    );
#else
    ctx->chunk_acc = alloc_acc_matrix(
        GEMMINI_SAFE_MAX_I,
        GEMMINI_SAFE_MAX_J,
        "Gemmini chunk accumulator"
    );
    ctx->zero_bias = alloc_acc_matrix(
        GEMMINI_SAFE_MAX_I,
        GEMMINI_SAFE_MAX_J,
        "Gemmini zero bias"
    );
#endif

    if (opts->gemmini_vcpu >= 0) {
        gemmini_executor_start(&ctx->executor, opts->gemmini_vcpu);
        ctx->executor_ptr = &ctx->executor;
    } else {
        prepare_gemmini_state();
    }
    if (opts->profile_latency) {
        ctx->profile_ptr = &ctx->profile;
    }
}

static void power_backend_destroy(PowerBackendContext *ctx)
{
    if (ctx->executor_ptr != NULL) {
        gemmini_executor_stop(ctx->executor_ptr);
    }
    free(ctx->chunk_acc);
    free(ctx->zero_bias);
}

static void power_backend_matmul(
    const GemmOptions *opts,
    const PowerElem *a,
    const PowerElem *b,
    PowerAcc *acc,
    PowerBackendContext *ctx)
{
    const double t0_ms = now_ms();

    run_gemmini_matmul(
        opts,
        a,
        b,
        acc,
        ctx->chunk_acc,
        ctx->zero_bias,
        ctx->executor_ptr,
        ctx->profile_ptr
    );
    if (ctx->profile_ptr != NULL) {
        ctx->profile.backend_matmul_ms += now_ms() - t0_ms;
    }
}

static void power_backend_note_update(PowerBackendContext *ctx, double update_ms)
{
    if (ctx->profile_ptr != NULL) {
        ctx->profile.polybench_update_ms += update_ms;
    }
}

static double pct_of(double value, double total)
{
    return total > 0.0 ? (value / total) * 100.0 : 0.0;
}

static void power_backend_write_profile(
    FILE *fp,
    const GemmOptions *opts,
    const PowerBackendContext *ctx,
    double elapsed_ms)
{
    const GemminiLatencyProfile *p = ctx->profile_ptr;
    double executor_overhead_ms = 0.0;
    double fence_flush_total_ms = 0.0;
    double measured_profile_ms = 0.0;
    double common_backend_wall_ms = 0.0;
    double common_acc_memset_ms = 0.0;
    double common_other_gemm_ms = 0.0;
    long long common_macs_total = 0;

    if (p == NULL) {
        fprintf(fp, "  \"profile_supported\": true");
        return;
    }

    executor_overhead_ms = p->executor_submit_wait_ms - p->run_chunk_total_ms;
    if (executor_overhead_ms < 0.0) {
        executor_overhead_ms = 0.0;
    }
    fence_flush_total_ms = p->pre_fence_flush_ms + p->post_fence_ms;
    measured_profile_ms =
        p->full_acc_memset_ms +
        p->executor_submit_wait_ms +
        p->copy_chunk_acc_ms +
        p->polybench_update_ms;
    if (ctx->executor_ptr == NULL) {
        measured_profile_ms =
            p->full_acc_memset_ms +
            p->run_chunk_total_ms +
            p->copy_chunk_acc_ms +
            p->polybench_update_ms;
    }
    common_backend_wall_ms = ctx->executor_ptr != NULL ? p->executor_submit_wait_ms : p->run_chunk_total_ms;
    common_acc_memset_ms = p->full_acc_memset_ms + p->chunk_acc_memset_ms + p->zero_bias_memset_ms;
    common_other_gemm_ms = p->backend_matmul_ms - common_backend_wall_ms - p->full_acc_memset_ms - p->copy_chunk_acc_ms;
    if (common_other_gemm_ms < 0.0) {
        common_other_gemm_ms = 0.0;
    }
    common_macs_total = (long long)opts->reps * (long long)opts->m * (long long)opts->n * (long long)opts->k;

    fprintf(fp, "  \"profile_supported\": true,\n");
    fprintf(fp, "  \"profile_mode\": \"%s\",\n", ctx->executor_ptr != NULL ? "current_executor" : "inline");
    fprintf(fp, "  \"profile_chunks_total\": %lld,\n", p->chunks_total);
#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
    fprintf(fp, "  \"profile_expected_chunks_per_rep\": %d,\n", 1);
    fprintf(fp, "  \"profile_expected_chunks_total\": %d,\n", opts->reps);
#else
    fprintf(fp, "  \"profile_expected_chunks_per_rep\": %d,\n",
            ((opts->m + GEMMINI_SAFE_MAX_I - 1) / GEMMINI_SAFE_MAX_I) *
            ((opts->n + GEMMINI_SAFE_MAX_J - 1) / GEMMINI_SAFE_MAX_J));
    fprintf(fp, "  \"profile_expected_chunks_total\": %d,\n",
            opts->reps *
            ((opts->m + GEMMINI_SAFE_MAX_I - 1) / GEMMINI_SAFE_MAX_I) *
            ((opts->n + GEMMINI_SAFE_MAX_J - 1) / GEMMINI_SAFE_MAX_J));
#endif
    fprintf(fp, "  \"profile_backend_matmul_ms\": %.6f,\n", p->backend_matmul_ms);
    fprintf(fp, "  \"profile_executor_submit_wait_ms\": %.6f,\n", p->executor_submit_wait_ms);
    fprintf(fp, "  \"profile_executor_overhead_est_ms\": %.6f,\n", executor_overhead_ms);
    fprintf(fp, "  \"profile_run_chunk_total_ms\": %.6f,\n", p->run_chunk_total_ms);
    fprintf(fp, "  \"profile_full_acc_memset_ms\": %.6f,\n", p->full_acc_memset_ms);
    fprintf(fp, "  \"profile_chunk_acc_memset_ms\": %.6f,\n", p->chunk_acc_memset_ms);
    fprintf(fp, "  \"profile_zero_bias_memset_ms\": %.6f,\n", p->zero_bias_memset_ms);
    fprintf(fp, "  \"profile_pre_fence_flush_ms\": %.6f,\n", p->pre_fence_flush_ms);
    fprintf(fp, "  \"profile_tiled_matmul_ms\": %.6f,\n", p->tiled_matmul_ms);
    fprintf(fp, "  \"profile_post_fence_ms\": %.6f,\n", p->post_fence_ms);
    fprintf(fp, "  \"profile_fence_flush_total_ms\": %.6f,\n", fence_flush_total_ms);
    fprintf(fp, "  \"profile_copy_chunk_acc_ms\": %.6f,\n", p->copy_chunk_acc_ms);
    fprintf(fp, "  \"profile_polybench_update_ms\": %.6f,\n", p->polybench_update_ms);
    fprintf(fp, "  \"profile_measured_profile_ms\": %.6f,\n", measured_profile_ms);
    fprintf(fp, "  \"profile_tiled_matmul_pct_of_elapsed\": %.6f,\n", pct_of(p->tiled_matmul_ms, elapsed_ms));
    fprintf(fp, "  \"profile_fence_flush_pct_of_elapsed\": %.6f,\n", pct_of(fence_flush_total_ms, elapsed_ms));
    fprintf(fp, "  \"profile_executor_overhead_pct_of_elapsed\": %.6f,\n", pct_of(executor_overhead_ms, elapsed_ms));
    fprintf(fp, "  \"profile_copy_chunk_acc_pct_of_elapsed\": %.6f,\n", pct_of(p->copy_chunk_acc_ms, elapsed_ms));
    fprintf(fp, "  \"profile_polybench_update_pct_of_elapsed\": %.6f,\n", pct_of(p->polybench_update_ms, elapsed_ms));
    fprintf(fp, "  \"profile_measured_profile_pct_of_elapsed\": %.6f,\n", pct_of(measured_profile_ms, elapsed_ms));
    fprintf(fp, "  \"backend_profile\": {\n");
    fprintf(fp, "    \"schema\": \"gemmini_backend_profile_v1\",\n");
    fprintf(fp, "    \"backend\": \"%s\",\n", POLYBENCH_GEMM_BACKEND_NAME);
    fprintf(fp, "    \"operation\": \"polybench_gemm\",\n");
    fprintf(fp, "    \"mode\": \"%s\",\n", ctx->executor_ptr != NULL ? "current_executor" : "inline");
    fprintf(fp, "    \"linear_calls\": %d,\n", opts->reps);
    fprintf(fp, "    \"chunks\": %lld,\n", p->chunks_total);
    fprintf(fp, "    \"rows\": %lld,\n", (long long)opts->reps * (long long)opts->m);
    fprintf(fp, "    \"output_elements\": %lld,\n", (long long)opts->reps * (long long)opts->m * (long long)opts->n);
    fprintf(fp, "    \"macs\": %lld,\n", common_macs_total);
    fprintf(fp, "    \"linear_total_ms\": %.6f,\n", p->backend_matmul_ms);
    fprintf(fp, "    \"quantize_ms\": 0.000000,\n");
    fprintf(fp, "    \"backend_wall_ms\": %.6f,\n", common_backend_wall_ms);
    fprintf(fp, "    \"backend_active_ms\": %.6f,\n", p->run_chunk_total_ms);
    fprintf(fp, "    \"backend_overhead_ms\": %.6f,\n", executor_overhead_ms);
    fprintf(fp, "    \"tiled_matmul_ms\": %.6f,\n", p->tiled_matmul_ms);
    fprintf(fp, "    \"fence_flush_ms\": %.6f,\n", fence_flush_total_ms);
    fprintf(fp, "    \"acc_memset_ms\": %.6f,\n", common_acc_memset_ms);
    fprintf(fp, "    \"copy_chunk_acc_ms\": %.6f,\n", p->copy_chunk_acc_ms);
    fprintf(fp, "    \"postprocess_ms\": %.6f,\n", p->polybench_update_ms);
    fprintf(fp, "    \"other_linear_ms\": %.6f,\n", common_other_gemm_ms);
    fprintf(fp, "    \"linear_pct_of_elapsed\": %.6f,\n", pct_of(p->backend_matmul_ms, elapsed_ms));
    fprintf(fp, "    \"backend_wall_pct_of_elapsed\": %.6f,\n", pct_of(common_backend_wall_ms, elapsed_ms));
    fprintf(fp, "    \"backend_active_pct_of_elapsed\": %.6f,\n", pct_of(p->run_chunk_total_ms, elapsed_ms));
    fprintf(fp, "    \"tiled_matmul_pct_of_elapsed\": %.6f,\n", pct_of(p->tiled_matmul_ms, elapsed_ms));
    fprintf(fp, "    \"fence_flush_pct_of_elapsed\": %.6f,\n", pct_of(fence_flush_total_ms, elapsed_ms));
    fprintf(fp, "    \"backend_overhead_pct_of_elapsed\": %.6f\n", pct_of(executor_overhead_ms, elapsed_ms));
    fprintf(fp, "  }");
}

#endif

#if POLYBENCH_GEMM_POWER_ONLY
#include "polybench_gemm_power_common.h"

int main(int argc, char **argv)
{
    return polybench_gemm_run_power(
        argc,
        argv,
        POLYBENCH_GEMM_BACKEND_MODE,
        POLYBENCH_GEMM_BACKEND_NAME
    );
}
#else
int main(int argc, char **argv)
{
    GemmOptions opts;
    FILE *timeline;
    const double alpha = 1.5;
    const double beta = 1.2;
    elem_t *a;
    elem_t *b;
    acc_t *gemmini_acc;
    acc_t *reference_acc;
    acc_t *chunk_acc;
    acc_t *zero_bias;
    double *c;
    double *reference_c;
    double t0_ms;
    double elapsed_ms;
    double checksum;
    double reference_checksum;
    double checksum_abs_diff;
    double acc_max_abs_diff;
    double output_max_abs_diff;
    double rep_start_ms;
    long long acc_mismatches;
    int acc_mismatched_reps;
    int rep_acc_mismatches;
    int checksum_match;
    int output_mismatches;
    int main_pinned_cpu;
    long long guard_acc;
    GemminiExecutor gemmini_executor;
    GemminiExecutor *gemmini_executor_ptr = NULL;

    parse_gemm_options(argc, argv, &opts);
    g_log_backend_calls = opts.verbose_chunks;
    main_pinned_cpu = pin_thread_to_vcpu(opts.vcpu_offset);
    printf("[main-pin] target_vcpu=%d cpu=%d\n", opts.vcpu_offset, main_pinned_cpu);
    fflush(stdout);

    if (opts.gemmini_vcpu >= 0) {
        gemmini_executor_start(&gemmini_executor, opts.gemmini_vcpu);
        gemmini_executor_ptr = &gemmini_executor;
    } else {
        prepare_gemmini_state();
    }

    mkdir_p(opts.output_dir);

    timeline = open_output_file(&opts, "timeline.log");
    write_config(&opts, POLYBENCH_GEMM_BACKEND_MODE);

    a = alloc_elem_matrix((size_t)opts.m, (size_t)opts.k, "A");
    b = alloc_elem_matrix((size_t)opts.k, (size_t)opts.n, "B");
    gemmini_acc = alloc_acc_matrix((size_t)opts.m, (size_t)opts.n, "Gemmini accumulator");
#if !POLYBENCH_GEMM_POWER_ONLY
    reference_acc = alloc_acc_matrix((size_t)opts.m, (size_t)opts.n, "CPU reference accumulator");
#else
    reference_acc = NULL;
#endif
#if POLYBENCH_GEMM_DIRECT_GEMMINI_BACKEND
    chunk_acc = NULL;
    zero_bias = alloc_acc_matrix((size_t)opts.m, (size_t)opts.n, "Gemmini direct zero bias");
#else
    chunk_acc = alloc_acc_matrix(GEMMINI_SAFE_MAX_I, GEMMINI_SAFE_MAX_J, "Gemmini chunk accumulator");
    zero_bias = alloc_acc_matrix(GEMMINI_SAFE_MAX_I, GEMMINI_SAFE_MAX_J, "Gemmini zero bias");
#endif
    c = alloc_matrix((size_t)opts.m, (size_t)opts.n, "C");
#if !POLYBENCH_GEMM_POWER_ONLY
    reference_c = alloc_matrix((size_t)opts.m, (size_t)opts.n, "CPU reference C");
#else
    reference_c = NULL;
#endif

    t0_ms = now_ms();
    emit_timeline(timeline, "init_begin", -1, t0_ms);
    init_gemmini_inputs(opts.m, opts.n, opts.k, a, b, c);
#if !POLYBENCH_GEMM_POWER_ONLY
    memcpy(reference_c, c, (size_t)opts.m * (size_t)opts.n * sizeof(double));
    run_cpu_int8_matmul(&opts, a, b, reference_acc);
#endif
    emit_timeline(timeline, "init_end", -1, t0_ms);

    elapsed_ms = 0.0;
    acc_max_abs_diff = 0.0;
    acc_mismatches = 0;
    acc_mismatched_reps = 0;
    for (int rep = 0; rep < opts.reps; rep++) {
        emit_timeline(timeline, "gemm_begin", rep, t0_ms);
        rep_start_ms = now_ms();
        run_gemmini_matmul(
            &opts,
            a,
            b,
            gemmini_acc,
            chunk_acc,
            zero_bias,
            gemmini_executor_ptr,
            NULL
        );
        apply_polybench_update(&opts, alpha, beta, c, gemmini_acc);
        elapsed_ms += now_ms() - rep_start_ms;
        emit_timeline(timeline, "gemm_end", rep, t0_ms);
#if !POLYBENCH_GEMM_POWER_ONLY
        apply_polybench_update(&opts, alpha, beta, reference_c, reference_acc);
        rep_acc_mismatches = count_acc_mismatches(
            &opts,
            gemmini_acc,
            reference_acc,
            &acc_max_abs_diff
        );
        acc_mismatches += rep_acc_mismatches;
        if (rep_acc_mismatches > 0) {
            acc_mismatched_reps++;
        }
#endif
    }

    if (opts.housekeeping_vcpu >= 0) {
        int housekeeping_cpu = pin_thread_to_vcpu(opts.housekeeping_vcpu);

        emit_timeline(timeline, "post_gemm_repin", -1, t0_ms);
        printf("[main-repin] reason=post_gemm target_vcpu=%d cpu=%d\n",
               opts.housekeeping_vcpu,
               housekeeping_cpu);
        fflush(stdout);
    }

#if POLYBENCH_GEMM_POWER_ONLY
    checksum = checksum_matrix(c, (size_t)opts.m, (size_t)opts.n);
    guard_acc = (long long)gemmini_acc[(size_t)(opts.m - 1) * (size_t)opts.n + (size_t)(opts.n - 1)];
    write_checksum(&opts, checksum);
    write_gemmini_power_metrics(&opts, elapsed_ms, checksum, guard_acc);
    emit_timeline(timeline, "done", -1, t0_ms);

    printf("[done] mode=%s backend=%s output_dir=%s elapsed_ms=%.6f checked_reps=0 vcpu_offset=%d gemmini_vcpu=%d checksum=%.17g guard_acc=%lld\n",
           POLYBENCH_GEMM_BACKEND_MODE,
           POLYBENCH_GEMM_BACKEND_NAME,
           opts.output_dir,
           elapsed_ms,
           opts.vcpu_offset,
           opts.gemmini_vcpu,
           checksum,
           guard_acc);
#else
    checksum = checksum_matrix(c, (size_t)opts.m, (size_t)opts.n);
    reference_checksum = checksum_matrix(reference_c, (size_t)opts.m, (size_t)opts.n);
    checksum_abs_diff = fabs(checksum - reference_checksum);
    checksum_match = checksum_abs_diff <= 1.0e-6;
    compare_outputs(&opts, c, reference_c, &output_max_abs_diff, &output_mismatches);
    write_checksum(&opts, checksum);
    write_gemmini_metrics(
        &opts,
        elapsed_ms,
        checksum,
        reference_checksum,
        checksum_abs_diff,
        checksum_match,
        acc_max_abs_diff,
        acc_mismatches,
        acc_mismatched_reps,
        output_max_abs_diff,
        output_mismatches
    );
    emit_timeline(timeline, "done", -1, t0_ms);

    printf("[done] mode=%s output_dir=%s elapsed_ms=%.6f checked_reps=%d vcpu_offset=%d gemmini_vcpu=%d checksum=%.17g reference_checksum=%.17g checksum_match=%s acc_mismatches=%lld output_mismatches=%d\n",
           POLYBENCH_GEMM_BACKEND_MODE,
           opts.output_dir,
           elapsed_ms,
           opts.reps,
           opts.vcpu_offset,
           opts.gemmini_vcpu,
           checksum,
           reference_checksum,
           checksum_match ? "true" : "false",
           acc_mismatches,
           output_mismatches);
#endif

    if (gemmini_executor_ptr != NULL) {
        gemmini_executor_stop(gemmini_executor_ptr);
    }

    fclose(timeline);
    free(a);
    free(b);
    free(gemmini_acc);
    free(reference_acc);
    free(chunk_acc);
    free(zero_bias);
    free(c);
    free(reference_c);
    return 0;
}
#endif
