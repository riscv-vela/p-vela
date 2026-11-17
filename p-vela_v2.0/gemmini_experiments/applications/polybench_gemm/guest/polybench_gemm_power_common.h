#ifndef POLYBENCH_GEMM_POWER_COMMON_H
#define POLYBENCH_GEMM_POWER_COMMON_H

#define POLYBENCH_GEMM_POWER_OPERATION "C=beta*C+alpha*sum_k(A_int8*B_int8)"
#define POLYBENCH_GEMM_ALPHA 1.5
#define POLYBENCH_GEMM_BETA 1.2

typedef struct {
    PowerElem *a;
    PowerElem *b;
    PowerAcc *acc;
    double *c;
} PolybenchPowerWorkspace;

static inline void power_qemu_count_marker(void)
{
#if defined(__riscv)
    asm volatile("fence i, i" : : : "memory");
#else
    asm volatile("" : : : "memory");
#endif
}

static int power_pin_thread_to_vcpu(int guest_cpu)
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

static PowerElem power_quantize_unit_interval(double value)
{
    long q = lround((value * 2.0 - 1.0) * 63.0);

    if (q > 127) {
        q = 127;
    } else if (q < -128) {
        q = -128;
    }
    return (PowerElem)q;
}

static void *power_aligned_alloc_or_die(size_t alignment, size_t bytes, const char *name)
{
    void *ptr = NULL;
    const int rc = posix_memalign(&ptr, alignment, bytes);

    if (rc != 0 || ptr == NULL) {
        fatalf("failed to allocate %s: %s", name, strerror(rc));
    }
    memset(ptr, 0, bytes);
    return ptr;
}

static PowerElem *power_alloc_elem_matrix(size_t rows, size_t cols, const char *name)
{
    size_t count;

    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols) {
        fatalf("invalid elem matrix size for %s", name);
    }
    count = rows * cols;
    if (count > SIZE_MAX / sizeof(PowerElem)) {
        fatalf("elem matrix allocation is too large for %s", name);
    }
    return (PowerElem *)power_aligned_alloc_or_die(64, count * sizeof(PowerElem), name);
}

static PowerAcc *power_alloc_acc_matrix(size_t rows, size_t cols, const char *name)
{
    size_t count;

    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols) {
        fatalf("invalid acc matrix size for %s", name);
    }
    count = rows * cols;
    if (count > SIZE_MAX / sizeof(PowerAcc)) {
        fatalf("acc matrix allocation is too large for %s", name);
    }
    return (PowerAcc *)power_aligned_alloc_or_die(64, count * sizeof(PowerAcc), name);
}

static void power_workspace_init(const GemmOptions *opts, PolybenchPowerWorkspace *ws)
{
    ws->a = power_alloc_elem_matrix((size_t)opts->m, (size_t)opts->k, "A");
    ws->b = power_alloc_elem_matrix((size_t)opts->k, (size_t)opts->n, "B");
    ws->acc = power_alloc_acc_matrix((size_t)opts->m, (size_t)opts->n, "accumulator");
    ws->c = alloc_matrix((size_t)opts->m, (size_t)opts->n, "C");
}

static void power_workspace_free(PolybenchPowerWorkspace *ws)
{
    free(ws->a);
    free(ws->b);
    free(ws->acc);
    free(ws->c);
}

static void power_init_inputs(
    int m,
    int n,
    int k,
    PowerElem *a,
    PowerElem *b,
    double *c)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[(size_t)i * (size_t)n + (size_t)j] =
                (double)((i * j + 1) % m) / (double)m;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            const double value = (double)(i * (kk + 1) % k) / (double)k;
            a[(size_t)i * (size_t)k + (size_t)kk] = power_quantize_unit_interval(value);
        }
    }

    for (int kk = 0; kk < k; kk++) {
        for (int j = 0; j < n; j++) {
            const double value = (double)(kk * (j + 2) % n) / (double)n;
            b[(size_t)kk * (size_t)n + (size_t)j] = power_quantize_unit_interval(value);
        }
    }
}

static void power_apply_polybench_update(
    const GemmOptions *opts,
    double *c,
    const PowerAcc *acc)
{
    const size_t count = (size_t)opts->m * (size_t)opts->n;

    for (size_t idx = 0; idx < count; idx++) {
        c[idx] = POLYBENCH_GEMM_BETA * c[idx] +
            POLYBENCH_GEMM_ALPHA * (double)acc[idx];
    }
}

static long long power_guard_acc(const GemmOptions *opts, const PowerAcc *acc)
{
    return (long long)acc[(size_t)(opts->m - 1) * (size_t)opts->n + (size_t)(opts->n - 1)];
}

static void power_write_metrics(
    const GemmOptions *opts,
    const char *mode,
    const char *backend,
    double elapsed_ms,
    double checksum,
    int pinned_cpu,
    long long guard_acc,
    const PowerBackendContext *backend_ctx)
{
    FILE *fp = open_output_file(opts, "metrics.json");

    fprintf(fp, "{\n");
    fprintf(fp, "  \"workload\": \"polybench_gemm\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", mode);
    fprintf(fp, "  \"backend\": \"%s\",\n", backend);
    fprintf(fp, "  \"operation\": \"%s\",\n", POLYBENCH_GEMM_POWER_OPERATION);
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
    fprintf(fp, "  \"pinned_cpu\": %d,\n", pinned_cpu);
    fprintf(fp, "  \"gemmini_vcpu\": %d,\n", opts->gemmini_vcpu);
    fprintf(fp, "  \"housekeeping_vcpu\": %d,\n", opts->housekeeping_vcpu);
    fprintf(fp, "  \"serialize_gemmini\": %s,\n", opts->serialize_gemmini ? "true" : "false");
    fprintf(fp, "  \"verbose_chunks\": %s,\n", opts->verbose_chunks ? "true" : "false");
    fprintf(fp, "  \"profile_latency\": %s", opts->profile_latency ? "true" : "false");
    if (opts->profile_latency) {
        fprintf(fp, ",\n");
        power_backend_write_profile(fp, opts, backend_ctx, elapsed_ms);
        fprintf(fp, "\n");
    } else {
        fprintf(fp, "\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
}

static int polybench_gemm_run_power(
    int argc,
    char **argv,
    const char *mode,
    const char *backend)
{
    GemmOptions opts;
    PolybenchPowerWorkspace ws;
    PowerBackendContext backend_ctx;
    FILE *timeline;
    double t0_ms;
    double elapsed_ms;
    double rep_start_ms;
    double update_start_ms;
    double checksum;
    long long guard_acc;
    int main_pinned_cpu;

    parse_gemm_options(argc, argv, &opts);
    main_pinned_cpu = power_pin_thread_to_vcpu(opts.vcpu_offset);
    printf("[main-pin] target_vcpu=%d cpu=%d\n", opts.vcpu_offset, main_pinned_cpu);
    fflush(stdout);

    power_backend_init(&backend_ctx, &opts);

    mkdir_p(opts.output_dir);
    timeline = open_output_file(&opts, "timeline.log");
    write_config(&opts, mode);

    power_workspace_init(&opts, &ws);

    t0_ms = now_ms();
    emit_timeline(timeline, "init_begin", -1, t0_ms);
    power_init_inputs(opts.m, opts.n, opts.k, ws.a, ws.b, ws.c);
    emit_timeline(timeline, "init_end", -1, t0_ms);

    printf("[sample-start] order=0 idx=0 thread=0 cpu=%d\n", sched_getcpu());
    fflush(stdout);

    power_qemu_count_marker();
    elapsed_ms = 0.0;
    for (int rep = 0; rep < opts.reps; rep++) {
        emit_timeline(timeline, "gemm_begin", rep, t0_ms);
        rep_start_ms = now_ms();
        power_backend_matmul(&opts, ws.a, ws.b, ws.acc, &backend_ctx);
        update_start_ms = now_ms();
        power_apply_polybench_update(&opts, ws.c, ws.acc);
        power_backend_note_update(&backend_ctx, now_ms() - update_start_ms);
        elapsed_ms += now_ms() - rep_start_ms;
        emit_timeline(timeline, "gemm_end", rep, t0_ms);
    }
    power_qemu_count_marker();

    printf(
        "[sample-done] order=0 idx=0 thread=0 cpu=%d elapsed_ms=%.6f\n",
        sched_getcpu(),
        elapsed_ms
    );
    fflush(stdout);

    if (opts.housekeeping_vcpu >= 0) {
        int housekeeping_cpu = power_pin_thread_to_vcpu(opts.housekeeping_vcpu);

        emit_timeline(timeline, "post_gemm_repin", -1, t0_ms);
        printf("[main-repin] reason=post_gemm target_vcpu=%d cpu=%d\n",
               opts.housekeeping_vcpu,
               housekeeping_cpu);
        fflush(stdout);
    }

    checksum = checksum_matrix(ws.c, (size_t)opts.m, (size_t)opts.n);
    guard_acc = power_guard_acc(&opts, ws.acc);
    write_checksum(&opts, checksum);
    power_write_metrics(&opts, mode, backend, elapsed_ms, checksum, main_pinned_cpu, guard_acc, &backend_ctx);
    emit_timeline(timeline, "done", -1, t0_ms);
    printf(
        "[done] mode=%s backend=%s output_dir=%s elapsed_ms=%.6f reps=%d vcpu_offset=%d gemmini_vcpu=%d checksum=%.17g guard_acc=%lld\n",
        mode,
        backend,
        opts.output_dir,
        elapsed_ms,
        opts.reps,
        opts.vcpu_offset,
        opts.gemmini_vcpu,
        checksum,
        guard_acc
    );
    fflush(stdout);

    fclose(timeline);
    power_workspace_free(&ws);
    power_backend_destroy(&backend_ctx);
    return 0;
}

#endif
