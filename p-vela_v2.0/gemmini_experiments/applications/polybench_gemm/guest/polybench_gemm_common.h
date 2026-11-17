#ifndef POLYBENCH_GEMM_COMMON_H
#define POLYBENCH_GEMM_COMMON_H

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    int m;
    int n;
    int k;
    int reps;
    int threads;
    int vcpu_offset;
    int gemmini_vcpu;
    int housekeeping_vcpu;
    bool timeline;
    bool serialize_gemmini;
    bool verbose_chunks;
    bool profile_latency;
    char output_dir[PATH_MAX];
} GemmOptions;

#ifndef POLYBENCH_GEMM_DATASET_NAME
#if defined(MINI_DATASET)
#define POLYBENCH_GEMM_DATASET_NAME "MINI_DATASET"
#elif defined(SMALL_DATASET)
#define POLYBENCH_GEMM_DATASET_NAME "SMALL_DATASET"
#elif defined(MEDIUM_DATASET)
#define POLYBENCH_GEMM_DATASET_NAME "MEDIUM_DATASET"
#elif defined(LARGE_DATASET)
#define POLYBENCH_GEMM_DATASET_NAME "LARGE_DATASET"
#elif defined(EXTRALARGE_DATASET)
#define POLYBENCH_GEMM_DATASET_NAME "EXTRALARGE_DATASET"
#else
#define POLYBENCH_GEMM_DATASET_NAME "CUSTOM"
#endif
#endif

static void fatalf(const char *fmt, ...)
{
    va_list ap;

    fprintf(stderr, "[error] ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static long parse_positive_long(const char *arg_name, const char *value)
{
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 1) {
        fatalf("%s must be a positive integer, got '%s'", arg_name, value);
    }
    return parsed;
}

static void gemm_options_init(GemmOptions *opts)
{
    memset(opts, 0, sizeof(*opts));
#if defined(NI) && defined(NJ) && defined(NK)
    opts->m = NI;
    opts->n = NJ;
    opts->k = NK;
#else
    opts->m = 256;
    opts->n = 256;
    opts->k = 256;
#endif
    opts->reps = 1;
    opts->threads = 1;
    opts->vcpu_offset = 0;
    opts->gemmini_vcpu = -1;
    opts->housekeeping_vcpu = -1;
    opts->timeline = true;
    opts->serialize_gemmini = false;
    opts->verbose_chunks = false;
    opts->profile_latency = false;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --reps R --output-dir PATH "
            "[--timeline] [--threads T] [--vcpu-offset N] [--gemmini-vcpu N] "
            "[--housekeeping-vcpu N] "
            "[--serialize-gemmini] [--verbose-chunks] [--profile-latency]\n",
            prog);
}

static void parse_gemm_options(int argc, char **argv, GemmOptions *opts)
{
    bool have_output_dir = false;

    gemm_options_init(opts);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0) {
            if (++i >= argc) fatalf("missing value for --reps");
            opts->reps = (int)parse_positive_long("--reps", argv[i]);
        } else if (strcmp(argv[i], "--output-dir") == 0) {
            if (++i >= argc) fatalf("missing value for --output-dir");
            if (snprintf(opts->output_dir, sizeof(opts->output_dir), "%s", argv[i]) >=
                (int)sizeof(opts->output_dir)) {
                fatalf("output directory path is too long");
            }
            have_output_dir = true;
        } else if (strcmp(argv[i], "--timeline") == 0) {
            opts->timeline = true;
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) fatalf("missing value for --threads");
            opts->threads = (int)parse_positive_long("--threads", argv[i]);
        } else if (strcmp(argv[i], "--gemmini-vcpu") == 0) {
            if (++i >= argc) fatalf("missing value for --gemmini-vcpu");
            opts->gemmini_vcpu = atoi(argv[i]);
        } else if (strcmp(argv[i], "--vcpu-offset") == 0) {
            if (++i >= argc) fatalf("missing value for --vcpu-offset");
            opts->vcpu_offset = atoi(argv[i]);
        } else if (strcmp(argv[i], "--housekeeping-vcpu") == 0) {
            if (++i >= argc) fatalf("missing value for --housekeeping-vcpu");
            opts->housekeeping_vcpu = atoi(argv[i]);
        } else if (strcmp(argv[i], "--serialize-gemmini") == 0) {
            opts->serialize_gemmini = true;
        } else if (strcmp(argv[i], "--verbose-chunks") == 0) {
            opts->verbose_chunks = true;
        } else if (strcmp(argv[i], "--profile-latency") == 0) {
            opts->profile_latency = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            usage(argv[0]);
            fatalf("unknown argument: %s", argv[i]);
        }
    }

    if (!have_output_dir) {
        usage(argv[0]);
        fatalf("--output-dir is required");
    }
    if (opts->threads != 1) {
        fatalf("polybench_gemm currently supports --threads 1 only");
    }
    if (opts->vcpu_offset < 0) {
        fatalf("vcpu_offset must be non-negative");
    }
    if (opts->gemmini_vcpu < -1) {
        fatalf("gemmini_vcpu must be -1 or non-negative");
    }
    if (opts->housekeeping_vcpu < -1) {
        fatalf("housekeeping_vcpu must be -1 or non-negative");
    }
}

static void mkdir_one(const char *path)
{
    if (mkdir(path, 0775) == 0) {
        return;
    }
    if (errno == EEXIST) {
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            return;
        }
    }
    fatalf("failed to create directory '%s': %s", path, strerror(errno));
}

static void mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    size_t len;

    if (path == NULL || path[0] == '\0') {
        fatalf("empty directory path");
    }
    if (snprintf(tmp, sizeof(tmp), "%s", path) >= (int)sizeof(tmp)) {
        fatalf("directory path is too long");
    }

    len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/') {
        tmp[--len] = '\0';
    }

    for (char *p = tmp + 1; *p != '\0'; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir_one(tmp);
            *p = '/';
        }
    }
    mkdir_one(tmp);
}

static double now_ms(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fatalf("clock_gettime failed: %s", strerror(errno));
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void join_output_path(
    const GemmOptions *opts,
    const char *name,
    char *out,
    size_t out_size)
{
    if (snprintf(out, out_size, "%s/%s", opts->output_dir, name) >= (int)out_size) {
        fatalf("output path is too long for %s", name);
    }
}

static FILE *open_output_file(const GemmOptions *opts, const char *name)
{
    char path[PATH_MAX];
    FILE *fp;

    join_output_path(opts, name, path, sizeof(path));
    fp = fopen(path, "w");
    if (fp == NULL) {
        fatalf("failed to open '%s': %s", path, strerror(errno));
    }
    return fp;
}

static void emit_timeline(FILE *timeline, const char *event, int rep, double t0_ms)
{
    const double rel_ms = now_ms() - t0_ms;

    if (rep >= 0) {
        printf("[timeline] t_ms=%.3f event=%s rep=%d\n", rel_ms, event, rep);
        if (timeline != NULL) {
            fprintf(timeline, "[timeline] t_ms=%.3f event=%s rep=%d\n", rel_ms, event, rep);
        }
    } else {
        printf("[timeline] t_ms=%.3f event=%s\n", rel_ms, event);
        if (timeline != NULL) {
            fprintf(timeline, "[timeline] t_ms=%.3f event=%s\n", rel_ms, event);
        }
    }
    fflush(stdout);
    if (timeline != NULL) {
        fflush(timeline);
    }
}

static double *alloc_matrix(size_t rows, size_t cols, const char *name)
{
    size_t count;
    double *data;

    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols) {
        fatalf("invalid matrix size for %s", name);
    }
    count = rows * cols;
    if (count > SIZE_MAX / sizeof(double)) {
        fatalf("matrix allocation is too large for %s", name);
    }

    data = (double *)calloc(count, sizeof(double));
    if (data == NULL) {
        fatalf("failed to allocate %s matrix", name);
    }
    return data;
}

static double checksum_matrix(const double *data, size_t rows, size_t cols)
{
    const size_t count = rows * cols;
    double sum = 0.0;

    for (size_t i = 0; i < count; i++) {
        sum += data[i];
    }
    return sum;
}

static void write_config(const GemmOptions *opts, const char *mode)
{
    FILE *fp = open_output_file(opts, "config.txt");

    fprintf(fp, "workload=polybench_gemm\n");
    fprintf(fp, "mode=%s\n", mode);
    fprintf(fp, "dataset=%s\n", POLYBENCH_GEMM_DATASET_NAME);
    fprintf(fp, "m=%d\n", opts->m);
    fprintf(fp, "n=%d\n", opts->n);
    fprintf(fp, "k=%d\n", opts->k);
    fprintf(fp, "reps=%d\n", opts->reps);
    fprintf(fp, "threads=%d\n", opts->threads);
    fprintf(fp, "vcpu_offset=%d\n", opts->vcpu_offset);
    fprintf(fp, "gemmini_vcpu=%d\n", opts->gemmini_vcpu);
    fprintf(fp, "housekeeping_vcpu=%d\n", opts->housekeeping_vcpu);
    fprintf(fp, "serialize_gemmini=%d\n", opts->serialize_gemmini ? 1 : 0);
    fprintf(fp, "verbose_chunks=%d\n", opts->verbose_chunks ? 1 : 0);
    fprintf(fp, "profile_latency=%d\n", opts->profile_latency ? 1 : 0);
    fclose(fp);
}

static void write_checksum(const GemmOptions *opts, double checksum)
{
    FILE *fp = open_output_file(opts, "checksum.txt");

    fprintf(fp, "%.17g\n", checksum);
    fclose(fp);
}

static void write_metrics(
    const GemmOptions *opts,
    const char *mode,
    double elapsed_ms,
    double checksum)
{
    FILE *fp = open_output_file(opts, "metrics.json");

    fprintf(fp, "{\n");
    fprintf(fp, "  \"workload\": \"polybench_gemm\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", mode);
    fprintf(fp, "  \"dataset\": \"%s\",\n", POLYBENCH_GEMM_DATASET_NAME);
    fprintf(fp, "  \"m\": %d,\n", opts->m);
    fprintf(fp, "  \"n\": %d,\n", opts->n);
    fprintf(fp, "  \"k\": %d,\n", opts->k);
    fprintf(fp, "  \"reps\": %d,\n", opts->reps);
    fprintf(fp, "  \"elapsed_ms\": %.6f,\n", elapsed_ms);
    fprintf(fp, "  \"avg_ms_per_rep\": %.6f,\n", elapsed_ms / (double)opts->reps);
    fprintf(fp, "  \"checksum\": %.17g,\n", checksum);
    fprintf(fp, "  \"threads\": %d,\n", opts->threads);
    fprintf(fp, "  \"vcpu_offset\": %d,\n", opts->vcpu_offset);
    fprintf(fp, "  \"gemmini_vcpu\": %d,\n", opts->gemmini_vcpu);
    fprintf(fp, "  \"housekeeping_vcpu\": %d\n", opts->housekeeping_vcpu);
    fprintf(fp, "}\n");
    fclose(fp);
}

#endif
