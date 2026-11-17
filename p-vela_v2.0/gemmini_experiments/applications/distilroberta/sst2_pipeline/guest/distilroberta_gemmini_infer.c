#define _GNU_SOURCE

/*
 * distilroberta_gemmini_infer.c
 *
 * Role in the pipeline:
 * - Simplified Gemmini-backed guest implementation for the int8 DistilRoBERTa path.
 * - Reuses the validated vCPU baseline for all non-linear / control-flow ops.
 * - Replaces only the int8 linear/GEMM path with one direct tiled_matmul_auto() call.
 * - Compares per-sample outputs against the stored vCPU baseline run.
 *
 * What this program does:
 * - Loads model weights/scales and evaluation samples from generated/{model_bin,eval_bin}.
 * - Loads the latest vCPU baseline predictions/logits from outputs/c_vcpu/run_<timestamp>/predictions.jsonl.
 * - Runs guest inference with a minimal Gemmini linear path first, then scales up.
 * - Compares selected intermediate tensors against exported debug references.
 * - Writes metrics, predictions, and debug summaries for Gemmini-vs-vCPU comparison.
 */

#include <dirent.h>
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "gemmini.h"

#ifndef DISTILROBERTA_POWER_ONLY
#define DISTILROBERTA_POWER_ONLY 0
#endif

#ifndef DISTILROBERTA_USE_GEMMINI_BACKEND
#define DISTILROBERTA_USE_GEMMINI_BACKEND 1
#endif

#if !DISTILROBERTA_USE_GEMMINI_BACKEND && !DISTILROBERTA_POWER_ONLY
#error "The CPU backend shares this runner for power-only builds."
#endif

#if DISTILROBERTA_USE_GEMMINI_BACKEND
#define DISTILROBERTA_BACKEND_NAME "gemmini"
#define DISTILROBERTA_BACKEND_ACTOR "gemmini"
#define DISTILROBERTA_BACKEND_TMP_ROOT "/tmp/distilroberta_gemmini"
#define DISTILROBERTA_DEFAULT_OUTPUT_DIR "../outputs/gemmini"
#else
#define DISTILROBERTA_BACKEND_NAME "cpu"
#define DISTILROBERTA_BACKEND_ACTOR "vcpu"
#define DISTILROBERTA_BACKEND_TMP_ROOT "/tmp/distilroberta_cpu_power"
#define DISTILROBERTA_DEFAULT_OUTPUT_DIR "../outputs/cpu_power"
#endif

#define MAX_LAYERS 12
#define DEFAULT_THREADS 1
#define LAYERNORM_EPSILON 1.0e-5f
#define LAYERNORM_REDUCE_CHUNK 4
#define LAYERNORM_MAX_PARTS (((768) + LAYERNORM_REDUCE_CHUNK - 1) / LAYERNORM_REDUCE_CHUNK)
#ifndef GEMMINI_SAFE_MAX_I
#define GEMMINI_SAFE_MAX_I 16
#endif
#ifndef GEMMINI_SAFE_MAX_J
#define GEMMINI_SAFE_MAX_J 144
#endif
#if GEMMINI_SAFE_MAX_I <= 0 || GEMMINI_SAFE_MAX_J <= 0
#error "GEMMINI_SAFE_MAX_I and GEMMINI_SAFE_MAX_J must be positive"
#endif

static bool g_enable_timeline = false;
static bool g_log_backend_calls = false;

static const char *backend_name(void) {
    return DISTILROBERTA_BACKEND_NAME;
}

static const char *linear_backend_actor(void) {
    return DISTILROBERTA_BACKEND_ACTOR;
}

static bool argv_has_timeline_flag(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--timeline") == 0) {
            return true;
        }
    }
    return false;
}

static void emit_early_timeline_event(const char *actor, const char *label) {
    if (!g_enable_timeline) {
        return;
    }
    printf(
        "[timeline] actor=%s label=%s sample=0 thread=0 cpu=%d\n",
        actor,
        label,
        sched_getcpu()
    );
    fflush(stdout);
}
/* ==================== Model/Data Structures ==================== */

typedef struct {
    int in_features;
    int in_features_padded;
    int out_features;
    int out_features_padded;
    float input_scale;
    float output_scale;
    float inv_output_scale;
    int8_t *weight_q;
    int8_t *weight_q_t;
    float *weight_scale;
    float *bias_fp32;
    float *acc_scale;
} LinearLayer;

typedef struct {
    float *weight;
    float *bias;
    int dim;
} LayerNorm;

typedef struct {
    LinearLayer q_lin;
    LinearLayer k_lin;
    LinearLayer v_lin;
    LinearLayer out_lin;
    LinearLayer ffn_lin1;
    LinearLayer ffn_lin2;
    LayerNorm sa_layer_norm;
    LayerNorm output_layer_norm;
} TransformerLayer;

typedef struct {
    int n_layers;
    int dim;
    int hidden_dim;
    int n_heads;
    int max_position_embeddings;
    int type_vocab_size;
    int pad_token_id;
    int num_labels;
    int vocab_size;
    int seq_len;
    int eval_count;
    float *word_embeddings;
    float *position_embeddings;
    float *token_type_embeddings;
    LayerNorm embeddings_layer_norm;
    TransformerLayer layers[MAX_LAYERS];
    LinearLayer pre_classifier;
    LinearLayer classifier;
    int32_t *indices;
    int32_t *input_ids;
    int32_t *attention_mask;
    int32_t *labels;
    int32_t *host_preds;
    float *host_logits;
} DistilRobertaModel;

typedef struct {
    /* Scratch buffers reused across samples to avoid reallocating per layer. */
    float *hidden_a;
    float *hidden_b;
    float *q_buf;
    float *k_buf;
    float *v_buf;
    float *context_buf;
    float *ffn_buf;
    float *cls_buf;
    double *softmax_buf;
    elem_t *gemm_a;
    elem_t *gemm_b;
    acc_t *gemm_acc;
} Workspace;

typedef struct {
    int count;
    int *indices;
    int *preds;
    float *logits;
} BaselineReference;

typedef struct {
    int sample_idx;
    char point[32];
    int rows;
    int cols;
    float max_abs;
    float mean_abs;
    float cosine;
} DebugRecord;

typedef struct {
    char generated_dir[PATH_MAX];
    char output_dir[PATH_MAX];
    bool dump_debug_bin;
    DebugRecord *records;
    size_t count;
    size_t capacity;
    pthread_mutex_t mutex;
} DebugStore;

typedef struct {
    int idx;
    int label;
    int baseline_pred;
    int pred;
    float logits[2];
    float baseline_logits[2];
    float logit_max_abs_diff;
    double elapsed_ms;
} SampleResult;

typedef struct {
    int next_order;
    int total;
    pthread_mutex_t mutex;
} WorkQueue;

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    const LinearLayer *layer;
    int rows;
    int out_base;
    int chunk_cols;
    Workspace *ws;
    bool serialize_gemmini;
    bool has_job;
    bool job_done;
    bool stop;
    int target_vcpu;
    int pinned_cpu;
} GemminiExecutor;

typedef struct {
    const DistilRobertaModel *model;
    const BaselineReference *baseline;
    int start_order;
    WorkQueue *queue;
    SampleResult *results;
    DebugStore *debug_store;
    pthread_mutex_t *log_mutex;
    int thread_id;
    int vcpu_offset;
    int pinned_cpu;
    bool use_baseline;
    bool serialize_gemmini;
    GemminiExecutor *gemmini_executor;
} WorkerArgs;

typedef struct {
    char generated_dir[PATH_MAX];
    char output_dir[PATH_MAX];
    char baseline_predictions[PATH_MAX];
    int start_order;
    int limit;
    int num_threads;
    int vcpu_offset;
    int gemmini_vcpu;
    bool use_baseline;
    bool dump_debug_bin;
    bool serialize_gemmini;
} ProgramOptions;

typedef struct {
    char run_name[128];
    char temp_dir[PATH_MAX];
    char host_dir[PATH_MAX];
    char temp_metrics[PATH_MAX];
    char temp_predictions[PATH_MAX];
    char temp_debug_summary[PATH_MAX];
    char host_metrics[PATH_MAX];
    char host_predictions[PATH_MAX];
    char host_debug_summary[PATH_MAX];
} ResultPaths;

typedef struct {
    int correct;
    int baseline_correct;
    int pred_mismatches_vs_baseline;
    float max_logit_abs;
    double mean_logit_abs;
    int *mismatch_sample_indices;
    int mismatch_count;
} ExperimentSummary;

typedef struct {
    uint64_t linear_calls;
    uint64_t gemmini_chunks;
    uint64_t rows_total;
    uint64_t output_elements;
    uint64_t macs_total;
    uint64_t linear_total_ns;
    uint64_t quantize_ns;
    uint64_t submit_wait_ns;
    uint64_t executor_active_ns;
    uint64_t acc_memset_ns;
    uint64_t fence_flush_ns;
    uint64_t tiled_matmul_ns;
    uint64_t postprocess_ns;
} GemminiLinearProfile;

static pthread_mutex_t g_gemmini_linear_profile_mutex = PTHREAD_MUTEX_INITIALIZER;
static GemminiLinearProfile g_gemmini_linear_profile;

static void fatalf(const char *fmt, ...) {
    va_list args;
    fprintf(stderr, "[fatal] ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}

/* ==================== Basic Utilities ==================== */

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fatalf("clock_gettime failed: %s", strerror(errno));
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fatalf("clock_gettime failed: %s", strerror(errno));
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double ns_to_ms(uint64_t ns) {
    return (double)ns / 1.0e6;
}

static uint64_t ns_delta_or_zero(uint64_t lhs, uint64_t rhs) {
    return lhs > rhs ? lhs - rhs : 0;
}

static void add_gemmini_linear_profile(const GemminiLinearProfile *local) {
    pthread_mutex_lock(&g_gemmini_linear_profile_mutex);
    g_gemmini_linear_profile.linear_calls += local->linear_calls;
    g_gemmini_linear_profile.gemmini_chunks += local->gemmini_chunks;
    g_gemmini_linear_profile.rows_total += local->rows_total;
    g_gemmini_linear_profile.output_elements += local->output_elements;
    g_gemmini_linear_profile.macs_total += local->macs_total;
    g_gemmini_linear_profile.linear_total_ns += local->linear_total_ns;
    g_gemmini_linear_profile.quantize_ns += local->quantize_ns;
    g_gemmini_linear_profile.submit_wait_ns += local->submit_wait_ns;
    g_gemmini_linear_profile.executor_active_ns += local->executor_active_ns;
    g_gemmini_linear_profile.acc_memset_ns += local->acc_memset_ns;
    g_gemmini_linear_profile.fence_flush_ns += local->fence_flush_ns;
    g_gemmini_linear_profile.tiled_matmul_ns += local->tiled_matmul_ns;
    g_gemmini_linear_profile.postprocess_ns += local->postprocess_ns;
    pthread_mutex_unlock(&g_gemmini_linear_profile_mutex);
}

static void add_gemmini_executor_active_ns(uint64_t executor_active_ns) {
    pthread_mutex_lock(&g_gemmini_linear_profile_mutex);
    g_gemmini_linear_profile.executor_active_ns += executor_active_ns;
    pthread_mutex_unlock(&g_gemmini_linear_profile_mutex);
}

static void add_gemmini_chunk_detail_profile(
    uint64_t acc_memset_ns,
    uint64_t fence_flush_ns,
    uint64_t tiled_matmul_ns
) {
    pthread_mutex_lock(&g_gemmini_linear_profile_mutex);
    g_gemmini_linear_profile.acc_memset_ns += acc_memset_ns;
    g_gemmini_linear_profile.fence_flush_ns += fence_flush_ns;
    g_gemmini_linear_profile.tiled_matmul_ns += tiled_matmul_ns;
    pthread_mutex_unlock(&g_gemmini_linear_profile_mutex);
}

static GemminiLinearProfile snapshot_gemmini_linear_profile(void) {
    GemminiLinearProfile snapshot;
    pthread_mutex_lock(&g_gemmini_linear_profile_mutex);
    snapshot = g_gemmini_linear_profile;
    pthread_mutex_unlock(&g_gemmini_linear_profile_mutex);
    return snapshot;
}

static void print_gemmini_linear_profile(double total_elapsed_ms) {
    GemminiLinearProfile profile = snapshot_gemmini_linear_profile();
    uint64_t measured_ns =
        profile.quantize_ns + profile.submit_wait_ns + profile.postprocess_ns;
    uint64_t other_ns = ns_delta_or_zero(profile.linear_total_ns, measured_ns);
    uint64_t submit_overhead_ns =
        ns_delta_or_zero(profile.submit_wait_ns, profile.executor_active_ns);

    printf(
        "[backend-profile] backend=%s linear_calls=%llu chunks=%llu rows=%llu output_elements=%llu macs=%llu linear_total_ms=%.3f quantize_ms=%.3f backend_wall_ms=%.3f backend_active_ms=%.3f backend_overhead_ms=%.3f tiled_matmul_ms=%.3f fence_flush_ms=%.3f acc_memset_ms=%.3f postprocess_ms=%.3f other_linear_ms=%.3f elapsed_ms=%.3f linear_pct=%.3f backend_active_pct=%.3f tiled_matmul_pct=%.3f backend_overhead_pct=%.3f\n",
        backend_name(),
        (unsigned long long)profile.linear_calls,
        (unsigned long long)profile.gemmini_chunks,
        (unsigned long long)profile.rows_total,
        (unsigned long long)profile.output_elements,
        (unsigned long long)profile.macs_total,
        ns_to_ms(profile.linear_total_ns),
        ns_to_ms(profile.quantize_ns),
        ns_to_ms(profile.submit_wait_ns),
        ns_to_ms(profile.executor_active_ns),
        ns_to_ms(submit_overhead_ns),
        ns_to_ms(profile.tiled_matmul_ns),
        ns_to_ms(profile.fence_flush_ns),
        ns_to_ms(profile.acc_memset_ns),
        ns_to_ms(profile.postprocess_ns),
        ns_to_ms(other_ns),
        total_elapsed_ms,
        total_elapsed_ms > 0.0 ? ns_to_ms(profile.linear_total_ns) / total_elapsed_ms * 100.0 : 0.0,
        total_elapsed_ms > 0.0 ? ns_to_ms(profile.executor_active_ns) / total_elapsed_ms * 100.0 : 0.0,
        total_elapsed_ms > 0.0 ? ns_to_ms(profile.tiled_matmul_ns) / total_elapsed_ms * 100.0 : 0.0,
        total_elapsed_ms > 0.0 ? ns_to_ms(submit_overhead_ns) / total_elapsed_ms * 100.0 : 0.0
    );
    fflush(stdout);
}

static void write_gemmini_linear_profile_json(FILE *fp, double total_elapsed_ms) {
    GemminiLinearProfile profile = snapshot_gemmini_linear_profile();
    uint64_t measured_ns =
        profile.quantize_ns + profile.submit_wait_ns + profile.postprocess_ns;
    uint64_t other_ns = ns_delta_or_zero(profile.linear_total_ns, measured_ns);
    uint64_t submit_overhead_ns =
        ns_delta_or_zero(profile.submit_wait_ns, profile.executor_active_ns);

    fprintf(fp, "  \"backend_profile\": {\n");
    fprintf(fp, "    \"schema\": \"gemmini_backend_profile_v1\",\n");
    fprintf(fp, "    \"backend\": \"%s\",\n", backend_name());
    fprintf(fp, "    \"linear_calls\": %llu,\n", (unsigned long long)profile.linear_calls);
    fprintf(fp, "    \"chunks\": %llu,\n", (unsigned long long)profile.gemmini_chunks);
    fprintf(fp, "    \"rows\": %llu,\n", (unsigned long long)profile.rows_total);
    fprintf(fp, "    \"output_elements\": %llu,\n", (unsigned long long)profile.output_elements);
    fprintf(fp, "    \"macs\": %llu,\n", (unsigned long long)profile.macs_total);
    fprintf(fp, "    \"linear_total_ms\": %.6f,\n", ns_to_ms(profile.linear_total_ns));
    fprintf(fp, "    \"quantize_ms\": %.6f,\n", ns_to_ms(profile.quantize_ns));
    fprintf(fp, "    \"backend_wall_ms\": %.6f,\n", ns_to_ms(profile.submit_wait_ns));
    fprintf(fp, "    \"backend_active_ms\": %.6f,\n", ns_to_ms(profile.executor_active_ns));
    fprintf(fp, "    \"backend_overhead_ms\": %.6f,\n", ns_to_ms(submit_overhead_ns));
    fprintf(fp, "    \"tiled_matmul_ms\": %.6f,\n", ns_to_ms(profile.tiled_matmul_ns));
    fprintf(fp, "    \"fence_flush_ms\": %.6f,\n", ns_to_ms(profile.fence_flush_ns));
    fprintf(fp, "    \"acc_memset_ms\": %.6f,\n", ns_to_ms(profile.acc_memset_ns));
    fprintf(fp, "    \"postprocess_ms\": %.6f,\n", ns_to_ms(profile.postprocess_ns));
    fprintf(fp, "    \"other_linear_ms\": %.6f,\n", ns_to_ms(other_ns));
    fprintf(fp, "    \"linear_pct_of_elapsed\": %.6f,\n", total_elapsed_ms > 0.0 ? ns_to_ms(profile.linear_total_ns) / total_elapsed_ms * 100.0 : 0.0);
    fprintf(fp, "    \"backend_wall_pct_of_elapsed\": %.6f,\n", total_elapsed_ms > 0.0 ? ns_to_ms(profile.submit_wait_ns) / total_elapsed_ms * 100.0 : 0.0);
    fprintf(fp, "    \"backend_active_pct_of_elapsed\": %.6f,\n", total_elapsed_ms > 0.0 ? ns_to_ms(profile.executor_active_ns) / total_elapsed_ms * 100.0 : 0.0);
    fprintf(fp, "    \"tiled_matmul_pct_of_elapsed\": %.6f,\n", total_elapsed_ms > 0.0 ? ns_to_ms(profile.tiled_matmul_ns) / total_elapsed_ms * 100.0 : 0.0);
    fprintf(fp, "    \"fence_flush_pct_of_elapsed\": %.6f,\n", total_elapsed_ms > 0.0 ? ns_to_ms(profile.fence_flush_ns) / total_elapsed_ms * 100.0 : 0.0);
    fprintf(fp, "    \"backend_overhead_pct_of_elapsed\": %.6f\n", total_elapsed_ms > 0.0 ? ns_to_ms(submit_overhead_ns) / total_elapsed_ms * 100.0 : 0.0);
    fprintf(fp, "  },\n");
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fatalf("malloc failed for %zu bytes", size);
    }
    return ptr;
}

static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        fatalf("calloc failed for %zu x %zu bytes", count, size);
    }
    return ptr;
}

static void *xaligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0 || ptr == NULL) {
        fatalf("posix_memalign failed for %zu-byte aligned %zu-byte allocation", alignment, size);
    }
    memset(ptr, 0, size);
    return ptr;
}

static bool file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static int pin_thread_to_vcpu(int guest_cpu) {
    cpu_set_t cpuset;
    int rc = 0;
    CPU_ZERO(&cpuset);
    CPU_SET(guest_cpu, &cpuset);
    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fatalf("pthread_setaffinity_np failed for guest_cpu=%d: %s", guest_cpu, strerror(rc));
    }
    return sched_getcpu();
}

static void prepare_gemmini_state(void) {
#if DISTILROBERTA_USE_GEMMINI_BACKEND
    /*
     * Reset Gemmini state once before worker launch. Doing this per worker can
     * interfere with other active threads because gemmini_flush() affects the
     * device globally, not just the calling thread.
     */
    gemmini_flush(0);
    gemmini_fence();
#endif
}

static void join_path(char *dst, size_t dst_size, const char *lhs, const char *rhs) {
    int written = snprintf(dst, dst_size, "%s/%s", lhs, rhs);
    if (written < 0 || (size_t)written >= dst_size) {
        fatalf("path too long: %s/%s", lhs, rhs);
    }
}

static void snprintf_or_die(char *dst, size_t dst_size, const char *fmt, ...) {
    va_list args;
    int written;
    va_start(args, fmt);
    written = vsnprintf(dst, dst_size, fmt, args);
    va_end(args);
    if (written < 0 || (size_t)written >= dst_size) {
        fatalf("formatted string too long");
    }
}

static bool starts_with(const char *text, const char *prefix) {
    return strncmp(text, prefix, strlen(prefix)) == 0;
}

static int pad_to_dim(int value) {
    return ((value + DIM - 1) / DIM) * DIM;
}

static inline void cpu_memory_fence(void) {
    asm volatile("fence rw, rw" : : : "memory");
}

#if DISTILROBERTA_USE_GEMMINI_BACKEND
static pthread_mutex_t g_gemmini_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static void ensure_dir_recursive(const char *path) {
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) {
        fatalf("invalid directory path: %s", path);
    }
    memcpy(tmp, path, len + 1);
    for (size_t i = 1; i < len; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                fatalf("mkdir failed for %s: %s", tmp, strerror(errno));
            }
            tmp[i] = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        fatalf("mkdir failed for %s: %s", tmp, strerror(errno));
    }
}

static bool try_ensure_dir_recursive(const char *path) {
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) {
        return false;
    }
    memcpy(tmp, path, len + 1);
    for (size_t i = 1; i < len; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return false;
            }
            tmp[i] = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

static void *load_binary_file(const char *root_dir, const char *file_name, size_t elem_size, size_t *count_out) {
    char path[PATH_MAX];
    struct stat st;
    void *buffer;
    FILE *fp;

    join_path(path, sizeof(path), root_dir, file_name);
    if (stat(path, &st) != 0) {
        fatalf("stat failed for %s: %s", path, strerror(errno));
    }
    if ((size_t)st.st_size % elem_size != 0) {
        fatalf("file size mismatch for %s", path);
    }
    *count_out = (size_t)st.st_size / elem_size;
    buffer = xmalloc((size_t)st.st_size);
    fp = fopen(path, "rb");
    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }
    if (fread(buffer, elem_size, *count_out, fp) != *count_out) {
        fclose(fp);
        fatalf("fread failed for %s", path);
    }
    fclose(fp);
    return buffer;
}

static void *load_binary_file_from_path(const char *path, size_t elem_size, size_t *count_out) {
    struct stat st;
    FILE *fp;
    void *buffer;

    if (stat(path, &st) != 0) {
        fatalf("stat failed for %s: %s", path, strerror(errno));
    }
    if ((size_t)st.st_size % elem_size != 0) {
        fatalf("file size mismatch for %s", path);
    }
    *count_out = (size_t)st.st_size / elem_size;
    buffer = xmalloc((size_t)st.st_size);
    fp = fopen(path, "rb");
    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }
    if (fread(buffer, elem_size, *count_out, fp) != *count_out) {
        fclose(fp);
        fatalf("fread failed for %s", path);
    }
    fclose(fp);
    return buffer;
}

/* ==================== Model Loading ==================== */

static int32_t load_i32_scalar(const char *root_dir, const char *file_name) {
    size_t count = 0;
    int32_t *data = (int32_t *)load_binary_file(root_dir, file_name, sizeof(int32_t), &count);
    int32_t value;
    if (count != 1) {
        free(data);
        fatalf("expected scalar int32 in %s/%s", root_dir, file_name);
    }
    value = data[0];
    free(data);
    return value;
}

static float load_f32_scalar(const char *root_dir, const char *file_name) {
    size_t count = 0;
    float *data = (float *)load_binary_file(root_dir, file_name, sizeof(float), &count);
    float value;
    if (count != 1) {
        free(data);
        fatalf("expected scalar float32 in %s/%s", root_dir, file_name);
    }
    value = data[0];
    free(data);
    return value;
}

static float *load_f32_array(const char *root_dir, const char *file_name, size_t expected_count) {
    size_t count = 0;
    float *data = (float *)load_binary_file(root_dir, file_name, sizeof(float), &count);
    if (count != expected_count) {
        free(data);
        fatalf("unexpected float32 count for %s/%s: got %zu expected %zu",
               root_dir, file_name, count, expected_count);
    }
    return data;
}

static int32_t *load_i32_array(const char *root_dir, const char *file_name, size_t expected_count) {
    size_t count = 0;
    int32_t *data = (int32_t *)load_binary_file(root_dir, file_name, sizeof(int32_t), &count);
    if (count != expected_count) {
        free(data);
        fatalf("unexpected int32 count for %s/%s: got %zu expected %zu",
               root_dir, file_name, count, expected_count);
    }
    return data;
}

static int8_t *load_i8_array(const char *root_dir, const char *file_name, size_t expected_count) {
    size_t count = 0;
    int8_t *data = (int8_t *)load_binary_file(root_dir, file_name, sizeof(int8_t), &count);
    if (count != expected_count) {
        free(data);
        fatalf("unexpected int8 count for %s/%s: got %zu expected %zu",
               root_dir, file_name, count, expected_count);
    }
    return data;
}

static void load_layer_norm(const char *model_dir, const char *base_name, int dim, LayerNorm *layer_norm) {
    char weight_file[128];
    char bias_file[128];
    snprintf(weight_file, sizeof(weight_file), "%s__weight.bin", base_name);
    snprintf(bias_file, sizeof(bias_file), "%s__bias.bin", base_name);
    layer_norm->dim = dim;
    layer_norm->weight = load_f32_array(model_dir, weight_file, (size_t)dim);
    layer_norm->bias = load_f32_array(model_dir, bias_file, (size_t)dim);
}

static void load_linear_layer(const char *model_dir, const char *base_name, LinearLayer *layer) {
    char file_name[160];
    size_t out_features = 0;
    size_t in_features = 0;

    snprintf(file_name, sizeof(file_name), "%s__in_features.bin", base_name);
    layer->in_features = load_i32_scalar(model_dir, file_name);
    in_features = (size_t)layer->in_features;
    layer->in_features_padded = pad_to_dim(layer->in_features);

    snprintf(file_name, sizeof(file_name), "%s__out_features.bin", base_name);
    layer->out_features = load_i32_scalar(model_dir, file_name);
    out_features = (size_t)layer->out_features;
    layer->out_features_padded = pad_to_dim(layer->out_features);

    snprintf(file_name, sizeof(file_name), "%s__input_scale.bin", base_name);
    layer->input_scale = load_f32_scalar(model_dir, file_name);

    snprintf(file_name, sizeof(file_name), "%s__output_scale.bin", base_name);
    layer->output_scale = load_f32_scalar(model_dir, file_name);
    layer->inv_output_scale = 1.0f / layer->output_scale;

    snprintf(file_name, sizeof(file_name), "%s__weight_q.bin", base_name);
    layer->weight_q = load_i8_array(
        model_dir,
        file_name,
        in_features * out_features
    );

    /*
     * Gemmini matmul prefers B laid out as (K x J), while the export stores
     * weights as (J x K). Cache the transposed view once during model load so
     * each inference call can issue a single Gemmini matmul without per-call
     * weight transposition overhead.
     */
    layer->weight_q_t = (int8_t *)xcalloc((size_t)layer->in_features_padded * (size_t)layer->out_features_padded, sizeof(int8_t));
    for (size_t out_idx = 0; out_idx < out_features; ++out_idx) {
        for (size_t in_idx = 0; in_idx < in_features; ++in_idx) {
            layer->weight_q_t[in_idx * (size_t)layer->out_features_padded + out_idx] =
                layer->weight_q[out_idx * in_features + in_idx];
        }
    }

    snprintf(file_name, sizeof(file_name), "%s__weight_scale.bin", base_name);
    layer->weight_scale = load_f32_array(model_dir, file_name, out_features);

    snprintf(file_name, sizeof(file_name), "%s__bias_fp32.bin", base_name);
    layer->bias_fp32 = load_f32_array(model_dir, file_name, out_features);

    layer->acc_scale = (float *)xmalloc(out_features * sizeof(float));
    /* int32 accumulator -> fp32 rescale factor for each output channel. */
    for (size_t i = 0; i < out_features; ++i) {
        layer->acc_scale[i] = layer->input_scale * layer->weight_scale[i];
    }
}

static void load_model(DistilRobertaModel *model, const char *generated_dir) {
    char model_dir[PATH_MAX];
    char eval_dir[PATH_MAX];
    size_t count = 0;

    memset(model, 0, sizeof(*model));

    join_path(model_dir, sizeof(model_dir), generated_dir, "model_bin");
    join_path(eval_dir, sizeof(eval_dir), generated_dir, "eval_bin");

    emit_early_timeline_event("vcpu", "model_cfg");
    model->n_layers = load_i32_scalar(model_dir, "config__n_layers.bin");
    model->dim = load_i32_scalar(model_dir, "config__dim.bin");
    model->hidden_dim = load_i32_scalar(model_dir, "config__hidden_dim.bin");
    model->n_heads = load_i32_scalar(model_dir, "config__n_heads.bin");
    model->max_position_embeddings = load_i32_scalar(model_dir, "config__max_position_embeddings.bin");
    model->type_vocab_size = load_i32_scalar(model_dir, "config__type_vocab_size.bin");
    model->pad_token_id = load_i32_scalar(model_dir, "config__pad_token_id.bin");
    model->num_labels = load_i32_scalar(model_dir, "config__num_labels.bin");

    if (model->n_layers <= 0 || model->n_layers > MAX_LAYERS) {
        fatalf("unsupported n_layers=%d", model->n_layers);
    }
    if (model->dim <= 0 || model->hidden_dim <= 0 || model->n_heads <= 0) {
        fatalf("invalid model config");
    }
    if (model->dim % model->n_heads != 0) {
        fatalf("dim=%d not divisible by n_heads=%d", model->dim, model->n_heads);
    }

    emit_early_timeline_event("vcpu", "model_word_emb");
    model->word_embeddings = (float *)load_binary_file(
        model_dir,
        "roberta__embeddings__word_embeddings__weight.bin",
        sizeof(float),
        &count
    );
    if (count % (size_t)model->dim != 0) {
        fatalf("word embedding size mismatch");
    }
    model->vocab_size = (int)(count / (size_t)model->dim);

    emit_early_timeline_event("vcpu", "model_pos_emb");
    model->position_embeddings = (float *)load_binary_file(
        model_dir,
        "roberta__embeddings__position_embeddings__weight.bin",
        sizeof(float),
        &count
    );
    if (count != (size_t)model->max_position_embeddings * (size_t)model->dim) {
        fatalf("position embedding size mismatch");
    }

    emit_early_timeline_event("vcpu", "model_type_emb");
    model->token_type_embeddings = (float *)load_binary_file(
        model_dir,
        "roberta__embeddings__token_type_embeddings__weight.bin",
        sizeof(float),
        &count
    );
    if (count != (size_t)model->type_vocab_size * (size_t)model->dim) {
        fatalf("token type embedding size mismatch");
    }

    emit_early_timeline_event("vcpu", "model_emb_ln");
    load_layer_norm(
        model_dir,
        "roberta__embeddings__LayerNorm",
        model->dim,
        &model->embeddings_layer_norm
    );

    emit_early_timeline_event("vcpu", "model_encoder");
    for (int layer_idx = 0; layer_idx < model->n_layers; ++layer_idx) {
        TransformerLayer *layer = &model->layers[layer_idx];
        char base_name[128];

        snprintf(base_name, sizeof(base_name), "layer%d__q_lin", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->q_lin);
        snprintf(base_name, sizeof(base_name), "layer%d__k_lin", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->k_lin);
        snprintf(base_name, sizeof(base_name), "layer%d__v_lin", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->v_lin);
        snprintf(base_name, sizeof(base_name), "layer%d__out_lin", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->out_lin);
        snprintf(base_name, sizeof(base_name), "layer%d__ffn_lin1", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->ffn_lin1);
        snprintf(base_name, sizeof(base_name), "layer%d__ffn_lin2", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->ffn_lin2);

        snprintf(base_name, sizeof(base_name), "roberta__encoder__layer__%d__attention__output__LayerNorm", layer_idx);
        load_layer_norm(model_dir, base_name, model->dim, &layer->sa_layer_norm);
        snprintf(base_name, sizeof(base_name), "roberta__encoder__layer__%d__output__LayerNorm", layer_idx);
        load_layer_norm(model_dir, base_name, model->dim, &layer->output_layer_norm);
    }

    emit_early_timeline_event("vcpu", "model_head");
    load_linear_layer(model_dir, "pre_classifier", &model->pre_classifier);
    load_linear_layer(model_dir, "classifier", &model->classifier);

    emit_early_timeline_event("vcpu", "eval_meta");
    model->labels = (int32_t *)load_binary_file(eval_dir, "labels.bin", sizeof(int32_t), &count);
    model->eval_count = (int)count;

    model->indices = load_i32_array(eval_dir, "indices.bin", (size_t)model->eval_count);
    model->host_preds = load_i32_array(eval_dir, "host_int8_preds.bin", (size_t)model->eval_count);
    model->host_logits = load_f32_array(
        eval_dir,
        "host_int8_logits.bin",
        (size_t)model->eval_count * (size_t)model->num_labels
    );
}

static void load_eval_inputs(DistilRobertaModel *model, const char *generated_dir) {
    char eval_dir[PATH_MAX];
    size_t input_count = 0;
    size_t mask_count = 0;

    join_path(eval_dir, sizeof(eval_dir), generated_dir, "eval_bin");

    emit_early_timeline_event("vcpu", "eval_input_ids");
    model->input_ids = (int32_t *)load_binary_file(eval_dir, "input_ids.bin", sizeof(int32_t), &input_count);
    if (model->eval_count <= 0 || input_count % (size_t)model->eval_count != 0) {
        fatalf("input_ids count does not match eval_count");
    }
    /* Sequence length is inferred from exported evaluation tensors, not hardcoded. */
    model->seq_len = (int)(input_count / (size_t)model->eval_count);

    emit_early_timeline_event("vcpu", "eval_attention_mask");
    model->attention_mask = (int32_t *)load_binary_file(
        eval_dir,
        "attention_mask.bin",
        sizeof(int32_t),
        &mask_count
    );
    if (mask_count != input_count) {
        fatalf("attention_mask size does not match input_ids size");
    }
}

static void init_model(DistilRobertaModel *model, const char *generated_dir) {
    load_model(model, generated_dir);
    load_eval_inputs(model, generated_dir);
}

static void free_baseline_reference(BaselineReference *baseline) {
    free(baseline->indices);
    free(baseline->preds);
    free(baseline->logits);
    memset(baseline, 0, sizeof(*baseline));
}

static bool parse_predictions_line(
    const char *line,
    int *idx_out,
    int *label_out,
    int *pred_out,
    float *logit0_out,
    float *logit1_out
) {
    int matched = sscanf(
        line,
        "{\"idx\": %d, \"label\": %d, \"pred\": %d, %*[^[][%f, %f]",
        idx_out,
        label_out,
        pred_out,
        logit0_out,
        logit1_out
    );
    return matched == 5;
}

static void resolve_latest_baseline_predictions(char *dst, size_t dst_size, const char *output_dir) {
    DIR *dir = opendir(output_dir);
    struct dirent *entry = NULL;
    char best_name[NAME_MAX] = {0};
    bool found = false;

    if (dir == NULL) {
        fatalf("opendir failed for %s: %s", output_dir, strerror(errno));
    }

    emit_early_timeline_event("vcpu", "baseline_scan");
    while ((entry = readdir(dir)) != NULL) {
        if (starts_with(entry->d_name, "run_")) {
            if (!found || strcmp(entry->d_name, best_name) > 0) {
                size_t name_len = strlen(entry->d_name);
                if (name_len >= sizeof(best_name)) {
                    closedir(dir);
                    fatalf("baseline run directory name too long: %s", entry->d_name);
                }
                memcpy(best_name, entry->d_name, name_len + 1);
                found = true;
            }
        }
    }
    closedir(dir);

    if (!found) {
        fatalf("no run_* baseline directory found under %s", output_dir);
    }

    snprintf_or_die(dst, dst_size, "%s/%s/predictions.jsonl", output_dir, best_name);
}

static void load_baseline_reference(
    BaselineReference *baseline,
    const char *predictions_path,
    const DistilRobertaModel *model,
    int start_order,
    int total
) {
    FILE *fp = fopen(predictions_path, "r");
    char line[4096];
    int line_idx = 0;
    int global_order = 0;

    memset(baseline, 0, sizeof(*baseline));
    if (fp == NULL) {
        fatalf("failed to open baseline predictions %s: %s", predictions_path, strerror(errno));
    }

    emit_early_timeline_event("vcpu", "baseline_alloc");
    baseline->count = total;
    baseline->indices = (int *)xcalloc((size_t)total, sizeof(int));
    baseline->preds = (int *)xcalloc((size_t)total, sizeof(int));
    baseline->logits = (float *)xcalloc((size_t)total * (size_t)model->num_labels, sizeof(float));

    emit_early_timeline_event("vcpu", "baseline_parse");
    while (fgets(line, sizeof(line), fp) != NULL) {
        int idx = 0;
        int label = 0;
        int pred = 0;
        float logit0 = 0.0f;
        float logit1 = 0.0f;

        if (!parse_predictions_line(line, &idx, &label, &pred, &logit0, &logit1)) {
            fclose(fp);
            fatalf("failed to parse baseline predictions line %d in %s", global_order + 1, predictions_path);
        }

        if (global_order < start_order) {
            ++global_order;
            continue;
        }
        if (line_idx >= total) {
            break;
        }

        if (idx != model->indices[start_order + line_idx]) {
            fclose(fp);
            fatalf(
                "baseline idx mismatch at global_order %d: got %d expected %d",
                start_order + line_idx,
                idx,
                model->indices[start_order + line_idx]
            );
        }
        if (label != model->labels[start_order + line_idx]) {
            fclose(fp);
            fatalf(
                "baseline label mismatch at global_order %d: got %d expected %d",
                start_order + line_idx,
                label,
                model->labels[start_order + line_idx]
            );
        }

        baseline->indices[line_idx] = idx;
        baseline->preds[line_idx] = pred;
        baseline->logits[(size_t)line_idx * (size_t)model->num_labels + 0] = logit0;
        baseline->logits[(size_t)line_idx * (size_t)model->num_labels + 1] = logit1;
        ++line_idx;
        ++global_order;
    }

    fclose(fp);

    if (line_idx < total) {
        fatalf("baseline predictions shorter than requested sample count: got %d expected %d", line_idx, total);
    }
}

static void init_workspace(const DistilRobertaModel *model, Workspace *ws) {
    size_t token_count = (size_t)model->seq_len * (size_t)model->dim;
    size_t ffn_count = (size_t)model->seq_len * (size_t)model->hidden_dim;
    int max_in_features = model->hidden_dim > model->dim ? model->hidden_dim : model->dim;
    int max_out_features = model->hidden_dim > model->dim ? model->hidden_dim : model->dim;
    size_t max_gemm_a_count = (size_t)model->seq_len * (size_t)max_in_features;
    size_t max_gemm_b_count = (size_t)max_in_features * (size_t)max_out_features;
    size_t max_gemm_acc_count = (size_t)model->seq_len * (size_t)max_out_features;

    memset(ws, 0, sizeof(*ws));
    ws->hidden_a = (float *)xcalloc(token_count, sizeof(float));
    ws->hidden_b = (float *)xcalloc(token_count, sizeof(float));
    ws->q_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->k_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->v_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->context_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->ffn_buf = (float *)xcalloc(ffn_count, sizeof(float));
    ws->cls_buf = (float *)xcalloc((size_t)model->dim, sizeof(float));
    ws->softmax_buf = (double *)xcalloc((size_t)model->seq_len, sizeof(double));
    ws->gemm_a = (elem_t *)xaligned_alloc(64, max_gemm_a_count * sizeof(elem_t));
    ws->gemm_b = (elem_t *)xaligned_alloc(64, max_gemm_b_count * sizeof(elem_t));
    ws->gemm_acc = (acc_t *)xaligned_alloc(64, max_gemm_acc_count * sizeof(acc_t));
}

/* ==================== Numeric Kernels ==================== */

static inline int32_t quantize_ties_to_even(float value, float scale) {
    float q = nearbyintf(value / scale);
    if (q > 127.0f) {
        q = 127.0f;
    } else if (q < -128.0f) {
        q = -128.0f;
    }
    return (int32_t)q;
}

static void quantize_linear_input(
    const LinearLayer *layer,
    const float *input,
    int rows,
    Workspace *ws
) {
    const size_t a_stride = (size_t)layer->in_features;
    const size_t gemm_a_count = (size_t)rows * a_stride;

    memset(ws->gemm_a, 0, gemm_a_count * sizeof(elem_t));
    for (int row = 0; row < rows; ++row) {
        const float *row_input = input + (size_t)row * (size_t)layer->in_features;
        elem_t *row_a = ws->gemm_a + (size_t)row * a_stride;
        for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
            row_a[in_idx] = (elem_t)quantize_ties_to_even(row_input[in_idx], layer->input_scale);
        }
    }
}

#if !DISTILROBERTA_USE_GEMMINI_BACKEND
static void cpu_matmul_acc_chunk(
    const LinearLayer *layer,
    int rows,
    int out_base,
    int chunk_cols,
    Workspace *ws
) {
    const size_t a_stride = (size_t)layer->in_features;
    const size_t c_stride = (size_t)chunk_cols;
    const size_t gemm_acc_count = (size_t)rows * c_stride;

    memset(ws->gemm_acc, 0, gemm_acc_count * sizeof(acc_t));
    for (int row = 0; row < rows; ++row) {
        const elem_t *row_a = ws->gemm_a + (size_t)row * a_stride;
        acc_t *row_acc = ws->gemm_acc + (size_t)row * c_stride;
        for (int local_out = 0; local_out < chunk_cols; ++local_out) {
            const int out_idx = out_base + local_out;
            const int8_t *weight_row =
                layer->weight_q + (size_t)out_idx * (size_t)layer->in_features;
            acc_t acc = 0;
            for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
                acc += (acc_t)row_a[in_idx] * (acc_t)weight_row[in_idx];
            }
            row_acc[local_out] = acc;
        }
    }
}
#endif

#if DISTILROBERTA_USE_GEMMINI_BACKEND
static void gemmini_matmul_acc_chunk(
    const LinearLayer *layer,
    int rows,
    int out_base,
    int chunk_cols,
    Workspace *ws,
    bool serialize_gemmini
) {
    const size_t a_stride = (size_t)layer->in_features;
    const size_t b_stride = (size_t)layer->out_features_padded;
    const size_t c_stride = (size_t)chunk_cols;
    const size_t gemm_acc_count = (size_t)rows * c_stride;
    const elem_t *weight_chunk = (const elem_t *)layer->weight_q_t + (size_t)out_base;
    uint64_t phase_start_ns = now_ns();
    uint64_t acc_memset_ns = 0;
    uint64_t fence_flush_ns = 0;
    uint64_t tiled_matmul_ns = 0;

    memset(ws->gemm_acc, 0, gemm_acc_count * sizeof(acc_t));
    acc_memset_ns += now_ns() - phase_start_ns;

    phase_start_ns = now_ns();
    cpu_memory_fence();
    fence_flush_ns += now_ns() - phase_start_ns;

    if (serialize_gemmini) {
        pthread_mutex_lock(&g_gemmini_mutex);
    }

    phase_start_ns = now_ns();
    gemmini_flush(0);
    gemmini_fence();
    fence_flush_ns += now_ns() - phase_start_ns;

    phase_start_ns = now_ns();
    tiled_matmul_auto(
        (size_t)rows,
        (size_t)chunk_cols,
        (size_t)layer->in_features,
        ws->gemm_a,
        weight_chunk,
        NULL,
        ws->gemm_acc,
        a_stride,
        b_stride,
        c_stride,
        c_stride,
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
    tiled_matmul_ns += now_ns() - phase_start_ns;

    phase_start_ns = now_ns();
    gemmini_fence();
    cpu_memory_fence();
    fence_flush_ns += now_ns() - phase_start_ns;

    add_gemmini_chunk_detail_profile(acc_memset_ns, fence_flush_ns, tiled_matmul_ns);

    if (serialize_gemmini) {
        pthread_mutex_unlock(&g_gemmini_mutex);
    }
}

static void *gemmini_executor_main(void *arg) {
    GemminiExecutor *executor = (GemminiExecutor *)arg;

    executor->pinned_cpu = pin_thread_to_vcpu(executor->target_vcpu);
    if (g_log_backend_calls) {
        printf("[backend-executor-start] backend=%s target_vcpu=%d cpu=%d\n", backend_name(), executor->target_vcpu, executor->pinned_cpu);
        fflush(stdout);
    }
#if !DISTILROBERTA_POWER_ONLY
    prepare_gemmini_state();
#endif

    pthread_mutex_lock(&executor->mutex);
    while (!executor->stop) {
        while (!executor->has_job && !executor->stop) {
            pthread_cond_wait(&executor->cond, &executor->mutex);
        }
        if (executor->stop) {
            break;
        }

        const LinearLayer *layer = executor->layer;
        const int rows = executor->rows;
        const int out_base = executor->out_base;
        const int chunk_cols = executor->chunk_cols;
        Workspace *ws = executor->ws;
        const bool serialize_gemmini = executor->serialize_gemmini;

        executor->has_job = false;
        pthread_mutex_unlock(&executor->mutex);

        if (g_log_backend_calls) {
            printf(
                "[backend-call] backend=%s target_vcpu=%d cpu=%d rows=%d out_base=%d chunk_cols=%d\n",
                backend_name(),
                executor->target_vcpu,
                sched_getcpu(),
                rows,
                out_base,
                chunk_cols
            );
            fflush(stdout);
        }
        uint64_t executor_start_ns = now_ns();
        gemmini_matmul_acc_chunk(layer, rows, out_base, chunk_cols, ws, serialize_gemmini);
        add_gemmini_executor_active_ns(now_ns() - executor_start_ns);

        pthread_mutex_lock(&executor->mutex);
        executor->job_done = true;
        pthread_cond_broadcast(&executor->cond);
    }
    pthread_mutex_unlock(&executor->mutex);
    return NULL;
}

static void gemmini_executor_start(GemminiExecutor *executor, int target_vcpu) {
    memset(executor, 0, sizeof(*executor));
    executor->target_vcpu = target_vcpu;
    executor->pinned_cpu = -1;
    pthread_mutex_init(&executor->mutex, NULL);
    pthread_cond_init(&executor->cond, NULL);
    if (pthread_create(&executor->thread, NULL, gemmini_executor_main, executor) != 0) {
        fatalf("pthread_create failed for gemmini executor");
    }
}

static void gemmini_executor_stop(GemminiExecutor *executor) {
    pthread_mutex_lock(&executor->mutex);
    executor->stop = true;
    pthread_cond_broadcast(&executor->cond);
    pthread_mutex_unlock(&executor->mutex);
    if (pthread_join(executor->thread, NULL) != 0) {
        fatalf("pthread_join failed for gemmini executor");
    }
    pthread_cond_destroy(&executor->cond);
    pthread_mutex_destroy(&executor->mutex);
}

static void gemmini_executor_run_chunk(
    GemminiExecutor *executor,
    const LinearLayer *layer,
    int rows,
    int out_base,
    int chunk_cols,
    Workspace *ws,
    bool serialize_gemmini
) {
    pthread_mutex_lock(&executor->mutex);
    while (executor->has_job && !executor->stop) {
        pthread_cond_wait(&executor->cond, &executor->mutex);
    }
    if (executor->stop) {
        pthread_mutex_unlock(&executor->mutex);
        fatalf("gemmini executor stopped before job submit");
    }

    executor->layer = layer;
    executor->rows = rows;
    executor->out_base = out_base;
    executor->chunk_cols = chunk_cols;
    executor->ws = ws;
    executor->serialize_gemmini = serialize_gemmini;
    executor->job_done = false;
    executor->has_job = true;
    pthread_cond_broadcast(&executor->cond);

    while (!executor->job_done && !executor->stop) {
        pthread_cond_wait(&executor->cond, &executor->mutex);
    }
    if (executor->stop) {
        pthread_mutex_unlock(&executor->mutex);
        fatalf("gemmini executor stopped before job completion");
    }
    pthread_mutex_unlock(&executor->mutex);
}
#endif

static void linear_postprocess_chunk(
    const LinearLayer *layer,
    int rows,
    int out_base,
    int chunk_cols,
    float *output,
    Workspace *ws
) {
    const size_t c_stride = (size_t)chunk_cols;

    for (int row = 0; row < rows; ++row) {
        float *row_output = output + (size_t)row * (size_t)layer->out_features;
        const acc_t *row_acc = ws->gemm_acc + (size_t)row * c_stride;
        for (int local_out = 0; local_out < chunk_cols; ++local_out) {
            const int out_idx = out_base + local_out;
            volatile float scaled = (float)row_acc[local_out] * layer->acc_scale[out_idx];
            volatile float y = scaled + layer->bias_fp32[out_idx];
            int32_t y_q = quantize_ties_to_even(y, layer->output_scale);
            row_output[out_idx] = (float)y_q * layer->output_scale;
        }
    }
}

static void linear_forward(
    const LinearLayer *layer,
    const float *input,
    int rows,
    float *output,
    Workspace *ws,
    bool serialize_gemmini,
    GemminiExecutor *gemmini_executor
) {
    /*
     * Minimal Gemmini-backed counterpart of Python QuantizedLinear.forward().
     * Gemmini is responsible only for int8 x int8 -> int32 accumulation.
     * Per-output-channel rescale, fp32 bias, and requant stay on CPU so the
     * numeric contract stays identical to the validated vCPU baseline.
     */
    GemminiLinearProfile local_profile;
    uint64_t total_start_ns = now_ns();

    memset(&local_profile, 0, sizeof(local_profile));
    local_profile.linear_calls = 1;
    local_profile.rows_total = (uint64_t)rows;
    local_profile.output_elements = (uint64_t)rows * (uint64_t)layer->out_features;
    local_profile.macs_total =
        (uint64_t)rows * (uint64_t)layer->in_features * (uint64_t)layer->out_features;

    for (int row_base = 0; row_base < rows; row_base += GEMMINI_SAFE_MAX_I) {
        const int row_remaining = rows - row_base;
        const int chunk_rows = row_remaining < GEMMINI_SAFE_MAX_I ? row_remaining : GEMMINI_SAFE_MAX_I;
        const float *input_chunk = input + (size_t)row_base * (size_t)layer->in_features;
        float *output_chunk = output + (size_t)row_base * (size_t)layer->out_features;
        uint64_t phase_start_ns = now_ns();

        quantize_linear_input(layer, input_chunk, chunk_rows, ws);
        local_profile.quantize_ns += now_ns() - phase_start_ns;

        for (int out_base = 0; out_base < layer->out_features; out_base += GEMMINI_SAFE_MAX_J) {
            const int remaining = layer->out_features - out_base;
            const int chunk_cols = remaining < GEMMINI_SAFE_MAX_J ? remaining : GEMMINI_SAFE_MAX_J;

            local_profile.gemmini_chunks += 1;
            phase_start_ns = now_ns();
#if DISTILROBERTA_USE_GEMMINI_BACKEND
            if (gemmini_executor != NULL) {
                gemmini_executor_run_chunk(
                    gemmini_executor,
                    layer,
                    chunk_rows,
                    out_base,
                    chunk_cols,
                    ws,
                    serialize_gemmini
                );
                local_profile.submit_wait_ns += now_ns() - phase_start_ns;
            } else {
                gemmini_matmul_acc_chunk(layer, chunk_rows, out_base, chunk_cols, ws, serialize_gemmini);
                {
                    uint64_t backend_ns = now_ns() - phase_start_ns;
                    local_profile.submit_wait_ns += backend_ns;
                    local_profile.executor_active_ns += backend_ns;
                }
            }
#else
            (void)serialize_gemmini;
            (void)gemmini_executor;
            cpu_matmul_acc_chunk(layer, chunk_rows, out_base, chunk_cols, ws);
            {
                uint64_t backend_ns = now_ns() - phase_start_ns;
                local_profile.submit_wait_ns += backend_ns;
                local_profile.executor_active_ns += backend_ns;
            }
#endif
            phase_start_ns = now_ns();
            linear_postprocess_chunk(layer, chunk_rows, out_base, chunk_cols, output_chunk, ws);
            local_profile.postprocess_ns += now_ns() - phase_start_ns;
        }
    }

    local_profile.linear_total_ns = now_ns() - total_start_ns;
    add_gemmini_linear_profile(&local_profile);
}

static void add_inplace(float *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

static float pairwise_reduce_f32(float *values, int count) {
    while (count > 1) {
        int next_count = 0;
        int idx = 0;
        for (; idx + 1 < count; idx += 2) {
            values[next_count++] = values[idx] + values[idx + 1];
        }
        if (idx < count) {
            values[next_count++] = values[idx];
        }
        count = next_count;
    }
    return values[0];
}

static void layer_norm_inplace(float *tensor, int rows, const LayerNorm *layer_norm) {
    int dim = layer_norm->dim;
    float *row_copy = (float *)xmalloc((size_t)dim * sizeof(float));
    float *partials = (float *)xmalloc((size_t)LAYERNORM_MAX_PARTS * sizeof(float));
    for (int row = 0; row < rows; ++row) {
        float *row_ptr = tensor + (size_t)row * (size_t)dim;
        float mean = 0.0f;
        float var = 0.0f;
        float inv_std = 0.0f;
        float partial_sum = 0.0f;
        int part_count = 0;

        memcpy(row_copy, row_ptr, (size_t)dim * sizeof(float));

        for (int col = 0; col < dim; col += LAYERNORM_REDUCE_CHUNK) {
            int end = col + LAYERNORM_REDUCE_CHUNK;
            if (end > dim) {
                end = dim;
            }
            partial_sum = 0.0f;
            for (int inner = col; inner < end; ++inner) {
                partial_sum += row_copy[inner];
            }
            partials[part_count++] = partial_sum;
        }
        mean = pairwise_reduce_f32(partials, part_count);
        mean /= (float)dim;

        part_count = 0;
        for (int col = 0; col < dim; col += LAYERNORM_REDUCE_CHUNK) {
            int end = col + LAYERNORM_REDUCE_CHUNK;
            if (end > dim) {
                end = dim;
            }
            partial_sum = 0.0f;
            for (int inner = col; inner < end; ++inner) {
                float delta = row_copy[inner] - mean;
                partial_sum += delta * delta;
            }
            partials[part_count++] = partial_sum;
        }
        var = pairwise_reduce_f32(partials, part_count);
        var /= (float)dim;
        inv_std = 1.0f / sqrtf(var + LAYERNORM_EPSILON);
        for (int col = 0; col < dim; ++col) {
            float normalized = (row_copy[col] - mean) * inv_std;
            row_ptr[col] = normalized * layer_norm->weight[col] + layer_norm->bias[col];
        }
    }
    free(partials);
    free(row_copy);
}

static void gelu_inplace(float *values, size_t count) {
    const float inv_sqrt_2 = 0.7071067811865475f;
    for (size_t i = 0; i < count; ++i) {
        float x = values[i];
        values[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt_2));
    }
}

static void tanh_inplace(float *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        values[i] = tanhf(values[i]);
    }
}

static void trace_timeline_event(
    pthread_mutex_t *mutex,
    int thread_id,
    int sample_idx,
    const char *actor,
    const char *label
) {
    if (!g_enable_timeline) {
        return;
    }
    pthread_mutex_lock(mutex);
    printf(
        "[timeline] actor=%s label=%s sample=%d thread=%d cpu=%d\n",
        actor,
        label,
        sample_idx,
        thread_id,
        sched_getcpu()
    );
    fflush(stdout);
    pthread_mutex_unlock(mutex);
}

/* ==================== Embedding / Attention ==================== */

static void embeddings_forward(
    const DistilRobertaModel *model,
    const int32_t *input_ids,
    float *hidden
) {
    const float *token_type_src = model->token_type_embeddings;
    int32_t running_position = model->pad_token_id;

    /* RoBERTa uses padding-aware position ids and always adds token-type embeddings. */
    for (int pos = 0; pos < model->seq_len; ++pos) {
        int32_t token_id = input_ids[pos];
        float *dst = hidden + (size_t)pos * (size_t)model->dim;
        const float *word_src;
        const float *pos_src;
        int32_t position_id = model->pad_token_id;

        if (token_id < 0 || token_id >= model->vocab_size) {
            fatalf("token_id out of range: %d", token_id);
        }
        if (token_id != model->pad_token_id) {
            running_position += 1;
            position_id = running_position;
        }
        if (position_id < 0 || position_id >= model->max_position_embeddings) {
            fatalf("position_id out of range: %d", position_id);
        }
        word_src = model->word_embeddings + (size_t)token_id * (size_t)model->dim;
        pos_src = model->position_embeddings + (size_t)position_id * (size_t)model->dim;
        for (int dim_idx = 0; dim_idx < model->dim; ++dim_idx) {
            dst[dim_idx] = word_src[dim_idx] + pos_src[dim_idx] + token_type_src[dim_idx];
        }
    }
    layer_norm_inplace(hidden, model->seq_len, &model->embeddings_layer_norm);
}

static void attention_forward(
    const DistilRobertaModel *model,
    const float *q_buf,
    const float *k_buf,
    const float *v_buf,
    const int32_t *attention_mask,
    Workspace *ws
) {
    const int head_dim = model->dim / model->n_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float masked_bias = -FLT_MAX;
    double *context_acc = (double *)xcalloc((size_t)head_dim, sizeof(double));

    /* Context is rebuilt per attention call from Q/K/V projections. */
    memset(ws->context_buf, 0, (size_t)model->seq_len * (size_t)model->dim * sizeof(float));

    for (int head = 0; head < model->n_heads; ++head) {
        int head_offset = head * head_dim;
        for (int q_pos = 0; q_pos < model->seq_len; ++q_pos) {
            const float *q_vec = q_buf + (size_t)q_pos * (size_t)model->dim + (size_t)head_offset;
            float *context_dst = ws->context_buf + (size_t)q_pos * (size_t)model->dim + (size_t)head_offset;
            double max_score = -INFINITY;
            double sum = 0.0;

            for (int k_pos = 0; k_pos < model->seq_len; ++k_pos) {
                const float *k_vec = k_buf + (size_t)k_pos * (size_t)model->dim + (size_t)head_offset;
                float dot = 0.0f;
                double score = 0.0;

                for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    dot += q_vec[dim_idx] * k_vec[dim_idx];
                }
                score = (double)dot * (double)scale + (attention_mask[k_pos] != 0 ? 0.0 : (double)masked_bias);
                ws->softmax_buf[k_pos] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            memset(context_acc, 0, (size_t)head_dim * sizeof(double));

            for (int k_pos = 0; k_pos < model->seq_len; ++k_pos) {
                double exp_value = exp(ws->softmax_buf[k_pos] - max_score);
                ws->softmax_buf[k_pos] = exp_value;
                sum += exp_value;
            }

            if (sum <= 0.0f) {
                continue;
            }

            for (int k_pos = 0; k_pos < model->seq_len; ++k_pos) {
                const float *v_vec;
                double prob = ws->softmax_buf[k_pos] / sum;
                v_vec = v_buf + (size_t)k_pos * (size_t)model->dim + (size_t)head_offset;
                for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    context_acc[dim_idx] += prob * (double)v_vec[dim_idx];
                }
            }
            for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                context_dst[dim_idx] = (float)context_acc[dim_idx];
            }
        }
    }

    free(context_acc);
}

/* ==================== Debug Tensor Capture ==================== */

static void write_f32_file(const char *path, const float *data, size_t count) {
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }
    if (fwrite(data, sizeof(float), count, fp) != count) {
        fclose(fp);
        fatalf("fwrite failed for %s", path);
    }
    fclose(fp);
}

static void append_debug_record(DebugStore *store, const DebugRecord *record) {
    pthread_mutex_lock(&store->mutex);
    if (store->count == store->capacity) {
        size_t next_capacity = store->capacity == 0 ? 16 : store->capacity * 2;
        DebugRecord *next = (DebugRecord *)realloc(store->records, next_capacity * sizeof(DebugRecord));
        if (next == NULL) {
            pthread_mutex_unlock(&store->mutex);
            fatalf("realloc failed for debug records");
        }
        store->records = next;
        store->capacity = next_capacity;
    }
    store->records[store->count++] = *record;
    pthread_mutex_unlock(&store->mutex);
}

static void capture_debug_tensor(
    DebugStore *store,
    int sample_idx,
    const char *point,
    const float *tensor,
    int rows,
    int cols
) {
#if DISTILROBERTA_POWER_ONLY
    (void)store;
    (void)sample_idx;
    (void)point;
    (void)tensor;
    (void)rows;
    (void)cols;
    return;
#else
    char ref_path[PATH_MAX];
    char ref_file[256];
    char guest_dir[PATH_MAX];
    char guest_file[256];
    char guest_path[PATH_MAX];
    size_t elem_count = (size_t)rows * (size_t)cols;
    size_t compare_count = 0;
    int compare_rows = rows;
    size_t ref_count = 0;
    float *ref = NULL;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    DebugRecord record;

    snprintf_or_die(ref_file, sizeof(ref_file), "int8__sample_%d__%s.bin", sample_idx, point);
    join_path(guest_dir, sizeof(guest_dir), store->generated_dir, "debug_bin");
    join_path(ref_path, sizeof(ref_path), guest_dir, ref_file);
    /* Skip silently when that debug point was not exported by the host pipeline. */
    if (!file_exists(ref_path)) {
        return;
    }

    ref = (float *)load_binary_file_from_path(ref_path, sizeof(float), &ref_count);
    if (ref_count == elem_count) {
        compare_count = elem_count;
        compare_rows = rows;
    } else if (cols > 0 && ref_count % (size_t)cols == 0 && ref_count < elem_count) {
        compare_count = ref_count;
        compare_rows = (int)(ref_count / (size_t)cols);
    } else {
        free(ref);
        fatalf(
            "debug tensor size mismatch for sample=%d point=%s ref_count=%zu tensor_count=%zu",
            sample_idx,
            point,
            ref_count,
            elem_count
        );
    }

    for (size_t i = 0; i < compare_count; ++i) {
        float diff = fabsf(tensor[i] - ref[i]);
        max_abs = diff > max_abs ? diff : max_abs;
        mean_abs += diff;
        dot += (double)tensor[i] * (double)ref[i];
        lhs_norm += (double)tensor[i] * (double)tensor[i];
        rhs_norm += (double)ref[i] * (double)ref[i];
    }
    mean_abs /= (float)compare_count;

    if (store->dump_debug_bin) {
        /* Optional raw dump to inspect guest tensors side-by-side with host refs. */
        join_path(guest_dir, sizeof(guest_dir), store->output_dir, "debug_bin");
        ensure_dir_recursive(guest_dir);
        snprintf_or_die(guest_file, sizeof(guest_file), "guest__sample_%d__%s.bin", sample_idx, point);
        join_path(guest_path, sizeof(guest_path), guest_dir, guest_file);
        write_f32_file(guest_path, tensor, compare_count);
    }

    memset(&record, 0, sizeof(record));
    record.sample_idx = sample_idx;
    snprintf(record.point, sizeof(record.point), "%s", point);
    record.rows = compare_rows;
    record.cols = cols;
    record.max_abs = max_abs;
    record.mean_abs = mean_abs;
    if (lhs_norm <= 1.0e-12 || rhs_norm <= 1.0e-12) {
        record.cosine = 0.0f;
    } else {
        record.cosine = (float)(dot / sqrt(lhs_norm * rhs_norm));
    }
    append_debug_record(store, &record);
    free(ref);
#endif
}

/* ==================== End-to-End Forward Pass ==================== */

static void forward_sample(
    const DistilRobertaModel *model,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    int sample_idx,
    Workspace *ws,
    DebugStore *debug_store,
    pthread_mutex_t *log_mutex,
    int thread_id,
    bool serialize_gemmini,
    GemminiExecutor *gemmini_executor,
    float *logits_out
) {
    size_t token_count = (size_t)model->seq_len * (size_t)model->dim;
    char label[64];

    /* Top-level guest-side inference for one sample. */
    trace_timeline_event(log_mutex, thread_id, sample_idx, "vcpu", "embeddings");
    embeddings_forward(model, input_ids, ws->hidden_a);

    for (int layer_idx = 0; layer_idx < model->n_layers; ++layer_idx) {
        const TransformerLayer *layer = &model->layers[layer_idx];

        /*
         * The overall DistilRoBERTa flow is reproduced here in C.
         * The actual int8 GEMM work is delegated to linear_forward().
         */
        /* Self-attention block: Q/K/V projections -> attention -> output projection. */
        snprintf(label, sizeof(label), "L%d_qkv", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, linear_backend_actor(), label);
        linear_forward(&layer->q_lin, ws->hidden_a, model->seq_len, ws->q_buf, ws, serialize_gemmini, gemmini_executor);
        linear_forward(&layer->k_lin, ws->hidden_a, model->seq_len, ws->k_buf, ws, serialize_gemmini, gemmini_executor);
        linear_forward(&layer->v_lin, ws->hidden_a, model->seq_len, ws->v_buf, ws, serialize_gemmini, gemmini_executor);
        if (layer_idx == 0) {
            capture_debug_tensor(debug_store, sample_idx, "layer0_q_lin", ws->q_buf, model->seq_len, model->dim);
        }

        snprintf(label, sizeof(label), "L%d_attn", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, "vcpu", label);
        attention_forward(model, ws->q_buf, ws->k_buf, ws->v_buf, attention_mask, ws);
        snprintf(label, sizeof(label), "L%d_out", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, linear_backend_actor(), label);
        linear_forward(&layer->out_lin, ws->context_buf, model->seq_len, ws->hidden_b, ws, serialize_gemmini, gemmini_executor);
        if (layer_idx == 0) {
            capture_debug_tensor(debug_store, sample_idx, "layer0_attn_out", ws->hidden_b, model->seq_len, model->dim);
        }

        add_inplace(ws->hidden_b, ws->hidden_a, token_count);
        snprintf(label, sizeof(label), "L%d_sa_norm", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, "vcpu", label);
        layer_norm_inplace(ws->hidden_b, model->seq_len, &layer->sa_layer_norm);

        /* Feed-forward block: lin1 -> GELU -> lin2, then residual + LayerNorm. */
        snprintf(label, sizeof(label), "L%d_ffn1", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, linear_backend_actor(), label);
        linear_forward(&layer->ffn_lin1, ws->hidden_b, model->seq_len, ws->ffn_buf, ws, serialize_gemmini, gemmini_executor);
        if (layer_idx == 0) {
            capture_debug_tensor(
                debug_store,
                sample_idx,
                "layer0_ffn_lin1",
                ws->ffn_buf,
                model->seq_len,
                model->hidden_dim
            );
        }
        snprintf(label, sizeof(label), "L%d_gelu", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, "vcpu", label);
        gelu_inplace(ws->ffn_buf, (size_t)model->seq_len * (size_t)model->hidden_dim);

        snprintf(label, sizeof(label), "L%d_ffn2", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, linear_backend_actor(), label);
        linear_forward(&layer->ffn_lin2, ws->ffn_buf, model->seq_len, ws->hidden_a, ws, serialize_gemmini, gemmini_executor);
        add_inplace(ws->hidden_a, ws->hidden_b, token_count);
        snprintf(label, sizeof(label), "L%d_out_norm", layer_idx);
        trace_timeline_event(log_mutex, thread_id, sample_idx, "vcpu", label);
        layer_norm_inplace(ws->hidden_a, model->seq_len, &layer->output_layer_norm);
    }

    /*
     * Sequence classification head:
     * use the first token hidden state ([CLS] position) and map it to 2-class logits.
     */
    capture_debug_tensor(debug_store, sample_idx, "last_cls_hidden", ws->hidden_a, 1, model->dim);
    trace_timeline_event(log_mutex, thread_id, sample_idx, linear_backend_actor(), "pre_cls");
    linear_forward(&model->pre_classifier, ws->hidden_a, 1, ws->cls_buf, ws, serialize_gemmini, gemmini_executor);
    trace_timeline_event(log_mutex, thread_id, sample_idx, "vcpu", "cls_tanh");
    tanh_inplace(ws->cls_buf, (size_t)model->dim);
    trace_timeline_event(log_mutex, thread_id, sample_idx, linear_backend_actor(), "cls");
    linear_forward(&model->classifier, ws->cls_buf, 1, logits_out, ws, serialize_gemmini, gemmini_executor);
    capture_debug_tensor(debug_store, sample_idx, "logits", logits_out, 1, model->num_labels);
}

/* ==================== Multi-threaded Execution ==================== */

static void print_thread_log(pthread_mutex_t *mutex, const char *fmt, ...) {
    va_list args;
    pthread_mutex_lock(mutex);
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
    pthread_mutex_unlock(mutex);
}

static bool queue_pop(WorkQueue *queue, int *order_out) {
    bool has_work = false;
    pthread_mutex_lock(&queue->mutex);
    if (queue->next_order < queue->total) {
        *order_out = queue->next_order++;
        has_work = true;
    }
    pthread_mutex_unlock(&queue->mutex);
    return has_work;
}

static void *worker_main(void *arg) {
    WorkerArgs *worker = (WorkerArgs *)arg;
    const DistilRobertaModel *model = worker->model;
    const BaselineReference *baseline = worker->baseline;
    Workspace ws;
    int order = 0;

    if (fesetround(FE_TONEAREST) != 0) {
        fatalf("worker failed to set FE_TONEAREST");
    }

    trace_timeline_event(worker->log_mutex, worker->thread_id, 0, "vcpu", "worker_workspace");
    init_workspace(model, &ws);
    trace_timeline_event(worker->log_mutex, worker->thread_id, 0, "vcpu", "worker_pin");
    worker->pinned_cpu = pin_thread_to_vcpu(worker->vcpu_offset + worker->thread_id);

    /* Dynamic work stealing by sample order keeps thread utilization simple. */
    while (queue_pop(worker->queue, &order)) {
        const int global_order = worker->start_order + order;
        const int sample_idx = model->indices[global_order];
        const int label = model->labels[global_order];
        int baseline_pred = -1;
        float baseline_logit0 = 0.0f;
        float baseline_logit1 = 0.0f;
        const int32_t *input_ids = model->input_ids + (size_t)global_order * (size_t)model->seq_len;
        const int32_t *attention_mask = model->attention_mask + (size_t)global_order * (size_t)model->seq_len;
        float logits[2] = {0.0f, 0.0f};
        double start_ms = now_ms();
        int pred = 0;
        float diff0 = 0.0f;
        float diff1 = 0.0f;
        float logit_max_abs_diff = 0.0f;

        if (worker->use_baseline) {
            baseline_pred = baseline->preds[order];
            baseline_logit0 = baseline->logits[(size_t)order * (size_t)model->num_labels + 0];
            baseline_logit1 = baseline->logits[(size_t)order * (size_t)model->num_labels + 1];
        }

        print_thread_log(
            worker->log_mutex,
            "[sample-start] order=%d idx=%d thread=%d cpu=%d\n",
            order,
            sample_idx,
            worker->thread_id,
            sched_getcpu()
        );

        forward_sample(
            model,
            input_ids,
            attention_mask,
            sample_idx,
            &ws,
            worker->debug_store,
            worker->log_mutex,
            worker->thread_id,
            worker->serialize_gemmini,
            worker->gemmini_executor,
            logits
        );
        pred = logits[1] > logits[0] ? 1 : 0;

        worker->results[order].idx = sample_idx;
        worker->results[order].label = label;
        worker->results[order].baseline_pred = baseline_pred;
        worker->results[order].pred = pred;
        worker->results[order].logits[0] = logits[0];
        worker->results[order].logits[1] = logits[1];
        worker->results[order].baseline_logits[0] = baseline_logit0;
        worker->results[order].baseline_logits[1] = baseline_logit1;
        if (worker->use_baseline) {
            diff0 = fabsf(logits[0] - baseline_logit0);
            diff1 = fabsf(logits[1] - baseline_logit1);
            logit_max_abs_diff = diff0 > diff1 ? diff0 : diff1;
        }
        worker->results[order].logit_max_abs_diff = logit_max_abs_diff;
        worker->results[order].elapsed_ms = now_ms() - start_ms;

        if (worker->use_baseline) {
            print_thread_log(
                worker->log_mutex,
                "[sample-done] order=%d idx=%d thread=%d cpu=%d pinned_cpu=%d pred=%d label=%d baseline_pred=%d logits=[%.9g, %.9g] baseline_logits=[%.9g, %.9g] logit_max_abs_diff=%.9g elapsed_ms=%.3f\n",
                order,
                sample_idx,
                worker->thread_id,
                sched_getcpu(),
                worker->pinned_cpu,
                pred,
                label,
                baseline_pred,
                worker->results[order].logits[0],
                worker->results[order].logits[1],
                worker->results[order].baseline_logits[0],
                worker->results[order].baseline_logits[1],
                worker->results[order].logit_max_abs_diff,
                worker->results[order].elapsed_ms
            );
        } else {
            print_thread_log(
                worker->log_mutex,
                "[sample-done] order=%d idx=%d thread=%d cpu=%d pinned_cpu=%d pred=%d label=%d logits=[%.9g, %.9g] elapsed_ms=%.3f\n",
                order,
                sample_idx,
                worker->thread_id,
                sched_getcpu(),
                worker->pinned_cpu,
                pred,
                label,
                worker->results[order].logits[0],
                worker->results[order].logits[1],
                worker->results[order].elapsed_ms
            );
        }
    }

    return NULL;
}

static void init_debug_store(
    DebugStore *store,
    const char *generated_dir,
    const char *output_dir,
    bool dump_debug_bin
) {
    memset(store, 0, sizeof(*store));
    snprintf(store->generated_dir, sizeof(store->generated_dir), "%s", generated_dir);
    snprintf(store->output_dir, sizeof(store->output_dir), "%s", output_dir);
    store->dump_debug_bin = dump_debug_bin;
    pthread_mutex_init(&store->mutex, NULL);
}

/* ==================== Result Serialization ==================== */

static void compute_experiment_summary(
    const BaselineReference *baseline,
    const DistilRobertaModel *model,
    const SampleResult *results,
    int count,
    bool use_baseline,
    ExperimentSummary *summary
) {
    memset(summary, 0, sizeof(*summary));
    summary->mismatch_sample_indices = (int *)xcalloc((size_t)count, sizeof(int));

    for (int i = 0; i < count; ++i) {
        float sample_max = 0.0f;

        if (results[i].pred == results[i].label) {
            ++summary->correct;
        }
        if (use_baseline) {
            float host0 = baseline->logits[(size_t)i * (size_t)model->num_labels + 0];
            float host1 = baseline->logits[(size_t)i * (size_t)model->num_labels + 1];
            float diff0 = fabsf(results[i].logits[0] - host0);
            float diff1 = fabsf(results[i].logits[1] - host1);
            sample_max = diff0 > diff1 ? diff0 : diff1;

            if (results[i].baseline_pred == results[i].label) {
                ++summary->baseline_correct;
            }
            if (results[i].pred != results[i].baseline_pred) {
                summary->mismatch_sample_indices[summary->mismatch_count++] = results[i].idx;
                ++summary->pred_mismatches_vs_baseline;
            }
            if (sample_max > summary->max_logit_abs) {
                summary->max_logit_abs = sample_max;
            }
            summary->mean_logit_abs += sample_max;
        }
    }

    if (use_baseline) {
        summary->mean_logit_abs /= (double)count;
    }
}

static void write_predictions_jsonl(const char *path, const SampleResult *results, int count, bool use_baseline) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }
    for (int i = 0; i < count; ++i) {
        if (use_baseline) {
            fprintf(
                fp,
                "{\"idx\": %d, \"label\": %d, \"pred\": %d, \"baseline_pred\": %d, \"logits\": [%.9g, %.9g], \"baseline_logits\": [%.9g, %.9g], \"logit_max_abs_diff\": %.9g, \"elapsed_ms\": %.3f}\n",
                results[i].idx,
                results[i].label,
                results[i].pred,
                results[i].baseline_pred,
                results[i].logits[0],
                results[i].logits[1],
                results[i].baseline_logits[0],
                results[i].baseline_logits[1],
                results[i].logit_max_abs_diff,
                results[i].elapsed_ms
            );
        } else {
            fprintf(
                fp,
                "{\"idx\": %d, \"label\": %d, \"pred\": %d, \"logits\": [%.9g, %.9g], \"elapsed_ms\": %.3f}\n",
                results[i].idx,
                results[i].label,
                results[i].pred,
                results[i].logits[0],
                results[i].logits[1],
                results[i].elapsed_ms
            );
        }
    }
    fclose(fp);
}

static void write_debug_summary_json(const char *path, const DebugStore *store) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }

    fprintf(fp, "[\n");
    for (size_t i = 0; i < store->count; ++i) {
        const DebugRecord *record = &store->records[i];
        if (record->rows == 1) {
            fprintf(
                fp,
                "  {\"sample_idx\": %d, \"point\": \"%s\", \"shape\": [%d], \"max_abs\": %.9g, \"mean_abs\": %.9g, \"cosine\": %.9g}%s\n",
                record->sample_idx,
                record->point,
                record->cols,
                record->max_abs,
                record->mean_abs,
                record->cosine,
                (i + 1 == store->count) ? "" : ","
            );
        } else {
            fprintf(
                fp,
                "  {\"sample_idx\": %d, \"point\": \"%s\", \"shape\": [%d, %d], \"max_abs\": %.9g, \"mean_abs\": %.9g, \"cosine\": %.9g}%s\n",
                record->sample_idx,
                record->point,
                record->rows,
                record->cols,
                record->max_abs,
                record->mean_abs,
                record->cosine,
                (i + 1 == store->count) ? "" : ","
            );
        }
    }
    fprintf(fp, "]\n");
    fclose(fp);
}

static void write_metrics_json(
    const char *path,
    const ExperimentSummary *summary,
    int seq_len,
    int count,
    int num_threads,
    bool use_baseline,
    double total_elapsed_ms
) {
    FILE *fp = fopen(path, "w");

    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"backend\": \"%s\",\n", backend_name());
    fprintf(fp, "  \"num_samples\": %d,\n", count);
    fprintf(fp, "  \"seq_len\": %d,\n", seq_len);
    fprintf(fp, "  \"threads\": %d,\n", num_threads);
    fprintf(fp, "  \"baseline_compare_enabled\": %s,\n", use_baseline ? "true" : "false");
    fprintf(fp, "  \"backend_accuracy\": %.9f,\n", (double)summary->correct / (double)count);
    fprintf(fp, "  \"gemmini_accuracy\": %.9f,\n", (double)summary->correct / (double)count);
    if (use_baseline) {
        fprintf(fp, "  \"baseline_accuracy\": %.9f,\n", (double)summary->baseline_correct / (double)count);
        fprintf(fp, "  \"prediction_mismatches_vs_vcpu_baseline\": %d,\n", summary->pred_mismatches_vs_baseline);
        fprintf(fp, "  \"prediction_mismatch_sample_indices\": [");
        for (int i = 0; i < summary->mismatch_count; ++i) {
            fprintf(fp, "%s%d", i == 0 ? "" : ", ", summary->mismatch_sample_indices[i]);
        }
        fprintf(fp, "],\n");
        fprintf(fp, "  \"mean_logit_max_abs_vs_vcpu_baseline\": %.9f,\n", summary->mean_logit_abs);
        fprintf(fp, "  \"max_logit_max_abs_vs_vcpu_baseline\": %.9f,\n", summary->max_logit_abs);
    }
    write_gemmini_linear_profile_json(fp, total_elapsed_ms);
    fprintf(fp, "  \"elapsed_ms\": %.3f\n", total_elapsed_ms);
    fprintf(fp, "}\n");
    fclose(fp);
}

static bool copy_file(const char *src_path, const char *dst_path) {
    FILE *src = fopen(src_path, "rb");
    FILE *dst = NULL;
    char buffer[8192];
    size_t nread = 0;

    if (src == NULL) {
        return false;
    }

    dst = fopen(dst_path, "wb");
    if (dst == NULL) {
        fclose(src);
        return false;
    }

    while ((nread = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(buffer, 1, nread, dst) != nread) {
            fclose(src);
            fclose(dst);
            return false;
        }
    }

    fclose(src);
    fclose(dst);
    return true;
}

static void build_result_paths(const ProgramOptions *opts, ResultPaths *paths) {
    struct timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        fatalf("clock_gettime(CLOCK_REALTIME) failed: %s", strerror(errno));
    }

    memset(paths, 0, sizeof(*paths));
    snprintf_or_die(
        paths->run_name,
        sizeof(paths->run_name),
        "run_%lld_%d",
        (long long)ts.tv_sec,
        (int)getpid()
    );

    /* Stage files under guest-local /tmp first, then publish one completed run dir. */
    join_path(paths->temp_dir, sizeof(paths->temp_dir), DISTILROBERTA_BACKEND_TMP_ROOT, paths->run_name);
    join_path(paths->host_dir, sizeof(paths->host_dir), opts->output_dir, paths->run_name);

    join_path(paths->temp_metrics, sizeof(paths->temp_metrics), paths->temp_dir, "metrics.json");
    join_path(paths->temp_predictions, sizeof(paths->temp_predictions), paths->temp_dir, "predictions.jsonl");
    join_path(paths->temp_debug_summary, sizeof(paths->temp_debug_summary), paths->temp_dir, "debug_summary.json");

    join_path(paths->host_metrics, sizeof(paths->host_metrics), paths->host_dir, "metrics.json");
    join_path(paths->host_predictions, sizeof(paths->host_predictions), paths->host_dir, "predictions.jsonl");
    join_path(paths->host_debug_summary, sizeof(paths->host_debug_summary), paths->host_dir, "debug_summary.json");
}

/* ==================== CLI / Program Entry ==================== */

static void parse_args(int argc, char **argv, ProgramOptions *opts) {
    snprintf(opts->generated_dir, sizeof(opts->generated_dir), "generated");
    snprintf(opts->output_dir, sizeof(opts->output_dir), "%s", DISTILROBERTA_DEFAULT_OUTPUT_DIR);
    opts->baseline_predictions[0] = '\0';
    opts->start_order = 0;
    opts->limit = -1;
    opts->num_threads = DEFAULT_THREADS;
    opts->vcpu_offset = 0;
    opts->gemmini_vcpu = -1;
    opts->use_baseline = !DISTILROBERTA_POWER_ONLY;
    opts->dump_debug_bin = false;
    opts->serialize_gemmini = false;
    g_enable_timeline = false;
    g_log_backend_calls = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--generated-dir") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --generated-dir");
            }
            snprintf(opts->generated_dir, sizeof(opts->generated_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--output-dir") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --output-dir");
            }
            snprintf(opts->output_dir, sizeof(opts->output_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--baseline-predictions") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --baseline-predictions");
            }
            snprintf(opts->baseline_predictions, sizeof(opts->baseline_predictions), "%s", argv[++i]);
            opts->use_baseline = true;
        } else if (strcmp(argv[i], "--no-baseline") == 0) {
            opts->use_baseline = false;
        } else if (strcmp(argv[i], "--limit") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --limit");
            }
            opts->limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--start-order") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --start-order");
            }
            opts->start_order = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --threads");
            }
            opts->num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vcpu-offset") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --vcpu-offset");
            }
            opts->vcpu_offset = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gemmini-vcpu") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --gemmini-vcpu");
            }
            opts->gemmini_vcpu = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dump-debug-bin") == 0) {
            opts->dump_debug_bin = true;
        } else if (strcmp(argv[i], "--serialize-gemmini") == 0) {
            opts->serialize_gemmini = true;
        } else if (strcmp(argv[i], "--timeline") == 0) {
            g_enable_timeline = true;
        } else if (strcmp(argv[i], "--log-backend-calls") == 0) {
            g_log_backend_calls = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf(
                "Usage: %s [--generated-dir PATH] [--output-dir PATH] [--baseline-predictions PATH] [--no-baseline] [--start-order N] [--limit N] [--threads N] [--vcpu-offset N] [--gemmini-vcpu N] [--dump-debug-bin] [--serialize-gemmini] [--timeline] [--log-backend-calls]\n",
                argv[0]
            );
            exit(0);
        } else {
            fatalf("unknown argument: %s", argv[i]);
        }
    }

    if (opts->num_threads <= 0) {
        fatalf("threads must be positive");
    }
    if (opts->vcpu_offset < 0) {
        fatalf("vcpu_offset must be non-negative");
    }
    if (opts->gemmini_vcpu < -1) {
        fatalf("gemmini_vcpu must be -1 or non-negative");
    }
    if (opts->gemmini_vcpu >= 0 && opts->num_threads != 1) {
        fatalf("--gemmini-vcpu split mode currently supports --threads 1 only");
    }
#if !DISTILROBERTA_USE_GEMMINI_BACKEND
    if (opts->gemmini_vcpu >= 0) {
        fatalf("--gemmini-vcpu requires the gemmini backend");
    }
    opts->serialize_gemmini = false;
#endif
    if (opts->start_order < 0) {
        fatalf("start_order must be non-negative");
    }
}

int main(int argc, char **argv) {
    ProgramOptions opts;
    DistilRobertaModel model;
    BaselineReference baseline;
    DebugStore debug_store;
    ResultPaths result_paths;
    ExperimentSummary summary;
    SampleResult *results;
    WorkQueue queue;
    pthread_t *threads;
    WorkerArgs *worker_args;
#if DISTILROBERTA_USE_GEMMINI_BACKEND
    GemminiExecutor gemmini_executor;
#endif
    GemminiExecutor *gemmini_executor_ptr = NULL;
    pthread_mutex_t log_mutex;
    int total = 0;
    double start_ms = 0.0;
    double elapsed_ms = 0.0;
    bool host_export_ok = false;

    g_enable_timeline = argv_has_timeline_flag(argc, argv);
    emit_early_timeline_event("vcpu", "process_start");
    parse_args(argc, argv, &opts);
    int main_pinned_cpu = pin_thread_to_vcpu(opts.vcpu_offset);
    printf("[main-pin] target_vcpu=%d cpu=%d\n", opts.vcpu_offset, main_pinned_cpu);
    fflush(stdout);
    emit_early_timeline_event("vcpu", "main_pin");
    emit_early_timeline_event("vcpu", "args_parse_done");
    pthread_mutex_init(&log_mutex, NULL);

    /* Match the rounding mode assumed during host-side quantized export. */
    if (fesetround(FE_TONEAREST) != 0) {
        fatalf("failed to set FE_TONEAREST");
    }
    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "round_mode");

    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "model_load");
    init_model(&model, opts.generated_dir);
    if (opts.start_order >= model.eval_count) {
        fatalf("start_order=%d out of range for eval_count=%d", opts.start_order, model.eval_count);
    }
    total = model.eval_count - opts.start_order;
    if (opts.limit > 0 && opts.limit < total) {
        total = opts.limit;
    }
    if (total <= 0) {
        fatalf("no evaluation samples available");
    }
    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "sample_window");

    memset(&baseline, 0, sizeof(baseline));
    if (opts.use_baseline) {
        if (opts.baseline_predictions[0] == '\0') {
            trace_timeline_event(&log_mutex, 0, 0, "vcpu", "baseline_resolve");
            resolve_latest_baseline_predictions(opts.baseline_predictions, sizeof(opts.baseline_predictions), "../outputs/c_vcpu");
        }
        trace_timeline_event(&log_mutex, 0, 0, "vcpu", "baseline_load");
        load_baseline_reference(&baseline, opts.baseline_predictions, &model, opts.start_order, total);
    } else {
        trace_timeline_event(&log_mutex, 0, 0, "vcpu", "baseline_skip");
    }

    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "output_dir_setup");
    ensure_dir_recursive(opts.output_dir);
    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "debug_store_init");
    init_debug_store(&debug_store, opts.generated_dir, opts.output_dir, opts.dump_debug_bin);

    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "result_alloc");
    results = (SampleResult *)xcalloc((size_t)total, sizeof(SampleResult));
    threads = (pthread_t *)xcalloc((size_t)opts.num_threads, sizeof(pthread_t));
    worker_args = (WorkerArgs *)xcalloc((size_t)opts.num_threads, sizeof(WorkerArgs));

    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "queue_setup");
    pthread_mutex_init(&queue.mutex, NULL);
    queue.next_order = 0;
    queue.total = total;

    printf("[config] backend=%s generated_dir=%s output_dir=%s use_baseline=%d baseline_predictions=%s start_order=%d samples=%d threads=%d seq_len=%d vcpu_offset=%d gemmini_vcpu=%d dump_debug_bin=%d serialize_gemmini=%d timeline=%d log_backend_calls=%d\n",
           backend_name(), opts.generated_dir, opts.output_dir, opts.use_baseline ? 1 : 0, opts.baseline_predictions, opts.start_order, total, opts.num_threads, model.seq_len, opts.vcpu_offset, opts.gemmini_vcpu, opts.dump_debug_bin ? 1 : 0, opts.serialize_gemmini ? 1 : 0, g_enable_timeline ? 1 : 0, g_log_backend_calls ? 1 : 0);
    fflush(stdout);

    if (opts.gemmini_vcpu >= 0) {
#if DISTILROBERTA_USE_GEMMINI_BACKEND
#if DISTILROBERTA_POWER_ONLY
        trace_timeline_event(&log_mutex, 0, 0, "vcpu", "gemmini_executor_start");
#else
        trace_timeline_event(&log_mutex, 0, 0, "vcpu", "gemmini_prepare");
#endif
        gemmini_executor_start(&gemmini_executor, opts.gemmini_vcpu);
        gemmini_executor_ptr = &gemmini_executor;
#else
        fatalf("--gemmini-vcpu requires the gemmini backend");
#endif
    } else {
        trace_timeline_event(&log_mutex, 0, 0, "vcpu", DISTILROBERTA_USE_GEMMINI_BACKEND ? "gemmini_prepare" : "cpu_backend_prepare");
        prepare_gemmini_state();
    }

    trace_timeline_event(&log_mutex, 0, 0, "vcpu", "worker_spawn");
    start_ms = now_ms();
    for (int thread_idx = 0; thread_idx < opts.num_threads; ++thread_idx) {
        worker_args[thread_idx].model = &model;
        worker_args[thread_idx].baseline = &baseline;
        worker_args[thread_idx].start_order = opts.start_order;
        worker_args[thread_idx].queue = &queue;
        worker_args[thread_idx].results = results;
        worker_args[thread_idx].debug_store = &debug_store;
        worker_args[thread_idx].log_mutex = &log_mutex;
        worker_args[thread_idx].thread_id = thread_idx;
        worker_args[thread_idx].vcpu_offset = opts.vcpu_offset;
        worker_args[thread_idx].pinned_cpu = -1;
        worker_args[thread_idx].use_baseline = opts.use_baseline;
        worker_args[thread_idx].serialize_gemmini = opts.serialize_gemmini;
        worker_args[thread_idx].gemmini_executor = gemmini_executor_ptr;
        if (pthread_create(&threads[thread_idx], NULL, worker_main, &worker_args[thread_idx]) != 0) {
            fatalf("pthread_create failed");
        }
    }

    for (int thread_idx = 0; thread_idx < opts.num_threads; ++thread_idx) {
        if (pthread_join(threads[thread_idx], NULL) != 0) {
            fatalf("pthread_join failed");
        }
    }
    elapsed_ms = now_ms() - start_ms;
    if (gemmini_executor_ptr != NULL) {
#if DISTILROBERTA_USE_GEMMINI_BACKEND
        gemmini_executor_stop(gemmini_executor_ptr);
#endif
    }

    build_result_paths(&opts, &result_paths);
    ensure_dir_recursive(result_paths.temp_dir);

    compute_experiment_summary(&baseline, &model, results, total, opts.use_baseline, &summary);

    print_gemmini_linear_profile(elapsed_ms);
    write_metrics_json(result_paths.temp_metrics, &summary, model.seq_len, total, opts.num_threads, opts.use_baseline, elapsed_ms);
    write_predictions_jsonl(result_paths.temp_predictions, results, total, opts.use_baseline);
    write_debug_summary_json(result_paths.temp_debug_summary, &debug_store);

    if (try_ensure_dir_recursive(result_paths.host_dir)) {
        bool copied_metrics = copy_file(result_paths.temp_metrics, result_paths.host_metrics);
        bool copied_predictions = copy_file(result_paths.temp_predictions, result_paths.host_predictions);
        bool copied_debug = copy_file(result_paths.temp_debug_summary, result_paths.host_debug_summary);
        host_export_ok = copied_metrics && copied_predictions && copied_debug;
    }

    if (host_export_ok) {
        if (opts.use_baseline) {
            printf(
                "[done] backend=%s host_run_dir=%s mismatch=%d baseline_accuracy=%.9f backend_accuracy=%.9f metrics=%s predictions=%s debug_summary=%s tmp_backup=%s elapsed_ms=%.3f\n",
                backend_name(),
                result_paths.host_dir,
                summary.pred_mismatches_vs_baseline,
                (double)summary.baseline_correct / (double)total,
                (double)summary.correct / (double)total,
                result_paths.host_metrics,
                result_paths.host_predictions,
                result_paths.host_debug_summary,
                result_paths.temp_dir,
                elapsed_ms
            );
        } else {
            printf(
                "[done] backend=%s host_run_dir=%s baseline_compare=disabled backend_accuracy=%.9f metrics=%s predictions=%s debug_summary=%s tmp_backup=%s elapsed_ms=%.3f\n",
                backend_name(),
                result_paths.host_dir,
                (double)summary.correct / (double)total,
                result_paths.host_metrics,
                result_paths.host_predictions,
                result_paths.host_debug_summary,
                result_paths.temp_dir,
                elapsed_ms
            );
        }
    } else {
        if (opts.use_baseline) {
            printf(
                "[done] backend=%s host_export=failed tmp_run_dir=%s mismatch=%d baseline_accuracy=%.9f backend_accuracy=%.9f metrics=%s predictions=%s debug_summary=%s elapsed_ms=%.3f\n",
                backend_name(),
                result_paths.temp_dir,
                summary.pred_mismatches_vs_baseline,
                (double)summary.baseline_correct / (double)total,
                (double)summary.correct / (double)total,
                result_paths.temp_metrics,
                result_paths.temp_predictions,
                result_paths.temp_debug_summary,
                elapsed_ms
            );
        } else {
            printf(
                "[done] backend=%s host_export=failed tmp_run_dir=%s baseline_compare=disabled backend_accuracy=%.9f metrics=%s predictions=%s debug_summary=%s elapsed_ms=%.3f\n",
                backend_name(),
                result_paths.temp_dir,
                (double)summary.correct / (double)total,
                result_paths.temp_metrics,
                result_paths.temp_predictions,
                result_paths.temp_debug_summary,
                elapsed_ms
            );
        }
    }
    if (opts.use_baseline) {
        if (summary.mismatch_count > 0) {
            printf("[summary] mismatch_sample_indices=");
            for (int i = 0; i < summary.mismatch_count; ++i) {
                printf("%s%d", i == 0 ? "[" : ", ", summary.mismatch_sample_indices[i]);
            }
            printf("]\n");
        } else {
            printf("[summary] mismatch_sample_indices=[]\n");
        }
    } else {
        printf("[summary] baseline_compare=disabled\n");
    }
    fflush(stdout);
    if (opts.use_baseline) {
        free_baseline_reference(&baseline);
    }
    return 0;
}
