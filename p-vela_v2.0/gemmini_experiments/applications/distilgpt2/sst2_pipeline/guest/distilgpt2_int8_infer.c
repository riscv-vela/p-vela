#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

/*
 * distilgpt2_int8_infer.c
 *
 * Role in the pipeline:
 * - CPU-only reference implementation for the guest-side int8 DistilGPT2 path.
 * - Reproduces the host int8 model exported by 03_export_c_model.py.
 * - Keeps non-GEMM logic in plain C so later Gemmini integration can replace
 *   only the projection/GEMM part and compare against this baseline.
 *
 * What this program does:
 * - Loads model weights/scales and evaluation samples from generated/{model_bin,eval_bin}.
 * - Runs multi-threaded inference on the guest/vCPU.
 * - Compares selected intermediate tensors against exported debug references.
 * - Writes metrics, predictions, and debug summaries for downstream vCPU/Gemmini comparison.
 */

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MAX_LAYERS 6
#define DEFAULT_THREADS 4
#define LAYERNORM_EPSILON 1.0e-5f
#define GPT2_GELU_COEFF 0.044715f

#ifndef DISTILGPT2_CPU_POWER_ONLY
#define DISTILGPT2_CPU_POWER_ONLY 0
#endif

/* ==================== Model/Data Structures ==================== */

typedef struct {
    int in_features;
    int out_features;
    float input_scale;
    float output_scale;
    int lora_rank;
    float lora_scaling;
    int8_t *weight_q;
    float *weight_scale;
    float *bias_fp32;
    float *acc_scale;
    float *lora_a;
    float *lora_b;
} LinearLayer;

typedef struct {
    float *weight;
    float *bias;
    int dim;
} LayerNorm;

typedef struct {
    LinearLayer qkv_lin;
    LinearLayer out_lin;
    LinearLayer ffn_lin1;
    LinearLayer ffn_lin2;
    LayerNorm ln_1;
    LayerNorm ln_2;
} Gpt2Block;

typedef struct {
    int n_layers;
    int dim;
    int hidden_dim;
    int n_heads;
    int max_position_embeddings;
    int num_labels;
    int vocab_size;
    int pad_token_id;
    int eos_token_id;
    int seq_len;
    int eval_count;
    float *word_embeddings;
    float *position_embeddings;
    LayerNorm final_layer_norm;
    Gpt2Block layers[MAX_LAYERS];
    float *score_weight;
    int32_t *indices;
    int32_t *input_ids;
    int32_t *attention_mask;
    int32_t *input_length;
    int32_t *last_token_idx;
    int32_t *labels;
    int32_t *host_preds;
    float *host_logits;
} DistilGpt2Model;

typedef struct {
    /* Scratch buffers reused across samples to avoid reallocating per layer. */
    float *hidden_a;
    float *hidden_b;
    float *norm_buf;
    float *qkv_buf;
    float *q_buf;
    float *k_buf;
    float *v_buf;
    float *context_buf;
    float *ffn_buf;
    float *lora_buf;
    double *softmax_buf;
    int32_t *quant_buf;
} Workspace;

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
    int host_pred;
    int pred;
    float logits[2];
    float host_logits[2];
    float logit_max_abs_diff;
    double elapsed_ms;
} SampleResult;

typedef struct {
    int next_order;
    int total;
    pthread_mutex_t mutex;
} WorkQueue;

typedef struct {
    const DistilGpt2Model *model;
    WorkQueue *queue;
    SampleResult *results;
    DebugStore *debug_store;
    pthread_mutex_t *log_mutex;
    int thread_id;
    int vcpu_offset;
    int pinned_cpu;
} WorkerArgs;

typedef struct {
    char generated_dir[PATH_MAX];
    char output_dir[PATH_MAX];
    int limit;
    int num_threads;
    int vcpu_offset;
    bool dump_debug_bin;
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
    int host_correct;
    int pred_mismatches_vs_host;
    float max_logit_abs;
    double mean_logit_abs;
    int *mismatch_sample_indices;
    int mismatch_count;
} ExperimentSummary;

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

static int pin_current_thread_to_vcpu(int guest_cpu) {
    cpu_set_t cpuset;
    int rc = 0;

    if (guest_cpu < 0) {
        return sched_getcpu();
    }

    CPU_ZERO(&cpuset);
    CPU_SET(guest_cpu, &cpuset);
    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fatalf("pthread_setaffinity_np failed for guest_cpu=%d: %s", guest_cpu, strerror(rc));
    }
    return sched_getcpu();
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

static bool file_exists(const char *path) {
    return access(path, F_OK) == 0;
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
    char file_path[PATH_MAX];
    size_t out_features = 0;

    memset(layer, 0, sizeof(*layer));

    snprintf(file_name, sizeof(file_name), "%s__in_features.bin", base_name);
    layer->in_features = load_i32_scalar(model_dir, file_name);

    snprintf(file_name, sizeof(file_name), "%s__out_features.bin", base_name);
    layer->out_features = load_i32_scalar(model_dir, file_name);
    out_features = (size_t)layer->out_features;

    snprintf(file_name, sizeof(file_name), "%s__input_scale.bin", base_name);
    layer->input_scale = load_f32_scalar(model_dir, file_name);

    snprintf(file_name, sizeof(file_name), "%s__output_scale.bin", base_name);
    layer->output_scale = load_f32_scalar(model_dir, file_name);

    snprintf(file_name, sizeof(file_name), "%s__weight_q.bin", base_name);
    layer->weight_q = load_i8_array(
        model_dir,
        file_name,
        (size_t)layer->in_features * (size_t)layer->out_features
    );

    snprintf(file_name, sizeof(file_name), "%s__weight_scale.bin", base_name);
    layer->weight_scale = load_f32_array(model_dir, file_name, out_features);

    snprintf(file_name, sizeof(file_name), "%s__bias_fp32.bin", base_name);
    layer->bias_fp32 = load_f32_array(model_dir, file_name, out_features);

    layer->acc_scale = (float *)xmalloc(out_features * sizeof(float));
    /* int32 accumulator -> fp32 rescale factor for each output channel. */
    for (size_t i = 0; i < out_features; ++i) {
        layer->acc_scale[i] = layer->input_scale * layer->weight_scale[i];
    }

    snprintf(file_name, sizeof(file_name), "%s__lora_rank.bin", base_name);
    join_path(file_path, sizeof(file_path), model_dir, file_name);
    if (!file_exists(file_path)) {
        return;
    }

    layer->lora_rank = load_i32_scalar(model_dir, file_name);
    if (layer->lora_rank <= 0) {
        fatalf("invalid lora_rank=%d for %s", layer->lora_rank, base_name);
    }

    snprintf(file_name, sizeof(file_name), "%s__lora_scaling.bin", base_name);
    layer->lora_scaling = load_f32_scalar(model_dir, file_name);

    snprintf(file_name, sizeof(file_name), "%s__lora_a.bin", base_name);
    layer->lora_a = load_f32_array(
        model_dir,
        file_name,
        (size_t)layer->lora_rank * (size_t)layer->in_features
    );

    snprintf(file_name, sizeof(file_name), "%s__lora_b.bin", base_name);
    layer->lora_b = load_f32_array(
        model_dir,
        file_name,
        (size_t)layer->out_features * (size_t)layer->lora_rank
    );
}

static void load_model(DistilGpt2Model *model, const char *generated_dir) {
    char model_dir[PATH_MAX];
    char eval_dir[PATH_MAX];
    size_t count = 0;

    memset(model, 0, sizeof(*model));

    join_path(model_dir, sizeof(model_dir), generated_dir, "model_bin");
    join_path(eval_dir, sizeof(eval_dir), generated_dir, "eval_bin");

    model->n_layers = load_i32_scalar(model_dir, "config__n_layers.bin");
    model->dim = load_i32_scalar(model_dir, "config__dim.bin");
    model->hidden_dim = load_i32_scalar(model_dir, "config__hidden_dim.bin");
    model->n_heads = load_i32_scalar(model_dir, "config__n_heads.bin");
    model->max_position_embeddings = load_i32_scalar(model_dir, "config__max_position_embeddings.bin");
    model->num_labels = load_i32_scalar(model_dir, "config__num_labels.bin");
    model->pad_token_id = load_i32_scalar(model_dir, "config__pad_token_id.bin");
    model->eos_token_id = load_i32_scalar(model_dir, "config__eos_token_id.bin");

    if (model->n_layers <= 0 || model->n_layers > MAX_LAYERS) {
        fatalf("unsupported n_layers=%d", model->n_layers);
    }
    if (model->dim <= 0 || model->hidden_dim <= 0 || model->n_heads <= 0) {
        fatalf("invalid model config");
    }
    if (model->dim % model->n_heads != 0) {
        fatalf("dim=%d not divisible by n_heads=%d", model->dim, model->n_heads);
    }

    model->word_embeddings = (float *)load_binary_file(
        model_dir,
        "transformer__wte__weight.bin",
        sizeof(float),
        &count
    );
    if (count % (size_t)model->dim != 0) {
        fatalf("word embedding size mismatch");
    }
    model->vocab_size = (int)(count / (size_t)model->dim);

    model->position_embeddings = (float *)load_binary_file(
        model_dir,
        "transformer__wpe__weight.bin",
        sizeof(float),
        &count
    );
    if (count != (size_t)model->max_position_embeddings * (size_t)model->dim) {
        fatalf("position embedding size mismatch");
    }

    load_layer_norm(model_dir, "transformer__ln_f", model->dim, &model->final_layer_norm);

    for (int layer_idx = 0; layer_idx < model->n_layers; ++layer_idx) {
        Gpt2Block *layer = &model->layers[layer_idx];
        char base_name[128];

        snprintf(base_name, sizeof(base_name), "layer%d__qkv_lin", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->qkv_lin);
        snprintf(base_name, sizeof(base_name), "layer%d__out_lin", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->out_lin);
        snprintf(base_name, sizeof(base_name), "layer%d__ffn_lin1", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->ffn_lin1);
        snprintf(base_name, sizeof(base_name), "layer%d__ffn_lin2", layer_idx);
        load_linear_layer(model_dir, base_name, &layer->ffn_lin2);

        snprintf(base_name, sizeof(base_name), "transformer__h__%d__ln_1", layer_idx);
        load_layer_norm(model_dir, base_name, model->dim, &layer->ln_1);
        snprintf(base_name, sizeof(base_name), "transformer__h__%d__ln_2", layer_idx);
        load_layer_norm(model_dir, base_name, model->dim, &layer->ln_2);
    }

    {
        char score_path[PATH_MAX];
        join_path(score_path, sizeof(score_path), model_dir, "score__modules_to_save__default__weight.bin");
        if (!file_exists(score_path)) {
            join_path(score_path, sizeof(score_path), model_dir, "score__original_module__weight.bin");
        }
        model->score_weight = (float *)load_binary_file_from_path(score_path, sizeof(float), &count);
        if (count != (size_t)model->num_labels * (size_t)model->dim) {
            fatalf("score weight size mismatch");
        }
    }

    model->labels = (int32_t *)load_binary_file(eval_dir, "labels.bin", sizeof(int32_t), &count);
    model->eval_count = (int)count;

    model->indices = load_i32_array(eval_dir, "indices.bin", (size_t)model->eval_count);
    model->host_preds = load_i32_array(eval_dir, "host_int8_preds.bin", (size_t)model->eval_count);
    model->input_length = load_i32_array(eval_dir, "input_length.bin", (size_t)model->eval_count);
    model->last_token_idx = load_i32_array(eval_dir, "last_token_idx.bin", (size_t)model->eval_count);
    model->host_logits = load_f32_array(
        eval_dir,
        "host_int8_logits.bin",
        (size_t)model->eval_count * (size_t)model->num_labels
    );
}

static void load_eval_inputs(DistilGpt2Model *model, const char *generated_dir) {
    char eval_dir[PATH_MAX];
    size_t input_count = 0;
    size_t mask_count = 0;

    join_path(eval_dir, sizeof(eval_dir), generated_dir, "eval_bin");

    model->input_ids = (int32_t *)load_binary_file(eval_dir, "input_ids.bin", sizeof(int32_t), &input_count);
    if (model->eval_count <= 0 || input_count % (size_t)model->eval_count != 0) {
        fatalf("input_ids count does not match eval_count");
    }
    /* Sequence length is inferred from exported evaluation tensors, not hardcoded. */
    model->seq_len = (int)(input_count / (size_t)model->eval_count);

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

static void init_model(DistilGpt2Model *model, const char *generated_dir) {
    load_model(model, generated_dir);
    load_eval_inputs(model, generated_dir);
}

static void init_workspace(const DistilGpt2Model *model, Workspace *ws) {
    size_t token_count = (size_t)model->seq_len * (size_t)model->dim;
    size_t ffn_count = (size_t)model->seq_len * (size_t)model->hidden_dim;
    size_t qkv_count = (size_t)model->seq_len * (size_t)model->dim * 3u;
    int max_in_features = model->hidden_dim > model->dim ? model->hidden_dim : model->dim;
    int max_lora_rank = 1;

    for (int layer_idx = 0; layer_idx < model->n_layers; ++layer_idx) {
        const Gpt2Block *layer = &model->layers[layer_idx];
        if (layer->qkv_lin.lora_rank > max_lora_rank) {
            max_lora_rank = layer->qkv_lin.lora_rank;
        }
        if (layer->out_lin.lora_rank > max_lora_rank) {
            max_lora_rank = layer->out_lin.lora_rank;
        }
        if (layer->ffn_lin1.lora_rank > max_lora_rank) {
            max_lora_rank = layer->ffn_lin1.lora_rank;
        }
        if (layer->ffn_lin2.lora_rank > max_lora_rank) {
            max_lora_rank = layer->ffn_lin2.lora_rank;
        }
    }

    memset(ws, 0, sizeof(*ws));
    ws->hidden_a = (float *)xcalloc(token_count, sizeof(float));
    ws->hidden_b = (float *)xcalloc(token_count, sizeof(float));
    ws->norm_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->qkv_buf = (float *)xcalloc(qkv_count, sizeof(float));
    ws->q_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->k_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->v_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->context_buf = (float *)xcalloc(token_count, sizeof(float));
    ws->ffn_buf = (float *)xcalloc(ffn_count, sizeof(float));
    ws->lora_buf = (float *)xcalloc((size_t)max_lora_rank, sizeof(float));
    ws->softmax_buf = (double *)xcalloc((size_t)model->seq_len, sizeof(double));
    ws->quant_buf = (int32_t *)xcalloc((size_t)max_in_features, sizeof(int32_t));
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

static void linear_forward(
    const LinearLayer *layer,
    const float *input,
    int rows,
    float *output,
    int32_t *quant_buf,
    float *lora_buf
) {
    for (int row = 0; row < rows; ++row) {
        const float *row_input = input + (size_t)row * (size_t)layer->in_features;
        float *row_output = output + (size_t)row * (size_t)layer->out_features;

        for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
            quant_buf[in_idx] = quantize_ties_to_even(row_input[in_idx], layer->input_scale);
        }

        for (int out_idx = 0; out_idx < layer->out_features; ++out_idx) {
            const int8_t *weight_row = layer->weight_q + (size_t)out_idx * (size_t)layer->in_features;
            int32_t acc = 0;
            for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
                acc += quant_buf[in_idx] * (int32_t)weight_row[in_idx];
            }
            row_output[out_idx] = (float)acc * layer->acc_scale[out_idx] + layer->bias_fp32[out_idx];
        }

        if (layer->lora_rank > 0) {
            for (int rank_idx = 0; rank_idx < layer->lora_rank; ++rank_idx) {
                const float *lora_a_row = layer->lora_a + (size_t)rank_idx * (size_t)layer->in_features;
                double acc = 0.0;
                for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
                    acc += (double)row_input[in_idx] * (double)lora_a_row[in_idx];
                }
                lora_buf[rank_idx] = (float)acc;
            }

            for (int out_idx = 0; out_idx < layer->out_features; ++out_idx) {
                const float *lora_b_row = layer->lora_b + (size_t)out_idx * (size_t)layer->lora_rank;
                double delta = 0.0;
                for (int rank_idx = 0; rank_idx < layer->lora_rank; ++rank_idx) {
                    delta += (double)lora_b_row[rank_idx] * (double)lora_buf[rank_idx];
                }
                row_output[out_idx] += (float)(delta * (double)layer->lora_scaling);
            }
        }
    }
}

static void add_inplace(float *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

static void layer_norm_inplace(float *tensor, int rows, const LayerNorm *layer_norm) {
    int dim = layer_norm->dim;
    for (int row = 0; row < rows; ++row) {
        float *row_ptr = tensor + (size_t)row * (size_t)dim;
        double mean = 0.0;
        double var = 0.0;
        double inv_std = 0.0;

        for (int col = 0; col < dim; ++col) {
            mean += (double)row_ptr[col];
        }
        mean /= (double)dim;

        for (int col = 0; col < dim; ++col) {
            double delta = (double)row_ptr[col] - mean;
            var += delta * delta;
        }
        var /= (double)dim;
        inv_std = 1.0 / sqrt(var + (double)LAYERNORM_EPSILON);
        for (int col = 0; col < dim; ++col) {
            float normalized = (float)(((double)row_ptr[col] - mean) * inv_std);
            row_ptr[col] = normalized * layer_norm->weight[col] + layer_norm->bias[col];
        }
    }
}

static void gelu_inplace(float *values, size_t count) {
    const double sqrt_2_over_pi = 0.7978845608028654;
    for (size_t i = 0; i < count; ++i) {
        double x = (double)values[i];
        double x3 = x * x * x;
        double inner = sqrt_2_over_pi * (x + (double)GPT2_GELU_COEFF * x3);
        values[i] = (float)(0.5 * x * (1.0 + tanh(inner)));
    }
}

/* ==================== Embedding / Attention ==================== */

static void embeddings_forward(
    const DistilGpt2Model *model,
    const int32_t *input_ids,
    int seq_len,
    float *hidden
) {
    for (int pos = 0; pos < seq_len; ++pos) {
        int32_t token_id = input_ids[pos];
        float *dst = hidden + (size_t)pos * (size_t)model->dim;
        const float *word_src;
        const float *pos_src;

        if (token_id < 0 || token_id >= model->vocab_size) {
            fatalf("token_id out of range: %d", token_id);
        }
        word_src = model->word_embeddings + (size_t)token_id * (size_t)model->dim;
        pos_src = model->position_embeddings + (size_t)pos * (size_t)model->dim;
        for (int dim_idx = 0; dim_idx < model->dim; ++dim_idx) {
            dst[dim_idx] = word_src[dim_idx] + pos_src[dim_idx];
        }
    }
}

static void split_qkv(
    const DistilGpt2Model *model,
    const float *qkv_buf,
    int seq_len,
    Workspace *ws
) {
    for (int pos = 0; pos < seq_len; ++pos) {
        const float *src = qkv_buf + (size_t)pos * (size_t)model->dim * 3u;
        float *q_dst = ws->q_buf + (size_t)pos * (size_t)model->dim;
        float *k_dst = ws->k_buf + (size_t)pos * (size_t)model->dim;
        float *v_dst = ws->v_buf + (size_t)pos * (size_t)model->dim;
        memcpy(q_dst, src, (size_t)model->dim * sizeof(float));
        memcpy(k_dst, src + model->dim, (size_t)model->dim * sizeof(float));
        memcpy(v_dst, src + (size_t)model->dim * 2u, (size_t)model->dim * sizeof(float));
    }
}

static void classifier_forward(
    const DistilGpt2Model *model,
    const float *hidden_vec,
    float *logits_out
) {
    for (int label_idx = 0; label_idx < model->num_labels; ++label_idx) {
        const float *weight_row = model->score_weight + (size_t)label_idx * (size_t)model->dim;
        double acc = 0.0;
        for (int dim_idx = 0; dim_idx < model->dim; ++dim_idx) {
            acc += (double)hidden_vec[dim_idx] * (double)weight_row[dim_idx];
        }
        logits_out[label_idx] = (float)acc;
    }
}

static void attention_forward(
    const DistilGpt2Model *model,
    const float *q_buf,
    const float *k_buf,
    const float *v_buf,
    const int32_t *attention_mask,
    int seq_len,
    Workspace *ws
) {
    const int head_dim = model->dim / model->n_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const double masked_bias = -1.0e30;
    double *context_acc = (double *)xcalloc((size_t)head_dim, sizeof(double));

    memset(ws->context_buf, 0, (size_t)seq_len * (size_t)model->dim * sizeof(float));

    for (int head = 0; head < model->n_heads; ++head) {
        int head_offset = head * head_dim;
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            const float *q_vec = q_buf + (size_t)q_pos * (size_t)model->dim + (size_t)head_offset;
            float *context_dst = ws->context_buf + (size_t)q_pos * (size_t)model->dim + (size_t)head_offset;
            double max_score = -INFINITY;
            double sum = 0.0;

            for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                const float *k_vec = k_buf + (size_t)k_pos * (size_t)model->dim + (size_t)head_offset;
                double dot = 0.0;
                double score = 0.0;

                for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    dot += (double)q_vec[dim_idx] * (double)k_vec[dim_idx];
                }
                if (k_pos > q_pos || attention_mask[k_pos] == 0) {
                    score = masked_bias;
                } else {
                    score = dot * (double)scale;
                }
                ws->softmax_buf[k_pos] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            memset(context_acc, 0, (size_t)head_dim * sizeof(double));

            for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                double exp_value = exp(ws->softmax_buf[k_pos] - max_score);
                ws->softmax_buf[k_pos] = exp_value;
                sum += exp_value;
            }

            if (sum <= 0.0) {
                continue;
            }

            for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
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

    if (DISTILGPT2_CPU_POWER_ONLY || store == NULL) {
        return;
    }

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
}

/* ==================== End-to-End Forward Pass ==================== */

static void forward_sample(
    const DistilGpt2Model *model,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    int input_length,
    int last_token_idx,
    int sample_idx,
    Workspace *ws,
    DebugStore *debug_store,
    float *logits_out
) {
    size_t token_count = (size_t)input_length * (size_t)model->dim;

    embeddings_forward(model, input_ids, input_length, ws->hidden_a);

    for (int layer_idx = 0; layer_idx < model->n_layers; ++layer_idx) {
        const Gpt2Block *layer = &model->layers[layer_idx];

        memcpy(ws->norm_buf, ws->hidden_a, token_count * sizeof(float));
        layer_norm_inplace(ws->norm_buf, input_length, &layer->ln_1);

        linear_forward(&layer->qkv_lin, ws->norm_buf, input_length, ws->qkv_buf, ws->quant_buf, ws->lora_buf);
        if (layer_idx == 0) {
            capture_debug_tensor(debug_store, sample_idx, "layer0_qkv_lin", ws->qkv_buf, input_length, model->dim * 3);
        }
        split_qkv(model, ws->qkv_buf, input_length, ws);

        attention_forward(model, ws->q_buf, ws->k_buf, ws->v_buf, attention_mask, input_length, ws);
        linear_forward(&layer->out_lin, ws->context_buf, input_length, ws->hidden_b, ws->quant_buf, ws->lora_buf);
        if (layer_idx == 0) {
            capture_debug_tensor(debug_store, sample_idx, "layer0_out_lin", ws->hidden_b, input_length, model->dim);
        }

        add_inplace(ws->hidden_b, ws->hidden_a, token_count);
        memcpy(ws->norm_buf, ws->hidden_b, token_count * sizeof(float));
        layer_norm_inplace(ws->norm_buf, input_length, &layer->ln_2);

        linear_forward(&layer->ffn_lin1, ws->norm_buf, input_length, ws->ffn_buf, ws->quant_buf, ws->lora_buf);
        if (layer_idx == 0) {
            capture_debug_tensor(
                debug_store,
                sample_idx,
                "layer0_ffn_lin1",
                ws->ffn_buf,
                input_length,
                model->hidden_dim
            );
        }
        gelu_inplace(ws->ffn_buf, (size_t)input_length * (size_t)model->hidden_dim);

        linear_forward(&layer->ffn_lin2, ws->ffn_buf, input_length, ws->hidden_a, ws->quant_buf, ws->lora_buf);
        add_inplace(ws->hidden_a, ws->hidden_b, token_count);
    }

    layer_norm_inplace(ws->hidden_a, input_length, &model->final_layer_norm);
    if (last_token_idx < 0 || last_token_idx >= input_length) {
        fatalf("invalid last_token_idx=%d", last_token_idx);
    }
    capture_debug_tensor(
        debug_store,
        sample_idx,
        "last_token_hidden",
        ws->hidden_a + (size_t)last_token_idx * (size_t)model->dim,
        1,
        model->dim
    );
    classifier_forward(
        model,
        ws->hidden_a + (size_t)last_token_idx * (size_t)model->dim,
        logits_out
    );
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
    const DistilGpt2Model *model = worker->model;
    Workspace ws;
    int order = 0;
    int target_vcpu = worker->vcpu_offset >= 0 ? worker->vcpu_offset + worker->thread_id : -1;

    worker->pinned_cpu = pin_current_thread_to_vcpu(target_vcpu);
    init_workspace(model, &ws);
    print_thread_log(
        worker->log_mutex,
        "[worker-pin] thread=%d target_vcpu=%d cpu=%d\n",
        worker->thread_id,
        target_vcpu,
        worker->pinned_cpu
    );

    /* Dynamic work stealing by sample order keeps thread utilization simple. */
    while (queue_pop(worker->queue, &order)) {
        const int sample_idx = model->indices[order];
        const int label = model->labels[order];
        const int host_pred = DISTILGPT2_CPU_POWER_ONLY ? -1 : model->host_preds[order];
        const float host_logit0 = DISTILGPT2_CPU_POWER_ONLY ? 0.0f : model->host_logits[(size_t)order * (size_t)model->num_labels + 0];
        const float host_logit1 = DISTILGPT2_CPU_POWER_ONLY ? 0.0f : model->host_logits[(size_t)order * (size_t)model->num_labels + 1];
        const int32_t *input_ids = model->input_ids + (size_t)order * (size_t)model->seq_len;
        const int32_t *attention_mask = model->attention_mask + (size_t)order * (size_t)model->seq_len;
        const int input_length = model->input_length[order];
        const int last_token_idx = model->last_token_idx[order];
        float logits[2] = {0.0f, 0.0f};
        double start_ms = now_ms();
        int pred = 0;
        float diff0 = 0.0f;
        float diff1 = 0.0f;
        float logit_max_abs_diff = 0.0f;

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
            input_length,
            last_token_idx,
            sample_idx,
            &ws,
            worker->debug_store,
            logits
        );
        pred = logits[1] > logits[0] ? 1 : 0;

        worker->results[order].idx = sample_idx;
        worker->results[order].label = label;
        worker->results[order].host_pred = host_pred;
        worker->results[order].pred = pred;
        worker->results[order].logits[0] = logits[0];
        worker->results[order].logits[1] = logits[1];
        worker->results[order].host_logits[0] = host_logit0;
        worker->results[order].host_logits[1] = host_logit1;
        if (!DISTILGPT2_CPU_POWER_ONLY) {
            diff0 = fabsf(logits[0] - host_logit0);
            diff1 = fabsf(logits[1] - host_logit1);
            logit_max_abs_diff = diff0 > diff1 ? diff0 : diff1;
        }
        worker->results[order].logit_max_abs_diff = logit_max_abs_diff;
        worker->results[order].elapsed_ms = now_ms() - start_ms;

        if (DISTILGPT2_CPU_POWER_ONLY) {
            print_thread_log(
                worker->log_mutex,
                "[sample-done] order=%d idx=%d thread=%d cpu=%d pred=%d label=%d elapsed_ms=%.3f\n",
                order,
                sample_idx,
                worker->thread_id,
                sched_getcpu(),
                pred,
                label,
                worker->results[order].elapsed_ms
            );
        } else {
            print_thread_log(
                worker->log_mutex,
                "[sample-done] order=%d idx=%d thread=%d cpu=%d pred=%d label=%d host_pred=%d logits=[%.9g, %.9g] host_logits=[%.9g, %.9g] logit_max_abs_diff=%.9g elapsed_ms=%.3f\n",
                order,
                sample_idx,
                worker->thread_id,
                sched_getcpu(),
                pred,
                label,
                host_pred,
                worker->results[order].logits[0],
                worker->results[order].logits[1],
                worker->results[order].host_logits[0],
                worker->results[order].host_logits[1],
                worker->results[order].logit_max_abs_diff,
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
    const DistilGpt2Model *model,
    const SampleResult *results,
    int count,
    ExperimentSummary *summary
) {
    memset(summary, 0, sizeof(*summary));
    summary->mismatch_sample_indices = (int *)xcalloc((size_t)count, sizeof(int));

    for (int i = 0; i < count; ++i) {
        float host0 = model->host_logits[(size_t)i * (size_t)model->num_labels + 0];
        float host1 = model->host_logits[(size_t)i * (size_t)model->num_labels + 1];
        float diff0 = fabsf(results[i].logits[0] - host0);
        float diff1 = fabsf(results[i].logits[1] - host1);
        float sample_max = diff0 > diff1 ? diff0 : diff1;

        if (results[i].pred == results[i].label) {
            ++summary->correct;
        }
        if (results[i].host_pred == results[i].label) {
            ++summary->host_correct;
        }
        if (results[i].pred != results[i].host_pred) {
            summary->mismatch_sample_indices[summary->mismatch_count++] = results[i].idx;
            ++summary->pred_mismatches_vs_host;
        }
        if (sample_max > summary->max_logit_abs) {
            summary->max_logit_abs = sample_max;
        }
        summary->mean_logit_abs += sample_max;
    }

    summary->mean_logit_abs /= (double)count;
}

static void write_predictions_jsonl(const char *path, const SampleResult *results, int count) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }
    for (int i = 0; i < count; ++i) {
        fprintf(
            fp,
            "{\"idx\": %d, \"label\": %d, \"pred\": %d, \"host_pred\": %d, \"logits\": [%.9g, %.9g], \"host_logits\": [%.9g, %.9g], \"logit_max_abs_diff\": %.9g, \"elapsed_ms\": %.3f}\n",
            results[i].idx,
            results[i].label,
            results[i].pred,
            results[i].host_pred,
            results[i].logits[0],
            results[i].logits[1],
            results[i].host_logits[0],
            results[i].host_logits[1],
            results[i].logit_max_abs_diff,
            results[i].elapsed_ms
        );
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
    double total_elapsed_ms
) {
    FILE *fp = fopen(path, "w");

    if (fp == NULL) {
        fatalf("fopen failed for %s: %s", path, strerror(errno));
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"num_samples\": %d,\n", count);
    fprintf(fp, "  \"seq_len\": %d,\n", seq_len);
    fprintf(fp, "  \"threads\": %d,\n", num_threads);
    fprintf(fp, "  \"accuracy\": %.9f,\n", (double)summary->correct / (double)count);
    fprintf(fp, "  \"host_int8_accuracy\": %.9f,\n", (double)summary->host_correct / (double)count);
    fprintf(fp, "  \"prediction_mismatches_vs_host_int8\": %d,\n", summary->pred_mismatches_vs_host);
    fprintf(fp, "  \"prediction_mismatch_sample_indices\": [");
    for (int i = 0; i < summary->mismatch_count; ++i) {
        fprintf(fp, "%s%d", i == 0 ? "" : ", ", summary->mismatch_sample_indices[i]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"mean_logit_max_abs_vs_host_int8\": %.9f,\n", summary->mean_logit_abs);
    fprintf(fp, "  \"max_logit_max_abs_vs_host_int8\": %.9f,\n", summary->max_logit_abs);
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
    join_path(paths->temp_dir, sizeof(paths->temp_dir), "/tmp/distilgpt2_c_vcpu", paths->run_name);
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
    snprintf(opts->output_dir, sizeof(opts->output_dir), "../outputs/c_vcpu");
    opts->limit = -1;
    opts->num_threads = DEFAULT_THREADS;
    opts->vcpu_offset = -1;
    opts->dump_debug_bin = false;

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
        } else if (strcmp(argv[i], "--limit") == 0) {
            if (i + 1 >= argc) {
                fatalf("missing value for --limit");
            }
            opts->limit = atoi(argv[++i]);
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
        } else if (strcmp(argv[i], "--dump-debug-bin") == 0) {
            opts->dump_debug_bin = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf(
                "Usage: %s [--generated-dir PATH] [--output-dir PATH] [--limit N] [--threads N] [--vcpu-offset N] [--dump-debug-bin]\n",
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
}

int main(int argc, char **argv) {
    ProgramOptions opts;
    DistilGpt2Model model;
    DebugStore debug_store;
#if !DISTILGPT2_CPU_POWER_ONLY
    ResultPaths result_paths;
    ExperimentSummary summary;
#endif
    SampleResult *results;
    WorkQueue queue;
    pthread_t *threads;
    WorkerArgs *worker_args;
    pthread_mutex_t log_mutex;
    DebugStore *debug_store_ptr = NULL;
    int total = 0;
    double start_ms = 0.0;
    double elapsed_ms = 0.0;
#if !DISTILGPT2_CPU_POWER_ONLY
    bool host_export_ok = false;
#endif

    parse_args(argc, argv, &opts);

    /* Match the rounding mode assumed during host-side quantized export. */
    if (fesetround(FE_TONEAREST) != 0) {
        fatalf("failed to set FE_TONEAREST");
    }

    init_model(&model, opts.generated_dir);
    total = model.eval_count;
    if (opts.limit > 0 && opts.limit < total) {
        total = opts.limit;
    }
    if (total <= 0) {
        fatalf("no evaluation samples available");
    }

    if (!DISTILGPT2_CPU_POWER_ONLY) {
        ensure_dir_recursive(opts.output_dir);
        init_debug_store(&debug_store, opts.generated_dir, opts.output_dir, opts.dump_debug_bin);
        debug_store_ptr = &debug_store;
    }

    results = (SampleResult *)xcalloc((size_t)total, sizeof(SampleResult));
    threads = (pthread_t *)xcalloc((size_t)opts.num_threads, sizeof(pthread_t));
    worker_args = (WorkerArgs *)xcalloc((size_t)opts.num_threads, sizeof(WorkerArgs));

    pthread_mutex_init(&queue.mutex, NULL);
    pthread_mutex_init(&log_mutex, NULL);
    queue.next_order = 0;
    queue.total = total;

    if (DISTILGPT2_CPU_POWER_ONLY) {
        printf("[config] generated_dir=%s samples=%d threads=%d seq_len=%d vcpu_offset=%d power_only=1\n",
               opts.generated_dir, total, opts.num_threads, model.seq_len, opts.vcpu_offset);
    } else {
        printf("[config] generated_dir=%s output_dir=%s samples=%d threads=%d seq_len=%d vcpu_offset=%d dump_debug_bin=%d power_only=0\n",
               opts.generated_dir, opts.output_dir, total, opts.num_threads, model.seq_len, opts.vcpu_offset, opts.dump_debug_bin ? 1 : 0);
    }
    fflush(stdout);

    start_ms = now_ms();
    for (int thread_idx = 0; thread_idx < opts.num_threads; ++thread_idx) {
        worker_args[thread_idx].model = &model;
        worker_args[thread_idx].queue = &queue;
        worker_args[thread_idx].results = results;
        worker_args[thread_idx].debug_store = debug_store_ptr;
        worker_args[thread_idx].log_mutex = &log_mutex;
        worker_args[thread_idx].thread_id = thread_idx;
        worker_args[thread_idx].vcpu_offset = opts.vcpu_offset;
        worker_args[thread_idx].pinned_cpu = -1;
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

    if (DISTILGPT2_CPU_POWER_ONLY) {
        int correct = 0;
        for (int i = 0; i < total; ++i) {
            if (results[i].pred == results[i].label) {
                ++correct;
            }
        }
        printf(
            "[done] cpu_power_only=1 samples=%d threads=%d accuracy=%.9f elapsed_ms=%.3f\n",
            total,
            opts.num_threads,
            (double)correct / (double)total,
            elapsed_ms
        );
        fflush(stdout);
        return 0;
    }

#if !DISTILGPT2_CPU_POWER_ONLY
    build_result_paths(&opts, &result_paths);
    ensure_dir_recursive(result_paths.temp_dir);

    compute_experiment_summary(&model, results, total, &summary);

    write_metrics_json(result_paths.temp_metrics, &summary, model.seq_len, total, opts.num_threads, elapsed_ms);
    write_predictions_jsonl(result_paths.temp_predictions, results, total);
    write_debug_summary_json(result_paths.temp_debug_summary, &debug_store);

    if (try_ensure_dir_recursive(result_paths.host_dir)) {
        bool copied_metrics = copy_file(result_paths.temp_metrics, result_paths.host_metrics);
        bool copied_predictions = copy_file(result_paths.temp_predictions, result_paths.host_predictions);
        bool copied_debug = copy_file(result_paths.temp_debug_summary, result_paths.host_debug_summary);
        host_export_ok = copied_metrics && copied_predictions && copied_debug;
    }

    if (host_export_ok) {
        printf(
            "[done] host_run_dir=%s mismatch=%d host_accuracy=%.9f vcpu_accuracy=%.9f metrics=%s predictions=%s debug_summary=%s tmp_backup=%s elapsed_ms=%.3f\n",
            result_paths.host_dir,
            summary.pred_mismatches_vs_host,
            (double)summary.host_correct / (double)total,
            (double)summary.correct / (double)total,
            result_paths.host_metrics,
            result_paths.host_predictions,
            result_paths.host_debug_summary,
            result_paths.temp_dir,
            elapsed_ms
        );
    } else {
        printf(
            "[done] host_export=failed tmp_run_dir=%s mismatch=%d host_accuracy=%.9f vcpu_accuracy=%.9f metrics=%s predictions=%s debug_summary=%s elapsed_ms=%.3f\n",
            result_paths.temp_dir,
            summary.pred_mismatches_vs_host,
            (double)summary.host_correct / (double)total,
            (double)summary.correct / (double)total,
            result_paths.temp_metrics,
            result_paths.temp_predictions,
            result_paths.temp_debug_summary,
            elapsed_ms
        );
    }
    if (summary.mismatch_count > 0) {
        printf("[summary] mismatch_sample_indices=");
        for (int i = 0; i < summary.mismatch_count; ++i) {
            printf("%s%d", i == 0 ? "[" : ", ", summary.mismatch_sample_indices[i]);
        }
        printf("]\n");
    } else {
        printf("[summary] mismatch_sample_indices=[]\n");
    }
    fflush(stdout);
    return 0;
#endif
}
