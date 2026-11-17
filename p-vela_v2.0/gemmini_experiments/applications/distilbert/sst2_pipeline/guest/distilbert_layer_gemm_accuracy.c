#define DISTILBERT_NO_MAIN 1
#define DISTILBERT_USE_GEMMINI_BACKEND 1

#include "distilbert_gemmini_infer.c"

typedef struct {
    const char *name;
    const LinearLayer *layer;
    int rows;
} LayerCase;

static void fill_deterministic_a(elem_t *a, int rows, int cols, uint32_t seed) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            uint32_t x = seed ^ (uint32_t)(row * 1103515245u) ^ (uint32_t)(col * 2654435761u);
            x ^= x >> 16;
            a[(size_t)row * (size_t)cols + (size_t)col] = (elem_t)((int)(x % 255u) - 127);
        }
    }
}

static void cpu_linear_acc_full(const LinearLayer *layer, const elem_t *a, int rows, acc_t *out) {
    for (int row = 0; row < rows; ++row) {
        for (int out_idx = 0; out_idx < layer->out_features; ++out_idx) {
            const int8_t *weight_row =
                layer->weight_q + (size_t)out_idx * (size_t)layer->in_features;
            acc_t acc = 0;
            for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
                acc += (acc_t)a[(size_t)row * (size_t)layer->in_features + (size_t)in_idx] *
                       (acc_t)weight_row[in_idx];
            }
            out[(size_t)row * (size_t)layer->out_features + (size_t)out_idx] = acc;
        }
    }
}

static void quantize_float_input_to_a(const LinearLayer *layer, const float *input, int rows, elem_t *a) {
    for (int row = 0; row < rows; ++row) {
        for (int in_idx = 0; in_idx < layer->in_features; ++in_idx) {
            const float value = input[(size_t)row * (size_t)layer->in_features + (size_t)in_idx];
            a[(size_t)row * (size_t)layer->in_features + (size_t)in_idx] =
                (elem_t)quantize_ties_to_even(value, layer->input_scale);
        }
    }
}

static void cpu_linear_forward_float(
    const LinearLayer *layer,
    const float *input,
    int rows,
    float *output,
    elem_t *a,
    acc_t *acc
) {
    quantize_float_input_to_a(layer, input, rows, a);
    cpu_linear_acc_full(layer, a, rows, acc);
    for (int row = 0; row < rows; ++row) {
        for (int out_idx = 0; out_idx < layer->out_features; ++out_idx) {
            const size_t idx = (size_t)row * (size_t)layer->out_features + (size_t)out_idx;
            volatile float scaled = (float)acc[idx] * layer->acc_scale[out_idx];
            volatile float y = scaled + layer->bias_fp32[out_idx];
            int32_t y_q = quantize_ties_to_even(y, layer->output_scale);
            output[idx] = (float)y_q * layer->output_scale;
        }
    }
}

static void copy_a_chunk(
    elem_t *dst,
    const elem_t *src,
    int row_base,
    int rows,
    int cols
) {
    for (int row = 0; row < rows; ++row) {
        memcpy(
            dst + (size_t)row * (size_t)cols,
            src + (size_t)(row_base + row) * (size_t)cols,
            (size_t)cols * sizeof(elem_t)
        );
    }
}

static void run_gemmini_chunked_acc(
    const LinearLayer *layer,
    const elem_t *a,
    int rows,
    Workspace *ws,
    acc_t *out
) {
    memset(out, 0, (size_t)rows * (size_t)layer->out_features * sizeof(acc_t));
    for (int row_base = 0; row_base < rows; row_base += GEMMINI_SAFE_MAX_I) {
        const int chunk_rows = rows - row_base < GEMMINI_SAFE_MAX_I
            ? rows - row_base
            : GEMMINI_SAFE_MAX_I;
        copy_a_chunk(ws->gemm_a, a, row_base, chunk_rows, layer->in_features);
        for (int out_base = 0; out_base < layer->out_features; out_base += GEMMINI_SAFE_MAX_J) {
            const int chunk_cols = layer->out_features - out_base < GEMMINI_SAFE_MAX_J
                ? layer->out_features - out_base
                : GEMMINI_SAFE_MAX_J;
            gemmini_matmul_acc_chunk(layer, chunk_rows, out_base, chunk_cols, ws, false);
            for (int row = 0; row < chunk_rows; ++row) {
                memcpy(
                    out + (size_t)(row_base + row) * (size_t)layer->out_features + (size_t)out_base,
                    ws->gemm_acc + (size_t)row * (size_t)chunk_cols,
                    (size_t)chunk_cols * sizeof(acc_t)
                );
            }
        }
    }
}

static void run_gemmini_direct_acc(
    const LinearLayer *layer,
    const elem_t *a,
    int rows,
    Workspace *ws,
    acc_t *out
) {
    memcpy(ws->gemm_a, a, (size_t)rows * (size_t)layer->in_features * sizeof(elem_t));
    gemmini_matmul_acc_chunk(layer, rows, 0, layer->out_features, ws, false);
    memcpy(out, ws->gemm_acc, (size_t)rows * (size_t)layer->out_features * sizeof(acc_t));
}

static void compare_and_print(
    const char *name,
    const char *mode,
    const LinearLayer *layer,
    int rows,
    const acc_t *ref,
    const acc_t *got,
    double elapsed_ms
) {
    const size_t total = (size_t)rows * (size_t)layer->out_features;
    size_t mismatches = 0;
    int max_abs_diff = 0;
    int first_row = -1;
    int first_col = -1;
    acc_t first_ref = 0;
    acc_t first_got = 0;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < layer->out_features; ++col) {
            const size_t idx = (size_t)row * (size_t)layer->out_features + (size_t)col;
            const int diff = abs((int)(ref[idx] - got[idx]));
            if (diff != 0) {
                if (mismatches == 0) {
                    first_row = row;
                    first_col = col;
                    first_ref = ref[idx];
                    first_got = got[idx];
                }
                mismatches++;
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                }
            }
        }
    }

    printf(
        "layer=%s mode=%s shape=%dx%dx%d mismatches=%zu total=%zu max_abs_diff=%d first=(row=%d,col=%d,ref=%d,got=%d) elapsed_ms=%.3f\n",
        name,
        mode,
        rows,
        layer->out_features,
        layer->in_features,
        mismatches,
        total,
        max_abs_diff,
        first_row,
        first_col,
        first_ref,
        first_got,
        elapsed_ms
    );
    fflush(stdout);
}

static void run_int8_case(
    const char *name,
    const LinearLayer *layer,
    int rows,
    elem_t *a,
    acc_t *ref,
    acc_t *got,
    Workspace *ws,
    uint32_t seed
) {
    const size_t out_count = (size_t)rows * (size_t)layer->out_features;
    double start_ms = 0.0;

    fill_deterministic_a(a, rows, layer->in_features, seed);
    memset(ref, 0, out_count * sizeof(acc_t));
    memset(got, 0, out_count * sizeof(acc_t));
    cpu_linear_acc_full(layer, a, rows, ref);

    start_ms = now_ms();
    run_gemmini_chunked_acc(layer, a, rows, ws, got);
    compare_and_print(name, "chunked_stress", layer, rows, ref, got, now_ms() - start_ms);

    memset(got, 0, out_count * sizeof(acc_t));
    start_ms = now_ms();
    run_gemmini_direct_acc(layer, a, rows, ws, got);
    compare_and_print(name, "direct_stress", layer, rows, ref, got, now_ms() - start_ms);
}

static void run_float_input_case(
    const char *name,
    const LinearLayer *layer,
    const float *input,
    int rows,
    elem_t *a,
    acc_t *ref,
    acc_t *got,
    Workspace *ws
) {
    const size_t out_count = (size_t)rows * (size_t)layer->out_features;
    double start_ms = 0.0;

    quantize_float_input_to_a(layer, input, rows, a);
    memset(ref, 0, out_count * sizeof(acc_t));
    memset(got, 0, out_count * sizeof(acc_t));
    cpu_linear_acc_full(layer, a, rows, ref);

    start_ms = now_ms();
    run_gemmini_chunked_acc(layer, a, rows, ws, got);
    compare_and_print(name, "chunked_actual", layer, rows, ref, got, now_ms() - start_ms);

    memset(got, 0, out_count * sizeof(acc_t));
    start_ms = now_ms();
    run_gemmini_direct_acc(layer, a, rows, ws, got);
    compare_and_print(name, "direct_actual", layer, rows, ref, got, now_ms() - start_ms);
}

static int add_layer_case(
    LayerCase *cases,
    int count,
    const char *name,
    const LinearLayer *layer,
    int rows
) {
    cases[count].name = name;
    cases[count].layer = layer;
    cases[count].rows = rows;
    return count + 1;
}

int main(int argc, char **argv) {
    const char *generated_dir = "generated";
    int vcpu = 0;
    DistilBertModel model;
    Workspace ws;
    LayerCase cases[16];
    int case_count = 0;
    elem_t *a = NULL;
    acc_t *ref = NULL;
    acc_t *got = NULL;
    size_t max_a = 0;
    size_t max_out = 0;
    const int sample_order = 0;
    const int32_t *input_ids = NULL;
    const int32_t *attention_mask = NULL;
    TransformerLayer *layer0 = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--generated-dir") == 0 && i + 1 < argc) {
            generated_dir = argv[++i];
        } else if (strcmp(argv[i], "--vcpu") == 0 && i + 1 < argc) {
            vcpu = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--generated-dir PATH] [--vcpu N]\n", argv[0]);
            return 2;
        }
    }

    pin_thread_to_vcpu(vcpu);
    if (fesetround(FE_TONEAREST) != 0) {
        fatalf("failed to set FE_TONEAREST");
    }
    init_model(&model, generated_dir);
    init_workspace(&model, &ws);
    prepare_gemmini_state();

    case_count = add_layer_case(cases, case_count, "layer0.q_lin", &model.layers[0].q_lin, model.seq_len);
    case_count = add_layer_case(cases, case_count, "layer0.out_lin", &model.layers[0].out_lin, model.seq_len);
    case_count = add_layer_case(cases, case_count, "layer0.ffn_lin1", &model.layers[0].ffn_lin1, model.seq_len);
    case_count = add_layer_case(cases, case_count, "layer0.ffn_lin2", &model.layers[0].ffn_lin2, model.seq_len);
    case_count = add_layer_case(cases, case_count, "pre_classifier", &model.pre_classifier, 1);
    case_count = add_layer_case(cases, case_count, "classifier", &model.classifier, 1);

    for (int i = 0; i < case_count; ++i) {
        const LinearLayer *layer = cases[i].layer;
        const size_t a_count = (size_t)cases[i].rows * (size_t)layer->in_features;
        const size_t out_count = (size_t)cases[i].rows * (size_t)layer->out_features;
        if (a_count > max_a) {
            max_a = a_count;
        }
        if (out_count > max_out) {
            max_out = out_count;
        }
    }
    a = (elem_t *)xaligned_alloc(64, max_a * sizeof(elem_t));
    ref = (acc_t *)xaligned_alloc(64, max_out * sizeof(acc_t));
    got = (acc_t *)xaligned_alloc(64, max_out * sizeof(acc_t));

    printf("distilbert_layer_gemm_accuracy cases=%d chunk_i=%d chunk_j=%d\n", case_count, GEMMINI_SAFE_MAX_I, GEMMINI_SAFE_MAX_J);
    input_ids = model.input_ids + (size_t)sample_order * (size_t)model.seq_len;
    attention_mask = model.attention_mask + (size_t)sample_order * (size_t)model.seq_len;
    layer0 = &model.layers[0];

    printf("section=actual_sample0_layer0\n");
    embeddings_forward(&model, input_ids, ws.hidden_a);
    run_float_input_case("actual.layer0.q_lin", &layer0->q_lin, ws.hidden_a, model.seq_len, a, ref, got, &ws);
    cpu_linear_forward_float(&layer0->q_lin, ws.hidden_a, model.seq_len, ws.q_buf, a, ref);
    cpu_linear_forward_float(&layer0->k_lin, ws.hidden_a, model.seq_len, ws.k_buf, a, ref);
    cpu_linear_forward_float(&layer0->v_lin, ws.hidden_a, model.seq_len, ws.v_buf, a, ref);

    attention_forward(&model, ws.q_buf, ws.k_buf, ws.v_buf, attention_mask, &ws);
    run_float_input_case("actual.layer0.out_lin", &layer0->out_lin, ws.context_buf, model.seq_len, a, ref, got, &ws);
    cpu_linear_forward_float(&layer0->out_lin, ws.context_buf, model.seq_len, ws.hidden_b, a, ref);
    add_inplace(ws.hidden_b, ws.hidden_a, (size_t)model.seq_len * (size_t)model.dim);
    layer_norm_inplace(ws.hidden_b, model.seq_len, &layer0->sa_layer_norm);

    run_float_input_case("actual.layer0.ffn_lin1", &layer0->ffn_lin1, ws.hidden_b, model.seq_len, a, ref, got, &ws);
    cpu_linear_forward_float(&layer0->ffn_lin1, ws.hidden_b, model.seq_len, ws.ffn_buf, a, ref);
    gelu_inplace(ws.ffn_buf, (size_t)model.seq_len * (size_t)model.hidden_dim);

    run_float_input_case("actual.layer0.ffn_lin2", &layer0->ffn_lin2, ws.ffn_buf, model.seq_len, a, ref, got, &ws);

    printf("section=deterministic_int8_shape_stress\n");
    for (int i = 0; i < case_count; ++i) {
        const LayerCase *tc = &cases[i];
        const LinearLayer *layer = tc->layer;
        const size_t a_count = (size_t)tc->rows * (size_t)layer->in_features;

        run_int8_case(
            tc->name,
            layer,
            tc->rows,
            a,
            ref,
            got,
            &ws,
            (uint32_t)(0xC0FFEEu + (uint32_t)i * 7919u)
        );
        memset(a, 0, a_count * sizeof(elem_t));
    }

    free(a);
    free(ref);
    free(got);
    return 0;
}
