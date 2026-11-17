#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "gemmini_params.h"

#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define MAX_SHAPES 64

typedef struct {
    const char *name;
    size_t m;
    size_t n;
    size_t k;
    size_t calls_per_sample;
} LogicalGemm;

typedef struct {
    const char *name;
    const LogicalGemm *gemms;
    size_t gemm_count;
    size_t chunk_m;
    size_t chunk_n;
    size_t expected_logical_calls;
    size_t expected_api_calls;
    size_t expected_loop_ws_calls;
} Application;

typedef struct {
    size_t m;
    size_t n;
    size_t k;
    size_t count;
} ApiShapeCount;

typedef struct {
    size_t i;
    size_t j;
    size_t k;
} AutoTile;

typedef struct {
    size_t i;
    size_t j;
    size_t k;
    size_t pad_i;
    size_t pad_j;
    size_t pad_k;
    bool accumulate;
    bool write_output;
    size_t count;
} LoopWsCount;

static const LogicalGemm distilbert_gemms[] = {
    {"Q/K/V/attention_out", 128, 768, 768, 24},
    {"ffn_lin1", 128, 3072, 768, 6},
    {"ffn_lin2", 128, 768, 3072, 6},
    {"pre_classifier", 1, 768, 768, 1},
    {"classifier", 1, 2, 768, 1},
};

static const LogicalGemm distilgpt2_gemms[] = {
    {"qkv_lin", 128, 2304, 768, 6},
    {"attention_out", 128, 768, 768, 6},
    {"ffn_lin1", 128, 3072, 768, 6},
    {"ffn_lin2", 128, 768, 3072, 6},
};

static const Application applications[] = {
    {
        "DistilBERT",
        distilbert_gemms,
        ARRAY_SIZE(distilbert_gemms),
        16,
        144,
        38,
        2503,
        4231,
    },
    {
        "DistilRoBERTa",
        distilbert_gemms,
        ARRAY_SIZE(distilbert_gemms),
        16,
        144,
        38,
        2503,
        4231,
    },
    {
        "DistilGPT2",
        distilgpt2_gemms,
        ARRAY_SIZE(distilgpt2_gemms),
        16,
        144,
        24,
        2400,
        4128,
    },
};

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

static size_t floor_sqrt(size_t value) {
    size_t root = 0;

    while (root + 1 <= value / (root + 1)) {
        root++;
    }
    return root;
}

static size_t total_spad_rows(size_t tile_i, size_t tile_j, size_t tile_k) {
    return (tile_i * tile_k + tile_k * tile_j) * DIM;
}

static size_t total_acc_rows(size_t tile_i, size_t tile_j) {
    return tile_i * tile_j * DIM;
}

// Mirrors the WS double-buffered tile selection in gemmini.h:tiled_matmul_auto().
static AutoTile choose_auto_tile(size_t dim_i, size_t dim_j, size_t dim_k) {
    const size_t blocks_i = ceil_div(dim_i, DIM);
    const size_t blocks_j = ceil_div(dim_j, DIM);
    const size_t blocks_k = ceil_div(dim_k, DIM);
    const size_t max_spad_rows = BANK_NUM * BANK_ROWS / 2;
    const size_t max_acc_rows = ACC_ROWS / 2;
    const size_t db_partition_rows = (BANK_NUM * BANK_ROWS / 2) / 2;
    const size_t db_mats_in_partition = db_partition_rows / DIM;
    const size_t db_mats_in_acc = (ACC_ROWS / 2) / DIM;
    const size_t db_max_tile_i_j = floor_sqrt(db_mats_in_acc);
    const size_t db_max_tile_k = db_mats_in_partition / db_max_tile_i_j;
    AutoTile tile = {
        min_size(blocks_i, db_max_tile_i_j),
        min_size(blocks_j, db_max_tile_i_j),
        min_size(blocks_k, db_max_tile_k),
    };

    assert(dim_i > 0 && dim_j > 0 && dim_k > 0);
    assert(tile.i > 0 && tile.j > 0 && tile.k > 0);

    for (;;) {
        bool increased = false;

        if (total_spad_rows(tile.i, tile.j + 1, tile.k) <= max_spad_rows &&
            total_acc_rows(tile.i, tile.j + 1) <= max_acc_rows &&
            tile.j + 1 <= blocks_j) {
            tile.j++;
            increased = true;
        }
        if (total_spad_rows(tile.i + 1, tile.j, tile.k) <= max_spad_rows &&
            total_acc_rows(tile.i + 1, tile.j) <= max_acc_rows &&
            tile.i + 1 <= blocks_i) {
            tile.i++;
            increased = true;
        }
        if (total_spad_rows(tile.i, tile.j, tile.k + 1) <= max_spad_rows &&
            tile.k + 1 <= blocks_k) {
            tile.k++;
            increased = true;
        }
        if (!increased) {
            return tile;
        }
    }
}

static void add_api_shape(
    ApiShapeCount *shapes,
    size_t *shape_count,
    size_t m,
    size_t n,
    size_t k,
    size_t count
) {
    for (size_t index = 0; index < *shape_count; index++) {
        ApiShapeCount *shape = &shapes[index];

        if (shape->m == m && shape->n == n && shape->k == k) {
            shape->count += count;
            return;
        }
    }

    assert(*shape_count < MAX_SHAPES);
    shapes[*shape_count] = (ApiShapeCount){m, n, k, count};
    (*shape_count)++;
}

static size_t chunk_logical_gemm(
    const LogicalGemm *gemm,
    size_t chunk_m,
    size_t chunk_n,
    ApiShapeCount *chunks
) {
    size_t chunk_count = 0;

    for (size_t m_base = 0; m_base < gemm->m; m_base += chunk_m) {
        const size_t m = min_size(chunk_m, gemm->m - m_base);

        for (size_t n_base = 0; n_base < gemm->n; n_base += chunk_n) {
            const size_t n = min_size(chunk_n, gemm->n - n_base);
            add_api_shape(chunks, &chunk_count, m, n, gemm->k, 1);
        }
    }

    return chunk_count;
}

static bool same_loop_ws(const LoopWsCount *a, const LoopWsCount *b) {
    return a->i == b->i &&
           a->j == b->j &&
           a->k == b->k &&
           a->pad_i == b->pad_i &&
           a->pad_j == b->pad_j &&
           a->pad_k == b->pad_k &&
           a->accumulate == b->accumulate &&
           a->write_output == b->write_output;
}

static void add_loop_ws(
    LoopWsCount *commands,
    size_t *command_count,
    LoopWsCount command
) {
    for (size_t index = 0; index < *command_count; index++) {
        if (same_loop_ws(&commands[index], &command)) {
            commands[index].count += command.count;
            return;
        }
    }

    assert(*command_count < MAX_SHAPES);
    commands[*command_count] = command;
    (*command_count)++;
}

static size_t expand_api_to_loop_ws(
    size_t dim_i,
    size_t dim_j,
    size_t dim_k,
    LoopWsCount *commands
) {
    const AutoTile tile = choose_auto_tile(dim_i, dim_j, dim_k);
    const size_t blocks_i = ceil_div(dim_i, DIM);
    const size_t blocks_j = ceil_div(dim_j, DIM);
    const size_t blocks_k = ceil_div(dim_k, DIM);
    const size_t outer_i = ceil_div(blocks_i, tile.i);
    const size_t outer_j = ceil_div(blocks_j, tile.j);
    const size_t outer_k = ceil_div(blocks_k, tile.k);
    const size_t last_i = blocks_i % tile.i == 0 ? tile.i : blocks_i % tile.i;
    const size_t last_j = blocks_j % tile.j == 0 ? tile.j : blocks_j % tile.j;
    const size_t last_k = blocks_k % tile.k == 0 ? tile.k : blocks_k % tile.k;
    const size_t final_pad_i = blocks_i * DIM - dim_i;
    const size_t final_pad_j = blocks_j * DIM - dim_j;
    const size_t final_pad_k = blocks_k * DIM - dim_k;
    size_t command_count = 0;

    for (size_t i0 = 0; i0 < outer_i; i0++) {
        for (size_t j0 = 0; j0 < outer_j; j0++) {
            for (size_t k0 = 0; k0 < outer_k; k0++) {
                LoopWsCount command = {
                    .i = i0 + 1 == outer_i ? last_i : tile.i,
                    .j = j0 + 1 == outer_j ? last_j : tile.j,
                    .k = k0 + 1 == outer_k ? last_k : tile.k,
                    .pad_i = i0 + 1 == outer_i ? final_pad_i : 0,
                    .pad_j = j0 + 1 == outer_j ? final_pad_j : 0,
                    .pad_k = k0 + 1 == outer_k ? final_pad_k : 0,
                    .accumulate = k0 != 0,
                    .write_output = k0 + 1 == outer_k,
                    .count = 1,
                };
                add_loop_ws(commands, &command_count, command);
            }
        }
    }

    return command_count;
}

static size_t sum_api_counts(const ApiShapeCount *shapes, size_t shape_count) {
    size_t total = 0;

    for (size_t index = 0; index < shape_count; index++) {
        total += shapes[index].count;
    }
    return total;
}

static size_t sum_loop_ws_counts(const LoopWsCount *commands, size_t command_count) {
    size_t total = 0;

    for (size_t index = 0; index < command_count; index++) {
        total += commands[index].count;
    }
    return total;
}

static void self_check(void) {
    LoopWsCount commands[MAX_SHAPES];
    AutoTile tile;

    assert(DIM == 16);
    assert(BANK_NUM == 4);
    assert(BANK_ROWS == 4096);
    assert(ACC_ROWS == 1024);

    tile = choose_auto_tile(16, 144, 768);
    assert(tile.i == 1 && tile.j == 9 && tile.k == 48);
    assert(sum_loop_ws_counts(commands, expand_api_to_loop_ws(16, 144, 768, commands)) == 1);

    tile = choose_auto_tile(16, 144, 3072);
    assert(tile.i == 1 && tile.j == 8 && tile.k == 56);
    assert(sum_loop_ws_counts(commands, expand_api_to_loop_ws(16, 144, 3072, commands)) == 8);

    tile = choose_auto_tile(16, 48, 3072);
    assert(tile.i == 1 && tile.j == 3 && tile.k == 128);
    assert(sum_loop_ws_counts(commands, expand_api_to_loop_ws(16, 48, 3072, commands)) == 2);
}

static void print_application(const Application *application) {
    ApiShapeCount application_api[MAX_SHAPES] = {0};
    LoopWsCount application_loop_ws[MAX_SHAPES] = {0};
    size_t application_api_shape_count = 0;
    size_t application_loop_ws_shape_count = 0;
    size_t logical_calls = 0;

    printf("\n=== %s ===\n", application->name);
    printf(
        "APPLICATION_CHUNK M<=%zu N<=%zu K=unchanged\n",
        application->chunk_m,
        application->chunk_n
    );
    printf(
        "LOOP_WS_BOUNDS I=M tile blocks, J=N tile blocks, "
        "K=reduction tile blocks; one block=DIM=%d\n",
        DIM
    );
    printf(
        "LOGICAL_SIZE M=I*DIM-pad_I, N=J*DIM-pad_J, K=K_tiles*DIM-pad_K\n"
    );
    printf("ORIGINAL_TO_API\n");

    for (size_t gemm_index = 0; gemm_index < application->gemm_count; gemm_index++) {
        const LogicalGemm *gemm = &application->gemms[gemm_index];
        ApiShapeCount chunks[MAX_SHAPES] = {0};
        const size_t chunk_shape_count =
            chunk_logical_gemm(gemm, application->chunk_m, application->chunk_n, chunks);

        logical_calls += gemm->calls_per_sample;
        printf(
            "  ORIGINAL op=%s M=%zu N=%zu K=%zu calls_per_sample=%zu\n",
            gemm->name,
            gemm->m,
            gemm->n,
            gemm->k,
            gemm->calls_per_sample
        );

        for (size_t chunk_index = 0; chunk_index < chunk_shape_count; chunk_index++) {
            const ApiShapeCount *chunk = &chunks[chunk_index];
            const size_t total_count = chunk->count * gemm->calls_per_sample;

            printf(
                "    API M=%zu N=%zu K=%zu per_original=%zu total_per_sample=%zu\n",
                chunk->m,
                chunk->n,
                chunk->k,
                chunk->count,
                total_count
            );
            add_api_shape(
                application_api,
                &application_api_shape_count,
                chunk->m,
                chunk->n,
                chunk->k,
                total_count
            );
        }
    }

    printf("API_TO_LOOP_WS\n");
    for (size_t api_index = 0; api_index < application_api_shape_count; api_index++) {
        const ApiShapeCount *api = &application_api[api_index];
        const AutoTile tile = choose_auto_tile(api->m, api->n, api->k);
        LoopWsCount commands[MAX_SHAPES] = {0};
        const size_t command_shape_count =
            expand_api_to_loop_ws(api->m, api->n, api->k, commands);

        printf(
            "  API M=%zu N=%zu K=%zu calls_per_sample=%zu auto_tile=(%zu,%zu,%zu)\n",
            api->m,
            api->n,
            api->k,
            api->count,
            tile.i,
            tile.j,
            tile.k
        );

        for (size_t command_index = 0; command_index < command_shape_count; command_index++) {
            LoopWsCount command = commands[command_index];
            const size_t total_count = command.count * api->count;

            printf(
                "    LOOP_WS bounds=(I=%zu,J=%zu,K=%zu) size=(%zu,%zu,%zu) "
                "pad=(%zu,%zu,%zu) accumulate=%d write_output=%d "
                "per_api=%zu total_per_sample=%zu\n",
                command.i,
                command.j,
                command.k,
                command.i * DIM - command.pad_i,
                command.j * DIM - command.pad_j,
                command.k * DIM - command.pad_k,
                command.pad_i,
                command.pad_j,
                command.pad_k,
                command.accumulate,
                command.write_output,
                command.count,
                total_count
            );

            command.count = total_count;
            add_loop_ws(
                application_loop_ws,
                &application_loop_ws_shape_count,
                command
            );
        }
    }

    const size_t api_calls = sum_api_counts(application_api, application_api_shape_count);
    const size_t loop_ws_calls =
        sum_loop_ws_counts(application_loop_ws, application_loop_ws_shape_count);

    assert(logical_calls == application->expected_logical_calls);
    assert(api_calls == application->expected_api_calls);
    assert(loop_ws_calls == application->expected_loop_ws_calls);

    printf(
        "SUMMARY logical_gemm_calls=%zu api_calls=%zu loop_ws_calls=%zu\n",
        logical_calls,
        api_calls,
        loop_ws_calls
    );
}

int main(void) {
    self_check();

    printf(
        "GEMMINI_PARAMS DIM=%d BANK_NUM=%d BANK_ROWS=%d ACC_ROWS=%d dataflow=WS\n",
        DIM,
        BANK_NUM,
        BANK_ROWS,
        ACC_ROWS
    );
    for (size_t index = 0; index < ARRAY_SIZE(applications); index++) {
        print_application(&applications[index]);
    }
    printf("\nSELF_CHECK PASS\n");
    return 0;
}
