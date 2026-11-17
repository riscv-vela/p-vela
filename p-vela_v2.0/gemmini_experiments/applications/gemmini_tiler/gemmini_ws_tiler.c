#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../q-vela/qemu/test-code/gemmini_test_code/include/gemmini_params.h"

typedef struct {
    size_t dim_m;
    size_t dim_n;
    size_t dim_k;
    size_t padded_m;
    size_t padded_n;
    size_t padded_k;
    size_t tile_m;
    size_t tile_n;
    size_t tile_k;
    size_t outer_m;
    size_t outer_n;
    size_t outer_k;
    size_t last_m;
    size_t last_n;
    size_t last_k;
    size_t pad_m;
    size_t pad_n;
    size_t pad_k;
} WsPlan;

typedef struct {
    size_t m_tiles;
    size_t n_tiles;
    size_t k_tiles;
    size_t pad_m;
    size_t pad_n;
    size_t pad_k;
} LoopWsCommand;

static size_t ceil_div(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

static size_t floor_sqrt(size_t value) {
    size_t root = 0;
    while (root + 1 <= value / (root + 1)) {
        root++;
    }
    return root;
}

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static size_t total_spad_rows(size_t tile_m, size_t tile_n, size_t tile_k) {
    return (tile_m * tile_k + tile_k * tile_n) * DIM;
}

static size_t total_acc_rows(size_t tile_m, size_t tile_n) {
    return tile_m * tile_n * DIM;
}

static WsPlan make_ws_plan(size_t dim_m, size_t dim_n, size_t dim_k) {
    const size_t max_spad_rows = BANK_NUM * BANK_ROWS / 2;
    const size_t max_acc_rows = ACC_ROWS / 2;
    const size_t db_partition_rows = (BANK_NUM * BANK_ROWS / 2) / 2;
    const size_t db_mats_in_partition = db_partition_rows / DIM;
    const size_t db_mats_in_acc = (ACC_ROWS / 2) / DIM;
    const size_t db_max_tile_mn = floor_sqrt(db_mats_in_acc);
    const size_t db_max_tile_k = db_mats_in_partition / db_max_tile_mn;

    WsPlan plan = {
        .dim_m = dim_m,
        .dim_n = dim_n,
        .dim_k = dim_k,
        .padded_m = ceil_div(dim_m, DIM) * DIM,
        .padded_n = ceil_div(dim_n, DIM) * DIM,
        .padded_k = ceil_div(dim_k, DIM) * DIM,
    };

    const size_t m_blocks = plan.padded_m / DIM;
    const size_t n_blocks = plan.padded_n / DIM;
    const size_t k_blocks = plan.padded_k / DIM;

    plan.tile_m = min_size(m_blocks, db_max_tile_mn);
    plan.tile_n = min_size(n_blocks, db_max_tile_mn);
    plan.tile_k = min_size(k_blocks, db_max_tile_k);

    while (true) {
        bool increased = false;

        if (plan.tile_n < n_blocks &&
            total_spad_rows(plan.tile_m, plan.tile_n + 1, plan.tile_k) <= max_spad_rows &&
            total_acc_rows(plan.tile_m, plan.tile_n + 1) <= max_acc_rows) {
            plan.tile_n++;
            increased = true;
        }

        if (plan.tile_m < m_blocks &&
            total_spad_rows(plan.tile_m + 1, plan.tile_n, plan.tile_k) <= max_spad_rows &&
            total_acc_rows(plan.tile_m + 1, plan.tile_n) <= max_acc_rows) {
            plan.tile_m++;
            increased = true;
        }

        if (plan.tile_k < k_blocks &&
            total_spad_rows(plan.tile_m, plan.tile_n, plan.tile_k + 1) <= max_spad_rows) {
            plan.tile_k++;
            increased = true;
        }

        if (!increased) {
            break;
        }
    }

    plan.outer_m = ceil_div(m_blocks, plan.tile_m);
    plan.outer_n = ceil_div(n_blocks, plan.tile_n);
    plan.outer_k = ceil_div(k_blocks, plan.tile_k);
    plan.last_m = m_blocks % plan.tile_m == 0 ? plan.tile_m : m_blocks % plan.tile_m;
    plan.last_n = n_blocks % plan.tile_n == 0 ? plan.tile_n : n_blocks % plan.tile_n;
    plan.last_k = k_blocks % plan.tile_k == 0 ? plan.tile_k : k_blocks % plan.tile_k;
    plan.pad_m = plan.padded_m - dim_m;
    plan.pad_n = plan.padded_n - dim_n;
    plan.pad_k = plan.padded_k - dim_k;

    return plan;
}

static LoopWsCommand command_at(const WsPlan *plan, size_t m0, size_t n0, size_t k0) {
    return (LoopWsCommand) {
        .m_tiles = m0 + 1 == plan->outer_m ? plan->last_m : plan->tile_m,
        .n_tiles = n0 + 1 == plan->outer_n ? plan->last_n : plan->tile_n,
        .k_tiles = k0 + 1 == plan->outer_k ? plan->last_k : plan->tile_k,
        .pad_m = m0 + 1 == plan->outer_m ? plan->pad_m : 0,
        .pad_n = n0 + 1 == plan->outer_n ? plan->pad_n : 0,
        .pad_k = k0 + 1 == plan->outer_k ? plan->pad_k : 0,
    };
}

static size_t command_count(const WsPlan *plan) {
    return plan->outer_m * plan->outer_n * plan->outer_k;
}

static void print_plan(const WsPlan *plan) {
    const size_t count = command_count(plan);
    size_t index = 0;

    printf("API M=%zu N=%zu K=%zu\n", plan->dim_m, plan->dim_n, plan->dim_k);
    printf("AUTO_TILE M_TILES=%zu N_TILES=%zu K_TILES=%zu "
           "OUTER_M=%zu OUTER_N=%zu OUTER_K=%zu LOOP_WS_COUNT=%zu\n",
           plan->tile_m, plan->tile_n, plan->tile_k,
           plan->outer_m, plan->outer_n, plan->outer_k, count);

    for (size_t m0 = 0; m0 < plan->outer_m; m0++) {
        for (size_t n0 = 0; n0 < plan->outer_n; n0++) {
            for (size_t k0 = 0; k0 < plan->outer_k; k0++) {
                const LoopWsCommand cmd = command_at(plan, m0, n0, k0);
                const size_t logical_m = cmd.m_tiles * DIM - cmd.pad_m;
                const size_t logical_n = cmd.n_tiles * DIM - cmd.pad_n;
                const size_t logical_k = cmd.k_tiles * DIM - cmd.pad_k;

                index++;
                printf("LOOP_WS %zu/%zu M_TILES=%zu N_TILES=%zu K_TILES=%zu "
                       "PAD_M=%zu PAD_N=%zu PAD_K=%zu M=%zu N=%zu K=%zu "
                       "K_INDEX=%zu/%zu ACCUMULATE=%d FINAL_K=%d\n",
                       index, count,
                       cmd.m_tiles, cmd.n_tiles, cmd.k_tiles,
                       cmd.pad_m, cmd.pad_n, cmd.pad_k,
                       logical_m, logical_n, logical_k,
                       k0 + 1, plan->outer_k,
                       k0 != 0, k0 + 1 == plan->outer_k);
            }
        }
    }
}

static bool expect_plan(size_t m, size_t n, size_t k,
                        size_t tile_m, size_t tile_n, size_t tile_k,
                        size_t outer_m, size_t outer_n, size_t outer_k,
                        size_t pad_m, size_t pad_n, size_t pad_k) {
    const WsPlan plan = make_ws_plan(m, n, k);
    const bool pass =
        plan.tile_m == tile_m && plan.tile_n == tile_n && plan.tile_k == tile_k &&
        plan.outer_m == outer_m && plan.outer_n == outer_n && plan.outer_k == outer_k &&
        plan.pad_m == pad_m && plan.pad_n == pad_n && plan.pad_k == pad_k;

    if (!pass) {
        fprintf(stderr,
                "FAIL M=%zu N=%zu K=%zu: tile=%zu,%zu,%zu outer=%zu,%zu,%zu "
                "pad=%zu,%zu,%zu\n",
                m, n, k,
                plan.tile_m, plan.tile_n, plan.tile_k,
                plan.outer_m, plan.outer_n, plan.outer_k,
                plan.pad_m, plan.pad_n, plan.pad_k);
    }
    return pass;
}

static int self_test(void) {
    bool pass = true;

    pass &= expect_plan(16, 144, 768, 1, 9, 48, 1, 1, 1, 0, 0, 0);
    pass &= expect_plan(16, 48, 768, 1, 3, 48, 1, 1, 1, 0, 0, 0);
    pass &= expect_plan(1, 144, 768, 1, 9, 48, 1, 1, 1, 15, 0, 0);
    pass &= expect_plan(1, 48, 768, 1, 3, 48, 1, 1, 1, 15, 0, 0);
    pass &= expect_plan(1, 2, 768, 1, 1, 48, 1, 1, 1, 15, 14, 0);
    pass &= expect_plan(16, 144, 3072, 1, 8, 56, 1, 2, 4, 0, 0, 0);
    pass &= expect_plan(16, 48, 3072, 1, 3, 128, 1, 1, 2, 0, 0, 0);

    if (!pass) {
        return 1;
    }

    puts("PASS: all Transformer API shapes match expected WS tiling");
    return 0;
}

static bool parse_dimension(const char *text, size_t *value) {
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > (unsigned long long)SIZE_MAX) {
        return false;
    }

    *value = (size_t)parsed;
    return true;
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --self-test\n"
            "  %s M N K [M N K ...]\n",
            program, program);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }

    if (argc < 4 || (argc - 1) % 3 != 0) {
        print_usage(argv[0]);
        return 2;
    }

    printf("CONFIG DIM=%d BANK_NUM=%d BANK_ROWS=%d ACC_ROWS=%d DATAFLOW=WS D=NULL\n",
           DIM, BANK_NUM, BANK_ROWS, ACC_ROWS);

    for (int arg = 1; arg < argc; arg += 3) {
        size_t m;
        size_t n;
        size_t k;

        if (!parse_dimension(argv[arg], &m) ||
            !parse_dimension(argv[arg + 1], &n) ||
            !parse_dimension(argv[arg + 2], &k)) {
            fprintf(stderr, "Invalid positive M/N/K near argument %d\n", arg);
            return 2;
        }

        if (arg != 1) {
            putchar('\n');
        }
        const WsPlan plan = make_ws_plan(m, n, k);
        print_plan(&plan);
    }

    return 0;
}
