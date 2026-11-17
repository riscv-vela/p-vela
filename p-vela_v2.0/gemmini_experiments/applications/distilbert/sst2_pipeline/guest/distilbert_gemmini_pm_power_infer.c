/* distilbert_gemmini_power_infer + power management SW (q-vela power_mgr.h).
 * Same inference code; the PM is linked in and told where NPU tasks begin and
 * end. Whether it gates is decided at run time (PM_POLICY in /etc/rvss_pm.conf),
 * so the TC.10.001 baseline and proposed runs use this one binary. */
#define _GNU_SOURCE
#define DISTILBERT_POWER_ONLY 1
#define DISTILBERT_USE_GEMMINI_BACKEND 1
#include "gemmini.h"
#include "power_mgr.h"
#define NPU_TASK_BEGIN() pm_runtime_task_begin()
#define NPU_TASK_END()   pm_runtime_task_end()
#include "distilbert_gemmini_infer.c"
