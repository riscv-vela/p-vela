#!/usr/bin/env python3

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from gemmini_hw_cycle_model import (
    LoopWsBounds,
    estimate_workload,
    loop_ws_cycles,
)


class GemminiRtlCycleModelTest(unittest.TestCase):
    def test_transformer_bounds_match_rtl(self):
        self.assertEqual(loop_ws_cycles(LoopWsBounds(1, 9, 48)), 9747)
        self.assertEqual(loop_ws_cycles(LoopWsBounds(1, 3, 48)), 2339)
        self.assertEqual(
            loop_ws_cycles(LoopWsBounds(1, 9, 48, pad_m=15)), 9735
        )

    def test_application_totals_include_sequence_sharing(self):
        expected = {
            "distilbert": 29181073,
            "distilroberta": 29181073,
            "distilgpt2": 29260320,
            "polybench_gemm": 44574,
        }
        for workload, cycles in expected.items():
            with self.subTest(workload=workload):
                estimate = estimate_workload(workload)
                self.assertEqual(estimate.total_cycles, cycles)
                self.assertAlmostEqual(
                    estimate.rtl_time_s, cycles / 200_000_000.0
                )


if __name__ == "__main__":
    unittest.main()
