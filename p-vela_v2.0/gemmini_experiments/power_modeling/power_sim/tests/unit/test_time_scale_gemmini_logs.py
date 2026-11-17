#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from time_scale_gemmini_logs import (
    IndependentModelTimeScaler,
    PiecewiseTimeScaler,
    build_model_start_anchors,
    build_per_model_time_summary,
    resolve_model_rtl_time_scales,
    subtract_intervals,
)


class CpuGemminiTimeScalingTest(unittest.TestCase):
    def test_scales_only_inference_cpu_and_gemmini_segments(self):
        sample_intervals = [(101.0, 107.0)]
        gemmini_intervals = [(103.0, 105.0)]
        cpu_intervals = subtract_intervals(sample_intervals, gemmini_intervals)
        scaler = PiecewiseTimeScaler(
            gemmini_intervals,
            0.5,
            cpu_intervals=cpu_intervals,
            cpu_scale=0.25,
        )

        self.assertEqual(cpu_intervals, [(101.0, 103.0), (105.0, 107.0)])
        for raw_ts, expected_ts in (
            (101.0, 101.0),
            (103.0, 101.5),
            (105.0, 102.5),
            (107.0, 103.0),
            (108.0, 104.0),
        ):
            self.assertAlmostEqual(scaler.map_time(raw_ts), expected_ts)

    def test_cpu_scale_one_preserves_gemmini_only_behavior(self):
        scaler = PiecewiseTimeScaler([(2.0, 4.0)], 0.5)

        self.assertAlmostEqual(scaler.map_time(10.0), 9.0)

    def test_dual_apps_scale_cpu_on_each_model_timeline(self):
        sample_windows = [
            {"key": ("app_a", 0, 0), "start_ts": 100.0, "end_ts": 110.0},
            {"key": ("app_a", 1, -1), "start_ts": 102.0, "end_ts": 106.0},
            {"key": ("app_b", 0, 0), "start_ts": 100.0, "end_ts": 110.0},
        ]
        scaler = IndependentModelTimeScaler(
            [
                {"model_key": "app_a", "start_ts": 102.0, "end_ts": 106.0},
                {"model_key": "app_b", "start_ts": 104.0, "end_ts": 108.0},
            ],
            {"app_a": 0.5, "app_b": 0.4},
            sample_windows=sample_windows,
            cpu_scale=0.25,
        )

        self.assertEqual(scaler.sample_intervals_by_model["app_a"], [(100.0, 110.0)])
        self.assertEqual(
            scaler.cpu_intervals_by_model["app_a"],
            [(100.0, 102.0), (106.0, 110.0)],
        )
        self.assertEqual(
            scaler.cpu_intervals_by_model["app_b"],
            [(100.0, 104.0), (108.0, 110.0)],
        )
        for model, expected in {
            "app_a": ((102.0, 100.5), (106.0, 102.5), (110.0, 103.5)),
            "app_b": ((104.0, 101.0), (108.0, 102.6), (110.0, 103.1)),
        }.items():
            for raw_ts, expected_ts in expected:
                self.assertAlmostEqual(
                    scaler.map_model_time(model, raw_ts), expected_ts
                )

        per_model, cpu_rows = build_per_model_time_summary(scaler)
        self.assertAlmostEqual(per_model["app_a"]["inference_scaled_time_s"], 3.5)
        self.assertAlmostEqual(per_model["app_b"]["inference_scaled_time_s"], 3.1)
        self.assertAlmostEqual(per_model["app_a"]["cpu_scaled_time_s"], 1.5)
        self.assertAlmostEqual(per_model["app_b"]["cpu_scaled_time_s"], 1.5)
        self.assertEqual({row["model_key"] for row in cpu_rows}, {"app_a", "app_b"})

    def test_launch_scaling_preserves_dual_overlap(self):
        anchors = build_model_start_anchors(
            [
                {
                    "model_key": "app_a",
                    "kind": "timeline",
                    "label": "remote_launch",
                    "host_ts": 100.0,
                },
                {
                    "model_key": "app_b",
                    "kind": "timeline",
                    "label": "remote_launch",
                    "host_ts": 100.0,
                },
            ]
        )
        sample_windows = [
            {"key": ("app_a", 0, 0), "start_ts": 110.0, "end_ts": 120.0},
            {"key": ("app_b", 0, 0), "start_ts": 112.0, "end_ts": 122.0},
        ]
        scaler = IndependentModelTimeScaler(
            [
                {"model_key": "app_a", "start_ts": 112.0, "end_ts": 118.0},
                {"model_key": "app_b", "start_ts": 114.0, "end_ts": 120.0},
            ],
            {"app_a": 0.5, "app_b": 0.4},
            sample_windows=sample_windows,
            model_start_anchors=anchors,
            cpu_scale=0.25,
        )

        app_a_start = scaler.map_model_time("app_a", 110.0)
        app_a_end = scaler.map_model_time("app_a", 120.0)
        app_b_start = scaler.map_model_time("app_b", 112.0)
        app_b_end = scaler.map_model_time("app_b", 122.0)
        self.assertAlmostEqual(app_a_start, 102.5)
        self.assertAlmostEqual(app_b_start, 103.0)
        self.assertGreater(min(app_a_end, app_b_end) - max(app_a_start, app_b_start), 0.0)

        per_model, _ = build_per_model_time_summary(scaler)
        self.assertAlmostEqual(per_model["app_a"]["startup_scaled_time_s"], 2.5)
        self.assertAlmostEqual(per_model["app_b"]["startup_scaled_time_s"], 3.0)

    def test_dual_rtl_cycle_scales_are_resolved_per_model(self):
        entries = []
        for model, interval_count in (("distilbert", 26), ("distilgpt2", 24)):
            entries.append(
                {
                    "model_key": model,
                    "kind": "sample-start",
                    "actor": None,
                    "label": None,
                }
            )
            entries.extend(
                {
                    "model_key": model,
                    "kind": "timeline",
                    "actor": "gemmini",
                    "label": f"gemmini_{idx}",
                }
                for idx in range(interval_count)
            )

        labeled_intervals = [
            {"model_key": "distilbert", "start_ts": 100.0, "end_ts": 102.0},
            {"model_key": "distilgpt2", "start_ts": 100.0, "end_ts": 103.0},
        ]
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as bench:
            bench.write(
                "[HOST_TS_LINE] 100.0 [distilbert] "
                "[backend-profile] backend=gemmini tiled_matmul_ms=1500\n"
            )
            bench.write(
                "[HOST_TS_LINE] 100.0 [distilgpt2] "
                "[backend-profile] backend=gemmini tiled_matmul_ms=2500\n"
            )
            bench_path = bench.name
        self.addCleanup(os.unlink, bench_path)

        args = SimpleNamespace(
            bench=bench_path,
            cpu_time_scale=0.235,
            gemmini_frequency_mhz=200.0,
        )
        scales, model_rtl = resolve_model_rtl_time_scales(
            args,
            entries,
            labeled_intervals,
            {"distilbert": "distilbert", "distilgpt2": "distilgpt2"},
        )

        self.assertAlmostEqual(
            model_rtl["distilbert"]["raw_tiled_matmul_profile_s"], 1.5
        )
        self.assertAlmostEqual(
            model_rtl["distilgpt2"]["raw_tiled_matmul_profile_s"], 2.5
        )
        for model, raw_time in (("distilbert", 2.0), ("distilgpt2", 3.0)):
            rtl = model_rtl[model]
            self.assertAlmostEqual(rtl["embedded_cpu_raw_time_s"], 0.5)
            self.assertAlmostEqual(
                scales[model], (rtl["rtl_time_s"] + 0.5 * 0.235) / raw_time
            )
            self.assertEqual(
                rtl["rtl_expected_gemmini_intervals"],
                rtl["rtl_observed_gemmini_intervals"],
            )


if __name__ == "__main__":
    unittest.main()
