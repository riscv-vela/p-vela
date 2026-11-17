#!/usr/bin/env python3
"""Validated no-DMA Gemmini LOOP_WS cycle model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import cached_property

from power_model_config import number


DIM = 16
FIXED_OVERHEAD_CYCLES = 35
DEFAULT_FREQUENCY_MHZ = number("DEFAULT_GEMMINI_FREQUENCY_MHZ")


@dataclass(frozen=True)
class LoopWsBounds:
    m_tiles: int
    n_tiles: int
    k_tiles: int
    pad_m: int = 0
    pad_n: int = 0
    pad_k: int = 0

    @property
    def logical_m(self) -> int:
        return DIM * self.m_tiles - self.pad_m


@dataclass(frozen=True)
class WorkloadProfile:
    api_calls: int
    gemmini_intervals: int
    loop_ws_counts: tuple[tuple[LoopWsBounds, int], ...]

    @cached_property
    def loop_ws_calls(self) -> int:
        return sum(count for _, count in self.loop_ws_counts)

    @cached_property
    def cycles_per_execution(self) -> int:
        """Sum this fixed profile once per process, independent of run/frequency."""
        per_command_total = sum(
            loop_ws_cycles(bounds) * count
            for bounds, count in self.loop_ws_counts
        )
        # Back-to-back commands from one API share this fixed cost.
        return per_command_total - FIXED_OVERHEAD_CYCLES * (
            self.loop_ws_calls - self.api_calls
        )


@dataclass(frozen=True)
class WorkloadCycleEstimate:
    workload: str
    executions: int
    cycles_per_execution: int
    total_cycles: int
    frequency_mhz: float
    rtl_time_s: float
    api_calls: int
    loop_ws_calls: int
    gemmini_intervals: int


_TRANSFORMER_LOOP_WS_COUNTS = (
    (LoopWsBounds(1, 9, 48), 1968),
    (LoopWsBounds(1, 3, 48), 240),
    (LoopWsBounds(1, 8, 56), 720),
    (LoopWsBounds(1, 8, 24), 240),
    (LoopWsBounds(1, 1, 56), 720),
    (LoopWsBounds(1, 1, 24), 240),
    (LoopWsBounds(1, 3, 128), 48),
    (LoopWsBounds(1, 3, 64), 48),
    (LoopWsBounds(1, 9, 48, pad_m=15), 5),
    (LoopWsBounds(1, 3, 48, pad_m=15), 1),
    (LoopWsBounds(1, 1, 48, pad_m=15, pad_n=14), 1),
)

WORKLOAD_PROFILES = {
    "distilbert": WorkloadProfile(
        api_calls=2503,
        gemmini_intervals=26,
        loop_ws_counts=_TRANSFORMER_LOOP_WS_COUNTS,
    ),
    "distilroberta": WorkloadProfile(
        api_calls=2503,
        gemmini_intervals=26,
        loop_ws_counts=_TRANSFORMER_LOOP_WS_COUNTS,
    ),
    "distilgpt2": WorkloadProfile(
        api_calls=2400,
        gemmini_intervals=24,
        loop_ws_counts=(
            (LoopWsBounds(1, 9, 48), 2016),
            (LoopWsBounds(1, 3, 48), 96),
            (LoopWsBounds(1, 8, 56), 720),
            (LoopWsBounds(1, 8, 24), 240),
            (LoopWsBounds(1, 1, 56), 720),
            (LoopWsBounds(1, 1, 24), 240),
            (LoopWsBounds(1, 3, 128), 48),
            (LoopWsBounds(1, 3, 64), 48),
        ),
    ),
    "polybench_gemm": WorkloadProfile(
        api_calls=26,
        gemmini_intervals=1,
        loop_ws_counts=(
            (LoopWsBounds(1, 8, 15), 12),
            (LoopWsBounds(1, 6, 15, pad_n=4), 12),
            (LoopWsBounds(1, 8, 15, pad_m=8), 1),
            (LoopWsBounds(1, 6, 15, pad_m=8, pad_n=4), 1),
        ),
    ),
}


def loop_ws_cycles(bounds: LoopWsBounds) -> int:
    """Return accept-to-idle cycles for the RTL-validated M_tiles=1 scope."""
    if min(bounds.m_tiles, bounds.n_tiles, bounds.k_tiles) <= 0:
        raise ValueError("LOOP_WS tile counts must be positive")
    if bounds.m_tiles != 1:
        raise ValueError("the current RTL model is validated only for M_TILES=1")
    for name, value in (
        ("pad_m", bounds.pad_m),
        ("pad_n", bounds.pad_n),
        ("pad_k", bounds.pad_k),
    ):
        if not 0 <= value < DIM:
            raise ValueError(f"{name} must be in [0, {DIM})")

    m = bounds.logical_m
    phases = bounds.n_tiles * bounds.k_tiles
    compute_span = DIM * bounds.m_tiles * phases - bounds.pad_m
    m_drain_correction = max(0, 4 - m)
    backpressure_correction = DIM * max(0, phases - 257)
    return (
        FIXED_OVERHEAD_CYCLES
        + compute_span
        + m_drain_correction
        + backpressure_correction
    )


def workload_cycles(workload: str) -> int:
    return WORKLOAD_PROFILES[workload].cycles_per_execution


def estimate_workload(
    workload: str,
    executions: int = 1,
    frequency_mhz: float = DEFAULT_FREQUENCY_MHZ,
) -> WorkloadCycleEstimate:
    if workload not in WORKLOAD_PROFILES:
        raise ValueError(f"unsupported workload: {workload}")
    if executions <= 0:
        raise ValueError("executions must be positive")
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")

    profile = WORKLOAD_PROFILES[workload]
    cycles_per_execution = workload_cycles(workload)
    total_cycles = cycles_per_execution * executions
    return WorkloadCycleEstimate(
        workload=workload,
        executions=executions,
        cycles_per_execution=cycles_per_execution,
        total_cycles=total_cycles,
        frequency_mhz=frequency_mhz,
        rtl_time_s=total_cycles / (frequency_mhz * 1_000_000.0),
        api_calls=profile.api_calls * executions,
        loop_ws_calls=profile.loop_ws_calls * executions,
        gemmini_intervals=profile.gemmini_intervals * executions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=sorted(WORKLOAD_PROFILES))
    parser.add_argument("--executions", type=int, default=1)
    parser.add_argument(
        "--frequency-mhz", type=float, default=DEFAULT_FREQUENCY_MHZ
    )
    args = parser.parse_args()
    estimate = estimate_workload(
        args.workload, args.executions, args.frequency_mhz
    )
    print(json.dumps(asdict(estimate), indent=2))


if __name__ == "__main__":
    main()
