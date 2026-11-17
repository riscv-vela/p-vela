#!/usr/bin/env python3
"""Unit tests for the power model itself.

Covers the three functions that turn a utilization sample into a power value:
the CPU formula with its two regimes, the three-state engine model, and the
utilization threshold that feeds it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from power_model_config import number
from power_sim_gemmini import (
    BOOST_OFF_TH,
    DEFAULT_GEMMINI_IDLE_POWER_W,
    cpu_power,
    gemmini_power,
    threshold_gemmini_busy,
)

A_NOM, K_NOM = number("CPU_A_NOM"), number("CPU_K_NOM")
A_BOOST, K_BOOST = number("CPU_A_BOOST"), number("CPU_K_BOOST")


class CpuPowerTest(unittest.TestCase):
    """P = a * util * f^3 + k, with separate coefficients above the boost
    threshold. The static term k is what makes an idle core cost anything."""

    def test_nominal_regime_formula(self):
        f, u = 1.4, 0.5
        self.assertAlmostEqual(cpu_power(f, u), A_NOM * u * f**3 + K_NOM)

    def test_boost_regime_formula(self):
        f, u = 3.2, 0.5
        self.assertGreater(f, BOOST_OFF_TH)
        self.assertAlmostEqual(cpu_power(f, u), A_BOOST * u * f**3 + K_BOOST)

    def test_regime_switches_at_threshold(self):
        """The threshold itself is nominal; anything above it is boost."""
        at = cpu_power(BOOST_OFF_TH, 0.0)
        above = cpu_power(BOOST_OFF_TH + 1e-9, 0.0)
        self.assertAlmostEqual(at, K_NOM)
        self.assertAlmostEqual(above, K_BOOST)

    def test_idle_core_still_draws_static_power(self):
        self.assertAlmostEqual(cpu_power(1.4, 0.0), K_NOM)
        self.assertGreater(K_NOM, 0.0)

    def test_cubic_in_frequency(self):
        """Doubling the frequency within one regime raises dynamic power 8x."""
        dyn = lambda f: cpu_power(f, 1.0) - K_NOM
        self.assertAlmostEqual(dyn(2.0) / dyn(1.0), 8.0, places=6)

    def test_monotonic_in_utilization(self):
        vals = [cpu_power(1.4, u / 10) for u in range(11)]
        self.assertEqual(vals, sorted(vals))


class GemminiPowerTest(unittest.TestCase):
    """Three states. Busy% decides active vs idle, but the gating log
    overrides both -- that is the whole point of feeding it in separately."""

    FULL = 0.862

    def test_active_at_full_utilization(self):
        self.assertAlmostEqual(gemmini_power(1.0, self.FULL), self.FULL)

    def test_idle_but_powered(self):
        self.assertAlmostEqual(gemmini_power(0.0, self.FULL),
                               DEFAULT_GEMMINI_IDLE_POWER_W)
        self.assertGreater(DEFAULT_GEMMINI_IDLE_POWER_W, 0.0)

    def test_gated_is_zero(self):
        self.assertEqual(gemmini_power(0.0, self.FULL, gated=True), 0.0)

    def test_gating_overrides_a_busy_sample(self):
        """A sample can read busy and still be gated when the gate landed
        mid-interval; the gate log wins."""
        self.assertEqual(gemmini_power(1.0, self.FULL, gated=True), 0.0)


class ThresholdTest(unittest.TestCase):
    """Busy% is binarised before it reaches the power model: the engine is
    either running a command or it is not."""

    def test_at_or_above_threshold_is_full(self):
        self.assertEqual(threshold_gemmini_busy(80.0, 80.0), 100.0)
        self.assertEqual(threshold_gemmini_busy(91.3, 80.0), 100.0)

    def test_below_threshold_is_zero(self):
        self.assertEqual(threshold_gemmini_busy(79.9, 80.0), 0.0)
        self.assertEqual(threshold_gemmini_busy(0.08, 80.0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
