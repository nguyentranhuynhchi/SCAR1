"""Unified E2E Test Suite Runner for SCAR M2-Pro (Tiers 1-4).

Usage:
    python tests/e2e/test_suite.py              # Run full 4-tier suite
    python tests/e2e/test_suite.py --tier 1     # Run Tier 1 only
    python tests/e2e/test_suite.py --tier 2     # Run Tier 2 only
    python tests/e2e/test_suite.py --tier 3     # Run Tier 3 only
    python tests/e2e/test_suite.py --tier 4     # Run Tier 4 only
    python tests/e2e/test_suite.py --verbose    # Detailed test case output
"""
from __future__ import annotations

import argparse
import sys
import time
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.contract_helpers import get_m2pro_net, get_m2pro_loss
from tests.e2e.test_tier1_feature_coverage import TestTier1FeatureCoverage
from tests.e2e.test_tier2_boundary_corner import TestTier2BoundaryCorner
from tests.e2e.test_tier3_cross_feature import TestTier3CrossFeature
from tests.e2e.test_tier4_real_world import TestTier4RealWorld

TIER_MAP = {
    1: ("Tier 1: Feature Coverage", TestTier1FeatureCoverage),
    2: ("Tier 2: Boundary & Corner Cases", TestTier2BoundaryCorner),
    3: ("Tier 3: Cross-Feature Combinations", TestTier3CrossFeature),
    4: ("Tier 4: Real-World Application Scenarios", TestTier4RealWorld),
}


def build_suite(selected_tier: int | None = None) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    if selected_tier is not None:
        if selected_tier not in TIER_MAP:
            raise ValueError(f"Invalid tier {selected_tier}. Choose from 1, 2, 3, 4.")
        name, test_class = TIER_MAP[selected_tier]
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    else:
        for tier_idx in sorted(TIER_MAP):
            _, test_class = TIER_MAP[tier_idx]
            suite.addTests(loader.loadTestsFromTestCase(test_class))

    return suite


def run_e2e_suite(selected_tier: int | None = None, verbosity: int = 2) -> bool:
    import torch

    print("=" * 72)
    print("        SCAR M2-Pro: End-to-End (E2E) Test Suite (Tiers 1-4)")
    print("=" * 72)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA:         {'Available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Not available (CPU mode)'}")

    # Inspect model and loss implementation status
    _, net_is_live = get_m2pro_net(img_size=32)
    _, loss_is_live = get_m2pro_loss()
    print(f"M2ProNet:     {'[LIVE IMPLEMENTATION]' if net_is_live else '[INTERFACE CONTRACT REFERENCE]'}")
    print(f"M2ProLoss:    {'[LIVE IMPLEMENTATION]' if loss_is_live else '[INTERFACE CONTRACT REFERENCE]'}")
    if not (net_is_live and loss_is_live):
        print("Note:         Running progressive interface contract tests pending M1/M2 landing.")
    print("-" * 72)

    if selected_tier is not None:
        print(f"Target:       Running {TIER_MAP[selected_tier][0]}")
    else:
        print("Target:       Running all 4 Tiers (Full Requirement Coverage)")
    print("=" * 72)
    print()

    start_time = time.perf_counter()
    suite = build_suite(selected_tier)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 72)
    print("                           E2E TEST SUMMARY")
    print("=" * 72)
    print(f"Total Tests Run:     {result.testsRun}")
    print(f"Passed:              {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:            {len(result.failures)}")
    print(f"Errors:              {len(result.errors)}")
    print(f"Total Duration:      {elapsed:.3f} seconds")
    print("-" * 72)

    if result.wasSuccessful():
        print("RESULT: ALL E2E TESTS PASSED [SUCCESS]")
        print("=" * 72)
        return True
    else:
        print("RESULT: E2E TESTS FAILED [FAIL]")
        print("=" * 72)
        return False


def main():
    parser = argparse.ArgumentParser(description="SCAR M2-Pro E2E Test Suite Runner")
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Execute only the specified tier (1: Feature, 2: Boundary, 3: Combinations, 4: Real-World)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Run tests in verbose mode (default: True)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Run tests in quiet mode",
    )
    args = parser.parse_args()

    verbosity = 1 if args.quiet else (2 if args.verbose else 1)
    success = run_e2e_suite(selected_tier=args.tier, verbosity=verbosity)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
