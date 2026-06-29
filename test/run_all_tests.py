"""
Test runner - discovers and runs all tests in the test directory.
Usage: uv run python test/run_all_tests.py [options]
"""
import sys
import unittest
import time
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401

TEST_CASES_PATH = Path(__file__).resolve().parent / "testcase.md"


def discover_and_run(verbosity: int = 2, pattern: str = "test_*.py") -> unittest.TestResult:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def update_testcase_md(results: unittest.TestResult, elapsed: float) -> None:
    testcase_path = TEST_CASES_PATH
    if not testcase_path.exists():
        print(f"Warning: {testcase_path} not found, skipping update")
        return

    lines = []
    with open(testcase_path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.split("\n")

    # Update the status section
    new_lines = []
    in_status_block = False
    for line in lines:
        if line.strip().startswith("| `test_"):
            test_name = line.split("`")[1]
            status = "✅ Pass" if test_name not in [t for t, _ in results.failures] and test_name not in [t for t, _ in results.errors] else "❌ Fail"
            new_lines.append(f"| `{test_name}` | {status} |")
            continue
        if line.strip().startswith("| `"):
            test_name = line.split("`")[1]
            status = "✅ Pass" if test_name not in [t for t, _ in results.failures] and test_name not in [t for t, _ in results.errors] else "❌ Fail"
            new_lines.append(f"| `{test_name}` | {status} |")
            continue
        new_lines.append(line)

    with open(testcase_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    print(f"Updated {testcase_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all agent tests")
    parser.add_argument("-v", "--verbosity", type=int, default=2, help="Verbosity level (1 or 2)")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Test file pattern")
    parser.add_argument("--no-doc-update", action="store_true", help="Skip testcase.md update")
    args = parser.parse_args()

    start = time.time()
    result = discover_and_run(verbosity=args.verbosity, pattern=args.pattern)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  Tests: {result.testsRun}  |  Passed: {result.testsRun - len(result.failures) - len(result.errors)}  |  "
          f"Failures: {len(result.failures)}  |  Errors: {len(result.errors)}")
    print(f"  Time: {elapsed:.2f}s")
    if result.wasSuccessful():
        print(f"  Result: ✅ ALL TESTS PASSED")
    else:
        print(f"  Result: ❌ SOME TESTS FAILED")
    print(f"{'=' * 60}")

    if not args.no_doc_update:
        update_testcase_md(result, elapsed)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
