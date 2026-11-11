"""Run all tests without pytest: python -m tests.run_tests (pytest is not in the base repo lock file)."""
import importlib
import sys
import time
import traceback


def main():
    t0, failed, n = time.time(), 0, 0
    for mod in ["tests.test_models", "tests.test_ppo"]:
        m = importlib.import_module(mod)
        for name in sorted(k for k in dir(m) if k.startswith("test_")):
            n += 1
            try:
                getattr(m, name)()
                print(f"PASS {mod}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {mod}.{name}\n{traceback.format_exc()}")
    print(f"{n - failed} passed, {failed} failed in {time.time() - t0:.1f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
