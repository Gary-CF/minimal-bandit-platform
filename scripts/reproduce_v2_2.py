"""Run the remaining v2.2 experiment pipeline, with resumable smoke/full archives."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2_2_remaining_r1"))
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--frozen-test-run", type=Path, help="Also read-only verify and report the earlier 315-episode test")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output.resolve()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    def call(*cmd):
        print("RUN:", " ".join(map(str, cmd)), flush=True)
        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, check=True)
    if not args.verify_only:
        call(sys.executable, "-m", "pytest", "-q", "tests")
    if args.frozen_test_run:
        frozen = args.frozen_test_run.resolve()
        call(sys.executable, "-m", "scripts.run_frozen_logistic_test", "--output", frozen, "--verify-only")
    for profile in ("smoke", "full"):
        command = [sys.executable, "-m", "scripts.run_remaining_v2_2", "--profile", profile,
                   "--output", output / profile, "--jobs", str(args.jobs)]
        if args.verify_only:
            command.append("--verify-only")
        call(*command)
    if args.frozen_test_run and not args.verify_only:
        report = output / "original_frozen_test_report"
        if not report.exists():
            call(sys.executable, "-m", "scripts.report_frozen_logistic_test",
                 "--run", args.frozen_test_run.resolve(), "--output", report)
    print(f"ALL REMAINING EXPERIMENTS PASS: {output / 'full' / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
