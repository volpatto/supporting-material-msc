from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ALL_CASES = [
    "CampoB01bp",
    "CampoB02bp",
    "CampoB03bp",
    "CampoB01kp",
    "CampoB02kp",
    "CampoB03kp",
    "CampoM01bp",
    "CampoM02bp",
    "CampoM03bp",
    "CampoM01kp",
    "CampoM02kp",
    "CampoM03kp",
]
DEFAULT_SNAPSHOT_MONTHS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 60.0, 100.0, 120.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Chapter 5 Firedrake cases.")
    parser.add_argument("--cases", nargs="*", default=ALL_CASES, help="Subset of thesis case names.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/chapter5"))
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--months", type=float, default=180.0)
    parser.add_argument("--dt-months", type=float, default=1.0)
    parser.add_argument(
        "--snapshot-months",
        type=float,
        nargs="*",
        default=DEFAULT_SNAPSHOT_MONTHS,
    )
    parser.add_argument("--vtk-interval", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--max-picard", "--max-fixed-stress", dest="max_picard", type=int, default=40)
    parser.add_argument("--nonlinear-solver", choices=["picard", "newton"], default="picard")
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--newton-rtol", type=float, default=1.0e-8)
    parser.add_argument("--newton-atol", type=float, default=1.0e-10)
    parser.add_argument("--newton-stol", type=float, default=1.0e-10)
    parser.add_argument("--newton-monitor", action="store_true")
    parser.add_argument("--permeability-law", choices=["normalized", "tortuosity"], default="normalized")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show per-iteration residuals.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("compressible_hm_2D_fixed_stress_ch5.py")

    for case in args.cases:
        cmd = [
            sys.executable,
            str(script),
            "--case",
            case,
            "--output-root",
            str(args.output_root),
            "--nx",
            str(args.nx),
            "--ny",
            str(args.ny),
            "--months",
            str(args.months),
            "--dt-months",
            str(args.dt_months),
            "--snapshot-months",
            *[str(month) for month in args.snapshot_months],
            "--vtk-interval",
            str(args.vtk_interval),
            "--tolerance",
            str(args.tolerance),
            "--max-picard",
            str(args.max_picard),
            "--nonlinear-solver",
            args.nonlinear_solver,
            "--max-newton",
            str(args.max_newton),
            "--newton-rtol",
            str(args.newton_rtol),
            "--newton-atol",
            str(args.newton_atol),
            "--newton-stol",
            str(args.newton_stol),
            "--permeability-law",
            args.permeability_law,
        ]
        if args.newton_monitor:
            cmd.append("--newton-monitor")
        if not args.verbose:
            cmd.append("--quiet")

        print("\n==>", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
