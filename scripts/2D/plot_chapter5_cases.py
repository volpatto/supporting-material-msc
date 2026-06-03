from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LX = 10.0
LY = 50.0

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

COUPLING_LABELS = {"01": "rigid", "02": "one-way", "03": "two-way"}
DEPTH_LABELS = {"bp": "shallow", "kp": "deep"}
FORMATION_LABELS = {"B": "Barnett", "M": "Marcellus"}
STRIP_Y = {"B": 0.25 * LY, "M": 0.5 * LY, "T": 0.75 * LY}


def case_label(case: str) -> str:
    return f"{FORMATION_LABELS[case[5]]} {DEPTH_LABELS[case[8:10]]} {COUPLING_LABELS[case[6:8]]}"


def month_suffix(month: float) -> str:
    return f"{month:g}".replace(".", "p")


def load_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0:
        return None
    return np.atleast_1d(data)


def plot_horizontal_profiles(case: str, output_root: Path, plot_root: Path) -> None:
    case_dir = output_root / case
    months = sorted(
        float(path.stem.split("_m")[-1].replace("p", "."))
        for path in case_dir.glob("profiles_horizontal_m*.csv")
    )
    if not months:
        return

    fields = {
        "pressure": ("pressure", "Pressure [MPa]", 1.0e-6),
        "vx": ("vx", r"$v_x$ [m/s]", 1.0),
        "jx": ("jx", r"$J_x$ [kg/(m^2 s)]", 1.0),
    }

    for strip, y_value in STRIP_Y.items():
        for field_name, (column, ylabel, scale) in fields.items():
            fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
            for month in months:
                current = load_csv(case_dir / f"profiles_horizontal_m{month_suffix(month)}.csv")
                if current is None:
                    continue
                mask = current["strip"] == strip
                if not np.any(mask):
                    continue
                ax.plot(
                    current["x"][mask],
                    current[column][mask] * scale,
                    linewidth=1.6,
                    label=f"FD {month:g} mo",
                )

            ax.set_title(f"{case}: {field_name} on strip {strip}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, ncols=2, fontsize=7)
            outdir = plot_root / "profiles" / field_name
            outdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(outdir / f"{case}_{field_name}_{strip}.png", dpi=180)
            plt.close(fig)


def plot_production(cases: list[str], output_root: Path, plot_root: Path) -> None:
    outdir = plot_root / "production"
    outdir.mkdir(parents=True, exist_ok=True)

    for formation_code, formation in FORMATION_LABELS.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        for case in cases:
            if case[5] != formation_code:
                continue
            label = f"{DEPTH_LABELS[case[8:10]]} {COUPLING_LABELS[case[6:8]]}"
            current = load_csv(output_root / case / "production.csv")
            if current is not None and "cumulative_produced_kg" in current.dtype.names:
                ax.plot(
                    current["month"],
                    current["cumulative_produced_kg"],
                    linewidth=1.6,
                    label=f"FD {label}",
                )

        ax.set_title(f"{formation}: cumulative gas production")
        ax.set_xlabel("time [months]")
        ax.set_ylabel("produced gas [kg]")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, ncols=2, fontsize=7)
        fig.savefig(outdir / f"{formation}_production.png", dpi=180)
        plt.close(fig)


def plot_vertical_profiles(case: str, output_root: Path, plot_root: Path) -> None:
    case_dir = output_root / case
    months = sorted(
        float(path.stem.split("_m")[-1].replace("p", "."))
        for path in case_dir.glob("profiles_vertical_m*.csv")
    )
    if not months or case[6:8] == "01":
        return

    fields = {
        "uy": ("uy", r"$u_y$ [m]", 1.0),
        "sigma_xx": ("sigma_xx", r"$\sigma_{xx}$ [MPa]", 1.0e-6),
        "sigma_yy": ("sigma_yy", r"$\sigma_{yy}$ [MPa]", 1.0e-6),
        "sigma_T": ("sigma_T", r"$\widetilde{\sigma}_T$ [MPa]", 1.0e-6),
    }

    for field_name, (column, xlabel, scale) in fields.items():
        fig, ax = plt.subplots(figsize=(5.2, 6.2), constrained_layout=True)
        for month in months:
            current = load_csv(case_dir / f"profiles_vertical_m{month_suffix(month)}.csv")
            if current is None:
                continue
            mask = current["strip"] == "C"
            if np.any(mask):
                ax.plot(
                    current[column][mask] * scale,
                    current["y"][mask],
                    linewidth=1.6,
                    label=f"FD {month:g} mo",
                )

        ax.set_title(f"{case}: {field_name} on strip C")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, ncols=2, fontsize=7)
        outdir = plot_root / "geomechanics"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{case}_{field_name}_C.png", dpi=180)
        plt.close(fig)


def grid_from_cell_csv(data: np.ndarray, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    grid = np.full((len(ys), len(xs)), np.nan)
    x_index = {value: i for i, value in enumerate(xs)}
    y_index = {value: i for i, value in enumerate(ys)}
    for row in data:
        grid[y_index[row["y"]], x_index[row["x"]]] = row[column]
    return xs, ys, grid


def plot_cell_fields(case: str, output_root: Path, plot_root: Path) -> None:
    case_dir = output_root / case
    files = sorted(case_dir.glob("cell_fields_m*.csv"))[:8]
    if not files or case[6:8] == "01":
        return

    fields = {
        "sigma_T": (r"$\widetilde{\sigma}_T$ [MPa]", 1.0e-6),
        "source_rate": ("fixed-stress source rate", 1.0),
    }
    for column, (title, scale) in fields.items():
        fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0), constrained_layout=True, sharex=True, sharey=True)
        used = False
        for ax, path in zip(axes.ravel(), files):
            data = load_csv(path)
            if data is None:
                ax.axis("off")
                continue
            month = float(path.stem.split("_m")[-1].replace("p", "."))
            xs, ys, grid = grid_from_cell_csv(data, column)
            mesh = ax.pcolormesh(xs, ys, grid * scale, shading="auto")
            ax.set_title(f"{month:g} mo", fontsize=9)
            ax.set_aspect("equal")
            fig.colorbar(mesh, ax=ax, shrink=0.75)
            used = True
        for ax in axes[-1, :]:
            ax.set_xlabel("x [m]")
        for ax in axes[:, 0]:
            ax.set_ylabel("y [m]")
        fig.suptitle(f"{case}: {title}")
        outdir = plot_root / "fields"
        outdir.mkdir(parents=True, exist_ok=True)
        if used:
            fig.savefig(outdir / f"{case}_{column}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for Chapter 5 Firedrake runs.")
    parser.add_argument("--cases", nargs="*", default=ALL_CASES)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/chapter5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_root = args.output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        plot_horizontal_profiles(case, args.output_root, plot_root)
        plot_vertical_profiles(case, args.output_root, plot_root)
        plot_cell_fields(case, args.output_root, plot_root)
    plot_production(args.cases, args.output_root, plot_root)
    print(f"Wrote Chapter 5 plots to {plot_root.resolve()}")


if __name__ == "__main__":
    main()
