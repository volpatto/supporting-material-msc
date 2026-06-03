from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LX = 10.0
LY = 50.0
SECONDS_PER_MONTH = 2.628e6
BETA_F = 3.2e-8
GAS_MOLAR_MASS = 1.604246e-2
GAS_CONSTANT = 8.31446261815324
GAS_VISCOSITY = 1.2e-5

MONTHS_PROFILE = (1.0, 5.0, 10.0)
MONTHS_FIELD = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

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

STRIP_Y = {"B": 0.25 * LY, "M": 0.5 * LY, "T": 0.75 * LY}
STRIP_X = {"L": 0.25 * LX, "C": 0.5 * LX, "R": 0.75 * LX}


@dataclass(frozen=True)
class CaseInfo:
    name: str
    formation_code: str
    coupling_code: str
    depth_code: str
    alpha: float
    young: float
    poisson: float
    kappa0: float
    gas_temperature: float

    @property
    def bulk_modulus(self) -> float:
        shear = self.young / (2.0 * (1.0 + self.poisson))
        lame = self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        return lame + (2.0 / 3.0) * shear


def case_info(case_name: str) -> CaseInfo:
    formation = case_name[5]
    coupling = case_name[6:8]
    depth = case_name[8:10]
    if formation == "B":
        return CaseInfo(case_name, formation, coupling, depth, 0.69, 20.0e9, 0.23, 1.5e-19, 338.45)
    if formation == "M":
        return CaseInfo(case_name, formation, coupling, depth, 0.91, 6.0e9, 0.23, 6.0e-19, 352.55)
    raise ValueError(f"Unsupported Chapter 5 case name: {case_name}")


def month_suffix(month: float) -> str:
    return f"{month:g}".replace(".", "p")


def load_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return np.atleast_1d(data)


PR_Z_COEFFICIENTS: dict[tuple[str, str], tuple[float, ...]] = {
    ("B", "bp"): (
        1.00000056274788895e00,
        -1.43197954645747126e-08,
        3.40426009023314045e-16,
        7.59907671558197778e-24,
        -4.11054389510077150e-32,
        2.92822924331161605e-39,
        -1.37471922256066198e-45,
        7.20994329376561970e-53,
        -1.51051020984995609e-60,
        1.18116464423077240e-68,
    ),
    ("B", "kp"): (
        1.00002409272612436e00,
        -1.43625042045230104e-08,
        3.64744138512952407e-16,
        1.22208997121274891e-24,
        8.66687917902170691e-31,
        -7.29973759992517208e-38,
        2.46497588710645164e-45,
        -4.34345835012191355e-53,
        3.94296266089095450e-61,
        -1.44523209281623973e-69,
    ),
    ("M", "bp"): (
        1.00000419632392568e00,
        -1.23006255232045819e-08,
        3.32643309472354619e-16,
        3.20388430525322964e-24,
        3.23577439089293074e-31,
        -3.65645294498147964e-38,
        1.29940511760457422e-45,
        -2.34132797182499811e-53,
        2.18490537936984735e-61,
        -8.42413282878880619e-70,
    ),
    ("M", "kp"): (
        9.99918565691723660e-01,
        -1.21934261515046661e-08,
        2.93555158557085215e-16,
        9.56148103089321076e-24,
        -2.27683095381002984e-31,
        -8.82961283630558861e-39,
        4.63894880996091385e-46,
        -8.56167354166185042e-54,
        7.49014828700013440e-62,
        -2.60295362178205543e-70,
    ),
}


def pr_coefficients(case_name: str) -> list[float]:
    return list(PR_Z_COEFFICIENTS[(case_name[5], case_name[8:10])])


def z_factor_np(pressure: np.ndarray | float, coefficients: list[float]) -> np.ndarray:
    pressure_array = np.asarray(pressure, dtype=float)
    value = np.zeros_like(pressure_array, dtype=float)
    for power, coeff in enumerate(coefficients):
        value += coeff * pressure_array**power
    return value


def setup_curve_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle=(0, (1.5, 5.0)), color="black", linewidth=0.75)
    ax.tick_params(direction="in", top=True, right=True)


def reserve_legend_column(fig: plt.Figure) -> None:
    fig.subplots_adjust(right=0.68)


def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def apply_thesis_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 6.0),
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
            "lines.linewidth": 2.0,
            "mathtext.fontset": "cm",
            "axes.unicode_minus": True,
        }
    )


def profile_xy(output_root: Path, case: str, month: float, column: str, strip: str = "M") -> tuple[np.ndarray, np.ndarray]:
    data = load_csv(output_root / case / f"profiles_horizontal_m{month_suffix(month)}.csv")
    mask = data["strip"] == strip
    if not np.any(mask):
        raise ValueError(f"No strip {strip!r} in {case} at month {month:g}.")
    return data["x"][mask], data[column][mask]


def legend_heading(label: str) -> Line2D:
    return Line2D([], [], color="none", label=label)


def compare_case_name(case: str, coupling: str) -> str:
    return f"{case[:6]}{coupling}{case[8:]}"


def plot_single_profile_set(
    output_root: Path,
    case: str,
    column: str,
    ylabel: str,
    out_path: Path,
) -> None:
    colors = ["b", "g", "r"]
    fig, ax = plt.subplots()
    for color, month in zip(colors, MONTHS_PROFILE):
        x_values, values = profile_xy(output_root, case, month, column)
        ax.plot(x_values, values, color=color, label=f"{month:g} mês" if month == 1.0 else f"{month:g} meses")
    ax.set_xlabel(r"$x\ (m)$")
    ax.set_ylabel(ylabel)
    setup_curve_axes(ax)
    ax.set_xlim(0.0, LX)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=False, framealpha=1.0)
    reserve_legend_column(fig)
    save_pdf(fig, out_path)


def plot_rigid_twoway_comparison(
    output_root: Path,
    case: str,
    column: str,
    ylabel: str,
    out_path: Path,
) -> None:
    rigid_case = compare_case_name(case, "01")
    twoway_case = compare_case_name(case, "03")
    month_styles = [
        ("1 mês", "g", "r"),
        ("5 meses", "m", "y"),
        ("10 meses", "b", "g"),
    ]
    dotted = (0, (1.5, 4.0))

    fig, ax = plt.subplots()
    handles: list[Line2D] = []
    for month, (heading, rigid_color, twoway_color) in zip(MONTHS_PROFILE, month_styles):
        handles.append(legend_heading(heading))
        x_values, values = profile_xy(output_root, rigid_case, month, column)
        rigid_line = ax.plot(x_values, values, color=rigid_color, linestyle="-", label="rígido")[0]
        x_values, values = profile_xy(output_root, twoway_case, month, column)
        twoway_line = ax.plot(x_values, values, color=twoway_color, linestyle=dotted, label="duas-vias")[0]
        handles.extend([rigid_line, twoway_line])

    ax.set_xlabel(r"$x\ (m)$")
    ax.set_ylabel(ylabel)
    setup_curve_axes(ax)
    ax.set_xlim(0.0, LX)
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=False, framealpha=1.0)
    reserve_legend_column(fig)
    save_pdf(fig, out_path)


def plot_pressure_and_flux(output_root: Path, out_root: Path, cases: list[str]) -> None:
    for case in cases:
        target = out_root / case
        if case[6:8] == "01":
            plot_rigid_twoway_comparison(output_root, case, "pressure", r"$p\ (Pa)$", target / "P_M.pdf")
            plot_rigid_twoway_comparison(output_root, case, "jx", r"$J_x\ (kg/(m^2s))$", target / "Jx_M.pdf")
        else:
            plot_single_profile_set(output_root, case, "pressure", r"$p\ (Pa)$", target / "P_M.pdf")
            plot_single_profile_set(output_root, case, "jx", r"$J_x\ (kg/(m^2s))$", target / "Jx_M.pdf")


def plot_production(output_root: Path, out_root: Path, formation: str) -> None:
    if formation == "B":
        target = "CampoB01kp"
    else:
        target = "CampoM01kp"

    styles = [
        (f"Campo{formation}01kp", "b", "-", "Rígido"),
        (f"Campo{formation}02kp", "g", "-", "Uma-via"),
        (f"Campo{formation}03kp", "r", "-", "Duas-vias"),
        (f"Campo{formation}01bp", "c", (0, (1.5, 4.0)), r"Rígido $(raso)$"),
        (f"Campo{formation}02bp", "m", (0, (1.5, 4.0)), r"Uma-via $(raso)$"),
        (f"Campo{formation}03bp", "y", (0, (1.5, 4.0)), r"Duas-vias $(raso)$"),
    ]

    fig, ax = plt.subplots()
    for case, color, linestyle, label in styles:
        data = load_csv(output_root / case / "production.csv")
        if "cumulative_produced_kg" not in data.dtype.names:
            raise ValueError(f"{case}/production.csv has no cumulative production column.")
        ax.plot(data["month"], data["cumulative_produced_kg"], color=color, linestyle=linestyle, label=label)

    ax.set_xlabel(r"$t\ (meses)$")
    ax.set_ylabel("Produção (kg)")
    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(bottom=0.0)
    setup_curve_axes(ax)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0)
    save_pdf(fig, out_root / target / "Prod_compH.pdf")


def plot_kh(out_root: Path) -> None:
    info = case_info("CampoM01kp")
    coeffs = pr_coefficients("CampoM01kp")
    pressure = np.linspace(0.0, 6.4e7, 400)
    z_values = z_factor_np(pressure, coeffs)
    rho = GAS_MOLAR_MASS * pressure / (z_values * GAS_CONSTANT * info.gas_temperature)
    porosities = [(0.04, "b"), (0.07, "g"), (0.10, "r")]

    fig, ax = plt.subplots()
    for phi, color in porosities:
        permeability = info.kappa0 * (2.0 * phi / (3.0 - phi))
        hydraulic_conductivity = rho * permeability / GAS_VISCOSITY
        ax.plot(pressure, hydraulic_conductivity, color=color, label=f"Porosidade = {phi:.2f}")

    ax.set_xlabel("p (Pa)")
    ax.set_ylabel("Condutividade hidráulica (m/s)")
    ax.set_xlim(0.0, 6.4e7)
    ax.set_ylim(bottom=0.0)
    setup_curve_axes(ax)
    ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=1.0)
    save_pdf(fig, out_root / "CampoM01kp" / "Kh.pdf")


def grid_from_cell_csv(data: np.ndarray, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    grid = np.full((len(ys), len(xs)), np.nan)
    x_index = {value: i for i, value in enumerate(xs)}
    y_index = {value: i for i, value in enumerate(ys)}
    for row in data:
        grid[y_index[row["y"]], x_index[row["x"]]] = row[column]
    return xs, ys, grid


def cell_line(output_root: Path, case: str, month: float, column: str, strip: str) -> tuple[np.ndarray, np.ndarray]:
    data = load_csv(output_root / case / f"cell_fields_m{month_suffix(month)}.csv")
    xs, ys, grid = grid_from_cell_csv(data, column)
    if strip in STRIP_X:
        idx = int(np.argmin(np.abs(xs - STRIP_X[strip])))
        return ys, grid[:, idx]
    if strip in STRIP_Y:
        idx = int(np.argmin(np.abs(ys - STRIP_Y[strip])))
        return xs, grid[idx, :]
    raise ValueError(f"Unknown strip {strip!r}.")


def plot_single_stress_profile(output_root: Path, case: str, component: str, strip: str, out_path: Path) -> None:
    column = "sigma_xx" if component == "sigmax" else "sigma_yy"
    ylabel = r"$\sigma_x\ (Pa)$" if component == "sigmax" else r"$\sigma_y\ (Pa)$"
    colors = ["b", "g", "r"]
    labels = ["1 mês", "5 meses", "10 meses"]

    fig, ax = plt.subplots()
    for color, label, month in zip(colors, labels, MONTHS_PROFILE):
        x_values, values = cell_line(output_root, case, month, column, strip)
        ax.plot(
            x_values,
            values,
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )
    ax.set_xlabel(r"$y$" if strip in STRIP_X else r"$x$")
    ax.set_ylabel(ylabel)
    if strip in STRIP_X:
        ax.set_xlim(0.0, LY)
    else:
        ax.set_xlim(0.0, LX)
    setup_curve_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=False, framealpha=1.0)
    reserve_legend_column(fig)
    save_png(fig, out_path)


def plot_stress_pngs(output_root: Path, out_root: Path, cases: list[str]) -> None:
    for case in cases:
        if case[6:8] == "01":
            continue
        for component in ("sigmax", "sigmay"):
            for strip in ("L", "C", "R", "B", "M", "T"):
                plot_single_stress_profile(output_root, case, component, strip, out_root / case / f"{component}_{strip}.png")


def vertical_profile(output_root: Path, case: str, month: float, column: str, strip: str = "C") -> tuple[np.ndarray, np.ndarray]:
    data = load_csv(output_root / case / f"profiles_vertical_m{month_suffix(month)}.csv")
    mask = data["strip"] == strip
    if not np.any(mask):
        raise ValueError(f"No vertical strip {strip!r} in {case} at month {month:g}.")
    return data["y"][mask], data[column][mask]


def plot_depth_comparison(output_root: Path, out_root: Path, column: str, ylabel: str, filename: str, negate: bool = False) -> None:
    shallow_case = "CampoM03bp"
    deep_case = "CampoM03kp"
    month_styles = [
        ("1 mês", "g", "r"),
        ("5 meses", "m", "y"),
        ("10 meses", "b", "g"),
    ]
    dotted = (0, (1.5, 4.0))

    fig, ax = plt.subplots()
    handles: list[Line2D] = []
    for month, (heading, shallow_color, deep_color) in zip(MONTHS_PROFILE, month_styles):
        handles.append(legend_heading(heading))
        y_values, shallow_values = vertical_profile(output_root, shallow_case, month, column)
        _, deep_values = vertical_profile(output_root, deep_case, month, column)
        if negate:
            shallow_values = -shallow_values
            deep_values = -deep_values
        shallow_line = ax.plot(y_values, shallow_values, color=shallow_color, linestyle="-", label=r"$h = 2119\ m$")[0]
        deep_line = ax.plot(y_values, deep_values, color=deep_color, linestyle=dotted, label=r"$h = 2621\ m$")[0]
        handles.extend([shallow_line, deep_line])

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, LY)
    setup_curve_axes(ax)
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=False, framealpha=1.0)
    reserve_legend_column(fig)
    save_pdf(fig, out_root / "CampoM03kp" / filename)


def common_range(values_by_time: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(values, dtype=float).ravel() for values in values_by_time])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= delta
        vmax += delta
    return vmin, vmax


def plot_map(
    data: np.ndarray,
    values: np.ndarray,
    label: str,
    out_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    grid = np.full((len(ys), len(xs)), np.nan)
    x_index = {value: i for i, value in enumerate(xs)}
    y_index = {value: i for i, value in enumerate(ys)}
    for row, value in zip(data, values):
        grid[y_index[row["y"]], x_index[row["x"]]] = value

    fig, ax = plt.subplots()
    levels = np.linspace(vmin, vmax, 81)
    mesh = ax.contourf(xs, ys, grid, levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$x\ (m)$")
    ax.set_ylabel(r"$y\ (m)$")
    ax.tick_params(direction="in", top=True, right=True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label)
    save_png(fig, out_path)


def plot_field_sequences(output_root: Path, out_root: Path) -> None:
    case = "CampoM03kp"
    field_rows = []
    for month in MONTHS_FIELD:
        data = load_csv(output_root / case / f"cell_fields_m{month_suffix(month)}.csv")
        source_rate = data["source_rate"]
        if "source_rate_slightly_compressible" in data.dtype.names:
            compressible_source = data["source_rate_slightly_compressible"]
        else:
            coeffs = pr_coefficients(case)
            z_values = z_factor_np(data["pressure"], coeffs)
            compressible_source = source_rate / (BETA_F * (data["pressure"] / z_values))
        field_rows.append((data, data["sigma_T"], source_rate, compressible_source))

    sigma_t_range = common_range([row[1] for row in field_rows])
    source_range = common_range([row[2] for row in field_rows])
    compressible_source_range = common_range([row[3] for row in field_rows])

    for idx, (data, sigma_t, source_magnitude, compressible_source) in enumerate(field_rows, start=1):
        plot_map(data, sigma_t, r"$Pa$", out_root / case / f"sig_T{idx}.png", *sigma_t_range)
        plot_map(data, source_magnitude, r"$\frac{Pa}{s}$", out_root / case / f"sig_uu{idx}.png", *source_range)
        plot_map(
            data,
            compressible_source,
            r"$\frac{Pa}{s}$",
            out_root / case / f"sig_uuS{idx}.png",
            *compressible_source_range,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 thesis-format plots from Firedrake outputs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/chapter5"))
    parser.add_argument("--plot-root", type=Path, default=Path("outputs/chapter5/thesis_figures/firedrake"))
    parser.add_argument("--cases", nargs="*", default=ALL_CASES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_thesis_rcparams()
    args.plot_root.mkdir(parents=True, exist_ok=True)

    plot_pressure_and_flux(args.output_root, args.plot_root, args.cases)
    plot_production(args.output_root, args.plot_root, "B")
    plot_production(args.output_root, args.plot_root, "M")
    plot_kh(args.plot_root)
    plot_stress_pngs(args.output_root, args.plot_root, args.cases)
    plot_depth_comparison(args.output_root, args.plot_root, "uy", r"$u\ (m)$", "Uycomp_C.pdf", negate=True)
    plot_depth_comparison(args.output_root, args.plot_root, "sigma_yy", r"$\sigma_{yy}\ (Pa)$", "sigmay_C.pdf")
    plot_depth_comparison(args.output_root, args.plot_root, "sigma_xx", r"$\sigma_{xx}\ (Pa)$", "sigmax_C.pdf")
    plot_field_sequences(args.output_root, args.plot_root)

    print(f"Wrote thesis-format Chapter 5 figures to {args.plot_root.resolve()}")


if __name__ == "__main__":
    main()
