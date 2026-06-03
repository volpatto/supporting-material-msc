from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="The ``Function.at`` method is deprecated.*",
    category=FutureWarning,
)


_ORIGINAL_ARGV = sys.argv[:]
_SCRIPT_OPTIONS = {
    "--case",
    "--output-root",
    "--nx",
    "--ny",
    "--months",
    "--dt-months",
    "--snapshot-months",
    "--vtk-interval",
    "--tolerance",
    "--max-picard",
    "--max-fixed-stress",
    "--nonlinear-solver",
    "--max-newton",
    "--newton-rtol",
    "--newton-atol",
    "--newton-stol",
    "--newton-monitor",
    "--gas-saturation",
    "--water-residual",
    "--gas-residual",
    "--gas-viscosity",
    "--gas-molar-mass",
    "--gas-constant",
    "--grain-bulk-modulus",
    "--permeability-law",
    "--quiet",
}


def _argv_without_script_options(argv: list[str]) -> list[str]:
    filtered = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        key = arg.split("=", 1)[0]
        if key in _SCRIPT_OPTIONS:
            i += 1
            if "=" not in arg:
                while i < len(argv) and not argv[i].startswith("-"):
                    i += 1
            continue
        filtered.append(arg)
        i += 1
    return filtered


sys.argv = _argv_without_script_options(sys.argv)

from firedrake import (
    Constant,
    DirichletBC,
    FacetNormal,
    Function,
    FunctionSpace,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    RectangleMesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    as_vector,
    assemble,
    derivative,
    div,
    dot,
    ds,
    dx,
    exp,
    grad,
    inner,
    norm,
    solve,
    sym,
)
from firedrake.output import VTKFile


SECONDS_PER_MONTH = 2.628e6
THESIS_LX = 10.0
THESIS_LY = 50.0
BETA_F = 3.2e-8
DEFAULT_SNAPSHOT_MONTHS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 60.0, 100.0, 120.0]


@dataclass(frozen=True)
class Chapter5Case:
    name: str
    formation: str
    depth: str
    coupling: str
    alpha: float
    young: float
    poisson: float
    p_reservoir: float
    p_well: float
    kappa0: float
    gas_temperature: float
    phi_min: float
    phi_max: float

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lame_lambda(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    @property
    def bulk_modulus(self) -> float:
        return self.lame_lambda + (2.0 / 3.0) * self.shear_modulus


def case_from_name(case_name: str) -> Chapter5Case:
    if len(case_name) != 10 or not case_name.startswith("Campo"):
        raise ValueError("Expected a thesis case name such as CampoM03kp or CampoB01kp.")

    formation_code = case_name[5]
    coupling_code = case_name[6:8]
    depth_code = case_name[8:10]

    if formation_code == "B":
        formation = "Barnett"
        alpha = 0.69
        young = 20.0e9
        phi_min, phi_max = 0.04, 0.05
        kappa0 = 1.5e-19
        temperature = 338.45
        p_reservoir = 28.6e6 if depth_code == "bp" else 40.8e6
    elif formation_code == "M":
        formation = "Marcellus"
        alpha = 0.91
        young = 6.0e9
        phi_min, phi_max = 0.05, 0.08
        kappa0 = 6.0e-19
        temperature = 352.55
        p_reservoir = 52.0e6 if depth_code == "bp" else 64.0e6
    else:
        raise ValueError(f"Unsupported formation code in {case_name!r}.")

    if depth_code not in {"bp", "kp"}:
        raise ValueError(f"Unsupported depth code in {case_name!r}.")

    coupling_by_code = {
        "01": "rigid",
        "02": "oneway",
        "03": "twoway",
    }
    if coupling_code not in coupling_by_code:
        raise ValueError(f"Unsupported coupling code in {case_name!r}.")

    return Chapter5Case(
        name=case_name,
        formation=formation,
        depth="shallow" if depth_code == "bp" else "deep",
        coupling=coupling_by_code[coupling_code],
        alpha=alpha,
        young=young,
        poisson=0.23,
        p_reservoir=p_reservoir,
        p_well=5.0e5,
        kappa0=kappa0,
        gas_temperature=temperature,
        phi_min=phi_min,
        phi_max=phi_max,
    )


# Coefficients for Z(p) = sum_i c_i p^i.  These are the fitted
# Peng-Robinson polynomial coefficients used as model input data.
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
    formation = case_name[5]
    depth = case_name[8:10]
    return list(PR_Z_COEFFICIENTS[(formation, depth)])


def z_factor(pressure, coefficients: list[float]):
    value = 0.0
    for power, coeff in enumerate(coefficients):
        value += coeff * pressure**power
    return value


def effective_relative_permeability(s_g: float, s_wr: float, s_gr: float) -> float:
    s_w = 1.0 - s_g
    s_e = (s_w - s_wr) / (1.0 - s_wr - s_gr)
    return (1.0 - s_e**2) * (1.0 - s_e) ** 2


def initialize_porosity(phi0: Function, case: Chapter5Case) -> None:
    mesh = phi0.function_space().mesh()
    x, y = SpatialCoordinate(mesh)
    smooth = (
        0.5 * (case.phi_min + case.phi_max)
        + 0.5 * (case.phi_max - case.phi_min) * 0.5 * (
            1.0 + 0.45 * (2.0 * x / THESIS_LX - 1.0) + 0.35 * (2.0 * y / THESIS_LY - 1.0)
        )
    )
    fracture_enrichment = (0.1 - case.phi_max) * exp(-x / 0.35)
    phi0.interpolate(smooth + fracture_enrichment)


def pressure_points(nx: int, ny: int) -> np.ndarray:
    x_vals = np.linspace(0.0, THESIS_LX, nx + 1)
    y_vals = np.linspace(0.0, THESIS_LY, ny + 1)
    return np.array([(x, y) for x in x_vals for y in y_vals], dtype=float)


def write_pressure_snapshot(path: Path, pressure: Function, time_value: float, nx: int, ny: int) -> None:
    rows = []
    node_id = 1
    for x_val in np.linspace(0.0, THESIS_LX, nx + 1):
        for y_val in np.linspace(0.0, THESIS_LY, ny + 1):
            rows.append((node_id, time_value, x_val, y_val, pressure.at((float(x_val), float(y_val)))))
            node_id += 1
    np.savetxt(path, np.array(rows), fmt=["%8d", "%24.8E", "%18.8E", "%18.8E", "%18.8E"])


def point_value(function: Function, x_value: float, y_value: float) -> float:
    eps = 1.0e-10
    x_clamped = min(max(float(x_value), eps), THESIS_LX - eps)
    y_clamped = min(max(float(y_value), eps), THESIS_LY - eps)
    return float(function.at((x_clamped, y_clamped)))


def effective_permeability_value(
    phi_value: float,
    phi0_value: float,
    case: Chapter5Case,
    args: argparse.Namespace,
) -> float:
    k_rel = effective_relative_permeability(args.gas_saturation, args.water_residual, args.gas_residual)
    if args.permeability_law == "normalized":
        return k_rel * case.kappa0 * ((3.0 - phi0_value) / (2.0 * phi0_value)) * (
            2.0 * phi_value / (3.0 - phi_value)
        )
    return k_rel * case.kappa0 * (2.0 * phi_value / (3.0 - phi_value))


def effective_permeability_expression(phi, phi0, k_rel, kappa0, args: argparse.Namespace):
    if args.permeability_law == "normalized":
        return k_rel * kappa0 * ((3.0 - phi0) / (2.0 * phi0)) * (2.0 * phi / (3.0 - phi))
    return k_rel * kappa0 * (2.0 * phi / (3.0 - phi))


def strip_profiles(
    pressure: Function,
    porosity: Function,
    phi0: Function,
    case: Chapter5Case,
    args: argparse.Namespace,
    coefficients: list[float],
    npoints: int = 201,
) -> list[tuple[str, float, float, float, float, float, float, float]]:
    rows = []
    x_values = np.linspace(0.0, THESIS_LX, npoints)
    for strip, y_value in (("B", 0.25 * THESIS_LY), ("M", 0.5 * THESIS_LY), ("T", 0.75 * THESIS_LY)):
        p_values = np.array([point_value(pressure, x_value, y_value) for x_value in x_values])
        dpdx_values = np.gradient(p_values, x_values, edge_order=1)
        for x_value, p_value, dpdx_value in zip(x_values, p_values, dpdx_values):
            phi_value = point_value(porosity, x_value, y_value)
            phi0_value = point_value(phi0, x_value, y_value)
            z_value = float(z_factor(p_value, coefficients))
            density_factor = args.gas_molar_mass / (args.gas_constant * case.gas_temperature)
            k_eff = effective_permeability_value(phi_value, phi0_value, case, args)
            vx_value = -(k_eff / args.gas_viscosity) * dpdx_value
            jx_value = -density_factor * (p_value / z_value) * vx_value
            rows.append((strip, x_value, y_value, p_value, vx_value, jx_value, phi_value, phi0_value))
    return rows


def vertical_profiles(
    displacement: Function,
    sigma_xx: Function,
    sigma_yy: Function,
    sigma_t: Function,
    source_rate: Function,
    npoints: int = 201,
) -> list[tuple[str, float, float, float, float, float, float, float, float]]:
    rows = []
    y_values = np.linspace(0.0, THESIS_LY, npoints)
    for strip, x_value in (("L", 0.25 * THESIS_LX), ("C", 0.5 * THESIS_LX), ("R", 0.75 * THESIS_LX)):
        for y_value in y_values:
            ux_value, uy_value = displacement.at((float(x_value), float(y_value)))
            rows.append(
                (
                    strip,
                    x_value,
                    y_value,
                    float(ux_value),
                    float(uy_value),
                    point_value(sigma_xx, x_value, y_value),
                    point_value(sigma_yy, x_value, y_value),
                    point_value(sigma_t, x_value, y_value),
                    point_value(source_rate, x_value, y_value),
                )
            )
    return rows


def cell_field_rows(
    pressure: Function,
    phi: Function,
    sigma_t: Function,
    source_rate: Function,
    source_rate_without_sg: Function,
    source_rate_slightly_compressible: Function,
    dsigma_total_dt: Function,
    sigma_xx: Function,
    sigma_yy: Function,
) -> np.ndarray:
    q0 = phi.function_space()
    x, y = SpatialCoordinate(q0.mesh())
    x_cell = Function(q0).interpolate(x)
    y_cell = Function(q0).interpolate(y)
    pressure_cell = Function(q0).interpolate(pressure)
    return np.column_stack(
        [
            x_cell.dat.data_ro,
            y_cell.dat.data_ro,
            pressure_cell.dat.data_ro,
            phi.dat.data_ro,
            sigma_t.dat.data_ro,
            source_rate.dat.data_ro,
            source_rate_without_sg.dat.data_ro,
            source_rate_slightly_compressible.dat.data_ro,
            dsigma_total_dt.dat.data_ro,
            sigma_xx.dat.data_ro,
            sigma_yy.dat.data_ro,
        ]
    )


def write_snapshot_diagnostics(
    output_dir: Path,
    month: float,
    pressure: Function,
    porosity: Function,
    phi0: Function,
    displacement: Function,
    sigma_xx: Function,
    sigma_yy: Function,
    sigma_t: Function,
    source_rate: Function,
    source_rate_without_sg: Function,
    source_rate_slightly_compressible: Function,
    dsigma_total_dt: Function,
    case: Chapter5Case,
    args: argparse.Namespace,
    coefficients: list[float],
) -> None:
    suffix = f"{month:g}".replace(".", "p")
    strip_rows = strip_profiles(pressure, porosity, phi0, case, args, coefficients)
    np.savetxt(
        output_dir / f"profiles_horizontal_m{suffix}.csv",
        np.array(strip_rows, dtype=object),
        fmt=["%s", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e"],
        delimiter=",",
        header="strip,x,y,pressure,vx,jx,phi,phi0",
        comments="",
    )

    vertical_rows = vertical_profiles(displacement, sigma_xx, sigma_yy, sigma_t, source_rate)
    np.savetxt(
        output_dir / f"profiles_vertical_m{suffix}.csv",
        np.array(vertical_rows, dtype=object),
        fmt=["%s", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e"],
        delimiter=",",
        header="strip,x,y,ux,uy,sigma_xx,sigma_yy,sigma_T,source_rate",
        comments="",
    )

    np.savetxt(
        output_dir / f"cell_fields_m{suffix}.csv",
        cell_field_rows(
            pressure,
            porosity,
            sigma_t,
            source_rate,
            source_rate_without_sg,
            source_rate_slightly_compressible,
            dsigma_total_dt,
            sigma_xx,
            sigma_yy,
        ),
        fmt="%.10e",
        delimiter=",",
        header=(
            "x,y,pressure,phi,sigma_T,source_rate,"
            "source_rate_without_sg,source_rate_slightly_compressible,"
            "dsigma_total_dt,sigma_xx,sigma_yy"
        ),
        comments="",
    )


def sample_midheight_profile(pressure: Function, npoints: int = 201) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, THESIS_LX, npoints)
    values = np.array([pressure.at((float(x), THESIS_LY / 2.0)) for x in xs], dtype=float)
    return xs, values


def plot_midheight_profiles(
    output_path: Path,
    profiles: list[tuple[float, np.ndarray, np.ndarray]],
    case: Chapter5Case,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    for month, xs, values in profiles:
        ax.plot(xs, values / 1.0e6, marker="o", markersize=2.5, linewidth=1.4, label=f"{month:g} mo")

    ax.set_title(f"{case.name}: mid-height pressure profiles")
    ax.set_xlabel("x at y = 25 m")
    ax.set_ylabel("Pressure [MPa]")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def epsilon(u):
    return sym(grad(u))


def clean_generated_outputs(output_dir: Path) -> None:
    generated_patterns = [
        "production.csv",
        "run_metadata.txt",
        "midheight_profiles.png",
        "fields.pvd",
        "fieldP.*",
        "cell_fields_m*.csv",
        "profiles_horizontal_m*.csv",
        "profiles_vertical_m*.csv",
    ]
    for pattern in generated_patterns:
        for path in output_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
    fields_dir = output_dir / "fields"
    if fields_dir.is_dir():
        shutil.rmtree(fields_dir)


def run_fixed_stress(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = case_from_name(args.case)
    coefficients = pr_coefficients(case.name)

    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    output_dir = output_root / case.name
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(output_dir)

    mesh = RectangleMesh(args.nx, args.ny, THESIS_LX, THESIS_LY, quadrilateral=True)
    normal = FacetNormal(mesh)
    v_space = FunctionSpace(mesh, "CG", 1)
    q0_space = FunctionSpace(mesh, "DG", 0)
    w_space = VectorFunctionSpace(mesh, "CG", 1)

    p_trial = TrialFunction(v_space)
    v_test = TestFunction(v_space)
    p = Function(v_space, name="pressure")
    p_n = Function(v_space, name="pressure_previous")
    p_iter = Function(v_space, name="pressure_picard")

    phi0 = Function(q0_space, name="phi0")
    initialize_porosity(phi0, case)

    phi = Function(q0_space, name="porosity")
    phi_n = Function(q0_space, name="porosity_previous")
    phi_iter = Function(q0_space, name="porosity_picard")
    inv_n = Function(q0_space, name="inv_N")
    beta_r = Function(q0_space, name="beta_r")
    sigma_t = Function(q0_space, name="sigma_T")
    sigma_n = Function(q0_space, name="sigma_T_previous")
    sigma_iter = Function(q0_space, name="sigma_T_picard")
    sigma_xx = Function(q0_space, name="sigma_xx")
    sigma_yy = Function(q0_space, name="sigma_yy")
    source_rate = Function(q0_space, name="fixed_stress_source_rate")
    source_rate_without_sg = Function(q0_space, name="fixed_stress_source_rate_without_sg")
    source_rate_slightly_compressible = Function(q0_space, name="fixed_stress_source_rate_slightly_compressible")
    dsigma_total_dt = Function(q0_space, name="dsigma_total_dt")
    u = Function(w_space, name="displacement")

    p.assign(case.p_reservoir)
    p_n.assign(case.p_reservoir)
    p_iter.assign(case.p_reservoir)
    phi.assign(phi0)
    phi_n.assign(phi0)
    phi_iter.assign(phi0)
    if case.coupling == "twoway":
        sigma_n.assign(-case.alpha * case.p_reservoir)
        sigma_iter.assign(sigma_n)

    inv_n.interpolate((Constant(case.alpha) - phi0) / Constant(args.grain_bulk_modulus))
    beta_r.interpolate(inv_n + Constant(case.alpha**2 / case.bulk_modulus))

    p_bcs = [DirichletBC(v_space, case.p_well, 1)]
    u_bcs = [
        DirichletBC(w_space.sub(1), 0.0, 3),
        DirichletBC(w_space.sub(0), 0.0, 2),
    ]

    du = TrialFunction(w_space)
    w_test = TestFunction(w_space)
    mu_s = Constant(case.shear_modulus)
    lam_s = Constant(case.lame_lambda)
    traction = as_vector((0.0, -case.p_reservoir))
    a_elasticity = (
        2.0 * mu_s * inner(epsilon(du), epsilon(w_test)) * dx
        + lam_s * div(du) * div(w_test) * dx
    )

    pressure_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    pressure_newton_parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": args.newton_rtol,
        "snes_atol": args.newton_atol,
        "snes_stol": args.newton_stol,
        "snes_max_it": args.max_newton,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    if args.newton_monitor:
        pressure_newton_parameters["snes_monitor"] = None
    mechanics_solver = {"ksp_type": "preonly", "pc_type": "lu"}

    s_g = Constant(args.gas_saturation)
    alpha = Constant(case.alpha)
    bulk = Constant(case.bulk_modulus)
    mu_g = Constant(args.gas_viscosity)
    k_rel = Constant(effective_relative_permeability(args.gas_saturation, args.water_residual, args.gas_residual))
    kappa0 = Constant(case.kappa0)
    dt = Constant(args.dt_months * SECONDS_PER_MONTH)

    pressure_newton_solver = None
    if args.nonlinear_solver == "newton":
        storage_newton = beta_r if case.coupling != "rigid" else Constant(0.0)
        stress_increment_newton = sigma_iter - sigma_n if case.coupling == "twoway" else Constant(0.0)
        permeability_newton = effective_permeability_expression(phi_iter, phi0, k_rel, kappa0, args)
        pz = p / z_factor(p, coefficients)
        pz_n = p_n / z_factor(p_n, coefficients)
        f_pressure = (
            s_g * phi_iter * (pz - pz_n) * v_test * dx
            + s_g * storage_newton * pz * (p - p_n) * v_test * dx
            + dt * (permeability_newton / mu_g) * pz * inner(grad(p), grad(v_test)) * dx
            + s_g * pz * (alpha / bulk) * stress_increment_newton * v_test * dx
        )
        pressure_newton_problem = NonlinearVariationalProblem(
            f_pressure,
            p,
            bcs=p_bcs,
            J=derivative(f_pressure, p),
        )
        pressure_newton_solver = NonlinearVariationalSolver(
            pressure_newton_problem,
            solver_parameters=pressure_newton_parameters,
        )

    outfile = VTKFile(str(output_dir / "fields.pvd"))
    outfile.write(
        p,
        phi,
        sigma_t,
        source_rate,
        source_rate_without_sg,
        source_rate_slightly_compressible,
        dsigma_total_dt,
        sigma_xx,
        sigma_yy,
        u,
        time=0.0,
    )

    total_steps = int(round(args.months / args.dt_months))
    snapshot_months = {float(month) for month in args.snapshot_months}
    profiles: list[tuple[float, np.ndarray, np.ndarray]] = []
    density_factor_value = args.gas_molar_mass / (args.gas_constant * case.gas_temperature)
    initial_inventory = float(assemble(args.gas_saturation * phi0 * (p_n / z_factor(p_n, coefficients)) * dx))
    initial_physical_inventory_kg_per_m = density_factor_value * initial_inventory
    density_factor = Constant(density_factor_value)
    cumulative_produced_kg = 0.0
    cumulative_produced_kg_trapezoid = 0.0
    previous_integral_production_rate_kg_s = 0.0
    production_rows = []

    metadata = output_dir / "run_metadata.txt"
    metadata.write_text(
        "\n".join(
            [
                f"case={case.name}",
                f"formation={case.formation}",
                f"depth={case.depth}",
                f"coupling={case.coupling}",
                f"mesh={args.nx}x{args.ny}",
                f"months={args.months}",
                f"dt_months={args.dt_months}",
                f"output_root={output_root}",
                "initial_porosity=analytic_smooth_boundary_enriched",
                f"nonlinear_solver={args.nonlinear_solver}",
                f"max_fixed_stress_iterations={args.max_picard}",
                f"newton_max_iterations={args.max_newton}",
                f"newton_rtol={args.newton_rtol}",
                f"newton_atol={args.newton_atol}",
                f"newton_stol={args.newton_stol}",
                f"alpha={case.alpha}",
                f"E={case.young}",
                f"nu={case.poisson}",
                f"K_bulk={case.bulk_modulus}",
                f"kappa0={case.kappa0}",
                f"permeability_law={args.permeability_law}",
                f"p_reservoir={case.p_reservoir}",
                f"p_well={case.p_well}",
                f"gas_molar_mass={args.gas_molar_mass}",
                f"gas_constant={args.gas_constant}",
                f"initial_physical_inventory_kg_per_m={initial_physical_inventory_kg_per_m}",
                f"production_definition=integral_-rho_K_over_mu_gradp_dot_n_ds_on_fracture_boundary_no_Sg",
                f"source_rate_definition=-Sg_p_over_Z_alpha_over_K_dsigmaT_dt",
                f"source_rate_without_sg_definition=-p_over_Z_alpha_over_K_dsigmaT_dt",
            ]
        )
        + "\n"
    )

    print(f"Running {case.name} ({case.coupling}) on {args.nx}x{args.ny} cells")
    print(f"Nonlinear pressure backend: {args.nonlinear_solver}")
    print("Initial porosity: analytic_smooth_boundary_enriched")
    print(f"Bulk modulus K = {case.bulk_modulus:.6e} Pa")

    for step in range(1, total_steps + 1):
        time_seconds = float(step * args.dt_months * SECONDS_PER_MONTH)
        time_months = step * args.dt_months
        p_iter.assign(p_n)
        phi_iter.assign(phi_n)
        sigma_iter.assign(sigma_n)

        converged = False
        last_pressure_newton_iterations = 0
        for iteration in range(1, args.max_picard + 1):
            if args.nonlinear_solver == "picard":
                z_iter = z_factor(p_iter, coefficients)
                pz_iter = p_iter / z_iter
                pz_n = p_n / z_factor(p_n, coefficients)
                permeability = effective_permeability_expression(phi_iter, phi0, k_rel, kappa0, args)
                storage = beta_r if case.coupling != "rigid" else Constant(0.0)
                stress_increment = sigma_iter - sigma_n if case.coupling == "twoway" else Constant(0.0)

                a_pressure = (
                    s_g * phi_iter * (p_trial / z_iter) * v_test * dx
                    + s_g * storage * pz_iter * p_trial * v_test * dx
                    + dt * (permeability / mu_g) * pz_iter * inner(grad(p_trial), grad(v_test)) * dx
                )
                l_pressure = (
                    s_g * phi_iter * pz_n * v_test * dx
                    + s_g * storage * pz_iter * p_n * v_test * dx
                    - s_g * pz_iter * (alpha / bulk) * stress_increment * v_test * dx
                )
                solve(a_pressure == l_pressure, p, bcs=p_bcs, solver_parameters=pressure_solver)
                last_pressure_newton_iterations = 0
            else:
                p.assign(p_iter)
                pressure_newton_solver.solve()
                last_pressure_newton_iterations = int(pressure_newton_solver.snes.getIterationNumber())

            pressure_change = norm(p - p_iter, mesh=mesh) / max(norm(p, mesh=mesh), 1.0)

            if case.coupling != "rigid":
                l_elasticity = (
                    dot(w_test, traction) * ds(4)
                    + alpha * p * div(w_test) * dx
                )
                solve(a_elasticity == l_elasticity, u, bcs=u_bcs, solver_parameters=mechanics_solver)
                sigma_t.interpolate(bulk * div(u) - alpha * p)
                sigma_xx.interpolate(lam_s * div(u) + 2.0 * mu_s * epsilon(u)[0, 0])
                sigma_yy.interpolate(lam_s * div(u) + 2.0 * mu_s * epsilon(u)[1, 1])
                phi.interpolate(phi0 + alpha * div(u) + inv_n * (p - case.p_reservoir))
            else:
                u.assign(0.0)
                sigma_t.assign(0.0)
                sigma_xx.assign(0.0)
                sigma_yy.assign(0.0)
                phi.assign(phi0)

            phi_change = norm(phi - phi_iter, mesh=mesh) / max(norm(phi, mesh=mesh), 1.0)
            sigma_change = norm(sigma_t - sigma_iter, mesh=mesh) / max(norm(sigma_t, mesh=mesh), 1.0)

            p_iter.assign(p)
            phi_iter.assign(phi)
            sigma_iter.assign(sigma_t)

            if not args.quiet:
                backend_log = (
                    f" newton={last_pressure_newton_iterations:02d}"
                    if args.nonlinear_solver == "newton"
                    else ""
                )
                print(
                    f"    it={iteration:02d} "
                    f"dp={pressure_change:.3e} dphi={phi_change:.3e} dsigma={sigma_change:.3e}"
                    f"{backend_log}"
                )

            if max(pressure_change, phi_change, sigma_change) < args.tolerance:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"Fixed-stress iteration failed at step {step}, month {time_months:g}, "
                f"after {args.max_picard} iterations with {args.nonlinear_solver!r} pressure backend."
            )

        if case.coupling == "twoway":
            dsigma_total_dt.interpolate((sigma_t - sigma_n) / dt)
            source_rate.interpolate(
                -s_g * (p / z_factor(p, coefficients)) * (alpha / bulk) * dsigma_total_dt
            )
            source_rate_without_sg.interpolate(
                -(p / z_factor(p, coefficients)) * (alpha / bulk) * dsigma_total_dt
            )
            source_rate_slightly_compressible.interpolate(
                -s_g * (1.0 / BETA_F) * (alpha / bulk) * dsigma_total_dt
            )
        else:
            dsigma_total_dt.assign(0.0)
            source_rate.assign(0.0)
            source_rate_without_sg.assign(0.0)
            source_rate_slightly_compressible.assign(0.0)

        p_n.assign(p)
        phi_n.assign(phi)
        sigma_n.assign(sigma_t)

        if step % args.vtk_interval == 0 or step == total_steps:
            outfile.write(
                p,
                phi,
                sigma_t,
                source_rate,
                source_rate_without_sg,
                source_rate_slightly_compressible,
                dsigma_total_dt,
                sigma_xx,
                sigma_yy,
                u,
                time=time_seconds,
            )

        if any(abs(time_months - month) < 0.5 * args.dt_months for month in snapshot_months):
            suffix = int(round(time_months))
            write_pressure_snapshot(output_dir / f"fieldP.{suffix}", p, time_seconds, args.nx, args.ny)
            profiles.append((time_months, *sample_midheight_profile(p)))
            write_snapshot_diagnostics(
                output_dir,
                time_months,
                p,
                phi,
                phi0,
                u,
                sigma_xx,
                sigma_yy,
                sigma_t,
                source_rate,
                source_rate_without_sg,
                source_rate_slightly_compressible,
                dsigma_total_dt,
                case,
                args,
                coefficients,
            )

        inventory = float(assemble(args.gas_saturation * phi * (p / z_factor(p, coefficients)) * dx))
        produced_fraction = (initial_inventory - inventory) / initial_inventory
        output_permeability = effective_permeability_expression(phi, phi0, k_rel, kappa0, args)
        production_rate_kg_s = float(
            assemble(
                -density_factor
                * (p / z_factor(p, coefficients))
                * (output_permeability / mu_g)
                * dot(grad(p), normal)
                * ds(1)
            )
        )
        boundary_mass_flux_kg_m2_s = production_rate_kg_s / THESIS_LY
        cumulative_produced_kg += production_rate_kg_s * args.dt_months * SECONDS_PER_MONTH
        cumulative_produced_kg_trapezoid += (
            0.5
            * (previous_integral_production_rate_kg_s + production_rate_kg_s)
            * args.dt_months
            * SECONDS_PER_MONTH
        )
        previous_integral_production_rate_kg_s = production_rate_kg_s
        physical_inventory_kg_per_m = float(
            assemble(args.gas_saturation * phi * density_factor * (p / z_factor(p, coefficients)) * dx)
        )
        physical_inventory_change_kg_per_m = initial_physical_inventory_kg_per_m - physical_inventory_kg_per_m
        production_rows.append(
            [
                step,
                time_months,
                time_seconds,
                boundary_mass_flux_kg_m2_s,
                production_rate_kg_s,
                cumulative_produced_kg,
                cumulative_produced_kg_trapezoid,
                initial_physical_inventory_kg_per_m,
                physical_inventory_kg_per_m,
                physical_inventory_change_kg_per_m,
                inventory,
                produced_fraction,
                float(p.dat.data_ro.min()),
                float(p.dat.data_ro.max()),
                float(phi.dat.data_ro.min()),
                float(phi.dat.data_ro.max()),
                iteration,
                last_pressure_newton_iterations,
            ]
        )

        solver_log = (
            f"picard={iteration:02d}"
            if args.nonlinear_solver == "picard"
            else f"fixed_stress={iteration:02d} newton={last_pressure_newton_iterations:02d}"
        )
        print(
            f"step={step:04d} month={time_months:8.3f} "
            f"{solver_log} p=[{p.dat.data_ro.min():.4e}, {p.dat.data_ro.max():.4e}] "
            f"phi=[{phi.dat.data_ro.min():.4e}, {phi.dat.data_ro.max():.4e}]"
        )

    if profiles:
        plot_midheight_profiles(output_dir / "midheight_profiles.png", profiles, case)

    if production_rows:
        production_format = ["%d"] + ["%.10e"] * (len(production_rows[0]) - 3) + ["%d", "%d"]
        np.savetxt(
            output_dir / "production.csv",
            np.array(production_rows),
            fmt=production_format,
            delimiter=",",
            header=(
                "step,month,time_seconds,boundary_mass_flux_kg_m2_s,"
                "production_rate_kg_s,cumulative_produced_kg,"
                "cumulative_produced_kg_trapezoid,"
                "initial_physical_inventory_kg_per_m,"
                "physical_inventory_kg_per_m,"
                "physical_inventory_change_kg_per_m,"
                "inventory,produced_fraction,"
                "p_min,p_max,phi_min,phi_max,"
                "fixed_stress_iterations,pressure_newton_iterations"
            ),
            comments="",
        )

    print(f"Wrote outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate Chapter 5 shale-gas fixed-stress cases in Firedrake."
    )
    parser.add_argument("--case", default="CampoM03kp", help="Thesis case folder, e.g. CampoM03kp.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/chapter5"),
        help="Root folder where case output directories are written.",
    )
    parser.add_argument("--nx", type=int, default=40, help="Number of cells in x.")
    parser.add_argument("--ny", type=int, default=80, help="Number of cells in y.")
    parser.add_argument("--months", type=float, default=180.0, help="Final time in thesis months.")
    parser.add_argument("--dt-months", type=float, default=1.0, help="Time-step size in thesis months.")
    parser.add_argument("--snapshot-months", type=float, nargs="*", default=DEFAULT_SNAPSHOT_MONTHS)
    parser.add_argument("--vtk-interval", type=int, default=30, help="Write VTK every N time steps.")
    parser.add_argument("--tolerance", type=float, default=1.0e-5, help="Fixed-stress relative tolerance.")
    parser.add_argument(
        "--max-picard",
        "--max-fixed-stress",
        dest="max_picard",
        type=int,
        default=40,
        help="Maximum fixed-stress outer iterations.",
    )
    parser.add_argument(
        "--nonlinear-solver",
        choices=["picard", "newton"],
        default="picard",
        help="Pressure nonlinear backend inside each fixed-stress iteration.",
    )
    parser.add_argument("--max-newton", type=int, default=20, help="Maximum SNES iterations for the Newton pressure backend.")
    parser.add_argument("--newton-rtol", type=float, default=1.0e-8, help="SNES relative tolerance for Newton.")
    parser.add_argument("--newton-atol", type=float, default=1.0e-10, help="SNES absolute tolerance for Newton.")
    parser.add_argument("--newton-stol", type=float, default=1.0e-10, help="SNES step tolerance for Newton.")
    parser.add_argument("--newton-monitor", action="store_true", help="Enable PETSc SNES monitor for Newton solves.")
    parser.add_argument("--gas-saturation", type=float, default=0.90)
    parser.add_argument("--water-residual", type=float, default=0.05)
    parser.add_argument("--gas-residual", type=float, default=0.05)
    parser.add_argument("--gas-viscosity", type=float, default=1.2e-5)
    parser.add_argument("--gas-molar-mass", type=float, default=1.604246e-2, help="Methane molar mass [kg/mol].")
    parser.add_argument("--gas-constant", type=float, default=8.31446261815324, help="Universal gas constant [J/(mol K)].")
    parser.add_argument("--grain-bulk-modulus", type=float, default=40.0e9)
    parser.add_argument(
        "--permeability-law",
        choices=["normalized", "tortuosity"],
        default="normalized",
        help=(
            "normalized keeps kappa0 as the initial permeability scale; "
            "tortuosity uses only 2*phi/(3-phi)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-Picard iteration residual logs.")
    args, _ = parser.parse_known_args(_ORIGINAL_ARGV[1:])
    return args


if __name__ == "__main__":
    run_fixed_stress(parse_args())
