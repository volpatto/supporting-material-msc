import os
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    RectangleMesh,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    as_vector,
    div,
    dot,
    ds,
    dx,
    grad,
    inner,
    norm,
    project,
    solve,
    sym,
)
from firedrake.output import VTKFile

import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings(
    "ignore",
    message="The ``Function.at`` method is deprecated.*",
    category=FutureWarning,
)


# -----------------------------------------------------------------------------
# Geometry, mesh, and time grid
# -----------------------------------------------------------------------------
numel_x = 100
numel_y = 100
Lx, Ly = 100.0, 5.0
mesh = RectangleMesh(numel_x, numel_y, Lx, Ly, quadrilateral=True)

SECONDS_PER_DAY = 86400.0
T_days = 365.0
dt_days = 5.0
T_total = T_days * SECONDS_PER_DAY
dt_seconds = dt_days * SECONDS_PER_DAY
dt = Constant(dt_seconds)


# -----------------------------------------------------------------------------
# Function spaces
# -----------------------------------------------------------------------------
degree = 1
V = FunctionSpace(mesh, "CG", degree)
Q0 = FunctionSpace(mesh, "DG", 0)
W = VectorFunctionSpace(mesh, "CG", degree)


# -----------------------------------------------------------------------------
# Fluid, rock, and reservoir data
# -----------------------------------------------------------------------------
bar = 1.0e5

# Methane viscosity; the fitted Peng-Robinson polynomial below supplies Z(p).
gas_viscosity_value = 1.2e-5
gas_viscosity = Constant(gas_viscosity_value)

# I use 5% immobile water and 90% mobile gas. A water saturation of 0.95 would
# leave only 5% gas and the Corey gas relative permeability would be nearly zero.
gas_saturation = 0.90
water_residual = 0.05
gas_residual = 0.05

# Marcellus-like poromechanical parameters from the thesis cases.
young_modulus = 6.0e9
poisson_ratio = 0.23
grain_bulk_modulus = Constant(40.0e9)
alpha_biot_value = 0.91
initial_porosity_value = 0.08
initial_permeability_value = 6.0e-19
alpha_biot = Constant(alpha_biot_value)
initial_permeability = Constant(initial_permeability_value)
shear_modulus_value = young_modulus / (2.0 * (1.0 + poisson_ratio))
lame_lambda_value = (
    young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
)
bulk_modulus_value = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))

shear_modulus = Constant(shear_modulus_value)
lame_lambda = Constant(lame_lambda_value)
bulk_modulus = Constant(bulk_modulus_value)

# Chapter 5 uses the same p0 as initial pressure and overburden traction scale:
# t_t = -p0 n. Here p0 is inferred from a 2500 m lithostatic load.
rock_density = 2500.0
gravity = 9.81
depth = 2500.0
overburden_stress = rock_density * gravity * depth
p0 = overburden_stress
p_reservoir = p0
injection_overpressure = 10.0e6
p_well = p0 + injection_overpressure
top_traction = as_vector((0.0, -p0))


# -----------------------------------------------------------------------------
# Peng-Robinson compressibility factor, fitted polynomial Z(p)
# -----------------------------------------------------------------------------
def Z(p):
    # Same fitted Peng-Robinson polynomial used in compressible_hm_2D_transient_nl.py.
    zcoef = np.zeros(10)
    zcoef[0] = 9.99921144125617722409060661448165774345397949218750e-01
    zcoef[1] = -1.19919829040115086206268166821135856547897446944262e-08
    zcoef[2] = 2.95290097079410864772724180594317997860333752145959e-16
    zcoef[3] = 9.42231835327529024372629441226386885009463920848079e-24
    zcoef[4] = -2.46929568577390678055712987160325876321312377285599e-31
    zcoef[5] = -7.40016953399249021667823040632784926032215181166411e-39
    zcoef[6] = 4.21756086831535086775256556143298098377555103824524e-46
    zcoef[7] = -7.90995787006734393072263251251053413138923617742665e-54
    zcoef[8] = 6.96584174374744927048653426883823314263335677413527e-62
    zcoef[9] = -2.42926517319393920651606665298459161434867679132808e-70

    value = 0.0
    for i in range(len(zcoef)):
        value += zcoef[i] * p**i
    return value


def epsilon(u):
    return sym(grad(u))


def point_value(function, x_value, y_value):
    eps = 1.0e-10
    x_clamped = min(max(float(x_value), eps), Lx - eps)
    y_clamped = min(max(float(y_value), eps), Ly - eps)
    return float(function.at((x_clamped, y_clamped)))


def vector_point_value(function, x_value, y_value):
    eps = 1.0e-10
    x_clamped = min(max(float(x_value), eps), Lx - eps)
    y_clamped = min(max(float(y_value), eps), Ly - eps)
    return np.asarray(function.at((x_clamped, y_clamped)), dtype=float)


def sample_field(function, x_grid, y_grid):
    values = np.zeros_like(x_grid)
    for i in range(y_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            values[i, j] = point_value(function, x_grid[i, j], y_grid[i, j])
    return values


def sample_vector_field(function, x_grid, y_grid):
    first_value = vector_point_value(function, x_grid[0, 0], y_grid[0, 0])
    values = np.zeros(x_grid.shape + first_value.shape, dtype=float)
    for i in range(y_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            values[i, j, :] = vector_point_value(function, x_grid[i, j], y_grid[i, j])
    return values


def add_displacement_quiver(ax, ux_values, uy_values):
    stride_x = 20
    stride_y = 10
    xq = XX[::stride_y, ::stride_x]
    yq = YY[::stride_y, ::stride_x]
    uxq = ux_values[::stride_y, ::stride_x]
    uyq = uy_values[::stride_y, ::stride_x]
    max_arrow = float(np.max(np.hypot(uxq, uyq)))
    if max_arrow <= 0.0:
        return
    target_length = 0.35 * Ly
    ax.quiver(
        xq,
        yq,
        uxq,
        uyq,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=max_arrow / target_length,
        pivot="mid",
        width=0.0025,
        headwidth=3.5,
        headlength=4.5,
        alpha=0.90,
    )


def midheight_profile(pressure, sigma_total, source):
    x_values = np.linspace(0.0, Lx, 251)
    y_value = 0.5 * Ly
    pressure_values = np.array([point_value(pressure, x, y_value) for x in x_values])
    sigma_values = np.array([point_value(sigma_total, x, y_value) for x in x_values])
    source_values = np.array([point_value(source, x, y_value) for x in x_values])
    return x_values, pressure_values, sigma_values, source_values


def midheight_profile_projected(pressure, sigma_total, source):
    # Visualization only: the model still computes sigma_T and source_rate in DG0.
    sigma_visual = project(sigma_total, V)
    source_visual = project(source, V)
    return midheight_profile(pressure, sigma_visual, source_visual)


# -----------------------------------------------------------------------------
# Fields and initial state
# -----------------------------------------------------------------------------
p_trial = TrialFunction(V)
v = TestFunction(V)
p = Function(V, name="pressure")
p_n = Function(V, name="pressure_previous")
p_iter = Function(V, name="pressure_picard")

du = TrialFunction(W)
w = TestFunction(W)
u = Function(W, name="displacement")
u_reference = Function(W, name="reference_displacement")
u_increment = Function(W, name="injection_induced_displacement")

phi0 = Function(Q0, name="initial_porosity")
phi = Function(Q0, name="porosity")
phi_n = Function(Q0, name="porosity_previous")
phi_iter = Function(Q0, name="porosity_picard")
inv_n = Function(Q0, name="inverse_biot_modulus")
beta_r = Function(Q0, name="fixed_stress_storage")
div_u_reference = Function(Q0, name="reference_volumetric_strain")

sigma_t = Function(Q0, name="sigma_T")
sigma_n = Function(Q0, name="sigma_T_previous")
sigma_iter = Function(Q0, name="sigma_T_picard")
dsigma_total_dt = Function(Q0, name="dsigma_T_dt")
source_rate = Function(Q0, name="fixed_stress_source_rate")

p.assign(p_reservoir)
p_n.assign(p_reservoir)
p_iter.assign(p_reservoir)

phi0.assign(initial_porosity_value)
phi.assign(phi0)
phi_n.assign(phi0)
phi_iter.assign(phi0)
inv_n.interpolate((alpha_biot - phi0) / grain_bulk_modulus)
beta_r.interpolate(inv_n + alpha_biot**2 / bulk_modulus)


# -----------------------------------------------------------------------------
# Boundary conditions
#
# Firedrake RectangleMesh boundary labels:
#   1: x = 0, 2: x = Lx, 3: y = 0, 4: y = Ly.
# -----------------------------------------------------------------------------
pressure_bcs = [
    DirichletBC(V, p_well, 1),
]
mechanics_bcs = [
    DirichletBC(W.sub(1), 0.0, 3),
    DirichletBC(W.sub(0), 0.0, 2),
]


# -----------------------------------------------------------------------------
# Mechanics problem and initial geostatic reference state.
#
# The reservoir starts from uniform p0. The initial displacement is the elastic
# equilibrium induced by that pressure and the top traction -p0 n; subsequent
# porosity/source terms use this state as the geostatic reference.
# -----------------------------------------------------------------------------
a_elasticity = (
    2.0 * shear_modulus * inner(epsilon(du), epsilon(w)) * dx
    + lame_lambda * div(du) * div(w) * dx
)
l_elasticity = dot(w, top_traction) * ds(4) + alpha_biot * p * div(w) * dx

mechanics_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}
solve(a_elasticity == l_elasticity, u, bcs=mechanics_bcs, solver_parameters=mechanics_solver_parameters)

u_reference.assign(u)
u_increment.interpolate(u - u_reference)
div_u_reference.interpolate(div(u_reference))
sigma_t.interpolate(bulk_modulus * div(u) - alpha_biot * p)
sigma_n.assign(sigma_t)
sigma_iter.assign(sigma_t)


# -----------------------------------------------------------------------------
# Fixed-stress pressure solve data
# -----------------------------------------------------------------------------
def effective_relative_permeability(s_g, s_wr, s_gr):
    s_w = 1.0 - s_g
    s_e = (s_w - s_wr) / (1.0 - s_wr - s_gr)
    s_e = min(max(s_e, 0.0), 1.0)
    return (1.0 - s_e**2) * (1.0 - s_e) ** 2


k_rel_value = effective_relative_permeability(gas_saturation, water_residual, gas_residual)
k_rel = Constant(k_rel_value)
gas_saturation_constant = Constant(gas_saturation)


def effective_permeability(phi_current):
    return (
        k_rel
        * initial_permeability
        * ((3.0 - phi0) / (2.0 * phi0))
        * (2.0 * phi_current / (3.0 - phi_current))
    )


pressure_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}

fixed_stress_tolerance = 1.0e-5
max_fixed_stress_iterations = 40


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[2]
output_dir = repo_root / "outputs" / "2D" / "compressible_hm_2D_fixed_stress_injection"
output_dir.mkdir(parents=True, exist_ok=True)
for output_file in (
    "fields.pvd",
    "final_fields.png",
    "field_evolution.png",
    "midheight_profiles.png",
    "midheight_profiles_projected.png",
    "run_metadata.txt",
    "time_history.csv",
):
    path = output_dir / output_file
    if path.exists():
        path.unlink()
fields_dir = output_dir / "fields"
if fields_dir.exists():
    shutil.rmtree(fields_dir)

outfile = VTKFile(str(output_dir / "fields.pvd"))
outfile.write(p, phi, sigma_t, source_rate, u, u_increment, time=0.0)

x_plot = np.linspace(0.0, Lx, 301)
y_plot = np.linspace(0.0, Ly, 61)
XX, YY = np.meshgrid(x_plot, y_plot)

profile_snapshots = []
projected_profile_snapshots = []
profile_days = {5.0, 30.0, 180.0, 365.0}
field_snapshots = []
history_rows = []


# -----------------------------------------------------------------------------
# Time loop
# -----------------------------------------------------------------------------
total_steps = int(round(T_total / dt_seconds))
t = dt_seconds
step = 0
while step < total_steps:
    step += 1
    time_days = t / SECONDS_PER_DAY

    p_iter.assign(p_n)
    phi_iter.assign(phi_n)
    sigma_iter.assign(sigma_n)

    converged = False
    for iteration in range(1, max_fixed_stress_iterations + 1):
        z_iter = Z(p_iter)
        pz_iter = p_iter / z_iter
        pz_n = p_n / Z(p_n)
        permeability = effective_permeability(phi_iter)
        stress_increment = sigma_iter - sigma_n

        a_pressure = (
            gas_saturation_constant * phi_iter * (p_trial / z_iter) * v * dx
            + gas_saturation_constant * beta_r * pz_iter * p_trial * v * dx
            + dt * (permeability / gas_viscosity) * pz_iter * inner(grad(p_trial), grad(v)) * dx
        )
        l_pressure = (
            gas_saturation_constant * phi_iter * pz_n * v * dx
            + gas_saturation_constant * beta_r * pz_iter * p_n * v * dx
            - gas_saturation_constant * pz_iter * (alpha_biot / bulk_modulus) * stress_increment * v * dx
        )

        solve(a_pressure == l_pressure, p, bcs=pressure_bcs, solver_parameters=pressure_solver_parameters)
        pressure_change = norm(p - p_iter, mesh=mesh) / max(norm(p, mesh=mesh), 1.0)

        l_elasticity = dot(w, top_traction) * ds(4) + alpha_biot * p * div(w) * dx
        solve(a_elasticity == l_elasticity, u, bcs=mechanics_bcs, solver_parameters=mechanics_solver_parameters)
        u_increment.interpolate(u - u_reference)

        # Porosity evolves from the mechanically equilibrated reservoir state.
        # sigma_T itself is kept absolute, so sigma_T - sigma_T_previous is the
        # stress increment used by the fixed-stress source term.
        phi.interpolate(phi0 + alpha_biot * (div(u) - div_u_reference) + inv_n * (p - p_reservoir))
        sigma_t.interpolate(bulk_modulus * div(u) - alpha_biot * p)

        phi_change = norm(phi - phi_iter, mesh=mesh) / max(norm(phi, mesh=mesh), 1.0)
        sigma_change = norm(sigma_t - sigma_iter, mesh=mesh) / max(norm(sigma_t, mesh=mesh), 1.0)

        p_iter.assign(p)
        phi_iter.assign(phi)
        sigma_iter.assign(sigma_t)

        if max(pressure_change, phi_change, sigma_change) < fixed_stress_tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Fixed-stress iteration failed at day {time_days:g} after "
            f"{max_fixed_stress_iterations} iterations."
        )

    dsigma_total_dt.interpolate((sigma_t - sigma_n) / dt)
    source_rate.interpolate(
        -gas_saturation_constant * (p / Z(p)) * (alpha_biot / bulk_modulus) * dsigma_total_dt
    )

    p_n.assign(p)
    phi_n.assign(phi)
    sigma_n.assign(sigma_t)

    if step % 5 == 0 or step == total_steps:
        outfile.write(p, phi, sigma_t, source_rate, u, u_increment, time=t)

    if any(abs(time_days - snapshot_day) < 0.5 * dt_days for snapshot_day in profile_days):
        profile_snapshots.append((time_days, *midheight_profile(p, sigma_t, source_rate)))
        projected_profile_snapshots.append((time_days, *midheight_profile_projected(p, sigma_t, source_rate)))
        field_snapshots.append(
            {
                "time_days": time_days,
                "pressure": sample_field(p, XX, YY) / 1.0e6,
                "sigma_T": sample_field(sigma_t, XX, YY) / 1.0e6,
                "source": sample_field(source_rate, XX, YY) * SECONDS_PER_DAY / 1.0e6,
                "displacement": sample_vector_field(u_increment, XX, YY),
            }
        )

    history_rows.append(
        [
            step,
            time_days,
            float(p.dat.data_ro.min()),
            float(p.dat.data_ro.max()),
            float(phi.dat.data_ro.min()),
            float(phi.dat.data_ro.max()),
            float(sigma_t.dat.data_ro.min()),
            float(sigma_t.dat.data_ro.max()),
            float(source_rate.dat.data_ro.min()),
            float(source_rate.dat.data_ro.max()),
            iteration,
        ]
    )

    print(
        f"step={step:03d} day={time_days:7.2f} fixed_stress={iteration:02d} "
        f"p=[{p.dat.data_ro.min():.4e}, {p.dat.data_ro.max():.4e}] "
        f"sigma_T=[{sigma_t.dat.data_ro.min():.4e}, {sigma_t.dat.data_ro.max():.4e}]"
    )
    t += dt_seconds


# -----------------------------------------------------------------------------
# CSV diagnostics and plots
# -----------------------------------------------------------------------------
np.savetxt(
    output_dir / "time_history.csv",
    np.array(history_rows),
    fmt=["%d"] + ["%.10e"] * 9 + ["%d"],
    delimiter=",",
    header=(
        "step,time_days,p_min,p_max,phi_min,phi_max,"
        "sigma_T_min,sigma_T_max,source_rate_min,source_rate_max,"
        "fixed_stress_iterations"
    ),
    comments="",
)

(output_dir / "run_metadata.txt").write_text(
    "\n".join(
        [
            "case=2D_fixed_stress_injection_training",
            f"mesh={numel_x}x{numel_y}",
            f"domain_m={Lx}x{Ly}",
            f"T_days={T_days}",
            f"dt_days={dt_days}",
            f"p0_Pa={p0}",
            f"injection_overpressure_Pa={injection_overpressure}",
            f"p_well_Pa={p_well}",
            f"p_reservoir_Pa={p_reservoir}",
            f"overburden_stress_Pa={overburden_stress}",
            f"depth_m={depth}",
            f"rock_density_kg_m3={rock_density}",
            f"gas_saturation={gas_saturation}",
            f"gas_viscosity_Pa_s={gas_viscosity_value}",
            f"water_residual={water_residual}",
            f"gas_residual={gas_residual}",
            f"relative_permeability={k_rel_value}",
            f"initial_porosity={initial_porosity_value}",
            f"initial_permeability_m2={initial_permeability_value}",
            f"alpha_biot={alpha_biot_value}",
            f"young_modulus_Pa={young_modulus}",
            f"poisson_ratio={poisson_ratio}",
            f"bulk_modulus_Pa={bulk_modulus_value}",
            "permeability_law=normalized_phi_over_3_minus_phi",
            "pr_z_fit=same_coefficients_as_scripts/2D/compressible_hm_2D_transient_nl.py",
            "nonlinear_solution=fixed_stress_outer_picard_with_linearized_pressure_subproblem",
            "hydraulic_bcs=p_left_p0_plus_10MPa_other_boundaries_no_flow",
            "field_evolution_snapshot_days=5,30,180,365",
            "displacement_plot=injection_induced_displacement_magnitude_with_quiver_arrows",
            "projected_profile_plot=midheight_profiles_projected.png uses CG1 L2 projection for sigma_T and source_rate only",
            "initial_pressure_condition=uniform_p0",
            "mechanical_reference=equilibrated_at_uniform_p0_and_top_traction_minus_p0_n",
            "mechanical_bcs=uy_bottom_zero_ux_right_zero_shear_natural_zero",
            "source_rate_units=Pa_per_second",
        ]
    )
    + "\n"
)

pressure_grid = sample_field(p, XX, YY) / 1.0e6
sigma_grid = sample_field(sigma_t, XX, YY) / 1.0e6
source_grid = sample_field(source_rate, XX, YY) * SECONDS_PER_DAY / 1.0e6
displacement_grid = sample_vector_field(u_increment, XX, YY)
ux_grid = displacement_grid[:, :, 0]
uy_grid = displacement_grid[:, :, 1]
displacement_magnitude_grid = np.hypot(ux_grid, uy_grid)

fig, axes = plt.subplots(4, 1, figsize=(10.0, 8.7), constrained_layout=True)

field_data = [
    (
        pressure_grid,
        "Final pressure field",
        "Pressure [MPa]",
        "viridis",
        np.linspace(p_reservoir / 1.0e6, p_well / 1.0e6, 51),
        None,
        None,
    ),
    (sigma_grid, "Final mean total normal stress", "sigma_T [MPa]", "magma", None, None, None),
    (
        source_grid,
        "Final fixed-stress source term",
        "Source [MPa/day]",
        "coolwarm",
        None,
        max(np.max(np.abs(source_grid)), 1.0e-12),
        None,
    ),
    (
        displacement_magnitude_grid,
        "Final injection-induced displacement field",
        "|u| [m]",
        "cividis",
        None,
        None,
        (ux_grid, uy_grid),
    ),
]

for ax, (values, title, cbar_label, cmap, levels, symmetric_range, vector_values) in zip(axes, field_data):
    if levels is not None:
        contour = ax.contourf(XX, YY, values, levels=levels, cmap=cmap, extend="both")
    elif symmetric_range is None:
        contour = ax.contourf(XX, YY, values, levels=50, cmap=cmap, extend="both")
    else:
        contour = ax.contourf(
            XX,
            YY,
            values,
            levels=np.linspace(-symmetric_range, symmetric_range, 51),
            cmap=cmap,
            extend="both",
        )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_aspect("auto")
    ax.grid(True, color="white", alpha=0.20, linewidth=0.6)
    if vector_values is not None:
        add_displacement_quiver(ax, *vector_values)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)

fig.suptitle("2D methane injection with hydro-geomechanical fixed-stress coupling")
fig.savefig(output_dir / "final_fields.png", dpi=200)
plt.close(fig)

if field_snapshots:
    ncols = len(field_snapshots)
    fig, axes = plt.subplots(4, ncols, figsize=(3.4 * ncols, 8.8), constrained_layout=True, squeeze=False)

    sigma_values = np.concatenate([snapshot["sigma_T"].ravel() for snapshot in field_snapshots])
    source_values = np.concatenate([snapshot["source"].ravel() for snapshot in field_snapshots])
    displacement_values = np.concatenate(
        [np.hypot(snapshot["displacement"][:, :, 0], snapshot["displacement"][:, :, 1]).ravel() for snapshot in field_snapshots]
    )
    source_absmax = max(float(np.max(np.abs(source_values))), 1.0e-12)
    displacement_max = max(float(np.max(displacement_values)), 1.0e-12)

    evolution_rows = [
        (
            "pressure",
            "Pressure [MPa]",
            "viridis",
            np.linspace(p_reservoir / 1.0e6, p_well / 1.0e6, 51),
        ),
        (
            "sigma_T",
            "sigma_T [MPa]",
            "magma",
            np.linspace(float(sigma_values.min()), float(sigma_values.max()), 51),
        ),
        (
            "source",
            "Source [MPa/day]",
            "coolwarm",
            np.linspace(-source_absmax, source_absmax, 51),
        ),
        (
            "displacement",
            "|u| [m]",
            "cividis",
            np.linspace(0.0, displacement_max, 51),
        ),
    ]

    for row, (field_name, colorbar_label, cmap, levels) in enumerate(evolution_rows):
        last_contour = None
        for col, snapshot in enumerate(field_snapshots):
            ax = axes[row, col]
            if field_name == "displacement":
                ux_snapshot = snapshot["displacement"][:, :, 0]
                uy_snapshot = snapshot["displacement"][:, :, 1]
                values = np.hypot(ux_snapshot, uy_snapshot)
            else:
                ux_snapshot = None
                uy_snapshot = None
                values = snapshot[field_name]
            last_contour = ax.contourf(XX, YY, values, levels=levels, cmap=cmap, extend="both")
            if field_name == "displacement":
                add_displacement_quiver(ax, ux_snapshot, uy_snapshot)
            if row == 0:
                ax.set_title(f"{snapshot['time_days']:g} d")
            if col == 0:
                ax.set_ylabel("y [m]")
            if row == len(evolution_rows) - 1:
                ax.set_xlabel("x [m]")
            ax.set_xlim(0.0, Lx)
            ax.set_ylim(0.0, Ly)
            ax.set_aspect("auto")
            ax.grid(True, color="white", alpha=0.18, linewidth=0.5)
        cbar = fig.colorbar(last_contour, ax=axes[row, :])
        cbar.set_label(colorbar_label)

    fig.suptitle("Field evolution with common color scale per variable")
    fig.savefig(output_dir / "field_evolution.png", dpi=200)
    plt.close(fig)

if profile_snapshots:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
    for time_days, xs, p_values, sigma_values, source_values in profile_snapshots:
        label = f"{time_days:g} d"
        axes[0].plot(xs, p_values / 1.0e6, linewidth=1.8, label=label)
        axes[1].plot(xs, sigma_values / 1.0e6, linewidth=1.8, label=label)
        axes[2].plot(xs, source_values * SECONDS_PER_DAY / 1.0e6, linewidth=1.8, label=label)

    axes[0].set_ylabel("Pressure [MPa]")
    axes[1].set_ylabel("sigma_T [MPa]")
    axes[2].set_ylabel("Source [MPa/day]")
    axes[2].set_xlabel("x at y = 2.5 m [m]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncols=2, fontsize=8)
    fig.suptitle("Mid-height transient profiles")
    fig.savefig(output_dir / "midheight_profiles.png", dpi=200)
    plt.close(fig)

if projected_profile_snapshots:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
    for time_days, xs, p_values, sigma_values, source_values in projected_profile_snapshots:
        label = f"{time_days:g} d"
        axes[0].plot(xs, p_values / 1.0e6, linewidth=1.8, label=label)
        axes[1].plot(xs, sigma_values / 1.0e6, linewidth=1.8, label=label)
        axes[2].plot(xs, source_values * SECONDS_PER_DAY / 1.0e6, linewidth=1.8, label=label)

    axes[0].set_ylabel("Pressure [MPa]")
    axes[1].set_ylabel("sigma_T [MPa]")
    axes[2].set_ylabel("Source [MPa/day]")
    axes[2].set_xlabel("x at y = 2.5 m [m]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncols=2, fontsize=8)
    fig.suptitle("Mid-height transient profiles (CG1 visualization projection)")
    fig.savefig(output_dir / "midheight_profiles_projected.png", dpi=200)
    plt.close(fig)

print(f"Wrote VTK, CSV diagnostics, and PNG plots to {output_dir}")
