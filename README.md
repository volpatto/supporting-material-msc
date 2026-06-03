# supporting-material-msc

Supporting material for hydro-geomechanical methane reservoir simulations related to Diego Tavares Volpatto's 2016 MSc dissertation.

The repository contains small Firedrake scripts for learning and experimentation, plus a larger Chapter 5 reproduction script. The main physical ingredients are compressible gas flow, poroelasticity, and fixed-stress splitting.

## Who This Is For

This README is written for students and researchers who may not be used to programming every day. You do not need to understand the whole codebase before running a first example. The safest way to start is to run one of the standalone scripts, inspect the generated plots, and then move to the Chapter 5 cases.

## How To Cite

This repository is supporting material for the MSc dissertation:

> Diego Tavares Volpatto, *Modelagem computacional do acoplamento hidro-geomecânico em reservatórios não-convencionais de gás*, MSc Dissertation, Laboratório Nacional de Computação Científica, Petrópolis, 2016.

The official TEDE record is available at:

<https://tede.lncc.br/handle/tede/245>

If you use these scripts, reproduce figures, reuse the model equations, or compare against the Chapter 5 cases, please cite the dissertation. The citation provided by the LNCC TEDE record is:

```text
VOLPATTO, D. T. Modelagem computacional do acoplamento hidro-geomecânico em reservatórios não-convencionais de gás, 2016, xvi,73 f. Dissertação (Programa de Pós-Graduação em Modelagem Computacional) Laboratório Nacional de Computação Científica, Petrópolis, 2016.
```

A normalized BibTeX entry is:

```bibtex
@mastersthesis{volpatto2016hm,
  author = {Volpatto, Diego Tavares},
  title = {Modelagem computacional do acoplamento hidro-geomecânico em reservatórios não-convencionais de gás},
  school = {Laboratório Nacional de Computação Científica},
  address = {Petrópolis, RJ, Brasil},
  year = {2016},
  type = {Dissertação de Mestrado},
  url = {https://tede.lncc.br/handle/tede/245}
}
```

For reports or papers derived from this code, cite both the dissertation and any specific software tools that were essential for the computation, especially Firedrake and PETSc when appropriate.

## Important Requirement: Firedrake

This repository does not install Firedrake for you.

You must install Firedrake before running the scripts. Firedrake is a finite element Python framework; it includes PETSc and other numerical tools used by these simulations. Follow the official installation instructions:

<https://www.firedrakeproject.org/install.html>

After Firedrake is installed, activate its Python environment before running anything in this repository. On one local setup this is:

```bash
source ~/firedrake/venv-firedrake/bin/activate
```

If Firedrake was installed somewhere else, replace the path above with the path to your own Firedrake environment.

You can check that Firedrake is available with:

```bash
python -c "import firedrake; print('Firedrake is available')"
```

If this command fails with `ModuleNotFoundError: No module named 'firedrake'`, the Firedrake environment is not active or Firedrake is not installed.

## Repository Layout

```text
scripts/
  1D/
    darcy1D_transient_nl.py
    darcy1D_transient_nl_picard.py
  2D/
    compressible_hm_2D_fixed_stress_injection.py
    compressible_hm_2D_fixed_stress_injection_newton.py
    compressible_hm_2D_fixed_stress_ch5.py
    run_chapter5_cases.py
    plot_chapter5_cases.py
    plot_chapter5_thesis.py

outputs/
  Created automatically when simulations are run.
```

The `outputs/` directory is ignored by Git. This is intentional: numerical results, VTK files, CSV files, and plots can be large and are meant to be regenerated.

## Running From the Correct Place

Open a terminal in the repository root:

```bash
cd /path/to/supporting-material-msc
```

Then activate Firedrake:

```bash
source ~/firedrake/venv-firedrake/bin/activate
```

Run commands from the repository root, not from inside `scripts/`.

## First Recommended Run

Start with the standalone 2D injection case. It is simpler than the full Chapter 5 batch and produces plots directly.

Picard pressure backend:

```bash
python scripts/2D/compressible_hm_2D_fixed_stress_injection.py
```

Newton pressure backend:

```bash
python scripts/2D/compressible_hm_2D_fixed_stress_injection_newton.py
```

These scripts write results to:

```text
outputs/2D/compressible_hm_2D_fixed_stress_injection/
outputs/2D/compressible_hm_2D_fixed_stress_injection_newton/
```

Useful files to open first:

```text
final_fields.png
field_evolution.png
midheight_profiles.png
time_history.csv
run_metadata.txt
```

The PNG files are the easiest way to inspect the solution. The CSV files contain numerical values that can be opened in a spreadsheet program or read with Python.

## What the Standalone 2D Scripts Do

The two standalone scripts solve a 2D compressible methane injection problem with hydro-geomechanical coupling.

They include:

- methane compressibility through a fitted Peng-Robinson `Z(p)` polynomial;
- porosity and permeability changes;
- poroelastic deformation;
- fixed-stress coupling between pressure and mechanics;
- pressure, stress, source-term, and displacement plots.

The difference between the two scripts is the nonlinear pressure solve:

- `compressible_hm_2D_fixed_stress_injection.py` uses a Picard-style linearization.
- `compressible_hm_2D_fixed_stress_injection_newton.py` uses a PETSc/SNES Newton solve inside each fixed-stress pressure subproblem.

## Running One Chapter 5 Case

The Chapter 5 script is:

```text
scripts/2D/compressible_hm_2D_fixed_stress_ch5.py
```

Example:

```bash
python scripts/2D/compressible_hm_2D_fixed_stress_ch5.py \
  --case CampoM03kp \
  --nx 40 \
  --ny 80 \
  --months 180 \
  --dt-months 1
```

This writes one case under:

```text
outputs/chapter5/CampoM03kp/
```

For a first test, keep `--nx 40 --ny 80`. Larger meshes, for example `--nx 200 --ny 200`, are more expensive and may take much longer.

## Chapter 5 Case Names

Case names follow the thesis convention. For example:

```text
CampoM03kp
```

Read this as:

- `M`: Marcellus formation;
- `03`: two-way hydro-geomechanical coupling;
- `kp`: deep case.

The scripts also support Barnett cases (`B`) and shallow cases (`bp`).

## Running All Chapter 5 Cases

To run the full set of Chapter 5 cases:

```bash
python scripts/2D/run_chapter5_cases.py \
  --nx 40 \
  --ny 80 \
  --months 180 \
  --dt-months 1
```

To run only a subset:

```bash
python scripts/2D/run_chapter5_cases.py \
  --cases CampoM03kp CampoM01kp \
  --nx 40 \
  --ny 80 \
  --months 180 \
  --dt-months 1
```

To use the Newton pressure backend:

```bash
python scripts/2D/run_chapter5_cases.py \
  --nonlinear-solver newton \
  --output-root outputs/chapter5_newton
```

## Generating Chapter 5 Plots

After running Chapter 5 cases, generate general comparison plots with:

```bash
python scripts/2D/plot_chapter5_cases.py --output-root outputs/chapter5
```

Generate thesis-style plots with:

```bash
python scripts/2D/plot_chapter5_thesis.py --output-root outputs/chapter5
```

The plotting scripts read the CSV files produced by the Firedrake runs. They do not rerun the simulation.

## 1D Scripts

The 1D scripts are smaller examples for transient nonlinear Darcy flow:

```bash
python scripts/1D/darcy1D_transient_nl.py
python scripts/1D/darcy1D_transient_nl_picard.py
```

They are useful for understanding the pressure nonlinearity before moving to the 2D hydro-geomechanical model.

## Common Problems

### `ModuleNotFoundError: No module named 'firedrake'`

Firedrake is not active in the current terminal. Activate the Firedrake environment:

```bash
source ~/firedrake/venv-firedrake/bin/activate
```

If that path does not exist, Firedrake may be installed somewhere else, or it may not be installed yet.

### The simulation is slow

Start with a smaller mesh:

```bash
--nx 40 --ny 80
```

Use larger meshes only after the smaller run works.

### I do not see output files

Check that you ran the command from the repository root. Most scripts write to `outputs/...`, relative to the repository.

### The plots look different after changing the model

That is expected. These scripts are active research/training material. Boundary conditions, time step, nonlinear solver, and material parameters all affect pressure, stress, source terms, and displacement.

## Scientific Notes

The Chapter 5 implementation is self-contained: case metadata, material parameters, and fitted Peng-Robinson coefficients are encoded as model inputs. It does not read legacy thesis result fields.

The standalone injection scripts are training-oriented. They are useful for understanding fixed-stress coupling, Picard versus Newton pressure solves, and the interpretation of injection-induced displacement. They should not be treated as calibrated field simulations without further validation.

## AI Statement of Use

Following CNPq guidance that the use of generative artificial intelligence be declared with the tool and purpose specified, this repository records that OpenAI Codex was used to assist the implementation, code refactoring, documentation editing, and consistency checks of this supporting material derived from the original MSc work:

<https://www.gov.br/cnpq/pt-br/composicao/comissao-de-integridade/diretrizes>

The scientific model, interpretation of the equations, numerical assumptions, and final responsibility for the code and results remain with the repository author. AI-assisted changes should be reviewed critically before being used in teaching, publication, or scientific comparison.

## License

This repository is distributed under the MIT License. See [LICENSE](LICENSE).
