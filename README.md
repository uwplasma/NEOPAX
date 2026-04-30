<p align="center">
    <img src="https://github.com/uwplasma/NEOPAX/blob/main/docs/NEOPAX_logo_1.png" align="center" width="50%">
</p>
<p align="center">
    <em><code>❯  NEOPAX - Neoclassical Transport Package in JAX</code></em>
</p>
<p align="center">
    <img src="https://img.shields.io/github/license/uwplasma/NEOPAX?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
    <img src="https://img.shields.io/github/last-commit/uwplasma/NEOPAX?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/uwplasma/NEOPAX?style=default&color=0080ff" alt="repo-top-language">
    <a href="https://github.com/uwplasma/NEOPAX/actions/workflows/build_test.yml">
        <img src="https://github.com/uwplasma/NEOPAX/actions/workflows/build_test.yml/badge.svg" alt="Build Status">
    </a>
    <a href="https://codecov.io/gh/uwplasma/NEOPAX">
        <img src="https://codecov.io/gh/uwplasma/NEOPAX/branch/main/graph/badge.svg" alt="Coverage">
    </a>
    <a href="https://neopax.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/neopax/badge/?version=latest" alt="Documentation Status">
    </a>
</p>

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [From PyPI](#from-pypi)
    - [From Source](#from-source)
- [Usage](#usage)
- [Testing](#testing)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Overview
NEOPAX is a flexible JAX-native transport solver for the radial tranpsort equations in stellarators. It follows the modular philosophy of the tokamak transport solver torax, while specializing in stellarator physics. It can fulfill the roles of power balance steady state solver and predictive transport simulator using different solvers like kvaerno5 from diffrax, a custom implemented radau solver or a theta solver.

## Quick Start
To install NEOPAX you just need to:

```
git clone https://github.com/uwplasma/NEOPAX.git
cd NEOPAX 
pip install .
```

## Usage

After installation, NEOPAX can be launched in three equivalent ways.

### 1. Console script

```bash
NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml
```

Lowercase also works:

```bash
neopax examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml
```

### 2. Python module entry

```bash
python -m NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml
```

### 3. Direct Python API

This is the preferred path when NEOPAX is being driven programmatically or
embedded inside a larger JAX workflow:

```python
import NEOPAX

result = NEOPAX.run(
    "examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml",
    backend="radau",
    n_radial=65,
)
```

You can also use the CLI override layer for common runtime changes:

```bash
NEOPAX examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_theta.toml --mode fluxes --n-radial 65 --n-x 5
```

or generic dotted overrides:

```bash
NEOPAX my_case.toml --set turbulence.debug_heat_flux_scale=0.5 --set transport_solver.dt=1e-4
```

For the full usage guide, including:

- CLI override conventions
- direct API return objects
- `NEOPAX.prepare_config(...)`
- `NEOPAX.run_config(...)`
- when to prefer CLI vs direct API

see:

- `docs/methods_of_use.rst`
- `docs/getting_started.rst`

## Examples
To run an example simply do:

```bash
python examples/Calculate_Fluxes/Fluxes_Calculation.py
```

## Contributing
