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
NEOPAX aims to be a simple JAX framework to solve radial transport equations for stellarators allowing for the optimization of different neoclassical quantities of interest.

## Quick Start
To install NEOPAX you just need to:

```
git clone https://github.com/uwplasma/NEOPAX.git
cd NEOPAX 
pip install .
```

At the moment the code makes use of a Monkes-JAX monoenergetic database Dij(r,collisionality,Er). To obtain these you can use Monkes-JAX, see: https://github.com/monkes  

## Examples
To run an exanmple simply do:

'''
python ./examples/Calculate_Fluxes/Fluxes_Calculation_comparison.py
'''

## Contributing
