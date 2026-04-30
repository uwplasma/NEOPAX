"""Public package API for NEOPAX.

The package root intentionally exposes a curated compatibility surface instead
of re-exporting every internal helper via wildcard imports.
"""

from __future__ import annotations

from .api import RunResult, prepare_config, run
from .version import __version__, __version_tuple__, version, version_tuple

from ._database import Monoenergetic
from ._neoclassical import (
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_Faces,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
)
from ._parameters import Solver_Parameters
from ._species import Species

# Current transport/state API.
from ._boundary_conditions import BoundaryConditionModel, DirichletBC, NeumannBC, RobinBC
from ._model_api import ModelCapabilities, ModelValidationContext, make_validation_context
from ._source_models import build_source_models_from_config, get_source_model, register_source_model, source_model
from ._state import TransportState
from ._turbulence import get_Turbulent_Fluxes_Analytical, get_Turbulent_Fluxes_PowerOverN
from ._transport_equations import (
    ComposedEquationSystem,
    DensityEquation,
    ElectricFieldEquation,
    TemperatureEquation,
    build_equation_system,
    build_equation_system_from_config,
)
from ._transport_flux_models import (
    CombinedTransportFluxModel,
    FluxesRFileTransportModel,
    ZeroTransportModel,
    build_fluxes_r_file_transport_model,
    build_transport_flux_model,
    get_transport_flux_model_capabilities,
    get_transport_flux_model,
    register_transport_flux_model,
    transport_flux_model,
)
from ._transport_solvers import (
    DiffraxSolver,
    NewtonThetaMethodSolver,
    RADAUSolver,
    ThetaMethodSolver,
    build_time_solver,
)

def load_config(*args, **kwargs):
    from .main import load_config as _load_config

    return _load_config(*args, **kwargs)


def run_config(*args, **kwargs):
    from .main import run_config as _run_config

    return _run_config(*args, **kwargs)


def run_config_path(*args, **kwargs):
    from .main import run_config_path as _run_config_path

    return _run_config_path(*args, **kwargs)


__all__ = [
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
    "BoundaryConditionModel",
    "CombinedTransportFluxModel",
    "ComposedEquationSystem",
    "DensityEquation",
    "DiffraxSolver",
    "DirichletBC",
    "ElectricFieldEquation",
    "FluxesRFileTransportModel",
    "Monoenergetic",
    "ModelCapabilities",
    "ModelValidationContext",
    "make_validation_context",
    "NeumannBC",
    "NewtonThetaMethodSolver",
    "RADAUSolver",
    "RobinBC",
    "RunResult",
    "Solver_Parameters",
    "Species",
    "TemperatureEquation",
    "ThetaMethodSolver",
    "TransportState",
    "ZeroTransportModel",
    "build_equation_system",
    "build_equation_system_from_config",
    "build_fluxes_r_file_transport_model",
    "build_source_models_from_config",
    "build_time_solver",
    "build_transport_flux_model",
    "get_Neoclassical_Fluxes",
    "get_Neoclassical_Fluxes_Faces",
    "get_Neoclassical_Fluxes_With_Momentum_Correction",
    "get_source_model",
    "get_Turbulent_Fluxes_Analytical",
    "get_Turbulent_Fluxes_PowerOverN",
    "get_transport_flux_model",
    "get_transport_flux_model_capabilities",
    "load_config",
    "prepare_config",
    "register_transport_flux_model",
    "register_source_model",
    "run",
    "run_config",
    "run_config_path",
    "source_model",
    "transport_flux_model",
]
