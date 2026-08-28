"""Unit checks for optimization-only reuse of VMEX raw-block setup."""

import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp

from NEOPAX import _geometry_autodiff as geometry_ad
from NEOPAX import _optimization_initial_root_stage as initial_root_stage
from NEOPAX import optimization


def test_raw_block_solve_uses_prebuilt_stage_without_rebuilding_config(monkeypatch):
    """Trial deltas must be dynamic while the stage configuration stays shared."""

    entries = (
        {
            "family": "RBC",
            "m": 1,
            "n": 0,
            "input_field": "rbc",
            "n_offset": 1,
            "m_index": 1,
        },
    )

    class Implicit:
        def solve_implicit_with_aux(self, params, cfg):
            assert params == "trial-params"
            assert cfg is shared_cfg
            return "trial-state", "dof-mask"

    shared_cfg = object()
    stage = geometry_ad.GeometryRawBlockStage(
        implicit=Implicit(),
        implicit_cfg=shared_cfg,
        param_entries=entries,
    )
    context = SimpleNamespace()
    observed = {}

    monkeypatch.setattr(geometry_ad, "_using_current_vmec_jax_context", lambda _context: True)

    def build_trial_params(_context, implicit, deltas, staged_entries, *, solver_device):
        observed["implicit"] = implicit
        observed["deltas"] = deltas
        observed["entries"] = staged_entries
        observed["solver_device"] = solver_device
        return "trial-params"

    monkeypatch.setattr(geometry_ad, "_implicit_params_with_boundary_deltas", build_trial_params)

    result = geometry_ad.geometry_raw_block_solve_from_param_vector(
        context,
        jnp.asarray([0.25]),
        (("RBC", 1, 0),),
        solver_device="cpu",
        stage=stage,
    )

    assert result.implicit is stage.implicit
    assert result.implicit_cfg is shared_cfg
    assert result.param_entries == entries
    assert result.state == "trial-state"
    assert observed["entries"] == entries
    assert observed["solver_device"] == "cpu"


def test_raw_block_stage_rejects_a_different_parameter_layout(monkeypatch):
    stage = geometry_ad.GeometryRawBlockStage(
        implicit=object(),
        implicit_cfg=object(),
        param_entries=(
            {
                "family": "RBC",
                "m": 1,
                "n": 0,
                "input_field": "rbc",
                "n_offset": 1,
                "m_index": 1,
            },
        ),
    )
    monkeypatch.setattr(geometry_ad, "_using_current_vmec_jax_context", lambda _context: True)

    try:
        geometry_ad.geometry_raw_block_solve_from_param_vector(
            SimpleNamespace(),
            jnp.asarray([0.0]),
            (("ZBS", 1, 0),),
            stage=stage,
        )
    except ValueError as exc:
        assert "parameter layout" in str(exc)
    else:
        raise AssertionError("Expected the mismatched staged layout to be rejected.")


def test_repeated_evaluation_memory_samples_release_evaluations(monkeypatch):
    """The audit must invoke the existing evaluator without retaining results."""

    class Problem:
        x0 = jnp.asarray([0.0])

        def __init__(self):
            self.calls = 0

        def evaluate(self, _x):
            self.calls += 1
            return SimpleNamespace(
                residuals=jnp.asarray([3.0]),
                jacobian=jnp.asarray([[4.0]]),
            )

    memory = iter((100, 110, 120))
    monkeypatch.setattr(optimization, "_process_resident_memory_bytes", lambda: next(memory))
    problem = Problem()
    reported = []
    samples = optimization.repeated_evaluation_memory_samples(
        problem,
        warmup=1,
        repeats=3,
        on_sample=reported.append,
    )

    assert problem.calls == 4
    assert [sample.resident_memory_bytes for sample in samples] == [100, 110, 120]
    assert reported == list(samples)
    assert all(sample.residual_norm == 3.0 for sample in samples)
    assert all(sample.jacobian_shape == (1, 1) for sample in samples)


def test_initial_root_reverse_stage_owns_only_callable_kernel_identities():
    layout = initial_root_stage.InitialRootStageLayout(
        objective_names=("maxEr", "J_bootstrap"),
        geometry_param_specs=(("RBC", 1, 0),),
        n_r=51,
        n_theta=25,
        n_zeta=25,
        n_xi=64,
        surface_backend="vmec",
        flux_model="ntx_exact_lij_runtime",
    )

    def kernel(*args):
        return args

    stage = initial_root_stage.build_initial_root_reverse_optimization_stage(
        layout=layout,
        corrected_bootstrap_fluxes=kernel,
        bootstrap_state_pullback=kernel,
        bootstrap_geometry_pullback=kernel,
        bootstrap_support_pullback=kernel,
        root_geometry_residual_pullback=kernel,
    )

    assert stage.layout is layout
    assert stage.kernels.corrected_bootstrap_fluxes is kernel
    assert stage.kernels.root_geometry_residual_pullback is kernel


def test_initial_root_reverse_kernel_adapters_keep_trial_geometry_and_support_dynamic():
    @dataclasses.dataclass(frozen=True)
    class Model:
        geometry: object
        support: object

        def evaluate_momentum_corrected_fluxes(self, state):
            return ("flux", state, self.geometry, self.support)

        def pullback_momentum_corrected_upar_state_by_radius(self, state, bars):
            return ("state", state, bars, self.geometry, self.support)

        def pullback_momentum_corrected_upar_geometry_by_radius(
            self, state, bars, geometry, support
        ):
            return ("geometry", state, bars, geometry, support, self.geometry, self.support)

        def pullback_momentum_corrected_upar_support_by_radius(self, state, bars, support):
            return ("support", state, bars, support, self.geometry, self.support)

    dependencies = initial_root_stage.InitialRootReverseDependencies(
        root_geometry_residual_pullback=(
            lambda _state, _er, _geometry, _support, bars, _delta: bars
        ),
    )
    kernels = initial_root_stage.build_initial_root_reverse_kernels_optimization(
        neoclassical_model=Model(geometry="static-geometry", support="static-support"),
        dependencies=dependencies,
    )

    assert kernels.corrected_bootstrap_fluxes("state", "geometry", "support") == (
        "flux", "state", "geometry", "support"
    )
    assert kernels.bootstrap_state_pullback("state", "bars", "geometry", "support") == (
        "state", "state", "bars", "geometry", "support"
    )
    assert kernels.bootstrap_geometry_pullback("state", "bars", "geometry", "support") == (
        "geometry", "state", "bars", "geometry", "support", "geometry", "support"
    )
    assert kernels.bootstrap_support_pullback("state", "bars", "geometry", "support") == (
        "support", "state", "bars", "support", "geometry", "support"
    )

    geometry_bars = kernels.root_geometry_residual_pullback(
        jnp.asarray([0.0]),
        jnp.asarray([2.0]),
        jnp.asarray([3.0, 4.0]),
        jnp.asarray([0.0]),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.zeros((2,)),
    )
    assert jnp.array_equal(geometry_bars, jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
