"""Unit checks for optimization-only reuse of VMEX raw-block setup."""

import ast
import copy
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX import _geometry_autodiff as geometry_ad
from NEOPAX import _orchestrator as orchestrator
from NEOPAX import _optimization_full_transport_stage as full_transport_stage
from NEOPAX import _optimization_initial_root_stage as initial_root_stage
from NEOPAX import _reverse_ad_optimization as reverse_optimization
from NEOPAX import _reverse_ad_transport as reverse_transport
from NEOPAX import _transport_solvers as transport_solvers
from NEOPAX import optimization
from NEOPAX._reverse_ad_optimization import normalize_geometry_full_ad_objective_names
from NEOPAX._state import TransportState
from NEOPAX._transport_flux_models import DENSITY_STATE_TO_PHYSICAL
from NEOPAX._constants import elementary_charge


def test_database_transport_bootstrap_evolution_reuses_saved_states(tmp_path):
    class _DatabaseFluxModel:
        @staticmethod
        def evaluate_momentum_corrected_fluxes(state):
            return {"Upar_neo": state.density}

    saved_states = TransportState(
        density=jnp.asarray(
            [
                [[2.0, 4.0], [1.0, 3.0]],
                [[100.0, 100.0], [0.0, 0.0]],
                [[7.0, 11.0], [2.0, 5.0]],
            ]
        ),
        pressure=jnp.ones((3, 2, 2)),
        Er=jnp.zeros((3, 2)),
    )
    solution = {
        "ys": saved_states,
        "ts": jnp.asarray([0.0, 1.0, 2.0]),
        "accepted_mask": jnp.asarray([True, False, True]),
    }
    runtime = SimpleNamespace(
        database=object(),
        species=SimpleNamespace(charge_qp=jnp.asarray([1.0, -1.0])),
        models=SimpleNamespace(flux=_DatabaseFluxModel()),
    )

    result = orchestrator.write_transport_bootstrap_current_evolution(
        jnp.asarray([0.25, 0.75]),
        solution,
        tmp_path,
        runtime=runtime,
    )

    scale = DENSITY_STATE_TO_PHYSICAL * elementary_charge * 1.0e-5
    np.testing.assert_allclose(result["times"], [0.0, 2.0])
    np.testing.assert_allclose(
        result["profiles"],
        np.asarray([[1.0, 1.0], [5.0, 6.0]]) * scale,
    )
    assert (tmp_path / "bootstrap_current_evolution.csv").is_file()


def test_least_squares_does_not_reapply_problem_coordinate_scale(monkeypatch):
    """ESS/profile scaling lives in the problem coordinates, not SciPy twice."""

    captured = {}
    evaluation = SimpleNamespace(
        residuals=jnp.asarray([1.0]),
        jacobian=jnp.asarray([[0.25, 0.5]]),
    )

    class _Problem:
        parameter_count = 2
        x0 = jnp.asarray([0.0, 0.0])
        x_scale = jnp.asarray([0.25, 0.125])

        @staticmethod
        def evaluate(_values):
            return evaluation

    def fake_scipy_least_squares(fun, x0, *, jac, **kwargs):
        captured["x_scale"] = np.asarray(kwargs["x_scale"], dtype=float)
        np.testing.assert_allclose(fun(x0), np.asarray([1.0]))
        np.testing.assert_allclose(jac(x0), np.asarray([[0.25, 0.5]]))
        return SimpleNamespace(x=np.asarray(x0, dtype=float))

    monkeypatch.setattr("scipy.optimize.least_squares", fake_scipy_least_squares)

    optimization.least_squares(_Problem())

    np.testing.assert_array_equal(captured["x_scale"], np.ones(2))


def test_database_initial_root_unfolded_support_bars_sanitize_float0(monkeypatch):
    """The full-transport root adapter emits ordinary float-delta payloads."""

    @dataclasses.dataclass
    class _StateBars:
        Er: object

    geometry = {
        "coefficient": jnp.asarray([2.0]),
        "mode": jnp.asarray([1], dtype=jnp.int32),
    }
    database = {
        "table": jnp.asarray([3.0]),
        "index": jnp.asarray([0], dtype=jnp.int32),
    }
    direct_geometry = {
        "coefficient": jnp.asarray([[1.0]]),
        "mode": jnp.zeros((1, 1), dtype=jax.dtypes.float0),
    }
    residual_geometry = {
        "coefficient": jnp.asarray([[2.0]]),
        "mode": jnp.zeros((1, 1), dtype=jax.dtypes.float0),
    }
    table_bars = {
        "table": jnp.asarray([[4.0]]),
        "index": jnp.zeros((1, 1), dtype=jax.dtypes.float0),
    }
    direct_database = {
        "table": jnp.asarray([[5.0]]),
        "index": jnp.zeros((1, 1), dtype=jax.dtypes.float0),
    }

    monkeypatch.setattr(
        reverse_optimization,
        "initial_er_charge_flux_residual_er_derivative",
        lambda *args, **kwargs: jnp.ones((1,)),
    )
    monkeypatch.setattr(
        reverse_optimization,
        "compact_initial_er_state_pullback",
        lambda **kwargs: _StateBars(Er=jnp.zeros((1, 1))),
    )
    monkeypatch.setattr(
        reverse_optimization,
        "compact_initial_er_database_support_bars",
        lambda **kwargs: table_bars,
    )
    monkeypatch.setattr(
        reverse_optimization,
        "compact_initial_er_database_geometry_bars",
        lambda **kwargs: residual_geometry,
    )
    monkeypatch.setattr(reverse_optimization, "_add_trees", lambda lhs, rhs: lhs)

    _, support_rows = reverse_optimization._database_initial_root_to_unfolded_support_bars(
        fixed_runtime=object(),
        support={"geometry": geometry, "database": database},
        pre_root_state=object(),
        er_profile=jnp.zeros((1,)),
        finite_mask=jnp.ones((1,), dtype=bool),
        rooted_state_bars=_StateBars(Er=jnp.zeros((1, 1))),
        direct_geometry_bars=direct_geometry,
        direct_database_bars=direct_database,
        parameter_set=SimpleNamespace(profile_specs=()),
        profile_values_arr=jnp.zeros((0,)),
        pre_root_state_from_profile_values=lambda value: value,
        objective_count=1,
    )

    assert jnp.allclose(support_rows[0]["geometry"]["coefficient"], jnp.asarray([3.0]))
    assert support_rows[0]["geometry"]["mode"].dtype == jnp.float64
    assert jnp.allclose(support_rows[0]["database"]["table"], jnp.asarray([9.0]))
    assert support_rows[0]["database"]["index"].dtype == jnp.float64


def test_local_vmex_mercier_softmax_adapter_uses_state_runtime_path(monkeypatch):
    """The old VMEX branch exposes DMerc without WOUT or Boozer data."""

    runtime = object()
    calls = {}

    class _Stability:
        @staticmethod
        def mercier_stability_softmax(state, received_runtime, *, margin, smoothing, temperature):
            calls.update(
                state=state,
                runtime=received_runtime,
                margin=margin,
                smoothing=smoothing,
                temperature=temperature,
            )
            return jnp.asarray(2.5e-4)

    monkeypatch.setattr(geometry_ad, "_import_vmec_module", lambda name: _Stability)
    context = SimpleNamespace(static=SimpleNamespace(runtime=runtime))
    value = geometry_ad.vmec_mercier_stability_softmax_objective_from_state(
        context, "converged-state"
    )

    assert jnp.allclose(value, jnp.asarray(2.5e-4))
    assert calls == {
        "state": "converged-state",
        "runtime": runtime,
        "margin": 0.0,
        "smoothing": 1.0e-6,
        "temperature": 1.0e-3,
    }


def test_local_vmex_mercier_softmax_is_a_full_geometry_objective_row():
    names = geometry_ad.geometry_observable_names_for_kind("geometry_full_ad_objectives")
    assert names[-1] == "vmec_dmerc_stability_softmax"
    assert normalize_geometry_full_ad_objective_names(("dmerc",)) == (
        "vmec_dmerc_stability_softmax",
    )


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


def test_geometry_input_saving_problem_writes_each_unique_trial_before_evaluation(tmp_path):
    events = []

    class InputDeck:
        def __init__(self, values):
            self.values = values

        def to_indata(self, path):
            events.append(("write", self.values))
            path.write_text(",".join(str(value) for value in self.values), encoding="utf-8")

    class Problem:
        x0 = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        parameter_count = 2

        def input_from_scaled_parameters(self, values):
            return InputDeck(tuple(float(value) for value in values))

        def evaluate(self, values=None):
            host_values = self.x0 if values is None else values
            values_tuple = tuple(float(value) for value in host_values)
            events.append(("evaluate", values_tuple))
            if values_tuple == (2.0, 3.0):
                raise RuntimeError("trial failed")
            return SimpleNamespace(residuals=jnp.asarray([0.0]), jacobian=jnp.asarray([[0.0, 0.0]]))

    wrapped = optimization.GeometryInputSavingProblem(Problem(), tmp_path)
    wrapped.evaluate(jnp.asarray([0.0, 0.0]))
    wrapped.evaluate(jnp.asarray([0.0, 0.0]))
    try:
        wrapped.evaluate(jnp.asarray([2.0, 3.0]))
    except RuntimeError as exc:
        assert str(exc) == "trial failed"
    else:
        raise AssertionError("Expected the synthetic trial to fail.")

    paths = sorted(tmp_path.glob("input.geometry_eval_*"))
    assert [path.name for path in paths] == [
        "input.geometry_eval_0000",
        "input.geometry_eval_0001",
    ]
    assert paths[0].read_text(encoding="utf-8") == "0.0,0.0"
    assert paths[1].read_text(encoding="utf-8") == "2.0,3.0"
    assert events == [
        ("write", (0.0, 0.0)),
        ("evaluate", (0.0, 0.0)),
        ("evaluate", (0.0, 0.0)),
        ("write", (2.0, 3.0)),
        ("evaluate", (2.0, 3.0)),
    ]


def test_database_profile_diagnostics_rebuild_live_database_runtime(monkeypatch):
    """Plot helpers must not inject exact-Lij support into a database model."""

    @dataclasses.dataclass(frozen=True)
    class State:
        Er: object

    live_runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=jnp.asarray([0.2, 0.8])))
    calls = {}

    monkeypatch.setattr(
        optimization,
        "geometry_raw_block_solve_from_param_vector",
        lambda *args, **kwargs: SimpleNamespace(state="trial-vmec-state"),
    )

    def rebuild_runtime(config, context, state, *, n_r):
        calls.update(config=config, context=context, state=state, n_r=n_r)
        return live_runtime, object()

    monkeypatch.setattr(optimization, "build_runtime_context_for_vmec_state", rebuild_runtime)
    monkeypatch.setattr(
        optimization,
        "build_neopax_geometry_and_ntx_exact_lij_support_from_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("database diagnostics must not build exact-Lij support")
        ),
    )
    monkeypatch.setattr(
        optimization,
        "initial_er_selected_root_profile",
        lambda state, *, config, runtime: (
            jnp.asarray([1.0, -1.0]),
            jnp.asarray([True, True]),
        ),
    )

    boundary_spec = SimpleNamespace(as_tuple=lambda: ("RBC", 1, 0))
    problem = SimpleNamespace(
        x0=jnp.asarray([0.0]),
        _scaled_to_physical=lambda values: jnp.asarray(values),
        baseline_profile_values=jnp.zeros(6),
        parameter_set=SimpleNamespace(
            specs=(boundary_spec,),
            vmec_boundary_specs=(boundary_spec,),
        ),
        runtime=object(),
        context="geometry-context",
        geometry_max_iter=None,
        geometry_solver_device="default",
        n_r=51,
        n_theta=5,
        n_zeta=25,
        n_xi=31,
        surface_backend="vmec",
        config={"neoclassical": {"flux_model": "ntx_scan_runtime"}},
        _pre_root_state_from_profile_values=lambda values: State(Er=jnp.zeros(2)),
    )

    profile_helper = (
        optimization.GeometryInitialErRootLeastSquaresProblem
        ._initial_er_root_state_runtime_from_scaled_parameters
    )
    rho, rooted_state, runtime, finite_mask = profile_helper(problem, jnp.asarray([0.1]))

    assert runtime is live_runtime
    assert jnp.allclose(rho, jnp.asarray([0.2, 0.8]))
    assert jnp.allclose(rooted_state.Er, jnp.asarray([1.0, -1.0]))
    assert jnp.all(finite_mask)
    assert calls == {
        "config": problem.config,
        "context": "geometry-context",
        "state": "trial-vmec-state",
        "n_r": 51,
    }


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


def test_initial_er_transport_payload_adapter_rebuilds_only_floating_trial_leaves():
    baseline = {
        "geometry": {
            "metric": jnp.asarray([1.0, 2.0]),
            "mode_numbers": jnp.asarray([0, 1], dtype=jnp.int32),
            "label": "fixed",
        },
        "ntx_support": {
            "coefficients": jnp.asarray([[3.0]]),
            "radial_index": jnp.asarray([4], dtype=jnp.int32),
        },
    }
    adapter = initial_root_stage.InitialErTransportPayloadAdapter.from_payload(baseline)

    trial = {
        "geometry": {
            "metric": jnp.asarray([5.0, 6.0]),
            "mode_numbers": jnp.asarray([0, 1], dtype=jnp.int32),
            "label": "fixed",
        },
        "ntx_support": {
            "coefficients": jnp.asarray([[7.0]]),
            "radial_index": jnp.asarray([4], dtype=jnp.int32),
        },
    }
    geometry_leaves, support_leaves = adapter.dynamic_leaves(trial)
    rebuilt = adapter.rebuild(geometry_leaves, support_leaves)

    assert len(geometry_leaves) == 1
    assert len(support_leaves) == 1
    assert jnp.array_equal(rebuilt["geometry"]["metric"], trial["geometry"]["metric"])
    assert jnp.array_equal(rebuilt["ntx_support"]["coefficients"], trial["ntx_support"]["coefficients"])
    assert rebuilt["geometry"]["label"] == "fixed"
    assert jnp.array_equal(rebuilt["geometry"]["mode_numbers"], baseline["geometry"]["mode_numbers"])


def test_database_initial_er_payload_adapter_rebuilds_only_floating_trial_leaves():
    baseline = {
        "geometry": {
            "metric": jnp.asarray([1.0, 2.0]),
            "mode_numbers": jnp.asarray([0, 1], dtype=jnp.int32),
        },
        "database": {
            "D11_log": jnp.asarray([[3.0]]),
            "Er_list": jnp.asarray([-1.0, 0.0, 1.0]),
            "grid_size": jnp.asarray(3, dtype=jnp.int32),
        },
    }
    adapter = initial_root_stage.DatabaseInitialErTransportPayloadAdapter.from_payload(baseline)
    trial = {
        "geometry": {
            "metric": jnp.asarray([5.0, 6.0]),
            "mode_numbers": jnp.asarray([0, 1], dtype=jnp.int32),
        },
        "database": {
            "D11_log": jnp.asarray([[7.0]]),
            "Er_list": jnp.asarray([-2.0, 0.0, 2.0]),
            "grid_size": jnp.asarray(3, dtype=jnp.int32),
        },
    }
    geometry_leaves, database_leaves = adapter.dynamic_leaves(trial)
    rebuilt = adapter.rebuild(geometry_leaves, database_leaves)

    assert len(geometry_leaves) == 1
    assert len(database_leaves) == 2
    assert jnp.array_equal(rebuilt["geometry"]["metric"], trial["geometry"]["metric"])
    assert jnp.array_equal(rebuilt["database"]["D11_log"], trial["database"]["D11_log"])
    assert jnp.array_equal(rebuilt["database"]["Er_list"], trial["database"]["Er_list"])
    assert jnp.array_equal(rebuilt["database"]["grid_size"], baseline["database"]["grid_size"])


def test_database_full_transport_mode_uses_one_integrated_benchmark_builder(monkeypatch):
    """The optimization selector must not build the retired two-sweep bridge."""

    production_dependencies = (
        reverse_transport.default_realtime_geometry_support_reverse_dependencies()
    )
    assert production_dependencies.segment_replay_minimal_with_primal_records is None
    assert (
        production_dependencies.database_segment_reduced_cotangent_bwd_with_table_support
        is None
    )
    assert production_dependencies.database_bootstrap_objective_row is None
    assert production_dependencies.database_initial_root_selected_profile is None
    assert production_dependencies.database_initial_root_pullback is None
    assert production_dependencies.optimization_phase_probe is None

    expected_database_modes = {
        "reverse_stage_adjoint_solve_mode": "block",
        "reverse_rhs_transpose_mode": "explicit_database",
        "reverse_rhs_pullback_mode": "separate",
        "reverse_initial_cache_support_pullback_mode": "scalar",
        "reverse_rebuild_support_pullback_mode": "separate",
        "reverse_database_initial_support_mode": "reduced_zero",
        "reverse_database_initial_state_mode": "reduced_zero_rhs",
        "reverse_database_support_preparation_mode": "shared",
        "reverse_database_center_geometry_mode": "scalar_jvp",
        "reverse_database_stage_jacobian_mode": "independent",
        "reverse_database_support_objective_mode": "scalar",
        "reverse_database_segment_support_mode": "inline",
        "reverse_database_interpolation_transpose_mode": "legacy_sparse",
        "reverse_database_root_interpolation_transpose_mode": "legacy_sparse",
        "reverse_database_bootstrap_interpolation_transpose_mode": "legacy_sparse",
        "reverse_final_objective_cotangent_mode": "grouped_joint_vjp",
        "reverse_bootstrap_cotangent_mode": "joint_local_vjp_upar_only",
        "reverse_schedule_artifact_mode": "reuse_static_probe",
        "reverse_segment_start_replay_mode": "minimal",
        "reverse_segment_primal_record_mode": "reuse_segment_primal_record",
        "reverse_stage_cotangent_mode": "full",
        "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
        "reverse_stage_adjoint_memory_mode": "default",
    }

    context = object()
    runtime = object()
    baseline_state = SimpleNamespace(pressure=jnp.asarray([1.0]))
    vmec_spec = SimpleNamespace(as_tuple=lambda: ("RBC", 1, 0))
    parameterization = SimpleNamespace(
        specs=(vmec_spec,),
        x_scale=jnp.asarray([1.0]),
    )
    parameter_set = SimpleNamespace(specs=("parameter-spec",))
    table_context = object()

    monkeypatch.setattr(
        optimization,
        "_prepare_full_transport_config",
        lambda _config, *, device: {
            "geometry": {"vmec_input_file": "seed-input", "n_radial": 7},
            "neoclassical": {
                "flux_model": "ntx_scan_runtime",
                "ntx_exact_n_theta": 5,
                "ntx_exact_n_zeta": 25,
                "ntx_exact_n_xi": 31,
            },
        },
    )
    monkeypatch.setattr(
        optimization,
        "build_geometry_autodiff_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        optimization,
        "freeze_physical_qi_maxj_pitches",
        lambda value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        optimization,
        "vmex_boundary_parameterization",
        lambda *_args, **_kwargs: parameterization,
    )
    monkeypatch.setattr(
        optimization,
        "reverse_ad_optimization_parameter_set",
        lambda **_kwargs: parameter_set,
    )
    monkeypatch.setattr(
        optimization,
        "build_runtime_context",
        lambda _config: (runtime, baseline_state),
    )
    monkeypatch.setattr(
        optimization,
        "_profile_values_from_config",
        lambda _config, _dtype: jnp.zeros((6,), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        optimization,
        "realtime_geometry_transport_reverse_table_context",
        lambda **_kwargs: table_context,
    )
    root_stage = object()
    root_stage_builds = []

    def _root_stage_factory(**kwargs):
        root_stage_builds.append(kwargs)
        return root_stage

    monkeypatch.setattr(
        optimization,
        "build_database_initial_root_experiment_stage",
        _root_stage_factory,
    )
    raw_stage = SimpleNamespace(raw_block_stage=object())
    transpose_stage = object()
    payload_stage = object()
    runtime_stage = object()
    runtime_stage_builds = []
    monkeypatch.setattr(
        optimization,
        "geometry_raw_block_optimization_stage",
        lambda *_args, **_kwargs: raw_stage,
    )
    monkeypatch.setattr(
        optimization,
        "geometry_raw_block_transpose_optimization_stage",
        lambda *_args, **_kwargs: transpose_stage,
    )
    monkeypatch.setattr(
        optimization,
        "_prepare_initial_root_payload_static",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        optimization,
        "build_initial_root_payload_assembly_stage",
        lambda **_kwargs: payload_stage,
    )
    monkeypatch.setattr(
        optimization,
        "build_database_full_transport_runtime_optimization_stage",
        lambda **kwargs: runtime_stage_builds.append(kwargs) or runtime_stage,
    )

    for stage_mode in ("benchmark", "database_full_transport_optimization"):
        calls = []
        builder = lambda *_args, **_kwargs: None

        def _builder_factory(**kwargs):
            calls.append(kwargs)
            return builder

        monkeypatch.setattr(
            optimization,
            "internal_realtime_geometry_transport_reverse_table_result_builder",
            _builder_factory,
        )
        problem = optimization.geometry_full_transport_least_squares_problem(
            {},
            ((optimization.geometry.vmec_aspect_ratio, 0.0, 1.0),),
            max_mode=1,
            reverse_stage_mode=stage_mode,
            initial_er_root_ad="jax_selected_root",
            er_transition_left_index=25,
            er_transition_right_index=26,
            accepted_step_limit=16,
            reverse_segment_length=4,
            max_reverse_accepted_steps=16,
            reverse_stage_adjoint_solve_mode="block",
            reverse_rhs_transpose_mode="explicit_database",
            reverse_step_bwd_mode="reduced_cotangent_call_boundary",
            reverse_segment_input_diagnostics=True,
        )

        assert len(calls) == 1
        assert calls[0]["initial_er_root_ad"] == "jax_selected_root"
        assert calls[0]["er_transition_left_index"] == 25
        assert calls[0]["er_transition_right_index"] == 26
        assert calls[0]["accepted_step_limit"] == 16
        assert calls[0]["reverse_segment_length"] == 4
        assert calls[0]["max_reverse_accepted_steps"] == 16
        assert calls[0]["reverse_stage_adjoint_solve_mode"] == "block"
        assert calls[0]["reverse_rhs_transpose_mode"] == "explicit_database"
        assert calls[0]["reverse_step_bwd_mode"] == "reduced_cotangent_call_boundary"
        assert calls[0]["reverse_segment_input_diagnostics"] is True
        assert problem.options["reverse_segment_input_diagnostics"] is True
        if stage_mode == "benchmark":
            assert calls[0]["segment_replay_optimization_stage_builder"] is None
            assert calls[0]["bootstrap_optimization_stage_builder"] is None
            assert calls[0]["payload_assembly_optimization_stage"] is None
            assert calls[0]["initial_root_optimization_stage"] is None
            assert calls[0]["runtime_optimization_stage"] is None
            assert problem.raw_block_optimization_stage is None
            assert problem.raw_block_transpose_optimization_stage is None
        else:
            assert (
                calls[0]["segment_replay_optimization_stage_builder"]
                is full_transport_stage.build_database_full_transport_replay_optimization_stage
            )
            assert (
                calls[0]["bootstrap_optimization_stage_builder"]
                is full_transport_stage.build_database_full_transport_bootstrap_optimization_stage
            )
            assert calls[0]["payload_assembly_optimization_stage"] is payload_stage
            assert calls[0]["initial_root_optimization_stage"] is root_stage
            assert calls[0]["runtime_optimization_stage"] is runtime_stage
            assert problem.raw_block_optimization_stage is raw_stage
            assert problem.raw_block_transpose_optimization_stage is transpose_stage
        for name, expected in expected_database_modes.items():
            assert calls[0][name] == expected
            assert problem.options[name] == expected
        assert problem.options["reverse_stage_mode"] == stage_mode
        assert problem.options["Er_transition_left_index"] == 25
        assert problem.options["Er_transition_right_index"] == 26
        assert problem.table_result_builder is builder

    assert len(root_stage_builds) == 1
    assert root_stage_builds[0]["jit_selected_root"] is False
    assert root_stage_builds[0]["use_fresh_database_payload"] is True
    assert root_stage_builds[0]["objective_names"] == tuple(
        reverse_transport.TRANSPORT_REVERSE_OBJECTIVE_LABELS
    )
    assert runtime_stage_builds == [
        {
            "runtime": runtime,
            "geometry_context": context,
            "n_r": 7,
        }
    ]


def test_database_full_transport_changed_geometry_uses_persistent_live_scan_runtime(
    monkeypatch,
):
    """A changed trial uses the optimization owner, not common runtime setup."""

    class _PreparedInputsReached(RuntimeError):
        pass

    baseline_runtime = object()
    fresh_runtime = object()
    fixed_runtime = object()
    raw_state = object()
    geometry_context = object()
    calls = []

    monkeypatch.setattr(
        reverse_transport,
        "_validate_transport_reverse_parameter_set",
        lambda _parameter_set: None,
    )

    monkeypatch.setattr(
        reverse_transport,
        "build_runtime_context_for_vmec_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("optimization entered common runtime construction")
        ),
    )
    monkeypatch.setattr(
        reverse_transport,
        "build_neopax_geometry_and_ntx_exact_lij_support_from_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("database optimization entered the exact-Lij payload branch")
        ),
    )
    monkeypatch.setattr(
        reverse_transport,
        "realtime_geometry_payload_for_runtime",
        lambda runtime: {"kind": "ntx_scan_runtime"},
    )
    monkeypatch.setattr(
        reverse_transport,
        "split_recorded_ntx_database_runtime",
        lambda runtime: (SimpleNamespace(runtime=fixed_runtime), object()),
    )
    monkeypatch.setattr(
        reverse_transport,
        "prepare_reverse_static_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_PreparedInputsReached()),
    )

    table_context = reverse_transport.RealtimeGeometryTransportReverseTableContext(
        config={"geometry": {"n_radial": 7}},
        baseline_values=jnp.zeros((6,), dtype=jnp.float64),
        baseline_runtime=baseline_runtime,
        baseline_state=object(),
        profile_cfg={},
        neoclassical_cfg={"flux_model": "ntx_scan_runtime"},
    )
    runtime_stage = SimpleNamespace(
        runtime_for_vmec_state=lambda state: calls.append(state) or fresh_runtime,
        cache_size=lambda: 1,
    )
    builder = reverse_transport.internal_realtime_geometry_transport_reverse_table_result_builder(
        table_context=table_context,
        geometry_context=geometry_context,
        raw_block_solve=SimpleNamespace(state=raw_state),
        n_r=7,
        runtime_optimization_stage=runtime_stage,
    )

    try:
        builder(
            ("particle_flux",),
            object(),
            {
                "reverse_stage_mode": "database_full_transport_optimization",
                "use_runtime_payload": False,
            },
        )
    except _PreparedInputsReached:
        pass
    else:  # pragma: no cover - the sentinel must stop the heavy setup
        raise AssertionError("test did not reach reverse static setup")

    assert calls == [raw_state]
    assert builder.optimization_runtime_scan_cache_size() == 1


def test_full_transport_runtime_scan_record_crosses_jit_as_array_payload():
    """The optimization JIT must not return NTX's unregistered record class."""

    @dataclasses.dataclass(frozen=True)
    class _HostRecord:
        surfaces: tuple
        prepared: tuple
        Es: object
        nu_v: object
        grid: object

    template = _HostRecord(
        surfaces=(jnp.asarray([0.0]),),
        prepared=(jnp.asarray([0.0]),),
        Es=jnp.asarray([0.0]),
        nu_v=jnp.asarray([0.0]),
        grid="fixed-grid",
    )

    def _kernel(value):
        record = dataclasses.replace(
            template,
            surfaces=(value,),
            prepared=(2.0 * value,),
            Es=3.0 * value,
            nu_v=4.0 * value,
        )
        return full_transport_stage._scan_primal_record_payload(record)

    payload = jax.jit(_kernel)(jnp.asarray([2.0]))
    rebuilt = full_transport_stage._scan_primal_record_from_payload(
        template, payload
    )

    assert rebuilt.grid == "fixed-grid"
    assert jnp.array_equal(rebuilt.surfaces[0], jnp.asarray([2.0]))
    assert jnp.array_equal(rebuilt.prepared[0], jnp.asarray([4.0]))
    assert jnp.array_equal(rebuilt.Es, jnp.asarray([6.0]))
    assert jnp.array_equal(rebuilt.nu_v, jnp.asarray([8.0]))


def test_database_full_transport_bootstrap_stage_keeps_trial_values_dynamic(
    monkeypatch,
):
    """The terminal bootstrap JIT has one owner and no captured trial leaves."""

    class _Model:
        def __init__(self, geometry, database):
            self.geometry = geometry
            self.database = database

        def evaluate_momentum_corrected_upar_only(self, state):
            return state + self.geometry["g"] + self.database["d"]

        def pullback_momentum_corrected_upar_state_geometry_by_radius(
            self, state, upar_bar, geometry
        ):
            del upar_bar
            return jnp.ones_like(state), {"g": 2.0 * geometry["g"]}

        def pullback_momentum_corrected_upar_database_support_legacy_sparse_by_radius(
            self, state, upar_bar
        ):
            del state, upar_bar
            value = 3.0 * self.database["d"]
            return value, value, value, value, value

    def _runtime_with_payload(_runtime, *, geometry, database):
        return SimpleNamespace(
            models=SimpleNamespace(flux=(geometry, database)),
        )

    monkeypatch.setattr(
        full_transport_stage,
        "runtime_with_fresh_ntx_database_payload",
        _runtime_with_payload,
    )
    monkeypatch.setattr(
        full_transport_stage,
        "find_ntx_database_transport_model_in_model",
        lambda payload: _Model(*payload),
    )
    monkeypatch.setattr(
        reverse_transport,
        "bootstrap_current_softmax_abs_value_and_upar_bar",
        lambda state, _runtime, fluxes: (
            jnp.sum(fluxes["Upar"]),
            jnp.ones_like(state),
        ),
    )
    monkeypatch.setattr(
        reverse_transport,
        "_database_bootstrap_interpolation_bar",
        lambda database, _state, _upar_bar, **_kwargs: {
            "d": 4.0 * database["d"]
        },
    )

    reverse_setup = SimpleNamespace(
        prepared_rollout=SimpleNamespace(
            physics_context=SimpleNamespace(unpack_flat=lambda value: value)
        )
    )
    support0 = {"geometry": {"g": jnp.asarray(1.0)}, "database": {"d": jnp.asarray(2.0)}}
    stage = full_transport_stage.build_database_full_transport_bootstrap_optimization_stage(
        runtime=object(),
        reverse_setup=reverse_setup,
        support_payload=support0,
    )

    value0, final_bar0, support_bar0 = stage.evaluate(jnp.asarray([5.0]), support0)
    support1 = {"geometry": {"g": jnp.asarray(4.0)}, "database": {"d": jnp.asarray(6.0)}}
    value1, final_bar1, support_bar1 = stage.evaluate(jnp.asarray([7.0]), support1)

    assert stage.cache_size() == 1
    assert jnp.allclose(value0, 8.0)
    assert jnp.allclose(value1, 17.0)
    assert jnp.allclose(final_bar0, jnp.ones((1,)))
    assert jnp.allclose(final_bar1, jnp.ones((1,)))
    assert jnp.allclose(support_bar0["geometry"]["g"], 2.0)
    assert jnp.allclose(support_bar1["geometry"]["g"], 8.0)
    assert jnp.allclose(support_bar0["database"]["d"], 8.0)
    assert jnp.allclose(support_bar1["database"]["d"], 24.0)


def test_full_transport_parity_parent_does_not_import_gpu_stack():
    """The coordinator must not own a GPU client while its worker runs."""

    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "optimization"
        / "test_geometry_qi_max_er_transition_bootstrap_initial_root_database_full_transport_parity.py"
    )
    module = ast.parse(script.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in module.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_imports.append(node.module)
    forbidden = {
        "jax",
        "jax.numpy",
        "NEOPAX",
        "optimize_geometry_qi_max_er_transition_bootstrap_initial_root",
    }
    assert forbidden.isdisjoint(top_level_imports)


def test_full_transport_parity_preserves_validated_root_seed():
    """Full transport must start from the same seed as root-only parity."""

    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "optimization"
        / "test_geometry_qi_max_er_transition_bootstrap_initial_root_database_full_transport_parity.py"
    )
    module = ast.parse(script.read_text(encoding="utf-8"))
    builder = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_problem"
    )
    call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "geometry_full_transport_least_squares_problem"
    )
    vmec_input = next(keyword.value for keyword in call.keywords if keyword.arg == "vmec_input")
    assert isinstance(vmec_input, ast.IfExp)
    assert isinstance(vmec_input.test, ast.Compare)
    assert isinstance(vmec_input.body, ast.Attribute)
    assert isinstance(vmec_input.body.value, ast.Name)
    assert (vmec_input.body.value.id, vmec_input.body.attr) == ("base", "SEED_INPUT")
    assert isinstance(vmec_input.orelse, ast.Name)
    assert vmec_input.orelse.id == "vmec_input"


def test_full_transport_parity_covers_softmax_er_and_net_power():
    """Parity must include both terminal objectives used by full transport."""

    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "optimization"
        / "test_geometry_qi_max_er_transition_bootstrap_initial_root_database_full_transport_parity.py"
    )
    module = ast.parse(script.read_text(encoding="utf-8"))
    active_terms = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "active_terms"
    )
    selected_attributes = {
        node.attr for node in ast.walk(active_terms) if isinstance(node, ast.Attribute)
    }
    assert "softmax_Er" in selected_attributes
    assert "net_total_power_volume_average_mw_m3" in selected_attributes

    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"ER_TRANSITION_LEFT_INDEX", "ER_TRANSITION_RIGHT_INDEX"}
    }
    assert constants == {
        "ER_TRANSITION_LEFT_INDEX": 25,
        "ER_TRANSITION_RIGHT_INDEX": 26,
    }


def test_full_transport_transition_objective_indices_are_selectable():
    """Configured terminal-Er cells override only the two transition rows."""

    state = SimpleNamespace(Er=jnp.arange(51, dtype=jnp.float64))
    labels = tuple(reverse_transport.TRANSPORT_REVERSE_OBJECTIVE_LABELS)
    left = reverse_transport.objective_scalar_by_index(
        state,
        object(),
        labels.index("Er_transition_left"),
        er_transition_left_index=25,
        er_transition_right_index=26,
    )
    right = reverse_transport.objective_scalar_by_index(
        state,
        object(),
        labels.index("Er_transition_right"),
        er_transition_left_index=25,
        er_transition_right_index=26,
    )

    assert float(left) == 25.0
    assert float(right) == 26.0


def test_full_transport_smooth_transition_location_ignores_near_axis_zero():
    """The optional rows soft-select an ordered +Er to -Er radial face."""

    rho = jnp.linspace(0.0, 1.0, 11, dtype=jnp.float64)
    er = jnp.asarray(
        [0.0, 8.0, 12.0, 10.0, 6.0, 2.0, -4.0, -8.0, -9.0, -10.0, -10.0],
        dtype=jnp.float64,
    )
    location, strength = (
        reverse_transport.smooth_positive_to_negative_er_transition_metrics(
            er,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )
    flat_er = jnp.zeros_like(er)
    flat_location, flat_strength = (
        reverse_transport.smooth_positive_to_negative_er_transition_metrics(
            flat_er,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )
    flat_location_moment = (
        reverse_transport.smooth_positive_to_negative_er_transition_location_moment(
            flat_er,
            rho,
            target_rho=0.55,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )
    all_negative_strength = (
        reverse_transport.smooth_positive_to_negative_er_transition_strength(
            -jnp.linspace(1.0, 11.0, 11, dtype=jnp.float64),
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )
    all_positive_strength = (
        reverse_transport.smooth_positive_to_negative_er_transition_strength(
            jnp.linspace(11.0, 1.0, 11, dtype=jnp.float64),
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )
    reversed_strength = (
        reverse_transport.smooth_positive_to_negative_er_transition_strength(
            -er,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )
    flat_strength_gradient = jax.grad(
        lambda values: reverse_transport.smooth_positive_to_negative_er_transition_strength(
            values,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )(flat_er)
    all_negative_gradient = jax.grad(
        lambda values: reverse_transport.smooth_positive_to_negative_er_transition_strength(
            values,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            rho_prior=0.55,
            temperature_kv_m=1.0,
            rho_softness=0.05,
            softmax_beta=16.0,
        )
    )(-jnp.ones_like(er) * 10.0)
    all_positive_gradient = jax.grad(
        lambda values: reverse_transport.smooth_positive_to_negative_er_transition_strength(
            values,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            rho_prior=0.55,
            temperature_kv_m=1.0,
            rho_softness=0.05,
            softmax_beta=16.0,
        )
    )(jnp.ones_like(er) * 10.0)
    raw_location = reverse_transport.smooth_positive_to_negative_er_transition_rho(
        er,
        rho,
        rho_min=0.25,
        rho_max=0.75,
        temperature_kv_m=1.0,
    )
    gradient = jax.grad(
        lambda values: reverse_transport.smooth_positive_to_negative_er_transition_rho(
            values,
            rho,
            rho_min=0.25,
            rho_max=0.75,
            temperature_kv_m=1.0,
        )
    )(er)

    assert 0.50 < float(location) < 0.65
    assert float(raw_location) == pytest.approx(float(location))
    assert float(strength) > 0.9
    assert bool(jnp.isfinite(flat_location))
    assert float(flat_strength) < 1.0e-4
    assert float(all_negative_strength) < 0.0
    assert float(all_positive_strength) < 0.0
    assert float(reversed_strength) < 0.0
    assert abs(float(flat_location_moment)) < 1.0e-4
    assert bool(jnp.all(jnp.isfinite(flat_strength_gradient)))
    assert float(jnp.linalg.norm(flat_strength_gradient)) > 0.0
    # Maximizing the soft crossing strength pushes the target's inner side
    # positive and its outer side negative even from either one-sign limit.
    assert float(all_negative_gradient[5]) > 0.0
    assert float(all_positive_gradient[6]) < 0.0
    assert bool(jnp.all(jnp.isfinite(all_negative_gradient)))
    assert bool(jnp.all(jnp.isfinite(all_positive_gradient)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert "Er_transition_rho" not in (
        reverse_transport.TRANSPORT_REVERSE_OBJECTIVE_LABELS
    )
    assert reverse_transport.TRANSPORT_OPTIMIZATION_OPTIONAL_OBJECTIVE_LABELS == (
        "Er_transition_rho",
        "Er_transition_location_moment",
        "Er_transition_strength",
    )


def test_full_transport_parity_perturbation_reuses_only_trial_stage(
    monkeypatch, tmp_path
):
    """Perturbed parity must expose state retained across optimization calls."""

    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "optimization"
        / "test_geometry_qi_max_er_transition_bootstrap_initial_root_database_full_transport_parity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "full_transport_persistent_parity_test_script",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    written_inputs = []

    class Input:
        def to_indata(self, path):
            written_inputs.append(Path(path))
            Path(path).write_text("perturbed", encoding="utf-8")

    problem = SimpleNamespace(
        x0=jnp.asarray([0.0, 0.0]),
        x_scale=jnp.asarray([2.0, 3.0]),
        parameter_labels=("RBC:1:0", "ZBS:1:0"),
        terms=(SimpleNamespace(residual_label="transport:test"),),
        input_from_scaled_parameters=lambda _values: Input(),
    )
    evaluations = []
    build_calls = []

    def fake_build_problem(**kwargs):
        build_calls.append(kwargs)
        return problem

    monkeypatch.setattr(module, "build_problem", fake_build_problem)

    def fake_evaluate(_problem, values):
        point = np.asarray(values, dtype=float).copy()
        evaluations.append(point)
        return jnp.asarray([point.sum()]), jnp.asarray([point])

    monkeypatch.setattr(module, "evaluate", fake_evaluate)

    trial_output = tmp_path / "trial.npz"
    perturbed_input = tmp_path / "input.perturbed"
    module._worker(
        "trial",
        trial_output,
        parameter_index=1,
        parameter_offset=0.25,
        perturbed_input_output=perturbed_input,
    )
    assert written_inputs == [perturbed_input]
    assert len(evaluations) == 2
    np.testing.assert_array_equal(evaluations[0], np.asarray([0.0, 0.0]))
    np.testing.assert_array_equal(evaluations[1], np.asarray([0.0, 0.25]))

    evaluations.clear()
    reference_output = tmp_path / "reference.npz"
    module._worker(
        "reference",
        reference_output,
        parameter_index=1,
        parameter_offset=0.25,
        vmec_input=perturbed_input,
    )
    assert len(evaluations) == 1
    np.testing.assert_array_equal(evaluations[0], np.asarray([0.0, 0.0]))
    assert build_calls[-1]["vmec_input"] == perturbed_input

    evaluations.clear()
    fresh_trial_output = tmp_path / "trial_fresh.npz"
    module._worker(
        "trial_fresh",
        fresh_trial_output,
        parameter_index=1,
        parameter_offset=0.25,
    )
    assert len(evaluations) == 1
    np.testing.assert_array_equal(evaluations[0], np.asarray([0.0, 0.25]))
    assert build_calls[-1]["reverse_stage_mode"] == module.TRIAL_STAGE_MODE
    assert build_calls[-1]["vmec_input"] is None

    evaluations.clear()
    rebased_trial_output = tmp_path / "trial_rebased.npz"
    module._worker(
        "trial_rebased",
        rebased_trial_output,
        parameter_index=1,
        parameter_offset=0.25,
        vmec_input=perturbed_input,
    )
    assert len(evaluations) == 1
    np.testing.assert_array_equal(evaluations[0], np.asarray([0.0, 0.0]))
    assert build_calls[-1]["reverse_stage_mode"] == module.TRIAL_STAGE_MODE
    assert build_calls[-1]["vmec_input"] == perturbed_input

    evaluations.clear()
    materialized_input = tmp_path / "input.materialized"
    module._worker(
        "materialize",
        materialized_input,
        parameter_index=1,
        parameter_offset=0.25,
    )
    assert evaluations == []
    assert written_inputs[-1] == materialized_input


@pytest.mark.parametrize(
    "interpolation_transpose_mode",
    ("established", "legacy_sparse"),
)
def test_database_full_transport_replay_stage_keeps_support_dynamic_and_cache_stable(
    monkeypatch,
    interpolation_transpose_mode,
):
    """The optimization replay compiles once while current support stays live."""

    body_calls = []

    def database_bwd_body(
        execution_context,
        _cotangent_mode,
        segment_reduced_bars,
        step_start_carries,
        _step_primal_records,
        _segment_arrays,
        support,
    ):
        body_calls.append(execution_context)
        physics = execution_context.physics_context
        assert physics.reverse_direct_stage_adjoint is True
        assert physics.reverse_stage_adjoint_solve_mode == "block"
        assert physics.reverse_rhs_transpose_mode == "explicit_database"
        assert physics.reverse_step_bwd_mode == "reduced_cotangent_call_boundary"
        state_bar = physics.flat_rhs_direct_black_box_state_pullback(
            jnp.asarray(0.0),
            jnp.asarray(step_start_carries.y)[0],
            None,
            jnp.asarray(segment_reduced_bars.y)[0],
        )
        # The block/explicit-database stage matrix differentiates flat_rhs.
        # Exercise it here so a persistent template RHS cannot silently keep
        # the first evaluation's geometry while the direct pullback is live.
        state_bar = state_bar + physics.flat_rhs(
            jnp.asarray(0.0),
            jnp.asarray(step_start_carries.y)[0],
        )
        support_bar = physics.flat_rhs_direct_database_split_support_pullback(
            jnp.asarray(0.0),
            jnp.asarray(step_start_carries.y)[0],
            jnp.asarray(segment_reduced_bars.y)[0],
            support,
        )
        active_segment_reduced_bars = dataclasses.replace(
            segment_reduced_bars,
            y=jnp.broadcast_to(
                state_bar,
                jnp.asarray(segment_reduced_bars.y).shape,
            ),
        )
        objective_count = jnp.asarray(segment_reduced_bars.y).shape[0]
        support_bars = tuple(
            jnp.broadcast_to(
                jnp.asarray(leaf)[None, ...],
                (objective_count,) + jnp.shape(leaf),
            )
            for leaf in jax.tree_util.tree_leaves(support_bar)
        )
        return active_segment_reduced_bars, support_bars

    def database_bwd_call(*_args, **_kwargs):
        raise AssertionError("Optimization must not enter the benchmark outer BWD JIT.")

    database_bwd_call.__wrapped__ = database_bwd_body
    monkeypatch.setattr(
        full_transport_stage,
        "_radau_database_segment_reduced_cotangent_bwd_with_table_support_call",
        database_bwd_call,
    )

    def schedule_probe_body(
        execution_context,
        carry,
        *,
        max_total_steps,
        stop_after_accepted_steps,
        capture_segment_length,
    ):
        return (
            execution_context.physics_context.flat_rhs(jnp.asarray(0.0), carry.y),
            jnp.asarray(max_total_steps),
            jnp.asarray(stop_after_accepted_steps),
            jnp.asarray(capture_segment_length),
        )

    monkeypatch.setattr(
        full_transport_stage,
        "_radau_adaptive_schedule_rollout",
        schedule_probe_body,
    )

    @dataclasses.dataclass(frozen=True)
    class EquationSystem:
        support: object

        def with_realtime_geometry_support_payload(self, support):
            return dataclasses.replace(self, support=support)

        def vector_field(self, _t, state, *_args):
            scale = self.support["geometry"] * self.support["database"]
            return -(0.2 + scale) * state + scale

        def pullback_direct_rhs_state(self, _t, _state, _runtime, rhs_bar):
            scale = self.support["geometry"] * self.support["database"]
            return -(0.2 + scale) * rhs_bar

        def pullback_direct_rhs_database_split_support_payload(
            self,
            _t,
            _state,
            _runtime,
            rhs_bar,
            support,
            **_kwargs,
        ):
            # Deliberately read the bound equation owner. This mirrors the
            # production equation-geometry partial and catches a callback
            # retained from the stage's first optimization geometry.
            rhs_scale = jnp.sum(rhs_bar)
            return {
                "geometry": rhs_scale * self.support["geometry"],
                "database": rhs_scale * self.support["database"],
            }

    solver = transport_solvers.RADAUSolver(
        t0=0.0,
        t1=1.0e-3,
        dt=1.0e-4,
        maxiter=4,
        max_steps=4,
        rhs_mode="black_box",
    )
    state = jnp.asarray([0.4])
    support0 = {
        "geometry": jnp.asarray(0.3),
        "database": jnp.asarray(0.7),
    }
    equation_system = EquationSystem(support0)
    prepared = transport_solvers._build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state,
        vector_field=equation_system.vector_field,
        species=None,
    )
    execution_context = transport_solvers._build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared,
    )
    execution_context = dataclasses.replace(
        execution_context,
        physics_context=dataclasses.replace(
            execution_context.physics_context,
            reverse_direct_stage_adjoint=True,
            reverse_stage_adjoint_solve_mode="block",
            reverse_rhs_transpose_mode="explicit_database",
            reverse_step_bwd_mode="reduced_cotangent_call_boundary",
            reverse_database_interpolation_transpose_mode=(
                interpolation_transpose_mode
            ),
        ),
    )
    reverse_setup = SimpleNamespace(
        solver=solver,
        solve_vector_field=equation_system.vector_field,
        prepared_rollout=prepared,
        execution_context=execution_context,
    )
    segment_arrays = (
        jnp.asarray([True]),
        jnp.asarray([1.0e-4]),
        jnp.asarray([1.0e-4]),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([False]),
    )
    stage = full_transport_stage.build_database_full_transport_replay_optimization_stage(
        reverse_setup=reverse_setup,
        species=None,
        support_payload=support0,
        segment_start_carry=prepared.initial_carry,
        segment_arrays=segment_arrays,
    )

    def run(support, active_prepared=prepared):
        return jax.block_until_ready(
            stage.replay(
                initial_flat_state=active_prepared.initial_carry.y,
                support_payload=support,
                segment_start_carry=active_prepared.initial_carry,
                segment_arrays=segment_arrays,
            )
        )

    result0 = run(support0)
    support1 = {
        "geometry": jnp.asarray(0.6),
        "database": jnp.asarray(0.7),
    }
    equation_system1 = EquationSystem(support1)
    state1 = jnp.asarray([0.55])
    prepared1 = transport_solvers._build_prepared_radau_accepted_rollout(
        solver=solver,
        state=state1,
        vector_field=equation_system1.vector_field,
        species=None,
    )
    execution_context1 = transport_solvers._build_prepared_radau_execution_context(
        solver=solver,
        prepared_rollout=prepared1,
    )
    result1 = run(support1, prepared1)

    reference_result0 = jax.block_until_ready(
        transport_solvers._radau_segment_replay_minimal_with_primal_records_call(
            execution_context,
            prepared.initial_carry,
            segment_arrays,
        )
    )
    reference_result1 = jax.block_until_ready(
        transport_solvers._radau_segment_replay_minimal_with_primal_records_call(
            execution_context1,
            prepared1.initial_carry,
            segment_arrays,
        )
    )

    schedule0 = jax.block_until_ready(
        stage.schedule_probe(
            initial_carry=prepared.initial_carry,
            support_payload=support0,
            max_total_steps=4,
            stop_after_accepted_steps=1,
            capture_segment_length=1,
        )
    )
    schedule1 = jax.block_until_ready(
        stage.schedule_probe(
            initial_carry=prepared1.initial_carry,
            support_payload=support1,
            max_total_steps=4,
            stop_after_accepted_steps=1,
            capture_segment_length=1,
        )
    )
    assert stage.cache_size() == 1
    assert stage.schedule_probe_cache_size() == 1
    assert not jnp.allclose(result0[0].y, result1[0].y)
    assert not jnp.allclose(schedule0[0], schedule1[0])
    for trial, reference in (
        (result0, reference_result0),
        (result1, reference_result1),
    ):
        for trial_leaf, reference_leaf in zip(
            jax.tree_util.tree_leaves(trial),
            jax.tree_util.tree_leaves(reference),
            strict=True,
        ):
            assert jnp.allclose(
                trial_leaf,
                reference_leaf,
                rtol=1.0e-12,
                atol=1.0e-12,
                equal_nan=True,
            )

    def seeded_batched_reduced_cotangent(carry):
        def batched_zeros(value):
            return jnp.zeros((1,) + jnp.shape(value), dtype=jnp.asarray(value).dtype)

        return transport_solvers._RadauAcceptedStepReducedCotangent(
            y=jnp.ones((1,) + jnp.shape(carry.y), dtype=jnp.asarray(carry.y).dtype),
            lagged_response_cache=jax.tree_util.tree_map(
                batched_zeros, carry.lagged_response_cache
            ),
            lagged_reference_y=batched_zeros(carry.lagged_reference_y),
        )

    bwd0 = stage.database_segment_bwd(
        initial_flat_state=prepared.initial_carry.y,
        support_payload=support0,
        cotangent_mode="full",
        segment_reduced_bars=seeded_batched_reduced_cotangent(result0[0]),
        step_start_carries=result0[1],
        step_primal_records=result0[2],
        segment_arrays=segment_arrays,
    )
    bwd1 = stage.database_segment_bwd(
        initial_flat_state=prepared1.initial_carry.y,
        support_payload=support1,
        cotangent_mode="full",
        segment_reduced_bars=seeded_batched_reduced_cotangent(result1[0]),
        step_start_carries=result1[1],
        step_primal_records=result1[2],
        segment_arrays=segment_arrays,
    )
    jax.block_until_ready((bwd0, bwd1))
    assert len(body_calls) == 1
    assert stage.database_bwd_cache_size() == 1
    for bwd, support, replay_result in (
        (bwd0, support0, result0),
        (bwd1, support1, result1),
    ):
        scale = support["geometry"] * support["database"]
        step_state = replay_result[1].y[0]
        expected_state_bar = (
            -(0.2 + scale)
            + (-(0.2 + scale) * step_state + scale)
        )
        assert jnp.allclose(bwd[0].y[0], expected_state_bar)
        support_bar = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(support),
            tuple(leaf[0] for leaf in bwd[1]),
        )
        assert jnp.allclose(support_bar["geometry"], support["geometry"])
        assert jnp.allclose(support_bar["database"], support["database"])
    assert not jnp.allclose(bwd0[0].y, bwd1[0].y)
    assert not jnp.allclose(
        jax.tree_util.tree_leaves(bwd0[1])[-1],
        jax.tree_util.tree_leaves(bwd1[1])[-1],
    )

    incompatible_support = {
        "geometry": jnp.asarray(0.6),
        "database": jnp.asarray([0.7]),
    }
    try:
        run(incompatible_support, prepared1)
    except ValueError as exc:
        assert "support shape, dtype, or weak type changed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError(
            "A changed support layout must not create another cache entry."
        )
    assert stage.cache_size() == 1


def test_full_transport_fresh_equations_keep_ntss_density_indices_static(monkeypatch):
    """Optimization replay must not expose fixed species indices as tracers."""

    @dataclasses.dataclass(frozen=True)
    class ErEquation:
        ntss_density_indices: object
        name: str = "Er"

    @dataclasses.dataclass(frozen=True)
    class EquationSystem:
        equations: tuple
        density_equation: object = None
        temperature_equation: object = None
        er_equation: object = None
        config: object = None
        species: object = None
        shared_flux_model: object = None
        source_models: object = None
        solver_cfg: object = None
        boundary_models: object = None

    template_indices = np.asarray([0, 1], dtype=np.int32)
    template_er = ErEquation(template_indices)
    template = EquationSystem(
        equations=(template_er,),
        er_equation=template_er,
        config={},
        species=object(),
        shared_flux_model=object(),
        source_models=(),
        solver_cfg={},
        boundary_models={},
    )
    monkeypatch.setattr(
        full_transport_stage,
        "_replace_geometry_and_fresh_database_payload_in_model",
        lambda model, geometry, database: (model, True),
    )

    def build_equations(*, field, **_kwargs):
        traced_indices = jnp.asarray([0, 1], dtype=jnp.int32) + (
            jnp.asarray(field, dtype=jnp.int32) * 0
        )
        return (ErEquation(traced_indices),)

    monkeypatch.setattr(full_transport_stage, "build_equation_system", build_equations)

    @jax.jit
    def probe(value):
        active = full_transport_stage._equation_system_with_fresh_database_payload(
            template,
            {"geometry": value, "database": value},
            static_ntss_density_indices=template_indices,
        )
        # Mirror the established Radau setup conversion which requires these
        # fixed indices to remain concrete while the outer replay is traced.
        concrete_indices = np.asarray(active.er_equation.ntss_density_indices)
        return value + jnp.asarray(concrete_indices.sum(), dtype=value.dtype)

    assert float(probe(jnp.asarray(2.0))) == 3.0


def test_database_full_transport_replay_skips_template_database_rescaling(monkeypatch):
    """The persistent replay uses the root lane's fresh-database replacement."""

    @dataclasses.dataclass(frozen=True)
    class Equation:
        name: str

    @dataclasses.dataclass(frozen=True)
    class EquationSystem:
        config: object
        species: object
        shared_flux_model: object
        source_models: object
        solver_cfg: object
        boundary_models: object
        equations: tuple = ()
        density_equation: object = None
        temperature_equation: object = None
        er_equation: object = None

        def with_realtime_geometry_support_payload(self, _support):
            raise AssertionError("generic geometry/database replacement must not run")

    template = EquationSystem(
        config="config",
        species="species",
        shared_flux_model="old-model",
        source_models="sources",
        solver_cfg="solver",
        boundary_models="boundaries",
    )
    calls = {}

    def replace(model, geometry, database):
        calls["replacement"] = (model, geometry, database)
        return "fresh-model", True

    def build(**kwargs):
        calls["build"] = kwargs
        return (Equation("density"), Equation("temperature"), Equation("Er"))

    monkeypatch.setattr(
        full_transport_stage,
        "_replace_geometry_and_fresh_database_payload_in_model",
        replace,
    )
    monkeypatch.setattr(full_transport_stage, "build_equation_system", build)
    result = full_transport_stage._equation_system_with_fresh_database_payload(
        template,
        {"geometry": "new-geometry", "database": "new-database"},
    )

    assert calls["replacement"] == (
        "old-model",
        "new-geometry",
        "new-database",
    )
    assert calls["build"]["field"] == "new-geometry"
    assert calls["build"]["flux_model"] == "fresh-model"
    assert result.shared_flux_model == "fresh-model"
    assert result.density_equation.name == "density"
    assert result.temperature_equation.name == "temperature"
    assert result.er_equation.name == "Er"


def test_physical_qi_maxj_adapter_uses_frozen_pitch_and_actual_well_functions(monkeypatch):
    calls = {}

    class _QI:
        @staticmethod
        def j_invariant_qi_residual_from_boozer(**kwargs):
            calls["qi"] = kwargs
            return {"total": jnp.asarray(2.5)}

    class _MaxJ:
        @staticmethod
        def maximum_j_residual_from_boozer(**kwargs):
            calls["maxj"] = kwargs
            return {"total": jnp.asarray(3.5)}

    monkeypatch.setattr(
        geometry_ad,
        "_import_vmec_module",
        lambda name: _QI if name == "core.qi" else _MaxJ,
    )
    settings = geometry_ad.QImaxJBackendSettings(
        backend="physical",
        physical_pitches=(0.51, 0.73),
        physical_nalpha=7,
        physical_points_per_period=24,
        physical_num_periods=3,
        physical_max_wells=8,
        physical_quadrature_order=16,
        physical_maxj_target=-0.02,
    )
    context = SimpleNamespace(
        qi_maxj_settings=settings,
        qi_maxj_physical_pitches=settings.physical_pitches,
        cfg=SimpleNamespace(nfp=2),
        static=SimpleNamespace(s=jnp.asarray([0.0, 0.2, 0.6, 1.0])),
        surface_indices=jnp.asarray([0, 1]),
    )
    booz = {
        "bmnc_b": jnp.asarray([[1.0, 0.1], [1.1, 0.2]]),
        "ixm_b": jnp.asarray([0, 1]),
        "ixn_b": jnp.asarray([0, 0]),
        "iota_b": jnp.asarray([0.4, 0.5]),
        "bvco_b": jnp.asarray([3.0, 3.1]),
        "buco_b": jnp.asarray([0.2, 0.3]),
    }

    result = geometry_ad._vmec_j_invariant_qi_maxj_objectives_from_boozer(
        context, booz, include_qi=True, include_maxj=True
    )

    assert float(result["qi_objective"]) == 2.5
    assert float(result["maxj_objective"]) == 3.5
    assert jnp.allclose(calls["qi"]["pitch"], jnp.asarray([0.51, 0.73]))
    assert jnp.allclose(calls["maxj"]["pitch"], jnp.asarray([0.51, 0.73]))
    assert calls["maxj"]["target"] == -0.02
    assert calls["maxj"]["nalpha"] == 7
    assert jnp.allclose(calls["maxj"]["psi_b"], jnp.asarray([0.1, 0.4]))
    assert calls["maxj"]["psi_edge"] == 1.0


def test_qi_maxj_backend_settings_preserve_surrogate_default_and_aliases():
    assert geometry_ad.normalize_qi_maxj_backend_settings().backend == "surrogate"
    assert geometry_ad.normalize_qi_maxj_backend_settings("old").backend == "surrogate"
    assert geometry_ad.normalize_qi_maxj_backend_settings("new").backend == "physical"
    configured = geometry_ad.normalize_qi_maxj_backend_settings(
        {"backend": "physical", "physical_pitches": [0.5, 0.75]}
    )
    assert configured.physical_pitches == (0.5, 0.75)
    explicit = optimization._resolve_qi_maxj_settings(
        {"backend": "physical", "trapping_depths": [0.25, 0.8]},
    )
    assert explicit.backend == "physical"
    assert explicit.trapping_depths == (0.25, 0.8)


def test_database_full_transport_profile_problem_uses_configured_direct_builder(
    monkeypatch,
):
    """Profile optimization keeps the configured Er cells and database lane."""

    config = {
        "geometry": {"n_radial": 51},
        "neoclassical": {
            "flux_model": "ntx_scan_runtime",
            "ntx_scan_n_theta": 25,
            "ntx_scan_n_zeta": 31,
            "ntx_scan_n_xi": 64,
            "ntx_scan_surface_backend": "vmec",
        },
        "profiles": {
            "n0": 4.21,
            "T0": 17.8,
            "density_shape_power": 10.0,
            "temperature_shape_power": 2.0,
        },
        "transport_solver": {},
    }
    runtime = SimpleNamespace()
    baseline_state = SimpleNamespace(pressure=jnp.ones((1, 2), dtype=jnp.float64))
    builder = object()
    builder_calls = []
    root_stage = object()

    monkeypatch.setattr(
        optimization,
        "_prepare_full_transport_config",
        lambda value, **_kwargs: copy.deepcopy(value),
    )
    monkeypatch.setattr(
        optimization,
        "build_runtime_context",
        lambda _config: (runtime, baseline_state),
    )
    monkeypatch.setattr(
        optimization,
        "build_database_initial_root_experiment_stage",
        lambda **_kwargs: root_stage,
    )

    def _builder_factory(**kwargs):
        builder_calls.append(kwargs)
        return builder

    monkeypatch.setattr(
        optimization,
        "internal_realtime_geometry_transport_reverse_table_result_builder",
        _builder_factory,
    )

    problem = optimization.full_transport_profile_least_squares_problem(
        config,
        (
            (optimization.transport.Er_transition_left, 26.0, 0.09),
            (optimization.transport.Er_transition_right, -10.0, 0.09),
        ),
        accepted_step_limit=None,
        reverse_segment_length=50,
        max_reverse_accepted_steps=500,
        er_transition_left_index=25,
        er_transition_right_index=26,
        reverse_stage_adjoint_solve_mode="block",
        reverse_rhs_transpose_mode="explicit_database",
        reverse_step_bwd_mode="reduced_cotangent_call_boundary",
        reverse_stage_mode="database_full_transport_optimization",
    )

    assert len(builder_calls) == 1
    call = builder_calls[0]
    assert call["er_transition_left_index"] == 25
    assert call["er_transition_right_index"] == 26
    assert call["reverse_segment_length"] == 50
    assert call["max_reverse_accepted_steps"] == 500
    assert call["initial_root_optimization_stage"] is root_stage
    assert problem.table_result_builder is builder
    assert problem.run_grouped_report is None
    assert problem.options["Er_transition_left_index"] == 25
    assert problem.options["Er_transition_right_index"] == 26
    assert problem.options["reverse_stage_mode"] == (
        "database_full_transport_optimization"
    )
