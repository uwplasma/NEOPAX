"""Unit checks for optimization-only reuse of VMEX raw-block setup."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from NEOPAX import _geometry_autodiff as geometry_ad
from NEOPAX import _optimization_initial_root_stage as initial_root_stage
from NEOPAX import optimization
from NEOPAX._reverse_ad_optimization import normalize_geometry_full_ad_objective_names


def test_main_qi_maxj_adapter_reuses_the_existing_boozer_payload(monkeypatch):
    """QI/max-J consume one NEOPAX Boozer table, never a state-native transform."""

    calls = {}

    class _QI:
        @staticmethod
        def j_invariant_qi_residual_from_boozer(**kwargs):
            calls["qi"] = kwargs
            return {"total": jnp.sum(kwargs["bmnc_b"]) + jnp.sum(kwargs["G_b"])}

    class _MaxJ:
        @staticmethod
        def common_trapped_pitches(bmag, depths):
            calls["pitch_bmag"] = bmag
            calls["pitch_depths"] = depths
            return jnp.asarray([0.51, 0.63, 0.78])

        @staticmethod
        def maximum_j_residual_from_boozer(**kwargs):
            calls["maxj"] = kwargs
            return {"total": jnp.sum(kwargs["bmnc_b"]) + jnp.sum(kwargs["I_b"])}

    def _module(name):
        if name == "core.qi":
            return _QI
        if name == "core.maxj":
            return _MaxJ
        raise AssertionError(name)

    monkeypatch.setattr(geometry_ad, "_import_vmec_module", _module)
    context = SimpleNamespace(
        cfg=SimpleNamespace(nfp=2),
        static=SimpleNamespace(s=jnp.asarray([0.0, 0.3, 0.7, 1.0])),
        surface_indices=jnp.asarray([0, 1]),
        qi_maxj_trapping_depths=(0.35, 0.55, 0.75),
    )
    booz = {
        "bmnc_b": jnp.asarray([[2.0, 0.1], [2.1, 0.2]]),
        "ixm_b": jnp.asarray([0, 1]),
        "ixn_b": jnp.asarray([0, 0]),
        "iota_b": jnp.asarray([0.4, 0.5]),
        "bvco_b": jnp.asarray([3.0, 3.1]),
        "buco_b": jnp.asarray([0.2, 0.3]),
    }

    result = geometry_ad._vmec_j_invariant_qi_maxj_objectives_from_boozer(
        context, booz, include_qi=True, include_maxj=True
    )

    assert float(result["qi_objective"]) == 10.5
    assert float(result["maxj_objective"]) == 4.9
    assert calls["qi"]["G_b"] is booz["bvco_b"]
    assert calls["qi"]["I_b"] is booz["buco_b"]
    assert calls["maxj"]["psi_edge"] == 1.0
    assert jnp.allclose(calls["maxj"]["psi_b"], jnp.asarray([0.15, 0.5]))
    assert jnp.allclose(calls["qi"]["pitch"], jnp.asarray([0.51, 0.63, 0.78]))
    assert "common_trapped_pitches_state" not in _MaxJ.__dict__


def test_main_dmerc_adapter_uses_state_runtime_path(monkeypatch):
    """The main-VMEX DMerc lane is state-native and does not need WOUT."""

    runtime = object()
    calls = {}

    class _Stability:
        @staticmethod
        def d_merc_state(state, received_runtime):
            calls["state"] = state
            calls["runtime"] = received_runtime
            return jnp.asarray([0.0, 1.5, 2.5, 0.0])

    monkeypatch.setattr(geometry_ad, "_using_current_vmec_jax_context", lambda _context: True)
    monkeypatch.setattr(
        geometry_ad,
        "_import_vmec_module",
        lambda name: _Stability if name == "core.stability" else (_ for _ in ()).throw(AssertionError(name)),
    )
    context = SimpleNamespace(static=SimpleNamespace(runtime=runtime))

    result = geometry_ad._vmec_dmerc_profile_from_state(context, "converged-state")

    assert jnp.allclose(result, jnp.asarray([0.0, 1.5, 2.5, 0.0]))
    assert calls == {"state": "converged-state", "runtime": runtime}


def test_main_mercier_residual_adapter_preserves_vmex_row_contract(monkeypatch):
    """Reverse AD receives VMEX's interior softplus residual rows unchanged."""

    runtime = object()
    calls = {}

    class _Stability:
        @staticmethod
        def mercier_stability_residual(state, received_runtime, *, margin, smoothing):
            calls.update(
                state=state,
                runtime=received_runtime,
                margin=margin,
                smoothing=smoothing,
            )
            return jnp.asarray([0.0, 2.0e-4, 0.0])

    monkeypatch.setattr(geometry_ad, "_using_current_vmec_jax_context", lambda _context: True)
    monkeypatch.setattr(
        geometry_ad,
        "_import_vmec_module",
        lambda name: _Stability if name == "core.stability" else (_ for _ in ()).throw(AssertionError(name)),
    )
    context = SimpleNamespace(static=SimpleNamespace(runtime=runtime))

    rows = geometry_ad.vmec_mercier_stability_residual_from_state(
        context, "converged-state", margin=5.0e-4, smoothing=1.0e-5
    )

    assert jnp.allclose(rows, jnp.asarray([0.0, 2.0e-4, 0.0]))
    assert calls == {
        "state": "converged-state",
        "runtime": runtime,
        "margin": 5.0e-4,
        "smoothing": 1.0e-5,
    }


def test_main_mercier_softmax_objective_reduces_physical_rows(monkeypatch):
    """The benchmark exposes one zero-preserving smooth worst-row objective."""

    class _Stability:
        @staticmethod
        def mercier_stability_residual(_state, _runtime, *, margin, smoothing):
            assert margin == 0.0
            assert smoothing == 1.0e-6
            return jnp.asarray([0.0, 2.0e-4, 5.0e-4])

    monkeypatch.setattr(geometry_ad, "_using_current_vmec_jax_context", lambda _context: True)
    monkeypatch.setattr(
        geometry_ad,
        "_import_vmec_module",
        lambda name: _Stability if name == "core.stability" else (_ for _ in ()).throw(AssertionError(name)),
    )
    context = SimpleNamespace(static=SimpleNamespace(runtime=object()))

    value = geometry_ad.vmec_mercier_stability_softmax_objective_from_state(
        context, "converged-state", temperature=1.0e-3
    )
    rows = jnp.asarray([0.0, 2.0e-4, 5.0e-4])
    expected = jnp.sum(jax.nn.softmax(rows / 1.0e-3) * rows)
    assert jnp.allclose(value, expected)
    assert float(value) > float(jnp.mean(rows))
    assert float(value) < float(jnp.max(rows))


def test_main_mercier_softmax_is_a_canonical_full_geometry_row():
    """FD and reverse table selection refer to exactly the same scalar row."""

    names = geometry_ad.geometry_observable_names_for_kind(
        "geometry_full_ad_objectives"
    )
    assert names[-1] == "vmec_dmerc_stability_softmax"
    assert normalize_geometry_full_ad_objective_names(("dmerc",)) == (
        "vmec_dmerc_stability_softmax",
    )


def test_main_mercier_reverse_table_keeps_one_row_per_interior_surface(monkeypatch):
    """The table preserves VMEX's DMerc[2:-1] rows instead of averaging them."""

    class _Stability:
        @staticmethod
        def mercier_stability_residual(state, _runtime, *, margin, smoothing):
            assert margin == 0.0
            assert smoothing == 1.0e-6
            return state[2:-1]

    raw_block_solve = SimpleNamespace(
        state=jnp.asarray([9.0, 8.0, 0.1, 0.2, 0.3, 7.0]),
        param_entries="entries",
    )
    context = SimpleNamespace(
        static=SimpleNamespace(runtime=object(), s=jnp.linspace(0.0, 1.0, 6)),
    )
    monkeypatch.setattr(geometry_ad, "_using_current_vmec_jax_context", lambda _context: True)
    monkeypatch.setattr(
        geometry_ad,
        "_import_vmec_module",
        lambda name: _Stability if name == "core.stability" else (_ for _ in ()).throw(AssertionError(name)),
    )
    monkeypatch.setattr(
        geometry_ad,
        "geometry_raw_block_transpose_from_state_bars",
        lambda _solve, bars, **_kwargs: bars,
    )
    monkeypatch.setattr(
        geometry_ad,
        "_param_vector_gradient_from_implicit_param_grads",
        lambda bars, _entries: bars,
    )

    names, values, gradients = geometry_ad.geometry_dmerc_stability_reverse_table_from_param_vector(
        context,
        jnp.asarray([0.0]),
        (("RBC", 1, 0),),
        raw_block_solve=raw_block_solve,
    )

    assert names == (
        "vmec_dmerc_stability_s2",
        "vmec_dmerc_stability_s3",
        "vmec_dmerc_stability_s4",
    )
    assert jnp.allclose(values, jnp.asarray([0.1, 0.2, 0.3]))
    assert jnp.allclose(gradients, jnp.eye(6)[2:-1])


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
