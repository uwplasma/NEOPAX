"""FD/VMEX routing regressions; no equilibrium or transport solves."""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@dataclasses.dataclass
class Parameters:
    rbc: object
    zbs: object


def helpers(family):
    path = Path(__file__).resolve().parents[1] / "examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py"
    module = ast.parse(path.read_text())
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_param_unit_tangent_like", "_manual_implicit_forward_state_tangent",
        "_frozen_linearized_vmec_geometry_bundle",
        "_runtime_for_frozen_linearized_geometry_step", "_geometry_fd_objectives",
        "_state_with_frozen_linearized_initial_er_root",
    }]
    params = Parameters(jnp.ones((2, 3)) * 2.0, jnp.ones((2, 3)) * 3.0)
    cfg, state, mask, tangent, runtime = (object() for _ in range(5))
    context = SimpleNamespace(mboz=36, nboz=11, indata=object())
    entry = dict(input_field=family.lower(), n_offset=1, m_index=1,
                 baseline_coefficient=2.0 if family == "RBC" else 3.0)
    implicit = SimpleNamespace(
        implicit_state_tangent_raw_block=Mock(return_value=tangent),
        _dof_mask=Mock(side_effect=AssertionError("Unsupported default GPU force evaluator")),
        make_config=Mock(return_value=cfg), runtime_from_params=Mock(return_value=runtime),
    )
    namespace = dict(
        jax=jax, jnp=jnp, dataclasses=dataclasses, im=implicit,
        _geometry_context_from_config=Mock(return_value=context),
        _geometry_param_specs_from_parameter_name=Mock(return_value=((family, 1, 0),)),
        boundary_param_entries=Mock(return_value=(entry,)),
        _implicit_params_with_boundary_deltas=Mock(return_value=params),
        _solve_state_for_single_param=Mock(return_value=state),
        geometry_raw_block_solve_from_param_vector=Mock(return_value=SimpleNamespace(
            implicit_params=params, implicit_cfg=cfg, state=state, dof_mask=mask,
        )),
    )
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, *selected], type_ignores=[])), str(path), "exec"), namespace)
    return namespace, SimpleNamespace(params=params, cfg=cfg, state=state, mask=mask,
                                     tangent=tangent, runtime=runtime, context=context)


@pytest.mark.parametrize("family", ["RBC", "ZBS"])
@pytest.mark.parametrize("retain_raw_solve", [False, True])
def test_ad_fd_reuses_primal_mask_and_config_without_private_gpu_evaluator(family, retain_raw_solve):
    namespace, data = helpers(family)
    bundle = namespace["_frozen_linearized_vmec_geometry_bundle"](
        {"geometry": {"vmec_lane": "ad", "vmec_max_iter": 7, "n_radial": 9}},
        f"{family}:1:0", baseline_delta=0.125, retain_raw_solve=retain_raw_solve,
    )
    raw = namespace["geometry_raw_block_solve_from_param_vector"]
    assert bundle["raw_block_solve"] is (raw.return_value if retain_raw_solve else None)
    args, kwargs = raw.call_args
    assert args[0] is data.context and args[2] == ((family, 1, 0),)
    np.testing.assert_array_equal(args[1], [0.125])
    assert kwargs == {"max_iter": 7, "solver_device": "default"}
    implicit = namespace["im"]
    for call in (implicit._dof_mask, implicit.make_config, implicit.runtime_from_params,
                 namespace["_solve_state_for_single_param"], namespace["_implicit_params_with_boundary_deltas"]):
        call.assert_not_called()
    args, kwargs = implicit.implicit_state_tangent_raw_block.call_args
    assert all(actual is expected for actual, expected in zip(args[:4],
               (data.params, data.cfg, data.state, data.mask), strict=True))
    assert kwargs == {"probe_chunk_size": 1}
    for name in ("rbc", "zbs"):
        expected = np.zeros((2, 3))
        if name == family.lower():
            expected[1, 1] = 1.0
        np.testing.assert_array_equal(getattr(args[4], name), expected)
    assert bundle["state_star"] is data.state and bundle["state_tangent"] is data.tangent
    assert bundle["n_r"] == 9
    assert bundle["coefficient_value"] == (2.0 if family == "RBC" else 3.0) + 0.125


def test_forward_reference_still_linearizes_its_configured_forward_primal():
    namespace, data = helpers("RBC")
    implicit = namespace["im"]
    implicit._dof_mask.side_effect = None
    implicit._dof_mask.return_value = data.mask
    bundle = namespace["_frozen_linearized_vmec_geometry_bundle"](
        {"geometry": {"vmec_max_iter": 7, "vmec_step_size": 0.2}},
        "RBC:1:0", baseline_delta=0.125,
    )
    namespace["geometry_raw_block_solve_from_param_vector"].assert_not_called()
    call = namespace["_solve_state_for_single_param"].call_args
    assert call.args[0] is data.context and float(call.args[1]) == 0.125
    assert call.kwargs["lane"] == "forward" and call.kwargs["step_size"] == 0.2
    implicit.make_config.assert_called_once_with(data.context.indata, mode="cli", multigrid=True, max_iterations=7)
    implicit._dof_mask.assert_called_once_with(data.state, data.runtime, data.cfg)
    assert bundle["state_star"] is data.state and bundle["state_tangent"] is data.tangent


def test_geometry_fd_keeps_one_vmec_tangent_and_forwards_frozen_root_reference():
    namespace, _ = helpers("RBC")
    equilibrium = {"R": jnp.asarray([2.0, 3.0]), "Z": jnp.asarray([4.0, 5.0])}
    tangent = {"R": jnp.asarray([0.2, -0.1]), "Z": jnp.asarray([-0.3, 0.4])}
    context, baseline_root, residual, root_derivative, profile_values = (object() for _ in range(5))
    bundle = dict(state_star=equilibrium, state_tangent=tangent, context=context, n_r=51)
    build = Mock(side_effect=lambda cfg, ctx, state, **kw: (SimpleNamespace(vmec=state), object()))
    endpoint = Mock(return_value=object())
    namespace.update(build_runtime_context_for_vmec_state=build,
                     _objectives_on_realtime_geometry_frozen_trace=endpoint)
    config = {"geometry": {"vmec_param_delta": 0.125}}
    for h in (-1e-4, 1e-4):
        namespace["_geometry_fd_objectives"](
            config=config, geometry_parameter="RBC:1:0", geometry_delta=0.125 + h,
            profile_values=profile_values, profile_cfg={}, frozen_trace=object(),
            replay_mode="accepted", fd_accepted_replay_contract="reverse_frozen_metadata",
            baseline_replay_reference=object(), geometry_fd_lane="frozen_linearized",
            frozen_linearized_bundle=bundle, initial_er_root_ad="jax_selected_root",
            initial_er_root_fd_root_lane="frozen_linearized", baseline_er_profile=baseline_root,
            baseline_residual=residual, baseline_dres_der=root_derivative,
        )
        assert build.call_args.args[1] is context
        assert build.call_args.kwargs == {"n_r": 51}
        for name in equilibrium:
            np.testing.assert_allclose(build.call_args.args[2][name],
                                       equilibrium[name] + h * tangent[name], rtol=1e-14)
        kwargs = endpoint.call_args.kwargs
        assert kwargs["runtime"].vmec is build.call_args.args[2]
        assert kwargs["profile_values"] is profile_values
        assert kwargs["initial_er_root_fd_root_lane"] == "frozen_linearized"
        assert kwargs["baseline_er_profile"] is baseline_root
        assert kwargs["baseline_residual"] is residual
        assert kwargs["baseline_dres_der"] is root_derivative
    namespace["geometry_raw_block_solve_from_param_vector"].assert_not_called()
    namespace["_solve_state_for_single_param"].assert_not_called()


@pytest.mark.parametrize("parameter_kind", ["profile", "geometry"])
def test_frozen_root_branch_moves_by_the_implicit_rule_without_reselection(parameter_kind):
    namespace, _ = helpers("RBC")

    @dataclasses.dataclass
    class State:
        Er: object
        profile: object

    root = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float64)
    slope = jnp.asarray([2.0, -3.0, 0.0], dtype=jnp.float64)
    bias = jnp.asarray([0.01, -0.02, 0.03], dtype=jnp.float64)
    profile_partial = jnp.asarray([0.3, -0.7, 0.9], dtype=jnp.float64)
    geometry_partial = jnp.asarray([-0.4, 0.2, 0.8], dtype=jnp.float64)
    baseline = State(root, jnp.asarray(0.0))
    profile_builder = Mock(side_effect=lambda value, **kw: State(kw["baseline_state"].Er, value))

    def residual(state, er, *, runtime):
        return slope * (er - root) + bias + state.profile * profile_partial + runtime.geometry * geometry_partial

    namespace.update(_profile_state_from_values=profile_builder,
                     initial_er_charge_flux_residuals=residual)

    def endpoint(delta):
        return namespace["_state_with_frozen_linearized_initial_er_root"](
            runtime=SimpleNamespace(geometry=delta if parameter_kind == "geometry" else 0.0),
            baseline_state=baseline, profile_cfg={},
            profile_values=delta if parameter_kind == "profile" else 0.0,
            baseline_er_profile=root, baseline_residual=bias, baseline_dres_der=slope,
        ).Er

    partial = profile_partial if parameter_kind == "profile" else geometry_partial
    expected = np.asarray([-partial[0] / slope[0], -partial[1] / slope[1], 0.0])
    np.testing.assert_array_equal(endpoint(0.0), root)
    h = 1e-4
    np.testing.assert_allclose((endpoint(h) - endpoint(-h)) / (2 * h), expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(jax.jacfwd(endpoint)(0.0), expected, rtol=1e-13, atol=1e-14)
    assert all(call.kwargs["initial_er_root_ad"] == "off" for call in profile_builder.call_args_list)
