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
def test_ad_fd_reuses_primal_mask_and_config_without_private_gpu_evaluator(family):
    namespace, data = helpers(family)
    bundle = namespace["_frozen_linearized_vmec_geometry_bundle"](
        {"geometry": {"vmec_lane": "ad", "vmec_max_iter": 7, "n_radial": 9}},
        f"{family}:1:0", baseline_delta=0.125,
    )
    raw = namespace["geometry_raw_block_solve_from_param_vector"]
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
