import types
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
from ntx import GridSpec, example_surface, prepare_monoenergetic_system

from NEOPAX._orchestrator import (
    _build_database,
    _build_flux_model,
    _load_user_extensions,
    _load_ntss_reference_profiles,
    _normalize_solver_config,
    _resolve_reference_path,
)
from NEOPAX._monoenergetic import (
    MONOENERGETIC_KIND_GENERIC,
    MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED,
    MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D,
    load_monoenergetic_database,
    monoenergetic_database_kind,
)
from NEOPAX._database_preprocessed import (
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1D,
    PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU,
)
from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel
from NEOPAX._interpolators import get_Dij
from NEOPAX._source_models import get_source_model
from NEOPAX._transport_flux_models import (
    NTXExactLijRuntimeTransportModel,
    NTXRuntimeScanChannels,
    NTXRuntimeScanTransportModel,
    _sanitize_float_delta_bar_tree,
    _ntx_runtime_scan_to_neopax_monoenergetic,
    build_ntx_exact_lij_runtime_transport_model,
    build_ntx_runtime_scan_channels,
    build_ntx_runtime_scan_transport_model,
    get_transport_flux_model,
)


def test_ntx_exact_fused_lowdot_local_pullback_matches_ntx_helper():
    """The opt-in fused local NTX path must preserve the scalar local bars."""
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    common = dict(
        species=object(),
        energy_grid=energy_grid,
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    reference = NTXExactLijRuntimeTransportModel(**common)
    fused = reference.with_derivative_pullback_algebra("ntx_helper_lowdot_fused")
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    field_bars = (
        jnp.asarray(0.2),
        jnp.asarray([0.4, -0.2, 0.1, 0.3, -0.5, 0.2]),
        jnp.asarray([-0.3, 0.2, 0.4, -0.1, 0.5, 0.2]),
        jnp.asarray([0.1, 0.3, -0.2, 0.5, -0.4, 0.2]),
    )
    kwargs = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.5e-2, 2.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, -2.0e-3, 1.5e-3]),
        vth_a=jnp.asarray(2.3),
        field_bars=field_bars,
    )
    reference_bars = reference._pullback_interpolated_moment_reduced_local_outputs(**kwargs)
    fused_bars = fused._pullback_interpolated_moment_reduced_local_outputs(**kwargs)
    for fused_bar, reference_bar in zip(fused_bars, reference_bars, strict=True):
        assert jnp.allclose(fused_bar, reference_bar, rtol=1.0e-9, atol=1.0e-11)


def test_ntx_exact_factorized_two_directional_local_response_matches_generic_jvps():
    """The isolated rebuild primitive must match the existing local response."""
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    model = NTXExactLijRuntimeTransportModel(
        species=object(),
        energy_grid=energy_grid,
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(3, 3, 2))
    nu_hat = jnp.asarray([1.0e-2, 1.5e-2, 2.0e-2])
    epsi_hat = jnp.asarray([1.0e-3, -2.0e-3, 1.5e-3])
    vth_a = jnp.asarray(2.3)
    drds = jnp.asarray(1.2)

    def _response(prepared_value, drds_value, *, factorized):
        return model._interpolated_moment_reduced_local_outputs_from_primitives(
            prepared_value,
            drds_value=drds_value,
            nu_hat_a=nu_hat,
            epsi_hat_a=epsi_hat,
            vth_a=vth_a,
            use_factorized_ntx_two_directional_prepared_vjp=factorized,
        )

    reference = _response(prepared, drds, factorized=False)
    factorized = jax.jit(
        lambda prepared_value, drds_value: _response(
            prepared_value, drds_value, factorized=True
        )
    )(prepared, drds)
    for actual, expected in zip(factorized, reference, strict=True):
        assert jnp.allclose(actual, expected, rtol=1.0e-9, atol=1.0e-11)


def test_ntx_exact_support_only_prepared_pullback_matches_joint_helper():
    """The isolated rebuild helper preserves prepared, ``drds``, and primal fields."""
    energy_grid = types.SimpleNamespace(
        xWeights=jnp.asarray([0.2, 0.3, 0.5]),
        L11_weight=jnp.asarray([1.0, 0.8, 1.2]),
        L12_weight=jnp.asarray([0.1, -0.2, 0.3]),
        L22_weight=jnp.asarray([0.9, 1.1, 0.7]),
        L13_weight=jnp.asarray([0.4, 0.5, 0.6]),
        L23_weight=jnp.asarray([-0.3, 0.2, 0.1]),
        L33_weight=jnp.asarray([1.3, 0.6, 0.9]),
        v_norm=jnp.asarray([1.7, 1.8, 1.9]),
    )
    model = NTXExactLijRuntimeTransportModel(
        species=object(),
        energy_grid=energy_grid,
        geometry=object(),
        vmec_file=None,
        boozer_file=None,
    )
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    kwargs = dict(
        prepared=prepared,
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.5e-2, 2.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, -2.0e-3, 1.5e-3]),
        vth_a=jnp.asarray(2.3),
        field_bars=(
            jnp.asarray(0.2),
            jnp.asarray([0.4, -0.2, 0.1, 0.3, -0.5, 0.2]),
            jnp.asarray([-0.3, 0.2, 0.4, -0.1, 0.5, 0.2]),
            jnp.asarray([0.1, 0.3, -0.2, 0.5, -0.4, 0.2]),
        ),
    )
    joint = model._pullback_interpolated_moment_reduced_local_outputs_with_prepared_support_and_drds(
        **kwargs
    )
    support_only = jax.jit(
        lambda: model._pullback_interpolated_moment_prepared_support_and_drds_only(**kwargs)
    )()
    expected_primal = model._interpolated_moment_reduced_local_outputs_from_primitives(
        prepared,
        drds_value=kwargs["drds_value"],
        nu_hat_a=kwargs["reference_nu_hat"],
        epsi_hat_a=kwargs["reference_epsi_hat"],
        vth_a=kwargs["vth_a"],
    )
    actual_prepared, actual_drds, actual_primal = support_only
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual_prepared),
        jax.tree_util.tree_leaves(joint[3]),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=1.0e-9, atol=1.0e-11)
    assert jnp.allclose(actual_drds, joint[4], rtol=1.0e-9, atol=1.0e-11)
    for actual_field, expected_field in zip(actual_primal, expected_primal, strict=True):
        assert jnp.allclose(actual_field, expected_field, rtol=1.0e-9, atol=1.0e-11)

    batched_field_bars = tuple(
        jnp.stack([field_bar, field_bar], axis=0)
        for field_bar in kwargs["field_bars"]
    )
    def _sanitized_support_only(first_bar, second_bar, third_bar, fourth_bar):
        prepared_bar, drds_bar, primal_response = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only(
                prepared,
                drds_value=kwargs["drds_value"],
                reference_nu_hat=kwargs["reference_nu_hat"],
                reference_epsi_hat=kwargs["reference_epsi_hat"],
                vth_a=kwargs["vth_a"],
                field_bars=(first_bar, second_bar, third_bar, fourth_bar),
            )
        )
        return (
            *jax.tree_util.tree_leaves(
                _sanitize_float_delta_bar_tree(prepared, prepared_bar)
            ),
            drds_bar,
            *primal_response,
        )

    batched_support_only = jax.jit(jax.vmap(_sanitized_support_only))(
        *batched_field_bars
    )
    expected_batched = tuple(
        jnp.broadcast_to(value, (2,) + jnp.asarray(value).shape)
        for value in _sanitized_support_only(*kwargs["field_bars"])
    )
    for actual_leaf, expected_leaf in zip(batched_support_only, expected_batched, strict=True):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=1.0e-9, atol=1.0e-11)


def test_normalize_solver_config_prefers_transport_solver_section():
    config = {
        "transport_solver": {
            "transport_solver_backend": "theta_newton",
            "density_floor": 2.5e-6,
        },
        "solver": {
            "integrator": "radau",
        },
        "neoclassical": {"flux_model": "ntx_database"},
        "turbulence": {"flux_model": "none"},
    }

    out = _normalize_solver_config(config)
    assert out["transport_solver_backend"] == "theta_newton"
    assert out["integrator"] == "theta_newton"
    assert out["neoclassical_flux_model"] == "ntx_database"
    assert out["turbulence_flux_model"] == "none"
    assert out["density_floor"] == 2.5e-6
    assert out["Er_relax"] == 1.0
    assert out["DEr"] == 1.0


def test_normalize_solver_config_falls_back_to_legacy_solver_section():
    config = {
        "solver": {
            "integrator": "radau",
        },
        "neoclassical": {"flux_model": "none"},
        "turbulence": {"flux_model": "turbulent_power_analytical"},
    }

    out = _normalize_solver_config(config)
    assert out["transport_solver_backend"] == "radau"
    assert out["integrator"] == "radau"
    assert out["density_floor"] == 1.0e-6
    assert out["turbulence_flux_model"] == "turbulent_power_analytical"


def test_resolve_reference_path_handles_relative_paths(tmp_path, monkeypatch):
    ref = tmp_path / "ref.h5"
    ref.write_bytes(b"test")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_reference_path("ref.h5")
    assert resolved == ref.resolve()


def test_resolve_reference_path_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_reference_path("missing.h5") is None


def test_load_monoenergetic_database_dispatches_from_mode(monkeypatch):
    geometry = types.SimpleNamespace(a_b=1.2)

    monkeypatch.setattr(
        "NEOPAX._monoenergetic.PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU.read_ntx",
        classmethod(lambda cls, a_b, ntx_file: {"kind": "fixed", "a_b": a_b, "file": ntx_file}),
    )

    out = load_monoenergetic_database(
        geometry,
        "db.h5",
        "preprocessed_3d_ntss1d_fixed",
    )

    assert out == {"kind": "fixed", "a_b": 1.2, "file": "db.h5"}


def test_build_database_uses_shared_monoenergetic_loader(monkeypatch):
    captured = {}

    def fake_loader(geometry, ntx_file, interpolation_mode):
        captured["geometry"] = geometry
        captured["file"] = ntx_file
        captured["mode"] = interpolation_mode
        return "database"

    monkeypatch.setattr("NEOPAX._orchestrator.load_monoenergetic_database", fake_loader)

    geometry = types.SimpleNamespace(a_b=1.5)
    config = {"neoclassical": {"neoclassical_file": "scan.h5", "interpolation_mode": "preprocessed_ntss"}}
    out = _build_database(config, geometry)

    assert out == "database"
    assert captured == {"geometry": geometry, "file": "scan.h5", "mode": "preprocessed_ntss"}


def test_monoenergetic_database_kind_defaults_to_generic():
    assert monoenergetic_database_kind(object()) == MONOENERGETIC_KIND_GENERIC


def test_monoenergetic_database_kind_prefers_most_specific_subclass():
    fixed = object.__new__(PreprocessedMonoenergetic3DNTSSRadiusNTSS1DFixedNU)
    ntss1d = object.__new__(PreprocessedMonoenergetic3DNTSSRadiusNTSS1D)
    assert monoenergetic_database_kind(fixed) == MONOENERGETIC_KIND_PREPROCESSED_3D_NTSS1D_FIXED
    assert monoenergetic_database_kind(ntss1d) == MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D


def test_monoenergetic_interpolation_kernel_defaults_to_generic():
    assert monoenergetic_interpolation_kernel(object()) is get_Dij


def test_load_ntss_reference_profiles_interpolates_scalar_and_species_profiles(tmp_path, monkeypatch):
    path = tmp_path / "profiles.h5"
    with h5py.File(path, "w") as f:
        f["r"] = jnp.array([0.0, 0.5, 1.0])
        f["Er"] = jnp.array([0.0, 1.0, 2.0])
        f["ne"] = jnp.array([10.0, 20.0, 30.0])
        f["nD"] = jnp.array([1.0, 2.0, 3.0])
        f["Te"] = jnp.array([100.0, 200.0, 300.0])
        f["TD"] = jnp.array([400.0, 500.0, 600.0])
        f["Tt"] = jnp.array([700.0, 800.0, 900.0])
        f["Vr"] = jnp.ones(3)
        f["FluxQe"] = jnp.array([7.0, 8.0, 9.0])
        f["FluxQI"] = jnp.array([4.0, 5.0, 6.0])

    monkeypatch.chdir(tmp_path)
    rho = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = _load_ntss_reference_profiles("profiles.h5", rho)

    assert jnp.allclose(out["Er"], jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert jnp.allclose(out["density"]["e"], jnp.array([10.0, 15.0, 20.0, 25.0, 30.0]))
    assert jnp.allclose(out["density"]["D"], jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert jnp.allclose(out["density"]["T"], jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert jnp.allclose(out["temperature"]["e"], jnp.array([100.0, 150.0, 200.0, 250.0, 300.0]))
    assert jnp.allclose(out["temperature"]["D"], jnp.array([400.0, 450.0, 500.0, 550.0, 600.0]))
    assert jnp.allclose(out["temperature"]["T"], jnp.array([700.0, 750.0, 800.0, 850.0, 900.0]))
    assert jnp.allclose(out["flux_species"]["Q_total"]["e"], jnp.array([7.0, 7.5, 8.0, 8.5, 9.0]))


def test_load_user_extensions_imports_python_modules(monkeypatch):
    imported = []

    def fake_import_module(name):
        imported.append(name)
        return types.SimpleNamespace(__name__=name)

    monkeypatch.setattr("NEOPAX._orchestrator.importlib.import_module", fake_import_module)
    _load_user_extensions({"extensions": {"python_modules": ["pkg.a", "pkg.b"]}})
    assert imported == ["pkg.a", "pkg.b"]


def test_load_user_extensions_imports_python_files_relative_to_config_dir(tmp_path):
    mod_path = tmp_path / "user_models.py"
    mod_path.write_text("MARKER = 1\n", encoding="utf-8")
    _load_user_extensions(
        {
            "_config_dir": str(tmp_path),
            "extensions": {"python_files": ["user_models.py"]},
        }
    )


def test_load_user_extensions_registers_custom_models_from_python_file(tmp_path):
    mod_path = tmp_path / "user_models.py"
    mod_path.write_text(
        "\n".join(
            [
                "import dataclasses",
                "import jax.numpy as jnp",
                "import NEOPAX",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class FileFluxModel:",
                "    def __call__(self, state, geometry=None, params=None):",
                "        del geometry, params",
                "        base = jnp.ones_like(state.density)",
                "        return {'Gamma': base, 'Q': 2.0 * base, 'Upar': jnp.zeros_like(base)}",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class FileSourceModel:",
                "    def __call__(self, state):",
                "        return {'pressure_source': jnp.ones_like(state.pressure)}",
                "",
                "NEOPAX.register_transport_flux_model('file_registered_flux', FileFluxModel)",
                "NEOPAX.register_source_model('file_registered_source', FileSourceModel)",
            ]
        ),
        encoding="utf-8",
    )
    _load_user_extensions(
        {
            "_config_dir": str(tmp_path),
            "extensions": {"python_files": ["user_models.py"]},
        }
    )

    flux_builder = get_transport_flux_model("file_registered_flux")
    source_builder = get_source_model("file_registered_source")
    assert flux_builder is not None
    assert source_builder is not None


def test_load_user_extensions_registers_custom_models_from_python_module(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "userpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "\n".join(
            [
                "import dataclasses",
                "import jax.numpy as jnp",
                "import NEOPAX",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class ModuleFluxModel:",
                "    def __call__(self, state, geometry=None, params=None):",
                "        del geometry, params",
                "        base = jnp.ones_like(state.density)",
                "        return {'Gamma': base, 'Q': 3.0 * base, 'Upar': jnp.zeros_like(base)}",
                "",
                "@dataclasses.dataclass(frozen=True, eq=False)",
                "class ModuleSourceModel:",
                "    def __call__(self, state):",
                "        return {'pressure_source': 2.0 * jnp.ones_like(state.pressure)}",
                "",
                "NEOPAX.register_transport_flux_model('module_registered_flux', ModuleFluxModel)",
                "NEOPAX.register_source_model('module_registered_source', ModuleSourceModel)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    _load_user_extensions({"extensions": {"python_modules": ["userpkg"]}})

    flux_builder = get_transport_flux_model("module_registered_flux")
    source_builder = get_source_model("module_registered_source")
    assert flux_builder is not None
    assert source_builder is not None


def test_build_flux_model_passes_runtime_ntx_scan_inputs(monkeypatch):
    captured = {}

    def fake_get_transport_flux_model(name):
        def factory(*args, **kwargs):
            captured[name] = {"args": args, "kwargs": kwargs}
            return f"{name}_instance"

        return factory

    monkeypatch.setattr("NEOPAX._orchestrator.get_transport_flux_model", fake_get_transport_flux_model)
    monkeypatch.setattr(
        "NEOPAX._orchestrator.build_transport_flux_model",
        lambda neo, turb, classical, include_turbulent_particle_flux=True: {
            "neo": neo,
            "turb": turb,
            "classical": classical,
            "include_turbulent_particle_flux": include_turbulent_particle_flux,
        },
    )

    out = _build_flux_model(
        {
            "geometry": {
                "vmec_file": "wout.nc",
                "boozer_file": "boozmn.nc",
            },
            "neoclassical": {
                "flux_model": "ntx_scan_runtime",
                "ntx_scan_rho": [0.25, 0.5],
                "ntx_scan_nu_v": [1.0e-4, 1.0e-3],
                "ntx_scan_er_tilde": [0.0, 1.0e-4],
            },
            "turbulence": {"flux_model": "none"},
            "classical": {"flux_model": "none"},
        },
        species="species",
        energy_grid="grid",
        geometry="geometry",
        database="db",
        source_models=None,
    )

    assert out["neo"] == "ntx_scan_runtime_instance"
    assert captured["ntx_scan_runtime"]["kwargs"]["vmec_file"] == "wout.nc"
    assert captured["ntx_scan_runtime"]["kwargs"]["boozer_file"] == "boozmn.nc"
    assert captured["ntx_scan_runtime"]["kwargs"]["ntx_scan_rho"] == [0.25, 0.5]


def test_build_flux_model_passes_runtime_ntx_exact_lij_inputs(monkeypatch):
    captured = {}

    def fake_get_transport_flux_model(name):
        def factory(*args, **kwargs):
            captured[name] = {"args": args, "kwargs": kwargs}
            return f"{name}_instance"

        return factory

    monkeypatch.setattr("NEOPAX._orchestrator.get_transport_flux_model", fake_get_transport_flux_model)
    monkeypatch.setattr(
        "NEOPAX._orchestrator.build_transport_flux_model",
        lambda neo, turb, classical, include_turbulent_particle_flux=True: {
            "neo": neo,
            "turb": turb,
            "classical": classical,
            "include_turbulent_particle_flux": include_turbulent_particle_flux,
        },
    )

    out = _build_flux_model(
        {
            "geometry": {
                "vmec_file": "wout.nc",
                "boozer_file": "boozmn.nc",
            },
            "neoclassical": {
                "flux_model": "ntx_exact_lij_runtime",
                "ntx_exact_n_theta": 19,
                "ntx_exact_n_zeta": 21,
                "ntx_exact_n_xi": 48,
            },
            "turbulence": {"flux_model": "none"},
            "classical": {"flux_model": "none"},
        },
        species="species",
        energy_grid="grid",
        geometry="geometry",
        database="db",
        source_models=None,
    )

    assert out["neo"] == "ntx_exact_lij_runtime_instance"
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["vmec_file"] == "wout.nc"
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["boozer_file"] == "boozmn.nc"
    assert captured["ntx_exact_lij_runtime"]["kwargs"]["ntx_exact_n_theta"] == 19


def test_build_ntx_runtime_scan_transport_model_can_skip_prebuild():
    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        prebuild_database=False,
    )

    assert isinstance(model, NTXRuntimeScanTransportModel)
    assert model.database is None
    assert model.vmec_file == "wout.nc"
    assert model.boozer_file == "boozmn.nc"


def test_ntx_runtime_scan_database_keeps_radius_local_er_axis():
    scan = types.SimpleNamespace(
        rho=jnp.asarray([0.25, 0.5]),
        nu_v=jnp.asarray([1.0e-4, 1.0e-3]),
        Er=jnp.asarray(
            [
                [1.0e-6, 2.0e-6],
                [3.0e-6, 6.0e-6],
            ]
        ),
        drds=jnp.asarray([2.0, 4.0]),
        D11=jnp.ones((2, 2, 2)),
        D13=2.0 * jnp.ones((2, 2, 2)),
        D33=3.0 * jnp.ones((2, 2, 2)),
    )

    database = _ntx_runtime_scan_to_neopax_monoenergetic(scan, a_b=2.0)

    expected_er_list = jnp.log10(jnp.maximum(1.0e-8, jnp.abs(scan.Er) / (2.0 * scan.rho[:, None])))
    assert jnp.allclose(database.Er_list, expected_er_list)
    assert not jnp.allclose(database.Er_list[1], database.Er_list[0] + jnp.log10(scan.rho[0] / scan.rho[1]))
    assert jnp.allclose(10.0 ** database.D11_log, scan.D11 * scan.drds[:, None, None] ** 2)
    assert jnp.allclose(database.D13, scan.D13 * scan.drds[:, None, None])
    assert jnp.allclose(database.D33, scan.D33 * scan.nu_v[None, :, None])


def test_build_ntx_exact_lij_runtime_transport_model_can_skip_preload():
    model = build_ntx_exact_lij_runtime_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        preload_support=False,
    )

    assert isinstance(model, NTXExactLijRuntimeTransportModel)
    assert model.support is None
    assert model.vmec_file == "wout.nc"
    assert model.boozer_file == "boozmn.nc"


def test_build_ntx_runtime_scan_channels_uses_loader(monkeypatch):
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models._load_ntx_vmec_boozer_channels",
        lambda vmec_file, boozer_file, rho: {
            "a_b": 1.5,
            "psia": 2.5,
            "b00": rho + 1.0,
            "r00": rho + 2.0,
            "boozer_i": rho + 3.0,
            "boozer_g": rho + 4.0,
            "iota": rho + 5.0,
            "drds": rho + 6.0,
            "dr_tildedr": rho + 7.0,
            "dr_tildeds": rho + 8.0,
            "fac_reference_to_sfincs_11": rho + 9.0,
            "fac_reference_to_sfincs_31": rho + 10.0,
            "fac_reference_to_sfincs_33": rho + 11.0,
            "fac_sfincs_to_dkes_11": rho + 12.0,
            "fac_sfincs_to_dkes_31": rho + 13.0,
            "fac_sfincs_to_dkes_33": rho + 14.0,
            "fac_dkes_to_d11star": rho + 15.0,
            "fac_dkes_to_d31star": rho + 16.0,
            "fac_dkes_to_d33star": rho + 17.0,
        },
    )

    channels = build_ntx_runtime_scan_channels("wout.nc", "boozmn.nc", [0.25, 0.5])

    assert isinstance(channels, NTXRuntimeScanChannels)
    assert jnp.allclose(channels.rho, jnp.array([0.25, 0.5]))
    assert channels.a_b == 1.5
    assert jnp.allclose(channels.dr_tildeds, jnp.array([8.25, 8.5]))


def test_build_ntx_runtime_scan_transport_model_can_preload_channels(monkeypatch):
    sentinel = NTXRuntimeScanChannels(
        rho=jnp.array([0.25, 0.5]),
        a_b=1.0,
        psia=2.0,
        b00=jnp.array([1.0, 1.1]),
        r00=jnp.array([2.0, 2.1]),
        boozer_i=jnp.array([3.0, 3.1]),
        boozer_g=jnp.array([4.0, 4.1]),
        iota=jnp.array([5.0, 5.1]),
        drds=jnp.array([6.0, 6.1]),
        dr_tildedr=jnp.array([7.0, 7.1]),
        dr_tildeds=jnp.array([8.0, 8.1]),
        fac_reference_to_sfincs_11=jnp.array([9.0, 9.1]),
        fac_reference_to_sfincs_31=jnp.array([10.0, 10.1]),
        fac_reference_to_sfincs_33=jnp.array([11.0, 11.1]),
        fac_sfincs_to_dkes_11=jnp.array([12.0, 12.1]),
        fac_sfincs_to_dkes_31=jnp.array([13.0, 13.1]),
        fac_sfincs_to_dkes_33=jnp.array([14.0, 14.1]),
        fac_dkes_to_d11star=jnp.array([15.0, 15.1]),
        fac_dkes_to_d31star=jnp.array([16.0, 16.1]),
        fac_dkes_to_d33star=jnp.array([17.0, 17.1]),
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.build_ntx_runtime_scan_channels",
        lambda vmec_file, boozer_file, rho_scan: sentinel,
    )

    model = build_ntx_runtime_scan_transport_model(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        ntx_scan_rho=[0.25, 0.5],
        ntx_scan_nu_v=[1.0e-4, 1.0e-3],
        ntx_scan_er_tilde=[0.0, 1.0e-4],
        preload_channels=True,
        prebuild_database=False,
    )

    assert model.database is None
    assert model.channels is sentinel


def test_ntx_runtime_scan_transport_model_delegates_face_and_local_evaluators(monkeypatch):
    model = NTXRuntimeScanTransportModel(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        rho_scan=[0.25],
        nu_v_scan=[1.0e-4],
        er_tilde_scan=[0.0],
        database=None,
    )
    calls = []

    monkeypatch.setattr(
        NTXRuntimeScanTransportModel,
        "_build_runtime_database",
        lambda self: "runtime_db",
    )

    def fake_build_local(self, state):
        calls.append(("local", self.database, state))
        return "local_eval"

    def fake_face(self, state, face_state, **kwargs):
        calls.append(("face", self.database, state, face_state, kwargs))
        return "face_eval"

    def fake_build_lagged(self, state, **kwargs):
        calls.append(("build_lagged", self.database, state, kwargs))
        return "face_lagged_response"

    def fake_eval_lagged(self, state, lagged_response, **kwargs):
        calls.append(("eval_lagged", self.database, state, lagged_response, kwargs))
        return "lagged_face_fluxes"

    def fake_pullback_lagged(self, state, lagged_response_bar, **kwargs):
        calls.append(("pullback_lagged", self.database, state, lagged_response_bar, kwargs))
        return "state_bar"

    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.build_local_particle_flux_evaluator",
        fake_build_local,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.evaluate_face_fluxes",
        fake_face,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.build_lagged_response",
        fake_build_lagged,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.evaluate_with_lagged_response",
        fake_eval_lagged,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.pullback_build_lagged_response",
        fake_pullback_lagged,
    )

    assert model.build_local_particle_flux_evaluator("state") == "local_eval"
    assert model.evaluate_face_fluxes("state", "face_state", marker=True) == "face_eval"
    assert model.build_lagged_response("state", marker=True) == "face_lagged_response"
    assert model.evaluate_with_lagged_response("state", "response", marker=True) == "lagged_face_fluxes"
    assert model.pullback_build_lagged_response("state", "response_bar", marker=True) == "state_bar"
    assert calls[0] == ("local", "runtime_db", "state")
    assert calls[1] == ("face", "runtime_db", "state", "face_state", {"marker": True})
    assert calls[2] == ("build_lagged", "runtime_db", "state", {"marker": True})
    assert calls[3] == ("eval_lagged", "runtime_db", "state", "response", {"marker": True})
    assert calls[4] == ("pullback_lagged", "runtime_db", "state", "response_bar", {"marker": True})


def test_ntx_runtime_scan_transport_model_with_scan_inputs_preserves_channels_for_same_rho():
    channels = NTXRuntimeScanChannels(
        rho=jnp.array([0.25, 0.5]),
        a_b=1.0,
        psia=2.0,
        b00=jnp.array([1.0, 1.1]),
        r00=jnp.array([2.0, 2.1]),
        boozer_i=jnp.array([3.0, 3.1]),
        boozer_g=jnp.array([4.0, 4.1]),
        iota=jnp.array([5.0, 5.1]),
        drds=jnp.array([6.0, 6.1]),
        dr_tildedr=jnp.array([7.0, 7.1]),
        dr_tildeds=jnp.array([8.0, 8.1]),
        fac_reference_to_sfincs_11=jnp.array([9.0, 9.1]),
        fac_reference_to_sfincs_31=jnp.array([10.0, 10.1]),
        fac_reference_to_sfincs_33=jnp.array([11.0, 11.1]),
        fac_sfincs_to_dkes_11=jnp.array([12.0, 12.1]),
        fac_sfincs_to_dkes_31=jnp.array([13.0, 13.1]),
        fac_sfincs_to_dkes_33=jnp.array([14.0, 14.1]),
        fac_dkes_to_d11star=jnp.array([15.0, 15.1]),
        fac_dkes_to_d31star=jnp.array([16.0, 16.1]),
        fac_dkes_to_d33star=jnp.array([17.0, 17.1]),
    )
    model = NTXRuntimeScanTransportModel(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        rho_scan=[0.25, 0.5],
        nu_v_scan=[1.0e-4, 1.0e-3],
        er_tilde_scan=[0.0, 1.0e-4],
        channels=channels,
        database="cached_db",
    )

    updated = model.with_scan_inputs(
        nu_v_scan=[2.0e-4, 2.0e-3],
        er_tilde_scan=[1.0e-5, 2.0e-4],
    )

    assert updated.channels is channels
    assert updated.database is None
    assert updated.nu_v_scan == [2.0e-4, 2.0e-3]
    assert updated.er_tilde_scan == [1.0e-5, 2.0e-4]


def test_ntx_runtime_scan_transport_model_with_scan_inputs_drops_channels_for_new_rho():
    channels = NTXRuntimeScanChannels(
        rho=jnp.array([0.25, 0.5]),
        a_b=1.0,
        psia=2.0,
        b00=jnp.array([1.0, 1.1]),
        r00=jnp.array([2.0, 2.1]),
        boozer_i=jnp.array([3.0, 3.1]),
        boozer_g=jnp.array([4.0, 4.1]),
        iota=jnp.array([5.0, 5.1]),
        drds=jnp.array([6.0, 6.1]),
        dr_tildedr=jnp.array([7.0, 7.1]),
        dr_tildeds=jnp.array([8.0, 8.1]),
        fac_reference_to_sfincs_11=jnp.array([9.0, 9.1]),
        fac_reference_to_sfincs_31=jnp.array([10.0, 10.1]),
        fac_reference_to_sfincs_33=jnp.array([11.0, 11.1]),
        fac_sfincs_to_dkes_11=jnp.array([12.0, 12.1]),
        fac_sfincs_to_dkes_31=jnp.array([13.0, 13.1]),
        fac_sfincs_to_dkes_33=jnp.array([14.0, 14.1]),
        fac_dkes_to_d11star=jnp.array([15.0, 15.1]),
        fac_dkes_to_d31star=jnp.array([16.0, 16.1]),
        fac_dkes_to_d33star=jnp.array([17.0, 17.1]),
    )
    model = NTXRuntimeScanTransportModel(
        species="species",
        energy_grid="grid",
        geometry="geometry",
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        rho_scan=[0.25, 0.5],
        nu_v_scan=[1.0e-4, 1.0e-3],
        er_tilde_scan=[0.0, 1.0e-4],
        channels=channels,
        database="cached_db",
    )

    updated = model.with_scan_inputs(rho_scan=[0.2, 0.6])

    assert updated.channels is None
    assert updated.database is None
    assert updated.rho_scan == [0.2, 0.6]


def test_build_ntx_exact_lij_runtime_transport_model_can_preload_support(monkeypatch):
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.build_ntx_exact_lij_runtime_support",
        lambda *args, **kwargs: "sentinel_support",
    )

    model = build_ntx_exact_lij_runtime_transport_model(
        species="species",
        energy_grid="grid",
        geometry=types.SimpleNamespace(
            a_b=1.0,
            r_grid=jnp.array([0.25, 0.5]),
            r_grid_half=jnp.array([0.125, 0.375, 0.625]),
        ),
        vmec_file="wout.nc",
        boozer_file="boozmn.nc",
        preload_support=True,
    )

    assert model.support == "sentinel_support"
