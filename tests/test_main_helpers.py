from pathlib import Path
import types

import h5py
import jax.numpy as jnp

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
    build_ntx_exact_lij_runtime_transport_model,
    build_ntx_runtime_scan_channels,
    build_ntx_runtime_scan_transport_model,
    get_transport_flux_model,
)


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

    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.build_local_particle_flux_evaluator",
        fake_build_local,
    )
    monkeypatch.setattr(
        "NEOPAX._transport_flux_models.NTXDatabaseTransportModel.evaluate_face_fluxes",
        fake_face,
    )

    assert model.build_local_particle_flux_evaluator("state") == "local_eval"
    assert model.evaluate_face_fluxes("state", "face_state", marker=True) == "face_eval"
    assert calls[0] == ("local", "runtime_db", "state")
    assert calls[1] == ("face", "runtime_db", "state", "face_state", {"marker": True})


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
