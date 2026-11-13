"""Check reference accounting and the plotted overlay without physics solves."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from NEOPAX._orchestrator import _load_ntss_reference_profiles, plot_transport_solution
from NEOPAX._state import TransportState


def reference_file(path, *, radiation=None):
    with h5py.File(path, "w") as stream:
        stream["r"] = [0., .5, 1.]
        stream["ne"] = [2., 2., 2.]
        stream["Te"] = [4., 4., 4.]
        stream["Zeff"] = [1., 1., 1.]
        stream["AlphaPower"] = [10., 10., 10.]
        stream["ECRHPower"] = [2., 2., 2.]
        stream["NBIPower_e"] = [3., 3., 3.]
        stream["NBIPower_I"] = [4., 4., 4.]
        if radiation is not None:
            stream["Prad"] = [radiation] * 3


def test_explicit_ntss_radiation_is_subtracted_from_reference_heating(tmp_path):
    path = tmp_path / "reference.h5"
    reference_file(path, radiation=5.)
    data = _load_ntss_reference_profiles(path, jnp.asarray([0., .25, .5, .75, 1.]))
    np.testing.assert_allclose(data["scalar"]["power_sources_total"], 10. + 2. + 3. + 4. - 5.)
    np.testing.assert_allclose(data["scalar"]["alpha_power"], 10.)
    np.testing.assert_allclose(data["scalar"]["pbrems_power"], 5.)


def test_missing_ntss_radiation_uses_active_ntss_coefficient(tmp_path):
    path = tmp_path / "reference.h5"
    reference_file(path)
    data = _load_ntss_reference_profiles(path, jnp.asarray([0., .5, 1.]))
    radiation = .334 * 1. * 2.**2 * np.sqrt(4.) / 62.422
    np.testing.assert_allclose(data["scalar"]["pbrems_power"], radiation)
    np.testing.assert_allclose(data["scalar"]["power_sources_total"], 19. - radiation)


def test_source_plot_uses_net_ntss_overlay(tmp_path, monkeypatch):
    """Exercise the plot's nested loader, not only the standalone importer."""
    from matplotlib.figure import Figure

    path = tmp_path / "reference.h5"
    reference_file(path, radiation=5.)
    density = jnp.ones((1, 4, 3))
    saved = TransportState(density=density, pressure=density, Er=jnp.zeros((1, 3)))
    species = SimpleNamespace(
        species_idx={name: i for i, name in enumerate(("e", "D", "T", "He"))},
        names=("e", "D", "T", "He"), ion_indices=(1, 2, 3),
    )
    captured = {}

    def save_figure(figure, filename, **_kwargs):
        if Path(filename).name == "transport_power_sources_total.png":
            captured["title"] = figure.axes[0].get_title()
            captured["curves"] = {
                line.get_label(): np.asarray(line.get_ydata())
                for line in figure.axes[0].lines
            }

    monkeypatch.setattr(Figure, "savefig", save_figure)
    plot_transport_solution(
        jnp.asarray([0., .5, 1.]),
        {"ys": saved, "ts": jnp.asarray([0.])},
        tmp_path,
        final_reference_file=path,
        source_models={
            "temperature": lambda state: {
                "AlphaPower": jnp.full(3, 19. * 62.422),
                "PBrems": jnp.full(3, 5. * 62.422),
            }
        },
        species=species,
    )
    assert captured["title"] == "Net Power Sources vs rho"
    np.testing.assert_allclose(captured["curves"]["Final ref"], 14.)
    np.testing.assert_allclose(captured["curves"]["t=0"], 14.)
