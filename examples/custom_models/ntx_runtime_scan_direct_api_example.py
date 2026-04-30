"""Direct Python example for the runtime NTX scan transport model.

This example shows the intended "setup once, evaluate later" pattern:

1. preload the VMEC/Boozer-derived static channels once
2. construct the runtime NTX scan model with those channels attached
3. defer the runtime monoenergetic database build until the model is evaluated

The example is intentionally focused on model construction rather than running
the full transport solver.
"""

from __future__ import annotations

import NEOPAX


def build_runtime_ntx_model():
    vmec_file = "examples/inputs/wout_QI_nfp2_newNT_opt_hires.nc"
    boozer_file = "examples/inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc"

    rho_scan = [0.25, 0.5, 0.75]
    nu_v_scan = [1.0e-5, 1.0e-4, 1.0e-3]
    er_tilde_scan = [0.0, 1.0e-5, 3.0e-5, 1.0e-4]

    channels = NEOPAX.build_ntx_runtime_scan_channels(
        vmec_file,
        boozer_file,
        rho_scan,
    )

    model = NEOPAX.build_ntx_runtime_scan_transport_model(
        species="species-placeholder",
        energy_grid="energy-grid-placeholder",
        geometry="geometry-placeholder",
        vmec_file=vmec_file,
        boozer_file=boozer_file,
        ntx_scan_rho=rho_scan,
        ntx_scan_nu_v=nu_v_scan,
        ntx_scan_er_tilde=er_tilde_scan,
        ntx_scan_channels=channels,
        prebuild_database=False,
    )

    updated_model = model.with_scan_inputs(
        nu_v_scan=[2.0e-5, 2.0e-4, 2.0e-3],
        er_tilde_scan=[0.0, 2.0e-5, 6.0e-5, 2.0e-4],
    )

    return channels, model, updated_model


if __name__ == "__main__":
    channels, model, updated_model = build_runtime_ntx_model()
    print("Prepared runtime NTX scan channels for rho grid:", channels.rho)
    print("Model type:", type(model).__name__)
    print("Database prebuilt:", model.database is not None)
    print("Updated model reuses channels:", updated_model.channels is channels)
