import numpy as np
from netCDF4 import Dataset

from NEOPAX._geometry_models import VmecBoozer


def test_vmec_vprime_integrates_to_volume_p():
    vmec_path = "examples/inputs/wout_QI_nfp2_newNT_opt_hires.nc"
    booz_path = "examples/inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc"

    geometry = VmecBoozer(n_r=51, vmec=vmec_path, booz=booz_path)

    with Dataset(vmec_path, mode="r") as vfile:
        expected_volume = float(np.asarray(vfile.variables["volume_p"][:]).squeeze())

    integrated_volume = float(np.trapezoid(np.asarray(geometry.Vprime), x=np.asarray(geometry.r_grid)))

    np.testing.assert_allclose(integrated_volume, expected_volume, rtol=5e-3, atol=0.0)
