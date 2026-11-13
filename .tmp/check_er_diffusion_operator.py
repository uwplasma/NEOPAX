"""Five cross-checks of the Er diffusion claim; eight cells, no physics runtime."""
import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from NEOPAX._boundary_conditions import build_boundary_condition_model
from NEOPAX._orchestrator import _normalize_solver_config, _normalized_boundary_cfg_for_transport
from NEOPAX._state import TransportState
from NEOPAX._transport_equations import (
    _resolve_er_boundary_mode,
    build_electric_field_equation,
)
from NEOPAX._fem import conservative_update

root = Path(__file__).resolve().parents[1]
ntss = root.parent / "NTSS-software-master/NTSSfusion"
flux_code = (ntss / "CPlasma/CPlasma_Fluxes.cpp").read_text()
rhs_code = (ntss / "CPlasma/CPlasma_RHS.cpp").read_text()
checks = []

assert "Erk = U[EqEr]" in flux_code
assert "dEr  = Ur[EqEr]" in flux_code
assert "Ur = (u[k+1]-u[k])/h[k]" in flux_code
checks.append({"check":1,"name":"NTSS variables are physical Er and its radial derivative; no Er/r redefinition","passed":True})

expression = re.search(r"GEr\s*=\s*([^;]+);",flux_code).group(1)
assert re.sub(r"\s+","",expression) == "-dEr+Erk/rk"
assert "FE  = DEcoeff*GEr" in flux_code
assert "Fx[EqEr]  = FE" in flux_code
assert "Zk = -(A&(F15-F05)) + Src" in rhs_code
assert "D[EqEr]  = 1." in flux_code
assert "S[EqEr]  = Vprime" in flux_code
assert "double ViD = 0" in flux_code
assert "CPlasma_Fluxes.o" in (ntss / "linux/makefileNTSSfus").read_text()
checks.append({"check":2,"name":"NTSS build, conservative divergence and source retain the Er/r flux contribution","passed":True,"flux_expression":expression})

faces = jnp.linspace(0.,1.,9)
r = .5*(faces[:-1]+faces[1:])
field = SimpleNamespace(r_grid=r,r_grid_half=faces,Vprime=r,Vprime_half=faces,
    B0=jnp.ones_like(r),enlogation=jnp.ones_like(r),iota=jnp.ones_like(r))
cfg = tomllib.loads((root / "examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_match_THe.toml").read_text())
settings = _normalize_solver_config(cfg)
normalized_boundary = _normalized_boundary_cfg_for_transport(cfg["boundary"])
bc = build_boundary_condition_model(normalized_boundary["Er"],float(faces[1]-faces[0]))
boundary_mode = _resolve_er_boundary_mode(cfg,settings)
eq = build_electric_field_equation(field,tuple(cfg["species"]["names"]),None,
    jnp.array(cfg["species"]["mass_mp"]),jnp.array(cfg["species"]["charge_qp"]),None,bc,
    Er_relax=settings["Er_relax"],DEr=settings["DEr"],
    source_mode=settings["Er_source_mode"],permitivity_mode=settings["Er_permittivity_mode"],
    boundary_mode=boundary_mode)
interior = slice(1,-2)  # Do not compare different axis/outer boundary stencils.
linear_flux,linear_diff = eq._er_diffusion(2*r)
np.testing.assert_allclose(np.asarray(linear_flux)[1:-2],-2.,rtol=0,atol=1e-12)
np.testing.assert_allclose(np.asarray(linear_diff)[interior],np.asarray(2/r)[interior],rtol=0,atol=1e-11)
checks.append({"check":3,"name":"Actual NEOPAX factory/face_grad/FVM gives scalar radial diffusion, without hidden Er/r","passed":True})

assert eq.permitivity_mode == "ntss_like_midpoint"
assert eq.source_mode == "ambipolar_local"
assert eq.DEr == 2. and eq.Er_relax == .1
assert eq.boundary_mode == "floating_ambipolar_edge"
state = TransportState(density=jnp.ones((4,8)),pressure=jnp.ones((4,8)),Er=2*r)
rhs = eq(state,fluxes={"Gamma":jnp.zeros((4,8)),"Gamma_faces":jnp.zeros((4,9))})
np.testing.assert_allclose(np.asarray(rhs)[interior],np.asarray(.1*2*2/r)[interior],atol=1e-11)
checks.append({"check":4,"name":"Current TOML options and complete NEOPAX Er RHS retain this diffusion; ntss_like_midpoint does not replace it","passed":True})

examples = []
for power in (1,2):
    er = 2*r if power == 1 else r*r
    slope = jnp.full_like(faces,2.) if power == 1 else 2*faces
    ef = 2*faces if power == 1 else faces*faces
    # Evaluate the actual NTSS source expression at interior faces.
    ntss_flux = jnp.asarray([eval(expression,{"__builtins__":{}},{"dEr":float(slope[i]),"Erk":float(ef[i]),"rk":float(faces[i])}) for i in range(1,9)])
    ntss_flux = jnp.concatenate([jnp.array([0.]),ntss_flux])
    ntss_diff = conservative_update(ntss_flux,jnp.diff(faces),field.Vprime,field.Vprime_half)
    _,neopax_diff = eq._er_diffusion(er)
    expected_ntss = jnp.zeros_like(r) if power == 1 else jnp.full_like(r,2.)
    expected_neopax = 2/r if power == 1 else jnp.full_like(r,4.)
    np.testing.assert_allclose(np.asarray(ntss_diff)[interior],np.asarray(expected_ntss)[interior],atol=1e-11)
    np.testing.assert_allclose(np.asarray(neopax_diff)[interior],np.asarray(expected_neopax)[interior],atol=1e-11)
    examples.append({"Er":"2*r" if power==1 else "r*r","r":np.asarray(r)[interior].tolist(),
        "NEOPAX_diffusion_before_DEr_and_relax":np.asarray(neopax_diff)[interior].tolist(),
        "NTSS_expression_diffusion_before_DEr_and_relax":np.asarray(ntss_diff)[interior].tolist()})
checks.append({"check":5,"name":"Independent manufactured linear and quadratic fields distinguish both operators for Vprime proportional to r","passed":True,"examples":examples})
result={"scope":"Current checkout source and actual NEOPAX equation; no execution of historical NTSS binary or transport solve","checks":checks}
output=root / "outputs/whe_ntss_audit_20261031/er_diffusion_five_checks.json"
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
