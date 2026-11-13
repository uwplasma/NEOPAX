"""Inspect existing wHe results; no geometry, database, flux or transport solve."""
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/whe_ntss_audit_20261031"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {
    "THe": ("transport_solver_wHe_radau_ntss_match_THe", 0.316),
    "THe + work": ("transport_solver_wHe_radau_ntss_match_THe_workHe", 0.316),
    "THe + work + radiation": ("transport_solver_wHe_radau_ntss_match_THe_workHe_brems", 0.334),
}

def load(path):
    with h5py.File(path) as f:
        return {k: f[k][()] for k in f}

def brems(n, t, coefficient):
    zeff = (n[1] + n[2] + 4*n[3])/n[0]
    return coefficient*zeff*n[0]**2*np.sqrt(t[0])/62.422

def roots(rho, er):
    idx = np.where((er[:-1] > 0) & (er[1:] < 0))[0]
    return (rho[idx]-er[idx]*np.diff(rho)[idx]/np.diff(er)[idx]).tolist()

ref = load(ROOT / "examples/inputs/NTSS_Test_Final_flux_scaled_wHe.h5")
rx = ref["r"]/ref["r"][-1]
rn = np.array([ref["ne"], ref["nD"], ref["ne"]-ref["nD"]-2*ref["nHe"], ref["nHe"]])
rt = np.array([ref["Te"], ref["TD"], ref["Tt"], ref["TD"]])
ref_net = ref["AlphaPower"]-brems(rn, rt, .334)
report = {"method": "saved arrays and algebraic sources only; native-grid trapezoids", "reference": {
    "alpha_MW": float(np.trapezoid(ref["AlphaPower"]*ref["Vr"], x=ref["r"])),
    "net_MW": float(np.trapezoid(ref_net*ref["Vr"], x=ref["r"])),
    "volume_m3": float(np.trapezoid(ref["Vr"], x=ref["r"])),
    "downcrossings": roots(rx, ref["Er"]),
}, "runs": {}}
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for name, (folder, coefficient) in RUNS.items():
    f = load(ROOT / "outputs" / folder / "transport_solution.h5")
    rho, n, t = f["rho"], f["density"][-1], f["temperature"][-1]
    net = f["alpha_power_mw_m3"][-1]-brems(n, t, coefficient)
    power = float(np.trapezoid(net*f["Vprime"], x=f["r_grid"]))
    metrics = {
        "file": f"outputs/{folder}/transport_solution.h5",
        "final_time": float(f["ts"][-1]),
        "all_profiles_finite": all(np.isfinite(f[k]).all().item() for k in ("density", "temperature", "pressure", "Er")),
        "coefficient": coefficient,
        "net_MW": power,
        "net_relative_error_percent": 100*(power/report["reference"]["net_MW"]-1),
        "net_saved_discrepancy_MW": power-float(f["P_total_mw"][-1]),
        "alpha_MW": float(np.trapezoid(f["alpha_power_mw_m3"][-1]*f["Vprime"], x=f["r_grid"])),
        "volume_m3": float(np.trapezoid(f["Vprime"], x=f["r_grid"])),
        "max_THe_minus_TD_keV": float(np.max(abs(f["temperature"][:, 3]-f["temperature"][:, 1]))),
        "downcrossings": roots(rho, f["Er"][-1]),
        "common_radius": [],
    }
    for x in (.1, .3, .5, .7, .9):
        row = {"rho": x}
        for key, own, reference in (
            ("nHe", n[3], ref["nHe"]), ("ne", n[0], ref["ne"]),
            ("TD", t[1], ref["TD"]), ("Te", t[0], ref["Te"]),
            ("Er", f["Er"][-1], ref["Er"]), ("net_MW_m3", net, ref_net),
        ):
            a, b = float(np.interp(x, rho, own)), float(np.interp(x, rx, reference))
            row[key] = {"NEOPAX": a, "NTSS": b, "relative_error_percent": 100*(a/b-1)}
        row["nHe_change_last_interval"] = float(np.interp(x, rho, f["density"][-1,3]-f["density"][-2,3]))
        metrics["common_radius"].append(row)
    if coefficient == .334:
        relative_face_jump = abs(np.diff(t, axis=1))/(t[:, 1:]+t[:, :-1])
        metrics["upwind_vs_arithmetic_face_temperature_max_relative"] = np.max(relative_face_jump[:3], axis=1).tolist()
        metrics["upwind_vs_arithmetic_face_temperature_max_relative_rho_lt_08"] = np.max(relative_face_jump[:3, rho[1:] < .8], axis=1).tolist()
        latest = f
    report["runs"][name] = metrics
    axes[0,0].plot(rho, n[3], label=name)
    axes[0,1].plot(rho, t[1], label=name)
    axes[1,0].plot(rho, f["Er"][-1], label=name)
    axes[1,1].plot(rho, net, label=name)

for ax, reference, label in zip(axes.flat, (ref["nHe"],ref["TD"],ref["Er"],ref_net),
    (r"$n_{He}$ [$10^{20}$ m$^{-3}$]", r"$T_D$ [keV]", r"$E_r$ [kV/m]", "Net source [MW/m$^3$]")):
    ax.plot(rx, reference, "k--", label="NTSS, t=20 s", linewidth=2)
    ax.set(xlabel=r"$\rho$", ylabel=label, xlim=(0,1))
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle("Saved wHe profiles at t=20 s")
fig.savefig(OUT / "comparison.png", dpi=170)
plt.close(fig)

# Check the actual NEOPAX source functions against the formulas in NTSS.
# No transport or flux model is built: each source receives only 4 x 51 arrays.
import jax.numpy as jnp
from NEOPAX._sources import dt_reaction, fusion_power_fraction_electrons, bremsstrahlung_radiation, power_exchange
from NEOPAX._species import Species
from NEOPAX._state import TransportState
species = Species(4, jnp.arange(4), jnp.array([.000544617,2.,3.,4.]), jnp.array([-1.,1.,1.,2.]), names=("e","D","T","He"))
n, t = latest["density"][-1], latest["temperature"][-1]
state = TransportState(jnp.asarray(n), jnp.asarray(n*t), jnp.asarray(latest["Er"][-1]))
tt = t[2]
u = tt**(-1/3)
reaction = 8.972e-19*u*u*np.exp(-19.94*u)*((tt+1.0134)/(1+.006386*(tt+1.0134)**2)+1.877*np.exp(-.16176*tt**1.5))
he = 1e20*reaction*n[1]*n[2]
y2 = 88/t[0]
fraction = 1-2*(np.log((1-np.sqrt(y2)+y2)/(1+2*np.sqrt(y2)+y2))/6+.57735026*np.arctan(.57735026*(2*np.sqrt(y2)-1))+.30229987)/y2
ntss_exchange = np.zeros_like(t)
ntss_masses = [.000544662, 2., 3., 4.]
for a,b in ((0,1),(0,2),(1,2)):
    ma, mb, ta, tb, na, nb = ntss_masses[a],ntss_masses[b],t[a],t[b],n[a],n[b]
    if a == 0:
        lnl = np.full_like(ta,15.)
        lnl = np.where((ta < .01) & (tb*ma < ta*mb),17.24+.5*np.log(ta**3/na),lnl)
        lnl = np.where((ta > .01) & (tb*ma < .01*mb),16.1+.5*np.log(ta**2/na),lnl)
        lnl = np.where(ta*mb < tb*ma,24.24+.5*np.log(tb*(tb*mb)**2/nb),lnl)
    else:
        lnl = 17.24-np.log((ma+mb)/(ma*tb+mb*ta)*np.sqrt(na/ta+nb/tb))
    pair = 663*np.sqrt(ma*mb)*na*nb*lnl*(tb-ta)/(ma*tb+mb*ta)**1.5
    ntss_exchange[a] += pair
    ntss_exchange[b] -= pair
rate_ad, he_ad, alpha_ad = map(np.asarray,dt_reaction(state,species))
rad_ad = np.asarray(bremsstrahlung_radiation(state,species,coefficient_mode="ntssfusion")[0])
exchange_ad = np.asarray(power_exchange(state,species,jnp.array([True,True,False,True,False,False]),coulomb_log_mode="ntssfusion"))
def relative_max(actual,expected):
    return float(np.max(abs(np.asarray(actual)-expected))/max(float(np.max(abs(expected))),1e-30))
report["same_state_source_checks_relative_max_error"] = {
    "reaction":relative_max(rate_ad,reaction), "He_source":relative_max(he_ad,he),
    "alpha":relative_max(alpha_ad,3520*he),
    "alpha_electron_fraction":relative_max(fusion_power_fraction_electrons(state,species),fraction),
    "radiation":relative_max(rad_ad,brems(n,t,.334)*62.422),
    "exchange_including_actual_electron_mass_difference":relative_max(exchange_ad[:3],ntss_exchange[:3]),
    "exchange_species_sum_max_abs":float(np.max(abs(np.sum(exchange_ad,axis=0)))),
    "stored_alpha_vs_formula":relative_max(latest["alpha_power_mw_m3"][-1],alpha_ad/62.422),
}
rho, r, vp = latest["rho"], latest["r_grid"], latest["Vprime"]
tt_reference = np.interp(rho, rx, ref["Tt"])
u_reference = tt_reference**(-1/3)
alpha_tt_swap = 3520*1e20*n[1]*n[2]*8.972e-19*u_reference**2*np.exp(-19.94*u_reference)*(
    (tt_reference+1.0134)/(1+.006386*(tt_reference+1.0134)**2)
    +1.877*np.exp(-.16176*tt_reference**1.5))/62.422
report["saved_state_decomposition"] = {
    "latest_alpha_MW":float(np.trapezoid(latest["alpha_power_mw_m3"][-1]*vp,x=r)),
    "latest_radiation_MW":float(np.trapezoid(brems(n,t,.334)*vp,x=r)),
    "ref_alpha_MW":report["reference"]["alpha_MW"],
    "ref_radiation_MW":float(np.trapezoid(brems(rn,rt,.334)*ref["Vr"],x=ref["r"])),
    "net_if_only_tritium_temperature_replaced_by_NTSS_MW":float(np.trapezoid((alpha_tt_swap-brems(n,t,.334))*vp,x=r)),
    "ref_net_integrated_on_NP_grid_MW":float(np.trapezoid(np.interp(rho,rx,ref_net)*vp,x=r)),
    "NP_fv_volume_m3":float(np.sum(vp*np.diff(latest["r_grid_half"]))),
    "NP_trap_volume_m3":float(np.trapezoid(vp,x=r)),
    "NTSS_trap_volume_m3":report["reference"]["volume_m3"],
    "Vp_relative_error_percent_max":float(100*np.max(abs(vp/np.interp(rho,rx,ref["Vr"])-1))),
}
rf = latest["r_grid_half"]
er_over_r = np.divide(latest["Er_faces"][-1],rf,out=np.zeros_like(rf),where=rf>0)
er_over_r[0] = latest["Er"][-1,0]/r[0]  # Regular axis limit; Vprime_half[0] = 0.
correction = -.1*2*np.diff(latest["Vprime_half"]*er_over_r)/(vp*np.diff(rf))
report["missing_NTSS_Er_over_r_flux"] = {
    "meaning":"extra NTSS diffusion RHS on saved NEOPAX field; no new evolution, edge row excluded",
    "rhs_correction_at_common_radii_kVm_per_s":{str(x):float(np.interp(x,rho,correction)) for x in (.1,.3,.5,.7,.9)},
    "rhs_correction_interior_max_abs_kVm_per_s":float(np.max(abs(correction[:-1]))),
}
(OUT / "summary.json").write_text(json.dumps(report,indent=2)+"\n")
with (OUT / "common_radius.csv").open("w",newline="") as stream:
    writer=csv.writer(stream)
    writer.writerow(["case","rho","quantity","NEOPAX","NTSS","relative_error_percent"])
    for name,metrics in report["runs"].items():
        for row in metrics["common_radius"]:
            for key,value in row.items():
                if isinstance(value,dict):
                    writer.writerow([name,row["rho"],key,value["NEOPAX"],value["NTSS"],value["relative_error_percent"]])
print(json.dumps({"reference":report["reference"],"sources":report["same_state_source_checks_relative_max_error"],"latest":report["runs"]["THe + work + radiation"]},indent=2))
print(f"Wrote {OUT}")
