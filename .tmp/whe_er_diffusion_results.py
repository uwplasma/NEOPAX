"""Compare saved Er-diffusion experiments; NumPy/HDF5 only, no solver."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--face-temperature",action="store_true")
args=parser.parse_args()
experiment="face_temperature_comparison" if args.face_temperature else "er_diffusion_comparison"
old_label="Upwind temperature" if args.face_temperature else "Previous diffusion"
new_label="Shared face temperature" if args.face_temperature else "NTSS diffusion"
out = root / "outputs/whe_ntss_audit_20261031" / experiment
out.mkdir(parents=True, exist_ok=True)

def load(path):
    with h5py.File(path) as stream:
        return {key: stream[key][()] for key in stream}

def radiation(n, t):
    return .334*(n[1]+n[2]+4*n[3])*n[0]*np.sqrt(t[0])/62.422

def roots(rho, er):
    indices = np.where((er[:-1]>0)&(er[1:]<0))[0]
    return (rho[indices]-er[indices]*np.diff(rho)[indices]/np.diff(er)[indices]).tolist()

folder = "transport_solver_wHe_radau_ntss_match_THe_workHe_brems"
old_folder=folder+"_er_ntss" if args.face_temperature else folder
new_folder=folder+"_er_ntss_face_T" if args.face_temperature else folder+"_er_ntss"
old = load(root / "outputs" / old_folder / "transport_solution.h5")
new = load(root / "outputs" / new_folder / "transport_solution.h5")
ref = load(root / "examples/inputs/NTSS_Test_Final_flux_scaled_wHe.h5")
rx = ref["r"]/ref["r"][-1]
rn = np.array([ref["ne"],ref["nD"],ref["ne"]-ref["nD"]-2*ref["nHe"],ref["nHe"]])
rt = np.array([ref["Te"],ref["TD"],ref["Tt"],ref["TD"]])
ref_net = ref["AlphaPower"]-radiation(rn, rt)
report = {
    "method":"Existing saved HDF5 arrays only; common-rho interpolation and native-grid trapezoidal power integrals",
    "identical_initial_arrays":{key:bool(np.array_equal(old[key][0],new[key][0])) for key in ("density","temperature","pressure","Er")},
    "identical_grids_and_save_times":all(np.array_equal(old[key],new[key]) for key in ("rho","r_grid","r_grid_half","ts")),
    "reference":{"net_MW":float(np.trapezoid(ref_net*ref["Vr"],x=ref["r"])),"alpha_MW":float(np.trapezoid(ref["AlphaPower"]*ref["Vr"],x=ref["r"])),"downcrossings":roots(rx,ref["Er"])},
    "terminal_max_abs_change":{key:float(np.max(abs(new[key][-1]-old[key][-1]))) for key in ("density","temperature","pressure","Er")},
    "runs":{},
}
fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
rows=[]
for label,f,style in ((old_label,old,"-"),(new_label,new,"--")):
    n,t,rho = f["density"][-1],f["temperature"][-1],f["rho"]
    net = f["alpha_power_mw_m3"][-1]-radiation(n,t)
    alpha=float(np.trapezoid(f["alpha_power_mw_m3"][-1]*f["Vprime"],x=f["r_grid"]))
    metrics={"net_MW":float(f["P_total_mw"][-1]),"alpha_MW":alpha,"radiation_MW":alpha-float(f["P_total_mw"][-1]),
        "net_relative_error_percent":100*(float(f["P_total_mw"][-1])/report["reference"]["net_MW"]-1),
        "final_time":float(f["ts"][-1]),"all_profiles_finite":all(np.isfinite(f[k]).all().item() for k in ("density","temperature","pressure","Er")),
        "max_THe_minus_TD_keV":float(np.max(abs(f["temperature"][:,3]-f["temperature"][:,1]))),
        "downcrossings":roots(rho,f["Er"][-1]),"common_radius":[]}
    for x in (.1,.3,.5,.7,.9):
        values={"rho":x}
        for key,own,reference in (("nHe",n[3],ref["nHe"]),("TD",t[1],ref["TD"]),("Te",t[0],ref["Te"]),("Er",f["Er"][-1],ref["Er"])):
            a,b=float(np.interp(x,rho,own)),float(np.interp(x,rx,reference))
            values[key]={"NEOPAX":a,"NTSS":b,"relative_error_percent":100*(a/b-1)}
            rows.append([label,x,key,a,b,100*(a/b-1)])
        values["nHe_change_last_interval"]=float(np.interp(x,rho,f["density"][-1,3]-f["density"][-2,3]))
        metrics["common_radius"].append(values)
    report["runs"][label]=metrics
    for ax,profile in zip(axes.flat,(n[3],t[1],f["Er"][-1],net)):
        ax.plot(rho,profile,style,label=label,linewidth=1.8)
for ax,profile,ylabel in zip(axes.flat,(ref["nHe"],ref["TD"],ref["Er"],ref_net),
    (r"$n_{He}$ [$10^{20}$ m$^{-3}$]",r"$T_D$ [keV]",r"$E_r$ [kV/m]","Net source [MW/m$^3$]")):
    ax.plot(rx,profile,"k:",label="NTSS reference",linewidth=2.2)
    ax.set(xlabel=r"$\rho$",ylabel=ylabel,xlim=(0,1))
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle(("Effect of shared face temperature" if args.face_temperature else "Effect of matching Er diffusion")+", saved profiles at t=20 s")
fig.savefig(out/"comparison.png",dpi=170)
plt.close(fig)
report["net_change_MW"]=report["runs"][new_label]["net_MW"]-report["runs"][old_label]["net_MW"]
report["net_change_percent"]=100*report["net_change_MW"]/report["runs"][old_label]["net_MW"]
(out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
with (out/"common_radius.csv").open("w",newline="") as stream:
    writer=csv.writer(stream)
    writer.writerow(["case","rho","quantity","NEOPAX","NTSS","relative_error_percent"])
    writer.writerows(rows)
print(json.dumps(report,indent=2))
