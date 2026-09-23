"""Plot captured deuterium/tritium NTX inputs at the failing edge state."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


energy = np.arange(1, 5)
nu_over_v = {
    "Deuterium": np.array([3.11696484e-2, 1.89855985e-3, 3.29860953e-4, 8.17441525e-5]),
    "Tritium": np.array([2.60969312e-2, 1.70676716e-3, 3.12776339e-4, 7.93771493e-5]),
}
er_over_v = {
    "Deuterium": np.array([-1.4253734e-1, -6.126793e-2, -3.800663e-2, -2.641044e-2]),
    "Tritium": np.array([-1.7458592e-1, -7.504362e-2, -4.655217e-2, -3.234865e-2]),
}

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
for name, color in (("Deuterium", "#d62728"), ("Tritium", "#1f77b4")):
    axes[0].plot(energy, nu_over_v[name], "o-", color=color, label=name)
    axes[1].plot(energy, er_over_v[name], "o-", color=color, label=name)

axes[0].set_yscale("log")
axes[0].set_title(r"Collision coordinate $\nu/v$")
axes[0].set_ylabel(r"$\nu/v$")
axes[1].set_title(r"Electric coordinate $E_r/v$")
axes[1].set_ylabel(r"$E_r/v$")
for axis in axes:
    axis.set_xlabel("NTX energy node")
    axis.set_xticks(energy)
    axis.grid(True, alpha=0.3)
    axis.legend()

fig.suptitle(r"Captured outer-edge inputs: $E_{r,edge}=-36.970860$ kV/m")
output = Path(__file__).with_name("captured_dt_ntx_inputs.png")
fig.savefig(output, dpi=180)
print(output)
