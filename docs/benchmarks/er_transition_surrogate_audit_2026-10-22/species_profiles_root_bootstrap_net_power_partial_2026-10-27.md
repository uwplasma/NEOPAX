# Independent-species stage 2: partial evaluations 4 through 9

The supplied excerpt is from the new species-profile path. It starts during
the reverse sweep for evaluation 4 and ends after evaluation 9. It contains
neither the initial state/cost/parameter-count report, accepted-step
notifications, nor a final result or exception. The code's default is 15
profile DOFs, but the setup/manifest is not included in this excerpt.
`objectives=13` counts raw reverse objective rows, not optimizer variables.

The [raw excerpt](species_profiles_root_bootstrap_net_power_partial_2026-10-27.txt),
[evaluation table](species_profiles_root_bootstrap_net_power_partial_2026-10-27.csv)
and [memory samples](species_profiles_root_bootstrap_net_power_partial_2026-10-27_memory.csv)
are saved without inventing acceptance states for trial evaluations.

| Evaluation | Total transport cost | Transition radius | Bootstrap observable | Net power MW |
| --- | --- | --- | --- | --- |
| 4 | 194.033934 | 0.438889174 | 0.475685181 | 235.314686 |
| 5 | 13.0629262 | 0.497814962 | 0.512491652 | 283.388495 |
| 6 | 23.8682554 | 0.497810949 | 0.506751201 | 277.351772 |
| 7 | 26.7808301 | 0.497810349 | 0.505657372 | 275.983149 |
| 8 | 27.3304535 | 0.497810503 | 0.505630414 | 275.733666 |
| 9 | 27.2372699 | 0.497810909 | 0.505915820 | 275.775986 |

The bootstrap observable here is the printed penalty plus the configured
limit 0.1. At evaluation 9, root-location cost is 0.131043335, bootstrap
cost is 0.329535305, and net-power cost is 26.7766913, about 98.31% of the
total. These proposals worsen root position and power relative to the known
saved seed (root approximately 0.517424, power approximately 299.89 MW),
while bootstrap decreases. The position/strength/power/bootstrap formulas
still rank such a tradeoff as expensive. The strength threshold remains
satisfied in these six proposals.

The same final front location near 0.49781 and power near 275.8 MW appears
as in the six-shared-DOF experiment. Expanding the parameter set has therefore
not eliminated these trial forward outcomes. This does not establish that
the optimization has accepted a bad root, that the expanded problem is
converged or unsuccessful, or that the DOFs are ineffective. The trial
coordinates, initial-root fields and step-model predictions are absent;
the excerpt cannot identify why the response jumps or independently
validate the physical transport Jacobian.

## Reverse passes and memory

The displayed sweeps complete with 173, 214, 213, 226, 216 and 223 support
rebuilds for evaluations 4 through 9 respectively. No nonfinite reverse
error, accepted-step guard exception, or OOM is reported in this excerpt.
Completion without an exception is not an independent AD/FD accuracy check.
Geometry and disabled-objective nan fields are reporting placeholders:
the new fixed-geometry profile example omits geometry cost rows and disables
maximum-Er/Er-left/Er-right. They do not indicate nonfinite active transport
costs. The reported transport costs are finite.

Host RSS increases from 9488.2 to 9499.2 MiB (+11 MiB). Device bytes in use
increase from 45871616 to 45874432 (about 43.75 MiB), with an unchanged peak
5257060352 bytes (about 4.90 GiB). Live JAX arrays rise from 561 to 577 when
the replay needs another segment, then remain 577 through evaluation 9.
The replay/backward/probe/bootstrap cache counts remain one. runtime_scan
cache size is zero, consistent with the profile-only table returning profile
columns directly without a VMEC/recorded-scan parameter transpose. This
short interval does not establish long-run host or total-driver GPU memory
behavior.

## Inspect existing predictions while the run continues

The following uses only saved JSON. It prints the actual independent
parameter labels and each completed trial's step norm and predicted versus
observed total cost reduction. It adds no physics solve and can be run while
the optimizer writes later evaluations. The output-directory name below is
the one in the provided run command.

```bash
cd /home/exouser/NEOPAX || exit 1
/home/exouser/jax_env/bin/python - <<'PY'
import json
from pathlib import Path

case = Path("outputs/species_profiles_root_bootstrap_net_power_20261027")
print("DOFs:", json.loads((case / "case_manifest.json").read_text())["parameter_labels"])
files = list((case / "optimization_diagnostics").glob("run_*/events.jsonl"))
path = max(files, key=lambda p: p.stat().st_mtime)
print("Diagnostics:", path)
for line in path.open():
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        break  # The last line may still be being written.
    if e.get("event") != "trial":
        continue
    m = e.get("model", {})
    print(json.dumps(dict(trial=e["trial"], finite=e["finite"], cost=e["cost"],
        from_accepted_trial=m.get("from_accepted_trial"),
        scaled_profile_step=m.get("profile_step_norm"),
        predicted_reduction=m.get("predicted_reduction"),
        actual_reduction=m.get("actual_reduction"))))
PY
```

Those existing predictions are needed to distinguish a useful local descent
model followed by a finite forward change from inaccurate local sensitivities.
There is no basis here to change weights or restart another optimization.
No implementation change or local physics test was made for this update.
