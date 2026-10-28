# Profile-only SciPy continuation on the saved optimized geometry

The supplied completed run uses
`optimize_geometry_profiles_from_geometry_result.py --profile-only`, with
position/strength, bootstrap and net power enabled together. The source is
`outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization`;
the new output is `outputs/profiles_only_root_bootstrap_net_power_20261027`.
The setup confirms six variables, skipped geometry optimization, mode 2,
the 25/25/33 database mesh and a 1000-attempt transport budget. CUDA command
buffers are disabled. The initial Jacobian has 11 residuals and six columns.

Evidence is preserved in the [verbatim attachment](profiles_only_root_bootstrap_net_power_2026-10-27.txt),
[28 evaluation rows](profiles_only_root_bootstrap_net_power_2026-10-27.csv) and
[28 memory samples](profiles_only_root_bootstrap_net_power_2026-10-27_memory.csv).
The attachment starts with an interrupted fragment of the separate
bootstrap-candidate pilot and shell paste artifacts. Its incomplete
iteration-8 record does not establish another accepted pilot point; the
tables here cover only the subsequent completed SciPy run.

## Result

| Quantity | Initial | Final |
| --- | --- | --- |
| Total cost | 0.626058955 | 0.625752622 |
| Transition radius | 0.517423590 | 0.517423569 |
| Location-moment cost | 0.00586048341 | 0.00586041383 |
| Transition strength | 1.62479777 | 1.62482794 |
| Bootstrap excess above 0.1 | 0.438060861 | 0.438070680 |
| Bootstrap cost | 0.383794636 | 0.383811841 |
| Net power, fixed-reference-volume MW | 299.890707 | 299.930314 |
| Net-power cost | 0.000545062086 | 0.000221592794 |

Total cost falls by 0.0489304%. The root moves inward by only about 2.1e-8;
bootstrap slightly worsens. Strength is already above 0.9. Most of the tiny
cost improvement is the reduction in the already-small net-power error.
The fixed-geometry penalties sum to approximately 0.235858774 and do not
change as profiles vary. The active transport costs agree, within printed
precision, with weights 1000 for location moment, 100 for strength deficit,
4 for bootstrap excess and 10000 for net-power volume-average error. Their
targets are 0.514, 0.9, 0.1 and 300/331.0187969899648 MW/m^3 respectively.

The baseline root, strength, bootstrap and power reproduce the earlier
verified optimized-geometry starting observables. A wrong starting geometry
is not indicated by these data. The remote Git output does show a modified
geometry-only example as well as version.py; no diff or source digest for
that remote modification is supplied. Matching reported observables and
weights should not be presented as proof of complete source identity.

SciPy terminates after 28 function evaluations with status 2, the ftol
condition, and first-order optimality 18.5. Accepted diagnostic trials after
the initial point are 9, 11, 14, 16, 17, 18, 19, 27 and 28. This is a
small-cost-change termination, not evidence that the root/bootstrap targets
were reached or that a stationary optimum was established.

## Root jumps remain with the production optimizer

Many unaccepted trials have transition radius approximately 0.497811 and
power approximately 275.9 MW, despite later accepted evaluations remaining
near radius 0.517424 and 299.9 MW. For example:

| Evaluation | Status | Root | Net power MW | Total cost |
| --- | --- | --- | --- | --- |
| 19 | Accepted | 0.517423592 | 299.937868 | 0.625754098 |
| 20 | Unaccepted | 0.497811333 | 275.917531 | 27.1616835 |
| 27 | Accepted | 0.517423579 | 299.933682 | 0.625752637 |
| 28 | Accepted | 0.517423569 | 299.930314 | 0.625752622 |

At evaluation 20 the net-power cost alone is 26.4646845; the root cost is
0.131036463 and bootstrap cost is 0.330103811. Bootstrap improves at that
proposal, while the power and root penalties make its total cost much worse.
This is consistent with rejection by the joint objective. The root-only
controls previously rejected the same kind of jump without net power, so
net-power weighting alone does not explain the underlying forward response.

Changing from the candidate gradient-descent pilot to the production
trust-region least-squares optimizer has not removed the jump. These logs
contain neither trial parameter vectors nor initial Er fields, so they do
not establish continuity, identify the root-selection/evolution boundary
where the response changes, or validate/refute the local root Jacobian.
They also do not prove that the two sets of outcomes are two distinct
local minima or that increasing radial resolution is required.

## One trial hits the accepted-step guard

Evaluation 2 is rejected because its adaptive forward probe reaches 500
accepted steps at t=1.2991752688024072 before the requested t=2.0. This is
the explicit max_reverse_accepted_steps guard in prepare_reverse_static_setup;
it is not a reported nonfinite reverse cotangent or a memory-exhaustion
exception. That incomplete trajectory is assigned a penalty and is not
used as a full-time objective/Jacobian evaluation. The other 27 evaluations
complete and produce finite printed costs. No nonfinite-AD failure is
reported in this attachment; nan placeholders for disabled objectives do
not establish a nonfinite active residual/Jacobian.

Increasing budgets could permit that one proposal to finish, but it does
not explain the many completed, high-cost, shifted-root proposals or the
near-stagnant accepted results. A new optimization with larger budgets is
not the first required diagnosis.

## Memory

| Recorded metric | First evaluation | Final evaluation |
| --- | --- | --- |
| Host RSS MiB | 10797.4 | 11418.3 |
| Live JAX arrays | 689 | 744 |
| Device bytes in use | 48647680 | 48655616 |
| Peak device bytes | 5949773568 | 5950168064 |

After the failed-trial transient at evaluation 2, live JAX arrays settle at
744 from evaluation 4 onward, device bytes remain about 46.4 MiB, and the
five persistent stage-cache counts remain one after warm-up. These samples
do not show the previous unbounded device-executable/CUDA-graph growth.
They measure the recorded JAX allocator counters, not total GPU memory in
the driver. Host RSS grows by 620.9 MiB during this finite run, including
continued increases in later evaluations. It should not be described as
flat or as proof that all memory growth is fixed. Its cause and long-run
behavior remain unestablished.

## Next: inspect existing step predictions

The diagnostic directory is
`outputs/profiles_only_root_bootstrap_net_power_20261027/optimization_diagnostics/run_htoj9ejr`.
Its trial NPZ files and events.jsonl already contain coordinates, physical
step sizes, weighted Jacobians, cost contributions and local predictions.
The following reads those existing JSON records only; it imports no JAX or
NEOPAX and launches no physics solve:

```bash
cd /home/exouser/NEOPAX || exit 1
/home/exouser/jax_env/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/profiles_only_root_bootstrap_net_power_20261027/"
            "optimization_diagnostics/run_htoj9ejr/events.jsonl")
for line in path.open():
    e = json.loads(line)
    if e.get("event") != "trial" or e.get("trial") not in (1, 19, 20, 27, 28):
        continue
    m = e.get("model", {})
    selected = []
    for i, row in enumerate(e["rows"]):
        if not any(key in row["label"] for key in
                   ("transition_location", "bootstrap", "net_total_power")):
            continue
        detail = {k: row[k] for k in
                  ("label", "residual", "cost", "jacobian_norm_profile")}
        if m:
            detail.update(predicted_cost=float(m["predicted_cost"][i]),
                          actual_cost_change=float(m["actual_cost_change"][i]))
        selected.append(detail)
    print(json.dumps(dict(trial=e["trial"], finite=e["finite"], cost=e["cost"],
          from_trial=m.get("from_accepted_trial"),
          physical_step=m.get("physical_step"),
          predicted_reduction=m.get("predicted_reduction"),
          actual_reduction=m.get("actual_reduction"), rows=selected)))
PY
```

This will quantify how small the switching proposals are and which
Jacobian predictions fail to describe their nonlinear response. That is
the next evidence needed before changing weights, loss formulas or solver
settings again. No production source or optimizer was changed for this
analysis, and no local transport/VMEC/NTX/AD test was run.
