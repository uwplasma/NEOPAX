# Database reverse AD versus FD status

## Completed one-step database reverse run

The black-box `ntx_scan_runtime` database benchmark completed its one accepted
step / one reverse segment validation run with `RBC:1:0,ZBS:1:0` and all
objectives.  The run used the fixed-table segment transpose, one recorded scan
fold, and compact VMEC JVP/bar contraction.

Observed successful boundary diagnostics:

```text
database final recorded-scan fold ready ... contract=one_batched_scan_transpose
compact_payload_tangent_contract=True
raw_block_param_bar_all_finite=True
raw_block_param_bar_first_nonfinite=None
```

The run completed and wrote:

```text
outputs/autodiff_transport_lagged_ntx/reverse_ad/
transport_reverse_ad_only_full_transport_shared_payload_smoke.json
```

This validates the database reverse plumbing for the one-step case.  It is not
yet a 16-step FD validation.

## Pure geometry objective comparison

The pure geometry rows should be independent of whether transport uses the
Lij flux model or the fixed scan database.  They are compared below with the
existing realtime eight-parameter snapshot in
`shared_payload_8param_benchmark_snapshot.md`.  That snapshot uses the same
VMEC geometry parameters, `RBC:1:0` and `ZBS:1:0`.

| Geometry objective | Database `d/dRBC:1:0` | Existing realtime | Relative difference | Database `d/dZBS:1:0` | Existing realtime | Relative difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boozer_qi_objective` | 5.939289118755e+00 | 5.939292752373e+00 | 6.118e-07 | -1.236555009546e-01 | -1.236549986209e-01 | 4.062e-06 |
| `boozer_maxj_objective` | -3.843135187617e+03 | -3.843140049060e+03 | 1.265e-06 | -1.920508280424e+03 | -1.920508192140e+03 | 4.597e-08 |
| `vmec_aspect_ratio` | -5.400678418701e+00 | -5.400678418701e+00 | 0 | -5.522688575132e+00 | -5.522688575132e+00 | 0 |
| `vmec_iota_mean` | 2.440514060926e-01 | 2.440514067774e-01 | 2.806e-09 | 1.456752601982e-01 | 1.456752605577e-01 | 2.468e-09 |
| `vmec_magnetic_well` | -1.109011606553e-02 | -1.109011249287e-02 | 3.221e-07 | -4.168202748224e-02 | -4.168202646912e-02 | 2.431e-08 |
| `vmec_mirror_ratio` | -5.935909471405e-01 | -5.935909468960e-01 | 4.118e-10 | 4.143700612540e-01 | 4.143700613673e-01 | 2.735e-10 |

All six geometry rows agree with the existing realtime values to within
`4.1e-06` relative difference.  The differences are numerical-level changes
from the payload-transpose route; the pure VMEC rows are effectively identical.

## Remaining validation

1. Run the same database configuration at 16 accepted steps / four segments.
2. Compare database transport-objective geometry derivatives with a matching
   frozen-schedule FD run.  The table above validates only the transport-model
   independent geometry rows.
