## Transport Reverse Benchmark Contract

- Differentiate the accepted-step composition only.
- The accepted-step schedule/time map is fixed by the primal forward pass.
- Reverse mode must target the same accepted-step time map as forward mode.
- Rejected attempts may contribute nondifferentiated support/replay metadata only.
- Rejected attempts must not become differentiated transitions in this benchmark reverse map.
- Do not widen the differentiated map just because another library might do so.
- Preserve the current forward AD path during this reverse refactor; use its
  values as the reference that reverse must match before any forward-path
  optimization is considered.

## Current OOM Reduction Rules

- Do not solve the full reverse rollout by saving full per-step payloads for
  all accepted steps.
- Prioritize making the custom reverse step cheaper, following the same reduced
  contract philosophy already used by the forward accepted-step AD path.
- For rollout experiments, prefer:
  - transient per-segment payload collection
  - optional sparse checkpoints
  - `checkpoint_count=0` meaning no stored checkpoints
- The main current blocker is the dynamic runtime reverse-payload contract for
  the one-step custom rule, not the accepted-step local algebra itself.
