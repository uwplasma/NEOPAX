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

## Temporary Diagnostic Rule For Large Payload / CPU Capture

- Using a large saved reverse payload and/or CPU-only payload capture is allowed
  **only as a temporary diagnostic tool** when needed to isolate payload
  corruption or collector bugs.
- Do not treat:
  - full saved-payload collection
  - large payload preservation
  - CPU-only payload capture
  as acceptable final architecture for the reverse rollout path.
- The current later-step payload probe may use:
  - `--payload-source last-from-rollout`
  - `--rollout-accepted-step-limit 128`
  - `--rollout-max-total-steps-multiplier 4`
  - `--payload-capture-device cpu`
  but this is diagnostic-only, not a solution path.
- The later-step payload `nan` bug that first appeared around accepted step 91
  has now been fixed by rewriting
  `_radau_collect_realized_accepted_step_payloads(...)` to collect payloads from
  the true adaptive attempt path instead of a second accepted-only replay.
- So the CPU-heavy `last-from-rollout` probe is now considered **retired as the
  main test path**. Keep it only as a fallback diagnostic if a new payload
  corruption issue appears.
- After that fix, return immediately to the main design rules:
  - reduce the reverse contract to match forward mode as closely as possible
  - avoid saving broad per-step payloads
  - avoid CPU-only rollout capture as a steady-state reverse strategy
