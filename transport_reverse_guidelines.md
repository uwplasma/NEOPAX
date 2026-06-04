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
