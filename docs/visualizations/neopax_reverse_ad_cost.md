# NEOPAX: checkpointed reverse AD cost animation

Open [the interactive animation](neopax_reverse_ad_cost.html) in a browser, or
view the [looping GIF](neopax_reverse_ad_cost.gif) or
[H.264 MP4 video](neopax_reverse_ad_cost.mp4). The HTML works offline and
has play/pause, stepping, a scrubber, speed and segment-size controls, and SVG
frame export. It runs no numerical transport solver.

The **Arrow scheme** style matches the supplied reference: a white background,
green forward and replay arrows, red rejected-trial branches, black checkpoint
ticks, and orange reverse arrows. Exports:
[GIF](neopax_reverse_ad_arrows.gif), [MP4](neopax_reverse_ad_arrows.mp4), and
[static vector SVG](neopax_reverse_ad_arrows.svg). The replay row has numbered
segment labels and checkpoint labels above its black boundary marks. As the
forward solve accepts each step, a small green dot appears at the corresponding
position in the replay row. These persistent dots represent recorded accepted
timesteps (lightweight schedule metadata), not full saved states or internal
stage records; the black checkpoints represent saved full carries. Each green
replay arrow segment disappears when its backward sweep has completed and its
temporary records are released; the dots and checkpoint marks remain visible. The
animation uses the same segment ordering as the detailed view. Trial branches
are schematic rather than measured attempt data. The minimal exports use a
1280 x 588 canvas. The static SVG shows an intermediate backward sweep.

The GIF is 1120 x 778 pixels; the MP4 is 1280 x 888 pixels. Both show the same
27.56-second sequence. The GIF loops automatically. MP4 repeat behavior is set
in the application that plays or embeds the video. No PowerPoint presentation
is included.

For motion inside a PDF, embed the H.264 MP4 as rich media in a compatible PDF
authoring tool. Adobe Acrobat offers **All tools > Add rich media > Video**;
desktop Acrobat/Reader supports H.264 media playback. A GIF inserted as a
normal PDF image does not retain its animation. Standard PowerPoint-to-PDF
export does not preserve media playback, so embedding in the PDF is a separate
step. Playback depends on the PDF viewer; do not assume a browser's built-in
PDF viewer supports it. See
[Adobe's multimedia embedding documentation](https://helpx.adobe.com/acrobat/using/adding-multimedia-pdfs.html),
[Adobe's playback formats](https://helpx.adobe.com/acrobat/using/playing-video-audio-multimedia-formats.html),
and [Microsoft's PDF conversion notes](https://powerpoint.cloud.microsoft/create/en/blog/convert-powerpoint-to-pdf/).

The schematic shows 12 accepted steps split into three segments of four:

1. An adaptive forward solve produces the trajectory, accepted schedule, and
   sparse checkpoint carries. Rejected trials are not physical trajectory steps.
2. Restore the last segment's checkpoint. Replay its accepted steps forward,
   retaining their stage records in a temporary segment buffer.
3. Propagate adjoints backward through that segment. Accumulate parameter and
   shared-support cotangents, release its temporary records, and repeat for the
   preceding segment.

The lower two lanes alternate. Reconstructing all segments visits every
accepted step once in total; the adjoint also visits each accepted step once.
If these three types of work had equal cost, their total would be approximately
three forward equivalents. This is an explanatory estimate, **not a measured
NEOPAX performance result or a guarantee of a 3x wall-time ratio**. Replay skips
the adaptive search; implicit-stage adjoints, batched objective rows, database
and geometry pullbacks, initialization and compilation have different costs.

The animation assumes the initial adaptive run supplied checkpoint carries.
Some paths replay the accepted schedule separately to collect those carries.
That is additional work, not another adaptive solve. A fixed schedule by itself
does not eliminate the need for suitable checkpoint states or local stage
reconstruction.

## Memory interpretation

The persistent data include sparse checkpoint carries and O(N) lightweight
schedule metadata. The temporary buffer contains stage records for only the
current segment. It is cleared before the preceding segment is reconstructed;
records do not accumulate across the reverse sweep. Shared transport/database
data and objective cotangents are additional storage.

The drawing counts records, not bytes. A full carry and a stage record have
different sizes. With N accepted steps and segment length L, the schematic
retains approximately ceil(N/L) segment-start carries plus up to L local step
records, as well as the final state and shared data. Changing L shows this
tradeoff without changing the total number of reconstructed accepted steps.
The all-12-steps option illustrates retaining a whole segment's stage records;
it is not an implementation of a separate full-tape AD algorithm.

## Source correspondence

Inspected on 2026-11-01:

- [Checkpoint assembly](../../NEOPAX/_transport_solvers.py):
  `_radau_adaptive_final_y_realized_schedule_vjp_fwd_from_schedule_artifact`
  reuses supplied `segment_start_carries_artifact` and `final_carry`. Without
  them, the multi-segment fallback performs a fixed accepted-schedule scan to
  collect segment starts.
- [Support reverse path](../../NEOPAX/_reverse_ad_transport.py):
  `realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector`
  constructs/reuses the
  realized schedule residuals, seeds terminal-objective cotangents and runs the
  segmented cotangent sweep. The record-consuming segment path first obtains
  `step_start_carries` and `step_primal_records`, then runs accepted-step
  pullbacks in reverse slot order.
- In recorded-forward Jacobian reuse mode, accepted-step Newton reuse decisions
  are retained without replaying or differentiating rejected attempts. A
  checkpoint represents the necessary complete carry, not only density,
  temperature and electric-field arrays.
- The backward arrows represent derivative pullbacks through accepted step
  maps. NEOPAX uses implicit-stage adjoint solves; it does not solve the plasma
  dynamics backward in physical time. Controller differentiation is outside
  this accepted-schedule contract.

The model intentionally avoids fabricated timing or gradient values. The HTML's
`NEOPAXCost.check()` checks event ordering, the one-segment buffer bound and
release, reverse ordering, and the three equal-cost work counts for every
offered segment length. No VMEC, NTX or transport calculation is needed.
