# kernels/inkling2 — a virtualized kernel declaration

This directory holds the unified-backend declaration for the
Inkling-2-Mini transformer block, as one multiversal phext artifact:

- `inkling2.phext` — the artifact. One file holding the declaration
  scrolls (the block's behavior stated declaratively: shapes, the
  loop, attention, MoE routing, NVFP4/MXFP8 precision and the
  dequantization rule, residual mixing) and the implementation
  scrolls (the published MLX runner as reference; the Linux CPU
  kernel this fork's patch adds; the run record of the first
  inflation). Scroll map lives in the artifact's Home scroll at
  1.1.1/1.1.1/1.1.1.
- `build_declaration.py` — the generator. The phext is built from
  the scroll texts in this script and round-trip verified on every
  build: `python3 build_declaration.py inkling2.phext`.

The design rule (Will, 2026-11-21): a virtualized kernel describes
how a kernel behaves declaratively; a valid distribution always
includes the declaration and at least one implementation; the runner
inflates the platform kernel it is missing. This directory is the
first instance of that shape, and its placement inside the fork is
provisional.

The CPU implementation itself is a build-system patch, recorded in
the fork's PATCHES.md: `OLLAMA_MLX_BACKENDS=cpu` builds MLX's CPU
backend (libmlxc.so) on Linux, and the inkling2 model runs on it
through the MLX runner's graph fallbacks, dequantizing NVFP4/MXFP8
in-graph from the weights' shipped packed form — no precision
change, no converted checkpoint.
