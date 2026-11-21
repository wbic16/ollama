# PATCHES.md — wbic16/ollama divergence record

This fork is upstream ollama/ollama plus the patches recorded here.
Base at fork time: upstream `main` @
2bd4d6aabdf4caa39e7095e201dcda4cc1a3a406 (2026-11-21, "mlx: fuse
Inkling residual mixing on Metal (#19808)"); distance 0 ahead /
0 behind. Every divergence from upstream is listed per file below —
if it isn't in this file, it isn't a fork change. Each patch is
stated so it can be re-applied to a future upstream tree.

## Patch 001 — MLX CPU backend: inkling-2-mini on Linux x86_64 (2026-11-21)

Why: upstream's MLX runner is the only runner that implements the
inkling2 architecture (mlxrunner/model/inkling2), but upstream ships
MLX backends only for Metal (macOS) and CUDA 13 (Linux/Windows), and
the Ollama library package for inkling-2-mini ships macOS and DGX
Spark manifests only — no linux/amd64. The ranch fleet has no
discrete GPUs, so the platform kernel to inflate for the unified
backend program (first declaration: kernels/inkling2/) is MLX's CPU
backend. This patch adds that backend to the build dispatch, plus
one Go fix the litmus exposed (mlx/io.go, below). The MLX loader
accepts any variant it can dlopen on non-macOS platforms, and the
inkling2 model's GPU kernels disable themselves where they cannot
launch, leaving the graph fallbacks — which compute the same
outputs — to execute.

Files:

- `cmake/local.cmake` (+6/−1)
  - The `OLLAMA_MLX_BACKENDS` doc string gains `;cpu`.
  - A `cpu` branch in the MLX backend dispatch:
    `ollama_add_mlx_build(cpu, PRESET mlx_cpu, RUNNER_DIR mlx_cpu)`,
    appending target `ollama-mlx-cpu`. Previously any backend other
    than cuda_v13/metal_v3/metal_v4 hit a FATAL_ERROR.
- `cmake/mlx/CMakePresets.json` (+15)
  - New configure preset `mlx_cpu` (inherits `default`,
    binaryDir build/mlx_cpu): `MLX_BUILD_CUDA=OFF`,
    `MLX_BUILD_METAL=OFF`, `OLLAMA_RUNNER_DIR=mlx_cpu`. With CUDA and
    Metal off, MLX builds its CPU backend and produces libmlxc.so.
  - New build preset `mlx_cpu` (targets mlx, mlxc), matching the
    other backends.
- `mlx/io.go` (+7/−1)
  - `loadSafetensorsStream` chose the load stream by OS: CPU stream
    on macOS, GPU stream everywhere else. On a CPU-only MLX build
    the GPU stream does not exist, `mlx_load_safetensors` fails,
    and `Load`'s iterator swallows the error — every weight blob
    silently yields zero tensors and the runner reports its first
    tensor missing. The fix uses the CPU stream whenever
    `GPUIsAvailable()` is false, the same availability fact the
    package already branches on elsewhere. Behavior on GPU
    platforms is unchanged. (Observed hazard, not changed here:
    `Load` discards the native loader's error; surfacing it is an
    upstream API decision, not part of this patch.)
- `kernels/inkling2/inkling2.phext` (new, 16,922 bytes, 10 scrolls,
  SHA-256 9953d0ec87aa86d70f2f6d708c47ab297f843e8fa99c3384ef3402081b9b4735)
  - The virtualized-kernel declaration for the inkling block in
    multiversal phext: declaration scrolls (structure, the loop,
    attention, MoE, NVFP4/MXFP8 precision and dequantization rule,
    residual mixing) plus implementation scrolls (the published MLX
    runner as reference; this CPU kernel; the first-inflation run
    record). Home and scroll map at 1.1.1/1.1.1/1.1.1.
- `kernels/inkling2/build_declaration.py` (new)
  - The declaration's generator; rebuilds the phext byte-identically
    and round-trip verifies it.
- `kernels/inkling2/README.md` (new)
  - What the declaration is, in plain text for repository readers.
- `PATCHES.md` (new) — this file.

NVFP4 route: in-kernel (in-graph) dequantization. Weights stay in
their shipped NVFP4/MXFP8 packed form in memory and in the model
store; the graph fallbacks apply the declaration's dequantization
rule (NVFP4: E2M1 code × E4M3 scale per 16-value group; MXFP8: E4M3
code × 2^ue8m0 scale per 32×32 block) as they compute. Cost: the
dequantization arithmetic is paid on every matmul, in tokens per
second. Precision change: none — the release is the precision, and
no higher-precision master weights exist. The build-time conversion
alternative was measured and rejected: materializing the experts in
bf16 takes 24.46B expert parameters × 2 bytes ≈ 48.9 GB, more RAM
than any ranch node has (fleet maximum 44 GB; the litmus node has
30 GB). Requantization was rejected outright: it would change the
trained precision and produce a different artifact, not an
implementation of the declared one.

Re-apply notes: the cmake changes are purely additive (one branch,
one doc token, two presets) and the io.go change is one condition
plus its comment. On an upstream rebase, expect conflicts only if
upstream restructures the MLX backend dispatch, the preset file, or
the safetensors load path; re-apply by re-inserting the same pieces.

Verification (2026-11-21, elven-path, Linux x86_64, CPU only, no
discrete GPU on the node or anywhere in the probed fleet):
- Unmodified tree built and served a stock model (smollm2:135m)
  through its API before the patch was applied.
- `go test ./mlx/quant/ ./create/ ./mlx/ ./mlxrunner/model/inkling2/`
  — all ok; the mlx package tests load libmlxc.so (CPU) through the
  fork's own dynamic loader.
- Patched tree built with `-DOLLAMA_MLX_BACKENDS=cpu`; payload at
  build/lib/ollama/mlx_cpu/libmlxc.so (libmlx.so links the bundled
  OpenBLAS — the CPU backend is real).
- Before/after, same binary and store: with the CPU payload absent,
  `ollama create` failed with "MLX runtime is not available: failed
  to load MLX dynamic library"; with it present, the create imported
  inkling-2-mini (1204 layers, 19 GB), the runner loaded 1244
  tensors, and the model answered through /api/generate.
- Weights verified against the publisher's tree (Hugging Face
  thinkingmachines/Inkling-2-Mini): model safetensors 19,526,019,168
  bytes, SHA-256 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1;
  mtp and tokenizer hashes also match.
- Serving: prompt "What is the capital of France? Answer in one
  sentence." produced coherent, correct text (", Paris is the
  capital of France." followed by correctly answered self-posed
  follow-ups). Decode 120 tokens in 672.0 s = 0.18 tok/s on CPU;
  runner RSS 20.3 GB of the node's 30 GB. Full record: the
  declaration's run-record scroll (1.1.1/1.2.1/1.1.3), including the
  kernel-optimization look — in-graph dequantization adopted (its
  per-token re-dequantization is the measured bottleneck), bf16
  materialization rejected by arithmetic (48.9 GB > any ranch
  node's RAM), requantization rejected on fidelity.
