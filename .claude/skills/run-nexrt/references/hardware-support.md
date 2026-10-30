# Hardware support

The supported configuration is **8×NVIDIA H100, H200, or H20-3e, TP=8**, with NVLS multicast
support, CUDA Toolkit 12.9+, and PyTorch 2.6+. H20-3e runs 1-batch decoding only.

## Fixed execution geometry

`bridge.make_server_args` sets `tp_size=8`. Weight shards, the four-rank context-parallel
attention groups, and multicast reductions use this geometry.

`nex_n2_5_pro/kernel_build.py` compiles kernels for `sm_90a`. The full-attention implementations
use Hopper WGMMA. Changing a compiler flag alone does not establish support for another
architecture; that requires a separate kernel port and validation.

## H20-3e

8×H20-3e (78 SMs) runs the single-token decode path, which is the 1-batch mode. It runs with and
without context parallelism (CP), for prompts up to 261,888 tokens. `--mtp` and `--dflash` are not
ported to H20-3e yet.

The H20-3e validation used the Qwen3.5-397B-A17B-FP8 checkpoint. It covered correctness gates,
determinism runs, and a 4-hour soak at prompt lengths from 1,024 to 261,888 tokens.

H100 and H200 keep the released kernels, grids, and split counts. The launcher selects an H20
variant only on a GPU where 128 blocks do not fit at once. The [README](../../../../README.md)
lists the three variables that override this selection for testing: `NEXRT_H20_GEOMETRY`,
`NEXRT_FUSED_L2_CTAS`, and `NEXRT_CP_SPLITS`.

## Preflight checks

`scripts/preflight.sh` checks:

- Eight visible GPUs with compute capability 9.0, MIG disabled, and active NVLink links.
- Idle GPUs, unless `--allow-busy` was explicitly selected.
- CUDA Toolkit 12.9+, PyTorch 2.6+, and the symmetric-memory API used by NexRT.
- The selected Python environment's required imports, including SGLang for prefill.
- Configured weight directories, expected files, and writable build-lock storage.

These checks do not prove that NVLS multicast works, that a checkpoint's tensors match the
adapter, or that a new SGLang version preserves the expected layouts. Run the serving smoke
test to check the full pipeline. When a check fails, follow the
[troubleshooting guide](troubleshooting.md).
