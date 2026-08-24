# llamacpp-hip

Pinned [llama.cpp](https://github.com/ggml-org/llama.cpp) HIP builds for AMD MI300X
(`gfx942`), baked into a bootable image and published to GHCR.

Push a tag → GitHub Actions builds → GHCR stores it → a RunPod template boots
straight to an OpenAI-compatible endpoint. No local Docker, nothing large ever
leaves a home connection.

```bash
git tag llama-6036c635e && git push origin llama-6036c635e
```

produces `ghcr.io/<owner>/llamacpp-hip:6036c635e`.

`LLAMA_REF` is the only thing that changes on a routine update — it comes from the
tag name. `:latest` is never published automatically; pinned tags are the product,
and a moving `:latest` would defeat the reproducibility the pin exists for. It is
available as an opt-in checkbox on manual `workflow_dispatch` runs.

## Provenance

Encodes the recipe validated in the **2026-08-20 Batch-1 Engine Shootout**, where
llama.cpp won single-user decode on MI300X by ~1.6× over SGLang (SGLang won prefill
by ~5×). Reference anchor: Qwen3.5-122B-A10B Q8_0 at 91.3 tok/s (llama-bench tg128)
and 88.6 tok/s served.

## Base image: `rocm/dev-ubuntu-24.04:7.14.0-full`

Verified against the registry rather than assumed, because tag names are hypotheses:

| Tag | Compressed | Updated | Notes |
|---|---|---|---|
| `7.14.0-full` | 7.95 GB | 2026-07-15 | newest; byte-identical to `:latest` |
| `7.2.4-complete` | 7.40 GB | 2026-05-28 | last of the `-complete` line |
| `7.2.4` (bare) | 1.22 GB | 2026-05-28 | runtime-only variant |

Two findings that shape the Dockerfile:

- **AMD renamed the dev-toolchain variant `-complete` → `-full` at 7.14.** There is
  no `7.14.0-complete`. Anything referencing one is wrong.
- **The 7.14 line ships only the fat variant** — no bare `7.14.0` runtime tag. Slim
  and "newest ROCm" are therefore not simultaneously available from Docker Hub.

`7.14.0-full` also sets `PATH=/opt/rocm/bin:...` and `ROCM_PATH=/opt/rocm`, which
`7.2.4-complete` does *not* — that is why upstream's Dockerfile resolves the
compiler through `hipconfig` instead of trusting `PATH`. This image does the same,
so it stays correct on either line.

### Fat vs slim

Both stages currently use the same fat base. Deliberate:

- **Version parity with the shootout.** The 91.3 anchor was measured on ROCm 7.14.0.
  Dropping to the 7.2.x line to get the 1.22 GB runtime tag would change rocBLAS
  under the benchmark and make new anchors non-comparable.
- **No missing-library class of bug.** The full ROCm runtime is already present.
- **It costs ~2–4 min of pull instead of ~1** at datacenter bandwidth, so both fit
  inside the 10-minute cold-boot budget. Size here is registry hygiene, not latency.

This overshoots the "well under 10 GB" target. The slim variant is a follow-up tag:
the build stage writes `/opt/llama/DEPS.txt` (`ldd` of the real `llama-server`) so
the runtime package list is **derived, not guessed** — the one input that variant
needs. Nothing else changes.

## Numerics: the fast-math floor

Builds are IEEE-conformant. Upstream removed `-funsafe-math-optimizations` from
`ggml-hip` in **`e79e4bf66` (#26696, 2026-08-12)** because it enables
`-fassociative-math`, which reassociates FP reductions, can flip greedy argmax, and
**desyncs MTP speculative decode from the non-speculative baseline**.

The Dockerfile enforces this as a hard gate before the expensive build:

1. `git merge-base --is-ancestor e79e4bf66 HEAD` — refuses any pin older than the fix.
2. greps `ggml/src/ggml-hip/CMakeLists.txt` for fast-math flags.

`-use_fast_math` in `ggml-cuda/CMakeLists.txt` is guarded by `CUDAToolkit_FOUND` and
is **not** applied to HIP builds, so it is deliberately not checked. HIP compiles the
same `.cu` sources through `ggml-hip/CMakeLists.txt`, which never sets those flags.

## CI notes

- **Runner disk.** The ROCm dev base plus the build tree exceeds a stock runner.
  Mitigated with `ggml-org/free-disk-space@v1.3.1` — the same action and settings
  upstream llama.cpp uses for its own rocm image.
- **No `repo.radeon.com` apt pin is needed.** The docker tag *is* the version pin.
  The previous bootstrap used `.../rocm/apt/latest`, which was unpinned; baking
  removes that code path entirely.
- **Auth is the built-in `GITHUB_TOKEN`** with `packages: write`. No PATs.
- Single-target `gfx942`. Multi-target (`gfx90a;gfx942;gfx1201`) multiplies compile
  time and is a stretch goal only once this is green.

## serve-bootstrap.sh

Baked at `/usr/local/bin/serve-bootstrap.sh`. Resolution order:

1. **Baked binaries** if `/opt/llama/bin/llama-server` exists and `LLAMACPP_REF` is
   unset or matches `/opt/llama/REVISION`.
2. **Volume build** if one exists and its `git rev-parse` matches `LLAMACPP_REF`.
3. **Build at `LLAMACPP_REF`** on the volume — the escape hatch for experiments.

The previous version skipped the build whenever *any* working `llama-server` was on
the volume, which silently ignored `LLAMACPP_REF` on a reused volume.

The model-size sanity floor is now `MIN_MODEL_BYTES` (default 100 GB) instead of a
hardcoded Qwen-sized constant that false-failed smaller models.

### Environment

| Var | Default | Notes |
|---|---|---|
| `MODEL_REPO` | *(required)* | HF repo id |
| `MODEL_GLOB` | `*Q8_0*.gguf` | shard filter |
| `LLAMACPP_REF` | *(unset)* | unset ⇒ use baked binaries |
| `SERVER_ARGS` | `-c 16384 -ngl 999 -fa on --no-mmap` | |
| `MIN_MODEL_BYTES` | `100000000000` | download sanity floor |
| `DRAFT_REPO` | *(unset)* | usually unnecessary — see below |
| `API_KEY` | *(unset)* | sets `--api-key` |

**On draft models:** since #27005 and #26814 llama.cpp auto-detects the MTP draft
type from GGUF metadata, and #26458 resolves the DSpark sidecar. If the model ships
its own MTP head, leave `DRAFT_REPO` unset and let llama.cpp find it. Set it only to
override with a separate (e.g. Q4) draft.

## DSpark / MTP

DeepSeek-V4 MTP + DSpark landed upstream in `596a5795b` (#25784, 2026-08-02) — there
is no MTP work to implement here, only a commit to pin. Correcting a claim in the
original plan: the shootout pin `07822bddf` **does** already carry `LLM_ARCH_DEEPSEEK4`,
the V4 Flash 0731 chat template (#26398), and MTP draft auto-detection (#27005). It is
not V4-Flash-incapable.

Current master is still the better first pin, for a different reason — three fixes
land on top of `07822bddf`:

| Commit | Change |
|---|---|
| `b0539c43e` | DeepseekV4: fix rollback with multi-seq (#26756) — the serving path |
| `2c6b141ef` | common: fix draft-mtp with embeddings (#26352, #27299) |
| `bf0a29cc1` | Deepseek 4: `-sm tensor` (#26490) |

### Probe result (2026-08-24, off-pod, $0)

Range-read the GGUF headers of every quant in `unsloth/DeepSeek-V4-Flash-0731-GGUF`
directly from HuggingFace. Four findings, all of which change the serving plan:

**1. There is no Q8. The repo tops out at IQ4.**

| Quant | Shards | Size | Tensors |
|---|---|---|---|
| UD-IQ1_S / UD-IQ1_M | 3 | 82.5 / 86.9 GB | 1328 |
| UD-IQ2_XXS / UD-IQ2_M | 3 | 90.9 GB | 1328 |
| UD-Q2_K_XL | 3 | 96.8 GB | 1328 |
| UD-Q3_K_M | 3 | 98.8 GB | **1032 — suspect** |
| UD-IQ3_XXS / UD-IQ3_S | 4 | 104.2 / 116.1 GB | 1328 |
| **UD-IQ4_XS / UD-IQ4_NL** | 4 | **136.7 GB** | 1328 |

`MODEL_GLOB=*UD-Q8_K_XL*` matches nothing. And Q8 was never viable here: the model
is `256x8.4B` (43 layers, 256 experts, 6 active), so Q8 would land near 280–300 GB
against MI300X's 192 GB HBM. **UD-IQ4_XS at 136.7 GB is the right target** — it fits
with roughly 55 GB left for KV cache.

**2. `UD-Q3_K_M` carries 1032 tensors where every other quant carries 1328.** That is
296 missing, not a different layout. Treat it as incomplete and avoid it.

**3. No DSpark / MTP tensors in any base quant.** All 1328 tensors are `blk.0`–`blk.42`
plus `output.weight`, `output_norm.weight`, `token_embd.weight`, and the
hyper-connection tensors `output_hc_{base,fn,scale}.weight`. Zero matches for
`markov_w1`, `conf_proj`, `nextn`, `dspark`, or `eh_proj`.

The upstream base model *does* declare `num_nextn_predict_layers: 1` plus
`dspark_block_size` / `dspark_markov_rank` / `dspark_target_layer_ids` in its
`config.json` — the conversion dropped them. So the draft module is **not** in these
GGUFs.

**Consequence: the draft head ships separately, and most community conversions
will not load.** The unsloth repo is base-only, but standalone DSpark drafter GGUFs
do exist. Their `general.architecture` strings were probed against llama.cpp's
registry in `src/llama-arch.cpp`:

| Drafter repo | Arch string | Size | Verdict |
|---|---|---|---|
| `alessandrobologna/...0731-DSpark-Drafter-GGUF` | `deepseek_v4_flash_dspark_draft` | 7.0–10.9 GB | **rejected** — not registered |
| `bleysg/...DSpark-drafter-GGUF` | `deepseek4-dspark` | 6.97 GB | **rejected** — not registered |
| `Lucebox/...0731-DSpark-GGUF` | `deepseek4-dflash-draft` | 10.7–11.3 GB | **rejected** — not registered |
| **`Unkto/...0731-DSpark-Drafter-IQ1M-IQ2XXS-GGUF`** | **`dflash`** | **4.8 / 5.5 GB** | **loads** |

Only `dflash` is registered (`llama-arch.cpp:142`), and only the Unkto conversion
uses it — with exactly the canonical tensor names the loader expects: `markov_w1`,
`markov_w2`, `conf_proj` (`llama-arch.cpp:651-653`). The other three carry the right
weights under architecture strings llama.cpp does not know, and will fail at load.
Download counts are not a signal here: the most-downloaded drafter is one of the
rejected ones.

Recommended pairing: base `UD-IQ4_XS` (136.7 GB) + drafter `IQ2_XXS` (5.54 GB) =
**142.2 GB**, leaving ~50 GB for KV cache on a 192 GB MI300X. The drafter is a
3-layer `256x594M` MoE. IQ1_M/IQ2_XXS is more aggressive than the Q4 the original
plan suggested, but draft quality is laundered by verification — the cost is
acceptance rate, not output quality.

**4. VRAM budget is settled without renting anything.** 136.7 GB of weights on 192 GB.
The original 162 GB / 182 GB figures corresponded to no actual file. Note
`context_length` is 1048576 with `rope.scaling.original_context_length` 65536 — the
default `-c 16384` stays appropriate.
