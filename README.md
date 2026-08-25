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

### Model survey (corrected 2026-08-25)

**An earlier revision of this file was wrong.** It claimed no Q8 quant existed, that
Q8 could never fit an MI300X, that `UD-Q3_K_M` was damaged, and that no DSpark GGUF
shipped in the unsloth repo. All four were artifacts of one mistake: the HuggingFace
tree API paginates via a `Link: rel="next"` header, and the probe read page 1 of 2 —
36 of 54 files. The truncation cut the listing off alphabetically at `UD-Q3_K_M`,
hiding every quant after it.

The actual repo contents:

| Quant | Shards | Size |
|---|---|---|
| UD-IQ1_S / UD-IQ1_M | 3 | 82.5 / 86.9 GB |
| UD-IQ2_XXS / UD-IQ2_M | 3 | 90.9 GB |
| UD-Q2_K_XL | 3 | 96.8 GB |
| UD-IQ3_XXS / UD-IQ3_S | 4 | 104.2 / 116.1 GB |
| UD-Q3_K_M / UD-Q3_K_XL | 4 | 128.1 / 128.2 GB |
| UD-IQ4_XS / UD-IQ4_NL | 4 | 136.7 GB |
| UD-Q4_K_XL | 5 | 155.1 GB |
| **UD-Q8_K_XL** | 5 | **161.9 GB** |
| `dspark/…-BF16.gguf` | 1 | 11.3 GB |
| `…-dspark-…-Q8_0.gguf` (root) | 1 | 10.9 GB |

Unsloth documents `UD-Q8_K_XL` as full-precision lossless. **Any paginated listing
must follow `Link: rel="next"`** — a truncated one looks complete and is not.

### DSpark drafters

The model repo ships its own, and both use llama.cpp's canonical `dflash`
architecture with the expected `markov_w1` / `markov_w2` / `conf_proj` tensors:

```
…-dspark-…-Q8_0.gguf    10.9 GB   arch=dflash  block_count=3  block_size=5
dspark/…-BF16.gguf      11.3 GB   arch=dflash  block_count=3  block_size=5
```

Third-party conversions are a trap worth knowing about. Of four popular ones probed,
only `Unkto/…-DSpark-Drafter-IQ1M-IQ2XXS-GGUF` declares `dflash`; the others use
`deepseek_v4_flash_dspark_draft`, `deepseek4-dspark`, and `deepseek4-dflash-draft`,
none of which llama.cpp registers — `common/speculative.cpp` gates on
`arch != "dflash"`, so they carry correct weights and still fail at load. The
most-downloaded drafter is one of the rejected ones.

### Measured on MI300X, 2026-08-25

`UD-Q8_K_XL` + the in-repo `Q8_0` drafter, one MI300X (206.1 GB usable):

| KV cache | Context | VRAM | Decode | Draft accept |
|---|---|---|---|---|
| **q8_0** | **1,048,576 (full 1M)** | 192.9 GB (94%) | **45.2 tok/s** | 70.3% |
| bf16 | 524,288 | 190.9 GB (93%) | 40.7 tok/s | 64.6% |

q8_0 KV yields 2x the context of bf16 for the same memory and runs ~11% faster. The
model's full 1M trained context fits only because of it.

An earlier IQ4_XS run measured speculation's contribution directly: **31.2 tok/s with
no drafter versus 49.9–51.9 with one, ~1.6x.**

### `-c 0` is unsafe with a draft model

llama.cpp logs `failed to measure the memory of the extra model, fitting without it`
— the memory fitter sizes the context **ignoring the drafter's ~11 GB**. With q8_0 KV
that landed at 94% and worked; with bf16 it overcommitted, OOMed, and put the pod in
an unrecoverable crash loop. Set `-c` explicitly whenever a drafter is loaded.

## The pin

`6036c635e` (2026-08-24 10:43 +0300). This is a real commit on `origin/master`, but
it is **not** master's tip — at time of writing `origin/master` is `f280b2698`, nine
commits ahead. `6036c635e` is deliberately chosen anyway: none of those nine touch
`deepseek4`, `dflash`, `common/speculative.*`, or `ggml-hip` (the only one in that
area is `a14dba686`, cosmetic CUDA/Metal device naming), so they add nothing here
while subtracting soak time.

Every requirement below is met only at or after this commit:

| Requirement | Landed | In `07822bddf`? |
|---|---|---|
| `deepseek4` base arch (V4 Flash + MTP + DSpark) | `596a5795b` 08-02 | yes |
| `dflash` draft arch + `draft-dspark` spec type | `d1b34251b` 06-28 | yes |
| IEEE-conformant HIP (no unsafe-math) | `e79e4bf66` 08-13 | yes |
| `draft-mtp` with embeddings fix | `2c6b141ef` 08-22 | **no** |
| DeepseekV4 rollback with multi-seq | `b0539c43e` 08-23 | **no** |
| `ggml_clamp` fix | `6036c635e` 08-24 | **no** |

Two of those are load-bearing now that speculative decoding is in play.
`b0539c43e` fixes rollback under concurrent sequences — the exact path a server
exercises on every draft rejection. And `dflash` reads `swiglu_clamp_exp` /
`swiglu_clamp_shexp` (`src/models/dflash.cpp:36`), which the drafter GGUF supplies,
so the `ggml_clamp` fix touches an op this model family depends on. (The specific
failure mode of that bug was not verified here; pinning at or after it is simply the
safe side.)

Trade-off: `6036c635e` is a day old with limited soak. `bf0a29cc1` is the
conservative alternative — it keeps the rollback and embeddings fixes and gives up
only the clamp fix. Going the other way, `f280b2698` (master tip) adds only Metal,
WebGPU, and convert-script changes irrelevant to a HIP DeepSeek-V4 build.

### Provenance of this analysis

The architecture findings above were read from a local llama.cpp checkout detached at
`6036c635e`, whose only working-tree modification is `ggml/src/ggml-cuda/CMakeLists.txt`
(a local `-use_fast_math` removal). That file is CUDA-only, guarded by
`CUDAToolkit_FOUND`, and is not among the files the analysis reads — so it does not
affect any conclusion here. The image itself clones upstream at `LLAMA_REF` and is
independent of that checkout.

### Verified drafter configuration

`Unkto/DeepSeek-V4-Flash-0731-DSpark-Drafter-IQ1M-IQ2XXS-GGUF`, probed header:

- `general.architecture = dflash`, `markov_w1.weight` present → auto-detects as
  `draft-dspark` (`common/speculative.cpp:2265`). No `--spec-type` needed.
- `dflash.block_size = 5` is present, so the loader does **not** fall back to its
  default of 16. `dflash.sample_from_anchor` is absent → false.
- Therefore `n_draft_max = block_size - 1 = 4`, which is exactly what the bootstrap
  passes as `--spec-draft-n-max 4`.
- `mask_token_id = 128799` matches the base model's `dspark_noise_token_id`.
- 3 layers, `256x594M`, imatrix-quantized.

### Template environment

```
MODEL_REPO      = unsloth/DeepSeek-V4-Flash-0731-GGUF
MODEL_GLOB      = *UD-Q8_K_XL*
DRAFT_REPO      = unsloth/DeepSeek-V4-Flash-0731-GGUF
DRAFT_GLOB      = *dspark*Q8_0*
LLAMACPP_REF    = (unset - use baked binaries)
MIN_MODEL_BYTES = 155000000000
SERVER_ARGS     = -c 1048576 -ngl 999 -fa on --no-mmap -ctk q8_0 -ctv q8_0
```

Total resident: 161.9 GB weights + 10.9 GB draft + q8_0 KV at 1M ctx = **192.9 GB**
of 206.1 GB, measured.
