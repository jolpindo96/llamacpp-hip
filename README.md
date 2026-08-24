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

Whether the `UD-Q8_K_XL` GGUF bundles the ~20B draft module is answerable off-pod by
range-reading the GGUF header for the DSpark tensors named in `src/models/dflash.cpp`
(`markov_w1`, `markov_w2`, `conf_proj`). *Finding to be recorded here.*
