#!/usr/bin/env bash
# =============================================================================
# llamacpp-serve bootstrap - one-click llama.cpp OpenAI endpoint on MI300X
# v2: binaries ship BAKED in the image; the volume build is now an escape hatch.
# Idempotent across stop/start. Durable state on /workspace only.
# Log: /workspace/serve-bootstrap.log   Status: /workspace/STATUS.md
# =============================================================================
set -u
WS=/workspace
exec >>"$WS/serve-bootstrap.log" 2>&1
echo "=== bootstrap start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

BAKED_BIN=/opt/llama/bin
BAKED_REV_FILE=/opt/llama/REVISION
LLAMA_DIR=$WS/llama.cpp
VOL_BIN=$LLAMA_DIR/build/bin
MODELS=$WS/models

LLAMA_REF="${LLAMACPP_REF:-}"                     # unset => use baked binaries
MODEL_REPO="${MODEL_REPO:?set MODEL_REPO env}"
MODEL_GLOB="${MODEL_GLOB:-*Q8_0*.gguf}"
DRAFT_GLOB="${DRAFT_GLOB:-*.gguf}"                # which draft quant to fetch
DRAFT_REPO="${DRAFT_REPO:-}"                      # usually unnecessary; see note in §4
SERVER_ARGS="${SERVER_ARGS:--c 16384 -ngl 999 -fa on --no-mmap}"
# Was hardcoded at 100GB (Qwen-sized) and false-failed smaller models.
MIN_MODEL_BYTES="${MIN_MODEL_BYTES:-100000000000}"

fail(){ echo "FAILED: $1"; echo "FAILED: $1 ($(date -u))" > $WS/STATUS.md; exit 1; }

BAKED_REV="$(cat $BAKED_REV_FILE 2>/dev/null || echo '')"

# 1. Resolve which binaries to serve with ---------------------------------------
# Order: baked (ref unset or matches baked) -> matching volume build -> build it.
# The old check skipped the build whenever ANY working llama-server existed,
# which silently ignored LLAMACPP_REF on a reused volume.
ref_matches(){ # $1 = candidate rev, $2 = requested ref (possibly abbreviated)
  [ -n "$1" ] && [ -n "$2" ] && case "$1" in "$2"*) return 0;; esac; return 1
}

BIN=""
if [ -x "$BAKED_BIN/llama-server" ] && { [ -z "$LLAMA_REF" ] || ref_matches "$BAKED_REV" "$LLAMA_REF"; }; then
    BIN="$BAKED_BIN"
    ACTIVE_REV="$BAKED_REV"
    SOURCE="baked image"
    echo "using BAKED binaries at ${BAKED_REV:0:9} (LLAMACPP_REF=${LLAMA_REF:-<unset>})"
elif [ -x "$VOL_BIN/llama-server" ] \
     && ref_matches "$(cd $LLAMA_DIR 2>/dev/null && git rev-parse HEAD 2>/dev/null)" "$LLAMA_REF"; then
    BIN="$VOL_BIN"
    ACTIVE_REV="$(cd $LLAMA_DIR && git rev-parse HEAD)"
    SOURCE="volume build"
    echo "using VOLUME build at ${ACTIVE_REV:0:9}"
else
    echo "building llama.cpp at ref '$LLAMA_REF' on the volume (escape hatch)"
    [ -n "$LLAMA_REF" ] || fail "no baked binaries and LLAMACPP_REF unset"
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v hipcc >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y -qq build-essential cmake ccache git curl \
            libcurl4-openssl-dev rocm-hip-sdk || fail "apt toolchain"
    fi
    [ -d "$LLAMA_DIR/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" || fail "clone"
    cd "$LLAMA_DIR" && git fetch --all --tags -q && git checkout -q "$LLAMA_REF" || fail "checkout $LLAMA_REF"
    # Same numerics gate the image enforces (see Dockerfile / README).
    git merge-base --is-ancestor e79e4bf66 HEAD || fail "ref predates e79e4bf66 unsafe-math removal"
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx942 -DGGML_CCACHE=ON \
          -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null || fail "cmake configure"
    cmake --build build -j"$(nproc)" --target llama-server llama-bench >/dev/null || fail "build"
    BIN="$VOL_BIN"
    ACTIVE_REV="$(git rev-parse HEAD)"
    SOURCE="volume build (fresh)"
fi

# 2. Model shards (volume; datacenter-side pull, never the home uplink) ---------
mkdir -p "$MODELS"
if ! find "$MODELS" -name "*.gguf" -print -quit 2>/dev/null | grep -q .; then
    python3 -m venv $WS/hfenv 2>/dev/null
    $WS/hfenv/bin/pip install -q -U "huggingface_hub[hf_transfer,cli]" || fail "pip hf"
    HF_HUB_ENABLE_HF_TRANSFER=1 $WS/hfenv/bin/hf download "$MODEL_REPO" \
        --include "$MODEL_GLOB" --local-dir "$MODELS" || fail "model download"
fi
# Shootout lesson: success codes lie - verify sizes, not exit codes.
TOTAL=$(du -sb "$MODELS" | cut -f1)
[ "$TOTAL" -gt "$MIN_MODEL_BYTES" ] \
    || fail "model dir only $TOTAL bytes (floor $MIN_MODEL_BYTES) - incomplete download"
# unsloth publishes shards under a per-quant subdirectory (UD-IQ4_XS/...), so
# these must recurse. MAIN is a full path, not a basename.
MAIN=$(find "$MODELS" -name "*00001-of*.gguf" | sort | head -1)
[ -n "$MAIN" ] || MAIN=$(find "$MODELS" -name "*.gguf" | sort | head -1)
[ -n "$MAIN" ] || fail "no .gguf found under $MODELS"

# 3. Optional explicit draft head ----------------------------------------------
# Usually NOT needed: since #27005 / #26814 llama.cpp auto-detects the MTP draft
# type from GGUF metadata, and #26458 resolves the DSpark sidecar. Set DRAFT_REPO
# only to override with a separate (e.g. Q4) draft.
DRAFT_FLAG=""
if [ -n "$DRAFT_REPO" ]; then
    mkdir -p $WS/draft
    find $WS/draft -name "*.gguf" -print -quit 2>/dev/null | grep -q . || \
        HF_HUB_ENABLE_HF_TRANSFER=1 $WS/hfenv/bin/hf download \
        "$DRAFT_REPO" --include "$DRAFT_GLOB" --local-dir $WS/draft || fail "draft download"
    DRAFT_FLAG="--spec-draft-model $(find $WS/draft -name '*.gguf' | sort | head -1) --spec-draft-n-max 4"
fi

# 4. Status + serve -------------------------------------------------------------
cat > $WS/STATUS.md <<EOF
# READY ($(date -u +%Y-%m-%dT%H:%M:%SZ))
- binaries: $SOURCE @ ${ACTIVE_REV:0:9}  (baked image rev: ${BAKED_REV:0:9})
- model: $MAIN ($TOTAL bytes, floor $MIN_MODEL_BYTES)
- draft: ${DRAFT_REPO:-auto-detected from GGUF metadata if the model ships an MTP head}
- endpoint: :8000 OpenAI-compatible | GPU: $(rocm-smi --showuniqueid 2>/dev/null | grep -o '0x[0-9a-f]*' | head -1)
- args: $SERVER_ARGS $DRAFT_FLAG
EOF
echo "=== serving ($SOURCE @ ${ACTIVE_REV:0:9}) ==="
exec "$BIN/llama-server" \
  -m "$MAIN" $DRAFT_FLAG $SERVER_ARGS \
  --host 0.0.0.0 --port 8000 ${API_KEY:+--api-key "$API_KEY"}
