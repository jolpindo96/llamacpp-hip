#!/usr/bin/env bash
# =============================================================================
# llamacpp-serve bootstrap - one-click llama.cpp OpenAI endpoint on AMD Instinct
# v4: binaries AND the Qwen3.8-Flash-Next vision projector ship BAKED in the image;
#     the volume build is an escape hatch. Flags are parsed before any download,
#     and a failure leaves the pod up with STATUS.md instead of a restart loop.
# Idempotent across stop/start. Durable state on /workspace only.
# Log: /workspace/serve-bootstrap.log   Status: /workspace/STATUS.md
# =============================================================================
set -u
WS=/workspace
BAKED_BIN=/opt/llama/bin
SERVER_ARGS="${SERVER_ARGS:--c 16384 -ngl 999 -fa on --load-mode none}"

# Parse-only check of every flag this script will pass, against one llama-server:
# all arguments are parsed, then --help prints usage and exits 0. A removed flag
# exits 1 instead - upstream deleted --no-mmap outright in 14a9d09f7, and a stale
# flag used to surface only after a 100+ GB download, as a crash loop. Parsing
# opens no files, so /dev/null stands in for paths. $2=all adds every optional flag
# and also fails on a DEPRECATED warning: --no-mmap printed one for seven weeks
# before it was deleted, so the image build treats deprecation as removal.
# Otherwise a deprecation is only reported, and the pod still serves.
args_ok(){
    local draft="" mmproj="" out dep
    if [ -n "${DRAFT_REPO:-}" ] || [ "${2:-}" = all ]; then
        draft="--spec-draft-model /dev/null --spec-draft-n-max 4"
    fi
    if [ -n "${MMPROJ_REPO:-}${MMPROJ:-}" ] || [ "${2:-}" = all ]; then
        mmproj="--mmproj /dev/null"
    fi
    if ! out=$("$1/llama-server" -m /dev/null $draft $mmproj $SERVER_ARGS \
              --host 0.0.0.0 --port 8000 ${API_KEY:+--api-key x} --help 2>&1 >/dev/null); then
        echo "llama-server in $1 rejected the flags: $(printf '%s\n' "$out" | grep -m1 '^error' || printf '%s\n' "$out" | tail -n1)"
        return 1
    fi
    dep=$(printf '%s\n' "$out" | grep -m1 'DEPRECATED' | sed 's/.*DEPRECATED/DEPRECATED/')
    [ -z "$dep" ] && return 0
    echo "llama-server in $1 warns about the flags: $dep"
    [ "${2:-}" != all ]
}

# The image build runs this against the baked binary, so a pin that drops a flag
# this script passes fails the build instead of every pod.
if [ "${1:-}" = "--check-args" ]; then args_ok "$BAKED_BIN" all; exit $?; fi

exec >>"$WS/serve-bootstrap.log" 2>&1
echo "=== bootstrap start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

BAKED_REV_FILE=/opt/llama/REVISION
LLAMA_DIR=$WS/llama.cpp
VOL_BIN=$LLAMA_DIR/build/bin
MODELS=$WS/models

LLAMA_REF="${LLAMACPP_REF:-}"                     # unset => use baked binaries
MODEL_REPO="${MODEL_REPO:?set MODEL_REPO env}"
MODEL_GLOB="${MODEL_GLOB:-*Q8_0*.gguf}"
DRAFT_GLOB="${DRAFT_GLOB:-*.gguf}"                # which draft quant to fetch
DRAFT_REPO="${DRAFT_REPO:-}"                      # a separate draft model; see §3
# Was hardcoded at 100GB (Qwen-sized) and false-failed smaller models.
MIN_MODEL_BYTES="${MIN_MODEL_BYTES:-100000000000}"

# Never exit: the start command execs this script, so an exit ends the container,
# takes sshd with it, and RunPod restarts it forever with nothing left to read.
# Record why in STATUS.md, keep the pod up for SSH, and let a stop end it.
fail(){
    echo "FAILED: $1"
    { echo "# FAILED ($(date -u +%Y-%m-%dT%H:%M:%SZ))"; echo; echo "$1"; echo
      echo "The pod stays up for SSH and bills until stopped. Last log lines:"
      echo '```'; tail -n 20 "$WS/serve-bootstrap.log"; echo '```'
    } > $WS/STATUS.md
    trap 'exit 0' TERM INT
    sleep infinity & wait $!
    exit 1
}

# The hf CLI lives in a venv on the volume. It used to be created only inside the
# model-download branch, so a cached model plus DRAFT_REPO/MMPROJ_REPO would call
# a binary that was never installed. Idempotent; safe to call repeatedly.
HF=$WS/hfenv/bin/hf
ensure_hf(){
    [ -x "$HF" ] && return 0
    python3 -m venv $WS/hfenv 2>/dev/null
    # huggingface_hub 1.x ships the hf CLI and the Xet transfer backend itself;
    # the old [hf_transfer,cli] extras no longer exist.
    $WS/hfenv/bin/pip install -q -U huggingface_hub || fail "pip hf"
}

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
    # Build for the GPU this pod actually has - one arch compiles fastest - and fall
    # back to the image default pair if the enumerator is missing.
    ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -m1 -v '^gfx000$')
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS="${ARCH:-gfx942;gfx950}" -DGGML_CCACHE=ON \
          -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null || fail "cmake configure"
    cmake --build build -j"$(nproc)" --target llama-server llama-bench >/dev/null || fail "build"
    BIN="$VOL_BIN"
    ACTIVE_REV="$(git rev-parse HEAD)"
    SOURCE="volume build (fresh)"
fi

# 1b. Parse every flag against these binaries before any download --------------
FLAG_WARNING=$(args_ok "$BIN") || fail "$FLAG_WARNING"
echo "flags parse on this llama-server: $SERVER_ARGS"
[ -z "$FLAG_WARNING" ] || echo "WARNING: $FLAG_WARNING"

# 2. Model shards (volume; datacenter-side pull, never the home uplink) ---------
mkdir -p "$MODELS"
# A shard set is "complete" only if every shard the 00001-of-N name promises is
# present. Checking mere presence is what bricked a pod: hf materialises each
# shard as a real .gguf the moment it finishes, so an interrupted 162GB pull left
# real .gguf files behind, the download was skipped on the next boot, and
# llama-server got an incomplete shard set -> exit -> restart -> forever, with
# sshd dying alongside the container so the pod could not be rescued.
shards_complete() {
    local dir="$1" first expected actual
    first=$(find "$dir" -name "*00001-of-*.gguf" 2>/dev/null | sort | head -1)
    if [ -z "$first" ]; then
        # unsharded model: a single .gguf is complete on its own
        find "$dir" -name "*.gguf" -print -quit 2>/dev/null | grep -q .
        return $?
    fi
    expected=$(basename "$first" | sed -n "s/.*of-0*\([0-9][0-9]*\)\.gguf/\1/p")
    [ -n "$expected" ] || return 1
    actual=$(find "$dir" -name "*-of-*.gguf" 2>/dev/null | wc -l)
    [ "$actual" -eq "$expected" ]
}

if ! shards_complete "$MODELS"; then
    ensure_hf
    # hf download resumes; re-running on a partial set is safe and cheap.
    HF_XET_HIGH_PERFORMANCE=1 "$HF" download "$MODEL_REPO" \
        --include "$MODEL_GLOB" --local-dir "$MODELS" || fail "model download"
    shards_complete "$MODELS" || fail "shard set still incomplete after download"
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

# 3. Optional separate draft model ----------------------------------------------
# A draft file passed here needs no --spec-type: llama.cpp infers it from the
# file's own GGUF metadata (dflash + markov head => draft-dspark). An MTP head
# embedded in the MAIN GGUF is not inferred - --spec-type defaults to none - so
# put --spec-type draft-mtp (and --spec-draft-n-max) in SERVER_ARGS instead.
DRAFT_FLAG=""
if [ -n "$DRAFT_REPO" ]; then
    mkdir -p $WS/draft
    ensure_hf
    shards_complete "$WS/draft" || \
        HF_XET_HIGH_PERFORMANCE=1 "$HF" download \
        "$DRAFT_REPO" --include "$DRAFT_GLOB" --local-dir $WS/draft || fail "draft download"
    DRAFT_FLAG="--spec-draft-model $(find $WS/draft -name '*.gguf' | sort | head -1) --spec-draft-n-max 4"
fi

# 3b. Vision projector (mmproj) -------------------------------------------------
# Qwen3.8-Flash-Next is a VLM, but the backbone GGUF does NOT carry the vision
# tower - llama.cpp loads it separately via --mmproj. A BF16 projector for
# Qwen3.8-Flash-Next is baked at $BAKED_MMPROJ (clip.projector_type
# qwen3vl_merger, which llama.cpp already registers as PROJECTOR_TYPE_QWEN3VL).
#
# OPT-IN, defaulting to off: passing --mmproj to a text-only model such as
# DeepSeek-V4-Flash would break that template, and both share this image.
#   MMPROJ=baked     use the projector baked into the image
#   MMPROJ=<path>    use an explicit file
#   MMPROJ_REPO=...  download one (with MMPROJ_GLOB) to $WS/mmproj, overriding
#                    the baked BF16 - the escape hatch if BF16 misbehaves.
BAKED_MMPROJ=/opt/llama/mmproj/qwen38-flash-next-bf16.gguf
MMPROJ="${MMPROJ:-}"
MMPROJ_REPO="${MMPROJ_REPO:-}"
MMPROJ_GLOB="${MMPROJ_GLOB:-*mmproj*.gguf}"
MMPROJ_FLAG=""
MMPROJ_PATH=""
if [ -n "$MMPROJ_REPO" ]; then
    ensure_hf
    mkdir -p $WS/mmproj
    find $WS/mmproj -name '*.gguf' -print -quit 2>/dev/null | grep -q . || \
        HF_XET_HIGH_PERFORMANCE=1 "$HF" download \
        "$MMPROJ_REPO" --include "$MMPROJ_GLOB" --local-dir $WS/mmproj || fail "mmproj download"
    MMPROJ_PATH=$(find $WS/mmproj -name '*.gguf' | sort | head -1)
elif [ "$MMPROJ" = "baked" ]; then
    MMPROJ_PATH="$BAKED_MMPROJ"
elif [ -n "$MMPROJ" ]; then
    MMPROJ_PATH="$MMPROJ"
fi
if [ -n "$MMPROJ_PATH" ]; then
    [ -s "$MMPROJ_PATH" ] || fail "mmproj missing or empty: $MMPROJ_PATH"
    [ "$(head -c4 "$MMPROJ_PATH")" = "GGUF" ] || fail "mmproj is not a GGUF: $MMPROJ_PATH"
    MMPROJ_FLAG="--mmproj $MMPROJ_PATH"
    echo "vision projector: $MMPROJ_PATH ($(stat -c%s "$MMPROJ_PATH") bytes)"
fi

# 4. Status + serve -------------------------------------------------------------
status(){ # $1 = STARTING, then READY once /health answers
cat > $WS/STATUS.md <<EOF
# $1 ($(date -u +%Y-%m-%dT%H:%M:%SZ))
- binaries: $SOURCE @ ${ACTIVE_REV:0:9}  (baked image rev: ${BAKED_REV:0:9})
- model: $MAIN ($TOTAL bytes, floor $MIN_MODEL_BYTES)
- draft: ${DRAFT_REPO:-none separate (an MTP head in the model needs --spec-type draft-mtp in args)}
- endpoint: :8000 OpenAI-compatible | GPU: $(rocm-smi --showuniqueid 2>/dev/null | grep -o '0x[0-9a-f]*' | head -1)
- vision: ${MMPROJ_PATH:-off (text-only; set MMPROJ=baked for Qwen3.8-Flash-Next)}
- args: $SERVER_ARGS $DRAFT_FLAG $MMPROJ_FLAG
- flag warnings: ${FLAG_WARNING:-none}
EOF
}
status STARTING
echo "=== serving ($SOURCE @ ${ACTIVE_REV:0:9}) ==="
# Supervised, not exec'd. As PID 1, any llama-server exit - an OOM at load, a model
# it cannot read - ended the container into a restart loop. Now an exit before
# /health ever answers is a config fault (fail: STATUS.md, pod stays up); an exit
# after serving is a runtime fault, so exit and let RunPod restart it as before.
# /health is public even with --api-key.
"$BIN/llama-server" \
  -m "$MAIN" $DRAFT_FLAG $MMPROJ_FLAG $SERVER_ARGS \
  --host 0.0.0.0 --port 8000 ${API_KEY:+--api-key "$API_KEY"} &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null; wait $SRV; exit 0' TERM INT
SERVED=0
while kill -0 $SRV 2>/dev/null; do
    if curl -sf -o /dev/null http://127.0.0.1:8000/health; then
        SERVED=1; status READY; echo "=== /health answered ==="; break
    fi
    sleep 5
done
wait $SRV; RC=$?
[ "$SERVED" = 1 ] || fail "llama-server exited with code $RC before /health ever answered"
echo "llama-server exited with code $RC after serving; exiting so RunPod restarts the pod"
exit $RC
