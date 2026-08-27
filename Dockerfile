# syntax=docker/dockerfile:1
#
# Pinned llama.cpp HIP build for MI300X (gfx942), baked into a bootable image.
#
# Base: rocm/dev-ubuntu-24.04:7.14.0-full
#   AMD renamed the dev-toolchain variant from "-complete" to "-full" at 7.14;
#   there is NO 7.14.0-complete and NO bare 7.14.0 runtime tag. 7.14.0-full is
#   byte-identical to :latest as of 2026-07-15. This is the exact ROCm the
#   2026-08-20 Batch-1 Engine Shootout ran on, so anchor numbers stay comparable.
#
# Both stages use the same fat base on purpose (see README "Fat vs slim").

ARG ROCM_TAG=7.14.0-full
ARG BASE=rocm/dev-ubuntu-24.04:${ROCM_TAG}

# --------------------------------------------------------------- assets stage
# Vision projector for Qwen3.8-Flash-Next.
#
# The backbone GGUF does NOT contain the vision tower; llama.cpp serves vision
# through a separate mmproj GGUF passed with --mmproj. Baking it removes a boot
# dependency and guarantees the projector matches what was validated.
#
# Its own stage on purpose: this layer must NOT be invalidated when LLAMA_REF
# changes, or every pin bump re-downloads 0.9GB.
#
# BF16 as requested. F16 is the conventional mmproj dtype and clip.cpp carries no
# BF16-specific handling (it relies on generic ggml paths), so if the vision path
# misbehaves, MMPROJ_REPO/MMPROJ_GLOB override this at boot without a rebuild.
FROM ${BASE} AS assets

ARG MMPROJ_URL=https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-BF16.gguf
ARG MMPROJ_SHA256=b115ede4c82c66393ed208015ad6ac9ec5094537051666be52a1d98e3f767344
ARG MMPROJ_DEST=/opt/llama/mmproj/qwen38-flash-next-bf16.gguf

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Verified by content, not by exit code (shootout lesson): checksum, GGUF magic,
# and the projector type llama.cpp actually registers (PROJECTOR_TYPE_QWEN3VL).
RUN set -eu; \
    mkdir -p "$(dirname "${MMPROJ_DEST}")"; \
    curl -fsSL --retry 5 --retry-delay 3 -o "${MMPROJ_DEST}" "${MMPROJ_URL}"; \
    echo "${MMPROJ_SHA256}  ${MMPROJ_DEST}" | sha256sum -c -; \
    [ "$(head -c4 "${MMPROJ_DEST}")" = "GGUF" ] || { echo "FAIL: not a GGUF"; exit 1; }; \
    head -c 1048576 "${MMPROJ_DEST}" | grep -aqm1 qwen3vl_merger \
        || { echo "FAIL: projector type is not qwen3vl_merger"; exit 1; }; \
    echo "PASS: mmproj $(stat -c%s "${MMPROJ_DEST}") bytes, qwen3vl_merger"

# ---------------------------------------------------------------- build stage
FROM ${BASE} AS build

ARG LLAMA_REF
ARG GPU_TARGETS=gfx942

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache libcurl4-openssl-dev libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN test -n "${LLAMA_REF}" || { echo "LLAMA_REF build-arg is required"; exit 1; } \
 && git clone https://github.com/ggml-org/llama.cpp . \
 && git checkout -q "${LLAMA_REF}" \
 && git rev-parse HEAD | tee /REVISION

# --- Numerics gate: HIP builds must stay IEEE-conformant. -------------------
# Upstream removed -funsafe-math-optimizations from ggml-hip in e79e4bf66
# (#26696, 2026-08-12) because it enables -fassociative-math, which reassociates
# FP reductions, can flip greedy argmax, and desyncs MTP speculative decode from
# the non-speculative baseline. Any pin older than that silently reintroduces it.
# Note: -use_fast_math in ggml-cuda/CMakeLists.txt is guarded by CUDAToolkit_FOUND
# and is NOT applied to HIP builds, so it is deliberately not checked here.
RUN set -eu; \
    echo "--- HIP_CLEAN_OF_FASTMATH ---"; \
    if ! git merge-base --is-ancestor e79e4bf66 HEAD; then \
        echo "FAIL: pin predates e79e4bf66 (unsafe-math removal); refusing to build"; \
        exit 1; \
    fi; \
    if grep -rEn 'ffast-math|funsafe-math|fassociative-math|use_fast_math' \
            ggml/src/ggml-hip/CMakeLists.txt; then \
        echo "FAIL: fast-math flag present in ggml-hip"; \
        exit 1; \
    fi; \
    echo "PASS: HIP build is clean of fast-math"

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DGPU_TARGETS="${GPU_TARGETS}" \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --config Release -j"$(nproc)" \
        --target llama-server llama-bench llama-quantize

# Collect binaries + the shared libs they were built against.
RUN set -eu; \
    mkdir -p /opt/llama/bin /opt/llama/lib; \
    cp build/bin/llama-server build/bin/llama-bench build/bin/llama-quantize /opt/llama/bin/; \
    find build -name '*.so*' -exec cp -P {} /opt/llama/lib/ \; ; \
    cp /REVISION /opt/llama/REVISION

# Record the real linkage. Feeds the slim variant (README "Fat vs slim") so the
# runtime package list is derived, never guessed. Also fails loudly on "not found".
RUN set -eu; \
    LD_LIBRARY_PATH=/opt/llama/lib:/opt/rocm/lib ldd /opt/llama/bin/llama-server \
        | sort > /opt/llama/DEPS.txt; \
    cat /opt/llama/DEPS.txt; \
    if grep -q 'not found' /opt/llama/DEPS.txt; then \
        echo "FAIL: unresolved shared libraries in llama-server"; exit 1; \
    fi

# -------------------------------------------------------------- runtime stage
FROM ${BASE} AS runtime

ARG LLAMA_REF
LABEL org.opencontainers.image.revision="${LLAMA_REF}" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.title="llamacpp-hip" \
      org.opencontainers.image.description="Pinned llama.cpp HIP (gfx942) server image for RunPod MI300X"

# serve-bootstrap.sh needs: curl (health/probes), python3-venv (hf download),
# openssh-server (RunPod exec), ca-certificates (HTTPS to HF/GHCR).
# apt's openssh-server postinst generates SSH host keys at BUILD time. Baked into
# a public image that means every pod boots with the same host keypair, whose
# private half anyone can pull. They are deleted here; the RunPod start command
# runs 'ssh-keygen -A' to generate fresh per-pod keys at boot.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3-venv openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_*

COPY --from=build  /opt/llama /opt/llama
# Must follow the build COPY: that one writes the whole /opt/llama tree.
COPY --from=assets /opt/llama/mmproj /opt/llama/mmproj

# MANDATORY (shootout lesson): ROCm debs do not register their libs with the
# dynamic linker. Without this the binaries build fine and fail at runtime with
# unresolvable libs. /opt/llama/lib carries the ggml/llama shared objects.
RUN printf '/opt/rocm/lib\n/opt/llama/lib\n' > /etc/ld.so.conf.d/rocm.conf \
 && ldconfig

COPY serve-bootstrap.sh /usr/local/bin/serve-bootstrap.sh
RUN chmod +x /usr/local/bin/serve-bootstrap.sh

ENV PATH=/opt/llama/bin:${PATH}
ENV LLAMA_ARG_HOST=0.0.0.0

# Verify the baked binary actually runs in the runtime image, rather than
# trusting that COPY succeeded (shootout lesson: exit codes lie).
RUN llama-server --version \
 && llama-bench --help >/dev/null \
 && test -s /opt/llama/mmproj/qwen38-flash-next-bf16.gguf

EXPOSE 8000
CMD ["/usr/local/bin/serve-bootstrap.sh"]
