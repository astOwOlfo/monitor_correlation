#!/usr/bin/env bash
# Install flash-attn into $VENV_DIR.
#
# This is not a locked dependency: PyPI publishes no wheel for the torch build this repo pins
# (torch 2.11 / CUDA 13), and the source build needs CUDA_HOME plus a job count that a lockfile
# cannot express. It is also not optional - Verl's model engine unpads every log-prob batch
# through flash_attn.bert_padding, so training does not start without it.
#
# Order of preference:
#   1. A prebuilt wheel matching this exact (flash-attn, CUDA, torch, python, arch) combination.
#      https://github.com/mjun0812/flash-attention-prebuild-wheels publishes these; installing one
#      takes seconds instead of the ~30-90 minutes a source build costs.
#   2. Compiling from source, targeting only the compute capability of the local GPU.
set -uo pipefail

FA_VERSION="${FA_VERSION:-2.8.3}"
WHEEL_REPO="${WHEEL_REPO:-mjun0812/flash-attention-prebuild-wheels}"
VENV_PY="${VENV_DIR:?VENV_DIR must be set}/bin/python"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

cat > "${WORK}/probe.py" <<'PY'
import platform, sys, torch

torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cuda = (torch.version.cuda or "").replace(".", "")
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
arch = "linux_x86_64" if platform.machine() == "x86_64" else "linux_aarch64"
cc = "%d.%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else ""
print(f"{torch_mm} {cuda} {py} {arch} {cc}")
PY

PROBE="$("$VENV_PY" "${WORK}/probe.py")" || {
  echo "flash-attn: could not inspect the torch install in ${VENV_DIR}" >&2
  exit 1
}
read -r TORCH_VER CUDA_VER PY_TAG ARCH_TAG CC <<< "${PROBE}"

WHEEL_NAME="flash_attn-${FA_VERSION}+cu${CUDA_VER}torch${TORCH_VER}-${PY_TAG}-${PY_TAG}-${ARCH_TAG}.whl"
echo "flash-attn: looking for a prebuilt ${WHEEL_NAME}"

# Newest release publishing this wheel wins; the repo spreads its builds across many tags.
cat > "${WORK}/find_wheel.py" <<'PY'
import json, sys

want = sys.argv[1]
for release in json.load(sys.stdin):
    for asset in release.get("assets", []):
        if asset["name"] == want:
            print(asset["browser_download_url"])
            raise SystemExit
PY

WHEEL_URL="$(curl -sL --max-time 120 "https://api.github.com/repos/${WHEEL_REPO}/releases?per_page=100" \
  | "$VENV_PY" "${WORK}/find_wheel.py" "${WHEEL_NAME}" 2>/dev/null)"

if [ -n "${WHEEL_URL}" ]; then
  # Keep the wheel's real filename: uv rejects a wheel whose name it cannot parse.
  if curl -fsSL --max-time 900 -o "${WORK}/${WHEEL_NAME}" "${WHEEL_URL}"; then
    # --no-deps so the local version tag (+cu130torch2.11) does not trigger a resolve.
    if uv pip install --python "$VENV_PY" --no-deps "${WORK}/${WHEEL_NAME}" \
       && "$VENV_PY" -c "import flash_attn, flash_attn_2_cuda" 2>/dev/null; then
      echo "flash-attn: installed prebuilt ${WHEEL_NAME}"
      exit 0
    fi
    echo "flash-attn: prebuilt wheel unusable, falling back to a source build" >&2
  fi
fi

echo "flash-attn: compiling ${FA_VERSION} from source (this takes a while)"
: "${CUDA_HOME:?CUDA_HOME must point at a CUDA toolkit matching the torch CUDA ${CUDA_VER}}"

# Build from outside the project so uv does not apply the project's build settings, and target
# only this GPU's compute capability. MAX_JOBS is memory-bound rather than core-bound here: these
# kernels peak at several GB each, so an unbounded job count gets the build OOM-killed.
(
  cd /tmp && \
  MAX_JOBS="${MAX_JOBS:-16}" \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  TORCH_CUDA_ARCH_LIST="${CC}" \
  uv pip install --python "$VENV_PY" --no-build-isolation "flash_attn==${FA_VERSION}"
) || exit 1

"$VENV_PY" -c "import flash_attn, flash_attn_2_cuda; print('flash-attn: built', flash_attn.__version__)"
