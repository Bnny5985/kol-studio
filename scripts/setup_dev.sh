#!/usr/bin/env bash
# Development environment for Kol Studio (macOS / Linux). Needs uv: https://docs.astral.sh/uv/
set -euo pipefail
cd "$(dirname "$0")/.."
BLUE_REF="db047ea"   # maxmelichov/BlueTTS commit tested with the BlueV3 v3.0.1 bundle
uv venv --python 3.12 .venv
uv pip install --python .venv -e .
# BlueTTS without its dependencies: phonemizer / espeak-ng (GPL) are only for non-Hebrew languages.
uv pip install --python .venv --no-deps "blue-onnx @ git+https://github.com/maxmelichov/BlueTTS@${BLUE_REF}"
uv pip install --python .venv pyinstaller
# polyg2p (Apache-2.0): reads English words inside Hebrew text. Its wheels ship only in the BlueV3 bundle (macOS, Linux).
WHEELS=$(mktemp -d)
.venv/bin/python - "$WHEELS" <<'PY'
import platform, sys
from huggingface_hub import hf_hub_download
tag = {"Darwin-arm64": "macosx_11_0_arm64", "Darwin-x86_64": "macosx_10_12_x86_64",
       "Linux-x86_64": "manylinux_2_34_x86_64"}.get(f"{platform.system()}-{platform.machine()}")
if tag:
    hf_hub_download("notmax123/BlueV3-onnx", f"frontend/polyg2p-0.1.0-cp39-abi3-{tag}.whl", local_dir=sys.argv[1])
PY
if ls "$WHEELS"/frontend/*.whl >/dev/null 2>&1; then
  uv pip install --python .venv --no-index --find-links "$WHEELS/frontend" polyg2p==0.1.0
fi
rm -rf "$WHEELS"
echo "ready: .venv/bin/python -m kol_studio.desktop"
