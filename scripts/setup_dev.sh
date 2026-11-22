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
echo "ready: .venv/bin/python -m kol_studio.desktop"
