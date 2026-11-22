#!/usr/bin/env bash
# Run the Kol Studio server for development at http://127.0.0.1:8765 (no app window).
# KOL_MODELS_DIR can point at an existing models folder so nothing is downloaded again.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -c "from kol_studio.server import main; main(8765)"
