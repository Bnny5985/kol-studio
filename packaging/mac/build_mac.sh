#!/usr/bin/env bash
# Build KolStudio.app and a drag-to-install DMG (Apple Silicon).
#   ./packaging/mac/build_mac.sh            -> dist/KolStudio-<version>-mac-arm64.dmg
# Without an Apple Developer ID the app is ad-hoc signed: on first open, right-click the app -> Open.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
ARCH=$(uname -m)

[ -x .venv/bin/python ] || ./scripts/setup_dev.sh
.venv/bin/python scripts/make_icon.py
rm -rf build dist
.venv/bin/pyinstaller --noconfirm --log-level WARN packaging/kol_studio.spec

# Ad-hoc signature (required to run on Apple Silicon). Replace "-" with a Developer ID to sign for real.
codesign --force --deep --sign - dist/KolStudio.app

STAGE=$(mktemp -d)
cp -R dist/KolStudio.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="dist/KolStudio-${VERSION}-mac-${ARCH}.dmg"
hdiutil create -volname "KolStudio ${VERSION}" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
(cd dist && shasum -a 256 "$(basename "$DMG")" | tee "$(basename "$DMG").sha256")
echo "built: $DMG"
