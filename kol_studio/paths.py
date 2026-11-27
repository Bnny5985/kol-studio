"""Where Kol Studio keeps its files, on every operating system.

Override for development:  KOL_DATA_DIR (everything)  ·  KOL_MODELS_DIR (models only).
"""
import os
import sys
from pathlib import Path

APP_NAME = "KolStudio"


def _data_dir() -> Path:
    if os.environ.get("KOL_DATA_DIR"):
        return Path(os.environ["KOL_DATA_DIR"]).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME


DATA = _data_dir()
MODELS = Path(os.environ["KOL_MODELS_DIR"]).expanduser() if os.environ.get("KOL_MODELS_DIR") else DATA / "models"
CHARS = DATA / "characters"
CHARS_DELETED = DATA / "characters_deleted"
TMP = DATA / "tmp"
SETTINGS_FILE = DATA / "settings.json"
SERVER_FILE = DATA / "server.json"  # port of the running engine, read by the MCP server
OUTPUT_DEFAULT = Path.home() / "Documents" / APP_NAME

# index.html: next to this file in development, inside the bundle when frozen (PyInstaller)
RESOURCES = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
if (RESOURCES / "kol_studio").is_dir():
    RESOURCES = RESOURCES / "kol_studio"

for d in (DATA, MODELS, CHARS, TMP):
    d.mkdir(parents=True, exist_ok=True)
