# PyInstaller spec for Kol Studio (macOS .app and Windows folder build).
#   macOS:   pyinstaller packaging/kol_studio.spec   -> dist/KolStudio.app
#   Windows: pyinstaller packaging/kol_studio.spec   -> dist/KolStudio/KolStudio.exe
# Models are NOT bundled: the app downloads them on first run.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
VERSION = "0.2.0"

datas, binaries, hiddenimports = [], [], []
for pkg in [
    "kol_studio", "blue_onnx", "renikud_onnx", "phonikud_onnx", "faster_whisper", "ctranslate2", "onnxruntime",
    "heb_tts_normalizer", "hebrew_num2words", "num2words", "librosa", "soxr", "lazy_loader", "docx", "pptx", "pypdf",
    "av", "soundfile", "tokenizers", "uvicorn", "fastapi", "python_multipart", "multipart",
    "webview", "huggingface_hub", "mcp", "mcp_types", "polyg2p",
]:
    try:
        d, b, h = collect_all(pkg, filter_submodules=lambda n: not n.startswith("mcp.cli"), on_error="ignore")
    except Exception:  # optional packages, e.g. polyg2p has no Windows build
        print(f"skip {pkg}: not installed")
        continue
    datas += d
    binaries += b
    hiddenimports += h
datas += [(str(ROOT / "kol_studio" / "index.html"), "kol_studio")]

# GPL components used by BlueTTS only for non-Hebrew languages are left out on purpose.
excludes = ["phonemizer", "espeakng_loader", "yiddish_labels", "typer", "mcp.cli", "torch", "tensorflow", "matplotlib",
            "IPython", "jupyter", "tkinter", "pytest"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
                                   "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
icon_mac = ROOT / "packaging" / "mac" / "icon.icns"
icon_win = ROOT / "packaging" / "windows" / "icon.ico"

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="KolStudio",
    console=False,
    icon=str(icon_win) if sys.platform.startswith("win") and icon_win.exists() else None,
    target_arch=None,
)
# The MCP server for AI agents: a small console program next to the app (stdin/stdout transport).
a_mcp = Analysis([str(ROOT / "packaging" / "mcp_launcher.py")], pathex=[str(ROOT)],
                 hiddenimports=collect_all("mcp", filter_submodules=lambda n: not n.startswith("mcp.cli"),
                                           on_error="ignore")[2] + collect_all("mcp_types", on_error="ignore")[2],
                 excludes=excludes + ["typer", "mcp.cli"], noarchive=False)
pyz_mcp = PYZ(a_mcp.pure)
exe_mcp = EXE(pyz_mcp, a_mcp.scripts, [], exclude_binaries=True, name="kolstudio-mcp", console=True)

coll = COLLECT(exe, a.binaries, a.datas, exe_mcp, a_mcp.binaries, a_mcp.datas, name="KolStudio")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="KolStudio.app",
        icon=str(icon_mac) if icon_mac.exists() else None,
        bundle_identifier="io.github.kolstudio",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "קול־סטודיו",
            "CFBundleName": "KolStudio",
            "CFBundleShortVersionString": VERSION,
            "NSMicrophoneUsageDescription": "קול־סטודיו משתמש במיקרופון כדי להקליט את הקול שלך לשכפול.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "LSApplicationCategoryType": "public.app-category.productivity",
        },
    )
