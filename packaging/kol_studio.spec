# PyInstaller spec for Kol Studio (macOS .app and Windows folder build).
#   macOS:   pyinstaller packaging/kol_studio.spec   -> dist/KolStudio.app
#   Windows: pyinstaller packaging/kol_studio.spec   -> dist/KolStudio/KolStudio.exe
# Models are NOT bundled: the app downloads them on first run.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
VERSION = "0.1.0"

datas, binaries, hiddenimports = [], [], []
for pkg in [
    "kol_studio", "blue_onnx", "renikud_onnx", "phonikud_onnx", "faster_whisper", "ctranslate2", "onnxruntime",
    "heb_tts_normalizer", "hebrew_num2words", "num2words", "librosa", "soxr", "lazy_loader", "docx", "pptx", "pypdf",
    "av", "soundfile", "tokenizers", "uvicorn", "fastapi", "python_multipart", "multipart",
    "webview", "huggingface_hub",
]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
datas += [(str(ROOT / "kol_studio" / "index.html"), "kol_studio")]

# GPL components used by BlueTTS only for non-Hebrew languages are left out on purpose.
excludes = ["phonemizer", "espeakng_loader", "polyg2p", "yiddish_labels", "torch", "tensorflow", "matplotlib",
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
coll = COLLECT(exe, a.binaries, a.datas, name="KolStudio")

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
