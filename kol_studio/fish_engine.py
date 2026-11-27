"""Optional engine: Fish Audio S2-Pro with the Hebrew adapter (notmax123/Fish-Audio-S2-Pro-He).

Higher expressiveness (free-form emotion tags such as [laughing], [whisper]) at the cost of speed and size
(~13 GB, several times slower than real time on Apple M1). It runs in its own Python environment under
<data>/engines/fish so its PyTorch stack never touches the app; the app talks to a resident worker process
(fish_worker.py) over stdin/stdout.

License: Fish Audio Research License — research and non-commercial use only. "Built with Fish Audio."
"""
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from . import paths

DIR = paths.DATA / "engines" / "fish"
SRC = DIR / "src"
VENV = DIR / "venv"
REFS = DIR / "refs"
CKPT = SRC / "checkpoints"
FISH_COMMIT = "95672e3"  # maxmelichov/fish-speech, tested with the Hebrew adapter (step 2200)
LICENSE_URL = "https://github.com/maxmelichov/fish-speech/blob/main/LICENSE"
SIZE = "13GB"

# Only what generation needs (the fork's own list pulls in gradio, wandb, training tools…)
DEPS = ["torch==2.8.0", "torchaudio==2.8.0", "transformers<=4.57.3", "einops", "einx[torch]==0.2.2", "loguru",
        "hydra-core", "natsort", "loralib", "safetensors", "tiktoken", "ormsgpack", "pydantic==2.9.2", "librosa",
        "rich", "pyrootutils", "lightning", "cachetools", "zstandard", "soundfile", "click", "kui",
        "renikud-plus==0.5.0", "onnxruntime>=1.27,<1.30", "argbind", "julius", "randomname", "flatten-dict",
        "markdown2", "pyloudnorm", "ffmpy", "importlib-resources", "tensorboard"]
NO_DEPS = ["descript-audio-codec", "descript-audiotools"]  # their tensorboard/grpc chain is not needed

# A neutral ~9 s sentence used to give built-in and mixed voices a reference recording for Fish
REF_TEXT = "שָׁלוֹם, אֲנִי מְדַבֵּר בְּקֶצֶב רָגִיל וּבָרוּר. הַיּוֹם נִקְרָא יַחַד סִפּוּר קָצָר עַל יְלָדִים, עַל חַיּוֹת וְעַל הַרְבֵּה הַרְפַּתְקָאוֹת."


def python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def installed() -> bool:
    return (python().exists() and (CKPT / "s2-pro-he-ipa" / "config.json").exists()
            and (CKPT / "hebrew" / "hebrew_lora_step2200.safetensors").exists()
            and (CKPT / "s2-pro" / "codec.pth").exists())


def device() -> str:
    if sys.platform == "darwin":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


# ---------- installation ----------

def _uv(job: dict) -> str:
    """uv (MIT/Apache) builds the engine's Python environment; downloaded once if it is not on this computer."""
    found = shutil.which("uv") or shutil.which("uv", path=str(DIR / "bin"))
    if found:
        return found
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}[platform.machine()]
    target = {"darwin": f"{arch}-apple-darwin", "win32": f"{arch}-pc-windows-msvc"}.get(sys.platform,
                                                                                       f"{arch}-unknown-linux-gnu")
    ext = "zip" if sys.platform.startswith("win") else "tar.gz"
    url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.{ext}"
    job["current"] = "מוריד את uv"
    data = urllib.request.urlopen(url, timeout=120).read()
    (DIR / "bin").mkdir(parents=True, exist_ok=True)
    name = "uv.exe" if sys.platform.startswith("win") else "uv"
    if ext == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(m for m in z.namelist() if m.endswith(name))
            (DIR / "bin" / name).write_bytes(z.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("/" + name) or m.name == name)
            (DIR / "bin" / name).write_bytes(t.extractfile(member).read())
    (DIR / "bin" / name).chmod(0o755)
    return str(DIR / "bin" / name)


def _run(cmd: list, job: dict, what: str, cwd: Path | None = None):
    job["current"] = what
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{what} נכשל: {(r.stderr or r.stdout)[-600:]}")


def install(job: dict):
    """Fetch the code, build the environment, download the models (~11 GB) and build the Hebrew checkpoint.
    Every step skips what is already there, so an interrupted install can simply be started again."""
    from huggingface_hub import HfApi, hf_hub_download

    DIR.mkdir(parents=True, exist_ok=True)
    job.update(total=100, done=0, step="")

    def progress(step: str, frac: float, text: str):
        base = {"code": 0, "env": 3, "models": 15, "checkpoint": 97}[step]
        span = {"code": 3, "env": 12, "models": 82, "checkpoint": 3}[step]
        job.update(step=step, done=round(base + span * frac, 1), current=text)

    # 1. code (pinned commit of the Hebrew fork)
    if not (SRC / "fish_speech").is_dir():
        progress("code", 0, "מוריד את הקוד של Fish")
        data = urllib.request.urlopen(f"https://codeload.github.com/maxmelichov/fish-speech/zip/{FISH_COMMIT}",
                                      timeout=300).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(DIR / "_code")
        top = next((DIR / "_code").iterdir())
        if SRC.exists():
            shutil.rmtree(SRC / "fish_speech", ignore_errors=True)
        SRC.mkdir(exist_ok=True)
        for item in top.iterdir():
            shutil.move(str(item), SRC / item.name)
        shutil.rmtree(DIR / "_code", ignore_errors=True)

    # 2. Python environment (PyTorch for the Apple GPU, an NVIDIA card, or the CPU)
    if not python().exists() or not _has_torch():
        progress("env", 0, "מכין סביבת Python")
        uv = _uv(job)
        _run([uv, "venv", "--python", "3.12", VENV], job, "יצירת סביבה")
        progress("env", 0.2, "מתקין PyTorch וספריות (כ־2GB)")
        extra = ["--index-strategy", "unsafe-best-match", "--extra-index-url",
                 "https://download.pytorch.org/whl/cu128"] if device() == "cuda" else []
        _run([uv, "pip", "install", "--python", python(), *extra, *DEPS], job, "התקנת ספריות")
        _run([uv, "pip", "install", "--python", python(), "--no-deps", *NO_DEPS, "-e", SRC], job, "התקנת Fish")

    # 3. models: S2-Pro base (~11 GB) + Hebrew adapter
    api = HfApi()
    plan = [("fishaudio/s2-pro", f, s, CKPT / "s2-pro") for f, s in
            ((x.rfilename, x.size or 0) for x in api.model_info("fishaudio/s2-pro", files_metadata=True).siblings)
            if not f.endswith((".png", ".md")) or f == "LICENSE.md"]
    plan += [("notmax123/Fish-Audio-S2-Pro-He", f, s, CKPT / "hebrew") for f, s in
             ((x.rfilename, x.size or 0) for x in
              api.model_info("notmax123/Fish-Audio-S2-Pro-He", files_metadata=True).siblings)
             if f.endswith((".safetensors", ".json", ".pt")) or f.startswith("tokenizer/") or f in ("LICENSE", "NOTICE")]
    total, done = sum(p[2] for p in plan) or 1, 0
    for repo, fname, size, dest in plan:
        progress("models", done / total, f"מוריד מודלים · {Path(fname).name} · {done / 1e9:.1f}/{total / 1e9:.1f}GB")
        if not (dest / fname).exists():
            hf_hub_download(repo, fname, local_dir=str(dest))
        done += size

    # 4. the IPA-extended Hebrew checkpoint
    if not (CKPT / "s2-pro-he-ipa" / "config.json").exists():
        progress("checkpoint", 0, "בונה את הגרסה העברית של המודל")
        _run([python(), "tools/hebrew/build_ipa_checkpoint.py", "--base", "checkpoints/s2-pro", "--output",
              "checkpoints/s2-pro-he-ipa"], job, "בניית המודל העברי", cwd=SRC)
    (DIR / "NOTICE.txt").write_text(
        "This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.\n"
        "Built with Fish Audio.\nHebrew adapter: notmax123/Fish-Audio-S2-Pro-He.\n"
        f"License: {LICENSE_URL}\n", encoding="utf-8")
    progress("checkpoint", 1, "מוכן")


def _has_torch() -> bool:
    r = subprocess.run([str(python()), "-c", "import torch"], capture_output=True, check=False)
    return r.returncode == 0


def uninstall():
    stop()
    shutil.rmtree(DIR, ignore_errors=True)


# ---------- the resident worker ----------

_proc: subprocess.Popen | None = None
_lock = threading.Lock()
_state = {"status": "stopped", "error": "", "load_seconds": None}


def worker_script() -> Path:
    return paths.RESOURCES / "fish_worker.py"


def status() -> dict:
    alive = _proc is not None and _proc.poll() is None
    st = _state["status"] if alive or _state["status"] in ("error", "loading") else "stopped"
    return {"installed": installed(), "status": st,
            "error": _state["error"], "device": device(), "load_seconds": _state["load_seconds"], "size": SIZE,
            "license_url": LICENSE_URL}


def _start(renikud: Path):
    global _proc
    if not installed():
        raise RuntimeError("מנוע Fish לא מותקן — אפשר להתקין אותו בהגדרות")
    _state.update(status="loading", error="")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    log = open(DIR / "worker.log", "w")  # noqa: SIM115 - lives as long as the worker
    _proc = subprocess.Popen([str(python()), "-I", str(worker_script()), "--src", str(SRC), "--renikud", str(renikud),
                              "--device", device()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                             text=True, encoding="utf-8", bufsize=1, env=env)
    first = _proc.stdout.readline()
    try:
        ready = json.loads(first)
    except ValueError:
        _state.update(status="error", error=(DIR / "worker.log").read_text()[-800:])
        raise RuntimeError("מנוע Fish לא עלה. הפרטים נמצאים ב־worker.log") from None
    _state.update(status="ready", load_seconds=ready.get("load_seconds"))


def synth(text: str, out: Path, renikud: Path, ref_audio: Path | None = None, ref_text: str = "",
          seed: int | None = None) -> dict:
    """Generate one utterance into `out` (WAV). Starts the worker on first use (~1.5 min to load)."""
    with _lock:
        if _proc is None or _proc.poll() is not None:
            _start(renikud)
        req = {"text": text, "out": str(out), "ref_audio": str(ref_audio) if ref_audio else None,
               "ref_text": ref_text, "seed": seed}
        _state["status"] = "busy"
        _proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        _proc.stdin.flush()
        line = _proc.stdout.readline()
        _state["status"] = "ready"
    if not line:
        _state.update(status="error", error="the worker stopped")
        raise RuntimeError("מנוע Fish נעצר באמצע. נסה שוב")
    res = json.loads(line)
    if not res.get("ok"):
        raise RuntimeError(f"Fish: {res.get('error')}")
    return res


def stop():
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.stdin.close()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
    _state["status"] = "stopped"


def warm_up(renikud: Path):
    """Load the model in the background so the first line does not wait."""
    def go():
        try:
            with _lock:
                if _proc is None or _proc.poll() is not None:
                    _start(renikud)
        except Exception as e:  # shown in Settings
            _state.update(status="error", error=str(e))

    _state.update(status="loading", error="")
    threading.Thread(target=go, daemon=True).start()


def wait_ready(timeout: float = 300) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if _state["status"] in ("ready", "busy"):
            return True
        if _state["status"] == "error":
            return False
        time.sleep(1)
    return False
