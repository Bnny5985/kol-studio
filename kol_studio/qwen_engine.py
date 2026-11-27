"""Optional voice engine: QwenTTS-he — Qwen3-TTS 1.7B with the Hebrew adapter by notmax123 (Apache-2.0).

Very clear Hebrew and voice cloning from a reference recording; no emotion control. About 5x slower than real time
on an Apple M1 Max. Runs in its own Python environment under <data>/engines/qwen through a resident worker.
"""
import os
import shutil
from pathlib import Path

from . import engine_tools as et
from . import paths

DIR = paths.DATA / "engines" / "qwen"
VENV = DIR / "venv"
MODELS = DIR / "models"
BASE_REPO, BASE_REV = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"
ADAPTER_REPO = "notmax123/QwenTTS-he-1.7B"
SIZE = "4.5GB"
LICENSE = "Apache-2.0"
LICENSE_URL = "https://huggingface.co/notmax123/QwenTTS-he-1.7B"
DEPS = ["qwen-tts==0.1.1", "transformers==4.57.3", "peft==0.21.2", "accelerate==1.12.0", "huggingface_hub<1",
        "soundfile", "torch==2.9.1", "torchaudio==2.9.1", "renikud-plus==0.5.0", "onnxruntime>=1.27,<1.30"]


def python() -> Path:
    return et.venv_python(VENV)


def installed() -> bool:
    return python().exists() and (MODELS / "base" / ".complete").exists() and (MODELS / "adapter" / ".complete").exists()


def install(job: dict):
    from huggingface_hub import HfApi, hf_hub_download

    DIR.mkdir(parents=True, exist_ok=True)
    job.update(total=100, done=0)
    py = python()
    if not py.exists() or not et.has_module(py, "qwen_tts, peft, renikud_onnx"):
        job.update(done=1, current="מכין סביבת Python")
        uv = et.uv(job)
        if not py.exists():
            et.run([uv, "venv", "--python", "3.12", VENV], job, "יצירת סביבה")
        job.update(done=3, current="מתקין PyTorch וספריות (כ־2GB)")
        extra = ["--index-strategy", "unsafe-best-match", "--extra-index-url",
                 "https://download.pytorch.org/whl/cu128"] if et.gpu() == "cuda" else []
        et.run([uv, "pip", "install", "--python", py, *extra, *DEPS], job, "התקנת ספריות")
    api = HfApi()
    plan = []
    for repo, rev, sub in ((BASE_REPO, BASE_REV, "base"), (ADAPTER_REPO, None, "adapter")):
        if (MODELS / sub / ".complete").exists():
            continue
        info = api.model_info(repo, revision=rev, files_metadata=True)
        plan += [(repo, rev, sub, f.rfilename, f.size or 0) for f in info.siblings
                 if not f.rfilename.endswith((".png", ".jpg", ".gitattributes")) and not f.rfilename.startswith("samples/")]
    total, done = sum(p[4] for p in plan) or 1, 0
    for repo, rev, sub, fname, size in plan:
        job.update(done=round(15 + 84 * done / total, 1),
                   current=f"מוריד את המודל · {Path(fname).name} · {done / 1e9:.1f}/{total / 1e9:.1f}GB")
        if not (MODELS / sub / fname).exists():
            hf_hub_download(repo, fname, revision=rev, local_dir=str(MODELS / sub))
        done += size
    for sub in ("base", "adapter"):
        (MODELS / sub).mkdir(parents=True, exist_ok=True)
        (MODELS / sub / ".complete").write_text("ok")
    (DIR / "NOTICE.txt").write_text(f"{BASE_REPO} (Apache-2.0, Qwen team) + {ADAPTER_REPO} (Apache-2.0, notmax123)\n",
                                    encoding="utf-8")
    job.update(done=100, current="מוכן")


def uninstall():
    stop()
    shutil.rmtree(DIR, ignore_errors=True)


_worker = et.Worker("QwenTTS-he", DIR / "worker.log")


def _cmd(renikud: Path) -> tuple[list, dict]:
    if not installed():
        raise RuntimeError("מנוע QwenTTS-he לא מותקן — אפשר להתקין אותו בהגדרות ← מחשב ומודלים")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    return [python(), "-I", paths.RESOURCES / "qwen_worker.py", "--base", MODELS / "base", "--adapter",
            MODELS / "adapter", "--renikud", renikud, "--device", et.gpu()], env


def synth(text: str, out: Path, renikud: Path, ref_audio: Path) -> dict:
    cmd, env = _cmd(renikud)
    return _worker.call({"text": text, "out": str(out), "ref_audio": str(ref_audio)}, cmd, env)


def warm_up(renikud: Path):
    cmd, env = _cmd(renikud)
    _worker.warm_up(cmd, env)


def stop():
    _worker.stop()


def status() -> dict:
    return {"installed": installed(), "status": _worker.status(), "error": _worker.state["error"],
            "load_seconds": _worker.state["load_seconds"], "size": SIZE, "license": LICENSE,
            "license_url": LICENSE_URL, "device": et.gpu()}
