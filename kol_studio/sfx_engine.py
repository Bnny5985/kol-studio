"""Optional sound-effect engines. Each runs in its own Python environment under <data>/engines/sfx-<id>
and is driven through a resident worker (sfx_worker.py).

- moss: MOSS-SoundEffect v2.0 (OpenMOSS, Apache-2.0). Open, 48 kHz, up to 30 s. Slow on Apple GPUs.
- sa3:  Stable Audio 3 Small-SFX (Stability AI Community License). Fast, 44.1 kHz stereo. Gated on
        Hugging Face: the user accepts the terms there and pastes their own access token in Settings.
"""
import json
import os
import shutil
from pathlib import Path

from . import engine_tools as et
from . import paths

AUDIOTOOLS_DEPS = ["argbind", "julius", "randomname", "flatten-dict", "markdown2", "pyloudnorm", "ffmpy",
                   "importlib-resources", "rich", "tqdm", "scipy", "librosa", "matplotlib", "tensorboard"]

ENGINES = {
    "moss": {
        "name": "MOSS-SoundEffect v2", "by": "OpenMOSS", "repo": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
        "code": ("OpenMOSS/MOSS-TTS", "934d6826b084c46a0d033402174d5f8ac4ed2519", "moss_soundeffect_v2"),
        "deps": ["torch==2.9.0", "torchaudio==2.9.0", "numpy==1.26.4", "einops==0.8.2", "safetensors==0.7.0",
                 "transformers==4.57.1", "diffusers==0.37.1", "ftfy", "regex", "soundfile==0.13.1", "pillow",
                 "imageio", "typing-extensions>=4.10", "huggingface_hub<1", *AUDIOTOOLS_DEPS],
        "no_deps": ["descript-audiotools==0.7.2"], "check": "moss_soundeffect_v2, audiotools.ml",
        "size": "11GB", "min_ram": 16, "max_seconds": 30, "sample_rate": 48000, "gated": False,
        "license": "Apache-2.0", "license_url": "https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0",
        "speed": "איטי: כדקה לצליל קצר במחשב Mac",
        "note": "פתוח לגמרי (Apache-2.0). 48kHz, עד 30 שניות. מצוין לרעשי רקע ואווירה",
    },
    "sa3": {
        "name": "Stable Audio 3 · SFX", "by": "Stability AI", "repo": "stabilityai/stable-audio-3-small-sfx",
        "code": ("Stability-AI/stable-audio-3", "4007b6f2987bcb760626c2b4ae3e3541753984e9", ""),
        "deps": ["torch==2.7.1", "torchaudio==2.7.1", "einops>=0.8.2", "einops-exts>=0.0.4", "numpy>=2.2.6",
                 "packaging", "safetensors>=0.7.0", "tqdm", "huggingface-hub>=1.7.1", "transformers>=5.8.0",
                 "soundfile>=0.13.1", "sentencepiece"],
        "no_deps": [], "check": "stable_audio_3.model",
        "size": "2GB", "min_ram": 8, "max_seconds": 47, "sample_rate": 44100, "gated": True,
        "license": "Stability AI Community License",
        "license_url": "https://huggingface.co/stabilityai/stable-audio-3-small-sfx",
        "speed": "מהיר: כמה שניות לצליל",
        "note": "מהיר מאוד, סטריאו 44.1kHz. רישיון קהילתי של Stability (חינם עד הכנסה של מיליון דולר בשנה)",
    },
}


def dir_of(eid: str) -> Path:
    return paths.DATA / "engines" / f"sfx-{eid}"


def python(eid: str) -> Path:
    return et.venv_python(dir_of(eid) / "venv")


def model_dir(eid: str) -> Path:
    return dir_of(eid) / "model"


def installed(eid: str) -> bool:
    return (python(eid).exists() and (dir_of(eid) / "src").is_dir()
            and (model_dir(eid) / ".complete").exists())


def install(job: dict, eid: str, token: str = ""):
    """Code, Python environment, model. Each step skips what is already there, so an interrupted
    install can simply be started again."""
    from huggingface_hub import HfApi, hf_hub_download

    e, d = ENGINES[eid], dir_of(eid)
    d.mkdir(parents=True, exist_ok=True)
    job.update(total=100, done=0)

    def progress(base: float, span: float, frac: float, text: str):
        job.update(done=round(base + span * frac, 1), current=text)

    if not (d / "src").is_dir():
        progress(0, 3, 0, f"מוריד את הקוד של {e['name']}")
        repo, commit, sub = e["code"]
        et.fetch_code(repo, commit, d / "src", job, f"מוריד את הקוד של {e['name']}", sub)

    py = python(eid)
    if not py.exists() or not et.has_module(py, e["check"]):
        progress(3, 12, 0, "מכין סביבת Python")
        uv = et.uv(job)
        if not py.exists():
            et.run([uv, "venv", "--python", "3.12", d / "venv"], job, "יצירת סביבה")
        progress(3, 12, 0.2, "מתקין PyTorch וספריות (כ־2GB)")
        extra = ["--index-strategy", "unsafe-best-match", "--extra-index-url",
                 "https://download.pytorch.org/whl/cu128"] if et.gpu() == "cuda" else []
        et.run([uv, "pip", "install", "--python", py, *extra, *e["deps"]], job, "התקנת ספריות")
        et.run([uv, "pip", "install", "--python", py, "--no-deps", *e["no_deps"], "-e", d / "src"], job,
               f"התקנת {e['name']}")

    if not (model_dir(eid) / ".complete").exists():
        api = HfApi(token=token or None)
        try:
            files = api.model_info(e["repo"], files_metadata=True).siblings
        except Exception as ex:
            if e["gated"]:
                raise RuntimeError("Hugging Face לא נתן גישה למודל. צריך לאשר את התנאים בדף המודל "
                                   "ולהדביק בהגדרות אסימון גישה (token) של החשבון שאישר") from ex
            raise
        plan = [(f.rfilename, f.size or 0) for f in files if not f.rfilename.endswith((".png", ".jpg", ".gitattributes"))]
        total, done = sum(s for _, s in plan) or 1, 0
        for fname, size in plan:
            progress(15, 85, done / total, f"מוריד את המודל · {Path(fname).name} · {done / 1e9:.1f}/{total / 1e9:.1f}GB")
            if not (model_dir(eid) / fname).exists():
                try:
                    hf_hub_download(e["repo"], fname, local_dir=str(model_dir(eid)), token=token or None)
                except Exception as ex:
                    if e["gated"] and ("401" in str(ex) or "403" in str(ex) or "gated" in str(ex).lower()):
                        raise RuntimeError("Hugging Face לא נתן גישה למודל. צריך לאשר את התנאים בדף המודל "
                                           "ולהדביק בהגדרות אסימון גישה (token) של החשבון שאישר") from ex
                    raise
            done += size
        (model_dir(eid) / ".complete").write_text(json.dumps({"repo": e["repo"]}))
    (d / "NOTICE.txt").write_text(f"{e['name']} by {e['by']}\nLicense: {e['license']}\n{e['license_url']}\n",
                                  encoding="utf-8")
    progress(0, 100, 1, "מוכן")


def uninstall(eid: str):
    stop(eid)
    shutil.rmtree(dir_of(eid), ignore_errors=True)


# ---------- workers ----------

_workers = {eid: et.Worker(ENGINES[eid]["name"], dir_of(eid) / "worker.log") for eid in ENGINES}


def worker_script() -> Path:
    return paths.RESOURCES / "sfx_worker.py"


def _cmd(eid: str) -> tuple[list, dict]:
    if not installed(eid):
        raise RuntimeError(f"{ENGINES[eid]['name']} לא מותקן — אפשר להתקין אותו בהגדרות")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TORCHDYNAMO_DISABLE": "1",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    return [python(eid), "-I", worker_script(), "--engine", eid, "--model", model_dir(eid),
            "--device", et.gpu()], env


def generate(eid: str, prompt: str, seconds: float, outs: list[Path], seed: int, quality: str = "normal",
             negative: str = "") -> dict:
    """Generate len(outs) variations of one prompt into WAV files."""
    cmd, env = _cmd(eid)
    req = {"prompt": prompt, "seconds": seconds, "outs": [str(o) for o in outs], "seed": seed,
           "quality": quality, "negative": negative}
    return _workers[eid].call(req, cmd, env)


def warm_up(eid: str):
    cmd, env = _cmd(eid)
    _workers[eid].warm_up(cmd, env)


def stop(eid: str):
    _workers[eid].stop()


def status() -> dict:
    out = {}
    for eid, e in ENGINES.items():
        w = _workers[eid]
        out[eid] = {k: e[k] for k in ("name", "by", "size", "min_ram", "max_seconds", "sample_rate", "gated",
                                      "license", "license_url", "speed", "note")}
        out[eid].update(installed=installed(eid), status=w.status(), error=w.state["error"],
                        load_seconds=w.state["load_seconds"])
    return out
