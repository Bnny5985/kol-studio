"""Settings, hardware scan, model catalogue, updates and the external-LLM client."""
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import paths

FILE = paths.SETTINGS_FILE  # stays on this computer; holds the API key in plain text

VERSION = "0.1.0"
UPDATE_REPO = ""  # "owner/name" of the GitHub repository; set when the repository exists

DEFAULTS = {
    "app": {"update_repo": UPDATE_REPO, "check_updates_on_start": True, "threads": 0, "output_dir": "",
            "part_chars": 1200, "reader_prefetch": 2, "default_voice": "noa"},
    "models": {"tts_precision": "int8", "nikud": "phonikud", "asr": "ivrit-turbo", "asr_device": "auto"},
    "llm": {"provider": "lmstudio", "base_url": "http://127.0.0.1:1234/v1", "api_key": "", "model": ""},
    "theme": {"mode": "auto", "accent": "#2f5bd3", "tp_bg": "#0e0f13", "tp_ink": "#8b8f9c", "tp_now": "#ffd25e",
              "font": "Assistant", "radius": 12, "text_size": 19},
}

PROVIDERS = {
    "lmstudio": {"name": "LM Studio (מקומי)", "base_url": "http://127.0.0.1:1234/v1", "key": False},
    "ollama": {"name": "Ollama (מקומי)", "base_url": "http://127.0.0.1:11434/v1", "key": False},
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "key": True},
    "anthropic": {"name": "Anthropic (Claude)", "base_url": "https://api.anthropic.com/v1", "key": True},
    "gemini": {"name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key": True},
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key": True},
    "custom": {"name": "שרת תואם OpenAI אחר", "base_url": "", "key": True},
}


def load() -> dict:
    data = json.loads(FILE.read_text()) if FILE.exists() else {}
    out = json.loads(json.dumps(DEFAULTS))
    for section, values in data.items():
        if section in out and isinstance(values, dict):
            out[section].update(values)
    return out


def save(data: dict):
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    try:
        os.chmod(FILE, 0o600)  # only this user can read the key
    except OSError:
        pass


def mask(key: str) -> str:
    return ("••••" + key[-4:]) if key else ""


def public(data: dict) -> dict:
    d = json.loads(json.dumps(data))
    d["llm"]["api_key"] = mask(d["llm"]["api_key"])
    return d


# ---------- hardware ----------

def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def scan() -> dict:
    info = {"os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}", "arch": platform.machine(),
            "cpu": platform.processor(), "cores": os.cpu_count(), "ram_gb": None, "gpu": "", "gpu_cores": None,
            "accelerators": [], "disk_free_gb": round(shutil.disk_usage(paths.DATA).free / 1e9)}
    if platform.system() == "Darwin":
        info["cpu"] = _sysctl("machdep.cpu.brand_string") or info["cpu"]
        info["ram_gb"] = round(int(_sysctl("hw.memsize") or 0) / 2**30)
        try:
            sp = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, timeout=10)
            g = json.loads(sp.stdout)["SPDisplaysDataType"][0]
            info["gpu"] = g.get("sppci_model", "")
            info["gpu_cores"] = int(g.get("sppci_cores", 0) or 0) or None
        except Exception:
            pass
    else:
        try:
            import psutil  # optional on other systems

            info["ram_gb"] = round(psutil.virtual_memory().total / 2**30)
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True)
            info["gpu"] = r.stdout.strip()
    try:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        local = {"CoreMLExecutionProvider": "CoreML", "CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML",
                 "ROCMExecutionProvider": "ROCm", "OpenVINOExecutionProvider": "OpenVINO"}
        info["accelerators"] += [local[p] for p in provs if p in local]  # Azure etc. are not local hardware
    except Exception:
        pass
    try:
        import ctranslate2

        if "float16" in ctranslate2.get_supported_compute_types("mps"):
            info["accelerators"].append("MPS (תמלול)")
    except Exception:
        pass
    ram = info["ram_gb"] or 8
    info["tier"] = "strong" if ram >= 32 else "normal" if ram >= 16 else "light"
    info["tier_name"] = {"strong": "חזק", "normal": "רגיל", "light": "קל"}[info["tier"]]
    return info


# ---------- model catalogue ----------

def catalogue(hw: dict, current: dict) -> list[dict]:
    m = paths.MODELS
    ram = hw.get("ram_gb") or 8
    items = [
        # category, id, name, size, min RAM, installed?, available?, note
        ("tts", "bluev3-int8", "BlueV3 · int8 (מהיר)", "0.6GB", 4, True, True,
         "עברית במקור. נמדד אצלך: פי 12 מזמן אמת", "tts_precision", "int8"),
        ("tts", "bluev3-fp32", "BlueV3 · fp32 (מדויק)", "0.8GB", 6, True, True,
         "אותו מודל בדיוק מלא. נמדד אצלך: פי 9.6 מזמן אמת", "tts_precision", "fp32"),
        ("tts", "fish-s2-he", "Fish-Audio S2 Pro · עברית", "~10GB", 24, False, False,
         "עברית במקור, תומך ברגשות מובנים. עוד לא חובר", None, None),
        ("tts", "piper-he", "Piper · נועה", "60MB", 2, False, False, "קל מאוד, למחשבים חלשים. עוד לא חובר", None, None),
        ("nikud", "phonikud", "Phonikud", "0.3GB", 2, (m / "phonikud-1.0.int8.onnx").exists(), True,
         "מבוסס Dicta. 20/20 בבדיקה, 52ms למשפט", "nikud", "phonikud"),
        ("nikud", "dicta", "Dicta (מודל מלא)", "1.2GB", 4, (m / "dicta-1.0.onnx").exists(), True,
         "אותה תוצאה בבדיקה, גדול ואיטי יותר", "nikud", "dicta"),
        ("asr", "ivrit-turbo", "ivrit.ai Whisper turbo", "1.5GB", 8, (m / "ivrit-turbo-ct2" / "model.bin").exists(), True,
         "מהיר. 2–6.5% שגיאות בבדיקה", "asr", "ivrit-turbo"),
        ("asr", "ivrit-large", "ivrit.ai Whisper large-v3", "3.1GB", 16, (m / "ivrit-large-ct2" / "model.bin").exists(), True,
         "הכי מדויק, איטי פי 2 בערך", "asr", "ivrit-large"),
        ("llm", "dictalm3-24b", "DictaLM 3.0 · 24B", "~14GB", 32, None, True,
         "עברית במקור. מריצים ב־LM Studio", None, None),
        ("llm", "dictalm2-7b", "DictaLM 2.0 · 7B", "~4.5GB", 12, None, True, "עברית במקור, קל יותר. ב־LM Studio", None, None),
    ]
    out = []
    for cat, mid, name, size, min_ram, installed, available, note, key, value in items:
        fits = ram >= min_ram
        out.append({"category": cat, "id": mid, "name": name, "size": size, "min_ram": min_ram, "fits": fits,
                    "installed": installed, "available": available, "note": note, "setting": key, "value": value,
                    "active": key is not None and current["models"].get(key) == value})
    # recommendation per category: the best model that fits this computer
    best = {"tts": "bluev3-int8", "nikud": "phonikud",
            "asr": "ivrit-large" if ram >= 32 else "ivrit-turbo",
            "llm": "dictalm3-24b" if ram >= 32 else "dictalm2-7b"}
    for it in out:
        it["recommended"] = best.get(it["category"]) == it["id"]
    return out


DOWNLOADS = {"ivrit-large": ("ivrit-ai/whisper-large-v3-ct2", "ivrit-large-ct2"),
             "ivrit-turbo": ("ivrit-ai/whisper-large-v3-turbo-ct2", "ivrit-turbo-ct2")}


def asr_path(model_id: str) -> Path:
    return paths.MODELS / DOWNLOADS.get(model_id, DOWNLOADS["ivrit-turbo"])[1]


# ---------- external LLM ----------

def _request(url: str, headers: dict, body: dict | None = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"השרת החזיר שגיאה {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"אין חיבור לשרת ({e.reason}). האם הוא פועל?")


def _headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"} if key else {}


def list_models(provider: str, base_url: str, key: str) -> list[str]:
    data = _request(base_url.rstrip("/") + "/models", _headers(provider, key), timeout=15)
    return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))


def chat(provider: str, base_url: str, key: str, model: str, prompt: str, max_tokens: int = 200) -> str:
    base = base_url.rstrip("/")
    if provider == "anthropic":
        data = _request(base + "/messages", _headers(provider, key),
                        {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in data.get("content", []))
    data = _request(base + "/chat/completions", _headers(provider, key),
                    {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]},
                    timeout=120)
    return data["choices"][0]["message"]["content"]


def detect_local() -> list[dict]:
    found = []
    for pid in ("lmstudio", "ollama"):
        try:
            models = list_models(pid, PROVIDERS[pid]["base_url"], "")
            found.append({"provider": pid, "name": PROVIDERS[pid]["name"], "models": models})
        except Exception:
            pass
    return found


# ---------- updates (GitHub releases) ----------

def _ver(v: str) -> tuple:
    nums = [int(x) for x in __import__("re").findall(r"\d+", v)[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


def check_update(repo: str, current: str) -> dict:
    """repo = "owner/name". Reads the latest GitHub release; nothing is downloaded or installed."""
    if not repo or "/" not in repo:
        return {"status": "no-repo", "message": "עוד לא הוגדר מאגר גיטהב לעדכונים"}
    try:
        data = _request(f"https://api.github.com/repos/{repo}/releases/latest",
                        {"Accept": "application/vnd.github+json", "User-Agent": "kol-studio-updater"}, timeout=15)
    except RuntimeError as e:
        if "404" in str(e):
            return {"status": "none", "message": "עדיין אין גרסאות שפורסמו במאגר"}
        raise
    latest = data.get("tag_name", "")
    assets = [{"name": a["name"], "url": a["browser_download_url"], "size_mb": round(a.get("size", 0) / 1e6)}
              for a in data.get("assets", [])]
    newer = _ver(latest) > _ver(current)
    return {"status": "update" if newer else "latest", "current": current, "latest": latest,
            "url": data.get("html_url"), "notes": (data.get("body") or "")[:1500], "assets": assets,
            "message": f"גרסה חדשה זמינה: {latest}" if newer else f"יש לך את הגרסה העדכנית ({current})"}


# ---------- search Hugging Face for models that suit this computer ----------

HEBREW_ORGS = {"dicta-il", "ivrit-ai", "notmax123", "Phonikud", "thewh1teagle", "imvladikon", "avichr", "onlplab", "HeNLP"}
TASKS = {"text-to-speech": "הקראה", "automatic-speech-recognition": "תמלול", "text-generation": "מודל שפה",
         "token-classification": "ניקוד / ניתוח"}


def search_models(task: str, ram_gb: int, query: str = "") -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    seen, out = set(), []
    searches = [dict(author=o) for o in ("dicta-il", "ivrit-ai", "notmax123", "Phonikud", "thewh1teagle")]
    searches += [dict(language="he"), dict(search="hebrew")]
    if query:
        searches = [dict(search=query)] + searches
    for kw in searches:
        try:
            found = api.list_models(pipeline_tag=task or None, sort="downloads", limit=40,
                                    expand=["downloads", "lastModified", "safetensors", "tags", "pipeline_tag"], **kw)
            for m in found:
                if m.id in seen:
                    continue
                seen.add(m.id)
                params = getattr(m, "safetensors", None)
                total = getattr(params, "total", None) if params else None
                size_gb = round(total * 2 / 1e9, 1) if total else None  # ~2 bytes/param at 16-bit
                tags = set(m.tags or [])
                import re as _re

                native = m.id.split("/")[0] in HEBREW_ORGS or bool(
                    _re.search(r"hebrew|ivrit|heb\b|-he\b|_he\b|dicta", m.id, _re.I))
                gguf = "gguf" in tags
                fits = None if size_gb is None else size_gb * (0.35 if gguf else 1.2) <= ram_gb * 0.6
                out.append({"id": m.id, "task": TASKS.get(m.pipeline_tag or "", m.pipeline_tag or ""),
                            "downloads": m.downloads or 0, "updated": str(m.last_modified or "")[:10],
                            "size_gb": size_gb, "fits": fits, "native": native, "gguf": gguf,
                            "url": f"https://huggingface.co/{m.id}"})
        except Exception:
            continue
    out.sort(key=lambda x: (x["id"].split("/")[0] not in HEBREW_ORGS, not x["native"], x["fits"] is False, -x["downloads"]))
    return out[:40]


def timed(fn, *a, **kw):
    t0 = time.time()
    return fn(*a, **kw), round(time.time() - t0, 2)
