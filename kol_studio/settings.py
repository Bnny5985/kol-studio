"""Settings file, hardware scan, model catalogue, update check, model search, folder picker and LLM client."""
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import fish_engine, models_setup, paths, qwen_engine, sfx_engine

FILE = paths.SETTINGS_FILE  # stays on this computer; holds the API key in plain text

VERSION = "0.4.0"
UPDATE_REPO = "Bnny5985/kol-studio"  # GitHub repository whose releases the app checks for updates

DEFAULTS = {
    "app": {"update_repo": UPDATE_REPO, "check_updates_on_start": True, "threads": 0, "output_dir": "",
            "part_chars": 1200, "reader_prefetch": 2},
    "models": {"tts_precision": "int8", "asr": "ivrit-turbo", "asr_device": "auto", "sfx_engine": "", "hf_token": ""},
    "llm": {"provider": "lmstudio", "base_url": "http://127.0.0.1:1234/v1", "api_key": "", "model": ""},
    "theme": {"design": 2, "mode": "dark", "accent": "#ffb547", "accent2": "#ff7a2f", "tp_bg": "#070b1f", "tp_ink": "#8e9ac8",
              "tp_now": "#ffc35c", "font": "rubik", "radius": 20, "text_size": 19},
    # measured while the app runs, so estimates fit this voice model on this computer
    "stats": {"chars_per_sec": 12.0, "sentence_sec": 4.0, "samples": 0},
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
    if data.get("theme", {}).get("design") != DEFAULTS["theme"]["design"]:
        data.pop("theme", None)  # a new look: colours saved for the previous design are dropped once
    for section, values in data.items():
        if section in out and isinstance(values, dict):
            out[section].update({k: v for k, v in values.items() if k in out[section]})  # drop retired keys
    return out


def save(data: dict):
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    with contextlib.suppress(OSError):
        os.chmod(FILE, 0o600)  # only this user can read the key


def mask(key: str) -> str:
    return ("••••" + key[-4:]) if key else ""


def public(data: dict) -> dict:
    d = json.loads(json.dumps(data))
    d["llm"]["api_key"] = mask(d["llm"]["api_key"])
    d["models"]["hf_token"] = mask(d["models"].get("hf_token", ""))
    return d


# ---------- hardware ----------

def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def scan() -> dict:
    info = {"os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}", "arch": platform.machine(),
            "cpu": platform.processor(), "cores": os.cpu_count() or 4, "ram_gb": None, "gpu": "", "gpu_cores": None,
            "accelerators": [], "disk_free_gb": round(shutil.disk_usage(paths.DATA).free / 1e9)}
    if sys.platform == "darwin":
        info["os"] = f"macOS {platform.mac_ver()[0]}"
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or info["cpu"]
        info["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30)
        with contextlib.suppress(Exception):
            g = json.loads(_run(["system_profiler", "SPDisplaysDataType", "-json"]))["SPDisplaysDataType"][0]
            info["gpu"] = g.get("sppci_model", "")
            info["gpu_cores"] = int(g.get("sppci_cores", 0) or 0) or None
    else:
        with contextlib.suppress(Exception):
            import psutil

            info["ram_gb"] = round(psutil.virtual_memory().total / 2**30)
        if shutil.which("nvidia-smi"):
            info["gpu"] = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    with contextlib.suppress(Exception):
        import onnxruntime as ort

        local = {"CoreMLExecutionProvider": "CoreML", "CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML",
                 "ROCMExecutionProvider": "ROCm", "OpenVINOExecutionProvider": "OpenVINO"}
        info["accelerators"] += [local[p] for p in ort.get_available_providers() if p in local]  # not Azure etc.
    with contextlib.suppress(Exception):
        import ctranslate2

        if sys.platform == "darwin" and "float16" in ctranslate2.get_supported_compute_types("mps"):
            info["accelerators"].append("MPS")
        elif ctranslate2.get_cuda_device_count() > 0:
            info["accelerators"].append("CUDA (תמלול)")
    ram = info["ram_gb"] or 8
    info["tier"] = "strong" if ram >= 32 else "normal" if ram >= 16 else "light"
    info["tier_name"] = {"strong": "חזק", "normal": "רגיל", "light": "קל"}[info["tier"]]
    info["default_threads"] = min(8, info["cores"])  # what the speech engine uses when set to "automatic"
    return info


# ---------- model catalogue ----------

def catalogue(hw: dict, current: dict) -> list[dict]:
    ram = hw.get("ram_gb") or 8
    m = current["models"]
    rows = [
        # category, id, name, size, min RAM, pack (None = nothing to download here), setting, value, note
        ("tts", "bluev3-int8", "BlueV3 · int8 (מהיר)", "0.6GB", 4, "speech", "tts_precision", "int8",
         "עברית במקור. הכי מהיר במעבד רגיל"),
        ("tts", "bluev3-fp32", "BlueV3 · fp32 (מדויק)", "0.8GB", 6, "speech", "tts_precision", "fp32",
         "אותו מודל בדיוק מלא, איטי בכ־20%"),
        ("tts", "fish-s2-he", "Fish-Audio S2 Pro · עברית", fish_engine.SIZE, 24, None, None, None,
         "רגשות ותגיות מובנים, שכפול מדויק. איטי (כ־5× מזמן אמת ב־M1). רישיון: לשימוש לא מסחרי בלבד"),
        ("tts", "qwen-he", "QwenTTS-he · 1.7B", qwen_engine.SIZE, 16, None, None, None,
         "הגייה עברית מדויקת מאוד ושכפול קול, בלי רגשות. איטי (כ־5× מזמן אמת ב־M1). רישיון Apache-2.0"),
        ("nikud", "phonikud", "Phonikud", "0.3GB", 2, "nikud", None, None, "מבוסס Dicta. 20/20 בבדיקה, 52ms למשפט"),
        ("asr", "ivrit-turbo", "ivrit.ai Whisper turbo", "1.6GB", 8, "asr", "asr", "ivrit-turbo",
         "מהיר. 2–6.5% שגיאות בבדיקה"),
        ("asr", "ivrit-large", "ivrit.ai Whisper large-v3", "3.1GB", 16, "asr-large", "asr", "ivrit-large",
         "הכי מדויק, איטי בערך פי 2"),
        ("llm", "dictalm3-24b", "DictaLM 3.0 · 24B", "~14GB", 32, None, None, None, "עברית במקור. מריצים ב־LM Studio"),
        ("llm", "dictalm2-7b", "DictaLM 2.0 · 7B", "~4.5GB", 12, None, None, None, "עברית במקור, קל יותר. ב־LM Studio"),
    ]
    best = {"tts": "bluev3-int8", "nikud": "phonikud", "asr": "ivrit-large" if ram >= 32 else "ivrit-turbo",
            "llm": "dictalm3-24b" if ram >= 32 else "dictalm2-7b"}
    for eid, e in sfx_engine.ENGINES.items():
        rows.append(("sfx", f"sfx-{eid}", f"{e['name']} · {e['by']}", e["size"], e["min_ram"], None, None, None,
                     f"{e['note']}. {e['speed']}"))
    best["sfx"] = "sfx-sa3"
    out = []
    for cat, mid, name, size, min_ram, pack, key, value, note in rows:
        out.append({"category": cat, "id": mid, "name": name, "size": size, "min_ram": min_ram, "fits": ram >= min_ram,
                    "pack": pack, "installed": (models_setup.installed(pack) if pack else
                                                fish_engine.installed() if mid == "fish-s2-he" else
                                                qwen_engine.installed() if mid == "qwen-he" else
                                                sfx_engine.installed(mid[4:]) if cat == "sfx" else None),
                    "engine": {"fish-s2-he": "fish", "qwen-he": "qwen"}.get(mid, mid if cat == "sfx" else None),
                    "available": pack is not None or cat in ("llm", "sfx") or mid in ("fish-s2-he", "qwen-he"), "setting": key, "value": value, "note": note,
                    "active": key is not None and m.get(key) == value, "recommended": best[cat] == mid})
    return out


# ---------- updates (GitHub releases) ----------

def _ver(v: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", v)[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


def check_update(repo: str, current: str) -> dict:
    """repo = "owner/name". Reads the latest GitHub release; nothing is downloaded or installed."""
    if not repo or "/" not in repo:
        return {"status": "no-repo", "message": "עוד לא הוגדר מאגר עדכונים"}
    try:
        data = _request(f"https://api.github.com/repos/{repo}/releases/latest",
                        {"Accept": "application/vnd.github+json", "User-Agent": "kol-studio-updater"}, timeout=15)
    except RuntimeError as e:
        if "404" in str(e):
            return {"status": "none", "message": "עדיין לא פורסמו גרסאות"}
        raise
    latest = data.get("tag_name", "")
    newer = _ver(latest) > _ver(current)
    return {"status": "update" if newer else "latest", "current": current, "latest": latest,
            "url": data.get("html_url"), "notes": (data.get("body") or "")[:1500],
            "assets": [{"name": a["name"], "url": a["browser_download_url"], "size_mb": round(a.get("size", 0) / 1e6)}
                       for a in data.get("assets", [])],
            "message": f"גרסה חדשה זמינה: {latest}" if newer else "יש לך את הגרסה העדכנית"}


# ---------- search Hugging Face for models that suit this computer ----------

HEBREW_ORGS = {"dicta-il", "ivrit-ai", "notmax123", "Phonikud", "thewh1teagle", "imvladikon", "avichr", "onlplab", "HeNLP"}
HEBREW_NAME = re.compile(r"hebrew|ivrit|heb\b|-he\b|_he\b|dicta", re.I)
TASKS = {"text-to-speech": "הקראה", "automatic-speech-recognition": "תמלול", "text-generation": "מודל שפה"}


def search_models(task: str, ram_gb: int, query: str = "") -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    searches = [{"search": query}] if query else []
    searches += [{"author": o} for o in ("dicta-il", "ivrit-ai", "notmax123", "Phonikud", "thewh1teagle")]
    searches += [{"language": "he"}, {"search": "hebrew"}]
    seen, out = set(), []
    for kw in searches:
        try:
            found = list(api.list_models(pipeline_tag=task or None, sort="downloads", limit=40,
                                         expand=["downloads", "lastModified", "safetensors", "tags", "pipeline_tag"], **kw))
        except Exception:  # one failed query should not hide the others
            continue
        for mdl in found:
            if mdl.id in seen:
                continue
            seen.add(mdl.id)
            total = getattr(getattr(mdl, "safetensors", None), "total", None)
            size_gb = round(total * 2 / 1e9, 1) if total else None  # ~2 bytes per parameter at 16-bit
            gguf = "gguf" in set(mdl.tags or [])
            out.append({"id": mdl.id, "task": TASKS.get(mdl.pipeline_tag or "", mdl.pipeline_tag or ""),
                        "downloads": mdl.downloads or 0, "updated": str(mdl.last_modified or "")[:10],
                        "size_gb": size_gb, "gguf": gguf, "url": f"https://huggingface.co/{mdl.id}",
                        "native": mdl.id.split("/")[0] in HEBREW_ORGS or bool(HEBREW_NAME.search(mdl.id)),
                        "fits": None if size_gb is None else size_gb * (0.35 if gguf else 1.2) <= ram_gb * 0.6})
    out.sort(key=lambda x: (x["id"].split("/")[0] not in HEBREW_ORGS, not x["native"], x["fits"] is False, -x["downloads"]))
    return out[:40]


# ---------- native folder picker (the server runs on the user's own computer) ----------

def pick_folder(start: str = "") -> str:
    """Open the system's folder dialog. Returns "" when cancelled or unsupported."""
    prompt = "בחר תיקייה לשמירת הקבצים"
    if sys.platform == "darwin":
        script = f'POSIX path of (choose folder with prompt "{prompt}"'
        if start and os.path.isdir(start):
            script += f' default location (POSIX file "{start}")'
        return _run(["osascript", "-e", script + ")"], timeout=600).rstrip("/")
    if sys.platform.startswith("win"):
        ps = ("Add-Type -AssemblyName System.Windows.Forms;"
              "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
              f"$d.Description='{prompt}';$d.UseDescriptionForTitle=$true;"
              "if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}")
        return _run(["powershell", "-NoProfile", "-STA", "-Command", ps], timeout=600)
    if shutil.which("zenity"):
        return _run(["zenity", "--file-selection", "--directory", f"--title={prompt}"], timeout=600)
    return ""


# ---------- external LLM ----------

def _request(url: str, headers: dict, body: dict | None = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"השרת החזיר שגיאה {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"אין חיבור לשרת ({e.reason}). האם הוא פועל?") from None


def _headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"} if key else {}


def list_models(provider: str, base_url: str, key: str) -> list[str]:
    data = _request(base_url.rstrip("/") + "/models", _headers(provider, key), timeout=15)
    return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))


def chat(provider: str, base_url: str, key: str, model: str, prompt: str, *, max_tokens: int = 200) -> str:
    base = base_url.rstrip("/")
    msg = [{"role": "user", "content": prompt}]
    if provider == "anthropic":
        data = _request(base + "/messages", _headers(provider, key), {"model": model, "max_tokens": max_tokens, "messages": msg})
        return "".join(b.get("text", "") for b in data.get("content", []))
    data = _request(base + "/chat/completions", _headers(provider, key),
                    {"model": model, "max_tokens": max_tokens, "messages": msg}, timeout=120)
    return data["choices"][0]["message"]["content"]


def detect_local() -> list[dict]:
    found = []
    for pid in ("lmstudio", "ollama"):
        with contextlib.suppress(RuntimeError):
            found.append({"provider": pid, "name": PROVIDERS[pid]["name"],
                          "models": list_models(pid, PROVIDERS[pid]["base_url"], "")})
    return found


def timed(fn, *a, **kw):
    t0 = time.time()
    return fn(*a, **kw), round(time.time() - t0, 2)
