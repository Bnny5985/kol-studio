"""Kol Studio server: Hebrew nikud -> speech (BlueV3), voice characters, stories, documents, reader, transcription.

Started by kol_studio.desktop (the app) or `python -m kol_studio.server` (development).
Everything stays on this computer (binds to 127.0.0.1 only).
"""
import contextlib
import difflib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import av
import numpy as np
import soundfile as sf
import uvicorn
from blue_onnx import BlueTTS, load_voice_style
from blue_onnx.style import export_voice_style, style_from_payload
from blue_onnx.text_norm import prepare_text_for_synthesis
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from phonikud_onnx import Phonikud
from renikud_onnx import G2P
from scipy.signal import resample_poly

from . import docs, fish_engine, latin, models_setup, paths, story
from . import settings as cfg

BUNDLE = paths.MODELS / "bluev3"
CHARS = paths.CHARS

PRESETS = {  # BlueV3 built-in voices (CC BY 4.0, LibriTTS-R speakers)
    "maya": "מאיה", "noa": "נועה", "lily": "לילי", "tamar": "תמר", "shira": "שירה",
    "yael": "יעל", "emma": "אמה", "sara": "שרה", "adam": "אדם", "daniel": "דניאל",
    "david": "דוד", "jonathan": "יונתן", "ethan": "איתן", "michael": "מיכאל", "ariel": "אריאל",
}
DEFAULT_CAST = ["noa", "adam", "maya", "daniel", "tamar", "david", "shira", "jonathan"]
CHAR_ID = re.compile(r"[0-9a-f]{10}")
REF_SECONDS = 9.0  # BlueV3 reads only the first ~9 s of a reference clip
PHONETIC_MARKS = re.compile("[ֽ֫֯|]")  # phonikud's stress / vocal-shva marks
NIKUD = re.compile(r"[֑-ׇ]")
UNSAFE_NAME = re.compile(r"[^\w֐-׿\- ]")
BAD_MEDIA = "לא הצלחתי לקרוא את הקובץ — ייתכן שהוא פגום או בפורמט שלא נתמך"

SETTINGS = cfg.load()


# ---------- folders ----------

def out_dir() -> Path:
    custom = SETTINGS["app"].get("output_dir", "").strip()
    d = Path(custom).expanduser() if custom else paths.OUTPUT_DEFAULT
    d.mkdir(parents=True, exist_ok=True)
    return d


def allowed_path(p: Path) -> bool:
    roots = {paths.OUTPUT_DEFAULT.resolve(), out_dir().resolve()}
    return any(p == r or r in p.parents for r in roots)


def safe_name(text: str, fallback: str, limit: int = 60) -> str:
    return UNSAFE_NAME.sub("", text).strip()[:limit] or fallback


def new_folder(title: str, fallback: str) -> Path:
    """A fresh output folder; never reuses one (two exports in the same minute get different names)."""
    base = out_dir() / f"{safe_name(title, fallback)} {time.strftime('%Y-%m-%d %H-%M')}"
    folder, n = base, 2
    while folder.exists():
        folder, n = base.with_name(f"{base.name} ({n})"), n + 1
    return folder


def clean_tmp(max_age_h: float = 24):
    for f in paths.TMP.glob("*"):
        with contextlib.suppress(OSError):
            if time.time() - f.stat().st_mtime > max_age_h * 3600:
                f.unlink()


# ---------- engines (load on first use, so the app can open and download models before they exist) ----------

def apply_threads():
    n = int(SETTINGS["app"].get("threads") or 0)
    if n > 0:
        os.environ["ORT_NUM_THREADS"] = str(n)
    else:
        os.environ.pop("ORT_NUM_THREADS", None)  # BlueTTS default: min(8, cores)


def load_tts(precision: str) -> BlueTTS:
    t = BlueTTS(onnx_dir=str(BUNDLE / "onnx"), style_json=str(BUNDLE / "voices/noa.json"), precision=precision)
    # Read the user's nikud as hard constraints instead of stripping it.
    t.tts.g2p.renikud = G2P(str(BUNDLE / "frontend/renikud/model.onnx"), niqqud="use")
    return t


tts = None
nakdan = None
_asr = None
_load_lock = threading.Lock()
_tts_lock = threading.Lock()
_asr_lock = threading.Lock()


def engines():
    global tts, nakdan
    if tts is None or nakdan is None:
        if not models_setup.ready():
            raise HTTPException(409, "המודלים עוד לא הורדו — פתח את מסך ההתקנה")
        with _load_lock:
            if tts is None:
                tts = load_tts(SETTINGS["models"]["tts_precision"])
            if nakdan is None:
                nakdan = Phonikud(str(paths.MODELS / models_setup.PACKS["nikud"].check))
    return tts, nakdan


def asr(force_cpu: bool = False) -> WhisperModel:
    global _asr
    with _asr_lock:
        if _asr is None or force_cpu:
            m = SETTINGS["models"]
            path = str(models_setup.asr_dir(m["asr"]))
            if not force_cpu and m["asr_device"] in ("auto", "gpu"):
                gpu = "mps" if sys.platform == "darwin" else "cuda"  # Apple GPU, or an NVIDIA card elsewhere
                with contextlib.suppress(Exception):
                    _asr = WhisperModel(path, device=gpu, compute_type="float16")
                    return _asr
            _asr = WhisperModel(path, device="cpu", compute_type="int8")
    return _asr


def transcribe_any(audio, **kw):
    """Transcribe on the GPU when possible. CTranslate2's MPS backend is experimental: on an error there,
    switch to the CPU for good and retry once. A bad file still fails, as a clear message."""
    if not models_setup.installed(models_setup.ASR_PACK.get(SETTINGS["models"]["asr"], "asr")) \
            and not models_setup.installed("asr"):
        raise HTTPException(409, "מודל התמלול לא מותקן — אפשר להוריד אותו בהגדרות")
    model = asr()
    try:
        segs, info = model.transcribe(audio, **kw)
        return list(segs), info
    except Exception as first:
        if getattr(model.model, "device", "cpu") == "cpu":
            raise HTTPException(400, BAD_MEDIA) from None
        print(f"[asr] GPU failed ({first!r}); using CPU")
        try:
            segs, info = asr(force_cpu=True).transcribe(audio, **kw)
            return list(segs), info
        except Exception:
            raise HTTPException(400, BAD_MEDIA) from None


# ---------- speech ----------

def add_nikud(text: str) -> str:
    """Numbers/dates -> words, then nikud. Lines that are already pointed are kept as they are."""
    out = []
    for line in text.split("\n"):
        head, body = ("# ", line.lstrip()[2:]) if line.lstrip().startswith("# ") else ("", line)  # keep part titles
        if not body.strip() or docs.is_pointed(body):
            out.append(head + body)
        else:
            norm = prepare_text_for_synthesis(body, lang="he", mark_slow=False)
            out.append(head + PHONETIC_MARKS.sub("", engines()[1].add_diacritics(norm)))
    return "\n".join(out)


def voice_payload(char_id: str) -> dict:
    if char_id in PRESETS:
        p = BUNDLE / "voices" / f"{char_id}.json"
    elif CHAR_ID.fullmatch(char_id):
        p = CHARS / char_id / "voice.json"
    else:
        raise HTTPException(400, "מזהה דמות לא תקין")
    if not p.exists():
        raise HTTPException(404, "דמות לא נמצאה")
    return json.loads(p.read_text())


def style_of(char_id: str):
    if char_id in PRESETS:
        return load_voice_style([str(BUNDLE / "voices" / f"{char_id}.json")])
    return style_from_payload(voice_payload(char_id))


def track_rate(text: str, seconds: float, speed: float):
    """Learn this computer's real speaking rate, so Settings can show honest estimates."""
    letters = len(NIKUD.sub("", re.sub(r"\s+", "", text)))
    if letters < 15 or seconds <= 0:
        return
    st = SETTINGS["stats"]
    st["chars_per_sec"] = round(0.9 * st["chars_per_sec"] + 0.1 * letters / seconds * speed, 2)
    n_sent = max(1, len(re.findall(r"[.!?…]", text)))
    st["sentence_sec"] = round(0.9 * st["sentence_sec"] + 0.1 * seconds / n_sent, 2)
    st["samples"] = st.get("samples", 0) + 1
    if st["samples"] % 25 == 0:  # the window may be closed without a clean shutdown
        with contextlib.suppress(OSError):
            cfg.save(SETTINGS)


def synth(pointed_text: str, character: str, speed: float) -> tuple[np.ndarray, int]:
    engine = engines()[0]  # first: a clear message when the models are not installed yet
    style = style_of(character)
    with _tts_lock:
        try:
            samples, sr = engine.synthesize(pointed_text, lang="he", speed=speed, style=style, normalize_text=False)
        except RuntimeError as e:  # English words without the English G2P (no polyg2p build for this system)
            if "polyg2p" not in str(e) or not latin.has_latin(pointed_text):
                raise
            samples, sr = engine.synthesize(latin.to_hebrew(pointed_text), lang="he", speed=speed, style=style,
                                            normalize_text=False)
    track_rate(pointed_text, len(samples) / sr, speed)
    return samples, sr


def wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_mp3(path: Path, samples: np.ndarray, sr: int, title: str = "", artist: str = ""):
    with av.open(str(path), "w", format="mp3") as out:
        if title:
            out.metadata["title"] = title
        if artist:
            out.metadata["artist"] = artist
        st = out.add_stream("libmp3lame", rate=sr, layout="mono")
        st.bit_rate = 96_000
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = sr
        for pkt in [*st.encode(frame), *st.encode(None)]:
            out.mux(pkt)


def srt_time(x: float) -> str:
    h, m, s = int(x // 3600), int(x % 3600 // 60), x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def norm_words(t: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", NIKUD.sub("", t)).split()


def transcript_match(transcript: str, heard: str) -> float:
    """Share of the user's transcript words that are heard in the recording, in order.
    A transcript of only part of the recording still scores high."""
    want, got = norm_words(transcript), norm_words(heard)
    if not want:
        return 0.0
    matched = sum(b.size for b in difflib.SequenceMatcher(None, want, got, autojunk=False).get_matching_blocks())
    return matched / len(want)


def save_upload(upload: UploadFile) -> Path:
    p = paths.TMP / f"up_{uuid.uuid4().hex}{Path(upload.filename or '').suffix.lower()}"
    p.write_bytes(upload.file.read())
    return p


def parts_json(text: str, max_chars: int = 1200) -> list[dict]:
    return [{"title": p["title"], "sentences": docs.split_sentences(p["text"])}
            for p in docs.split_parts(text, max_chars=max_chars)]


# ---------- background jobs ----------

JOBS: dict[str, dict] = {}


def run_job(fields: dict, fn) -> str:
    """Run fn(job) in a thread; the UI polls /api/jobs/<id>. Keeps the 50 newest jobs."""
    for old in list(JOBS)[:-50]:
        if JOBS[old]["state"] != "running":
            del JOBS[old]
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"state": "running", "done": 0, "total": 1, "current": "", "folder": "", **fields}

    def wrap():
        try:
            fn(JOBS[job_id])
            JOBS[job_id]["state"] = "done"
        except HTTPException as e:
            JOBS[job_id].update(state="error", error=e.detail)
        except Exception as e:  # report to the UI instead of dying silently
            JOBS[job_id].update(state="error", error=str(e))

    threading.Thread(target=wrap, daemon=True).start()
    return job_id


# ---------- API ----------

@contextlib.asynccontextmanager
async def lifespan(_app):
    yield
    with contextlib.suppress(OSError):
        cfg.save(SETTINGS)  # keep the learned speaking rate


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(paths.RESOURCES / "index.html")


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "לא נמצא")
    return JOBS[job_id]


# --- first run ---

@app.get("/api/setup/status")
def setup_status():
    return {"ready": models_setup.ready(), "packs": models_setup.status(), "models_dir": str(paths.MODELS)}


@app.post("/api/models/download")
def download_packs(packs: str = Form(...)):
    ids = [p for p in packs.split(",") if p in models_setup.PACKS and not models_setup.installed(p)]
    if not ids:
        raise HTTPException(400, "אין מה להוריד")
    return {"job": run_job({"current": "מתחבר…"}, lambda job: models_setup.download(ids, job))}


# --- speech ---

@app.post("/api/nikud")
def nikud(text: str = Form(...)):
    t0 = time.time()
    return {"text": add_nikud(text), "ms": int((time.time() - t0) * 1000)}


# --- Fish S2-Pro (optional engine) ---

TAG = re.compile(r"(\[[^\]]+\])")
RENIKUD = BUNDLE / "frontend" / "renikud" / "model.onnx"


def nikud_keep_tags(text: str) -> str:
    """Nikud for the Hebrew parts only; [tags] (Fish emotion tags) pass through untouched."""
    return "".join(part if TAG.fullmatch(part) else add_nikud(part) for part in TAG.split(text))


def fish_ref(char_id: str) -> tuple[Path, str]:
    """A reference recording + its text for a character, so Fish speaks in the same voice as BlueV3:
    the original recording for cloned voices, a BlueV3 sample of the voice otherwise (made once, cached)."""
    voice_payload(char_id)  # validates the id
    cdir = CHARS / char_id
    if (cdir / "reference.wav").exists() and (cdir / "reference_text.txt").exists():
        text = (cdir / "reference_text.txt").read_text(encoding="utf-8").strip()
        if text:
            return cdir / "reference.wav", text
    fish_engine.REFS.mkdir(parents=True, exist_ok=True)
    ref = fish_engine.REFS / f"{char_id}.wav"
    if not ref.exists():
        samples, sr = synth(fish_engine.REF_TEXT, char_id, 1.0)
        sf.write(ref, samples, sr, subtype="PCM_16")
    return ref, fish_engine.REF_TEXT


def fish_synth(text: str, character: str) -> tuple[np.ndarray, int]:
    engines()  # BlueV3 is needed for the voice references and the shared Hebrew G2P model
    ref, ref_text = fish_ref(character)
    out = paths.TMP / f"fish_{uuid.uuid4().hex}.wav"
    try:
        fish_engine.synth(text, out, RENIKUD, ref, ref_text)
        samples, sr = sf.read(out, dtype="float32")
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None
    finally:
        out.unlink(missing_ok=True)
    return samples, sr


def render_line(line, voice: str, speed: float, engine: str) -> tuple[np.ndarray, int]:
    if engine != "fish":
        return story.render_line(line_synth(voice), line, speed)
    parts, sr = [], 44100
    for kind, value in story.fish_segments(line):
        if kind == "pause":
            parts.append(story.silence(value, sr))
        else:
            samples, sr = fish_synth(nikud_keep_tags(value), voice)
            parts.append(samples)
    return (np.concatenate(parts) if parts else story.silence(0.1, sr)), sr


@app.post("/api/speak")
def speak(text: str = Form(...), character: str = Form("noa"), speed: float = Form(1.0),
          auto_nikud: bool = Form(False), engine: str = Form("blue")):
    if not text.strip():
        raise HTTPException(400, "אין טקסט")
    t0 = time.time()
    if engine == "fish":
        samples, sr = fish_synth(nikud_keep_tags(text) if auto_nikud else text, character)
    else:
        samples, sr = synth(add_nikud(text) if auto_nikud else text, character, speed)
    return Response(wav_bytes(samples, sr), media_type="audio/wav",
                    headers={"X-Synth-Seconds": f"{time.time() - t0:.2f}", "X-Audio-Seconds": f"{len(samples) / sr:.2f}"})


@app.post("/api/extract")
def extract(file: UploadFile = File(...)):  # noqa: B008 (FastAPI idiom)
    p = save_upload(file)
    try:
        text = docs.extract(p)
    except Exception as e:
        raise HTTPException(400, f"לא הצלחתי לקרוא את הקובץ: {e}") from None
    finally:
        p.unlink(missing_ok=True)
    if not text.strip():
        raise HTTPException(400, "לא נמצא טקסט בקובץ (אולי זה PDF סרוק כתמונה?)")
    return {"title": Path(file.filename or "מסמך").stem, "text": text, "parts": parts_json(text)}


@app.post("/api/split")
def split(text: str = Form(...), max_chars: int = Form(1200)):
    return {"parts": parts_json(text, max_chars=max(200, min(max_chars, 20000)))}


@app.post("/api/export")
def export(text: str = Form(...), title: str = Form("הקראה"), character: str = Form("noa"), speed: float = Form(1.0)):
    parts = docs.split_parts(text, max_chars=int(SETTINGS["app"]["part_chars"]))
    if not parts:
        raise HTTPException(400, "אין טקסט")
    folder = new_folder(title, "הקראה")
    voice_name = next((c["name"] for c in characters() if c["id"] == character), character)

    def work(job):
        (folder / "טקסט").mkdir(parents=True)
        m3u, index = ["#EXTM3U", f"#PLAYLIST:{title}"], [f"{title}\nקול: {voice_name}\n"]
        for i, part in enumerate(parts, 1):
            job["current"] = part["title"]
            pointed = add_nikud(part["text"])
            samples, sr = synth(pointed, character, speed)
            secs = round(len(samples) / sr)
            fname = f"{i:03d} - {safe_name(part['title'], 'חלק', 50)}.mp3"
            write_mp3(folder / fname, samples, sr, title=part["title"], artist=voice_name)
            (folder / "טקסט" / fname.replace(".mp3", ".txt")).write_text(pointed, encoding="utf-8")
            m3u += [f"#EXTINF:{secs},{part['title']}", fname]
            index.append(f"{i:03d}  {part['title']}  ({secs} שניות)")
            job["done"] = i
        (folder / "playlist.m3u8").write_text("\n".join(m3u) + "\n", encoding="utf-8")
        (folder / "תוכן עניינים.txt").write_text("\n".join(index) + "\n", encoding="utf-8")

    return {"job": run_job({"total": len(parts), "folder": str(folder)}, work)}


@app.post("/api/reveal")
def reveal(path: str = Form(...)):
    p = Path(path).resolve()
    if not allowed_path(p):
        raise HTTPException(400, "נתיב לא מורשה")
    if sys.platform == "darwin":
        subprocess.run(["open", str(p)], check=False)
    elif sys.platform.startswith("win"):
        os.startfile(str(p))  # noqa: S606 - opens the folder in Explorer
    else:
        subprocess.run(["xdg-open", str(p)], check=False)
    return {"ok": True}


@app.get("/api/file")
def get_file(path: str):
    p = Path(path).resolve()
    if not allowed_path(p) or not p.is_file():
        raise HTTPException(400, "נתיב לא מורשה")
    return FileResponse(p)


# --- characters ---

@app.get("/api/characters")
def characters():
    out = [{"id": k, "name": v, "source": "קול מובנה"} for k, v in PRESETS.items()]
    for d in sorted(CHARS.iterdir(), key=lambda p: p.stat().st_mtime):
        m = d / "character.json"
        if m.exists():
            out.append(json.loads(m.read_text()))
    return out


def save_character(name: str, payload: dict, meta_extra: dict) -> dict:
    cid = uuid.uuid4().hex[:10]
    d = CHARS / cid
    d.mkdir()
    (d / "voice.json").write_text(json.dumps(payload))
    meta = {"id": cid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M"), **meta_extra}
    (d / "character.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def best_window(total_sec: float, segments: list, seconds: float) -> tuple[float, float, list]:
    """The stretch of speech (by ASR segments) closest to `seconds` long, with as few gaps as possible."""
    if not segments:
        return 0.0, min(seconds, total_sec), []
    best = None
    for i, first in enumerate(segments):
        start = max(0.0, first.start - 0.15)
        used = [s for s in segments[i:] if s.end - start <= seconds + 0.5] or [first]
        end = min(used[-1].end + 0.2, start + seconds + 0.5)
        speech = sum(min(s.end, end) - max(s.start, start) for s in used)
        score = speech - 0.5 * max(0.0, (end - start) - speech)  # prefer dense speech
        if best is None or score > best[0]:
            best = (score, start, end, used)
    return best[1], best[2], best[3]


@app.post("/api/characters/clone")
def clone(name: str = Form(...), consent: bool = Form(False), transcript: str = Form(""),
          audio: UploadFile = File(...)):  # noqa: B008
    if not consent:
        raise HTTPException(400, "צריך לאשר שיש הסכמה של בעל הקול")
    raw = save_upload(audio)
    try:
        s44 = decode_audio(str(raw), sampling_rate=44100)  # any format: wav, mp3, m4a, webm, video…
    except Exception:
        raise HTTPException(400, BAD_MEDIA) from None
    finally:
        raw.unlink(missing_ok=True)
    if len(s44) < 44100 * 3:
        raise HTTPException(400, "ההקלטה קצרה מדי — צריך לפחות 3 שניות, עדיף 8–10")
    s16 = resample_poly(s44, 160, 441).astype(np.float32)

    # Find where the speech is, then keep the best ~9 s (BlueV3 reads only that much).
    segs, _ = transcribe_any(s16, language="he", beam_size=5, vad_filter=True)
    heard = " ".join(s.text.strip() for s in segs)
    start, end, used = best_window(len(s44) / 44100, segs, REF_SECONDS)

    quality, notes = None, []
    if transcript.strip():
        quality = transcript_match(transcript, heard)
        if quality < 0.7:
            notes.append("ההקלטה לא תואמת היטב לתמלול — אולי יש רעש, מוזיקה, או שהתמלול של קטע אחר")
    if end - start < 5:
        notes.append("פחות מ־5 שניות דיבור נקי — הדמיון לקול יהיה חלש יותר")

    clip = paths.TMP / f"clip_{uuid.uuid4().hex}.wav"
    sf.write(clip, s44[int(start * 44100): int(end * 44100)], 44100, subtype="PCM_16")
    payload = export_voice_style(str(clip), onnx_dir=str(BUNDLE / "onnx"), config=engines()[0].tts.cfgs)
    meta = save_character(name, payload, {"source": "שכפול מהקלטה", "ref_seconds": round(end - start, 1),
                                          "quality": None if quality is None else round(quality, 2), "notes": notes})
    cdir = CHARS / meta["id"]
    clip.rename(cdir / "reference.wav")
    (cdir / "transcript_user.txt").write_text(transcript.strip(), encoding="utf-8")
    (cdir / "transcript_heard.txt").write_text(heard, encoding="utf-8")
    (cdir / "reference_text.txt").write_text(" ".join(s.text.strip() for s in used), encoding="utf-8")
    return {**meta, "heard": heard}


@app.post("/api/characters/mix")
def mix(name: str = Form(...), a: str = Form(...), b: str = Form(...), ratio: float = Form(0.5)):
    """New voice = (1-ratio)·A + ratio·B in the model's style space."""
    r = min(max(ratio, 0.0), 1.0)
    pa, pb = voice_payload(a), voice_payload(b)
    payload = json.loads(json.dumps(pa))
    for key in ("style_ttl", "style_dp"):
        payload[key]["data"] = ((1 - r) * np.asarray(pa[key]["data"]) + r * np.asarray(pb[key]["data"])).tolist()
    sb = pb.get("speed_scale", {})
    payload["speed_scale"] = {k: (1 - r) * v + r * sb.get(k, v) for k, v in pa.get("speed_scale", {}).items()}
    payload["source"] = f"mix of {a} and {b}"
    style_from_payload(payload)  # validate before saving
    return save_character(name, payload, {"source": f"ערבוב {round((1 - r) * 100)}% / {round(r * 100)}%"})


@app.delete("/api/characters/{cid}")
def delete_character(cid: str):
    if not CHAR_ID.fullmatch(cid) or not (CHARS / cid).is_dir():
        raise HTTPException(400, "אי אפשר למחוק את הדמות הזו")
    paths.CHARS_DELETED.mkdir(exist_ok=True)
    (CHARS / cid).rename(paths.CHARS_DELETED / cid)  # kept, not destroyed: can be restored by moving it back
    return {"ok": True}


# --- stories: several characters + stage directions ---

def cast_of(cast_json: str, speakers: list[str]) -> dict:
    try:
        cast = json.loads(cast_json or "{}")
    except json.JSONDecodeError:
        cast = {}
    return {s: cast.get(s) or DEFAULT_CAST[i % len(DEFAULT_CAST)] for i, s in enumerate(speakers)}


def line_synth(voice: str):
    return lambda text, speed: synth(add_nikud(text), voice, speed)


@app.get("/api/story/catalogue")
def story_catalogue():
    return story.catalogue()


@app.post("/api/story/parse")
def story_parse(script: str = Form(...)):
    lines, speakers, errors = story.parse(script)
    return {"speakers": speakers, "errors": errors,
            "lines": [{"n": ln.n, "row": ln.row, "speaker": ln.speaker, "text": ln.text, "scene": ln.scene} for ln in lines]}


@app.post("/api/story/line")
def story_line(script: str = Form(...), row: int = Form(...), cast: str = Form("{}"), speed: float = Form(1.0),
               engine: str = Form("blue")):
    lines, speakers, _ = story.parse(script)
    if not lines:
        raise HTTPException(400, "אין שורות בסיפור")
    # the line the cursor is on, or the closest one above it
    line = max((ln for ln in lines if ln.row <= row), key=lambda ln: ln.row, default=lines[0])
    t0 = time.time()
    samples, sr = render_line(line, cast_of(cast, speakers)[line.speaker], speed, engine)
    return Response(wav_bytes(samples, sr), media_type="audio/wav",
                    headers={"X-Line": str(line.n), "X-Synth-Seconds": f"{time.time() - t0:.2f}"})


@app.post("/api/story/render")
def story_render(script: str = Form(...), cast: str = Form("{}"), title: str = Form("סיפור"),
                 speed: float = Form(1.0), gap: float = Form(0.35), engine: str = Form("blue")):
    lines, speakers, errors = story.parse(script)
    if not lines:
        raise HTTPException(400, "אין שורות בסיפור")
    voices = cast_of(cast, speakers)
    names = {c["id"]: c["name"] for c in characters()}
    safe = safe_name(title, "סיפור")
    folder = new_folder(title, "סיפור")
    scenes: list[tuple[str, list]] = []  # consecutive lines grouped by "# title" rows
    for line in lines:
        name = line.scene or "פתיחה"
        if not scenes or scenes[-1][0] != name:
            scenes.append((name, []))
        scenes[-1][1].append(line)

    def work(job):
        (folder / "סצנות").mkdir(parents=True)
        full, srt, m3u, t, sr = [], [], ["#EXTM3U", f"#PLAYLIST:{title}"], 0.0, 44100
        for si, (scene_name, scene_lines) in enumerate(scenes, 1):
            audio, scene_srt, st = [], [], 0.0
            for line in scene_lines:
                job["current"] = f"{scene_name} · {line.speaker}: {line.text[:30]}"
                samples, sr = render_line(line, voices[line.speaker], speed, engine)
                dur, cue = len(samples) / sr, f"{line.speaker}: {line.text}"
                scene_srt.append(f"{len(scene_srt) + 1}\n{srt_time(st)} --> {srt_time(st + dur)}\n{cue}\n")
                srt.append(f"{len(srt) + 1}\n{srt_time(t + st)} --> {srt_time(t + st + dur)}\n{cue}\n")
                audio += [samples, story.silence(gap, sr)]
                st += dur + gap
                job["done"] += 1
            scene_audio = np.concatenate(audio)
            base = f"{si:02d} - {safe_name(scene_name, 'סצנה', 50)}"
            write_mp3(folder / "סצנות" / f"{base}.mp3", scene_audio, sr, title=scene_name, artist=title)
            (folder / "סצנות" / f"{base}.srt").write_text("\n".join(scene_srt), encoding="utf-8")
            m3u += [f"#EXTINF:{round(st)},{scene_name}", f"סצנות/{base}.mp3"]
            full += [scene_audio, story.silence(1.0, sr)]  # a longer breath between scenes
            t += st + 1.0
        write_mp3(folder / f"{safe} - מלא.mp3", np.concatenate(full), sr, title=title)
        (folder / "כתוביות - מלא.srt").write_text("\n".join(srt), encoding="utf-8")
        (folder / "playlist.m3u8").write_text("\n".join(m3u) + "\n", encoding="utf-8")
        cast_txt = "\n".join(f"{s}: {names.get(v, v)}" for s, v in voices.items())
        (folder / "תסריט.txt").write_text(f"{title}\n\nליהוק:\n{cast_txt}\n\n{script}\n", encoding="utf-8")
        job.update(scenes=len(scenes), full=str(folder / f"{safe} - מלא.mp3"))

    return {"job": run_job({"total": len(lines), "folder": str(folder), "warnings": errors}, work)}


# --- settings ---

_HW = None


def hardware() -> dict:
    global _HW
    if _HW is None:
        _HW = cfg.scan()
    return _HW


@app.get("/api/settings")
def get_settings():
    return {"settings": cfg.public(SETTINGS), "providers": cfg.PROVIDERS, "version": cfg.VERSION,
            "output_default": str(paths.OUTPUT_DEFAULT), "output_dir": str(out_dir())}


@app.get("/api/engines")
def engines_status():
    return {"blue": {"installed": models_setup.ready()}, "fish": fish_engine.status()}


@app.post("/api/engines/fish/install")
def fish_install(accept_license: bool = Form(False)):
    if not accept_license:
        raise HTTPException(400, "צריך לאשר את רישיון Fish Audio Research (לשימוש לא מסחרי בלבד)")
    if any(j["state"] == "running" and j.get("kind") == "fish-install" for j in JOBS.values()):
        raise HTTPException(409, "ההתקנה כבר רצה")
    return {"job": run_job({"kind": "fish-install", "current": "מתחיל…"}, fish_engine.install)}


@app.post("/api/engines/fish/start")
def fish_start():
    if not fish_engine.installed():
        raise HTTPException(400, "מנוע Fish לא מותקן")
    engines()
    fish_engine.warm_up(RENIKUD)
    return fish_engine.status()


@app.delete("/api/engines/fish")
def fish_remove():
    fish_engine.uninstall()
    return {"ok": True}


@app.get("/api/system")
def system():
    hw = hardware()
    return {"hardware": hw, "models": cfg.catalogue(hw, SETTINGS)}


@app.post("/api/settings")
def set_settings(data: str = Form(...)):
    """Merge the posted sections; reload only the engines whose settings changed."""
    global tts, _asr
    try:
        new = json.loads(data)
    except json.JSONDecodeError:
        raise HTTPException(400, "הגדרות לא תקינות") from None
    old = json.loads(json.dumps(SETTINGS))
    for section in ("app", "models", "llm", "theme"):
        if isinstance(new.get(section), dict):
            SETTINGS[section].update({k: v for k, v in new[section].items() if k in SETTINGS[section]})
    if SETTINGS["llm"].get("api_key", "").startswith("••••"):  # the browser only sees the masked key
        SETTINGS["llm"]["api_key"] = old["llm"]["api_key"]
    out = SETTINGS["app"]["output_dir"].strip()
    if out and not Path(out).expanduser().is_dir():
        SETTINGS["app"]["output_dir"] = old["app"]["output_dir"]
        raise HTTPException(400, "התיקייה לא קיימת")
    reloaded = []
    m, o = SETTINGS["models"], old["models"]
    threads_changed = SETTINGS["app"]["threads"] != old["app"]["threads"]
    if threads_changed:
        apply_threads()
    if tts is not None and (threads_changed or m["tts_precision"] != o["tts_precision"]):
        with _tts_lock:
            tts = load_tts(m["tts_precision"])
        reloaded.append("הקראה")
    if m["asr"] != o["asr"] or m["asr_device"] != o["asr_device"]:
        with _asr_lock:
            _asr = None  # loads on next use
        reloaded.append("תמלול")
    cfg.save(SETTINGS)
    return {"ok": True, "reloaded": reloaded, "settings": cfg.public(SETTINGS), "output_dir": str(out_dir())}


ANTIGRAVITY_CONFIG = Path.home() / ".gemini" / "config" / "mcp_config.json"


def mcp_entry() -> dict:
    """How an AI client starts the Kol Studio MCP server on this computer."""
    if getattr(sys, "frozen", False):
        name = "kolstudio-mcp.exe" if sys.platform.startswith("win") else "kolstudio-mcp"
        return {"command": str(Path(sys.executable).with_name(name)), "args": []}
    root = Path(__file__).resolve().parent.parent
    return {"command": sys.executable, "args": ["-m", "kol_studio.mcp_server"], "cwd": str(root)}


@app.get("/api/mcp/info")
def mcp_info():
    entry = mcp_entry()
    installed = False
    with contextlib.suppress(OSError, ValueError):
        installed = "kol-studio" in json.loads(ANTIGRAVITY_CONFIG.read_text()).get("mcpServers", {})
    return {"entry": entry, "config": {"mcpServers": {"kol-studio": entry}},
            "antigravity_config": str(ANTIGRAVITY_CONFIG), "antigravity_installed": installed,
            "antigravity_present": ANTIGRAVITY_CONFIG.parent.is_dir()}


@app.post("/api/mcp/install-antigravity")
def mcp_install_antigravity():
    """Add (or update) the kol-studio entry in Antigravity's MCP config. Other servers are kept; a backup is saved."""
    if not ANTIGRAVITY_CONFIG.parent.is_dir():
        raise HTTPException(400, "Antigravity לא נמצא במחשב הזה")
    data = {}
    if ANTIGRAVITY_CONFIG.exists():
        try:
            data = json.loads(ANTIGRAVITY_CONFIG.read_text() or "{}")
        except ValueError:
            raise HTTPException(400, "קובץ ההגדרות של Antigravity לא תקין — לא שיניתי אותו") from None
        ANTIGRAVITY_CONFIG.with_name("mcp_config.backup.json").write_text(ANTIGRAVITY_CONFIG.read_text())
    data.setdefault("mcpServers", {})["kol-studio"] = mcp_entry()
    ANTIGRAVITY_CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return {"ok": True, "path": str(ANTIGRAVITY_CONFIG)}


@app.post("/api/pick-folder")
def pick_folder():
    return {"path": cfg.pick_folder(str(out_dir()))}


@app.post("/api/benchmark")
def benchmark():
    """Measure speech speed with different core counts on this computer; the UI offers the fastest."""
    engines()  # models must be installed
    cores = hardware()["cores"]
    counts = sorted({n for n in (1, 2, 4, 6, 8, cores) if n <= cores})
    text = add_nikud("המחקר עוסק בשפה העברית המודרנית, והוא פורסם בכתבי עת רבים בשנים האחרונות.")
    style = style_of("noa")

    def work(job):
        job["results"] = []
        try:
            for i, n in enumerate(counts):
                job["current"] = f"{n} ליבות"
                os.environ["ORT_NUM_THREADS"] = str(n)
                t = load_tts(SETTINGS["models"]["tts_precision"])
                t.synthesize(text, style=style, normalize_text=False)  # warm-up
                t0 = time.time()
                for _ in range(2):
                    s, sr = t.synthesize(text, style=style, normalize_text=False)
                job["results"].append({"threads": n, "x_realtime": round(len(s) / sr / ((time.time() - t0) / 2), 1)})
                job["done"] = i + 1
        finally:
            apply_threads()
        job["best"] = max(job["results"], key=lambda r: r["x_realtime"])["threads"]

    return {"job": run_job({"total": len(counts)}, work)}


@app.get("/api/update/check")
def update_check():
    try:
        return cfg.check_update(SETTINGS["app"]["update_repo"], cfg.VERSION)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from None


@app.get("/api/models/search")
def models_search(task: str = "", q: str = ""):
    return {"results": cfg.search_models(task, hardware().get("ram_gb") or 8, q)}


def llm_args(provider: str, base_url: str, api_key: str) -> tuple[str, str, str]:
    if not api_key or api_key.startswith("••••"):
        api_key = SETTINGS["llm"]["api_key"] if provider == SETTINGS["llm"]["provider"] else ""
    base_url = base_url or cfg.PROVIDERS.get(provider, {}).get("base_url", "")
    if not base_url:
        raise HTTPException(400, "חסרה כתובת שרת")
    return provider, base_url, api_key


@app.post("/api/llm/models")
def llm_models(provider: str = Form(...), base_url: str = Form(""), api_key: str = Form("")):
    try:
        return {"models": cfg.list_models(*llm_args(provider, base_url, api_key))}
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from None


@app.post("/api/llm/test")
def llm_test(provider: str = Form(...), base_url: str = Form(""), api_key: str = Form(""), model: str = Form(...)):
    p, b, k = llm_args(provider, base_url, api_key)
    try:
        reply, secs = cfg.timed(cfg.chat, p, b, k, model, "ענה במשפט אחד בעברית: מה תפקידו של ניקוד בעברית?")
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from None
    except (KeyError, IndexError):
        raise HTTPException(400, "השרת ענה בפורמט לא צפוי") from None
    return {"reply": reply.strip(), "seconds": secs}


@app.get("/api/llm/detect")
def llm_detect():
    return {"found": cfg.detect_local()}


# --- transcription ---

@app.post("/api/transcribe")
def transcribe(media: UploadFile = File(...)):  # noqa: B008
    raw = save_upload(media)
    t0 = time.time()
    try:
        segs, info = transcribe_any(str(raw), language="he", beam_size=5, vad_filter=True)
    finally:
        raw.unlink(missing_ok=True)
    rows = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
    srt = "\n".join(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n" for i, s in enumerate(rows, 1))
    return {"text": " ".join(s["text"] for s in rows), "srt": srt, "segments": rows,
            "audio_seconds": info.duration, "took_seconds": time.time() - t0}


def main(port: int = 8765):
    apply_threads()
    clean_tmp()
    paths.SERVER_FILE.write_text(json.dumps({"port": port, "pid": os.getpid(), "version": cfg.VERSION}))
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        with contextlib.suppress(OSError):
            if json.loads(paths.SERVER_FILE.read_text()).get("pid") == os.getpid():
                paths.SERVER_FILE.unlink()


if __name__ == "__main__":
    main()
