"""Kol Studio server: Hebrew nikud -> speech (BlueV3), voice characters, stories, documents, reader, transcription.

Started by kol_studio.desktop (the app) or `python -m kol_studio.server` (development).
Everything stays on this computer (binds to 127.0.0.1 only).
"""
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
import jiwer
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

from . import docs, models_setup, paths, story
from . import settings as cfg

MODELS = paths.MODELS
BUNDLE = MODELS / "bluev3"
CHARS = paths.CHARS
TMP = paths.TMP
OUTPUT = paths.OUTPUT_DEFAULT  # exported audio folders (changeable in Settings)

PRESETS = {  # BlueV3 built-in voices (CC BY 4.0, LibriTTS-R speakers)
    "maya": "מאיה", "noa": "נועה", "lily": "לילי", "tamar": "תמר", "shira": "שירה",
    "yael": "יעל", "emma": "אמה", "sara": "שרה", "adam": "אדם", "daniel": "דניאל",
    "david": "דוד", "jonathan": "יונתן", "ethan": "איתן", "michael": "מיכאל", "ariel": "אריאל",
}
REF_SECONDS = 9.0  # BlueV3 reads only the first ~9 s of a reference clip

SETTINGS = cfg.load()


def apply_threads():
    n = int(SETTINGS["app"].get("threads") or 0)
    if n > 0:
        os.environ["ORT_NUM_THREADS"] = str(n)
    else:
        os.environ.pop("ORT_NUM_THREADS", None)  # BlueTTS default: min(8, cores)


def out_dir() -> Path:
    custom = SETTINGS["app"].get("output_dir", "").strip()
    d = Path(custom).expanduser() if custom else OUTPUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def allowed_path(p: Path) -> bool:
    roots = {OUTPUT.resolve(), out_dir().resolve()}
    return any(p == r or r in p.parents for r in roots)


apply_threads()
_asr = None
_asr_lock = threading.Lock()
_tts_lock = threading.Lock()


def load_tts(precision: str):
    t = BlueTTS(onnx_dir=str(BUNDLE / "onnx"), style_json=str(BUNDLE / "voices/noa.json"), precision=precision)
    # Read the user's nikud as hard constraints instead of stripping it.
    t.tts.g2p.renikud = G2P(str(BUNDLE / "frontend/renikud/model.onnx"), niqqud="use")
    return t


def load_nakdan(name: str):
    if name == "dicta" and (MODELS / "dicta-1.0.onnx").exists():
        from dicta_onnx import Dicta

        return Dicta(str(MODELS / "dicta-1.0.onnx"))
    return Phonikud(str(MODELS / "phonikud-1.0.int8.onnx"))


# Engines load on first use, so the app opens (and can download models) before they exist.
tts = None
nakdan = None
_load_lock = threading.Lock()


def engines():
    global tts, nakdan
    if tts is None or nakdan is None:
        if not models_setup.ready():
            raise HTTPException(409, "המודלים עוד לא הורדו — פתח את מסך ההתקנה")
        with _load_lock:
            if tts is None:
                tts = load_tts(SETTINGS["models"]["tts_precision"])
            if nakdan is None:
                nakdan = load_nakdan(SETTINGS["models"]["nikud"])
    return tts, nakdan

PHONETIC_MARKS = re.compile("[ֽ֫֯|]")  # phonikud's stress / vocal-shva marks


# ---------- shared helpers ----------

def asr():
    global _asr
    with _asr_lock:
        if _asr is None:
            m = SETTINGS["models"]
            path = cfg.asr_path(m["asr"])
            if not (path / "model.bin").exists():
                path = cfg.asr_path("ivrit-turbo")
            if m["asr_device"] in ("auto", "gpu"):
                # Apple GPU (~2x faster than CPU on M1 Max) or an NVIDIA card on Windows/Linux
                gpu = "mps" if sys.platform == "darwin" else "cuda"
                try:
                    _asr = WhisperModel(str(path), device=gpu, compute_type="float16")
                    return _asr
                except Exception:
                    pass
            _asr = WhisperModel(str(path), device="cpu", compute_type="int8")
    return _asr


def transcribe_any(audio, **kw):
    """Transcribe on the Apple GPU; ctranslate2's MPS backend is experimental, so on any error
    switch to the CPU for good and retry."""
    global _asr
    try:
        segs, info = asr().transcribe(audio, **kw)
        return list(segs), info
    except Exception as e:
        print(f"[asr] GPU failed ({e!r}); using CPU")
        with _asr_lock:
            path = cfg.asr_path(SETTINGS["models"]["asr"])
            if not (path / "model.bin").exists():
                path = cfg.asr_path("ivrit-turbo")
            _asr = WhisperModel(str(path), device="cpu", compute_type="int8")
        segs, info = _asr.transcribe(audio, **kw)
        return list(segs), info


def add_nikud(text: str) -> str:
    """Numbers/dates -> words, then nikud. Lines that are already pointed are kept as they are."""
    out = []
    for line in text.split("\n"):
        if not line.strip() or docs.is_pointed(line):
            out.append(line)
            continue
        norm = prepare_text_for_synthesis(line, lang="he", mark_slow=False)
        out.append(PHONETIC_MARKS.sub("", engines()[1].add_diacritics(norm)))
    return "\n".join(out)


def voice_payload(char_id: str) -> dict:
    p = BUNDLE / "voices" / f"{char_id}.json" if char_id in PRESETS else CHARS / char_id / "voice.json"
    if not p.exists():
        raise HTTPException(404, "דמות לא נמצאה")
    return json.loads(p.read_text())


def style_of(char_id: str):
    if char_id in PRESETS:
        return load_voice_style([str(BUNDLE / "voices" / f"{char_id}.json")])
    return style_from_payload(voice_payload(char_id))


def synth(pointed_text: str, character: str, speed: float) -> tuple[np.ndarray, int]:
    style = style_of(character)
    with _tts_lock:
        return engines()[0].synthesize(pointed_text, lang="he", speed=speed, style=style, normalize_text=False)


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
        for pkt in st.encode(frame):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)


def norm_words(t: str) -> str:
    t = re.sub(r"[֑-ׇ]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def save_upload(upload: UploadFile) -> Path:
    p = TMP / f"up_{uuid.uuid4().hex}{Path(upload.filename or '').suffix.lower()}"
    p.write_bytes(upload.file.read())
    return p


# ---------- jobs (long exports run in the background) ----------

JOBS: dict[str, dict] = {}


def run_job(job_id: str, fn):
    def wrap():
        try:
            fn(JOBS[job_id])
            JOBS[job_id]["state"] = "done"
        except Exception as e:  # report to the UI instead of dying silently
            JOBS[job_id].update(state="error", error=str(e))

    threading.Thread(target=wrap, daemon=True).start()


# ---------- API ----------

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(paths.RESOURCES / "index.html")


@app.get("/api/setup/status")
def setup_status():
    return {"ready": models_setup.ready(), "packs": models_setup.status(), "models_dir": str(MODELS)}


@app.post("/api/setup/download")
def setup_download(packs: str = Form(...)):
    ids = [p for p in packs.split(",") if p in models_setup.PACKS]
    if not ids:
        raise HTTPException(400, "לא נבחרו חבילות")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"state": "running", "done": 0, "total": 1, "current": "מתחבר…", "folder": ""}
    run_job(job_id, lambda job: models_setup.download(ids, job))
    return {"job": job_id}


@app.get("/api/characters")
def characters():
    out = [{"id": k, "name": v, "source": "קול מובנה"} for k, v in PRESETS.items()]
    for d in sorted(CHARS.iterdir(), key=lambda p: p.stat().st_mtime):
        m = d / "character.json"
        if m.exists():
            out.append(json.loads(m.read_text()))
    return out


@app.post("/api/nikud")
def nikud(text: str = Form(...)):
    t0 = time.time()
    return {"text": add_nikud(text), "ms": int((time.time() - t0) * 1000)}


@app.post("/api/speak")
def speak(text: str = Form(...), character: str = Form("noa"), speed: float = Form(1.0),
          auto_nikud: bool = Form(False)):
    if not text.strip():
        raise HTTPException(400, "אין טקסט")
    t0 = time.time()
    pointed = add_nikud(text) if auto_nikud else text
    samples, sr = synth(pointed, character, speed)
    secs = time.time() - t0
    return Response(
        wav_bytes(samples, sr),
        media_type="audio/wav",
        headers={"X-Synth-Seconds": f"{secs:.2f}", "X-Audio-Seconds": f"{len(samples) / sr:.2f}"},
    )


@app.post("/api/extract")
def extract(file: UploadFile = File(...)):
    p = save_upload(file)
    try:
        text = docs.extract(p)
    except Exception as e:
        raise HTTPException(400, f"לא הצלחתי לקרוא את הקובץ: {e}")
    finally:
        p.unlink(missing_ok=True)
    if not text.strip():
        raise HTTPException(400, "לא נמצא טקסט בקובץ (אולי זה PDF סרוק כתמונה?)")
    parts = docs.split_parts(text)
    return {"title": Path(file.filename or "מסמך").stem, "text": text,
            "parts": [{"title": p["title"], "sentences": docs.split_sentences(p["text"])} for p in parts]}


@app.post("/api/split")
def split(text: str = Form(...)):
    parts = docs.split_parts(text)
    return {"parts": [{"title": p["title"], "sentences": docs.split_sentences(p["text"])} for p in parts]}


# --- characters ---

def save_character(name: str, payload: dict, meta_extra: dict) -> dict:
    cid = uuid.uuid4().hex[:10]
    d = CHARS / cid
    d.mkdir()
    (d / "voice.json").write_text(json.dumps(payload))
    meta = {"id": cid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M"), **meta_extra}
    (d / "character.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def best_window(samples16: np.ndarray, segments: list, seconds: float) -> tuple[float, float, list]:
    """Pick the continuous stretch of speech (by ASR segments) closest to `seconds` long, starting at speech."""
    if not segments:
        return 0.0, min(seconds, len(samples16) / 16000), []
    best = None
    for i in range(len(segments)):
        start = max(0.0, segments[i].start - 0.15)
        used = []
        for s in segments[i:]:
            if s.end - start > seconds + 0.5:
                break
            used.append(s)
        if not used:
            used = [segments[i]]
        end = min(used[-1].end + 0.2, start + seconds + 0.5)
        speech = sum(min(s.end, end) - max(s.start, start) for s in used)
        score = speech - 0.5 * max(0.0, (end - start) - speech)  # prefer dense speech, few gaps
        if best is None or score > best[0]:
            best = (score, start, end, used)
    return best[1], best[2], best[3]


@app.post("/api/characters/clone")
def clone(name: str = Form(...), consent: bool = Form(False), transcript: str = Form(""),
          audio: UploadFile = File(...)):
    if not consent:
        raise HTTPException(400, "צריך לאשר שיש הסכמה של בעל הקול")
    raw = save_upload(audio)
    try:
        s44 = decode_audio(str(raw), sampling_rate=44100)  # any format: wav, mp3, m4a, webm, video…
        s16 = decode_audio(str(raw), sampling_rate=16000)
    finally:
        raw.unlink(missing_ok=True)
    if len(s16) < 16000 * 3:
        raise HTTPException(400, "ההקלטה קצרה מדי — צריך לפחות 3 שניות, עדיף 8–10")

    # Transcribe to find where the speech is, then keep the best ~9 s (BlueV3 reads only that much).
    segs, _ = transcribe_any(s16, language="he", beam_size=5, vad_filter=True)
    heard = " ".join(s.text.strip() for s in segs)
    start, end, used = best_window(s16, segs, REF_SECONDS)
    clip = s44[int(start * 44100): int(end * 44100)]

    quality, notes = None, []
    if transcript.strip():
        quality = max(0.0, 1 - jiwer.wer(norm_words(transcript), norm_words(heard)))
        if quality < 0.7:
            notes.append("ההקלטה לא תואמת היטב לתמלול — אולי יש רעש, מוזיקה, או שהתמלול של קטע אחר")
    if end - start < 5:
        notes.append("פחות מ־5 שניות דיבור נקי — הדמיון לקול יהיה חלש יותר")

    d = TMP / f"clip_{uuid.uuid4().hex}.wav"
    sf.write(d, clip, 44100, subtype="PCM_16")
    payload = export_voice_style(str(d), onnx_dir=str(BUNDLE / "onnx"), config=engines()[0].tts.cfgs)
    meta = save_character(name, payload, {
        "source": "שכפול מהקלטה",
        "ref_seconds": round(end - start, 1),
        "quality": None if quality is None else round(quality, 2),
        "notes": notes,
    })
    cdir = CHARS / meta["id"]
    d.rename(cdir / "reference.wav")
    (cdir / "transcript_user.txt").write_text(transcript.strip())
    (cdir / "transcript_heard.txt").write_text(heard)
    (cdir / "reference_text.txt").write_text(" ".join(s.text.strip() for s in used))
    return {**meta, "heard": heard}


@app.post("/api/characters/mix")
def mix(name: str = Form(...), a: str = Form(...), b: str = Form(...), ratio: float = Form(0.5)):
    """New voice = (1-ratio)·A + ratio·B in the model's style space."""
    r = min(max(ratio, 0.0), 1.0)
    pa, pb = voice_payload(a), voice_payload(b)

    def lerp(x, y):
        return ((1 - r) * np.asarray(x, dtype=np.float64) + r * np.asarray(y, dtype=np.float64)).tolist()

    payload = json.loads(json.dumps(pa))
    for key in ("style_ttl", "style_dp"):
        payload[key]["data"] = lerp(pa[key]["data"], pb[key]["data"])
    sa_speed, sb_speed = pa.get("speed_scale", {}), pb.get("speed_scale", {})
    payload["speed_scale"] = {k: (1 - r) * v + r * sb_speed.get(k, v) for k, v in sa_speed.items()}
    payload["source"] = f"mix of {a} and {b}"
    style_from_payload(payload)  # validate
    return save_character(name, payload, {"source": f"ערבוב {round((1 - r) * 100)}% / {round(r * 100)}%"})


@app.delete("/api/characters/{cid}")
def delete_character(cid: str):
    d = CHARS / cid
    if cid in PRESETS or not d.is_dir() or not re.fullmatch(r"[0-9a-f]{10}", cid):
        raise HTTPException(400, "אי אפשר למחוק את הדמות הזו")
    trash = CHARS.parent / "characters_deleted"
    trash.mkdir(exist_ok=True)
    d.rename(trash / cid)  # kept, not destroyed: can be restored by moving it back
    return {"ok": True}


# --- export: folder of small MP3 files + playlist ---

@app.post("/api/export")
def export(text: str = Form(...), title: str = Form("הקראה"), character: str = Form("noa"),
           speed: float = Form(1.0)):
    parts = docs.split_parts(text, max_chars=int(SETTINGS["app"]["part_chars"]))
    if not parts:
        raise HTTPException(400, "אין טקסט")
    safe = re.sub(r"[^\w֐-׿\- ]", "", title).strip()[:60] or "הקראה"
    folder = out_dir() / f"{safe} {time.strftime('%Y-%m-%d %H-%M')}"
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"state": "running", "done": 0, "total": len(parts), "folder": str(folder), "current": ""}

    def work(job):
        folder.mkdir(parents=True)
        voice_name = next((c["name"] for c in characters() if c["id"] == character), character)
        m3u = ["#EXTM3U", f"#PLAYLIST:{title}"]
        index = [f"{title}\nקול: {voice_name}\n"]
        for i, part in enumerate(parts, 1):
            job["current"] = part["title"]
            pointed = add_nikud(part["text"])
            samples, sr = synth(pointed, character, speed)
            fname = f"{i:03d} - {re.sub(r'[^\w֐-׿\- ]', '', part['title'])[:50]}.mp3"
            write_mp3(folder / fname, samples, sr, title=part["title"], artist=voice_name)
            (folder / "טקסט").mkdir(exist_ok=True)
            (folder / "טקסט" / fname.replace(".mp3", ".txt")).write_text(pointed)
            m3u += [f"#EXTINF:{round(len(samples) / sr)},{part['title']}", fname]
            index.append(f"{i:03d}  {part['title']}  ({round(len(samples) / sr)} שניות)")
            job["done"] = i
        (folder / "playlist.m3u8").write_text("\n".join(m3u) + "\n", encoding="utf-8")
        (folder / "תוכן עניינים.txt").write_text("\n".join(index) + "\n", encoding="utf-8")

    run_job(job_id, work)
    return {"job": job_id, **JOBS[job_id]}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "לא נמצא")
    return JOBS[job_id]


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


# --- stories: several characters + stage directions ---

DEFAULT_CAST = ["noa", "adam", "maya", "daniel", "tamar", "david", "shira", "jonathan"]


def cast_of(cast_json: str, speakers: list[str]) -> dict:
    try:
        cast = json.loads(cast_json or "{}")
    except json.JSONDecodeError:
        cast = {}
    return {s: cast.get(s) or DEFAULT_CAST[i % len(DEFAULT_CAST)] for i, s in enumerate(speakers)}


def line_synth(voice: str):
    return lambda text, speed: synth(add_nikud(text), voice, speed)


def srt_time(x: float) -> str:
    h, m, s = int(x // 3600), int(x % 3600 // 60), x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


@app.get("/api/story/catalogue")
def story_catalogue():
    return story.catalogue()


@app.post("/api/story/parse")
def story_parse(script: str = Form(...)):
    lines, speakers, errors = story.parse(script)
    return {"speakers": speakers, "errors": errors,
            "lines": [{"n": l.n, "row": l.row, "speaker": l.speaker, "text": l.text, "scene": l.scene} for l in lines]}


@app.post("/api/story/line")
def story_line(script: str = Form(...), row: int = Form(...), cast: str = Form("{}"), speed: float = Form(1.0)):
    lines, speakers, _ = story.parse(script)
    if not lines:
        raise HTTPException(400, "אין שורות בסיפור")
    line = next((l for l in lines if l.row >= row), lines[-1])
    if line.row > row:  # cursor sits after the last line of a speaker block: take the previous line
        line = max((l for l in lines if l.row <= row), key=lambda l: l.row, default=line)
    voice = cast_of(cast, speakers)[line.speaker]
    t0 = time.time()
    samples, sr = story.render_line(line_synth(voice), line, speed)
    return Response(wav_bytes(samples, sr), media_type="audio/wav",
                    headers={"X-Line": str(line.n), "X-Synth-Seconds": f"{time.time() - t0:.2f}"})


@app.post("/api/story/render")
def story_render(script: str = Form(...), cast: str = Form("{}"), title: str = Form("סיפור"),
                 speed: float = Form(1.0), gap: float = Form(0.35)):
    lines, speakers, errors = story.parse(script)
    if not lines:
        raise HTTPException(400, "אין שורות בסיפור")
    voices = cast_of(cast, speakers)
    names = {c["id"]: c["name"] for c in characters()}
    safe = re.sub(r"[^\w֐-׿\- ]", "", title).strip()[:60] or "סיפור"
    folder = out_dir() / f"{safe} {time.strftime('%Y-%m-%d %H-%M')}"
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"state": "running", "done": 0, "total": len(lines), "folder": str(folder), "current": "",
                    "warnings": errors}

    # group consecutive lines into scenes ("# title" rows); lines before the first title form an opening scene
    scenes: list[tuple[str, list]] = []
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
                samples, sr = story.render_line(line_synth(voices[line.speaker]), line, speed)
                dur = len(samples) / sr
                cue = f"{line.speaker}: {line.text}"
                scene_srt.append(f"{len(scene_srt) + 1}\n{srt_time(st)} --> {srt_time(st + dur)}\n{cue}\n")
                srt.append(f"{len(srt) + 1}\n{srt_time(t + st)} --> {srt_time(t + st + dur)}\n{cue}\n")
                audio += [samples, story.silence(gap, sr)]
                st += dur + gap
                job["done"] += 1
            scene_audio = np.concatenate(audio)
            base = f"{si:02d} - {re.sub(r'[^\w֐-׿\- ]', '', scene_name)[:50]}"
            write_mp3(folder / "סצנות" / f"{base}.mp3", scene_audio, sr, title=scene_name, artist=title)
            (folder / "סצנות" / f"{base}.srt").write_text("\n".join(scene_srt), encoding="utf-8")
            m3u += [f"#EXTINF:{round(st)},{scene_name}", f"סצנות/{base}.mp3"]
            full += [scene_audio, story.silence(1.0, sr)]  # a longer breath between scenes
            t += st + 1.0
        write_mp3(folder / f"{safe} - מלא.mp3", np.concatenate(full), sr, title=title)
        (folder / "כתוביות - מלא.srt").write_text("\n".join(srt), encoding="utf-8")
        (folder / "playlist.m3u8").write_text("\n".join(m3u) + "\n", encoding="utf-8")
        job["scenes"] = len(scenes)
        cast_txt = "\n".join(f"{s}: {names.get(v, v)}" for s, v in voices.items())
        (folder / "תסריט.txt").write_text(f"{title}\n\nליהוק:\n{cast_txt}\n\n{script}\n", encoding="utf-8")
        job["full"] = str(folder / f"{safe} - מלא.mp3")

    run_job(job_id, work)
    return {"job": job_id, **JOBS[job_id]}


# --- settings: models for this computer, external LLM, colours ---

_HW = None


def hardware():
    global _HW
    if _HW is None:
        _HW = cfg.scan()
    return _HW


@app.get("/api/settings")
def get_settings():
    return {"settings": cfg.public(SETTINGS), "providers": cfg.PROVIDERS, "version": cfg.VERSION}


@app.get("/api/system")
def system():
    hw = hardware()
    return {"hardware": hw, "models": cfg.catalogue(hw, SETTINGS)}


@app.post("/api/settings")
def set_settings(data: str = Form(...)):
    """Merge the posted sections; reload only the engines whose settings changed."""
    global tts, nakdan, _asr
    new = json.loads(data)
    old = json.loads(json.dumps(SETTINGS))
    for section in ("app", "models", "llm", "theme"):
        if isinstance(new.get(section), dict):
            SETTINGS[section].update(new[section])
    key = SETTINGS["llm"].get("api_key", "")
    if key.startswith("••••"):  # the browser only ever sees the masked key: keep the saved one
        SETTINGS["llm"]["api_key"] = old["llm"]["api_key"]
    reloaded = []
    m, o = SETTINGS["models"], old["models"]
    if SETTINGS["app"]["threads"] != old["app"]["threads"]:
        apply_threads()
        o = {**o, "tts_precision": None}  # rebuild the speech engine with the new thread count
    if m["tts_precision"] != o["tts_precision"] and tts is not None:
        with _tts_lock:
            tts = load_tts(m["tts_precision"])
        reloaded.append("הקראה")
    if m["nikud"] != o["nikud"] and nakdan is not None:
        nakdan = load_nakdan(m["nikud"])
        reloaded.append("ניקוד")
    if m["asr"] != o["asr"] or m["asr_device"] != o["asr_device"]:
        with _asr_lock:
            _asr = None  # loads on next use
        reloaded.append("תמלול")
    cfg.save(SETTINGS)
    return {"ok": True, "reloaded": reloaded, "settings": cfg.public(SETTINGS)}


@app.get("/api/update/check")
def update_check():
    try:
        return cfg.check_update(SETTINGS["app"]["update_repo"], cfg.VERSION)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/models/search")
def models_search(task: str = "", q: str = ""):
    return {"results": cfg.search_models(task, hardware().get("ram_gb") or 8, q)}


@app.post("/api/models/download")
def download_model(model: str = Form(...)):
    if model not in cfg.DOWNLOADS:
        raise HTTPException(400, "המודל הזה לא זמין להורדה מכאן")
    repo, folder = cfg.DOWNLOADS[model]
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"state": "running", "done": 0, "total": 1, "current": repo, "folder": ""}

    def work(job):
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=str(MODELS / folder))
        job["done"] = 1

    run_job(job_id, work)
    return {"job": job_id}


def llm_args(provider: str, base_url: str, api_key: str):
    if not api_key or api_key.startswith("••••"):
        api_key = SETTINGS["llm"]["api_key"] if provider == SETTINGS["llm"]["provider"] else ""
    if not base_url:
        base_url = cfg.PROVIDERS.get(provider, {}).get("base_url", "")
    if not base_url:
        raise HTTPException(400, "חסרה כתובת שרת")
    return provider, base_url, api_key


@app.post("/api/llm/models")
def llm_models(provider: str = Form(...), base_url: str = Form(""), api_key: str = Form("")):
    try:
        return {"models": cfg.list_models(*llm_args(provider, base_url, api_key))}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/llm/test")
def llm_test(provider: str = Form(...), base_url: str = Form(""), api_key: str = Form(""), model: str = Form(...)):
    p, b, k = llm_args(provider, base_url, api_key)
    try:
        reply, secs = cfg.timed(cfg.chat, p, b, k, model, "ענה במשפט אחד בעברית: מה תפקידו של ניקוד בעברית?")
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except (KeyError, IndexError):
        raise HTTPException(400, "השרת ענה בפורמט לא צפוי")
    return {"reply": reply.strip(), "seconds": secs}


@app.get("/api/llm/detect")
def llm_detect():
    return {"found": cfg.detect_local()}


@app.get("/api/file")
def get_file(path: str):
    p = Path(path).resolve()
    if not allowed_path(p) or p.is_dir():
        raise HTTPException(400, "נתיב לא מורשה")
    return FileResponse(p)


# --- transcription ---

@app.post("/api/transcribe")
def transcribe(media: UploadFile = File(...)):
    raw = save_upload(media)
    t0 = time.time()
    try:
        segs, info = transcribe_any(str(raw), language="he", beam_size=5, vad_filter=True)
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
    finally:
        raw.unlink(missing_ok=True)

    def ts(x):
        h, m, s = int(x // 3600), int(x % 3600 // 60), x % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    srt = "\n".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for i, s in enumerate(segs, 1))
    return {"text": " ".join(s["text"] for s in segs), "srt": srt, "segments": segs,
            "audio_seconds": info.duration, "took_seconds": time.time() - t0}


def main(port: int = 8765):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
