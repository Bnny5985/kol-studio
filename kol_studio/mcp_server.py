"""Kol Studio MCP server: lets AI agents (Antigravity, Claude, Cursor…) use Kol Studio through the Model Context Protocol.

It talks to the Kol Studio engine running on this computer (http://127.0.0.1:<port>) and starts the app when it is
not running. Speech, nikud, voices and transcription all stay local.

Run:  kolstudio-mcp                       (installed app, next to the KolStudio executable)
      python -m kol_studio.mcp_server     (development)
Transport: stdio. Nothing but MCP messages may be written to stdout.
"""
import json
import mimetypes
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # its message reaches the agent

from . import paths
from .settings import VERSION

INSTRUCTIONS = """Kol Studio is a Hebrew voice studio that runs on the user's own computer.
Use it to add nikud (Hebrew vowel marks) to text, read Hebrew text aloud in many voices ("characters"),
turn documents into folders of MP3 files, produce multi-character stories with stage directions and subtitles,
transcribe Hebrew audio/video, and create new voices (clone from a recording with consent, or mix two voices).

Tips:
- Call list_characters first to see the voices; pass a character by id or by its Hebrew name.
- Speech follows the nikud exactly. To control pronunciation, call add_nikud, fix the vowels, then speak with auto_nikud=false.
- For stories call story_directions to learn the script format and the available directions.
- Long jobs (export_audio_folder, render_story) may return a job_id: poll job_status until it is done.
- Results are saved as files on the user's computer; tools return their paths.
- Only clone a voice when the user confirms they own it or have the speaker's consent."""

mcp = MCPServer(name="kol-studio", title="קול־סטודיו", instructions=INSTRUCTIONS, version=VERSION)

# ---------- connection to the engine ----------

_base: str | None = None


def _alive(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "api/setup/status", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _from_file() -> str | None:
    try:
        port = json.loads(paths.SERVER_FILE.read_text())["port"]
    except (OSError, ValueError, KeyError):
        return None
    return f"http://127.0.0.1:{port}/"


def _launch():
    """Start Kol Studio: the installed app (with its window), or a headless engine in development."""
    exe = Path(sys.executable)
    flags = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if getattr(sys, "frozen", False):
        app = exe.with_name("KolStudio.exe" if sys.platform.startswith("win") else "KolStudio")
        bundle = next((p for p in exe.parents if p.suffix == ".app"), None)
        if sys.platform == "darwin" and bundle:
            subprocess.Popen(["open", "-g", str(bundle)], **flags)
        else:
            subprocess.Popen([str(app)], **flags, **({"creationflags": 0x00000008} if sys.platform.startswith("win") else {}))
    else:
        from .desktop import free_port

        code = f"from kol_studio.server import main; main({free_port()})"
        subprocess.Popen([sys.executable, "-c", code], **flags, start_new_session=True)


def engine() -> str:
    """URL of a running engine; starts Kol Studio and waits for it when needed."""
    global _base
    for base in (_base, _from_file()):
        if base and _alive(base):
            _base = base
            return base
    _launch()
    end = time.time() + 120
    while time.time() < end:
        time.sleep(1)
        base = _from_file()
        if base and _alive(base):
            _base = base
            return base
    raise ToolError("Kol Studio did not start. Open the app once by hand and try again.")


def _call(method: str, path: str, fields: dict | None = None, files: dict | None = None, timeout: int = 600):
    """HTTP call to the engine. Returns (body bytes, headers). Raises RuntimeError with the engine's message."""
    url = engine() + path.lstrip("/")
    data, headers = None, {}
    if fields is not None or files:
        boundary = uuid.uuid4().hex
        body = bytearray()
        for k, v in (fields or {}).items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, fp in (files or {}).items():
            ctype = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
            body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fp.name}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n").encode()
            body += fp.read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        data, headers = bytes(body), {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except ValueError:
            detail = ""
        raise ToolError(detail or f"Kol Studio error {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise ToolError(f"Could not reach Kol Studio: {e}") from None


def get(path: str):
    return json.loads(_call("GET", path)[0])


def post(path: str, fields: dict | None = None, files: dict | None = None, timeout: int = 600):
    return json.loads(_call("POST", path, fields or {}, files, timeout)[0])


def _file(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_file():
        raise ToolError(f"File not found: {p}")
    return p


def _voice(character: str) -> str:
    """Accept a character id or name (Hebrew or English preset id)."""
    chars = get("api/characters")
    for c in chars:
        if character in (c["id"], c["name"]):
            return c["id"]
    names = ", ".join(c["name"] for c in chars)
    raise ToolError(f"No character called '{character}'. Available: {names}")


def _out_dir(sub: str) -> Path:
    d = Path(get("api/settings")["output_dir"]) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(text: str) -> str:
    words = re.sub(r"[^\w֐-׿ ]", "", re.sub(r"[֑-ׇ]", "", text)).split()[:5]
    return time.strftime("%Y-%m-%d %H-%M-%S") + (" " + " ".join(words) if words else "")


def _wait(job_id: str, limit: float = 150) -> dict:
    """Wait for a background job (inside the client's tool timeout); return its state either way."""
    end = time.time() + limit
    while True:
        job = get(f"api/jobs/{job_id}")
        if job["state"] != "running" or time.time() > end:
            job["job_id"] = job_id
            if job["state"] == "running":
                job["note"] = "Still working. Call job_status with this job_id in a little while."
            return job
        time.sleep(1.5)


# ---------- tools ----------

@mcp.tool()
def kol_status() -> dict:
    """Kol Studio status: version, whether the Hebrew models are installed, the output folder and the voice count.
    סטטוס קול־סטודיו."""
    st, s = get("api/setup/status"), get("api/settings")
    return {"engine": engine(), "version": s["version"], "models_ready": st["ready"],
            "models": {p["id"]: p["installed"] for p in st["packs"]}, "output_dir": s["output_dir"],
            "characters": len(get("api/characters"))}


@mcp.tool()
def list_characters() -> list[dict]:
    """All voices ("characters"): 15 built-in voices plus cloned and mixed ones. Use the id or name in other tools.
    רשימת הדמויות (הקולות)."""
    return [{"id": c["id"], "name": c["name"], "source": c["source"]} for c in get("api/characters")]


@mcp.tool()
def add_nikud(text: str) -> str:
    """Add nikud (Hebrew vowel marks) to Hebrew text. Numbers, dates and times become words first.
    Lines starting with '# ' are kept as part titles. ניקוד טקסט."""
    return post("api/nikud", {"text": text})["text"]


@mcp.tool()
def speak(text: str, character: str = "noa", speed: float = 1.0, auto_nikud: bool = True, file_name: str = "") -> dict:
    """Read Hebrew text aloud in a character's voice and save a WAV file. Returns the file path and duration.
    auto_nikud=false when the text already has the exact nikud you want (speech follows the nikud).
    speed: 0.6–1.6. הקראת טקסט בקול של דמות."""
    body, headers = _call("POST", "api/speak", {"text": text, "character": _voice(character), "speed": speed,
                                                "auto_nikud": str(auto_nikud).lower()})
    name = (re.sub(r"[^\w֐-׿\- ]", "", file_name).strip() or _stamp(text)) + ".wav"
    out = _out_dir("MCP") / name
    out.write_bytes(body)
    return {"file": str(out), "audio_seconds": float(headers.get("X-Audio-Seconds", 0)),
            "synth_seconds": float(headers.get("X-Synth-Seconds", 0))}


@mcp.tool()
def extract_document(file_path: str) -> dict:
    """Read the text of a document: Word (docx), PDF, PowerPoint (pptx), EPUB, txt, md, srt, html.
    Returns the title, the text ('# ' lines are part titles) and the parts with their sentence counts.
    קריאת טקסט ממסמך."""
    j = post("api/extract", files={"file": _file(file_path)})
    return {"title": j["title"], "text": j["text"],
            "parts": [{"title": p["title"], "sentences": len(p["sentences"])} for p in j["parts"]]}


@mcp.tool()
def export_audio_folder(text: str = "", file_path: str = "", title: str = "", character: str = "noa",
                        speed: float = 1.0) -> dict:
    """Turn text or a document into a folder of numbered MP3 files (one per part, '# ' lines start parts),
    with a playlist, a table of contents and the nikud text. Give either text or file_path.
    ייצוא לתיקיית קבצי MP3 עם פלייליסט."""
    if file_path:
        doc = post("api/extract", files={"file": _file(file_path)})
        text, title = text or doc["text"], title or doc["title"]
    if not text.strip():
        raise ToolError("Give text or file_path")
    job = post("api/export", {"text": text, "title": title or "הקראה", "character": _voice(character), "speed": speed})
    return _wait(job["job"])


@mcp.tool()
def story_directions() -> dict:
    """The story script format and every stage direction Kol Studio understands. Read this before render_story.
    הוראות הבימוי ופורמט התסריט."""
    return {
        "format": [
            "# Scene title            -> starts a scene (one MP3 + subtitles per scene)",
            "@Name: text              -> a line spoken by a character",
            "@Name [direction]: text  -> the direction applies to the whole line",
            "[direction] inside text  -> a style applies from there to the end of the line; an event happens there",
            "*word*                   -> emphasis",
            "// comment               -> not read",
            "A row without @ continues the previous speaker.",
        ],
        "example": "# סצנה 1: בספרייה\n@מספר: זה היה ערב שקט.\n@דנה [שמחה]: מצאתי! [צחוק]\n@יוסי [עצב]: באמת? [אנחה] [הפסקה 1] חשבתי שאיבדנו אותו.",
        "directions": get("api/story/catalogue"),
    }


@mcp.tool()
def render_story(script: str, cast: dict[str, str] | None = None, title: str = "סיפור", speed: float = 1.0,
                 gap: float = 0.35) -> dict:
    """Produce a multi-character story: full MP3, one MP3 + SRT subtitles per scene, full subtitles, playlist.
    cast maps each speaker name in the script to a character id or name (missing speakers get default voices).
    הפקת סיפור עם כמה דמויות."""
    resolved = {speaker: _voice(v) for speaker, v in (cast or {}).items()}
    check = post("api/story/parse", {"script": script})
    if not check["lines"]:
        raise ToolError("The script has no lines. " + " · ".join(check["errors"]))
    job = post("api/story/render", {"script": script, "cast": json.dumps(resolved, ensure_ascii=False), "title": title,
                                    "speed": speed, "gap": gap})
    result = _wait(job["job"], limit=160)
    result["script_warnings"] = check["errors"]
    return result


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Progress or result of a long job started by export_audio_folder or render_story. מצב משימה."""
    return _wait(job_id, limit=60)


@mcp.tool()
def transcribe(file_path: str) -> dict:
    """Transcribe Hebrew speech from an audio or video file (ivrit.ai Whisper). Saves SRT subtitles next to the
    other outputs and returns the text and the subtitle path. תמלול הקלטה או סרטון."""
    src = _file(file_path)
    j = post("api/transcribe", files={"media": src}, timeout=3600)
    srt = _out_dir("MCP") / f"{src.stem} - כתוביות.srt"
    srt.write_text(j["srt"], encoding="utf-8")
    return {"text": j["text"], "srt_file": str(srt), "audio_seconds": round(j["audio_seconds"], 1),
            "took_seconds": round(j["took_seconds"], 1)}


@mcp.tool()
def clone_voice(name: str, audio_path: str, owner_consent: bool, transcript: str = "") -> dict:
    """Create a new character from a voice recording (8–10 s of clean speech is best; longer is fine).
    owner_consent must be true ONLY if the user confirmed it is their voice or they have the speaker's consent.
    transcript (optional) = what is said in the recording; used to check recording quality.
    שכפול קול מהקלטה — רק בהסכמת בעל הקול."""
    if not owner_consent:
        raise ToolError("Voice cloning needs the voice owner's consent. Ask the user to confirm first.")
    return post("api/characters/clone", {"name": name, "consent": "true", "transcript": transcript},
                files={"audio": _file(audio_path)})


@mcp.tool()
def mix_voices(name: str, voice_a: str, voice_b: str, ratio: float = 0.5) -> dict:
    """Create a new character by mixing two voices (ratio 0 = all A, 1 = all B). The result belongs to no real person.
    ערבוב שני קולות לדמות חדשה."""
    return post("api/characters/mix", {"name": name, "a": _voice(voice_a), "b": _voice(voice_b), "ratio": ratio})


@mcp.tool()
def open_folder(path: str) -> str:
    """Open a Kol Studio output folder in Finder / Explorer for the user. פתיחת תיקייה."""
    post("api/reveal", {"path": path})
    return "opened"


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
