"""Multi-character stories with stage directions.

Script format (one spoken line per row):

    # סצנה 1                          -> scene title
    @מספר: היה היה פעם…               -> character line
    @דנה [שמחה]: שלום! [צחוק] איזה יום.
    @יוסי [עצב]: אני לא בטוח… [נשימה] [הפסקה 1.5] אולי מחר.

Tags in square brackets. A *style* tag (שמחה, עצב, לחישה…) applies from where it stands
to the end of the line, or until the next style tag. An *event* tag (הפסקה, נשימה, צחוק…)
happens at its place. `*מילה*` = emphasis.

BlueV3 has no emotion control of its own, so styles are rendered around the model:
tempo, pitch, loudness, punctuation, and a real whisper effect (LPC with noise excitation).
The tag names stay engine-neutral, so a future engine with native emotion tags can take them as they are.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, lfilter, resample_poly, sosfilt

# ---------- the direction catalogue (also sent to the UI) ----------

STYLES = {
    # name: tempo (speed multiplier), pitch (semitones), gain (dB), end mark, effect, description
    "רגיל": dict(tempo=1.0, pitch=0, gain=0, mark=None, fx=None, group="רגש", desc="ביטול סגנון קודם"),
    "שמחה": dict(tempo=1.07, pitch=1.5, gain=1, mark="!", fx=None, group="רגש", desc="מהיר וגבוה מעט"),
    "התרגשות": dict(tempo=1.13, pitch=2.5, gain=2, mark="!", fx=None, group="רגש", desc="מהיר, גבוה ונמרץ"),
    "עצב": dict(tempo=0.86, pitch=-1.5, gain=-3, mark="…", fx=None, group="רגש", desc="איטי, נמוך ושקט"),
    "כעס": dict(tempo=1.08, pitch=0.8, gain=4, mark="!", fx="drive", group="רגש", desc="חזק, חד ומהיר"),
    "פחד": dict(tempo=1.12, pitch=1.8, gain=-1, mark="…", fx="tremble", group="רגש", desc="מהיר ורועד"),
    "רוגע": dict(tempo=0.92, pitch=-0.5, gain=-2, mark=None, fx=None, group="רגש", desc="איטי ורך"),
    "הפתעה": dict(tempo=1.05, pitch=3.0, gain=2, mark="?!", fx=None, group="רגש", desc="גבוה, בסימן שאלה"),
    "לחישה": dict(tempo=0.95, pitch=0, gain=-4, mark=None, fx="whisper", group="אופן דיבור", desc="לחישה אמיתית, בלי קול"),
    "צעקה": dict(tempo=1.04, pitch=2.0, gain=6, mark="!", fx="drive", group="אופן דיבור", desc="חזק מאוד"),
    "שקט": dict(tempo=0.97, pitch=-0.5, gain=-7, mark=None, fx=None, group="אופן דיבור", desc="בעוצמה נמוכה"),
    "איטי": dict(tempo=0.82, pitch=0, gain=0, mark=None, fx=None, group="אופן דיבור", desc="קצב איטי"),
    "מהיר": dict(tempo=1.2, pitch=0, gain=0, mark=None, fx=None, group="אופן דיבור", desc="קצב מהיר"),
    "קול עמוק": dict(tempo=0.97, pitch=-3.0, gain=0, mark=None, fx=None, group="אופן דיבור", desc="נמוך יותר"),
    "קול גבוה": dict(tempo=1.0, pitch=3.0, gain=0, mark=None, fx=None, group="אופן דיבור", desc="גבוה יותר"),
}

EVENTS = {
    "הפסקה קצרה": dict(group="הפסקות", desc="0.3 שניות", insert="[הפסקה 0.3]"),
    "הפסקה": dict(group="הפסקות", desc="0.7 שניות (או [הפסקה 2] לשתי שניות)", insert="[הפסקה 0.7]"),
    "הפסקה ארוכה": dict(group="הפסקות", desc="1.5 שניות", insert="[הפסקה 1.5]"),
    "נשימה": dict(group="קולות", desc="שאיפת אוויר"),
    "אנחה": dict(group="קולות", desc="נשיפה ארוכה"),
    "צחוק": dict(group="קולות", desc="חַה חַה חַה"),
    "צחקוק": dict(group="קולות", desc="הִי הִי"),
    "היסוס": dict(group="קולות", desc="אֶה…"),
    "מממ": dict(group="קולות", desc="מְמְמ… חושב"),
    "בכי": dict(group="קולות", desc="יבבה קצרה (ניסיוני)"),
}

PAUSE_RE = re.compile(r"^הפסקה\s*([\d.]+)\s*(?:ש(?:ניות|')?|s)?$")
TAG_RE = re.compile(r"\[([^\]]+)\]")
LINE_RE = re.compile(r"^@\s*([^:\[\]]+?)\s*(?:\[([^\]]+)\])?\s*:\s*(.*)$")


def catalogue() -> dict:
    groups: dict[str, list] = {}
    for name, s in STYLES.items():
        groups.setdefault(s["group"], []).append({"name": name, "kind": "style", "desc": s["desc"], "insert": f"[{name}]"})
    for name, e in EVENTS.items():
        groups.setdefault(e["group"], []).append({"name": name, "kind": "event", "desc": e["desc"], "insert": e.get("insert", f"[{name}]")})
    groups["הדגשה"] = [{"name": "הדגשת מילה", "kind": "emphasis", "desc": "סמן מילה ואז לחץ: *מילה*", "insert": "**"}]
    return groups


# ---------- parsing ----------

@dataclass
class Piece:
    kind: str  # "text" | "pause" | "event"
    value: str = ""
    seconds: float = 0.0
    style: str = "רגיל"
    emphasis: bool = False


@dataclass
class Line:
    n: int
    row: int  # 1-based row in the script
    speaker: str
    text: str
    pieces: list = field(default_factory=list)
    scene: str = ""


def parse(script: str) -> tuple[list[Line], list[str], list[str]]:
    """Return (lines, speakers in order of appearance, errors)."""
    lines, speakers, errors = [], [], []
    scene, last_speaker = "", None
    for row, raw in enumerate(script.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("# "):
            scene = s[2:].strip()
            continue
        m = LINE_RE.match(s)
        if m:
            speaker, style, text = m.group(1).strip(), (m.group(2) or "רגיל").strip(), m.group(3).strip()
            if style not in STYLES:
                errors.append(f"שורה {row}: סגנון לא מוכר [{style}]")
                style = "רגיל"
            text = f"[{style}] {text}" if style != "רגיל" else text
        elif last_speaker:  # a row without @ continues the previous speaker
            speaker, text = last_speaker, s
        else:
            errors.append(f"שורה {row}: חסר שם דמות בתחילת השורה (למשל @מספר:)")
            continue
        last_speaker = speaker
        if speaker not in speakers:
            speakers.append(speaker)
        pieces, errs = _pieces(text, row)
        errors += errs
        if pieces:
            clean = re.sub(r"\s+([.,!?…:;])", r"\1", re.sub(r"\s{2,}", " ", TAG_RE.sub("", text).replace("*", ""))).strip()
            lines.append(Line(len(lines) + 1, row, speaker, clean, pieces, scene))
    return lines, speakers, errors


def _pieces(text: str, row: int) -> tuple[list[Piece], list[str]]:
    pieces, errors, style = [], [], "רגיל"
    pos = 0
    for m in TAG_RE.finditer(text):
        _add_text(pieces, text[pos:m.start()], style)
        tag = m.group(1).strip()
        pm = PAUSE_RE.match(tag)
        if tag in STYLES:
            style = tag
        elif pm:
            pieces.append(Piece("pause", seconds=min(float(pm.group(1)), 10.0)))
        elif tag in ("הפסקה קצרה", "הפסקה", "הפסקה ארוכה"):
            pieces.append(Piece("pause", seconds={"הפסקה קצרה": 0.3, "הפסקה": 0.7, "הפסקה ארוכה": 1.5}[tag]))
        elif tag in EVENTS:
            pieces.append(Piece("event", value=tag, style=style))
        else:
            errors.append(f"שורה {row}: הוראה לא מוכרת [{tag}]")
        pos = m.end()
    _add_text(pieces, text[pos:], style)
    return pieces, errors


def _add_text(pieces: list, chunk: str, style: str):
    parts = re.split(r"(\*[^*]+\*)", chunk)
    for p in parts:
        if not p.strip():
            continue
        emph = p.startswith("*") and p.endswith("*") and len(p) > 2
        pieces.append(Piece("text", value=p.strip("*").strip() if emph else p.strip(), style=style, emphasis=emph))


# ---------- audio effects ----------

def pitch_shift_keep_length(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Resample-based pitch shift. The caller pre-stretches duration by the same factor via the TTS speed,
    so the length comes out right (see render_text)."""
    if abs(semitones) < 0.05:
        return samples
    ratio = 2 ** (semitones / 12)
    up, down = _ratio(1 / ratio)
    return resample_poly(samples, up, down).astype(np.float32)


def _ratio(x: float, den: int = 200) -> tuple[int, int]:
    from math import gcd

    num = max(1, round(x * den))
    g = gcd(num, den)
    return num // g, den // g


def whisper(samples: np.ndarray, sr: int) -> np.ndarray:
    """Real whisper: keep the vocal-tract filter (LPC) of each frame, replace the voice source with noise."""
    frame, hop, order = int(0.025 * sr), int(0.0125 * sr), 28
    win = np.hanning(frame).astype(np.float64)
    out = np.zeros(len(samples) + frame, dtype=np.float64)
    norm = np.zeros_like(out)
    rng = np.random.default_rng(7)
    x = np.append(samples.astype(np.float64), np.zeros(frame))
    pre = lfilter([1, -0.95], [1], x)  # pre-emphasis
    for start in range(0, len(samples), hop):
        seg = pre[start:start + frame] * win
        energy = np.sqrt(np.mean(seg ** 2)) + 1e-9
        r = np.correlate(seg, seg, "full")[frame - 1:frame + order]
        if r[0] < 1e-9:
            continue
        r[0] *= 1.0001
        try:
            a = solve_toeplitz(r[:order], -r[1:order + 1])
        except Exception:
            continue
        noise = rng.standard_normal(frame)
        y = lfilter([1], np.concatenate([[1], a]), noise) * win
        y *= energy / (np.sqrt(np.mean(y ** 2)) + 1e-9)
        out[start:start + frame] += y * win
        norm[start:start + frame] += win ** 2
    out = out / np.maximum(norm, 1e-3)
    out = lfilter([1], [1, -0.95], out)[: len(samples)]  # de-emphasis
    sos = butter(4, [400, min(9000, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    out = sosfilt(sos, out)
    peak = np.max(np.abs(samples)) + 1e-9
    return (out / (np.max(np.abs(out)) + 1e-9) * peak * 0.8).astype(np.float32)


def drive(samples: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Soft saturation: a harder, more forceful edge for anger and shouting."""
    peak = np.max(np.abs(samples)) + 1e-9
    y = np.tanh(samples / peak * amount) / np.tanh(amount)
    return (y * peak).astype(np.float32)


def tremble(samples: np.ndarray, sr: int, rate: float = 6.5, depth: float = 0.18) -> np.ndarray:
    t = np.arange(len(samples)) / sr
    return (samples * (1 - depth / 2 + depth / 2 * np.sin(2 * np.pi * rate * t))).astype(np.float32)


def breath(sr: int, kind: str = "in", seed: int = 0) -> np.ndarray:
    """A synthetic breath: band-limited noise under a breath-shaped envelope."""
    rng = np.random.default_rng(seed)
    dur = {"in": 0.45, "out": 0.9}[kind]
    n = int(dur * sr)
    noise = rng.standard_normal(n)
    lo, hi = (700, 3500) if kind == "in" else (250, 1800)
    noise = sosfilt(butter(3, [lo, hi], btype="band", fs=sr, output="sos"), noise)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** (1.5 if kind == "in" else 0.8)
    if kind == "out":
        env *= np.exp(-1.8 * t)
    y = noise * env
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.06).astype(np.float32)


def apply_gain(samples: np.ndarray, db: float) -> np.ndarray:
    return (samples * (10 ** (db / 20))).astype(np.float32)


# ---------- rendering ----------

EVENT_TEXT = {  # events voiced by the character itself
    "צחוק": ("חַה חַה חַה!", 1.18, 2.0),
    "צחקוק": ("הִי הִי.", 1.2, 2.5),
    "היסוס": ("אֶה…", 0.8, -0.5),
    "מממ": ("מְמְמ…", 0.75, -1.0),
    "בכי": ("אוּהוּ…", 0.7, 1.5),
}


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def render_text(synth, text: str, style_name: str, emphasis: bool, base_speed: float):
    """synth(pointed_text, speed) -> (samples, sr). Returns (samples, sr)."""
    st = STYLES.get(style_name, STYLES["רגיל"])
    text = text.strip()
    if st["mark"] and text and text[-1] not in "!?…":
        text = text.rstrip(".,;:") + st["mark"]
    tempo = st["tempo"] * (0.85 if emphasis else 1.0)
    ratio = 2 ** (st["pitch"] / 12)
    # Ask the model for a longer/shorter take, then resample: duration ends at speed*tempo, pitch moves by ratio.
    samples, sr = synth(text, base_speed * tempo / ratio)
    samples = pitch_shift_keep_length(samples, st["pitch"])
    if st["fx"] == "whisper":
        samples = whisper(samples, sr)
    elif st["fx"] == "drive":
        samples = drive(samples)
    elif st["fx"] == "tremble":
        samples = tremble(samples, sr)
    samples = apply_gain(samples, st["gain"] + (3 if emphasis else 0))
    return samples, sr


def render_line(synth, line: Line, base_speed: float = 1.0) -> tuple[np.ndarray, int]:
    out, sr = [], 44100
    for i, p in enumerate(line.pieces):
        if p.kind == "pause":
            out.append(silence(p.seconds, sr))
        elif p.kind == "event":
            if p.value in ("נשימה", "אנחה"):
                out.append(breath(sr, "in" if p.value == "נשימה" else "out", seed=line.n * 31 + i))
                out.append(silence(0.08, sr))
            else:
                txt, tempo, pitch = EVENT_TEXT[p.value]
                ratio = 2 ** (pitch / 12)
                s, sr = synth(txt, base_speed * tempo / ratio)
                out.append(apply_gain(pitch_shift_keep_length(s, pitch), -2))
                out.append(silence(0.12, sr))
        else:
            s, sr = render_text(synth, p.value, p.style, p.emphasis, base_speed)
            out.append(s)
    if not out:
        return silence(0.1, sr), sr
    y = np.concatenate(out)
    peak = np.max(np.abs(y))
    if peak > 0.98:
        y = y / peak * 0.98
    return y.astype(np.float32), sr
