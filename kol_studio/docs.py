"""Extract readable Hebrew text from documents and split it into parts and sentences.

Output text format (shared by the UI and the exporter):
    # part title
    paragraph…

    paragraph…
A line starting with "# " opens a new part.
"""
import html
import re
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

HEB = re.compile(r"[א-ת]")
NIKUD = re.compile(r"[ְ-ׇּׁׂ]")
FINALS_FIRST = re.compile(r"(?:^|\s)[ךםןףץ][א-ת]")  # final letter at word start = reversed text


LTR_RUN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,:/%\-]*[0-9A-Za-z%]|[0-9A-Za-z]")
MIRROR = str.maketrans("()[]{}<>", ")(][}{><")


def _is_visual_order(text: str) -> bool:
    """Final letters (ם ן ץ ף ך) belong at word ends. If they mostly sit at word starts, the text is reversed."""
    at_start = len(FINALS_FIRST.findall(text))
    at_end = len(re.findall(r"[ךםןףץ](?=\s|$|[.,:;!?\"')\]])", text))
    return at_start > at_end


def _fix_rtl_line(line: str) -> str:
    """Flip a character-reversed (visual order) Hebrew line, then restore numbers/English runs."""
    if not HEB.search(line):
        return line
    flipped = line[::-1].translate(MIRROR)
    return LTR_RUN.sub(lambda m: m.group(0)[::-1], flipped)


def _from_pdf(path: Path) -> str:
    from pypdf import PdfReader

    out = []
    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        try:  # layout mode keeps each line intact (visual order), which _fix_rtl_line can reverse exactly
            text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            text = page.extract_text() or ""
        lines = [re.sub(r"\s{2,}", " ", ln.strip()) for ln in text.splitlines() if ln.strip()]
        if _is_visual_order(" ".join(lines)):  # decided per page: one-word lines have no clue of their own
            lines = [_fix_rtl_line(ln) for ln in lines]
        if lines:
            out.append(f"# עמוד {i}\n" + _join_wrapped(lines))
    return "\n\n".join(out)


def _join_wrapped(lines: list[str]) -> str:
    """PDF lines are hard-wrapped: join them, keep a break after sentence ends."""
    paras, cur = [], []
    for ln in lines:
        if not cur and len(ln.split()) <= 6 and not re.search(r"[.!?,:;״\"]$", ln):
            paras.append(ln)  # a short line with no end mark at a paragraph start reads as a heading
            continue
        cur.append(ln)
        if re.search(r"[.!?:״\"]$", ln):
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def _from_docx(path: Path) -> str:
    import docx

    out = []
    for p in docx.Document(str(path)).paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        out.append(f"# {t}" if style.startswith(("heading", "title", "כותרת")) else t)
    return "\n\n".join(out)


def _from_pptx(path: Path) -> str:
    from pptx import Presentation

    out = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(r.text for r in para.runs).strip()
                    if t:
                        texts.append(t)
        if texts:
            out.append(f"# שקופית {i}\n" + "\n\n".join(texts))
    return "\n\n".join(out)


class _Html2Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote"}

    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3"):
            self.parts.append("\n\n# ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        t = html.unescape("".join(self.parts))
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"#\s*\n+", "# ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def _html_to_text(raw: str) -> str:
    p = _Html2Text()
    p.feed(raw)
    return p.text()


def _from_epub(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = container.find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(z.read(opf_path))
        base = str(Path(opf_path).parent)
        items = {i.get("id"): i.get("href") for i in opf.iter("{*}item")}
        out = []
        for ref in opf.iter("{*}itemref"):
            href = items.get(ref.get("idref"))
            if not href:
                continue
            name = f"{base}/{href}" if base not in ("", ".") else href
            try:
                t = _html_to_text(z.read(name).decode("utf-8", "ignore"))
            except KeyError:
                continue
            if t.strip():
                out.append(t if t.lstrip().startswith("#") else f"# פרק {len(out) + 1}\n{t}")
        return "\n\n".join(out)


def _via_textutil(path: Path) -> str:
    """macOS converter for .doc .rtf .odt .html .webarchive."""
    import sys

    if sys.platform != "darwin":
        raise ValueError("קובץ מהסוג הזה נתמך כרגע רק במק. שמור אותו כ־DOCX ונסה שוב")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.txt"
        subprocess.run(["textutil", "-convert", "txt", "-output", str(out), str(path)], check=True, capture_output=True)
        return out.read_text("utf-8", "ignore")


def extract(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        t = _from_pdf(path)
    elif ext == ".docx":
        t = _from_docx(path)
    elif ext == ".pptx":
        t = _from_pptx(path)
    elif ext == ".epub":
        t = _from_epub(path)
    elif ext in (".html", ".htm"):
        t = _html_to_text(path.read_text("utf-8", "ignore"))
    elif ext in (".doc", ".rtf", ".odt", ".webarchive"):
        t = _via_textutil(path)
    else:  # .txt, .md, .srt …
        t = path.read_text("utf-8", "ignore")
        if ext == ".srt":
            t = "\n".join(ln for ln in t.splitlines() if ln.strip() and not ln.strip().isdigit() and "-->" not in ln)
        if ext == ".md":
            t = re.sub(r"^#{1,6}\s*", "# ", t, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def split_parts(text: str, max_chars: int = 1200) -> list[dict]:
    """'# title' lines open parts; long parts are cut at paragraph boundaries into small pieces."""
    parts, title, paras = [], None, []

    def flush():
        if not paras:
            return
        chunk, n = [], 0
        idx = 1
        for p in paras:
            if chunk and n + len(p) > max_chars:
                parts.append({"title": title if idx == 1 else f"{title or 'חלק'} ({idx})", "text": "\n\n".join(chunk)})
                chunk, n, idx = [], 0, idx + 1
            chunk.append(p)
            n += len(p)
        parts.append({"title": title if idx == 1 else f"{title or 'חלק'} ({idx})", "text": "\n\n".join(chunk)})

    for block in re.split(r"\n\s*\n", text):  # a blank line separates paragraphs
        body = []
        for raw in block.splitlines():
            line = raw.strip()
            if line.startswith("# "):
                if body:
                    paras.append(" ".join(body))
                    body = []
                flush()
                title, paras = line[2:].strip(), []
            elif line:
                body.append(line)
        if body:
            paras.append(" ".join(body))
    flush()
    for i, p in enumerate(parts, 1):
        p["title"] = p["title"] or f"חלק {i}"
    return parts


SENT_END = re.compile(r"(?<=[.!?…:;])\s+(?=\S)")


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    out = []
    for para in re.split(r"\n+", text):
        for raw in SENT_END.split(para.strip()):
            rest = raw.strip()
            while len(rest) > max_chars:  # cut very long sentences at a comma (or a space) near the limit
                cut = rest.rfind(",", 0, max_chars)
                cut = cut if cut > 40 else rest.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                out.append(rest[: cut + 1].strip())
                rest = rest[cut + 1:].strip()
            if re.search(r"\w", rest):
                out.append(rest)
    return out


def is_pointed(text: str) -> bool:
    letters = len(HEB.findall(text))
    return letters > 0 and len(NIKUD.findall(text)) / letters > 0.35
