"""Fallback for English words inside Hebrew text when the English G2P (polyg2p) is not available
(it ships only for macOS and Linux). Acronyms are spelled letter by letter; other words get a rough
Hebrew transliteration, so the sentence is still read instead of failing."""
import re

LETTER_NAMES = {
    "A": "אֵי", "B": "בִּי", "C": "סִי", "D": "דִּי", "E": "אִי", "F": "אֶף", "G": "גִ'י", "H": "אֵייץ'", "I": "אַיי",
    "J": "גֵ'יי", "K": "קֵיי", "L": "אֶל", "M": "אֶם", "N": "אֶן", "O": "אוֹ", "P": "פִּי", "Q": "קְיוּ", "R": "אַר",
    "S": "אֶס", "T": "טִי", "U": "יוּ", "V": "וִי", "W": "דַּאבֶּל יוּ", "X": "אֶקְס", "Y": "וַאי", "Z": "זֶד",
}
# longest first, so "sh" wins over "s"
SOUNDS = [("tion", "שְׁן"), ("wh", "ו"), ("sh", "ש"), ("ch", "צ'"), ("th", "ת'"), ("ph", "פ"), ("ck", "ק"), ("ee", "י"), ("oo", "ו"),
          ("ou", "או"), ("b", "ב"), ("c", "ק"), ("d", "ד"), ("f", "פ"), ("g", "ג"), ("h", "ה"), ("j", "ג'"), ("k", "ק"),
          ("l", "ל"), ("m", "מ"), ("n", "נ"), ("p", "פ"), ("q", "ק"), ("r", "ר"), ("s", "ס"), ("t", "ט"), ("v", "ו"),
          ("w", "ו"), ("x", "קס"), ("y", "י"), ("z", "ז"), ("a", "א"), ("e", "א"), ("i", "י"), ("o", "ו"), ("u", "ו")]
FINALS = str.maketrans({"מ": "ם", "נ": "ן", "צ": "ץ", "פ": "ף", "כ": "ך"})
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _word(w: str) -> str:
    if w.isupper() and len(w) <= 6:  # acronym: MCP -> אֶם סִי פִּי
        return " ".join(LETTER_NAMES[c] for c in w)
    w, out, i = w.lower().replace("'", ""), [], 0
    while i < len(w):
        for src, heb in SOUNDS:
            if w.startswith(src, i):
                # vowels inside a word are mostly left to the nikud model; keep them only at the start
                if src in "aeiou" and i > 0 and src not in "iou":
                    heb = ""
                out.append(heb)
                i += len(src)
                break
        else:
            i += 1
    heb = "".join(out) or w
    return heb[:-1] + heb[-1].translate(FINALS) if heb[-1:] in "מנצפכ" else heb


def to_hebrew(text: str) -> str:
    return LATIN_WORD.sub(lambda m: _word(m.group(0)), text)


def has_latin(text: str) -> bool:
    return bool(LATIN_WORD.search(text))
