"""The sound-effect library: ready-made Hebrew buttons with tested English prompts, styles,
and turning a free Hebrew description into the English prompt the models understand."""
import re

# category -> [(Hebrew label, English prompt, seconds)]
LIBRARY = {
    "משחק": [
        ("מטבע", "a bright short coin pickup chime, video game", 1.0),
        ("קפיצה", "a short springy jump sound effect, video game", 1.0),
        ("ניצחון", "a cheerful short victory fanfare jingle, video game", 3.0),
        ("הפסד", "a sad descending game over jingle, video game", 3.0),
        ("עלייה בשלב", "a magical rising level up chime with sparkles", 2.0),
        ("לחיצת כפתור", "a soft clean user interface button click", 0.5),
        ("פתיחת תיבה", "a wooden treasure chest creaking open with a magical sparkle", 2.5),
        ("תשובה נכונה", "a happy two-note correct answer ding", 1.0),
        ("תשובה שגויה", "a short low buzzer for a wrong answer", 1.0),
        ("קסם", "a shimmering magic spell with twinkling sparkles", 2.5),
        ("פיצוץ", "a big cartoon explosion boom", 2.0),
        ("לייזר", "a retro sci-fi laser zap", 1.0),
        ("איסוף כוכב", "a sparkling star collect sound, video game", 1.0),
        ("צעדים", "footsteps walking on a wooden floor", 4.0),
    ],
    "טבע": [
        ("גשם", "steady rain falling on leaves, ambience", 10.0),
        ("סערה ורעם", "a rolling thunderstorm with heavy rain and distant thunder", 10.0),
        ("רוח", "wind blowing through trees, ambience", 10.0),
        ("ים וגלים", "ocean waves gently crashing on a sandy beach", 10.0),
        ("נחל", "a small stream babbling over rocks", 10.0),
        ("ציפורים בבוקר", "birds chirping in a forest in the early morning", 10.0),
        ("צרצרים בלילה", "crickets chirping on a quiet summer night", 10.0),
        ("מדורה", "a crackling campfire", 10.0),
    ],
    "בעלי חיים": [
        ("כלב נובח", "a dog barking twice", 2.0),
        ("חתול מיילל", "a cat meowing", 2.0),
        ("תרנגול", "a rooster crowing in the morning", 3.0),
        ("פרה", "a cow mooing", 2.5),
        ("סוס דוהר", "a horse galloping on a dirt road", 4.0),
        ("אריה שואג", "a lion roaring", 3.0),
        ("ינשוף", "an owl hooting at night", 3.0),
        ("זבוב", "a fly buzzing around", 3.0),
    ],
    "בית ועיר": [
        ("דפיקה בדלת", "three knocks on a wooden door", 2.0),
        ("דלת חורקת", "an old wooden door slowly creaking open", 3.0),
        ("טריקת דלת", "a door slamming shut", 1.5),
        ("פעמון בדלת", "a doorbell ringing", 2.0),
        ("טלפון מצלצל", "an old telephone ringing", 4.0),
        ("שעון מתקתק", "a clock ticking", 5.0),
        ("מים מהברז", "water running from a kitchen tap", 4.0),
        ("מכונית חולפת", "a car passing by on a street", 4.0),
        ("צופר", "a car horn honking twice", 2.0),
        ("רכבת", "a train passing with its horn", 8.0),
        ("רחוב הומה", "a busy city street ambience with traffic and people", 10.0),
        ("כיתה", "a classroom ambience with children chatting quietly", 10.0),
    ],
    "אנשים": [
        ("מחיאות כפיים", "a small crowd clapping and applauding", 4.0),
        ("קהל צוחק", "an audience laughing", 4.0),
        ("קהל מריע", "a crowd cheering", 4.0),
        ("לב פועם", "a heartbeat, slow and steady", 5.0),
        ("לגימה", "a person sipping a drink", 1.5),
        ("פיהוק", "a person yawning", 2.0),
        ("שריקה", "a person whistling a short tune", 3.0),
    ],
    "מתח ופחד": [
        ("רוח רפאים", "a ghostly whispering wind, eerie", 5.0),
        ("צעדים בחושך", "slow footsteps in an empty hallway, tense", 5.0),
        ("גונג דרמטי", "a deep dramatic gong hit", 4.0),
        ("מתח עולה", "a rising tension drone, cinematic", 6.0),
        ("שרשראות", "heavy metal chains rattling", 3.0),
        ("זאב מיילל", "a wolf howling at night", 4.0),
    ],
    "מעברים": [
        ("ווש", "a fast whoosh transition", 1.0),
        ("ווש איטי", "a slow airy whoosh transition", 2.0),
        ("דפדוף", "a page turning in a book", 1.0),
        ("פופ", "a soft cartoon pop", 0.5),
        ("פעמון", "a single clear bell ding", 2.0),
        ("תיפוף", "a short drum roll ending with a cymbal crash", 3.0),
    ],
}

STYLES = {  # Hebrew label -> words added to the prompt
    "רגיל": "",
    "מציאותי": "realistic, high quality field recording",
    "קולנועי": "cinematic, dramatic, high fidelity",
    "מצויר (קרטון)": "cartoon style, playful",
    "משחק 8-ביט": "8-bit retro chiptune video game style",
    "רחוק": "heard from far away, distant",
    "קרוב": "close-up, intimate, detailed",
}

# A small dictionary for when no language model is set up: common Hebrew sound words -> English
WORDS = {
    "גשם": "rain", "רעם": "thunder", "ברק": "lightning", "סערה": "storm", "רוח": "wind", "ים": "sea", "גלים": "waves",
    "נחל": "stream", "מים": "water", "אש": "fire", "מדורה": "campfire", "יער": "forest", "לילה": "night",
    "בוקר": "morning", "ציפורים": "birds", "ציפור": "bird", "כלב": "dog", "חתול": "cat", "סוס": "horse",
    "אריה": "lion", "זאב": "wolf", "פרה": "cow", "תרנגול": "rooster", "ינשוף": "owl", "דבורה": "bee",
    "זבוב": "fly", "צפרדע": "frog", "דלת": "door", "חלון": "window", "פעמון": "bell", "שעון": "clock",
    "טלפון": "phone", "מכונית": "car", "רכבת": "train", "מטוס": "airplane", "אופנוע": "motorcycle",
    "צעדים": "footsteps", "צעד": "footstep", "מחיאות": "clapping", "כפיים": "hands", "קהל": "crowd",
    "צחוק": "laughter", "בכי": "crying", "צעקה": "scream", "לב": "heart", "דפיקה": "knock", "דפיקות": "knocks",
    "חורקת": "creaking", "חורק": "creaking", "טריקה": "slam", "נובח": "barking", "נביחה": "bark",
    "מיילל": "meowing", "שואג": "roaring", "פיצוץ": "explosion", "יריה": "shot", "חרב": "sword",
    "קסם": "magic", "מטבע": "coin", "קפיצה": "jump", "ניצחון": "victory", "לייזר": "laser", "חללית": "spaceship",
    "רובוט": "robot", "מפלצת": "monster", "רפאים": "ghost", "שרשראות": "chains", "זכוכית": "glass",
    "נשבר": "breaking", "נשברת": "breaking", "מתכת": "metal", "עץ": "wood", "אבן": "stone", "נייר": "paper",
    "ספר": "book", "דף": "page", "כיתה": "classroom", "ילדים": "children", "רחוב": "street", "עיר": "city",
    "שוק": "market", "מטבח": "kitchen", "ברז": "tap", "מקלחת": "shower", "שקט": "quiet", "חזק": "loud",
    "רחוק": "distant", "קרוב": "close", "ישנה": "old", "ישן": "old", "גדול": "big", "קטן": "small",
    "מהיר": "fast", "איטי": "slow", "מפחיד": "scary", "שמח": "happy", "עצוב": "sad", "טירה": "castle",
    "מערה": "cave", "מנהרה": "tunnel", "חלל": "space", "מוזיקה": "music", "תוף": "drum", "תופים": "drums",
    "גיטרה": "guitar", "פסנתר": "piano", "שריקה": "whistle", "צופר": "horn", "אזעקה": "siren",
    "מנוע": "engine", "מסוק": "helicopter", "סירה": "boat", "שלג": "snow", "קרח": "ice", "בוץ": "mud",
    "חול": "sand", "עלים": "leaves", "על": "on", "ב": "in", "של": "of", "עם": "with",
}

HEBREW = re.compile(r"[֐-׿]")


def catalogue() -> dict:
    return {"library": {cat: [{"label": h, "prompt": p, "seconds": s} for h, p, s in items]
                        for cat, items in LIBRARY.items()},
            "styles": [{"label": k, "words": v} for k, v in STYLES.items()]}


def find_preset(label: str) -> tuple[str, float] | None:
    for items in LIBRARY.values():
        for h, p, s in items:
            if h == label.strip():
                return p, s
    return None


def dictionary_translate(text: str) -> str:
    """Word-by-word fallback: good enough for short descriptions like "כלב נובח ביער בלילה"."""
    words = []
    for w in re.findall(r"[\w֐-׿'\-]+", text):
        if not HEBREW.search(w):
            words.append(w)
            continue
        key = w
        if key not in WORDS and len(key) > 2 and key[0] in "הובלמש":  # strip a one-letter prefix (ה, ו, ב, ל, מ, ש)
            key = key[1:]
        if key in WORDS:
            words.append(WORDS[key])
    return " ".join(words)


LLM_PROMPT = ("Turn this Hebrew description of a sound effect into one short English prompt for a text-to-audio "
              "model. Describe only the sound itself (source, action, place, mood), no music unless asked, no "
              "quotation marks, at most 25 words. Answer with the English prompt only.\n\nHebrew: {text}")


def to_prompt(text: str, style: str = "", llm=None) -> tuple[str, str]:
    """(English prompt, how it was made). llm(prompt) -> str is used when a language model is set up."""
    text = text.strip()
    preset = find_preset(text)
    if preset:
        base, how = preset[0], "ספרייה"
    elif not HEBREW.search(text):
        base, how = text, "כמו שכתבת"
    else:
        base, how = "", ""
        if llm is not None:
            try:
                base = llm(LLM_PROMPT.format(text=text)).strip().strip('"').splitlines()[0]
                how = "מודל השפה"
            except Exception:  # fall back to the dictionary
                base = ""
        if not base:
            base, how = dictionary_translate(text), "מילון מובנה"
        if not base:
            raise ValueError("לא הצלחתי להבין את התיאור. אפשר לכתוב אותו באנגלית, או להגדיר מודל שפה בהגדרות")
    words = STYLES.get(style, "")
    return (f"{base}, {words}" if words else base), how
