"""Fast tests that need no models: text handling, stories, effects, settings and server helpers.

Run: .venv/bin/pytest -q
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("KOL_DATA_DIR", tempfile.mkdtemp(prefix="kol-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kol_studio import docs, models_setup, settings, story  # noqa: E402


# ---------- documents ----------

def test_parts_follow_headings_and_split_long_sections():
    text = "# מבוא\nפסקה ראשונה.\n\nפסקה שנייה.\n\n# סיכום\n" + "\n\n".join(["משפט ארוך מאוד. " * 20] * 3)
    parts = docs.split_parts(text, max_chars=400)
    assert parts[0]["title"] == "מבוא"
    assert "פסקה ראשונה." in parts[0]["text"] and "פסקה שנייה." in parts[0]["text"]
    assert [p["title"] for p in parts[1:]] == ["סיכום", "סיכום (2)", "סיכום (3)"]


def test_text_without_headings_gets_numbered_parts():
    assert docs.split_parts("שלום.\n\nעולם.")[0]["title"] == "חלק 1"


def test_sentences_split_and_long_ones_are_cut():
    assert docs.split_sentences("שלום לכם. מה שלומכם? טוב!") == ["שלום לכם.", "מה שלומכם?", "טוב!"]
    long = "מילה " * 120
    pieces = docs.split_sentences(long, max_chars=100)
    assert all(len(p) <= 101 for p in pieces) and len(pieces) > 4


def test_sentence_without_spaces_does_not_loop_forever():
    pieces = docs.split_sentences("א" * 500, max_chars=100)
    assert "".join(pieces) == "א" * 500


def test_pointed_detection():
    assert docs.is_pointed("הַיֶּלֶד סָפַר אֶת הַכֶּסֶף")
    assert not docs.is_pointed("הילד ספר את הכסף")


def test_reversed_pdf_line_is_restored():
    # a real line from a Chrome-printed Hebrew PDF, as pypdf's layout mode returns it
    visual = ".לולמתל Whisper-ב שומיש השענ .2024-ב םיפתתשמ 150 ללכ (םילשוריב ךרענש) רקחמה"
    assert docs._is_visual_order(visual)
    assert docs._fix_rtl_line(visual) == "המחקר (שנערך בירושלים) כלל 150 משתתפים ב-2024. נעשה שימוש ב-Whisper לתמלול."


def test_logical_text_is_left_alone():
    assert not docs._is_visual_order("שלום עולם, זה טקסט רגיל לגמרי בשנת 2024.")


# ---------- stories ----------

SCRIPT = """# סצנה 1: בספרייה
@מספר: היה ערב שקט.
@דנה [שמחה]: מצאתי! [צחוק] *באמת*.
המשך של דנה [הפסקה 1.5] וסוף.
@יוסי [לא-קיים]: שורה.
// הערה שלא נקראת
בלי דובר בהתחלה? לא, זה המשך של יוסי.
"""


def test_story_parse_speakers_scenes_and_errors():
    lines, speakers, errors = story.parse(SCRIPT)
    assert speakers == ["מספר", "דנה", "יוסי"]
    assert [ln.speaker for ln in lines] == ["מספר", "דנה", "דנה", "יוסי", "יוסי"]
    assert all(ln.scene == "סצנה 1: בספרייה" for ln in lines)
    assert any("לא-קיים" in e for e in errors)
    assert lines[1].text == "מצאתי! באמת."  # tags and * removed, no stray spaces


def test_story_pieces():
    lines, _, _ = story.parse(SCRIPT)
    kinds = [(p.kind, p.style, p.emphasis) for p in lines[1].pieces]
    assert kinds[0] == ("text", "שמחה", False)
    assert ("event", "שמחה", False) in kinds
    assert ("text", "שמחה", True) in kinds
    pause = [p for p in lines[2].pieces if p.kind == "pause"][0]
    assert pause.seconds == 1.5


def test_first_line_without_speaker_is_an_error():
    _, _, errors = story.parse("סתם שורה")
    assert errors and "חסר שם דמות" in errors[0]


def test_catalogue_inserts_parse_back():
    for items in story.catalogue().values():
        for it in items:
            if it["kind"] in ("style", "event"):
                _, _, errors = story.parse(f"@א: שלום {it['insert']} עולם")
                assert not errors, it


def fake_synth(text, speed):
    sr = 44100
    return np.sin(np.linspace(0, 300, int(sr * 0.5 / speed))).astype(np.float32) * 0.5, sr


def test_render_line_produces_audio_for_every_piece():
    lines, _, _ = story.parse("@א [לחישה]: שלום [נשימה] [הפסקה 0.5] [צחוק] [כעס] עולם")
    samples, sr = story.render_line(fake_synth, lines[0])
    assert sr == 44100 and len(samples) > sr * 1.5 and np.max(np.abs(samples)) <= 0.98


def test_pitch_shift_with_tempo_keeps_length():
    # render_text asks the model for a take that is longer by the pitch ratio, then resamples it back
    plain, _ = story.render_text(fake_synth, "שלום", "רגיל", False, 1.0)
    high, _ = story.render_text(fake_synth, "שלום", "קול גבוה", False, 1.0)
    assert abs(len(high) - len(plain)) / len(plain) < 0.02


def test_whisper_keeps_length_and_level():
    x = fake_synth("", 1.0)[0]
    y = story.whisper(x, 44100)
    assert len(y) == len(x) and 0 < np.max(np.abs(y)) <= np.max(np.abs(x))


# ---------- settings ----------

def test_versions_compare():
    assert settings._ver("v0.2.0") > settings._ver("0.1.9")
    assert settings._ver("v1.0") == (1, 0, 0)


def test_key_is_masked_and_load_drops_retired_keys(tmp_path, monkeypatch):
    f = tmp_path / "s.json"
    f.write_text('{"app": {"threads": 4, "default_voice": "noa"}, "llm": {"api_key": "sk-123456"}}')
    monkeypatch.setattr(settings, "FILE", f)
    s = settings.load()
    assert s["app"]["threads"] == 4 and "default_voice" not in s["app"]
    assert settings.public(s)["llm"]["api_key"] == "••••3456"


def test_catalogue_recommends_by_memory():
    cur = settings.load()
    small = {m["id"]: m for m in settings.catalogue({"ram_gb": 16}, cur)}
    big = {m["id"]: m for m in settings.catalogue({"ram_gb": 64}, cur)}
    assert small["ivrit-turbo"]["recommended"] and big["ivrit-large"]["recommended"]
    assert not small["dictalm3-24b"]["fits"] and big["dictalm3-24b"]["fits"]


def test_update_check_without_repo():
    assert settings.check_update("", "0.1.0")["status"] == "no-repo"


def test_asr_dir_falls_back_to_installed_model():
    assert models_setup.asr_dir("ivrit-large").name == "ivrit-turbo-ct2"  # large is not installed in the test dir


# ---------- server helpers ----------

@pytest.fixture(scope="module")
def server():
    pytest.importorskip("blue_onnx")
    from kol_studio import server as srv

    return srv


def test_character_ids_are_validated(server):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        server.voice_payload("../../etc/passwd")
    assert e.value.status_code == 400


def test_new_folder_never_reuses_a_name(server, tmp_path, monkeypatch):
    monkeypatch.setitem(server.SETTINGS["app"], "output_dir", str(tmp_path))
    a = server.new_folder("בדיקה/../", "x")
    a.mkdir()
    b = server.new_folder("בדיקה/../", "x")
    assert a != b and a.parent == tmp_path and "/" not in a.name.replace(str(tmp_path), "")


def test_best_window_prefers_dense_speech(server):
    seg = lambda s, e: SimpleNamespace(start=s, end=e, text="x")  # noqa: E731
    segs = [seg(3.0, 6.0), seg(6.2, 9.0), seg(9.1, 12.0), seg(20.0, 21.0)]
    start, end, used = server.best_window(25.0, segs, 9.0)
    assert 2.5 <= start <= 3.0 and end - start <= 9.6 and len(used) == 3


def test_safe_name_and_srt_time(server):
    assert server.safe_name('a/b:c*"', "x") == "abc"
    assert server.srt_time(3725.5) == "01:02:05,500"


def test_transcript_match_handles_partial_and_wrong_text(server):
    heard = "שלום, אני מקליטה את הקול שלי לצורך יצירת דמות. אני מדברת בקצב רגיל וברור."
    assert server.transcript_match("שלום אני מקליטה את הקול שלי", heard) == 1.0   # only part of the recording
    assert server.transcript_match("היום בבוקר נסענו לים עם המשפחה", heard) < 0.3  # a different text
    assert server.transcript_match("", heard) == 0.0


def test_speech_without_models_gives_clear_message(server, tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(server, "tts", None)
    monkeypatch.setattr(server.models_setup, "ready", lambda: False)
    with pytest.raises(HTTPException) as e:
        server.synth("שָׁלוֹם", "noa", 1.0)
    assert e.value.status_code == 409


def test_nikud_keeps_part_titles(server, monkeypatch):
    fake = SimpleNamespace(add_diacritics=lambda t: t + "ַ")
    monkeypatch.setattr(server, "engines", lambda: (None, fake))
    out = server.add_nikud("# פתיחה\nשלום")
    assert out.splitlines()[0].startswith("# ") and len(docs.split_parts(out)) == 1
    assert docs.split_parts(out)[0]["title"].startswith("פתיחה")


def test_latin_fallback_spells_acronyms_and_transliterates():
    from kol_studio import latin

    assert latin.to_hebrew("דרך MCP.") == "דרך אֶם סִי פִּי."
    out = latin.to_hebrew("עם Whisper בתוך")
    assert not latin.has_latin(out) and out.startswith("עם ו")
    assert latin.to_hebrew("בלי אנגלית") == "בלי אנגלית"


def test_mcp_server_exposes_the_tools():
    import asyncio

    pytest.importorskip("mcp")
    from kol_studio import mcp_server

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"kol_status", "list_characters", "add_nikud", "speak", "render_story", "transcribe", "clone_voice",
            "sound_effect", "sfx_library", "list_sounds", "export_sounds"} <= names


def test_mcp_entry_points_at_this_python_in_development(server):
    entry = server.mcp_entry()
    assert entry["args"] == ["-m", "kol_studio.mcp_server"] and entry["command"]


def test_fish_segments_map_directions_to_tags():
    lines, _, _ = story.parse("@א [שמחה]: שלום [צחוק] [הפסקה 0.5] [לחישה] *בשקט* עכשיו")
    segs = story.fish_segments(lines[0])
    assert segs[0] == ("text", "[happy] שלום [laughing]")
    assert segs[1] == ("pause", 0.5)
    assert segs[2][0] == "text" and segs[2][1].startswith("[whisper]") and "[emphasis] בשקט" in segs[2][1]


def test_fish_nikud_keeps_tags(server, monkeypatch):
    monkeypatch.setattr(server, "add_nikud", lambda t: t.replace("שלום", "שָׁלוֹם"))
    assert server.nikud_keep_tags("[laughing] שלום [whisper]") == "[laughing] שָׁלוֹם [whisper]"


# ---------- sound effects ----------

def test_sfx_tags_parse_and_render():
    lines, _, errors = story.parse("@א: שלום [אפקט: מטבע] עולם\n@א: [רקע: גשם]")
    assert not errors
    kinds = [p.kind for p in lines[0].pieces]
    assert kinds == ["text", "sfx", "text"] and lines[0].pieces[1].value == "מטבע"
    assert lines[1].pieces[0].kind == "bg" and lines[1].text == ""
    calls = []

    def sfx(name, sr):
        calls.append(name)
        return np.ones(sr // 2, dtype=np.float32) * 0.1

    samples, sr = story.render_line(fake_synth, lines[0], sfx=sfx)
    assert calls == ["מטבע"] and len(samples) > sr * 1.4
    assert ("sfx", "מטבע") in story.fish_segments(lines[0])


def test_sfx_prompt_from_library_dictionary_and_llm():
    from kol_studio import sfx_library as lib

    assert lib.to_prompt("מטבע")[0].startswith("a bright short coin")
    assert lib.to_prompt("rain on a tin roof", "קולנועי")[0] == "rain on a tin roof, cinematic, dramatic, high fidelity"
    prompt, how = lib.to_prompt("כלב נובח ביער בלילה")
    assert how == "מילון מובנה" and prompt == "dog barking forest night"
    assert lib.to_prompt("כלב", llm=lambda p: '"a dog barking"\n')[0] == "a dog barking"
    assert lib.to_prompt("כלב", llm=lambda p: 1 / 0)[1] == "מילון מובנה"  # a failing model falls back
    with pytest.raises(ValueError):
        lib.to_prompt("קשקשת מבולבלת")


def test_sfx_edit_trims_fades_and_loops(server):
    sr = 1000
    y = np.ones((3000, 1), dtype=np.float32) * 0.5
    out = server.edit_audio(y, sr, 0.5, 2.5, 0, 0.1, 0.2, False, True)
    assert len(out) == 2000 and out[0, 0] == 0 and abs(out[1000, 0] - 0.89) < 1e-3 and out[-1, 0] < 0.01
    looped = server.edit_audio(y, sr, 0, 0, -6, 0, 0, True, False)
    assert len(looped) == 3000 - 300 and np.allclose(looped, 0.5 * 10 ** (-6 / 20), atol=1e-3)


def test_character_engine_choice(server, tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(server, "CHAR_ENGINES", tmp_path / "eng.json")
    monkeypatch.setattr(server, "voice_payload", lambda cid: {})
    monkeypatch.setattr(server.fish_engine, "installed", lambda: False)
    monkeypatch.setattr(server.qwen_engine, "installed", lambda: False)
    assert server.engine_for("noa") == "blue"
    with pytest.raises(HTTPException):
        server.set_character_engine("noa", "fish")  # not installed
    with pytest.raises(HTTPException):
        server.engine_for("noa", "fish")
    monkeypatch.setattr(server.fish_engine, "installed", lambda: True)
    server.set_character_engine("noa", "fish")
    assert server.engine_for("noa") == "fish" and server.engine_for("noa", "blue") == "blue"
    assert server.engine_for("adam") == "blue"
    assert {c["id"]: c["engine"] for c in server.characters()}["noa"] == "fish"


def test_sfx_engines_listed_in_catalogue():
    hw = {"ram_gb": 64}
    rows = [m for m in settings.catalogue(hw, settings.load()) if m["category"] == "sfx"]
    assert {m["id"] for m in rows} == {"sfx-moss", "sfx-sa3"} and all(m["available"] for m in rows)
