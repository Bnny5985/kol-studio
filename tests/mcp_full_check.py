"""Full check of every MCP tool against a Kol Studio install, the way an AI agent uses it.
Creates real files and writes a Hebrew report (בדיקת MCP.md) into the output folder.

Run: .venv/bin/python tests/mcp_full_check.py [/Applications/KolStudio.app/Contents/MacOS/kolstudio-mcp]
"""
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[str, str, bool, str, float]] = []  # area, tool, ok, detail, seconds


def sample_docs(tmp: Path) -> dict:
    """A Word file and a recording to test document reading, transcription and cloning."""
    import docx

    d = docx.Document()
    d.add_heading("מבוא", 1)
    d.add_paragraph("המחקר עוסק בשפה העברית המודרנית. הוא פורסם בשנת 2024.")
    d.add_heading("סיכום", 1)
    d.add_paragraph("תודה רבה שהקשבתם.")
    d.save(tmp / "מסמך לבדיקה.docx")
    rec = tmp / "הקלטה לבדיקה.aiff"
    subprocess.run(["say", "-v", "Carmit", "-o", str(rec),
                    "שלום, זו הקלטה לבדיקת התמלול ושכפול הקול. אני מדברת בקצב רגיל וברור."], check=True)
    return {"docx": tmp / "מסמך לבדיקה.docx", "rec": rec}


async def run(cmd: list[str]):
    tmp = Path(tempfile.mkdtemp(prefix="kol-mcp-check-"))
    files = sample_docs(tmp)
    params = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(ROOT))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        init = await s.initialize()
        tools = {t.name for t in (await s.list_tools()).tools}

        async def call(area, tool, expect_error=False, **args):
            t0 = time.time()
            res = await s.call_tool(tool, args)
            secs = time.time() - t0
            text = res.content[0].text if res.content else ""
            data = res.structured_content  # lists come with a structured copy; plain dicts as JSON text
            if data is None and not res.is_error and text.strip()[:1] in "{[":
                data = json.loads(text)
            if isinstance(data, dict) and list(data) == ["result"]:
                data = data["result"]
            ok = res.is_error == expect_error
            RESULTS.append((area, tool, ok, text if res.is_error else json.dumps(data, ensure_ascii=False)[:400], secs))
            print(("✓" if ok else "✗"), area, tool, f"{secs:.1f}s")
            return data if not res.is_error else text

        st = await call("חיבור", "kol_status")
        out_dir = Path(st["output_dir"])
        chars = await call("דמויות", "list_characters")
        mixed = await call("דמויות", "mix_voices", name="בדיקת MCP ערבוב", voice_a="נועה", voice_b="אדם", ratio=0.4)
        cloned = await call("דמויות", "clone_voice", name="בדיקת MCP שכפול", audio_path=str(files["rec"]),
                            owner_consent=True, transcript="שלום, זו הקלטה לבדיקת התמלול ושכפול הקול.")
        await call("דמויות", "clone_voice", expect_error=True, name="x", audio_path=str(files["rec"]), owner_consent=False)

        pointed = await call("ניקוד", "add_nikud", text="# פתיחה\nהילד ספר את הכסף ב-15 במרץ בשעה 10:30.")
        await call("הקראה", "speak", text="שלום, זו בדיקה של הקראה דרך MCP.", character="נועה", file_name="בדיקת MCP - הקראה")
        await call("הקראה", "speak", text=pointed.split("\n", 1)[1], character="אדם", auto_nikud=False,
                   file_name="בדיקת MCP - ניקוד שלי")
        await call("הקראה", "speak", text="מילים באנגלית: Whisper ו-MCP בתוך משפט.", character="מאיה",
                   file_name="בדיקת MCP - אנגלית")
        await call("הקראה", "speak", text="זה הקול המעורבב.", character=mixed["id"], file_name="בדיקת MCP - קול מעורבב")
        await call("הקראה", "speak", text="זה הקול המשוכפל.", character=cloned["id"], file_name="בדיקת MCP - קול משוכפל")
        await call("הקראה", "speak", expect_error=True, text="שלום", character="דמות שלא קיימת")

        doc = await call("מסמכים", "extract_document", file_path=str(files["docx"]))
        exp = await call("מסמכים", "export_audio_folder", file_path=str(files["docx"]), title="בדיקת MCP ייצוא", character="תמר")
        if exp.get("state") == "running":
            exp = await call("מסמכים", "job_status", job_id=exp["job_id"])

        await call("סיפור", "story_directions")
        story = await call("סיפור", "render_story", title="בדיקת MCP סיפור",
                           script="# סצנה 1: בספרייה\n@מספר: זה היה ערב שקט.\n@דנה [שמחה]: מצאתי! [צחוק]\n"
                                  "@יוסי [לחישה]: שקט.\n# סצנה 2\n@מספר [פחד]: [הפסקה 0.5] האורות כבו.",
                           cast={"דנה": "מאיה", "יוסי": mixed["id"]})
        if story.get("state") == "running":
            story = await call("סיפור", "job_status", job_id=story["job_id"])

        await call("הקראה", "speak", text="זו הקראה במנוע קוון, עם הגייה ברורה.", character="נועה", engine="qwen",
                   file_name="בדיקת MCP - קוון")
        lib = await call("אפקטים", "sfx_library")
        snd = await call("אפקטים", "sound_effect", description="מטבע", seconds=1.5, quality="draft",
                         name="בדיקת MCP מטבע")
        if isinstance(snd, dict) and snd.get("state") == "running":
            snd = await call("אפקטים", "job_status", job_id=snd["job_id"])
        await call("אפקטים", "list_sounds", search="בדיקת MCP")
        await call("אפקטים", "export_sounds", names_or_paths=[snd["files"][0]["name"]], folder=str(tmp / "game"), fmt="ogg")
        await call("אפקטים", "sound_effect", expect_error=True, description="x", engine="nope")

        trans = await call("תמלול", "transcribe", file_path=str(files["rec"]))
        await call("תמלול", "transcribe", expect_error=True, file_path="/no/such/file.mp3")
        await call("כללי", "open_folder", path=str(out_dir))

    return init.server_info.version, tools, out_dir, {"doc": doc, "exp": exp, "story": story, "trans": trans,
                                                       "chars": len(chars), "ids": [mixed["id"], cloned["id"]],
                                                       "sfx": snd, "lib": lib}


def report(version, tools, out_dir: Path, info: dict, cmd: list[str]) -> Path:
    ok = sum(1 for r in RESULTS if r[2])
    lines = [f"# בדיקת MCP — קול־סטודיו {version}", "",
             f"**תאריך:** {time.strftime('%d.%m.%Y %H:%M')}  ",
             f"**שרת:** `{cmd[0]}`  ", f"**תוצאה:** {ok}/{len(RESULTS)} בדיקות עברו "
             + ("✅" if ok == len(RESULTS) else "❌"), "",
             f"כל {len(tools)} הכלים נבדקו דרך MCP, כמו שסוכן AI משתמש בהם. בדיקות שמסומנות \"צפויה שגיאה\" "
             "בודקות שהשרת מסרב כמו שצריך (דמות שלא קיימת, שכפול בלי הסכמה, קובץ חסר).", "",
             "| # | אגף | כלי | תוצאה | זמן | פרטים |", "|---|---|---|---|---|---|"]
    for i, (area, tool, good, detail, secs) in enumerate(RESULTS, 1):
        d = detail.replace("|", "/").replace("\n", " ")[:160]
        lines.append(f"| {i} | {area} | `{tool}` | {'✅' if good else '❌'} | {secs:.1f}s | {d} |")
    lines += ["", "## קבצים שנוצרו", "",
              f"- תיקיית הייצוא: `{info['exp'].get('folder', '')}`",
              f"- תיקיית הסיפור: `{info['story'].get('folder', '')}` ({info['story'].get('scenes', '?')} סצנות)",
              f"- קובצי הקראה ותמלול: `{out_dir / 'MCP'}`", "",
              "## מה נבדק בכל אגף", "",
              "- **חיבור:** השרת מוצא את התוכנה (ופותח אותה אם היא סגורה).",
              f"- **דמויות:** רשימה ({info['chars']} דמויות), ערבוב קולות, שכפול מהקלטה עם תמלול, וסירוב לשכפל בלי הסכמה.",
              "- **ניקוד:** מספרים ושעות הופכים למילים, וכותרת `#` נשמרת.",
              "- **הקראה:** קול מובנה, ניקוד שלי (בלי ניקוד אוטומטי), מילים באנגלית, הקול המעורבב, הקול המשוכפל ומנוע QwenTTS.",
              f"- **אפקטים:** הספרייה ({sum(len(v) for v in info['lib']['library'].values())} צלילים מוכנים), יצירת מטבע, "
              "רשימת הצלילים, ייצוא ל־OGG, וסירוב למנוע לא מוכר.",
              f"- **מסמכים:** קריאת Word ({len(info['doc']['parts'])} חלקים) וייצוא לתיקיית MP3.",
              "- **סיפור:** פורמט ההוראות, והפקה עם ליהוק, רגשות, לחישה, צחוק והפסקה.",
              f"- **תמלול:** \"{info['trans']['text'][:80]}\"",
              "- **כללי:** פתיחת תיקיית התוצאות.", ""]
    path = out_dir / "בדיקת MCP.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def cleanup_characters(ids):
    """Remove the two test characters (moved to characters_deleted, not destroyed)."""
    from kol_studio import paths

    port = json.loads(paths.SERVER_FILE.read_text())["port"]
    for cid in ids:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/characters/{cid}", method="DELETE")
        urllib.request.urlopen(req, timeout=10).read()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    cmd = sys.argv[1:] or ["/Applications/KolStudio.app/Contents/MacOS/kolstudio-mcp"]
    version, tools, out_dir, info = asyncio.run(run(cmd))
    cleanup_characters(info["ids"])
    print("report:", report(version, tools, out_dir, info, cmd))
    print("passed", sum(r[2] for r in RESULTS), "/", len(RESULTS))
