"""End-to-end check of the MCP server with a real MCP client over stdio (needs the models installed).

Run: .venv/bin/python tests/mcp_smoke.py [command ...]
Default command: this venv's python -m kol_studio.mcp_server
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def main():
    cmd = sys.argv[1:] or [sys.executable, "-m", "kol_studio.mcp_server"]
    params = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(ROOT))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        info = await s.initialize()
        print("server:", info.server_info.name, info.server_info.version)
        tools = (await s.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))

        async def call(tool, **args):
            t0 = time.time()
            res = await s.call_tool(tool, args)
            text = res.content[0].text if res.content else ""
            status = "ERROR" if res.is_error else "ok"
            print(f"\n[{status}] {tool} ({time.time() - t0:.1f}s): {text[:300]}")
            if res.is_error:
                return text
            data = res.structured_content  # the full result (lists arrive as one text block per item)
            if data is None:
                return json.loads(text) if text.strip()[:1] in "{[" else text
            return data["result"] if list(data) == ["result"] else data

        await call("kol_status")
        chars = await call("list_characters")
        assert any(c["id"] == "noa" for c in chars)
        await call("add_nikud", text="הילד ספר את הכסף בשעה 10:30.")
        spoken = await call("speak", text="שלום, אני מדברת דרך MCP.", character="נועה")
        assert Path(spoken["file"]).stat().st_size > 10000
        await call("story_directions")
        story = await call("render_story", script="# סצנה 1\n@מספר: שלום.\n@דנה [שמחה]: מצאתי! [צחוק]",
                           cast={"דנה": "מאיה"}, title="בדיקת MCP")
        assert story["state"] == "done", story
        exp = await call("export_audio_folder", text="# א\nשלום.\n\n# ב\nלהתראות.", title="ייצוא MCP")
        assert exp["state"] == "done", exp
        bad = await call("speak", text="שלום", character="לא-קיים")
        assert "Available" in bad, bad  # the agent sees the list of names
        refused = await call("clone_voice", name="x", audio_path="/nope.wav", owner_consent=False)
        assert "consent" in refused, refused
        eng = await call("speak", text="שלום, אני מדברת דרך MCP עם Whisper.", character="noa")
        assert Path(eng["file"]).stat().st_size > 10000  # English words inside Hebrew text
        print("\nALL OK")
        return story["folder"], exp["folder"]


if __name__ == "__main__":
    print(asyncio.run(main()))
