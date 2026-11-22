"""Kol Studio desktop app: starts the local server and shows it in a native window.

Falls back to the default web browser when no native web view is available.
"""
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

TITLE = "קול־סטודיו"

SPLASH = """<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<body style="margin:0;height:100vh;display:grid;place-items:center;background:#151517;color:#ececef;
font:20px/1.6 -apple-system,'Segoe UI',Arial,sans-serif">
<div style="text-align:center"><div style="font-size:30px;font-weight:700">קול־סטודיו</div>
<div style="color:#9a9aa2">טוען את המנועים…</div></div></body></html>"""


def free_port(preferred: int = 8765) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def start_server(port: int):
    from .server import main as serve  # heavy imports happen here, after the window is up

    threading.Thread(target=serve, args=(port,), daemon=True).start()


def wait_ready(url: str, timeout: float = 120) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "api/setup/status", timeout=2):
                return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    try:
        import webview
    except ImportError:
        webview = None

    if webview is None or "--browser" in sys.argv:
        start_server(port)
        if wait_ready(url):
            webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    def boot(window):
        start_server(port)
        if wait_ready(url):
            window.load_url(url)
        else:
            window.load_html("<h2 dir='rtl'>המנוע לא עלה. נסה לפתוח שוב.</h2>")

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    window = webview.create_window(TITLE, html=SPLASH, width=1320, height=900, min_size=(900, 620))
    webview.start(boot, window)


if __name__ == "__main__":
    main()
