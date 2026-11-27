"""Shared helpers for the optional engines that live in their own Python environment (Fish, sound effects).

Each engine gets <data>/engines/<name>/venv built with uv, so its PyTorch stack never touches the app.
"""
import io
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from . import paths

BIN = paths.DATA / "engines" / "bin"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def gpu() -> str:
    if sys.platform == "darwin":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def uv(job: dict) -> str:
    """uv (MIT/Apache) builds the engines' Python environments; downloaded once if it is not on this computer."""
    legacy = paths.DATA / "engines" / "fish" / "bin"  # where versions up to 0.3.0 kept it
    found = shutil.which("uv") or shutil.which("uv", path=f"{BIN}{':' if sys.platform != 'win32' else ';'}{legacy}")
    if found:
        return found
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}[platform.machine()]
    target = {"darwin": f"{arch}-apple-darwin", "win32": f"{arch}-pc-windows-msvc"}.get(sys.platform,
                                                                                       f"{arch}-unknown-linux-gnu")
    ext = "zip" if sys.platform.startswith("win") else "tar.gz"
    url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.{ext}"
    job["current"] = "מוריד את uv"
    data = urllib.request.urlopen(url, timeout=120).read()
    BIN.mkdir(parents=True, exist_ok=True)
    name = "uv.exe" if sys.platform.startswith("win") else "uv"
    if ext == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(m for m in z.namelist() if m.endswith(name))
            (BIN / name).write_bytes(z.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("/" + name) or m.name == name)
            (BIN / name).write_bytes(t.extractfile(member).read())
    (BIN / name).chmod(0o755)
    return str(BIN / name)


def run(cmd: list, job: dict, what: str, cwd: Path | None = None, env: dict | None = None):
    job["current"] = what
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, check=False, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{what} נכשל: {(r.stderr or r.stdout)[-600:]}")


def has_module(python: Path, module: str) -> bool:
    return subprocess.run([str(python), "-c", f"import {module}"], capture_output=True, check=False).returncode == 0


def fetch_code(owner_repo: str, commit: str, dest: Path, job: dict, what: str, subdir: str = ""):
    """Download a pinned commit of a GitHub repository (optionally only one sub-folder) into dest."""
    job["current"] = what
    data = urllib.request.urlopen(f"https://codeload.github.com/{owner_repo}/zip/{commit}", timeout=300).read()
    tmp = dest.parent / "_code"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp)
    top = next(tmp.iterdir())
    src = top / subdir if subdir else top
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(src), dest)
    shutil.rmtree(tmp, ignore_errors=True)


class Worker:
    """A resident helper process in an engine's environment: one JSON request per line on stdin,
    one JSON answer per line on stdout. The first line it prints is {"ready": true, ...}."""

    def __init__(self, name: str, log: Path):
        self.name, self.log = name, log
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.state = {"status": "stopped", "error": "", "load_seconds": None}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> str:
        return self.state["status"] if self.alive() or self.state["status"] in ("error", "loading") else "stopped"

    def _start(self, cmd: list, env: dict):
        self.state.update(status="loading", error="")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log, "w")  # noqa: SIM115 - lives as long as the worker
        self.proc = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                     text=True, encoding="utf-8", bufsize=1, env=env)
        first = self.proc.stdout.readline()
        try:
            ready = json.loads(first)
        except ValueError:
            self.state.update(status="error", error=self.log.read_text(errors="replace")[-800:])
            raise RuntimeError(f"המנוע {self.name} לא עלה. הפרטים נמצאים ב־{self.log.name}") from None
        self.state.update(status="ready", load_seconds=ready.get("load_seconds"))

    def ensure(self, cmd: list, env: dict):
        with self.lock:
            if not self.alive():
                self._start(cmd, env)

    def call(self, req: dict, cmd: list, env: dict) -> dict:
        with self.lock:
            if not self.alive():
                self._start(cmd, env)
            self.state["status"] = "busy"
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            self.state["status"] = "ready"
        if not line:
            self.state.update(status="error", error="the worker stopped")
            raise RuntimeError(f"המנוע {self.name} נעצר באמצע. נסה שוב")
        res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(f"{self.name}: {res.get('error')}")
        return res

    def warm_up(self, cmd: list, env: dict):
        def go():
            try:
                self.ensure(cmd, env)
            except Exception as e:  # shown in the UI
                self.state.update(status="error", error=str(e))

        self.state.update(status="loading", error="")
        threading.Thread(target=go, daemon=True).start()

    def wait_ready(self, timeout: float = 300) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.state["status"] in ("ready", "busy"):
                return True
            if self.state["status"] == "error":
                return False
            time.sleep(1)
        return False

    def stop(self):
        if self.alive():
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.state["status"] = "stopped"
