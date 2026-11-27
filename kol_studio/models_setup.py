"""Model packs: what the app downloads from Hugging Face, where it goes, and how to tell it is installed.

Models are not inside the installer (several GB). The first-run screen downloads the required packs; the
Settings page downloads the optional ones.
"""
import fnmatch
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Pack:
    repo: str
    patterns: tuple[str, ...]
    dest: str  # folder under paths.MODELS ("" = the models folder itself)
    check: str  # a file whose presence means the pack is installed
    required: bool
    name: str
    size: str


PACKS = {
    "speech": Pack("notmax123/BlueV3-onnx",
                   ("onnx/*", "voices/*", "frontend/renikud/*", "LICENSES.md", "LICENSE.OpenRAIL-M"),
                   "bluev3", "bluev3/onnx/vector_estimator.int8.onnx", True, "הקראה ושכפול קול (BlueV3)", "2.0GB"),
    "english": Pack("notmax123/BlueV3-onnx",
                    ("frontend/polyg2p_data/en.lex", "frontend/polyg2p_data/en.g2p",
                     "frontend/polyg2p_data/en-large.g2p", "frontend/polyg2p_data/LICENSE",
                     "frontend/polyg2p_data/NOTICE"),
                    "bluev3", "bluev3/frontend/polyg2p_data/en-large.g2p", True,
                    "מילים באנגלית בתוך טקסט עברי", "9MB"),
    "nikud": Pack("Phonikud/phonikud-onnx", ("phonikud-1.0.int8.onnx", "README.md"),
                  "", "phonikud-1.0.int8.onnx", True, "ניקוד (Phonikud)", "0.3GB"),
    "asr": Pack("ivrit-ai/whisper-large-v3-turbo-ct2", ("*",),
                "ivrit-turbo-ct2", "ivrit-turbo-ct2/model.bin", False, "תמלול מהיר (ivrit.ai turbo)", "1.6GB"),
    "asr-large": Pack("ivrit-ai/whisper-large-v3-ct2", ("*",),
                      "ivrit-large-ct2", "ivrit-large-ct2/model.bin", False, "תמלול מדויק (ivrit.ai large-v3)", "3.1GB"),
}

ASR_PACK = {"ivrit-turbo": "asr", "ivrit-large": "asr-large"}  # transcription model setting -> pack


def installed(pack_id: str) -> bool:
    return (paths.MODELS / PACKS[pack_id].check).exists()


def status() -> list[dict]:
    return [{"id": k, "name": p.name, "size": p.size, "required": p.required, "installed": installed(k)}
            for k, p in PACKS.items() if k != "asr-large"]  # the large model is offered in Settings only


def ready() -> bool:
    return all(installed(k) for k, p in PACKS.items() if p.required)


def asr_dir(model: str) -> Path:
    """Folder of the chosen transcription model, falling back to the fast one when it is missing."""
    pack = PACKS[ASR_PACK.get(model, "asr")]
    if not installed(ASR_PACK.get(model, "asr")):
        pack = PACKS["asr"]
    return paths.MODELS / pack.dest


def download(pack_ids: list[str], job: dict):
    """Download packs file by file; job["done"] / job["total"] count bytes for the progress bar."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    plan = []
    for pid in pack_ids:
        p = PACKS[pid]
        dest = paths.MODELS / p.dest if p.dest else paths.MODELS
        for s in api.model_info(p.repo, files_metadata=True).siblings:
            if any(fnmatch.fnmatch(s.rfilename, pat) for pat in p.patterns):
                plan.append((p, s.rfilename, s.size or 0, dest))
    job["total"] = sum(size for _, _, size, _ in plan) or 1
    job["done"] = 0
    for p, fname, size, dest in plan:
        job["current"] = f"{p.name} · {Path(fname).name}"
        if not (dest / fname).exists():
            hf_hub_download(p.repo, fname, local_dir=str(dest))
        job["done"] += size
