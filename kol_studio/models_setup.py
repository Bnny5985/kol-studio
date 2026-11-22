"""First-run model download. Models are not inside the installer (several GB); they come from Hugging Face."""
import fnmatch
from pathlib import Path

from . import paths

# id: (repo, file patterns, destination folder under MODELS, required?, Hebrew label, approx size)
PACKS = {
    "speech": ("notmax123/BlueV3-onnx",
               ["onnx/*", "voices/*", "frontend/renikud/*", "LICENSES.md", "LICENSE.OpenRAIL-M"],
               "bluev3", True, "הקראה ושכפול קול (BlueV3)", "2.0GB"),
    "nikud": ("Phonikud/phonikud-onnx", ["phonikud-1.0.int8.onnx", "README.md"], "", True, "ניקוד (Phonikud)", "0.3GB"),
    "asr": ("ivrit-ai/whisper-large-v3-turbo-ct2", ["*"], "ivrit-turbo-ct2", False, "תמלול (ivrit.ai)", "1.6GB"),
}

CHECK = {  # a file whose presence means the pack is installed
    "speech": "bluev3/onnx/vector_estimator.int8.onnx",
    "nikud": "phonikud-1.0.int8.onnx",
    "asr": "ivrit-turbo-ct2/model.bin",
}


def status() -> list[dict]:
    return [{"id": k, "name": v[4], "size": v[5], "required": v[3], "installed": (paths.MODELS / CHECK[k]).exists()}
            for k, v in PACKS.items()]


def ready() -> bool:
    return all(p["installed"] for p in status() if p["required"])


def download(pack_ids: list[str], job: dict):
    """Download packs file by file; job["done"]/["total"] are bytes for the progress bar."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    plan = []
    for pid in pack_ids:
        repo, patterns, dest, *_ = PACKS[pid]
        info = api.model_info(repo, files_metadata=True)
        for s in info.siblings:
            if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
                plan.append((pid, repo, s.rfilename, s.size or 0, paths.MODELS / dest if dest else paths.MODELS))
    job["total"] = sum(p[3] for p in plan) or 1
    job["done"] = 0
    for pid, repo, fname, size, dest in plan:
        job["current"] = f"{PACKS[pid][4]} · {Path(fname).name}"
        if not (dest / fname).exists():
            hf_hub_download(repo, fname, local_dir=str(dest))
        job["done"] += size
