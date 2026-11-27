"""Kol Studio worker for the sound-effect engines. Runs inside the engine's own Python environment.

Loads the model once, then answers one JSON request per line on stdin with one JSON line on stdout:
  {"prompt": "...", "seconds": 4, "outs": ["/a.wav", "/b.wav"], "seed": 1, "quality": "normal", "negative": ""}
  -> {"ok": true, "files": [...], "seconds": 4.0, "synth_seconds": 12.3}   or   {"ok": false, "error": "..."}
The first line it prints is {"ready": true, ...} once the model is loaded.

Engines: "moss" = MOSS-SoundEffect v2.0 (OpenMOSS, Apache-2.0);
         "sa3"  = Stable Audio 3 Small-SFX (Stability AI Community License).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path


def mps_safe(module, torch):
    """Apple GPUs have no float64/complex128: narrow those buffers (MOSS keeps its RoPE tables in complex128).
    Idea from VladimirTalyzin/MOSS-SoundEffect_v2.0_MPS_ROCm (Apache-2.0)."""
    narrow = {torch.float64: torch.float32, torch.complex128: torch.complex64}
    for m in module.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and buf.dtype in narrow:
                m._buffers[name] = buf.to(narrow[buf.dtype])
        for name, p in list(m._parameters.items()):
            if p is not None and p.dtype in narrow:
                m._parameters[name] = torch.nn.Parameter(p.to(narrow[p.dtype]), requires_grad=p.requires_grad)


def load_moss(model_dir: Path, device: str):
    import importlib

    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    if device == "mps":
        def sinusoidal_f32(dim, position):  # the original computes the timestep embedding in float64
            half = dim // 2
            s = torch.outer(position.float(), torch.pow(
                10000, -torch.arange(half, dtype=torch.float32, device=position.device).div(half)))
            return torch.cat([torch.cos(s), torch.sin(s)], dim=1).to(position.dtype)

        for name in ("moss_soundeffect_v2.diffsynth.pipelines.wan_audio",
                     "moss_soundeffect_v2.diffsynth.models.wan_video_dit",
                     "moss_soundeffect_v2.diffsynth.models.wan_audio_dit"):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(mod, "sinusoidal_embedding_1d"):
                mod.sinusoidal_embedding_1d = sinusoidal_f32
        pipe = MossSoundEffectPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, device="cpu")
        mps_safe(pipe.engine, torch)
        pipe.to("mps")
    else:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        pipe = MossSoundEffectPipeline.from_pretrained(str(model_dir), torch_dtype=dtype, device=device)
    steps = {"draft": 25, "normal": 50, "high": 100}

    def gen(prompt, seconds, seed, quality, negative, canvas=None):
        # Trained on a 30 s canvas. Twice the sound's length (at least 10 s) is ~3x faster and keeps events distinct;
        # a tighter canvas smears short events (measured: three knocks became one long noise at 8 s)
        canvas = canvas or (30 if quality == "high" else min(30, max(10, math.ceil(seconds) * 2)))
        audio = pipe(prompt=prompt, seconds=seconds, num_inference_steps=steps.get(quality, 50), cfg_scale=4.0,
                     seed=seed, negative_prompt=negative, max_inference_seconds=canvas, progress_bar_cmd=_quiet)
        return audio[0].float().cpu().numpy().T, pipe.sample_rate  # (T, C)

    return gen


def load_sa3(model_dir: Path, device: str):
    import torch
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    config = json.loads((model_dir / "model_config.json").read_text())
    text_encoder = model_dir / "t5gemma-b-b-ul2"

    def local(node):  # point the T5Gemma text encoder at the copy inside the model folder (no network)
        if isinstance(node, dict):
            if node.get("model_name") == "google/t5gemma-b-b-ul2" or node.get("type") == "t5gemma":
                cfg = node.get("config", node)
                if isinstance(cfg, dict):
                    cfg["model_path"] = str(text_encoder)
                    cfg.pop("repo_id", None)
                    cfg.pop("subfolder", None)
            for v in node.values():
                local(v)
        elif isinstance(node, list):
            for v in node:
                local(v)

    if text_encoder.is_dir():
        local(config)
    model = load_diffusion_cond(config, str(model_dir / "model.safetensors"), device=device, model_half=False)
    model.use_lora, model.lora_names = False, []
    sa = StableAudioModel(model, config, device, False)
    steps = {"draft": 6, "normal": 8, "high": 12}

    def gen(prompt, seconds, seed, quality, negative, canvas=None):
        audio = sa.generate(prompt=prompt, negative_prompt=negative or None, duration=seconds,
                            steps=steps.get(quality, 8), seed=seed)
        return audio[0].float().cpu().numpy().T, int(config.get("sample_rate", 44100))

    del torch
    return gen


def _quiet(x, **_):
    return x


def finish(audio, sr):
    """Peak to -1 dBFS and a 15 ms fade at both ends, so a sound never clicks when it starts or stops."""
    import numpy as np

    audio = np.nan_to_num(audio.astype(np.float32))
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio * (0.89 / peak)
    n = min(int(sr * 0.015), len(audio) // 4)
    if n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["moss", "sa3"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out = sys.stdout
    sys.stdout = sys.stderr  # libraries print; keep stdout for our JSON lines only

    import soundfile as sf
    import torch

    device = a.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    t0 = time.time()
    gen = (load_moss if a.engine == "moss" else load_sa3)(Path(a.model), device)
    print(json.dumps({"ready": True, "device": device, "load_seconds": round(time.time() - t0, 1)}), file=out,
          flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t1, files = time.time(), []
            seconds = float(req.get("seconds", 4))
            for i, path in enumerate(req["outs"]):
                audio, sr = gen(req["prompt"], seconds, int(req.get("seed", 0)) + i * 1009,
                                req.get("quality", "normal"), req.get("negative", ""), req.get("canvas"))
                sf.write(path, finish(audio, sr), sr, subtype="PCM_16")
                files.append(path)
            print(json.dumps({"ok": True, "files": files, "seconds": seconds,
                              "synth_seconds": round(time.time() - t1, 2)}), file=out, flush=True)
        except Exception as e:  # report and keep serving
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), file=out, flush=True)


if __name__ == "__main__":
    main()
