"""Kol Studio worker for Fish Audio S2-Pro (Hebrew adapter). Runs inside the Fish engine's own Python environment.

Loads the model once, then answers one JSON request per line on stdin with one JSON line on stdout:
  {"text": "...", "out": "/path/out.wav", "ref_audio": "...", "ref_text": "...", "seed": 1, "temperature": 0.7}
  -> {"ok": true, "out": "...", "seconds": 3.2, "synth_seconds": 9.1}   or   {"ok": false, "error": "..."}
The first line it prints is {"ready": true, ...} once the model is loaded.

Fish Audio S2-Pro is licensed under the Fish Audio Research License (research and non-commercial use only).
"Built with Fish Audio."
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the fish-speech source folder")
    ap.add_argument("--renikud", required=True, help="RenikudPlus model.onnx (shared with BlueV3)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--precision", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    a = ap.parse_args()

    src = Path(a.src)
    sys.path.insert(0, str(src))
    out = sys.stdout
    sys.stdout = sys.stderr  # libraries print; keep stdout for our JSON lines only

    import numpy as np
    import soundfile as sf
    import torch

    import fish_speech.text.hebrew as heb
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        decode_one_token_ar,
        encode_audio,
        generate_long,
        init_model,
        load_codec_model,
    )

    def load_g2p():  # the model file BlueV3 already ships; the user's nikud is honoured, not stripped
        from renikud_onnx import G2P

        return G2P(a.renikud, niqqud="use")

    heb.load_g2p = load_g2p

    ckpt = src / "checkpoints" / "s2-pro-he-ipa"
    lora = src / "checkpoints" / "hebrew" / "hebrew_lora_step2200.safetensors"
    device = torch.device(a.device if a.device != "mps" or torch.backends.mps.is_available() else "cpu")
    if a.precision != "auto":
        precision = getattr(torch, a.precision)
    elif device.type == "mps":
        precision = torch.float16  # Apple GPUs before M2 have no bfloat16 hardware: float16 is several times faster
    else:
        precision = torch.bfloat16 if device.type == "cuda" else torch.float32
    t0 = time.time()
    frontend = heb.HebrewTextFrontend.from_checkpoint(ckpt, mode="raw")
    model, _ = init_model(str(ckpt), device=device, precision=precision, lora_config="r_32_alpha_16_core",
                          lora_checkpoint=lora, lora_scale=a.lora_scale, max_length=4096)
    codec = load_codec_model(str(ckpt / "codec.pth"), device=device, precision=precision)
    ref_cache: dict = {}
    print(json.dumps({"ready": True, "device": device.type, "load_seconds": round(time.time() - t0, 1)}), file=out, flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t1 = time.time()
            if req.get("seed") is not None:
                torch.manual_seed(int(req["seed"]))
            prompt_tokens = prompt_text = None
            if req.get("ref_audio"):
                key = (req["ref_audio"], req.get("ref_text", ""))
                if key not in ref_cache:
                    ref_cache[key] = (encode_audio(req["ref_audio"], codec, device).cpu(), frontend(req.get("ref_text", "")))
                tokens, rtext = ref_cache[key]
                prompt_tokens, prompt_text = [tokens], [rtext]
            sr = codec.sample_rate
            gap = np.zeros(int(sr * 0.12), dtype=np.float32)
            waves = []
            for chunk in frontend.chunk(req["text"].strip(), 200):
                codes = [r.codes for r in generate_long(
                    model=model, device=device, decode_one_token=decode_one_token_ar, text=chunk,
                    max_new_tokens=int(req.get("max_new_tokens", 1024)), temperature=float(req.get("temperature", 0.7)),
                    top_p=float(req.get("top_p", 0.7)), top_k=int(req.get("top_k", 30)), compile=False,
                    chunk_length=200, prompt_text=prompt_text, prompt_tokens=prompt_tokens) if r.action == "sample"]
                if codes:
                    waves += [decode_to_audio(torch.cat(codes, dim=1).to(device), codec).float().cpu().numpy().copy(), gap]
            if not waves:
                raise RuntimeError("nothing was generated")
            audio = np.concatenate(waves[:-1])
            peak = float(np.abs(audio).max())
            if peak > 0.98:
                audio = audio / peak * 0.98
            sf.write(req["out"], audio, sr)
            print(json.dumps({"ok": True, "out": req["out"], "seconds": round(len(audio) / sr, 2),
                              "synth_seconds": round(time.time() - t1, 2), "sample_rate": sr}), file=out, flush=True)
        except Exception as e:  # report and keep serving
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), file=out, flush=True)


if __name__ == "__main__":
    main()
