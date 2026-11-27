"""Kol Studio worker for QwenTTS-he (Qwen3-TTS-12Hz-1.7B-Base + the Hebrew LoRA notmax123/QwenTTS-he-1.7B).
Runs inside the engine's own Python environment.

Loads the model once, then answers one JSON request per line on stdin with one JSON line on stdout:
  {"text": "pointed Hebrew", "out": "/path/out.wav", "ref_audio": "/path/ref.wav"}
  -> {"ok": true, "out": "...", "seconds": 3.2, "synth_seconds": 9.1}   or   {"ok": false, "error": "..."}
The first line it prints is {"ready": true, ...} once the model is loaded.

Both the base model and the adapter are Apache-2.0. The model reads stressed IPA, so the text goes through
RenikudPlus (the model file BlueV3 already ships) with the user's nikud honoured.
"""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--renikud", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out = sys.stdout
    sys.stdout = sys.stderr  # libraries print; keep stdout for our JSON lines only

    import numpy as np
    import soundfile as sf
    import torch
    from peft import PeftModel
    from qwen_tts import Qwen3TTSModel
    from renikud_onnx import G2P

    device = a.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = {"mps": torch.float16, "cuda": torch.bfloat16}.get(device, torch.float32)
    t0 = time.time()
    g2p = G2P(a.renikud, niqqud="use")
    tts = Qwen3TTSModel.from_pretrained(a.base, device_map=device, dtype=dtype, attn_implementation="sdpa")
    tts.model.talker = PeftModel.from_pretrained(tts.model.talker, a.adapter).merge_and_unload()
    tts.model.eval()
    prompts: dict = {}
    print(json.dumps({"ready": True, "device": device, "load_seconds": round(time.time() - t0, 1)}), file=out,
          flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t1 = time.time()
            ref = req["ref_audio"]
            if ref not in prompts:
                prompts[ref] = tts.create_voice_clone_prompt(ref_audio=ref, x_vector_only_mode=True)
            waves, sr = [], 24000
            for sentence in [s for s in req["text"].split("\n") if s.strip()]:
                ipa = g2p.phonemize(sentence.strip())
                wavs, sr = tts.generate_voice_clone(text=ipa, voice_clone_prompt=prompts[ref], language="Auto",
                                                    non_streaming_mode=True, do_sample=False, subtalker_dosample=False,
                                                    repetition_penalty=1.0, max_new_tokens=2048)
                waves += [np.asarray(wavs[0], dtype=np.float32), np.zeros(int(sr * 0.25), dtype=np.float32)]
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
