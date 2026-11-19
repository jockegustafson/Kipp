"""Kipp: audible in-breaths and disfluent breaths in Swedish speech.
  det = BreathDetector()                                    # cuda, mps or cpu, whichever is there
  det.detect_file("interview.wav", words, channel=0)        # words: [(word, start, end)] of ONE speaker, or None (Silero VAD)
  -> [{"s", "e", "label": "in" | "disfluent" (| "out" with keep_out=True), "p", "pause": "pause" | "turnend" | "prestart", "db", "cls"}]
Breath types: label in / disfluent / out; cls = the breath-token class: S / M / L (< 0.30 s, 0.30-0.55 s, longer) x soft / loud (below /
above this recording's median breath level), "disfl" or "out". Out-breaths are found by the model but were counted as "none" when it
was trained (TTS needs in-breaths), so they are off by default and their recall is lower."""
import hashlib
import io
import json
from importlib import resources

import numpy as np
import torch

from .model import FR, GRU, MARG, MAXCLIP, SR, classify, events, pauses


def load_audio(path, channel=None):
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y[:, channel] if channel is not None and channel < y.shape[1] else y.mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr); y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def vad_words(x):
    """no words given: speech stretches from Silero VAD (pip install "kipp[vad]") stand in for words"""
    from silero_vad import get_speech_timestamps, load_silero_vad
    ts = get_speech_timestamps(torch.from_numpy(x), load_silero_vad(), sampling_rate=SR, min_silence_duration_ms=200)
    return [(t["start"] / SR, t["end"] / SR, "") for t in ts]


class BreathDetector:
    def __init__(self, device=None):
        from transformers import WavLMModel
        self.dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        ref = resources.files("kipp") / "weights"; man = json.loads((ref / "MANIFEST.json").read_text())
        raw = (ref / "breath_gru_v2.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != man["npz_sha256"]: raise RuntimeError("kipp weights do not match MANIFEST.json")
        z = np.load(io.BytesIO(raw)); self.manifest = man; self.thr = man["thr"]; self.layers = man["layers"]
        self.m = GRU(); self.m.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files}); self.m.to(self.dev).eval()
        self.wl = WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.dev).eval()
        if self.dev == "cuda": self.wl = self.wl.half()

    @torch.inference_mode()
    def probs(self, x16):
        """frame probabilities (none, in, out, disfluent) of one clip, 16 kHz mono"""
        inp = torch.from_numpy(x16)[None].to(self.dev); inp = inp.half() if self.dev == "cuda" else inp
        h = self.wl(inp, output_hidden_states=True).hidden_states
        f = torch.stack([h[k][0].float() for k in self.layers]).mean(0).cpu().numpy(); n = len(f); hop = int(FR * SR)
        e = np.array([10 * np.log10(np.mean(x16[i * hop:(i + 1) * hop] ** 2) + 1e-9) for i in range(n)])
        F = np.concatenate([f, ((e - np.percentile(e, 90)) / 20.0)[:, None]], 1).astype(np.float32)
        return torch.softmax(self.m(torch.from_numpy(F)[None].to(self.dev))[0], -1).float().cpu().numpy()

    def detect(self, x, words=None, thr=None, keep_out=False):
        """x: 16 kHz mono float32; words: [(start, end, ...)] of the speaker (None: Silero VAD) -> breath events (seconds)"""
        thr = self.thr if thr is None else thr; dur = len(x) / SR; out = []
        for kind, s, e in pauses(words if words is not None else vad_words(x)):
            t1 = min(e + MARG, dur); t0 = max(0.0, s - MARG, t1 - MAXCLIP); c = x[int(t0 * SR):int(t1 * SR)]
            if len(c) < SR * 0.5: continue
            c = (c / max(1e-4, float(np.abs(c).max())) * 0.9).astype(np.float32)      # peak-normalised as the training clips
            for v in events(self.probs(c), max(0.0, s - t0), e - t0, thr, keep_out):
                out.append({"s": round(t0 + v["s"], 3), "e": round(t0 + v["e"], 3), "label": v["label"], "p": v["p"], "pause": kind})
        return classify(x, out, words)

    def detect_window(self, x, t0, t1, thr=None):
        """the strongest in-breath in [t0, t1] (s), searched directly (not only in pauses between words): threshold `thr` (default the
        model's), then 0.3; None if there is none. For recordings where a breath is known to be there (calibration)."""
        t0, t1 = max(0.0, t0), min(len(x) / SR, t1); c = x[int(t0 * SR):int(t1 * SR)]
        if len(c) < SR * 0.3: return None
        c = (c / max(1e-4, float(np.abs(c).max())) * 0.9).astype(np.float32); P = self.probs(c)
        for th in (self.thr if thr is None else thr, 0.3):
            ev = [v for v in events(P, 0.0, t1 - t0, th) if v["label"] == "in"]
            if ev:
                v = max(ev, key=lambda v: (v["e"] - v["s"]) * v["p"])
                return {"s": round(t0 + v["s"], 3), "e": round(t0 + v["e"], 3), "label": "in", "p": v["p"], "pause": "window", "found_at": th}
        # last resort: the longest unvoiced stretch at least 12 dB above the window's quietest tenth (a breath is unvoiced noise)
        from .profile import voicing
        vo, db = voicing(c); loud = (~vo) & (db > np.percentile(db, 10) + 12); best, i = None, 0
        while i < len(loud):
            if not loud[i]: i += 1; continue
            j = i
            while j + 1 < len(loud) and (loud[j + 1] or (j + 2 < len(loud) and loud[j + 2])): j += 1
            if (j - i + 1) * 0.01 >= 0.2 and (best is None or j - i > best[1] - best[0]): best = (i, j)
            i = j + 1
        if best:
            return {"s": round(t0 + best[0] * 0.01, 3), "e": round(t0 + (best[1] + 1) * 0.01, 3), "label": "in", "p": None, "pause": "window", "found_at": "energy"}
        return None

    def detect_file(self, path, words=None, channel=None, thr=None, keep_out=False):
        """path: any audio soundfile reads; words: [(word, start, end)] of the speaker on `channel`, or None for Silero VAD"""
        x = load_audio(path, channel)
        w = [(float(s), float(e), wd) for wd, s, e in words] if words is not None else None
        return self.detect(x, w, thr, keep_out)
