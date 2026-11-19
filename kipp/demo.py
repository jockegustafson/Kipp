"""kipp-demo: a local web page for Kipp (breath detection, your breathing profile, calibration, and breath planning for a text).

    kipp-demo [--port 8766] [--device cuda|mps|cpu] [--host 127.0.0.1] [--no-browser]

then open http://localhost:8766 (the microphone works on localhost). Nothing leaves the machine. The page sends audio as float32 at its
sample rate; recordings over MAX_S = 300 s are cut.
  POST /analyze   {pcm, sr, words?, tier?, gender}  -> spectrogram, breaths, profile (kipp.profile.estimate)
  POST /calibrate {pcm, sr, gender}                 -> spectrogram, breaths, calibration (kipp.profile.calibrate)
  POST /plan      {text, voice | profile, calib?, gender, place, eff} -> the lung model's plan and its simulation
  GET  /voices                                      -> the measured voice profiles"""
import argparse, base64, json, os, tempfile, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import numpy as np

from .lung import VOICES, plan, simulate, voice_params
from .profile import calibrate, estimate, speech_from_voicing, voicing

MAX_S = 300.0; SR = 16000
DET = None; LOCK = threading.Lock()


def to16k(y, sr):
    y = np.asarray(y, np.float32)
    if int(sr) != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, int(sr)); y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return y[:int(MAX_S * SR)]


def spectrogram(y, n_fft=512, hop=160):
    if len(y) < n_fft: y = np.pad(y, (0, n_fft - len(y)))
    w = np.hanning(n_fft).astype(np.float32); n = 1 + (len(y) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0])) * w
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6); top = S.max()
    S = np.clip((S - (top - 80)) / 80, 0, 1)
    return {"frames": int(n), "bins": int(S.shape[1]), "hop_s": hop / SR, "data": base64.b64encode((S * 255).astype(np.uint8).tobytes()).decode()}


def read_words_text(text, tier):
    from .cli import read_words
    suffix = ".TextGrid" if "IntervalTier" in text else ".tsv"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f: f.write(text)
    try: return [(s, e, w or "") for w, s, e in read_words(f.name, tier or None)]
    finally: os.unlink(f.name)


def detect(y, words):
    """breaths in the pauses between the words, or between voicing-based speech stretches when there are no words; a dummy first word at
    0 s makes the pause before the first speech (where a recording's first in-breath sits) part of the search"""
    if not words:
        voiced, _ = voicing(y); words = [(0.0, 0.01, "")] + [(s, e, "") for s, e in speech_from_voicing(voiced)]
    with LOCK: return DET.detect(y, words), words


def analyze(req):
    y = to16k(np.frombuffer(base64.b64decode(req["pcm"]), np.float32), req.get("sr", SR)); g = req.get("gender", "m")
    words = read_words_text(req["words"], req.get("tier")) if req.get("words") else None
    br, used = detect(y, words)
    est = estimate(y, br, words=words, gender=g) if words else estimate(y, br, speech=[(s, e) for s, e, _ in used[1:]], gender=g)
    return {"dur": len(y) / SR, "spec": spectrogram(y), "breaths": br, **est}


def calib(req):
    """the deep breath is searched directly in the 4 s before the first speech and the final breath in the 4 s after the last (the pause
    search alone only looks between two stretches of speech, 2026-11-19); inside the speech, the pauses as usual"""
    y = to16k(np.frombuffer(base64.b64decode(req["pcm"]), np.float32), req.get("sr", SR)); br, used = detect(y, None)
    sp = [(s, e) for s, e, _ in used[1:] if e - s >= 0.15]
    if sp:
        with LOCK:
            pre = DET.detect_window(y, sp[0][0] - 4.0, sp[0][0] + 0.2); post = DET.detect_window(y, sp[-1][1] - 0.2, sp[-1][1] + 4.0)
        extra = [b for b in (pre, post) if b and not any(abs(b["s"] - o["s"]) < 0.2 for o in br)]
        if extra:
            from .model import classify
            br = classify(y, sorted(br + extra, key=lambda b: b["s"]), [(s, e, "") for s, e in sp])
    return {"dur": len(y) / SR, "spec": spectrogram(y), "breaths": br, "calibration": calibrate(y, br, req.get("gender", "m"))}


def make_plan(req):
    if req.get("voice") and req["voice"] != "mine": P = voice_params(req["voice"])
    else: P = voice_params(dict(req.get("profile") or {}, gender=req.get("gender", "m")))
    c = req.get("calib") or {}
    c = c if c.get("ok") else {}                                   # a calibration that failed its own checks is not used
    if c.get("vc"): P["vc"] = c["vc"]
    if c.get("flow_max"): P["flow"] = min(P["flow"], c["flow_max"])
    P["place"] = req.get("place", "phrase"); P["eff"] = float(req.get("eff", 1.0))
    w = plan(req["text"], P); sim = simulate(w, P)
    return {"P": P, "words": [{k: o.get(k) for k in ("w", "br", "dv", "bdur", "dur", "air")} for o in w],
            "sim": {"pts": sim["pts"], "end": sim["end"], "minV": sim["minV"], "words": sim["words"], "breaths": {str(k): v for k, v in sim["breaths"].items()}}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode(); self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"): self._send(200, (resources.files("kipp") / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/voices": self._send(200, json.dumps(VOICES, ensure_ascii=False), "application/json")
        else: self._send(404, "not found", "text/plain")

    def do_POST(self):
        fn = {"/analyze": analyze, "/calibrate": calib, "/plan": make_plan}.get(self.path)
        if not fn: return self._send(404, "not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, json.dumps(fn(req), ensure_ascii=False, default=float), "application/json")
        except SystemExit as ex: self._send(400, json.dumps({"error": str(ex)}), "application/json")
        except Exception as ex: self._send(500, json.dumps({"error": f"{type(ex).__name__}: {ex}"}), "application/json")


def main():
    global DET
    ap = argparse.ArgumentParser(description="Kipp demo page (local)")
    ap.add_argument("--port", type=int, default=8766); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--device")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    from . import BreathDetector
    print("loading the breath detector and WavLM-large (first run downloads it, ~1.2 GB) ...", flush=True)
    DET = BreathDetector(a.device)
    url = f"http://localhost:{a.port}"; print(f"Kipp demo ({DET.dev}) at {url}  - Ctrl-C to stop", flush=True)
    if not a.no_browser: threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
