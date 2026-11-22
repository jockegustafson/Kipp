"""How much air a breath brings in, and a breath planner fitted on data (Kipp 1.3, lung model v3, KTH 2026-11-22).

Breath volume. Over several breath cycles of one speaker, the air breathed in must match the air spent in speech, so a model of breath
volume can be fitted without a spirometer: log V = log s + a log t + c level + spectrum terms, with s the speaker's own scale (lung size,
microphone), fitted on how well the summed volumes of 5 consecutive breaths match the air spent between them (330 h of Swedish
conversation, 521 speakers; within-speaker r = 0.46 on held-out speakers, against 0.35 for breath length alone). V grows as t^0.54 (longer
breaths are slower), 10 dB louder at the same length means about 18 % more air, and a disfluent breath brings about a quarter of an
in-breath's air. Volumes are RELATIVE: 1 = a breath of the corpora's median length and level; the speaker's scale cannot be heard.

    from kipp.volume import breath_volume
    breath_volume(0.45, db=-14.0)                  # relative volume of a 0.45 s in-breath 14 dB below the speech level
    BreathDetector().detect_file(...)              # each breath also gets "vol" (computed from its audio: level, spectrum, noise floor)

Planner models (multi-class nets with one hidden layer, JSON): from what a synthesiser knows (air of the breath group ahead, air spent
since the last breath, the pause, speech level and rate), (1) is there an AUDIBLE breath at a pause (AUC 0.77 on held-out speakers, pause
length alone 0.62), (2) its length class S / M / L (65 %, majority 50 %), (3) soft / loud (60 %, majority 50 %); used by
kipp.lung.plan(..., sizing="data"). Where a planner refills, the refill is silent rather than audible when P(audible | refill) < 0.5, from
p_audible and the hidden lung-state model's rate of silent refills in breathless pauses (sigmoid(0.04 + 0.89 log pause)).
Speech levels (next_db, prev_db1) are dB re the speaker's median speech level; typical values at a breath are -2.7 and -3.5."""
import json
from importlib import resources

import numpy as np

SR = 16000
V3 = json.loads((resources.files("kipp") / "weights" / "lung_v3.json").read_text(encoding="utf-8"))
_VOL = V3["volume"]; WU, KDB = V3["air_cost"]["wu"], V3["air_cost"]["k"]
_REF = None


def _z(name, v): return (v - _VOL["mu"][name]) / _VOL["sd"][name]


def _logv(lt, db, snr=None, lcog=None, hf=None, flat=None, disfl=0.0):
    """the balance model's log volume (missing spectrum terms at their mean)"""
    th = _VOL["theta"]; z = th["lt"] * lt + th["db"] * _z("db", db) + th["disfl"] * disfl
    for k, v in (("snr", snr), ("lcog", lcog), ("hf", hf), ("flat", flat)):
        if v is not None and np.isfinite(v): z += th[k] * _z(k, v)
    return z


def _ref():
    global _REF
    if _REF is None: _REF = _logv(np.log(0.40), _VOL["mu"]["db"])          # a 0.40 s in-breath at the median level -> 1
    return _REF


def breath_volume(dur, db, disfluent=False, snr=None, cog=None, hf=None, flat=None):
    """relative air of one breath: dur (s), db = level re the speaker's speech level (as the detector's level_db); optional spectrum:
    snr (dB over the pause's noise floor), cog (Hz, 100-8000), hf (dB, 2-8 kHz re 0.1-2 kHz), flat (spectral flatness 0.3-6 kHz)"""
    lv = _logv(np.log(max(dur, 0.05)), db, snr, np.log(max(cog, 50)) if cog else None, hf, flat, float(disfluent))
    return float(np.exp(lv - _ref()))


def spectrum(seg):
    """cog (Hz), hf (dB), flatness of a breath's samples (16 kHz)"""
    if len(seg) < 256: return None, None, None
    n = 512; fr = [seg[i:i + n] * np.hanning(n) for i in range(0, len(seg) - n + 1, 256)] or [np.pad(seg, (0, n - len(seg))) * np.hanning(n)]
    P = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in fr], 0) + 1e-12; fq = np.fft.rfftfreq(n, 1 / SR)
    b = (fq >= 100) & (fq <= 8000); m = (fq >= 300) & (fq <= 6000)
    return (float((fq[b] * P[b]).sum() / P[b].sum()), float(10 * np.log10(P[(fq >= 2000) & (fq <= 8000)].sum() / P[(fq >= 100) & (fq < 2000)].sum())),
            float(np.exp(np.mean(np.log(P[m]))) / np.mean(P[m])))


def add_volumes(x, breaths):
    """adds "vol" (relative air) to detector breaths that have "db" (level re speech); out-breaths get None"""
    hop = 320
    for b in breaths:
        if b["label"] == "out": b["vol"] = None; continue
        a, e = int(b["s"] * SR), int(b["e"] * SR); cog, hf, flat = spectrum(x[a:e].astype(np.float64))
        lo, hi = max(0, a - SR), min(len(x), e + SR); w = x[lo:hi].astype(np.float64)
        fr = 10 * np.log10(np.mean(w[:len(w) // hop * hop].reshape(-1, hop) ** 2, 1) + 1e-12) if len(w) >= hop * 5 else None
        lvl = 10 * np.log10(np.mean(x[a:e].astype(np.float64) ** 2) + 1e-12)
        snr = float(lvl - np.percentile(np.convolve(fr, np.ones(5) / 5, "valid"), 2)) if fr is not None else None
        b["vol"] = round(breath_volume(b["e"] - b["s"], b["db"], b["label"] == "disfluent", snr, cog, hf, flat), 3)
    return breaths


def _net(m, x):
    z = (np.asarray(x, np.float64) - np.asarray(m["mu"])) / np.asarray(m["sd"])
    h = np.maximum(0, z @ np.asarray(m["W1"]) + np.asarray(m["b1"])); o = h @ np.asarray(m["W2"]) + np.asarray(m["b2"])
    if m["out"] == "logistic": p1 = 1 / (1 + np.exp(-o[0])); return np.array([1 - p1, p1])
    e = np.exp(o - o.max()); return e / e.sum()


def p_audible(spent, tsince, pause, next_air, to_end, prev_filler=False, next_db=0.0, prev_db1=0.0):
    """probability that a speaker takes an AUDIBLE in-breath in a pause: spent = air since the last breath, next_air = air of the speech
    up to the next pause, to_end = air to the end of the stretch (air: seconds of speech, voiceless time x WU, x 10^(KDB dB / 10))"""
    m = V3["planner"]["audible"]["mlp"]
    f = {"l_spent": np.log1p(spent), "l_tsince": np.log1p(tsince), "l_pause": np.log(max(pause, 0.05)), "l_next": np.log1p(next_air),
         "l_to_end": np.log1p(to_end), "prev_fill": float(prev_filler), "next_db": next_db, "prev_db1": prev_db1}
    return float(_net(m, [f[c] for c in m["features"]])[1])


def p_silent_refill(pause):
    """P(air is taken in without an audible breath) in a pause where no breath is heard (the hidden lung-state model's estimate)"""
    g = V3["silent_inflow"]; return float(1 / (1 + np.exp(-(g["g0"] + g["g1"] * np.log(max(pause, 0.05))))))


def p_audible_given_refill(p_aud, pause):
    """P(the refill is audible | the speaker refills in this pause), from p_audible and p_silent_refill"""
    q = (1 - p_aud) * p_silent_refill(pause); return float(p_aud / max(p_aud + q, 1e-9))


def breath_class(group_air, spent, pause, stretch_start=False, next_db=0.0, prev_db1=0.0, rate=5.0):
    """(class, length s) of an in-breath before a breath group of `group_air` air: S / M / L x soft / loud, and the class's median length"""
    f = {"l_group": np.log1p(group_air), "l_spent": np.log1p(spent), "l_pause": np.log(max(pause, 0.05)), "stretch_start": float(stretch_start),
         "next_db": next_db, "prev_db1": prev_db1, "rate": rate}
    ms, ml = V3["planner"]["size"]["mlp"], V3["planner"]["level"]["mlp"]
    size = int(ms["classes"][int(np.argmax(_net(ms, [f[c] for c in ms["features"]])))])
    loud = int(ml["classes"][int(np.argmax(_net(ml, [f[c] for c in ml["features"]])))])
    return "SML"[size] + ("loud" if loud else "soft"), V3["planner"]["median_dur_s"][str(size)]
