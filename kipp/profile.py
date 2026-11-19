"""A speaker's breathing profile from a recording: the numbers behind the lung model's voice settings, measured as the profiles in
kipp/weights/voices.json were measured on corpora (breath_voices.py / voice_air.py of the KTH breathing-TTS work), so a recording of
yourself gives your own planner settings.

Measured (with Kipp's breath detector and word times, or Silero VAD + syllable nuclei when there are no words):
  syll_per_s    syllables per second of speech (inside speech stretches; pauses excluded)
  breath_rate   share of the mid-speech pauses (0.25-3 s) that hold an audible in-breath
  breath_s      median in-breath length (s);  breath_db: median level re the speech (dB)
  group_phon    median voiced seconds between two audible in-breaths (one breath group's "out-breathing")
  group_s       median clock time between two in-breaths (speech + pauses)
Derived by the lung model's rules (voice_params): air per breath group = group_phon x 0.11 l/s, the inhale flow = that air / breath_s,
the highest volume from the group's air. Vital capacity cannot be heard; it stays at the default for the chosen gender (4.8 / 3.6 l).
Without words, speech stretches and pauses come from voicing and syllables are counted as loudness peaks in voiced stretches (de Jong &
Wempe 2009); both are estimates, best with one speaker on a close microphone, at least ~30 s of speech and 3 or more audible breaths.

Calibration (calibrate): a deep in-breath, then speech for as long as possible on that one breath. The voiced seconds until the next
in-breath, times 0.11 l/s, is the most air the speaker uses on one breath in the lung model's units; it sets the vital capacity of the
model (that air spans 20-90 % of it), and the deep breath's length gives the full-effort inhale flow. A model value, not spirometry."""
import numpy as np

from .lung import PHON_FLOW, syllables, voice_params
from .model import SR

FR = 0.01


def _frames(x, hop=int(FR * SR), win=int(0.03 * SR)):
    n = max(0, 1 + (len(x) - win) // hop)
    return np.lib.stride_tricks.as_strided(x, (n, win), (x.strides[0] * hop, x.strides[0])) if n else np.zeros((0, win), x.dtype)


def voicing(x):
    """per 10 ms frame: (voiced bool, level dB) - normalised autocorrelation peak >= 0.45 at 60-400 Hz and level within 30 dB of the top"""
    F = _frames(x.astype(np.float32)); F = F - F.mean(1, keepdims=True)
    db = 10 * np.log10((F ** 2).mean(1) + 1e-10)
    lo, hi = int(SR / 400), int(SR / 60)
    spec = np.fft.rfft(F, 2 * F.shape[1], axis=1); ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :F.shape[1]]
    peak = ac[:, lo:hi].max(1) / (ac[:, 0] + 1e-9)
    return (peak >= 0.45) & (db > np.percentile(db, 95) - 30), db


def syllable_nuclei(x, voiced, db):
    """loudness peaks in voiced frames, at least 1.5 dB above the dips around them, within 30 dB of the top, >= 50 ms apart (de Jong &
    Wempe 2009, simplified; tuned on 10 min of a podcast: 0.96 of the syllables of its word transcript)"""
    k = db; thr = np.percentile(k, 99) - 30; peaks = []
    for i in range(2, len(k) - 2):
        if voiced[i] and k[i] > thr and k[i] >= k[i - 1] and k[i] > k[i + 1]:
            left = k[max(0, i - 15):i].min(); right = k[i + 1:i + 16].min()
            if k[i] - max(left, right) >= 1.5 and (not peaks or i - peaks[-1] >= 5): peaks.append(i)
    return np.array(peaks) * FR


def speech_from_voicing(voiced, join=0.25, minrun=0.03):
    """speech stretches without words: voiced runs (>= 30 ms) joined across gaps shorter than 0.25 s (a breath is unvoiced, so it stays a
    pause; on 10 min of a two-speaker podcast this found 86 of the 167 word-time pauses where Silero VAD found at most 42)"""
    runs, s = [], None
    for i, v in enumerate(list(voiced) + [False]):
        if v and s is None: s = i
        elif not v and s is not None:
            if (i - s) * FR >= minrun: runs.append([s * FR, i * FR])
            s = None
    sp = []
    for a, b in runs:
        if sp and a - sp[-1][1] < join: sp[-1][1] = b
        else: sp.append([a, b])
    return [tuple(x) for x in sp]


def estimate(x, breaths, words=None, speech=None, gender="m"):
    """x: 16 kHz mono; breaths: Kipp detections (keep_out off); words: [(start, end, word)] or None, then speech: [(start, end)] (VAD)
    -> {"profile": voices.json-style numbers, "params": the lung model's settings for them, "n": counts behind the numbers}
    Without words or speech, speech stretches come from voicing (speech_from_voicing)."""
    voiced, db = voicing(x); t = (np.arange(len(voiced)) + 0.5) * FR
    if words:
        segs = sorted((float(s), float(e)) for s, e, *_ in words); n_syl = sum(syllables(w[2]) for w in words if len(w) > 2 and w[2])
    else:
        segs = sorted(speech or speech_from_voicing(voiced)); nuc = syllable_nuclei(x, voiced, db)
        n_syl = int(sum(((nuc >= s) & (nuc < e)).sum() for s, e in segs))
    talk = sum(e - s for s, e in segs)
    gaps = [(a1, b0) for (a0, a1), (b0, b1) in zip(segs, segs[1:]) if 0.25 <= b0 - a1 <= 3.0]
    ins = [b for b in breaths if b["label"] == "in"]
    has = [any(b["s"] < g1 and b["e"] > g0 for b in ins) for g0, g1 in gaps]
    # breath groups: voiced seconds (inside speech) and clock time between consecutive in-breaths
    in_speech = np.zeros(len(t), bool)
    for s, e in segs: in_speech |= (t >= s) & (t < e)
    vs = np.cumsum(voiced & in_speech) * FR
    starts = sorted(b["e"] for b in ins); grp, grp_s = [], []
    for a, b in zip(starts, starts[1:]):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b)
        grp.append(float(vs[min(j, len(vs) - 1)] - vs[min(i, len(vs) - 1)])); grp_s.append(b - a)
    prof = {"gender": gender, "syll_per_s": round(n_syl / talk, 2) if talk > 1 else None,
            "breath_rate": round(float(np.mean(has)), 3) if has else None,
            "breath_s": round(float(np.median([b["e"] - b["s"] for b in ins])), 3) if ins else None,
            "breath_db": round(float(np.median([b["db"] for b in ins])), 1) if ins else None,
            "group_phon": round(float(np.median(grp)), 2) if grp else None, "group_s": round(float(np.median(grp_s)), 2) if grp_s else None}
    use = {k: v for k, v in prof.items() if v is not None}
    P = voice_params(use)
    return {"profile": prof, "params": P, "air_per_group_l": round((use.get("group_phon") or 3.5) * PHON_FLOW, 3),
            "n": {"speech_s": round(talk, 1), "syllables": int(n_syl), "pauses": len(gaps), "in_breaths": len(ins), "groups": len(grp),
                  "source": "words" if words else "voicing + syllable nuclei"}}


def calibrate(x, breaths, gender="m"):
    """x: a recording that starts with the deepest possible in-breath and then speech for as long as possible (until the next in-breath or
    the end); breaths: Kipp detections of it (detect with speech=None -> words from voicing)
    -> {"max_voiced_s", "max_air_l", "deep_breath_s", "vc", "flow_max", "ok", "note"}"""
    voiced, db = voicing(x); t = (np.arange(len(voiced)) + 0.5) * FR
    ins = sorted((b for b in breaths if b["label"] == "in"), key=lambda b: b["s"])
    sp = [s for s in speech_from_voicing(voiced) if s[1] - s[0] >= 0.15]; first_voice = sp[0][0] if sp else None
    deep = next((b for b in reversed(ins) if first_voice is None or b["e"] <= first_voice + 0.3), None)   # the last breath before speech
    start = deep["e"] if deep else (first_voice or 0.0)
    nxt = next((b for b in ins if b["s"] > start + 1.0), None); end = nxt["s"] if nxt else len(x) / SR
    m = (t >= start) & (t < end); vs = float(voiced[m].sum() * FR)
    air = vs * PHON_FLOW; vc = float(min(7.0, max(2.0, air / 0.70)))
    dur = (deep["e"] - deep["s"]) if deep else None
    note = []
    if not deep: note.append("no deep in-breath found at the start: start recording, then breathe in")
    if not nxt: note.append("no second in-breath found: the speech was measured to the end of the recording")
    if vs < 5: note.append("under 5 s of voiced speech: talk for longer on the one breath")
    return {"max_voiced_s": round(vs, 2), "max_air_l": round(air, 3), "deep_breath_s": round(dur, 2) if dur else None,
            "speech_s": round(end - start, 2), "vc": round(vc, 1), "flow_max": round(min(3.0, max(0.3, 0.7 * vc / dur)), 2) if dur else None,
            "ok": bool(deep and vs >= 5), "note": "; ".join(note)}
