"""The lung model: where a speaker breathes in a text, how much air each breath takes, and its breath-token class.
A Python port of the planner of the KTH breathing-TTS page (breath_page.html, 2026-11-17: plan, effWords, simulate, and the voice
defaults of voiceChanged), giving the same plan for the same text and settings.

The model: a lung volume in litres, drained by every word (syllables / speech rate x 0.11 l per second of voiced phonation; voiceless
sounds cost `vl` times more; scaled by vocal effort) and refilled by breaths. A breath may only come before the text, after a sentence
end, after other punctuation, or (place="phrase") before a conjunction, a filler or a cue word, and always where an effort marker
changes the effort ([låg] / [mellan] / [hög] in the text). It is taken when the stretch up to the next possible breath would go below
the lowest volume, or at a sentence end when the next sentence would. With planning on, a breath is sized to what follows (up to the
next likely breath, plus a margin). Breath-token class: S / M / L from the inhale time (air / flow; < 0.30 s, 0.30-0.55 s, longer),
loud when the effort or the flow is high or the breath is large, else soft; a small refill (< 0.22 l) at a comma or phrase boundary is
a silent breath when the voice allows it (it breathes audibly in under half of its pauses).

  from kipp.lung import plan, simulate, voice_params, VOICES, to_text
  P = voice_params("v04")                          # one of the measured voices (VOICES), or voice_params(None) for a generic one
  words = plan("Först skär du löken. Sedan steker du den i smör, och sen häller du i grädden.", P)
  print(to_text(words))                            # the text with [Msoft] / [silent] marks before the words where breaths go
  sim = simulate(words, P)                         # estimated times: sim["words"], sim["breaths"], the volume curve sim["pts"]"""
import json
import re
from importlib import resources

PHON_FLOW = 0.11                 # l per second of voiced phonation at normal effort (the lung model's air unit)
FILLERS = {"eh", "ehm", "äh", "ähm", "öh", "öhm", "mm", "mmm", "mhm", "hm", "hmm"}
CUES = {"liksom", "typ", "alltså", "asså", "ba", "ju", "väl", "nej", "ja", "okej", "jaha", "oj",
        "wallah", "walla", "yani", "len", "bre", "mannen", "aboow", "khalas", "ayde", "jalla", "yalla", "tamam", "shoo", "salam", "inshallah"}
PHRASE_START = {"och", "men", "att", "som", "när", "om", "för", "eftersom", "så", "eller", "fast", "då", "innan", "medan", "tills",
                "därför", "utan", "vilket", "där", "dit"}
EFF_MARK = {"låg": "low", "mellan": "mid", "hög": "high", "low": "low", "mid": "mid", "high": "high"}
VOICES = json.loads((resources.files("kipp") / "weights" / "voices.json").read_text(encoding="utf-8"))
# the page's sliders (min, max, step): values snap to the step as an HTML range input does
SLIDERS = {"vc": (2, 7, 0.1), "start": (30, 90, 1), "rel": (15, 55, 1), "top": (40, 95, 1), "flow": (0.3, 3, 0.05), "eff": (0.5, 2, 0.05),
           "vl": (1, 6, 0.1), "rate": (3, 8, 0.05)}


def _snap(name, x):
    lo, hi, st = SLIDERS[name]; v = lo + round((min(hi, max(lo, x)) - lo) / st) * st
    return float(f"{min(hi, v):.10g}")


def bare(w): return re.sub(r"[^a-zåäöéü0-9]", "", w.lower())


def syllables(w):
    b = bare(w)
    if not b: return 0
    if b.isdigit(): return 2 * len(b)
    return max(1, len(re.findall(r"[aeiouyåäöéü]+", b)))


def voiceless_share(w):
    b = re.sub(r"sj|skj|stj|tj|ch|sch", "S", bare(w))
    return len(re.findall(r"[ptkfshcxS]", b)) / len(b) if b else 0.0


def punct_of(w):
    m = re.search(r"([.,!?;:–-])[\"')\]]*$", w); return m.group(1) if m else ""


def word_air(dur, vl, P): return dur * PHON_FLOW * P["eff"] * ((1 - vl) + vl * P["vl"])


def pause_after(p): return 0.18 if p == "," else 0.38 if p in (".", "!", "?") else 0.22 if p else 0.0


def voice_params(voice=None, **over):
    """the page's settings for a voice (its slider defaults; voiceChanged), any of them overridable:
    vc (vital capacity, l), start / rel / top (start, lowest and highest volume, share of vc), flow (inhale l/s), eff (effort), vl (cost of
    voiceless sounds), rate (syllables / s), plan, first (a breath before the text), silent_ok, place ("punct" | "phrase" | "any")"""
    v = VOICES.get(voice, {}) if isinstance(voice, str) else (voice or {})
    clamp = lambda x, a, b: min(b, max(a, x))  # noqa: E731
    vc = 4.8 if v.get("gender", "m") == "m" else 3.6; grp = (v.get("group_phon") or 3.5) * PHON_FLOW
    P = {"vc": _snap("vc", vc), "rate": _snap("rate", clamp(v.get("syll_per_s", 5.0), 3, 8)), "rel": _snap("rel", 35) / 100,
         "top": _snap("top", round(clamp(35 + 130 * grp / vc, 40, 95))) / 100,
         "flow": _snap("flow", float(f"{clamp(grp / max(0.15, v.get('breath_s', 0.4)), 0.3, 3):.2f}")),
         "silent_ok": v.get("breath_rate", 0.5) < 0.5, "start": _snap("start", 55) / 100, "eff": 1.0, "vl": 2.5,
         "plan": True, "first": True, "place": "phrase"}
    P.update(over); return P


def eff_words(text):
    """[låg] / [mellan] / [hög] set the effort of the words after them; a change of effort forces a breath"""
    out, cur, changed = [], None, False
    for w in text.split():
        m = re.match(r"^\[(\S+)\]$", w); e = EFF_MARK.get(m.group(1).lower()) if m else None
        if e: changed = changed or e != cur; cur = e; continue
        out.append({"w": w, "eff": cur, "eff_change": changed and len(out) > 0}); changed = False
    return out


def candidate(words, i, P):
    if i == 0: return "start"
    p = punct_of(words[i - 1]["w"])
    if p and p in ".!?": return "sentence"
    if p: return "punct"
    if P["place"] == "punct": return None
    b = bare(words[i]["w"])
    if P["place"] == "any": return "any"
    return "phrase" if b in PHRASE_START or b in FILLERS or b in CUES else None


def classify(dv, dur, P, kind):
    if P["silent_ok"] and dv < 0.22 and kind in ("punct", "phrase"): return "silent"
    size = "S" if dur < 0.30 else "M" if dur < 0.55 else "L"
    loud = P["eff"] >= 1.25 or P["flow"] >= 1.8 or dv / P["vc"] > 0.16
    return size + ("loud" if loud else "soft")


def plan(text, P=None):
    """text -> [{"w", "eff", "syl", "vl", "dur", "air", "br" (breath class before this word, or absent), "dv" (l), "bdur" (s), ...}]"""
    P = P or voice_params(None)
    words = []
    for x in eff_words(text):
        w = x["w"]; syl = syllables(w); vl = voiceless_share(w)
        words.append(dict(x, syl=syl, vl=vl, dur=syl / P["rate"] if syl else 0.05, filler=bare(w) in FILLERS, cue=bare(w) in CUES))
    for o in words:
        if o["filler"]: o["dur"] = max(o["dur"], 0.35)
        o["air"] = word_air(o["dur"], o["vl"], P)
    VC = P["vc"]; low, top = P["rel"] * VC, P["top"] * VC
    cands = [candidate(words, i, P) or ("sentence" if o["eff_change"] else None) for i, o in enumerate(words)]

    def air_to(i, stop):
        a = 0.0
        for j in range(i, len(words)):
            if j > i and stop(j): break
            a += words[j]["air"]
        return a
    V = P["start"] * VC
    for i in range(len(words)):
        kind = cands[i]
        if kind:
            to_next = air_to(i, lambda j: bool(cands[j])); to_sent = air_to(i, lambda j: cands[j] in ("sentence", "start"))
            if kind == "start": breathe = P["first"]
            elif words[i]["eff_change"]: breathe = True
            elif V - to_next < low: breathe = True
            elif kind == "sentence" and V - to_sent < low + 0.25 * (top - low): breathe = True
            else: breathe = False
            if breathe:
                target = top
                if P["plan"]:
                    a = 0.0
                    for j in range(i, len(words)):
                        if j > i and cands[j] == "sentence" and a > 0.35 * (top - low): break
                        if a + words[j]["air"] > top - low: break
                        a += words[j]["air"]
                    target = min(top, max(V + 0.12, low + a + 0.08 * VC))
                dv = max(0.0, target - V)
                if dv >= 0.05 or kind == "start" or words[i]["eff_change"]:
                    dur = max(0.15, dv / P["flow"])
                    words[i]["br"] = classify(dv, dur, P, kind); words[i]["dv"] = dv
                    words[i]["bdur"] = max(0.12, dur * 0.6) if words[i]["br"] == "silent" else dur
                    V = target
        V -= words[i]["air"]
    return words


def simulate(words, P):
    """the planned breaths walked at estimated times -> {"words": [{s, e}], "breaths": {i: {s, e, cls, dv}}, "pts": [[t, V]], "end", "minV"}"""
    VC = P["vc"]; V = P["start"] * VC; t = 0.0; pts = [[0.0, V]]; out = {"words": [], "breaths": {}}
    for i, o in enumerate(words):
        if o.get("br"):
            s, e = t, t + o["bdur"]
            pts.append([s, V]); V = min(VC, V + o["dv"]); pts.append([e, V]); out["breaths"][i] = {"s": s, "e": e, "cls": o["br"], "dv": o["dv"]}; t = e
        s, e = t, t + o["dur"]
        pts.append([s, V]); V -= o["air"]; pts.append([e, V]); out["words"].append({"s": s, "e": e})
        t = e + pause_after(punct_of(o["w"])) * (0.3 if i + 1 < len(words) and words[i + 1].get("br") else 1)
    out["pts"] = pts; out["end"] = max(t, max(p[0] for p in pts)); out["minV"] = min(p[1] for p in pts)
    return out


def to_text(words):
    return " ".join((f"[{o['br']}] " if o.get("br") else "") + o["w"] for o in words)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="kipp-plan", description="Kipp lung model: where a speaker breathes in a Swedish text")
    ap.add_argument("text"); ap.add_argument("--voice", help=f"one of {', '.join(VOICES)} (default: a generic voice)")
    ap.add_argument("--place", default="phrase", choices=["phrase", "punct", "any"]); ap.add_argument("--effort", type=float)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv); P = voice_params(a.voice, place=a.place, **({"eff": a.effort} if a.effort else {}))
    w = plan(a.text, P)
    print(json.dumps(w, ensure_ascii=False, indent=1) if a.json else to_text(w))
