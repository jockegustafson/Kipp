"""Kipp model code: the breath detector of the KTH svtx toolkit (svtx/breath.py), unchanged in what it computes.
A BiGRU over WavLM-large layers 6, 8, 10 (averaged) plus the frame log-energy relative to the clip's 90th percentile, 20 ms frames,
classes none / in-breath / out-breath / disfluent breath. A breath is looked for in the pauses between a speaker's words."""
import numpy as np
import torch

SR, FR, MARG, MAXCLIP = 16000, 0.02, 0.6, 4.0
LAB = {"none": 0, "in": 1, "out": 2, "disfluent": 3}
DUR_BINS = (0.30, 0.55)


class GRU(torch.nn.Module):
    def __init__(s, d=1025):
        super().__init__()
        s.inp = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Dropout(0.3), torch.nn.Linear(d, 192), torch.nn.GELU())
        s.rnn = torch.nn.GRU(192, 96, batch_first=True, bidirectional=True)
        s.out = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(192, 4))

    def forward(s, x): return s.out(s.rnn(s.inp(x))[0])


def events(P, g0, g1, thr=0.5, keep_out=False):
    """frame probabilities -> breath events inside the pause [g0, g1] (s, clip time)"""
    pb = 1 - P[:, 0]; t = (np.arange(len(P)) + 0.5) * FR; inside = (t >= g0) & (t <= g1)
    m = (pb > thr) & inside; ev, i = [], 0
    while i < len(m):
        if not m[i]: i += 1; continue
        j = i
        while j + 1 < len(m) and m[j + 1]: j += 1
        if (j - i + 1) * FR >= 0.08:
            cls = int(np.argmax(P[i:j + 1, 1:].sum(0))) + 1
            lab = [k for k, v in LAB.items() if v == cls][0]
            if lab != "out" or keep_out: ev.append({"s": round(i * FR, 2), "e": round((j + 1) * FR, 2), "label": lab, "p": round(float(pb[i:j + 1].mean()), 3)})
        i = j + 1
    return ev


def pauses(words, min_gap=0.25, max_gap=3.0):
    """words: [(start, end, ...)] of one speaker -> pauses [(kind, s, e)]"""
    ws = sorted((float(w[0]), float(w[1])) for w in words); out = []
    for (a0, a1), (b0, b1) in zip(ws, ws[1:]):
        g = b0 - a1
        if min_gap <= g <= max_gap: out.append(("pause", a1, b0))
        elif g > max_gap: out += [("turnend", a1, a1 + 1.2), ("prestart", b0 - 1.2, b0)]
    return out


def classify(x, breaths, words=None):
    """token class per breath: S/M/L by duration, soft/loud by level re this recording's median breath level"""
    if not breaths: return breaths
    hop = int(0.02 * SR); db = lambda a, b: 10 * np.log10(np.mean(x[int(a * SR):int(b * SR)] ** 2) + 1e-12)  # noqa: E731
    sp = [db(w[0], w[1]) for w in (words or [])[:400] if w[1] - w[0] > 0.1] or [np.percentile(
        [10 * np.log10(np.mean(x[i:i + hop] ** 2) + 1e-12) for i in range(0, len(x) - hop, hop)], 90)]
    ref = float(np.median(sp))
    for b in breaths: b["db"] = round(float(db(b["s"], b["e"]) - ref), 1)
    med = float(np.median([b["db"] for b in breaths if b["label"] != "out"] or [b["db"] for b in breaths]))   # as svtx: out-breaths left out
    for b in breaths:
        d = b["e"] - b["s"]; size = "S" if d < DUR_BINS[0] else "M" if d < DUR_BINS[1] else "L"
        b["cls"] = "disfl" if b["label"] == "disfluent" else "out" if b["label"] == "out" else size + ("loud" if b["db"] > med else "soft")
    return breaths
