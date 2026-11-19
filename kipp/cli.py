"""kipp AUDIO [WORDS] [--channel N] [--tier NAME] [--thr 0.5] [--keep-out] [--device cuda|mps|cpu] [-o out.tsv] [--json out.json]

WORDS: the words of ONE speaker (a breath is looked for in the pauses between them), as a Praat TextGrid (the first interval tier, or
--tier NAME) or a TSV/CSV with a header naming start, end and word (or without a header: start, end, word). Without WORDS, Silero VAD
finds the speech (pip install "kipp[vad]"). Writes TSV: start, end, label, p, class, level_db, pause."""
import argparse
import csv
import json
import sys


def read_words(path, tier=None):
    if str(path).lower().endswith(".textgrid"):
        import re
        txt = open(path, encoding="utf-8", errors="replace").read()
        for t in re.split(r'item \[\d+\]:', txt)[1:]:
            name = re.search(r'name = "(.*?)"', t).group(1)
            if 'IntervalTier' not in t or (tier and name != tier): continue
            iv = re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', t, re.S)
            return [(w.strip(), float(a), float(b)) for a, b, w in iv if w.strip()]
        sys.exit(f"no interval tier {tier or ''} in {path}")
    rows = [r for r in csv.reader(open(path, encoding="utf-8"), delimiter="\t" if str(path).endswith(".tsv") else ",") if r]
    head = [c.strip().lower() for c in rows[0]] if rows else []
    if "start" in head and "end" in head:
        i, j = head.index("start"), head.index("end"); k = head.index("word") if "word" in head else head.index("label")
        return [(r[k], float(r[i]), float(r[j])) for r in rows[1:]]
    out = []
    for r in rows:
        try: out.append((r[2], float(r[0]), float(r[1])))
        except (ValueError, IndexError): continue
    return out


def main():
    ap = argparse.ArgumentParser(description="Kipp: in-breaths and disfluent breaths in the pauses of Swedish speech")
    ap.add_argument("audio"); ap.add_argument("words", nargs="?"); ap.add_argument("--channel", type=int); ap.add_argument("--tier")
    ap.add_argument("--thr", type=float); ap.add_argument("--keep-out", action="store_true", help="also report out-breaths"); ap.add_argument("--device"); ap.add_argument("-o", "--out"); ap.add_argument("--json")
    a = ap.parse_args()
    from . import BreathDetector
    res = BreathDetector(a.device).detect_file(a.audio, read_words(a.words, a.tier) if a.words else None, channel=a.channel, thr=a.thr, keep_out=a.keep_out)
    f = open(a.out, "w", newline="") if a.out else sys.stdout; w = csv.writer(f, delimiter="\t")
    w.writerow(["start", "end", "label", "p", "class", "level_db", "pause"])
    for b in res: w.writerow([f'{b["s"]:.3f}', f'{b["e"]:.3f}', b["label"], b["p"], b["cls"], b["db"], b["pause"]])
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
