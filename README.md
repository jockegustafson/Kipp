# Kipp

**Breath detection for Swedish speech.** Kipp finds the audible **in-breaths** and **disfluent breaths** (breaths inside a hesitation)
in the pauses of a recording, with their start and end, and sorts each into a size and loudness class relative to the speaker, ready
for breath-aware speech synthesis or for studying breathing in dialogue. In cross-validation on 464 pauses labelled by ear it finds
breaths at **precision 0.87 and recall 0.88**. *Kippa efter andan* is Swedish for gasping for breath.

```bash
pip install "kipp @ git+https://github.com/jockegustafson/Kipp.git"            # + "kipp[vad] @ ..." to run without word times
kipp interview.wav words.tsv --channel 0 -o breaths.tsv    # words of ONE speaker: TSV (word, start, end) or a Praat TextGrid
kipp interview.wav -o breaths.tsv                          # no words: Silero VAD finds the speech
```

```
start    end      label      p      class   level_db  pause
```

**Breath types.** `label` is `in` (in-breath) or `disfluent` (a breath inside a hesitation), and with `--keep-out` also `out`
(out-breath; the model was trained with out-breaths counted as none, so expect lower recall for them). `class` is the breath-token class
used for speech synthesis: S / M / L (shorter than 0.30 s, 0.30–0.55 s, longer) × soft / loud (below or above this recording's median
breath level, out-breaths left out), `disfl` or `out`; `pause` says where the breath sat: between two words (`pause`), after the last word of a
turn (`turnend`) or before the first word of the next one (`prestart`).

## Run it from Python

```python
from kipp import BreathDetector

det = BreathDetector()                                         # GPU if there is one, Apple silicon (mps), else CPU
words = [("jag", 12.31, 12.40), ("tror", 12.40, 12.62), ...]   # (word, start s, end s) of one speaker
for b in det.detect_file("interview.wav", words, channel=0):
    print(b["s"], b["e"], b["label"], b["cls"], b["p"])
```

## How it works

- **Where it looks:** only in pauses, from the speaker's word times: every gap of 0.25–3 s between two words, and for longer gaps the
  1.2 s after the last word and the 1.2 s before the next one (where the in-breath before a new breath group sits). Each pause is cut
  with 0.6 s of speech on both sides (at most 4 s) and peak-normalised, as the training clips were.
- **Model:** WavLM-large (`microsoft/wavlm-large`) layers 6, 8 and 10 averaged, plus each 20 ms frame's energy relative to the clip;
  LayerNorm → Linear 192 → a bidirectional GRU (2 × 96) → four classes per frame: none, in-breath, out-breath, disfluent breath.
  A breath is a run of frames with P(breath) > 0.5 inside the pause, at least 80 ms long; out-breaths are not reported.

## Training and evaluation

Built in four listening rounds: an acoustic rule proposed candidates (unvoiced noise at least 6 dB above the pause floor), the owner
labelled them on a spectrogram page (in-breath, out-breath, disfluent, none), the detector was retrained, and the next round took the
pauses it was least sure of plus random ones. 464 labelled pauses from five kinds of Swedish speech: remote-call dialogue,
podcast, face-to-face dialogue, read speech and radio interviews; a fourth round re-checked the 89
pauses where model and label disagreed.

| Step | Labels | Model | Precision / recall |
| --- | --- | --- | --- |
| round 0 | 164 | rule | 0.67 / 0.86 |
| round 0 | 164 | frame MLP on WavLM layer 8 | 0.90 / 0.85 |
| rounds 0–2 | 464 | frame MLP | 0.81 / 0.77 |
| rounds 0–2 | 464 | BiGRU, WavLM layers 6/8/10 + level | 0.83 / 0.84 |
| **Kipp** | 464 | the same, out-breaths counted as none | **0.87 / 0.88** |

Precision and recall are 5-fold cross-validation over the labelled clips, with one listener's labels; there is no separate held-out
test set of new speakers yet. Peak normalisation matters: without it, recall on new recordings fell to about 0.4.

The model is the breath detector of the KTH svtx toolkit (`svtx breath`), used there to put breath tokens into Swedish TTS training
data.

## Licence

Non-commercial research use only (see `LICENSE`). Please cite this repository. Questions: Joakim Gustafson, KTH Speech, Music and
Hearing.
