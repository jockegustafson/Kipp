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

## The lung model: where a speaker breathes in a text

`kipp.lung` plans breaths for speech synthesis: given a text and a voice, where the speaker breathes, how much air each breath takes,
and its breath-token class. It is the planner of the KTH breathing-TTS demo, ported to Python (the same plan as the page for the same
settings: identical on 336 test cases).

```bash
kipp-plan "Först skär du löken. Sedan steker du den i smör, och sen häller du i grädden." --voice v15
# [Ssoft] Först skär du löken. [Ssoft] Sedan steker du den i smör, [Msoft] och sen häller du i grädden.
```

```python
from kipp.lung import plan, simulate, voice_params, to_text, VOICES
P = voice_params("v15", place="phrase")      # 18 measured voices (VOICES) or voice_params(None); every setting can be overridden
words = plan("[låg] Det var en gång ... [hög] Plötsligt hörde hon ett ljud!", P)
print(to_text(words)); sim = simulate(words, P)   # estimated word and breath times and the lung-volume curve
```

- **The model:** a lung volume in litres, drained by every word (syllables / speech rate × 0.11 l per second of voiced phonation;
  voiceless sounds cost more; scaled by vocal effort) and refilled by breaths. A breath may come before the text, after a sentence
  end or other punctuation, before a conjunction, filler or cue word (`place="phrase"`, the default; `"punct"` or `"any"` otherwise), and
  always where an effort marker (`[låg]`, `[mellan]`, `[hög]`) changes the effort. It is taken when the stretch to the next possible
  breath would go below the lowest volume, or at a sentence end when the next sentence would, and sized to what follows.
- **Classes:** the same breath tokens the detector gives: S / M / L by inhale time, soft / loud by effort, flow and size, and
  `silent` for a small refill at a comma or phrase boundary when the voice breathes audibly in under half of its pauses.
- **Voices:** 18 profiles measured with the detector on Swedish corpora (podcast, remote-call and face-to-face dialogue, radio
  interviews and monologue, parliament speech, a cooking show, book readings, audiobooks): speech rate, share of pauses with an audible
  breath, median breath length and level, and voiced seconds per breath group. They set the defaults, as the demo's sliders do:
  vital capacity by gender (4.8 / 3.6 l), the highest volume from one breath group's air, the inhale flow from the breath length.

## Try it in the browser: your own breathing

```bash
kipp-demo                       # http://localhost:8766 (everything runs on your machine)
```

Three tabs:
1. **Breaths in a recording.** Record yourself or load a file (optionally with a word file). The page draws the spectrogram with each
   breath as a box coloured by its class, lists the breaths, and measures your **breathing profile**: speech rate, the share of pauses
   with an audible in-breath, breath length and level, voiced seconds per breath group, and from these, by the lung model's rules, the air
   per breath group, your inhale speed and the highest volume. While you record, the profile updates every 5 s.
2. **Calibrate.** Breathe out, press Start, take your deepest breath, and talk for as long as you can on it; breathe and press Stop. The
   voiced seconds on that one breath give the most air you use on a breath (in the model's units), which sets the model's lung capacity
   for your voice, and the deep breath's length your full-effort inhale speed. Model values for the planner, not a lung-function test.
3. **Plan breaths for a text.** Type a text, choose one of the measured voices or **your voice** (from tabs 1–2), and see where you
   would breathe, each breath's class, and the lung volume over the text.

From Python: `kipp.profile.estimate(x, breaths, words)` and `kipp.profile.calibrate(x, breaths)`. Without word times, pauses come from
voicing and syllables from loudness peaks (de Jong & Wempe, 2009): estimates that work best with one voice on a close microphone and 30 s
or more of speech; the page shows how many breaths and pauses each number rests on. Edited podcasts often have their breaths removed.

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
