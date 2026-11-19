# Kipp: breath detection, breathing profiles and breath planning for Swedish speech

Joakim Gustafson (KTH Speech, Music and Hearing) with Claude Code. Status 2026-11-19. Models: the breath detector and the lung model of
the KTH svtx toolkit and its breathing-TTS work.

## Abstract

Kipp has three parts. (1) A **breath detector** finds audible in-breaths and disfluent breaths in the pauses of Swedish speech: a
bidirectional GRU over frozen WavLM-large layers and frame energy, trained in four listening rounds on 464 pauses from five kinds of
speech; in 5-fold cross-validation it reaches **precision 0.87 and recall 0.88**, against 0.67 / 0.86 for an acoustic rule. (2) A
**lung model** plans breaths for speech synthesis: a lung volume drained by the words and refilled by breaths, placing each breath and
choosing its token class (S / M / L × soft / loud, silent). Fitted on 818k pauses and 346k breath groups from ten corpora, the
corresponding statistical model predicts on held-out speakers whether a pause holds a breath at **AUC 0.78** (pause length alone 0.50).
(3) A **breathing profile** from a recording (speech rate, breaths per pause, breath length and level, voiced time per breath group) and
a one-breath **calibration** set the lung model for a new speaker.

## 1. Background and related work

**Speech breathing.** Speech uses a strongly asymmetric breathing cycle: a quick in-breath (about 10 % of the cycle) and a long
expiratory phase carrying the speech (Fuchs & Rochet-Capellan, 2021). Conversational speech typically starts a breath group around
55–65 % of vital capacity and ends near the resting expiratory level; loud speech starts higher and uses more (Hixon, Goldman & Mead,
1973; Winkworth et al., 1995). Speakers plan: they breathe in more, and for longer, before longer and louder utterances (Fuchs, Petrone,
Krivokapić & Hoole, 2013). In conversation, breathing is coordinated with turn-taking: in-breaths cluster at turn starts and holding the
floor (Rochet-Capellan & Fuchs, 2014; Włodarczak & Heldner, 2017; Aare, Włodarczak & Heldner, 2015). In-breaths are noisy, unvoiced,
vowel-like sounds whose loudness and spectral centre rise with inhalation speed (Werner, Fuchs, Trouvain & Möbius, 2021).

**Breath detection.** Early detectors used templates or hand-crafted features (Ruinskiy & Lavner, 2007). For speech synthesis from
found conversational data, Székely, Henter & Gustafson (2019) trained a CNN-LSTM breath detector on mel-spectrograms and zero-crossing
rate to segment spontaneous speech into breath groups, and Székely, Henter, Beskow & Gustafson (2019, 2020) showed that synthesising
breaths in context makes spontaneous TTS more natural and that breathing reflects speech planning. Frame-wise breath detection with
self-training has since been proposed (arXiv:2402.00288). Detection results are reported on different data and with different
definitions (frames vs events, speaker-dependent vs independent), so they are not comparable with ours; **we know of no published breath
detector trained on Swedish**.

**Breathing in TTS.** Most TTS treats breaths as silences or sounds tied to punctuation. Breath-group training (Székely et al., 2020) and
explicit respiratory state models (the literature review behind this work, *Speech Breathing Synthesis Models*, 2026) motivate the lung
model here: an explicit air budget decides where breaths go and how big they are.

## 2. The breath detector

**Model.** WavLM-large (Chen et al., 2022) layers 6, 8 and 10 averaged, plus each 20 ms frame's log energy relative to the clip's 90th
percentile → LayerNorm → dropout 0.3 → Linear 192 → GELU → bidirectional GRU (2 × 96) → dropout 0.2 → Linear 4 (none, in-breath,
out-breath, disfluent breath). AdamW, learning rate 10⁻³, weight decay 10⁻², 40 epochs, batches of 16 clips. Out-breaths were counted as
none in the released model (speech synthesis needs in-breaths).

**Where it looks.** Only in pauses, from the speaker's word times: gaps of 0.25–3 s, and for longer gaps the 1.2 s after the last word
and the 1.2 s before the next. Each pause is cut with 0.6 s of speech on each side (at most 4 s) and peak-normalised; a breath is a run of
frames with P(breath) > 0.5 inside the pause, at least 80 ms long.

**Data and rounds.** Round 0: an acoustic rule proposed candidates (unvoiced frames, normalised autocorrelation < 0.45 at 60–400 Hz, at
least 6 dB above the file's pause floor, 80–1200 ms), 40 clips per corpus group; rounds 1–2: the detector chose the next clips, half where
it was least sure and half at random; round 3: a second look at the 89 clips where model and label disagreed (62 kept, 27 changed).
One listener labelled each pause on a spectrogram page as in-breath, out-breath, disfluent or none.

| Speech | Recording |
| --- | --- |
| remote-call dialogue | two speakers, one channel each |
| podcast | one host's channel |
| face-to-face dialogue | lab recordings, close-talk microphones |
| read speech | one reader |
| radio interviews | broadcast |

**Results** (5-fold cross-validation over the labelled clips):

| Step | Labels | Model | Precision | Recall | Breath type correct |
| --- | --- | --- | --- | --- | --- |
| round 0 | 164 | acoustic rule | 0.67 | 0.86 | – |
| round 0 | 164 | frame MLP, WavLM layer 8 (±2 frames) | 0.90 | 0.85 | 0.84 |
| rounds 0–1 | 314 | frame MLP | 0.85 | 0.84 | 0.87 |
| rounds 0–2 | 464 | frame MLP | 0.81 | 0.77 | 0.89 |
| rounds 0–2 | 464 | BiGRU, WavLM 6/8/10 + level | 0.83 | 0.84 | 0.90 |
| **Kipp** | 464 | BiGRU, out-breaths as none | **0.87** | **0.88** | – |

A blind listening check on corpus output found all 5 detected breaths audible. On new recordings recall was 0.37–0.40 until the clips
were peak-normalised as the training clips were; then 0.70. The package reproduces the svtx detector: on a 48-minute podcast it found
469 breaths where svtx found 467, with 465 matching within 50 ms and all matched ones in the same class.

## 3. The lung model

**State.** A lung volume in litres (vital capacity 4.8 l for a male, 3.6 l for a female voice by default). Every word drains it:
syllables / speech rate × 0.11 l per second of voiced phonation (about 110 ml/s), voiceless sounds `vl` times more, scaled by vocal
effort. Breaths refill it.

**Where breaths go.** A breath may come before the text, after a sentence end or other punctuation, before a conjunction, filler or cue
word (the default; or only at punctuation, or anywhere), and always where an effort marker changes the effort. It is taken when the
stretch to the next possible breath would go below the lowest volume (35 % of capacity by default), or at a sentence end when the next
sentence would; with planning on, it is sized to what follows (up to the next likely breath plus a margin), never above the highest volume
(from one breath group's air, 40–95 %).

**Token class.** S / M / L from the inhale time (air / inhale flow; < 0.30 s, 0.30–0.55 s, longer); loud when the effort is ≥ 1.25,
the flow ≥ 1.8 l/s or the breath > 16 % of capacity; a refill under 0.22 l at a comma or phrase boundary is a *silent* breath when the
voice breathes audibly in under half of its pauses.

**The fitted model behind it** (from the detector's breaths in face-to-face and remote-call dialogue, a podcast, radio interviews,
a radio panel show, radio monologues and parliament speeches; 818k checked pauses, 346k breath groups; boosted trees, 5 folds by speaker):

| Speech | Pause length alone (AUC) | Air spent, time since the last breath, speech ahead (AUC) |
| --- | --- | --- |
| face-to-face dialogue | 0.72 | 0.71 |
| remote-call dialogue | 0.67 | 0.74 |
| podcast | 0.66 | 0.78 |
| parliament speeches (two sets) | 0.62–0.63 | 0.76–0.77 |
| radio monologues | 0.57 | 0.69 |
| radio interviews (two sets) | 0.47–0.48 | 0.81–0.82 |
| radio panel show | 0.39 | 0.80 |
| all, pooled | 0.50 (air spent alone 0.70) | **0.78** |

With the detected breaths hidden and as many put back per file as were there, 68 % land in the right pause (pause length alone 44 %,
chance 41 %). The breath's length class is predicted right 68 % of the time on held-out speakers (majority class 54 %), its level class
60 % (50 %). Breathing is individual: 54 face-to-face-dialogue speakers recorded in more than one session keep their median breath length (r = 0.76),
breath level (r = 0.62) and speech per breath group (r = 0.64) across sessions. Broadcast editing shortens pauses, which is why pause
length alone predicts so little in the radio interviews and the panel show.

**The planner in this package** is the rule model of the KTH breathing-TTS demo (lung volume, candidates, sizing, classes), ported to
Python; on 336 cases (4 texts × 7 voices × 3 placement rules × 2 effort levels × planning on/off) it gives the same breaths, classes,
sizes and times as the demo. The 18 voice profiles were measured with the detector on the corpora above, plus radio monologues,
a cooking show, book readings and audiobooks.

## 4. Breathing profiles and calibration

A profile is measured from a recording as the voice profiles were: speech rate (syllables per second of speech), the share of
mid-speech pauses (0.25–3 s) with an audible in-breath, median breath length and level, and median voiced seconds between two in-breaths.
From these, by the lung model's rules: air per breath group = voiced seconds × 0.11 l/s, inhale flow = that air / breath length, highest
volume from the group's air. Without word times, speech stretches come from voicing (voiced runs joined across gaps under 0.25 s; on 10
minutes of a two-speaker podcast this recovered 86 of the 167 word-time pauses, against at most 42 for Silero VAD) and syllables from
loudness peaks in voiced stretches (de Jong & Wempe, 2009; 0.96 of the transcript's syllables on the audio its thresholds were tuned on, so optimistic). With word times the
profile of that podcast host came out at 6.0 syllables/s, 47 % of pauses with a breath, 0.32 s breaths and 2.7 voiced seconds per group,
close to the measured podcast voice in the package (5.7, 43 %, 0.36 s, 3.5 s).

**Calibration.** The speaker breathes out, takes the deepest breath they can, and talks for as long as they can on it. The voiced
seconds until the next in-breath, × 0.11 l/s, give the most air the speaker uses on one breath in the model's units; taken to span 20–90 %
of capacity, they set the model's capacity, and the deep breath's length the full-effort inhale flow. These are model values that make
the planner fit a speaker, not lung-function measurements: vital capacity cannot be heard.

## 5. Limitations

- One listener's labels, 464 pauses; precision and recall are cross-validated over clips, not a held-out test of new speakers.
- Out-breaths are weak (counted as none in training).
- The lung model is an air budget, not physiology: there is no plethysmography in the data, and the 0.11 l/s unit, the capacities and
  the volume limits come from the literature, not from these speakers.
- Profiles without word times are estimates; edited recordings (podcasts, broadcast) often have breaths removed or pauses shortened.

## 6. Reproduction

Detector training and the lung-model fits are in the KTH svtx repository (`tts/breath/`, `lung_table.py`, `lung_fit.py`; TTS dialogue
report sections 17–23). This package holds the inference code, the detector weights (hash-checked), the 18 voice profiles, the planner
and its parity test against the demo (`tools/`).

## References

- Aare, K., Włodarczak, M., & Heldner, M. (2015). Inhalation amplitude and turn-taking in spontaneous Estonian conversations. *Fonetik
  2015*. [link](https://konferens.ht.lu.se/fileadmin/user_upload/sol/ovrigt/konferens_fonetik2015/docs/WP55_Aare.pdf)
- Chen, S., et al. (2022). WavLM: large-scale self-supervised pre-training for full stack speech processing. *IEEE Journal of Selected
  Topics in Signal Processing*, 16(6).
- de Jong, N. H., & Wempe, T. (2009). Praat script to detect syllable nuclei and measure speech rate automatically. *Behavior Research
  Methods*, 41(2), 385–390.
- Frame-wise breath detection with self-training (2024). arXiv:2402.00288. [link](https://arxiv.org/html/2402.00288v1)
- Fuchs, S., Petrone, C., Krivokapić, J., & Hoole, P. (2013). Acoustic and respiratory evidence for utterance planning in German.
  *Journal of Phonetics*, 41(1), 29–47.
- Fuchs, S., & Rochet-Capellan, A. (2021). The respiratory foundations of spoken language. *Annual Review of Linguistics*, 7, 13–30.
- Hixon, T. J., Goldman, M. D., & Mead, J. (1973). Kinematics of the chest wall during speech production: volume displacements of the
  rib cage, abdomen, and lung. *Journal of Speech and Hearing Research*, 16(1), 78–115.
- Rochet-Capellan, A., & Fuchs, S. (2014). Take a breath and take the turn: how breathing meets turns in spontaneous dialogue.
  *Philosophical Transactions of the Royal Society B*, 369(1658).
- Ruinskiy, D., & Lavner, Y. (2007). An effective algorithm for automatic detection and exact demarcation of breath sounds in speech and
  song signals. *IEEE Transactions on Audio, Speech, and Language Processing*, 15(3), 838–850.
- Székely, É., Henter, G. E., & Gustafson, J. (2019). Casting to corpus: segmenting and selecting spontaneous dialogue for TTS with a
  CNN-LSTM speaker-dependent breath detector. *ICASSP 2019*.
- Székely, É., Henter, G. E., Beskow, J., & Gustafson, J. (2019). Spontaneous conversational speech synthesis from found data.
  *Interspeech 2019*. [link](https://people.kth.se/~ghe/pubs/pdf/szekely2019spontaneous.pdf)
- Székely, É., Henter, G. E., Beskow, J., & Gustafson, J. (2020). Breathing and speech planning in spontaneous speech synthesis.
  *ICASSP 2020*. [link](https://www.speech.kth.se/connected/papers/2020_ICASSP_Breathing%20and%20speech%20planning%20in%20spontaneous%20speech%20synthesisbreathing.pdf)
- Werner, R., Fuchs, S., Trouvain, J., & Möbius, B. (2021). Inhalations in speech: acoustic and physiological characteristics.
  *Interspeech 2021*.
- Winkworth, A. L., Davis, P. J., Adams, R. D., & Ellis, E. (1995). Breathing patterns during spontaneous speech. *Journal of Speech and
  Hearing Research*, 38(1), 124–144.
- Włodarczak, M., & Heldner, M. (2017). Respiratory constraints in verbal and non-verbal communication. *Frontiers in Psychology*, 8, 708.
