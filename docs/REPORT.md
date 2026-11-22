# Kipp: breath detection, breathing profiles and breath planning for Swedish speech

Joakim Gustafson (KTH Speech, Music and Hearing) with Claude Code. Status 2026-11-22 (Kipp 1.3). Models: the breath detector and the lung model of
the KTH svtx toolkit and its breathing-TTS work.

## Abstract

Kipp has three parts. (1) A **breath detector** finds audible in-breaths and disfluent breaths in the pauses of Swedish speech: a
bidirectional GRU over frozen WavLM-large layers and frame energy, trained in four listening rounds on 464 pauses from five kinds of
speech; in 5-fold cross-validation it reaches **precision 0.87 and recall 0.88**, against 0.67 / 0.86 for an acoustic rule. (2) A
**lung model** plans breaths for speech synthesis: a lung volume drained by the words and refilled by breaths, placing each breath and
choosing its token class (S / M / L × soft / loud, silent). Fitted on 818k pauses and 346k breath groups from ten corpora, the
corresponding statistical model predicts on held-out speakers whether a pause holds a breath at **AUC 0.78** (pause length alone 0.50).
(3) A **breathing profile** from a recording (speech rate, breaths per pause, breath length and level, voiced time per breath group) and
a one-breath **calibration** set the lung model for a new speaker. (4) Since 1.3, **how much air each breath brings in**, fitted by
air balance over 330 hours of spontaneous Swedish conversation (the summed volumes of five consecutive breaths against the air spent
between them; within-speaker r = 0.46 on held-out speakers, breath length alone 0.35), a hidden lung-state model that infers silent
refills, and breath-class models fitted on the same data that the planner now uses by default (audible breath AUC 0.77; length class
65 %, level class 60 %, majorities 50 %).

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

## 3.1 How much air a breath brings in (Kipp 1.3)

**Why new tables.** The breath tables behind section 3 came from a speech-synthesis manifest, which drops a speaker's unusable words
(overlap, laughter, a cap on a host's speech), and for single-channel mixes the detector had only been run on pauses inside one manifest
segment: podcasts and broadcast programmes showed 6–10 detected breaths per minute of speech against 19 in face-to-face dialogue. For 1.3 the
tables were rebuilt from complete own-word timelines (every word the speaker said, usable or not) with every own pause through the detector.
The detector was run in padded batches with an attention mask, which gives the same breaths as one clip at a time (471 of 471 on a check
set), and earlier detections of the same clip were reused.

**Data.** 30 corpora of spontaneous Swedish conversation: face-to-face and remote-call dialogue, a podcast host's own track, radio
interviews, panel and talk shows, podcasts, and a three-party quiz game (read and scripted speech left out). Per corpus, up to 12 hours of
the speakers' own speech, speaker files ranked by their speech in long same-speaker stretches (at least 20 s with no own pause over 3 s and
no more than 1 s of other speech in a pause), at most 1.5 h per speaker: about 330 hours, 969 speaker files. Breath rates are now 11–16
per minute of speech in the podcasts and broadcast corpora.

**Air balance.** Over K consecutive breath cycles of one speaker the air breathed in must match the air spent in speech, so a volume model
can be fitted without a spirometer:

    log V_b = log s_speaker + a log t_b + c level_b + spectrum terms        (t = breath length, level re the speaker's speech)
    S_w = sum over the speech of the window of (voiced phone time + w x voiceless phone time) x 10^(k dB / 10)
    fit: within-speaker deviations of log sum_b V_b - log S_w, over windows of K detected breaths inside one stretch

The speaker scale s (lung size, microphone, units) is free, so volumes are relative. Planning noise (how much of a breath a speaker uses
before the next) averages out over K cycles, which a test on the next breath group alone cannot do. 15,850 non-overlapping windows of 5
breaths from 521 speakers; 5 folds by speaker; within-speaker r on held-out speakers:

| Breath measures | K = 1 | K = 3 | K = 5 | K = 8 |
| --- | --- | --- | --- | --- |
| length | 0.16 | 0.32 | 0.35 | 0.35 |
| length + level | 0.16 | 0.34 | 0.39 | 0.41 |
| length + level + spectrum (centre of gravity, 2–8 kHz tilt, flatness) | 0.17 | 0.38 | 0.43 | 0.45 |
| all (+ level re the pause's noise floor, disfluent) | 0.17 | 0.40 | **0.46** | 0.48 |

The best air cost has voiceless time costing w = 3 times voiced time and k = 0.1 per dB of speech level (louder speech costs more air).
The fitted volume grows as t^0.54: longer breaths bring in more air, but more slowly. At the same length, 10 dB louder means about 18 %
more air, so the level is a weak cue to the inhale speed (consistent with turbulent noise power growing steeply with flow). A disfluent
breath brings in about a quarter of an in-breath's air. Per corpus the within-speaker r is 0.42–0.58, except four podcast-like sets
(0.31 and below; one near zero), which suggests edited audio or difficult mixes there.

## 3.2 The hidden lung state and silent refills

A discrete state-space model per speaker file (forward–backward over 81 volume steps, in units of the speaker's median breath-cycle air):
speech drains the volume by its air cost, a detected breath refills it by the balance model, a pause without a detected breath refills
with a probability that grows with its length (sigmoid(g0 + g1 log pause); a silent or missed breath of a learned size), a long pause
relaxes the volume towards rest, and going below the floor is penalised. EM on training speakers, 5 folds by speaker. Fitted: g0 = 0.04,
g1 = 0.89, so about half of the breathless 1-second pauses hold a silent refill, of about 1.4 breath cycles of air. With a quarter of the
detected breaths hidden, the model finds them at AUC 0.665 (pause length alone 0.615), but it cannot tell their size (Spearman 0.05–0.08).
The inferred volume at the end of a stretch of speech does not predict creak there beyond the stretch's position (AUC 0.602 with and
without it; 352,070 stretches); see the Knarr report.

## 3.3 A breath planner fitted on data

From what a synthesiser knows before it speaks (the air of the breath group ahead and to the end of the stretch, the air spent since the
last breath and the time since it, the pause, the speech level before and after, the speech rate, a filler before the pause), three small
nets with one hidden layer (24 ReLU units; boosted trees reach the same) were fitted on the 1.3 tables, 5 folds by speaker:

| Decision | Held-out result | Baseline |
| --- | --- | --- |
| audible in-breath at a pause (423,197 pauses, 48 % with one) | AUC 0.77 | pause length alone 0.62 |
| its length class S / M / L (233,629 breaths) | 65 % | majority 50 % (section 3's planner: 68 % vs 54 %) |
| its level class soft / loud | 60 % | majority 50 % (section 3's planner: 60 % vs 50 %) |

`kipp.lung.plan(..., sizing="data")`, now the default, keeps the rule planner's places and refill sizes, and takes each breath's class from
these models; a refill at a comma or phrase boundary is silent when P(audible | refill) < 0.5, with P(audible | refill) = P(audible) /
(P(audible) + (1 − P(audible)) × P(silent refill)). The breath's length is its class's median (0.24 / 0.40 / 0.68 s). `sizing="rule"`
gives the demo page's rules unchanged, and the parity test uses it. The detector now also reports each breath's relative volume
(`kipp.volume`; 1 = a 0.40 s in-breath at the corpora's median level).

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
  the volume limits come from the literature, not from these speakers. The air-balance volumes of 1.3 are relative and explain under half
  of the balance's variance; a dialogue corpus with respiration-belt signals is the next test.
- Silent refills are inferred, not heard: the hidden-state model finds hidden breaths only somewhat better than pause length.
- Profiles without word times are estimates; edited recordings (podcasts, broadcast) often have breaths removed or pauses shortened.

## 6. Reproduction

Detector training and the lung-model fits are in the KTH svtx repository (`tts/breath/`, `lung_table.py`, `lung_fit.py`; TTS dialogue
report sections 17–23); the 1.3 tables, air balance, hidden-state model and planner fits (`lung3_tables.py`, `lung3_balance.py`,
`lung3_hmm.py`, `lung3_planner.py`) are kept with the lab's non-public analysis code, as the corpora are not redistributable. This package holds the inference code, the detector weights (hash-checked), the 18 voice profiles, the planner
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
