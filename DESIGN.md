# PerformanceCVAE — next-generation humanization model

This is the design for the second-generation model that replaces the
deterministic `PerformanceRegressor` (4-output L1 regressor). It is a
conditional VAE with a heteroscedastic head that predicts **distributions**
over expressive performance rather than the average performance.

## Why

The first model trained an L1 regressor, so it learned the *mean* expressive
deviation across all pianists. Averaging many valid interpretations produces a
timid, washed-out performance and gives the augmentation layer nothing real to
steer — sliders could only scale deviations the model had already chosen.

This model fixes that at the root:

| Goal | Mechanism |
| --- | --- |
| Stop predicting the mean | Heteroscedastic Gaussian head + NLL: per-note mean **and** variance |
| Multiple valid interpretations | Latent `z` per window (CVAE); sample the prior for different coherent renditions |
| Coherent trajectories | Stochasticity lives in `z` (whole-window), not per-note noise; full attention over the window |
| Harmony / key / tension awareness | New score-derived features (`score_features.py`) |
| Importance maps | Predicted per-note σ *is* the importance map — free, no extra supervised head |

## The three pieces

### 1. Shared note embedding (`model.py: PerformanceCVAE._embed_score`)
Each note → a vector from its score features. Existing features (pitch, metric
position, melodic/chord context, tempo, density, era) **plus** the new
harmonic/tension/phrase features below. `in_dim = 340`, projected to `d_model = 256`.

### 2. Recognition encoder `q(z | score, performance)` (`model.py: RecognitionEncoder`)
Training only. A 3-layer transformer reads the score embedding + the 4 real
targets, masked-mean-pools the window, and outputs `μ_z, logσ_z` for a
`z_dim = 48` latent. At inference there is no performance, so `z ~ N(0, I)`.

### 3. z-conditioned decoder + heteroscedastic head (`model.py: PerformanceCVAE.decode`)
The 6-layer transformer backbone, conditioned on `z` (added as a broadcast bias
to every note), outputs **8 numbers per note**: mean + log-variance for each of
timing offset, log-duration-ratio, velocity, pedal. Targets are standardized
(z-scored) so the Gaussian NLL isn't dominated by the largest-scale target.

## Control surface (for the augmentation layer / player — implemented later in infer.py)

```
deadpan   = decode(score, z = 0)           # μ0, the timid mean
rendition = decode(score, z ~ N(0, I))     # μ_z, one coherent interpretation
output    = μ0 + amount · (μ_z − μ0)        # per-target knobs
```

`amount = 0` → deadpan; `1` → a real human-like reading; `> 1` → exaggerated.
Because `μ_z − μ0` is large exactly where interpretations legitimately diverge
(cadences, phrase ends), "more rubato" adds rubato where it belongs instead of
scaling globally. The predicted σ gives per-note wiggle room and an optional
fine-texture term (`σ · ε`, with ε low-pass filtered to stay coherent).

## Score-derived features (`score_features.py`)

Computed from the **quantized score only** (so they're reproducible identically
at training and at inference from the flat MIDI the player feeds — no
train/inference skew). Implemented as a vectorized NumPy pipeline (~0.3 s/piece)
rather than literal `music21.chordify` / `romanNumeralFromChord`, which would
cost hours of recache and seconds-per-upload in the player for the same info.

- **Local key + mode** — windowed Krumhansl-Kessler correlation
- **Scale degree** — note pitch-class relative to local tonic
- **Chord root + quality** — template matching over the sounding pitch-class set
- **Roman degree** — chord root's scale degree within the local key (harmonic function)
- **Tonal tension ribbon** — spiral-array cloud diameter, tensile strain (distance
  from key center), and cloud momentum (harmonic motion)
- **Dissonance** — vertical interval-class roughness
- **Phrase** — gap/rest-based boundary strength, position-in-phrase, distance-to-boundary

> Chord recognition is an approximate contextual hint, not a transcription;
> augmented triads in particular over-fire slightly on pedal-blurred MIDI.

## Training (`train.py`)

Objective = heteroscedastic Gaussian NLL + KL(q‖N(0,I)). Stabilizers:

- **NLL warmup** (`nll_warmup_epochs = 3`): fit means with L1, variance + KL off,
  so the backbone is sane before the likelihood scales errors by variance.
- **KL annealing** (`kl_anneal_epochs = 12`) + **free bits** (`free_bits = 0.1`/dim):
  ramp KL weight `0 → kl_max` and floor per-dim KL, to stop latent collapse
  (which would kill the "multiple interpretations" capability).
- **β-NLL** (`beta_nll = 0.5`): reweight per-note NLL so high-error notes aren't
  neglected early.

Best checkpoint is chosen on **validation NLL** (only after warmup). Checkpoints
store the model config + target standardization stats so inference can rebuild
and de-standardize.

## Files

| File | Change |
| --- | --- |
| `score_features.py` | **new** — harmonic/tension/phrase feature extraction |
| `analyze_data.py` | dumps per-target standardization stats → `paired_data/target_stats.json` |
| `note_dataset.py` | caches score features (CACHE_VERSION 3→4), standardizes targets, exposes new fields; `prebuild` CLI |
| `model.py` | `PerformanceRegressor` → `PerformanceCVAE` (recognition encoder + z-conditioned heteroscedastic decoder) |
| `train.py` | β-NLL + annealed-KL ELBO, warmup, per-term logging, stats in checkpoint |
| `infer.py` / player | **not yet done** — must adopt the 8-output head + control surface, and run `score_features` on uploaded flat MIDI (cached per piece) |

## How to run (tonight)

> **Interpreter:** use the Store `python` (has torch+CUDA+music21). VS Code's
> Programs\Python312 has **no torch** and will fail.

```bash
python analyze_data.py            # 1. target_stats.json  (~minutes; once)
python note_dataset.py prebuild   # 2. build v4 caches    (~10-15 min; once)
python train.py                   # 3. train the CVAE
```

Step 2 is optional (training will build caches lazily on first epoch) but
recommended — it surfaces per-piece errors up front and avoids worker recompute
races. The previous-generation model is archived under
`archive/20260621_171708/`.
