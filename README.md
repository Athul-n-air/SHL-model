# SHL Grammar Scoring Engine

> Spoken-English grammar scoring from 45–60 second audio clips.
>
> A practical ML pipeline that combines **speech recognition, grammar analysis, acoustic fluency, and frozen speech representations** to predict a continuous grammar score from **0 to 5**.

---

## Current benchmark

**Best Kaggle RMSE so far: `0.4401`**

Lower RMSE is better. The latest notebook adds new experiments designed to beat this benchmark.

---

## What is this?

This project was built for the **SHL Grammar Scoring Engine** challenge.

The task is:

- **Input:** a short spoken-English audio clip
- **Output:** a grammar/proficiency score from `0` to `5`

The dataset is small, so the approach is not an end-to-end deep neural network trained from scratch. Instead, pretrained models are used as feature extractors and small regressors are trained on top.

---

## Pipeline

```text
                    AUDIO
                       │
          ┌────────────┴────────────┐
          │                         │
      Faster-Whisper             Audio signal
          │                         │
      Transcript              Fluency features
          │                         │
   ┌──────┼──────────┐       ┌──────┼────────────┐
   │      │          │       │      │            │
LanguageTool GPT-2  CoLA    Pauses  Pitch      MFCC
   │      │          │       │      │            │
   └──────┴──────────┘       └──────┴────────────┘
              │                       │
              └───────────┬───────────┘
                          │
                 Frozen speech models
                 ┌────────┼────────┐
                 │        │        │
               WavLM   Whisper   HuBERT
                 │        │        │
                 └────────┴────────┘
                          │
                   Small regressors
               ┌──────────┼──────────┐
               │          │          │
            LightGBM     SVR        Ridge
               │          │          │
               └──────────┼──────────┘
                          │
                  OOF model selection
                          │
                    Final prediction
                          │
                       Score 0–5
```

The key idea is to capture both **what the speaker says** and **how the speaker says it**.

---

## Data

The working Kaggle package contains:

- **769 training clips**
- **216 test clips**
- Audio around **45–60 seconds**
- Target score from **0–5**

The notebook uses the supplied `train.csv` and `test.csv`.

> The packaged `sample_submission.csv` was inconsistent with the current 216-row `test.csv`, so the notebook writes predictions in `test.csv` order.

---

## Feature engineering

### 1. Transcript / language features

Audio is transcribed with **Faster-Whisper `small`**.

Basic linguistic features include:

- Word count
- Sentence count
- Average sentence length
- Average word length
- Unique-word ratio
- Repeated-word count
- Filler count
- Filler ratio

### 2. LanguageTool

Grammar and writing-style signals:

- Total errors
- Grammar errors
- Spelling errors
- Style errors
- Errors per 100 words
- Grammar errors per sentence

### 3. GPT-2

Two views are used:

- Perplexity / log perplexity
- Token-level surprisal statistics

The token-level version summarizes local uncertainty instead of using one global number only.

### 4. RoBERTa-CoLA

Sentence acceptability features:

- Mean acceptability
- Minimum acceptability
- Fraction of low-scoring sentences

---

## Acoustic and fluency features

Because this is a spoken-language task, the transcript is only part of the signal.

The pipeline extracts:

### Timing

- Audio duration
- Speech duration
- Speech ratio
- Speech segment count
- Pause count
- Mean pause duration
- Maximum pause duration
- Long-pause frequency
- Speaking rate
- Articulation rate

### Signal features

- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Pitch statistics
- MFCC statistics

### Whisper timing

- Segment count
- Pause statistics
- Average log probability
- No-speech probability
- Compression ratio

### Whisper word timing

- Words per minute
- Articulation speed
- Word duration
- Pause distribution
- Word confidence
- Initial silence
- Trailing silence

---

## Frozen speech representations

### WavLM

WavLM is used as a frozen speech encoder. Mean pooling and reduced representations are tested.

### Whisper encoder

Hidden Whisper representations are tested as another speech view. The idea is to capture information that is not preserved by the final transcript alone.

### HuBERT

HuBERT is used as a third frozen speech representation. Multiple encoder layers are compared and the strongest layer is selected by OOF RMSE.

---

## Models

The project deliberately uses small regressors because the dataset is small.

### LightGBM

Captures nonlinear relationships in engineered text and audio features.

### SVR

RBF-SVR is used for smooth nonlinear regression.

### Ridge

Especially useful for high-dimensional frozen speech embeddings.

### KNN

A small distance-based experiment on reduced WavLM features.

### Stacking

Out-of-fold predictions from several models are combined with a small Ridge meta-model.

---

## Validation

All main model comparisons use:

```text
5-fold cross-validation
```

The same folds are reused so comparisons stay fair.

Important rule:

> **Test labels are never used.**

The stack is built from out-of-fold predictions so the meta-model is evaluated without directly seeing the labels it predicts.

The notebook also checks:

- Cross-fitted isotonic calibration
- Train/test domain weighting
- Controlled prediction rounding

A technique is kept only when the validation evidence supports it.

---

## Previous experiment results

| Model / Feature Set | RMSE | Pearson |
|---|---:|---:|
| Ridge + basic/grammar features | 1.1367 | 0.4214 |
| SVR + basic/grammar features | 1.1168 | 0.4751 |
| Ridge + MiniLM | 1.2619 | 0.4410 |
| Ridge + combined text features | 1.2376 | 0.4673 |
| SVR + combined features | 1.0285 | 0.5581 |
| LightGBM + combined features | 1.0130 | 0.5769 |
| LightGBM + GPT-2 | 0.9852 | 0.6088 |
| 70/30 LightGBM + SVR | 0.9771 | 0.6196 |
| WavLM-based system | ~0.58 OOF | — |
| Calibrated system | 0.5235 OOF | 0.9063 |
| **Kaggle benchmark** | **0.4401** | — |

> OOF and Kaggle scores are different measurements. The **0.4401 Kaggle RMSE** is the external benchmark we are trying to beat.

---

## New experiments in the latest notebook

### Whisper word timing

Adds information about:

- speaking speed
- articulation speed
- hesitation
- pauses
- word duration
- ASR confidence

### GPT-2 token surprisal

Adds local language-model uncertainty features instead of relying only on whole-response perplexity.

### Nonlinear WavLM models

Tests RBF-SVR and KNN in addition to Ridge.

### HuBERT

Adds a third frozen speech representation for model diversity.

### Domain adaptation

An unlabeled train-vs-test classifier estimates how similar each training example is to the test distribution. The resulting weights can be used when fitting selected models.

This uses **no test labels**.

### Leakage-safe stacking

Base model predictions are generated out-of-fold and then combined with a small Ridge meta-model.

### Controlled rounding

A small sweep checks whether rounding predictions improves RMSE. Rounding is applied only to the final predictions if validation supports it.

---

## Visualizations

The notebook includes charts at the important steps:

- Training score distribution
- Articulation rate vs grammar score
- Speech representation comparison
- HuBERT layer comparison
- Train/test domain shift
- Final model comparison
- OOF true-vs-predicted scatter
- Residual distribution
- Final test prediction distribution

---

## Suggested repository structure

```text
SHL-Grammar-Scoring/
│
├── README.md
├── notebooks/
│   └── SHL_Grammar_Scoring_Expert_v7.ipynb
│
├── outputs/
│   └── submission.csv
│
└── requirements.txt
```

The current workflow is notebook-first, so the `src/` package structure can be added later if the experiment is turned into a reusable application.

---

## Main technologies

| Area | Tools |
|---|---|
| Speech-to-text | Faster-Whisper |
| Speech embeddings | WavLM, Whisper Encoder, HuBERT |
| NLP | GPT-2, RoBERTa-CoLA |
| Grammar | LanguageTool |
| Audio processing | Librosa |
| Regression | LightGBM, SVR, Ridge, KNN |
| Validation | 5-fold OOF |
| Visualization | Matplotlib |
| Environment | Kaggle GPU |

---

## Running the notebook

The notebook is designed for a **Kaggle GPU** environment.

### 1. Open the notebook

Use:

```text
SHL_Grammar_Scoring_Expert_v7.ipynb
```

### 2. Attach the SHL competition dataset

The notebook searches `/kaggle/input` for the dataset containing:

```text
train.csv
test.csv
```

### 3. Run from top to bottom

Expensive feature extraction is cached under:

```text
/kaggle/working/shl_cache
```

when the Kaggle session preserves the working directory.

### 4. Output

The notebook writes:

```text
/kaggle/working/submission.csv
```

---

## Kaggle submission

```bash
kaggle competitions submit \
  -c shl-hiring-assessment-2026 \
  -f /kaggle/working/submission.csv \
  -m "WavLM + Whisper + HuBERT + grammar + acoustic features"
```

Current benchmark to beat:

```text
0.4401 RMSE
```

---

## Interview explanation

A simple way to explain the project:

> I treated the task as a multimodal regression problem. First, I transcribed each spoken response with Whisper. From the transcript I extracted grammar, fluency and language-model features. From the original audio I extracted pause, speaking-rate and acoustic features. I also used frozen WavLM, Whisper and HuBERT representations to capture speech information that the transcript cannot preserve. Finally, I compared small regressors with five-fold out-of-fold validation and combined the strongest independent predictions with a small Ridge stack.

---

## Design principles

### Small dataset → pretrained representations

The project uses pretrained speech and language models as feature extractors instead of training a large network from scratch on only 769 clips.

### Spoken task → audio + text

Grammar scoring is not treated as a normal text-only problem.

Both **what was said** and **how it was said** are useful signals.

### Honest validation

Test labels are never used. Out-of-fold predictions are used for model comparison and stacking.

### Explainability

Each major feature group and model has a clear reason for being there and can be explained in an interview.

### Avoid leakage

Filename numbers are not used as predictive features.

---

## Goal

```text
Current Kaggle benchmark
        0.4401
           │
           ▼
      Beat 0.4401
           │
           ▼
      Push into 0.32 range
```

The project focuses on **measured improvement, clean validation, and explainability** instead of adding complexity for its own sake.

---

## Author

**Athul Nair**

Built as an ML experimentation project for the SHL Grammar Scoring Engine challenge.
