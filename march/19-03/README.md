# 🛡️ Multilingual Abuse Detection System

## Progress Documentation — Phase 1 Complete

---

## What We Built So Far

```
  ███╗   ███╗██╗   ██╗██████╗ ██╗██╗
  ████╗ ████║██║   ██║██╔══██╗██║██║
  ██╔████╔██║██║   ██║██████╔╝██║██║
  ██║╚██╔╝██║██║   ██║██╔══██╗██║██║
  ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║███████╗
  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
  Multilingual Abuse & Slang Detection System
  Phase 1 — English Baseline Complete ✅
```

---

## 1. Full Pipeline (What Happens to Text)

```
  RAW INPUT TEXT
        │
        ▼
┌───────────────────┐
│   PREPROCESSING   │
│  ─────────────── │
│  • Remove URLs    │
│  • Strip HTML     │
│  • Emoji → text   │
│  • sooo → so      │
│  • wtf → expanded │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   TOKENIZER       │
│  ─────────────── │
│  MuRIL Tokenizer  │
│  SentencePiece    │
│  max_len = 128    │
│                   │
│  text → token IDs │
│  + attention mask │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   MuRIL MODEL     │
│  ─────────────── │
│  google/muril-    │
│  base-cased       │
│                   │
│  12 transformer   │
│  layers           │
│  768 hidden dim   │
│  [CLS] token      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ CLASSIFICATION    │
│      HEAD         │
│  ─────────────── │
│  Linear(768 → 1)  │
│  BCEWithLogits    │
│  + class weights  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    OUTPUT         │
│  ─────────────── │
│  sigmoid(logit)   │
│  > 0.5 = TOXIC    │
│  < 0.5 = CLEAN    │
└───────────────────┘
```

---

## 2. Dataset Pipeline

```
  kaggle.com
      │
      │  kaggle competitions download
      ▼
  jigsaw-toxic-comment-classification-challenge.zip
      │
      │  zipfile.ZipFile extract
      ▼
  data/
  ├── train.csv        ← 159,571 comments
  ├── test.csv
  ├── test_labels.csv
  └── sample_submission.csv
      │
      │  pandas read_csv
      ▼
  DataFrame
  ┌─────────────────────────────────────────────┐
  │ id │ comment_text │ toxic │ severe_toxic │.. │
  ├─────────────────────────────────────────────┤
  │    │ raw text...  │   0   │      0       │   │
  │    │ raw text...  │   1   │      1       │   │
  └─────────────────────────────────────────────┘
      │
      │  binary label = 1 if ANY label = 1
      ▼
  ┌──────────────────────┐
  │  label  │   count    │
  ├──────────────────────┤
  │    0    │  ~143,346  │  (90% clean)
  │    1    │  ~ 16,225  │  (10% toxic)
  └──────────────────────┘
      │
      │  train_test_split (90/10, stratified)
      ▼
  train_df (143k rows)  +  val_df (16k rows)
```

---

## 3. Preprocessing Pipeline

```
  INPUT: "Check this out http://x.com 😡😡 ur sooooo stupid wtf"
         │
         ▼
  ┌─────────────────────────────────────┐
  │  clean_text()                       │
  │  • re.sub URLs → ""                 │
  │  • re.sub HTML tags → ""            │
  │  • re.sub whitespace → " "          │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  "😡😡 ur sooooo stupid wtf"
                     │
  ┌──────────────────▼──────────────────┐
  │  handle_emojis()                    │
  │  • emoji.demojize()                 │
  │  • 😡 →  angry_face                │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  " angry_face  angry_face  ur sooooo stupid wtf"
                     │
  ┌──────────────────▼──────────────────┐
  │  normalize_repeated_chars()         │
  │  • re.sub (.)\1{2,} → \1\1         │
  │  • sooooo → so                      │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  " angry_face  angry_face  ur so stupid wtf"
                     │
  ┌──────────────────▼──────────────────┐
  │  normalize_abbreviations()          │
  │  • wtf → "what the hell"           │
  │  • ur  → "you"                     │
  │  • u   → "you"                     │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  OUTPUT: "angry_face angry_face you so stupid what the hell"
```

---

## 4. Model Architecture

```
  INPUT TOKENS (max 128)
  [CLS] you so stupid [SEP] [PAD] [PAD] ...
    │
    ▼
  ┌──────────────────────────────────────────┐
  │            EMBEDDING LAYER               │
  │   Token Embed + Position Embed +         │
  │   Segment Embed  →  768 dim vectors      │
  └─────────────────────┬────────────────────┘
                        │
            ┌───────────▼───────────┐
            │  Transformer Block 1  │
            │  ┌─────────────────┐  │
            │  │ Multi-Head      │  │
            │  │ Self-Attention  │  │
            │  │ (12 heads)      │  │
            │  └────────┬────────┘  │
            │  ┌────────▼────────┐  │
            │  │ Feed Forward    │  │
            │  │ Network         │  │
            │  └─────────────────┘  │
            └───────────┬───────────┘
                        │
                       ...  (12 blocks total)
                        │
            ┌───────────▼───────────┐
            │  Transformer Block 12 │
            └───────────┬───────────┘
                        │
                        ▼
              [CLS] token output
              (768-dim vector)
                        │
            ┌───────────▼───────────┐
            │  Classification Head  │
            │  Linear(768 → 1)      │
            │  + Dropout(0.1)       │
            └───────────┬───────────┘
                        │
                        ▼
                     logit
                        │
            ┌───────────▼───────────┐
            │  BCEWithLogitsLoss    │
            │  pos_weight = neg/pos │
            │  (~8.8x for imbalance)│
            └───────────┬───────────┘
                        │
                        ▼
              sigmoid(logit) → probability
              > 0.5  →  TOXIC  🔴
              ≤ 0.5  →  CLEAN  🟢
```

---

## 5. Training Setup

```
  ┌─────────────────────────────────────────────────┐
  │                TRAINING CONFIG                  │
  ├─────────────────────────────────────────────────┤
  │  Model          │  google/muril-base-cased       │
  │  Batch Size     │  32                            │
  │  Epochs         │  3                             │
  │  Learning Rate  │  2e-5                          │
  │  Max Length     │  128 tokens                    │
  │  Optimizer      │  AdamW (weight_decay=0.01)     │
  │  Scheduler      │  Linear warmup (500 steps)     │
  │  Loss           │  BCEWithLogitsLoss             │
  │  Precision      │  Mixed (float16 + GradScaler)  │
  │  Device         │  CUDA (T4 GPU - Google Colab)  │
  └─────────────────────────────────────────────────┘

  SPEED OPTIMIZATIONS APPLIED:
  ┌─────────────────────────────────────────────────┐
  │  ✅ torch.cuda.amp.autocast()  → 2-3x faster    │
  │  ✅ GradScaler                 → prevents NaN   │
  │  ✅ pin_memory=True            → faster transfer │
  │  ✅ num_workers=2              → parallel load   │
  │  ✅ non_blocking=True          → async GPU move  │
  │  ✅ batch_size=32              → better GPU util │
  └─────────────────────────────────────────────────┘
```

---

## 6. Experiment Tracking (W&B)

```
  TRAINING LOOP
       │
       │  every 100 steps
       ▼
  wandb.log(step_loss)────────────────────┐
       │                                  │
       │  every epoch                     │
       ▼                                  ▼
  wandb.log(                     ┌────────────────┐
    train_loss,                  │  wandb.ai      │
    val_f1,                      │  Dashboard     │
    epoch                        │                │
  )──────────────────────────────│  • Loss curve  │
                                 │  • F1 curve    │
  Best model auto-saved when     │  • Config      │
  val_f1 improves ──────────────▶│  • Run history │
  /content/best_model/           └────────────────┘
```

---

## 7. What's Done vs What's Left

```
  PHASE 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅ DONE
  │
  ├── ✅ Environment setup (Colab + W&B + Kaggle API)
  ├── ✅ Dataset download & extraction
  ├── ✅ EDA (shape, label distribution, nulls)
  ├── ✅ Preprocessing pipeline (clean, emoji, abbrev)
  ├── ✅ PyTorch Dataset class
  ├── ✅ MuRIL model loaded
  └── ✅ Training loop with mixed precision + tqdm

  PHASE 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ⏳ NEXT
  │
  ├── ⏳ Evaluation on test set
  ├── ⏳ Confusion matrix
  └── ⏳ Error analysis

  PHASE 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔜 SOON
  │
  ├── 🔜 Add HASOC dataset (Hinglish)
  ├── 🔜 Add Dravidian-CodeMix (Tanglish)
  └── 🔜 Re-train on multilingual data

  PHASE 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔜 SOON
  │
  ├── 🔜 Multi-label classification
  │       (hate_speech, profanity, cyberbullying)
  └── 🔜 Per-category confidence scores

  PHASE 5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔜 FUTURE
  │
  ├── 🔜 LIME explainability
  ├── 🔜 Attention visualization
  └── 🔜 Audit trail output

  PHASE 6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔜 FUTURE
  │
  ├── 🔜 FastAPI server
  ├── 🔜 ONNX conversion
  └── 🔜 /predict endpoint
```

---

---

# 🔧 Future Expansion Guide

## What to Change for Each Feature

---

### A. Add a New Language / Dataset

```
  WHAT TO CHANGE:
  ───────────────

  preprocess.py
  ┌─────────────────────────────────────────────────┐
  │  Add new abbreviation dict for that language    │
  │                                                 │
  │  tamil_abbrevs = {                              │
  │    "illa": "no",                                │
  │    "enna": "what",                              │
  │    ...                                          │
  │  }                                              │
  │                                                 │
  │  Add transliteration normalization              │
  │  using IndicNLP for that specific language      │
  └─────────────────────────────────────────────────┘

  data_loader.py  (NEW FILE TO CREATE)
  ┌─────────────────────────────────────────────────┐
  │  def load_hasoc():                              │
  │      df = pd.read_csv("hasoc.csv")              │
  │      df = df.rename(columns={...})              │
  │      df["label"] = ...                          │
  │      df["language"] = "hinglish"                │
  │      return df                                  │
  │                                                 │
  │  def load_dravidian():                          │
  │      ...same pattern...                         │
  │                                                 │
  │  def merge_all():                               │
  │      return pd.concat([                         │
  │          load_kaggle(),                         │
  │          load_hasoc(),       ← ADD HERE         │
  │          load_dravidian(),   ← ADD HERE         │
  │      ])                                         │
  └─────────────────────────────────────────────────┘

  train.py  ← NO CHANGES NEEDED
  (same training loop works for any language)
```

---

### B. Switch from Binary to Multi-Label

```
  WHAT TO CHANGE:
  ───────────────

  model.py
  ┌─────────────────────────────────────────────────┐
  │  BEFORE:                                        │
  │  num_labels = 1                                 │
  │                                                 │
  │  AFTER:                                         │
  │  num_labels = 4   # hate, profanity,            │
  │                   # cyberbullying, spam         │
  │  problem_type = "multi_label_classification"    │
  └─────────────────────────────────────────────────┘

  dataset.py
  ┌─────────────────────────────────────────────────┐
  │  BEFORE:                                        │
  │  "labels": torch.tensor(label, dtype=float)     │
  │                                                 │
  │  AFTER:                                         │
  │  "labels": torch.tensor(                        │
  │      [hate, profanity, cyberbullying, spam],    │
  │      dtype=torch.float                          │
  │  )                                              │
  └─────────────────────────────────────────────────┘

  train.py
  ┌─────────────────────────────────────────────────┐
  │  BEFORE:                                        │
  │  preds = (probs > 0.5).int()                    │
  │  f1_score(..., average="macro")                 │
  │                                                 │
  │  AFTER:                                         │
  │  preds = (probs > 0.5).int()  ← same!          │
  │  f1_score(...,                                  │
  │    average="macro",                             │
  │    zero_division=0   ← add this                 │
  │  )                                              │
  └─────────────────────────────────────────────────┘
```

---

### C. Upgrade the Model (MuRIL → XLM-RoBERTa)

```
  WHAT TO CHANGE:
  ───────────────

  Only ONE line in the entire codebase:

  ┌─────────────────────────────────────────────────┐
  │  BEFORE:                                        │
  │  MODEL_NAME = "google/muril-base-cased"         │
  │                                                 │
  │  AFTER (bigger, better, slower):                │
  │  MODEL_NAME = "xlm-roberta-large"               │
  │                                                 │
  │  AFTER (fastest):                               │
  │  MODEL_NAME = "xlm-roberta-base"                │
  └─────────────────────────────────────────────────┘

  Everything else — tokenizer, dataset, training
  loop — works automatically. Hugging Face handles
  the rest.

  ⚠️  XLM-R Large needs smaller batch size:
  BATCH_SIZE = 8 (or 16 with gradient accumulation)
```

---

### D. Add the FastAPI Server

```
  NEW FILE: api/main.py
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  LOAD MODEL ONCE AT STARTUP                     │
  │  ─────────────────────────                      │
  │  model = load from /content/best_model          │
  │  tokenizer = load from /content/best_model      │
  │                                                 │
  │  POST /predict                                  │
  │  ─────────────────────────                      │
  │  input:  { "text": "ur so stupid" }             │
  │                                                 │
  │  runs full_pipeline(text)                       │
  │  runs tokenizer(text)                           │
  │  runs model(tokens)                             │
  │  runs sigmoid(logits)                           │
  │                                                 │
  │  output: {                                      │
  │    "is_offensive": true,                        │
  │    "confidence": 0.94,                          │
  │    "categories": ["profanity"],                 │
  │    "explanation": { "stupid": 0.81, ... }       │
  │  }                                              │
  │                                                 │
  │  GET /health                                    │
  │  ─────────────────────────                      │
  │  output: { "status": "ok" }                     │
  └─────────────────────────────────────────────────┘
```

---

### E. Add LIME Explainability

```
  NEW FILE: explain.py
  ┌─────────────────────────────────────────────────┐
  │  from lime.lime_text import LimeTextExplainer   │
  │                                                 │
  │  def predict_proba(texts):                      │
  │      # wrapper that takes raw text list         │
  │      # returns [[prob_clean, prob_toxic], ...]  │
  │      ...                                        │
  │                                                 │
  │  explainer = LimeTextExplainer(                 │
  │      class_names=["clean", "toxic"]             │
  │  )                                              │
  │                                                 │
  │  explanation = explainer.explain_instance(      │
  │      text,                                      │
  │      predict_proba,                             │
  │      num_features=10                            │
  │  )                                              │
  │                                                 │
  │  # Returns which words pushed score up/down     │
  │  # e.g. "stupid" → +0.81 toxic                 │
  │  #      "hello"  → -0.02 toxic                 │
  └─────────────────────────────────────────────────┘

  PLUG INTO API:
  ┌─────────────────────────────────────────────────┐
  │  In main.py /predict endpoint:                  │
  │                                                 │
  │  explanation = explainer.explain_instance(      │
  │      text, predict_proba, num_features=5        │
  │  )                                              │
  │  word_scores = dict(explanation.as_list())      │
  │                                                 │
  │  return {                                       │
  │    "is_offensive": ...,                         │
  │    "confidence": ...,                           │
  │    "explanation": word_scores  ← ADD THIS       │
  │  }                                              │
  └─────────────────────────────────────────────────┘
```

---

### F. Convert to ONNX for Faster Inference

```
  NEW SCRIPT: convert_to_onnx.py
  ┌─────────────────────────────────────────────────┐
  │  # Run ONCE after training is done              │
  │                                                 │
  │  torch.onnx.export(                             │
  │      model,           ← your trained model      │
  │      dummy_input,     ← sample token tensor     │
  │      "model.onnx",    ← output file             │
  │      opset_version=14                           │
  │  )                                              │
  │                                                 │
  │  # Then in API use ONNX Runtime instead:        │
  │  import onnxruntime as ort                      │
  │  session = ort.InferenceSession("model.onnx")   │
  │  output = session.run(None, inputs)             │
  │                                                 │
  │  BENEFIT: 2-5x faster inference,               │
  │  no PyTorch needed at serving time              │
  └─────────────────────────────────────────────────┘
```

---

## Quick Reference — File Map

```
  multilingual-abuse-detection/
  │
  ├── preprocess.py        ← CHANGE FOR: new language, new slang
  │
  ├── dataset.py           ← CHANGE FOR: multi-label, new input format
  │
  ├── model.py             ← CHANGE FOR: different model, more labels
  │
  ├── train.py             ← CHANGE FOR: new hyperparams, new metrics
  │
  ├── data_loader.py       ← CHANGE FOR: adding new datasets
  │   (to be created)
  │
  ├── evaluate.py          ← CHANGE FOR: new evaluation metrics
  │   (to be created)
  │
  ├── explain.py           ← CREATE for LIME explainability
  │   (to be created)
  │
  ├── convert_to_onnx.py   ← CREATE for fast inference
  │   (to be created)
  │
  └── api/
      └── main.py          ← CREATE for FastAPI server
          (to be created)
```

---

*Phase 1 Complete — English baseline trained on Kaggle Toxic Comments using MuRIL + mixed precision on T4 GPU. Next: evaluation → HASOC multilingual data → multi-label → API.*
