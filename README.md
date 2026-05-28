# cnn-rnn-nlp-projects

Deep learning projects covering image captioning (CNN-RNN) and emotion classification (LSTM vs BERT vs LLM prompting).

---

## Table of Contents

1. [MS-COCO Image Captioning Project](#1-ms-coco-image-captioning-project)
2. [Emotion Classification: RNN vs BERT vs LLM](#2-emotion-classification-rnn-vs-bert-vs-llm)

---

## 1. MS-COCO Image Captioning Project

**Notebook:** `ms-coco-image-captioning-project.ipynb`

### Overview

This project builds an end-to-end **image captioning** system that takes a raw image as input and generates a descriptive natural-language sentence. It uses the classic **CNN-RNN encoder-decoder** architecture:

- **CNN (Encoder):** A pretrained InceptionV3 network extracts a fixed-size feature vector from each image.
- **RNN (Decoder):** A custom LSTM-based decoder uses the visual feature vector and previously generated words to predict the next word in the caption, one token at a time.

The model is trained and evaluated on the **MS-COCO 2014** dataset (50% subset), one of the most widely used benchmarks for image captioning.

---

### Dataset

- **Source:** [MS-COCO Dataset on Kaggle](https://www.kaggle.com/datasets/hariwh0/ms-coco-dataset) — downloaded via `kagglehub`.
- **Annotations file:** `captions_train2014.json`
- **Structure:** Each image can have multiple (5) associated captions.
- **Original dataset size:** 414,113 captions
- **Training subset (50% of images sampled):** 207,062 captions

**Sample raw data:**

| id | image_id | caption |
|----|----------|---------|
| 48 | 318556 | A very clean and well decorated empty bathroom |
| 126 | 318556 | A blue and white bathroom with butterfly theme... |
| 219 | 318556 | A bathroom with a border of butterflies and bl... |

---

### Data Preprocessing

#### Caption Cleaning

Each raw caption is processed through the following pipeline:

1. **Lowercased** — ensures uniform vocabulary (e.g., `"Park"` and `"park"` are treated as the same word).
2. **Punctuation removed** — strips all punctuation to avoid treating `"park"` and `"park."` as different tokens.
3. **Whitespace normalized** — extra spaces from punctuation removal are cleaned up.
4. **Special tokens added** — `<start>` and `<end>` tokens are prepended/appended so the RNN knows when to begin and stop generating.

**Example:**
```
Original:  A very clean and well decorated empty bathroom
Cleaned:   <start> a very clean and well decorated empty bathroom <end>
Tokenized: ['<start>', 'a', 'very', 'clean', 'and', 'well', 'decorated', 'empty', 'bathroom', '<end>']
```

#### Vocabulary Construction

- All captions are tokenized and word frequencies are counted.
- **Total unique words** in the raw dataset: **18,387**
- A vocabulary of the top **5,000** most frequent words is built (leaving room for 4 special tokens).
- **Final vocabulary size:** **4,998 words**
- Special tokens: `<pad>` (0), `<start>` (1), `<end>` (2), `<unk>` (3)
- Rare words not in the vocabulary are mapped to `<unk>`.

**Example numerical mapping:**
```
Tokenized:  ['<start>', 'a', 'very', 'clean', 'and', 'well', 'decorated', 'empty', 'bathroom', '<end>']
Sequence:   [1, 4, 143, 506, 10, 665, 395, 275, 58, 2]
```

#### Sequence Padding

- **Maximum sequence length** in the dataset: **51 words**
- All sequences are zero-padded (`<pad>` = 0) to this fixed length.
- **Final padded matrix shape:** `(207062, 51)` — one row per caption, 51 columns.

---

### CNN Encoder — InceptionV3

The CNN encoder uses **InceptionV3** pretrained on ImageNet as the image feature extractor.

**Architecture:**
- Base: `InceptionV3` (weights from ImageNet, top classification layer excluded)
- Base model weights are **frozen** (transfer learning).
- Output of the base model is passed through `GlobalAveragePooling2D` → shape `(batch, 2048)`
- A `Dense(256, activation='relu')` layer projects down to an embedding size of 256.
- `Dropout(0.5)` is applied for regularization.
- Input image size: **299×299×3** (as required by InceptionV3)
- Final output per image: a **256-dimensional** embedding vector.

**Why InceptionV3?**
InceptionV3 is a deep convolutional network known for high accuracy on ImageNet. By freezing its weights and only training the projection head, we get rich, generalizable visual representations without the cost of training the CNN from scratch.

**Key model details:**
- Hardware: **2× NVIDIA Tesla T4 GPUs** (Kaggle environment)
- The CNN processes images in batches, saving the resulting 256-d feature vectors to disk for reuse during RNN training (avoiding repeated forward passes through the large base model).

---

### CNN Feature Extraction

- CNN features are precomputed and saved to `/kaggle/working/cnn_features/` as `.npy` files.
- Each feature file is named by the image ID and stores a 256-dimensional numpy array.
- This allows the RNN training phase to load features directly from disk, dramatically reducing I/O and computation time.

---

### RNN Decoder — LSTM

The RNN decoder is a custom LSTM-based sequence model that generates captions word by word.

**Architecture:**
- **Image feature input:** 256-d vector (output of CNN encoder)
- **Word embedding layer:** Maps word indices → dense embedding vectors
- **LSTM layer:** Processes the sequence, conditioned on the image embedding
- **Dense output layer:** Projects LSTM hidden state → vocabulary-size logits
- **Softmax** is used to produce a probability distribution over the vocabulary at each timestep.

**Training procedure:**
- At each step, the model receives the image embedding + the current word and predicts the next word.
- **Teacher forcing** is used during training (ground-truth tokens are fed as input rather than the model's own predictions).
- Loss function: **Cross-entropy** over the vocabulary.

---

### Caption Generation (Inference)

At inference time, the model generates captions **greedily** (or with beam search):

1. The CNN encoder extracts a 256-d feature vector for the input image.
2. The RNN decoder is initialized with the `<start>` token.
3. At each step, the model predicts the most likely next word.
4. Generation stops when the `<end>` token is produced or when the maximum length is reached.

---

### Training Environment

- **Platform:** Kaggle Notebooks
- **GPU:** 2× NVIDIA Tesla T4
- **Framework:** TensorFlow / Keras
- **Python:** 3.12

---

## 2. Emotion Classification: RNN vs BERT vs LLM

**Notebook:** `emotion-classification-rnn-bert-llm.ipynb`

### Overview

This project compares **three fundamentally different approaches** to classifying the emotion expressed in short English text sentences. The same dataset and evaluation metrics are used across all three approaches, making it a clean head-to-head comparison.

**Task:** Given a sentence (e.g., `"i didnt feel humiliated"`), classify it into one of 6 emotion categories: **sadness**, **joy**, **love**, **anger**, **fear**, or **surprise**.

---

### Dataset — dair-ai/emotion

- **Source:** [Hugging Face Hub — `dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)
- **Format:** Text classification dataset with integer labels (0–5).

| Label | Emotion |
|-------|---------|
| 0 | sadness |
| 1 | joy |
| 2 | love |
| 3 | anger |
| 4 | fear |
| 5 | surprise |

**Dataset splits:**

| Split | Count |
|-------|-------|
| train | 16,000 |
| validation | 2,000 |
| test | 2,000 |

**First example:** `{'text': 'i didnt feel humiliated', 'label': 0}` → *sadness*

---

### Part 1 — Data Preprocessing & Exploration

#### Class Distribution (Training set)

| Emotion | Proportion |
|---------|-----------|
| sadness (0) | 29.16% |
| joy (1) | 33.51% |
| love (2) | 8.15% |
| anger (3) | 13.49% |
| fear (4) | 12.11% |
| surprise (5) | 3.58% |

**Note:** The dataset is moderately imbalanced. "joy" and "sadness" together account for over 60% of samples, while "surprise" is severely underrepresented (~3.6%).

#### Text Length Statistics (Training set)

| Statistic | Characters | Tokens (words) |
|-----------|-----------|----------------|
| Mean | 96.85 | 19.17 |
| Min | 7 | 2 |
| Max | 300 | 66 |

**Visualizations produced:**
- Bar chart of class distribution (emotion counts)
- Histogram of character-length distribution (with KDE curve)

---

### Part 2 — BiLSTM Model (RNN Approach)

#### Model Architecture

A **Bidirectional LSTM (BiLSTM)** is used as the recurrent baseline:

- **Embedding layer:** Converts token indices to dense word vectors.
- **BiLSTM layer(s):** Processes the sequence in both forward and backward directions, capturing context from both sides of each word.
- **Dropout:** Applied for regularization.
- **Dense output layer:** Projects to 6 logits (one per emotion class).

**Why Bidirectional?** Standard LSTMs only see context from the left. BiLSTMs also see the right context, which is more suitable for sentence-level classification tasks.

#### Training Configuration

- **Optimizer:** (standard for PyTorch LSTM models)
- **Loss:** Cross-entropy
- **Training epochs:** 10
- **Batch size:** (standard)
- **Hardware:** GPU (NVIDIA Tesla T4)

#### Training Results

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 1.4420 | 1.0310 |
| 2 | 0.6220 | 0.3957 |
| 3 | 0.2713 | 0.2668 |
| 4 | 0.1696 | 0.2187 |
| 5 | 0.1296 | 0.2521 |
| 6 | 0.1084 | 0.1900 |
| 7 | 0.0947 | 0.2200 |
| 8 | 0.0880 | 0.1933 |
| 9 | 0.0805 | 0.1950 |
| 10 | 0.0723 | 0.2040 |

- **Total training time:** ~1.43 minutes
- **Test set inference time:** 0.58 seconds

**Observations:**
- Loss drops sharply in the first 2 epochs, then gradually levels off.
- Validation loss is slightly higher than training loss by epoch 6+, indicating mild overfitting that Dropout helps control.

A **training/validation loss curve** plot is generated for visual diagnosis.

---

### Part 3 — DistilBERT Transformer Model

#### Model Architecture

**DistilBERT** is a smaller, faster distilled version of BERT that retains ~97% of BERT's performance at 60% of its size.

- **Pretrained model:** `distilbert-base-uncased` (from Hugging Face)
- **Fine-tuning strategy:** The pretrained DistilBERT backbone is fine-tuned end-to-end on the emotion classification task.
- **Classification head:** A linear layer on top of the `[CLS]` token output maps to 6 emotion classes.
- **Tokenizer:** DistilBERT's WordPiece tokenizer handles subword tokenization.

#### Why DistilBERT?

Unlike BiLSTM, BERT-based models use **self-attention** (transformer architecture), which captures long-range dependencies between all pairs of words simultaneously. DistilBERT specifically is chosen for its efficiency — it is much faster to fine-tune than full BERT while maintaining competitive accuracy.

#### Evaluation Results (Test Set)

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **92.90%** |
| **Macro F1 Score** | **0.8874** |

**Per-class F1 Scores:**

| Emotion | F1 Score |
|---------|---------|
| Sadness | 0.9659 |
| Joy | 0.9444 |
| Love | 0.8280 |
| Anger | 0.9338 |
| Fear | 0.9022 |
| Surprise | 0.7500 |

**Detailed Classification Report:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| sadness | 0.98 | 0.95 | 0.97 | 581 |
| joy | 0.94 | 0.95 | 0.94 | 695 |
| love | 0.84 | 0.82 | 0.83 | 159 |
| anger | 0.92 | 0.95 | 0.93 | 275 |
| fear | 0.90 | 0.91 | 0.90 | 224 |
| surprise | 0.77 | 0.73 | 0.75 | 66 |
| **weighted avg** | **0.93** | **0.93** | **0.93** | 2000 |

**Key observations:**
- DistilBERT achieves very high accuracy across all classes.
- "Surprise" has the lowest F1 (0.75), likely due to its small support (only 66 samples in the test set).
- "Sadness" has the highest F1 (0.9659), benefiting from the largest training set proportion.

A **confusion matrix heatmap** is generated to visualize where misclassifications occur.

---

### Part 4 — LLM with Chain-of-Thought (CoT) Prompting

#### Approach

Instead of fine-tuning a model, this approach uses a **pretrained LLM** in a **zero-shot / few-shot prompting** framework with **Chain-of-Thought (CoT)** reasoning.

**Model used:** A Microsoft Phi-based instruction-tuned model (deployed on GPU).

**Prompt format:**
```
<|system|>
You are an expert emotion classifier. Reason step-by-step then output the label on a new line starting with 'Final Emotion: ' followed by exactly one of: sadness, joy, love, anger, fear, or surprise.<|end|>
<|user|>
Text: "..."<|end|>
<|assistant|>
```

The model is asked to:
1. **Reason step-by-step** about the emotional cues in the text.
2. Output `Final Emotion: <label>` on a new line at the end.

The final answer is extracted using a regex pattern matching `Final Emotion: <word>`.

**Batch inference:** Run with batch size 32 across all 2,000 test samples.

#### Example Outputs

**Example 1:**
- Input: *"im feeling rather rotten so im not very ambitious right now"*
- Reasoning: *"The word 'rotten' suggests a feeling of being in a bad or undesirable situation, while 'not very ambitious' implies a lack of motivation..."*
- Output: `Final Emotion: sadness` ✅

**Example 2:**
- Input: *"im updating my blog because i feel shitty"*
- Output: `Final Emotion: sadness` ✅

**Example 3:**
- Input: *"i never make her separate from me because i don t ever want her to feel like i m ashamed with her"*
- Output: `Final Emotion: Love` ✅

**Example 4:**
- Input: *"i left with my bouquet of red and yellow tulips under my arm feeling slightly more optimistic than when i arrived"*
- Output: `Final Emotion: Joy` ✅

**Example 5:**
- Input: *"i was feeling a little vain when i did this one"*
- Output: `Final Emotion: Surprise` (note: this may not always be the "correct" label)

#### CoT Evaluation Results (Full Test Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | **49.35%** |
| **Macro F1** | **0.3671** |
| **Total Inference Time** | **1483.06 seconds (~25 minutes)** |

**Key observations:**
- CoT prompting with a small/medium LLM dramatically underperforms fine-tuned models (49% vs 93%).
- The model sometimes reasons correctly but fails to reliably follow the output format.
- Inference is extremely slow (~25 minutes for 2,000 samples on GPU vs 0.58 seconds for the BiLSTM).
- The confusion matrix shows systematic misclassification patterns, especially for minority classes like "surprise" and "love".

A **confusion matrix** is generated for the CoT results.

---

### Model Comparison Summary

| Model | Test Accuracy | Macro F1 | Inference Time (2k samples) |
|-------|--------------|---------|----------------------------|
| **BiLSTM** | — | — | 0.58 sec |
| **DistilBERT (fine-tuned)** | **92.90%** | **0.8874** | — |
| **LLM + CoT Prompting** | 49.35% | 0.3671 | 1,483 sec |

**Key Takeaways:**
- **Fine-tuned DistilBERT** is the clear winner in both accuracy and F1 across all emotion classes.
- **BiLSTM** trains very quickly (under 2 minutes) and is effective for a traditional RNN baseline.
- **LLM + CoT Prompting** (zero-shot) is much slower and less accurate for this structured multi-class classification task — demonstrating that prompting cannot replace task-specific fine-tuning for well-defined classification problems.
- The "surprise" class is the hardest for all models, consistent with its severe underrepresentation in the training data.

---

### Training Environment

- **Platform:** Kaggle Notebooks
- **GPU:** NVIDIA Tesla T4
- **Framework:** PyTorch + Hugging Face Transformers
- **Python:** 3.12
