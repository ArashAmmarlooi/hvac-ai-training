# Part 3 — Deep learning (DL) explained

## ML vs deep learning in one sentence

- **Classical ML:** you often **engineer features** (columns) by hand; the model is smaller (trees, logistic regression).
- **Deep learning:** a **neural network** learns features **from raw data** (especially images, audio, long text).

For **images**, deep learning usually means **CNNs** (Convolutional Neural Networks).

---

## What is a neural network?

Think of layers of math:

```
Input (pixels)  →  hidden layers  →  output (probabilities)
```

Each layer transforms numbers. During **training**, the network changes millions of **weights** to reduce error on labeled images.

**Input example:** image 224×224×3 (height, width, RGB channels) = a big array of numbers (0–255 or normalized 0–1).

**Output example:** `[0.05, 0.90, 0.05]` = 90% chance class “supply_fan”.

---

## Key ideas (what interviewers expect you to know)

### 1. Labeled data

Each training image has a **label** (symbol type). More variety in labels and images → usually better generalization.

### 2. Preprocessing must match in production

- Resize to same size as training
- Same color format (RGB vs BGR)
- Same normalization

If training used OpenCV BGR and production uses RGB, accuracy collapses.

### 3. Transfer learning (most common in industry)

You rarely train from zero. You take a **pretrained** model (ResNet, EfficientNet, YOLO) trained on ImageNet and **fine-tune** last layers on your symbols.

- Faster
- Less data needed
- Better results for small datasets

### 4. Object detection vs classification

| Task | Question | Tool idea |
|------|----------|-----------|
| **Classification** | “What is this crop?” | CNN on fixed-size patch |
| **Detection** | “Where are all symbols?” | YOLO, Faster R-CNN (boxes + classes) |

Your interview scenario: **legend** = classification on crops; **full plan** = detection + match to legend catalog.

### 5. When DL is worth it

Use DL when:

- Symbols rotate, scale, or look different scan-to-scan
- Template matching fails often
- You have (or can build) enough labeled examples

Skip DL for MVP when:

- Color layers are reliable (blue/red ducts)
- Symbol set is small and fixed
- You need explainable boxes from color segmentation

---

## Training loop (deep learning)

```
for each batch of images:
    predictions = model(images)
    loss = how wrong predictions are
    update weights to reduce loss
```

After training:

```
save model file  →  load in Python service  →  predict on new uploads
```

Same as classical ML: **version the model**, log metrics, monitor drift.

---

## DL and RAG (from your interview)

They solve **different** problems:

| Tool | Good for |
|------|----------|
| **CV / DL** | Finding shapes and symbols on a drawing |
| **OCR** | Reading text next to symbols |
| **RAG** | Answering questions from **documents** (manuals, specs) with citations |

A real product can use **all three**: detect symbol on plan (CV/DL), read tag (OCR), answer “what is RTU2?” from PDF manual (RAG).

---

## Honest skill statement (good in interviews)

“I have strong Python and ML pipeline thinking. For production vision I’d start with OpenCV and labeled evaluation, add a pretrained detector when rule-based matching isn’t enough, and keep humans in the loop for low-confidence matches.”

That is credible and matches the transcript discussion.
