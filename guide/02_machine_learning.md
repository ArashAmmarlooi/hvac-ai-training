# Part 2 — Machine learning (ML) explained

## What problem ML solves

You have **inputs** (features) and you want **outputs** (predictions) without writing hundreds of `if` rules by hand.

**Examples:**

- Input: transaction amount, time, merchant → Output: fraud or not
- Input: text features from OCR → Output: which symbol class (fan, duct, valve)
- Input: customer activity → Output: churn risk score

ML learns patterns from **past examples where you already know the answer** (labeled data).

---

## The ML workflow (same in every company)

```
Raw data  →  clean & features  →  train model  →  evaluate  →  deploy  →  monitor
```

### 1. Data

- **Features (X):** numbers or categories the model sees (columns in a table).
- **Label (y):** the correct answer for training (category, 0/1, price).

Example table (not images):

| amount | hour | country | **label: fraud** |
|--------|------|---------|------------------|
| 50     | 14   | CA      | 0                |
| 9000   | 3    | XX      | 1                |

### 2. Train / validation split

You hide part of the data (**validation**) so you can check: “Does the model work on data it never saw during training?”

If you test on the same data you trained on, you **cheat** and scores look too good.

### 3. Training (what “fit” means)

```python
model.fit(X_train, y_train)
```

In plain English:

- The algorithm **adjusts internal numbers** (weights, splits, trees) to minimize mistakes on `X_train` compared to `y_train`.
- For **XGBoost / Random Forest**: many decision rules on features.
- It does **not** “understand” like a human; it finds statistical patterns.

### 4. Predict

```python
prediction = model.predict(X_new)
```

Same **feature columns** as training, same order, same scaling (normalization). If production data is prepared differently than training, predictions break (**train/serve skew**).

### 5. Metrics (how good is it?)

- **Accuracy** — % correct (misleading if classes are imbalanced).
- **Precision** — of predicted positives, how many were right?
- **Recall** — of real positives, how many did we catch?
- **F1** — balance of precision and recall.

Pick metrics based on **business cost** (missing fraud vs false alarms).

---

## ML vs writing rules (OpenCV only)

| Approach | When it works |
|----------|----------------|
| **Rules / OpenCV** | Colors and shapes are fixed (blue duct always same color). Cheap, fast, explainable. |
| **Classical ML** | Tabular data, mixed features, labels available (XGBoost on transactions). |
| **ML on image features** | You extract numbers from images (size, aspect ratio, histogram) then classify. |

For the **HVAC drawing**, you often start with **rules + geometry**, then add **ML** when symbols vary or scans are noisy.

---

## Connection to your interview answers

You mentioned **XGBoost** for transaction categories:

- **Supervised learning**: each row has a true category.
- **Features**: amount, time, merchant type, etc.
- **Model**: learns non-linear boundaries and handles missing values well.

Same **mindset** applies to symbols if you label crops: features or CNN embeddings → classifier.

---

## What you do NOT need on day one

- Giant datasets
- Custom neural networks from scratch

You need: **clear label**, **consistent preprocessing**, **honest evaluation**, **job status for long runs** (upload → process → done).

See `examples/see_ml_flow.py` for a 30-line train/predict example with comments on every step.
