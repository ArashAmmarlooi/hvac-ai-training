# Rules and strategies — HVAC symbol project

This document is the durable plan for **how we build**, **what we train**, and **which rules we follow**. Use it when you forget the next step or need a consistent story for interviews.

Related guides: [Part 4 — Your project](04_your_interview_project.md), [Part 3 — Deep learning](03_deep_learning.md). Repo conventions: [`AGENTS.md`](../AGENTS.md) at the project root.

---

## Core strategy (one sentence)

**Classical OpenCV finds and cleans colored regions first; labeled bounding boxes train your own detector; OCR reads text separately; geometry links symbols to labels; JSON is the product output.**

Do not try to train one model to do symbols, duct colors, and text at once.

---

## Pipeline map

```
drawing (JPG/PNG/PDF→image)
    → load (NumPy array)
    → HSV color masks (blue / red / green)
    → clean masks (morphology)
    → regions / contours → bounding boxes
    → [DL] YOLO on annotated full pages
    → [OCR] text + boxes
    → match symbol ↔ nearest text
    → JSON (+ optional RAG on manuals)
```

---

## Roadmap and progress

| Phase | Step | Task | Status |
|-------|------|------|--------|
| **Foundations** | 1 | Load image with OpenCV / NumPy | Done |
| | 2 | Separate colors with HSV | Done |
| | 3 | Clean mask noise (morphology) | Done |
| | 4 | Find contours and bounding boxes | Done |
| **Create the dataset** | 5 | Define classes (vent, fan, RTU, diffuser, …) | **Current** |
| | 6 | Collect multiple HVAC drawings | |
| | 7 | Draw bounding-box annotations | |
| | 8 | Split train / validation / test | |
| **Train your model** | 9 | Understand CNNs and object detection | |
| | 10 | Fine-tune YOLO on your HVAC dataset | |
| | 11 | Evaluate precision, recall, mAP | |
| | 12 | Run predictions on unseen drawings | |
| **Complete the product** | 13 | OCR for text and coordinates | |
| | 14 | Match detected symbols with OCR text | |
| | 15 | Export structured JSON | |
| | 16 | RAG to search HVAC manuals (optional) | |

Update the **Status** column as you finish each step.

---

## Rules we follow in code

### Images and coordinates

| Rule | Why |
|------|-----|
| OpenCV loads color as **BGR**, not RGB | `cv2.imread` and `cv2.imshow` expect BGR |
| NumPy indexes as **`image[y, x]`** | Row first (height), then column (width) |
| `image.shape` is **`(height, width, channels)`** | Masks are `(height, width)` only — one value per pixel |
| Use **`pathlib.Path`** for file paths | Works on Windows; use `str(path)` for `cv2.imread` |
| On Windows, prefer **forward slashes** or raw strings in paths | Avoid `\U` unicode escape errors in strings |

### Masks and color

| Rule | Why |
|------|-----|
| Convert **BGR → HSV** before `inRange` | Hue bands separate blue/red/green ducts more reliably than BGR |
| Red needs **two** HSV ranges (wraps around 0/179) | OpenCV hue is 0–179 |
| Mask values are **0 or 255** | 255 = pixel kept, 0 = ignored |
| **OPEN** then optional **CLOSE** on masks | OPEN removes specks; CLOSE fills small gaps in lines |
| Tune HSV per customer / scan style | No universal magic numbers for every drawing |

### Deep learning (when you reach steps 9–12)

| Rule | Why |
|------|-----|
| **Same preprocessing in training and inference** | Resize, BGR/RGB, normalization must match |
| Prefer **fine-tune** (e.g. YOLO pretrained) over training from scratch | Less data, faster, better for small symbol sets |
| **Symbols** = detection or classification on crops; **text** = OCR | Different tools, different labels |
| Add DL when masks/templates fail often enough to justify labeling cost | Interview story: start simple, measure, then upgrade |

### Project hygiene

| Rule | Why |
|------|-----|
| HVAC source images live under **`projects/hvac/data/`** | e.g. `hvac_drawing.jpg` |
| Fruit source photo under **`projects/fruit/data/`**; class crops under **`projects/fruit/dataset/`** | Keep fruit and HVAC assets separate |
| Generated HVAC masks under **`projects/hvac/outputs/`**; pipeline JSON under **`projects/hvac/output/`** | Do not treat these as source |
| One small lesson step at a time | Run, inspect images, then advance |
| Validate Python with `python -m py_compile` after edits | Catch syntax errors early |

---

## What each technique is for

| Problem | First approach | Upgrade when needed |
|---------|----------------|---------------------|
| Blue / red / green ducts | HSV masks | Tune ranges; morphology |
| Symbol locations (MVP) | Contours on masks + aspect ratio / area | YOLO on labeled boxes |
| Symbol identity | Legend crops → classifier; then YOLO classes | More drawings + augmentation |
| Sizes and labels (`16X10`, `RTU2`) | OCR + nearest-neighbor geometry | Better OCR or line-following heuristics |
| Manual explanations | RAG on PDFs | Separate from detector training |

---

## Dataset strategy (steps 5–8)

1. **Classes** — Start from the **legend** on the drawing: each row is a class name (or your `S1`, `S2` ids with a description).
2. **Drawings** — One plan is enough to learn; **training** needs many pages or heavy augmentation (rotate, scale, noise, slight color shift).
3. **Annotations** — Use a bbox tool (e.g. LabelImg, CVAT, Roboflow) in **YOLO format** (class id + normalized or pixel box).
4. **Split** — Typical **70% train / 15% val / 15% test**; keep whole drawings in one split (do not split random crops from the same page across train and test).

Minimum viable dataset: every class appears in **train** at least several times; **val** for tuning; **test** only touched for final metrics.

---

## Target JSON (product shape)

Each detection should eventually look like:

```python
{
    "symbol_id": "S1",
    "class_name": "supply_terminal",
    "bbox": [x, y, width, height],
    "confidence": 0.87,
    "linked_text": "16X10",
    "text_bbox": [x, y, width, height],
}
```

`bbox` lists match OpenCV / connected-components style: top-left corner plus width and height in pixels.

---

## Learning workflow (how we use this repo)

1. Read the matching **guide** part for concepts.
2. Implement in **`projects/hvac/lesson_01_pixels.py`** (or the next numbered lesson file) — small, runnable.
3. Compare with **`projects/hvac/hvac_image_pipeline.py`** for a full classical pipeline reference.
4. Record progress in the roadmap table above.

Commands from the repo root:

```powershell
python projects/hvac/lesson_01_pixels.py
python projects/hvac/hvac_image_pipeline.py --input "projects/hvac/data/hvac_drawing.jpg" --output-dir projects/hvac/output/my_drawing
python -m py_compile projects/hvac/lesson_01_pixels.py
```

---

## Interview story (60 seconds)

We ingest legend and floor-plan images, build a symbol catalog from the legend, preprocess the plan with color segmentation and geometry, detect symbol candidates, and match them with a trained detector when rules are not enough. OCR supplies sizes and equipment tags; we associate text by proximity. Results are async job status plus JSON with boxes and confidence; low-confidence items can go to human review. RAG is optional for manual lookup, not for drawing parsing itself.

---

## Current lesson focus: step 3 — clean mask noise

**Goal:** Raw `inRange` mask → cleaned mask with fewer specks and more solid lines.

**Tools:** `cv2.getStructuringElement` or `np.ones((3, 3))`, `cv2.morphologyEx(..., cv2.MORPH_OPEN, kernel)`, optional `MORPH_CLOSE`.

**Success check:** Side-by-side save or display of raw vs clean mask; duct lines still continuous, small noise gone.

Next: **step 4** — `cv2.findContours` or `connectedComponentsWithStats` → list of bounding boxes; filter by `min_area` and aspect ratio (see `examples/hvac_image_pipeline.py`).
