# Part 4 — Your project (HVAC plan / symbols) end to end

This is the scenario from your interview: a **technical floor plan** (like Blue Foundry Bank duct drawing) with:

- **Colored lines** — blue supply, red return/exhaust, green equipment (RTU, fans)
- **Symbols** — vents, fans, terminals
- **Text** — sizes (`16X10`), labels (`RTU2`), dimensions

The company wants software that takes **images** (legend + diagram) and returns **structured data** (symbol id, description, position, confidence).

---

## Big picture (three layers)

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────────┐
│   Angular   │────▶│  .NET API   │────▶│  Queue + Python AI   │
│   upload UI │     │  job record │     │  image pipeline      │
└─────────────┘     └─────────────┘     └──────────────────────┘
       ▲                    │                      │
       └──── poll / WS ─────┴── status + JSON results
```

**Python’s job** is not “replace the whole app.” It is the **processing worker** that turns pixels into JSON.

---

## Stage A — Ingestion (no AI yet)

1. User uploads PNG or PDF.
2. If PDF, **rasterize** to image (high DPI so thin lines stay visible).
3. Store file in blob storage; create `job_id`, status `queued`.

**Python types you use:** paths (`pathlib`), maybe `dict` for job metadata.

---

## Stage B — Preprocessing (mostly OpenCV, not deep learning)

**Goal:** make the next steps reliable.

| Step | What it does | Why |
|------|----------------|-----|
| Load image | `cv2.imread` → NumPy array H×W×3 | Image = numbers |
| Color space | BGR → HSV | Easier to mask “blue” lines |
| Masks | `inRange` for blue/red/green | Separate duct systems |
| Contours | find connected regions | Bounding boxes around components |
| Denoise | morphology, blur | Remove specks |

This is **classical computer vision**. No training required. You tune color ranges per customer drawing style.

**Output:** masks, list of bounding boxes `(x, y, w, h)`.

---

## Stage C — Legend (symbol catalog)

Second image (or crop): **symbol list** with picture + description.

1. **Segment** table cells (lines or layout rules).
2. **Crop** each symbol icon.
3. **OCR** text description (Tesseract / cloud OCR).
4. Assign **`symbol_id`** (S1, S2, …) in your database.

**Data structure example:**

```python
catalog = {
    "S1": {"description": "Supply terminal", "embedding": [...]},
    "S2": {"description": "Return grille", "embedding": [...]},
}
```

`catalog` is a **dict** keyed by id — same Python you learned in Part 1.

---

## Stage D — Find symbols on the main drawing

**MVP (rules):**

- Template match legend crop against diagram regions
- Or match by color + shape (aspect ratio, area)

**When that fails (ML/DL):**

- Train **classifier** on cropped patches (legend labels = training labels)
- Or **object detector** (YOLO) on full page with boxes

For each detection:

```python
{
    "symbol_id": "S1",
    "description": "Supply terminal",
    "bbox": [120, 340, 44, 44],
    "confidence": 0.87,
}
```

---

## Stage E — Link text on the plan (OCR + geometry)

Run OCR on full diagram → list of text boxes with coordinates.

For each symbol bbox, find **nearest** text or follow **leader line**:

- Distance between centers
- Same horizontal band
- Optional: line detection along duct

This is **not** deep learning by default; it is geometry + heuristics. DL can help if layout is messy.

---

## Stage F — Product concerns (they asked you this)

**Async workflow:**

| Status | Meaning |
|--------|---------|
| `queued` | waiting in queue |
| `preprocessing` | OpenCV steps |
| `legend_parsed` | catalog ready |
| `detecting` | matching symbols |
| `done` / `failed` | JSON result or error |

Frontend **polls** or uses **WebSockets**. Python worker **updates** the same job row in DB.

**Why:** image processing can take seconds or minutes; user must see progress.

---

## Where ML and DL sit on THIS project

| Piece | Typical technique |
|-------|-------------------|
| Colored ducts | OpenCV HSV masks |
| Fixed symbol set | Template matching |
| Noisy scans, rotation | CNN classifier or YOLO |
| Descriptions | OCR |
| “What does RTU mean in our manual?” | RAG on PDFs (separate feature) |
| Fraud / usage analytics elsewhere | XGBoost on tables (your other interview topic) |

**Start simple, measure precision/recall per symbol type, add ML only when metrics justify cost.**

---

## One story to tell in the interview (60 seconds)

“We receive a legend and a floor plan. We build a symbol catalog from the legend with OCR. We preprocess the plan, detect candidates with color and geometry, match each to the catalog, and associate nearby OCR text. Results are JSON with bounding boxes and confidence. Processing runs async with job status for the UI. If template matching is not robust enough, we fine-tune a detector on labeled crops and keep human review for low confidence.”

---

## What to learn in code order

1. Python variables, dict/list, functions (Part 1)
2. Run `examples/see_ml_flow.py` (Part 2)
3. Run `examples/hvac_image_pipeline.py` and open `output/hvac_demo/annotated.png`
4. Compare the three `mask_*.png` files to the original drawing
5. Open `detections.json` and connect each JSON field to a Python dict/list
6. Read Part 3 so you can explain CNN vs rules

### What the OpenCV demo teaches

The demo deliberately starts with **classical computer vision**, not AI training:

1. The image becomes a NumPy array of pixel values.
2. OpenCV converts BGR pixels to HSV, where color ranges are easier to define.
3. `inRange` creates one black/white mask for each HVAC color layer.
4. Connected-components analysis groups neighboring colored pixels into regions.
5. Simple geometry labels each region as a duct/line, symbol candidate, or small mark.
6. Python serializes the regions to JSON for a .NET API or database.

This is a good MVP, but it does **not** understand every engineering symbol. Real drawings may need tuned color ranges, OCR for labels, template matching, or a trained object detector.

Run it on another drawing with:

```powershell
python examples/hvac_image_pipeline.py --input "C:\path\drawing.png" --output-dir output/my_drawing
```

You do **not** need to memorize every OpenCV function for the interview. You need the **pipeline story** and honest depth on Python + ML workflows.
