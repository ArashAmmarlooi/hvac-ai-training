# HVAC drawing project

Classical OpenCV pipeline for colored ducts and symbol candidates on shop drawings.

## Folders

| Folder | Purpose |
|--------|---------|
| `data/` | Source drawings (`hvac_drawing.jpg`, `hvacshopdrawing2.jpg`) |
| `outputs/` | Mask images from `lesson_01_pixels.py` |
| `output/` | Annotated images and JSON from `hvac_image_pipeline.py` |

## Commands (from repo root `Training/`)

```powershell
python projects/hvac/lesson_01_pixels.py
python projects/hvac/lesson_01_pixels.py --no-display
python projects/hvac/hvac_image_pipeline.py
python projects/hvac/hvac_image_pipeline.py --input "projects/hvac/data/hvac_drawing.jpg" --output-dir projects/hvac/output/my_run
```

Legacy shortcut from repo root: `python lesson_01_pixels.py` (runs the same HVAC lesson).
