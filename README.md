# Start here — Python, ML, DL, and your drawing project

**Read in order:**

1. [`guide/01_python_basics.md`](guide/01_python_basics.md) — variables, data structures, functions (every line explained)
2. [`guide/02_machine_learning.md`](guide/02_machine_learning.md) — what ML is and how training works
3. [`guide/03_deep_learning.md`](guide/03_deep_learning.md) — neural networks and images
4. [`guide/04_your_interview_project.md`](guide/04_your_interview_project.md) — HVAC plans, symbols, how AI fits
5. [`guide/05_rules_and_strategies.md`](guide/05_rules_and_strategies.md) — roadmap, coding rules, dataset and YOLO strategy

## Project folders

| Folder | What it is |
|--------|------------|
| [`projects/hvac/`](projects/hvac/README.md) | Shop drawings, masks, contours, HVAC OpenCV pipeline |
| [`projects/fruit/`](projects/fruit/README.md) | Fruit CNN practice (separate images and dataset) |
| [`examples/`](examples/) | Shared demos (`see_ml_flow.py`) |
| [`clients/hvac-classifier-ui/`](clients/README.md) | Angular UI for HVAC upload (C# + FastAPI later) |
| [`guide/`](guide/) | Concept lessons |

## Run the code demos

From this `Training` folder:

```powershell
python examples/see_ml_flow.py
python projects/hvac/lesson_01_pixels.py
python projects/hvac/hvac_image_pipeline.py
python projects/fruit/train_fruit_cnn.py
```

- `see_ml_flow.py` trains and evaluates a small classical ML model.
- HVAC scripts read images from `projects/hvac/data/` and write masks to `projects/hvac/outputs/` or pipeline results to `projects/hvac/output/`.

To process your own PNG/JPG drawing:

```powershell
python projects/hvac/hvac_image_pipeline.py --input "C:\path\drawing.png" --output-dir projects/hvac/output/my_drawing
```

Old folders `learn_python/` and `interview_prep/` are obsolete — you can delete them manually in Explorer if you want a clean folder.
