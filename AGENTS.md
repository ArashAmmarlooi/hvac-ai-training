# HVAC AI Training Project

## Project Purpose

This repository is a beginner-friendly learning project for interview preparation. It teaches Python, NumPy, OpenCV, machine learning, deep learning, OCR, symbol-to-text matching, and the eventual use of RAG for HVAC manuals and specifications.

The practical target is an HVAC drawing pipeline:

`image or PDF -> preprocessing -> symbol detection -> OCR -> geometric matching -> structured JSON -> optional RAG lookup`

## Primary Learning Goal

The learner wants to create and train a custom HVAC symbol-detection model. Classical OpenCV exercises are preparation for understanding images, establishing a baseline, and preparing training data; they are not the final objective. The practical model path should cover dataset creation, bounding-box annotation, train/validation/test splitting, transfer learning with an object detector such as YOLO, evaluation, inference, and integration with OCR and geometric matching.

## Teaching and Coding Preferences

- Communicate in English unless the user asks otherwise.
- Teach one small step at a time and wait for the learner's result before advancing.
- Explain the algorithm and purpose before presenting code.
- Avoid full-project code dumps unless explicitly requested.
- Connect each exercise to the HVAC drawing and interview questions.
- Correct errors directly but explain why they occurred.
- Prefer small runnable Python files with descriptive variable names.
- Use type hints when they improve understanding; avoid unnecessary abstractions.
- Remember that OpenCV stores colors as BGR and NumPy indexes images as `[y, x]`.
- Do not assume access to Cursor Agent/Composer history or unrelated ChatGPT chats. Ask for an attached file, selected text, or a short handoff when conversation-specific context is required.

## Active Areas

- `README.md`: project entry point and commands.
- `guide/`: conceptual lessons, interview transcript, and `05_rules_and_strategies.md` (roadmap and conventions).
- `projects/hvac/`: HVAC drawings (`data/`), `lesson_01_pixels.py`, `hvac_image_pipeline.py`, generated `outputs/` and `output/`.
- `projects/fruit/`: fruit photo (`data/`), class images (`dataset/`), crop and CNN training scripts (ML practice only).
- `clients/hvac-classifier-ui/`: Angular HVAC upload UI; future `hvac-classifier-api` (C#) and `services/hvac-inference-api` (FastAPI).
- `examples/`: shared demos such as `see_ml_flow.py` (not tied to one project).
- `lesson_01_pixels.py`: thin wrapper that runs `projects/hvac/lesson_01_pixels.py`.
- Do not treat generated folders under `projects/*/output*` or `outputs/` as source files.

## Common Commands

Run commands from the repository root:

```powershell
python lesson_01_pixels.py
python projects/hvac/lesson_01_pixels.py
python examples/see_ml_flow.py
python projects/hvac/hvac_image_pipeline.py
python projects/hvac/hvac_image_pipeline.py --input "projects/hvac/data/hvac_drawing.jpg" --output-dir projects/hvac/output/my_drawing
python projects/fruit/train_fruit_cnn.py
python -m py_compile lesson_01_pixels.py projects/hvac/lesson_01_pixels.py projects/hvac/hvac_image_pipeline.py projects/fruit/train_fruit_cnn.py examples/see_ml_flow.py
```

## Change Guidelines

- Keep changes focused on the current lesson or requested feature.
- Preserve learner-authored experiments unless a correction is requested.
- Do not modify generated files under `output/` manually.
- Validate changed Python files with `python -m py_compile` and run the narrowest relevant script.
- Do not add large frameworks or model dependencies before explaining why they are needed.
- Keep durable project facts in this file or checked-in documentation, not only in chat history.

## Obsolete Folders

- `learn_python/` is legacy material and should not receive new work.
- `interview_prep/` is legacy material and should not receive new work.
- Existing files in those folders may be consulted for historical context, but new lessons and implementations belong in `guide/`, `projects/hvac/`, `projects/fruit/`, or `examples/` for shared demos.
