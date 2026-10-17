"""Backward-compatible entry point — HVAC lesson lives under projects/hvac/."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    lesson = Path(__file__).resolve().parent / "projects" / "hvac" / "lesson_01_pixels.py"
    runpy.run_path(str(lesson), run_name="__main__")
