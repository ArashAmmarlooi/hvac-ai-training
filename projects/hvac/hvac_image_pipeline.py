"""Learn a practical OpenCV pipeline on an HVAC-style drawing.

Run from the Training folder:
    python projects/hvac/hvac_image_pipeline.py

Or process your own drawing:
    python projects/hvac/hvac_image_pipeline.py --input "projects/hvac/data/hvac_drawing.jpg"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HVAC_PROJECT_DIR = Path(__file__).resolve().parent

COLOR_RANGES = {
    "blue_supply": [((90, 80, 40), (140, 255, 255))],
    "red_return": [
        ((0, 80, 40), (12, 255, 255)),
        ((168, 80, 40), (180, 255, 255)),
    ],
    "green_equipment": [((35, 60, 40), (90, 255, 255))],
}

ANNOTATION_COLORS = {
    "blue_supply": (255, 120, 0),
    "red_return": (0, 0, 255),
    "green_equipment": (0, 170, 0),
}


def build_demo_drawing(width: int = 1200, height: int = 700) -> np.ndarray:
    """Create an HVAC-like image so the lesson runs without external files."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    wall_color = (150, 150, 150)
    cv2.rectangle(image, (45, 45), (1155, 650), wall_color, 3)
    cv2.rectangle(image, (90, 90), (1110, 600), wall_color, 2)
    cv2.line(image, (400, 90), (400, 600), wall_color, 2)
    cv2.line(image, (780, 90), (780, 600), wall_color, 2)
    cv2.line(image, (90, 350), (1110, 350), wall_color, 2)

    blue = (220, 20, 20)
    red = (20, 20, 220)
    green = (20, 200, 20)

    cv2.line(image, (160, 300), (720, 300), blue, 14)
    cv2.line(image, (520, 300), (520, 160), blue, 10)
    cv2.line(image, (720, 300), (900, 180), blue, 10)
    cv2.rectangle(image, (128, 268), (178, 318), blue, 5)
    cv2.line(image, (133, 273), (173, 313), blue, 3)
    cv2.line(image, (173, 273), (133, 313), blue, 3)
    cv2.rectangle(image, (875, 135), (925, 185), blue, 5)
    cv2.circle(image, (900, 160), 13, blue, -1)

    cv2.line(image, (150, 525), (900, 525), red, 14)
    cv2.line(image, (900, 525), (900, 390), red, 10)
    cv2.rectangle(image, (118, 470), (172, 524), red, 5)
    cv2.line(image, (123, 475), (167, 519), red, 3)
    cv2.line(image, (167, 475), (123, 519), red, 3)
    cv2.rectangle(image, (875, 365), (925, 415), red, 5)

    cv2.rectangle(image, (975, 430), (1080, 550), green, 5)
    cv2.circle(image, (1010, 490), 24, green, 4)
    cv2.circle(image, (1048, 490), 24, green, 4)
    cv2.rectangle(image, (980, 210), (1040, 300), green, 5)

    cv2.putText(image, "16X10 SUPPLY", (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
    cv2.putText(image, "18X8 RETURN", (430, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    cv2.putText(image, "RTU2", (988, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(image, "TF", (992, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    return image


def load_image(input_path: Path | None) -> tuple[np.ndarray, str]:
    if input_path is None:
        return build_demo_drawing(), "generated_demo"

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {input_path}")
    return image, str(input_path.resolve())


def make_color_mask(
    hsv_image: np.ndarray,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> np.ndarray:
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv_image, np.array(lower), np.array(upper)),
        )

    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def classify_region(width: int, height: int, fill_ratio: float) -> str:
    longest_side = max(width, height)
    shortest_side = max(1, min(width, height))
    aspect_ratio = longest_side / shortest_side

    if aspect_ratio >= 4:
        return "duct_or_line"
    if width >= 25 and height >= 25 and fill_ratio >= 0.08:
        return "symbol_or_equipment_candidate"
    return "small_mark_or_text"


def detect_regions(
    mask: np.ndarray,
    layer: str,
    min_area: int,
) -> list[dict[str, object]]:
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    detections: list[dict[str, object]] = []

    for component_id in range(1, component_count):
        x, y, width, height, pixel_area = stats[component_id]
        if int(pixel_area) < min_area:
            continue

        box_area = int(width) * int(height)
        fill_ratio = float(pixel_area) / box_area
        center_x, center_y = centroids[component_id]
        detections.append(
            {
                "layer": layer,
                "region_type": classify_region(int(width), int(height), fill_ratio),
                "bbox": [int(x), int(y), int(width), int(height)],
                "center": [round(float(center_x), 1), round(float(center_y), 1)],
                "colored_pixel_area": int(pixel_area),
                "fill_ratio": round(fill_ratio, 3),
            }
        )

    return detections


def annotate(image: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    annotated = image.copy()
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection["bbox"]
        layer = str(detection["layer"])
        color = ANNOTATION_COLORS[layer]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            annotated,
            str(index),
            (x, max(16, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return annotated


def process_drawing(
    image: np.ndarray,
    source: str,
    output_dir: Path,
    min_area: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections: list[dict[str, object]] = []

    for layer, ranges in COLOR_RANGES.items():
        mask = make_color_mask(hsv_image, ranges)
        cv2.imwrite(str(output_dir / f"mask_{layer}.png"), mask)
        detections.extend(detect_regions(mask, layer, min_area))

    result = {
        "source": source,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "technique": "HSV color segmentation plus connected components",
        "detection_count": len(detections),
        "detections": detections,
    }

    cv2.imwrite(str(output_dir / "input_used.png"), image)
    cv2.imwrite(str(output_dir / "annotated.png"), annotate(image, detections))
    (output_dir / "detections.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect colored regions in an HVAC drawing.")
    parser.add_argument("--input", type=Path, help="Optional path to a PNG or JPG drawing.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HVAC_PROJECT_DIR / "output" / "hvac_demo",
        help="Folder for masks, annotations, and JSON.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=80,
        help="Ignore connected colored regions smaller than this pixel area.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, source = load_image(args.input)
    result = process_drawing(image, source, args.output_dir, args.min_area)

    print(f"Processed: {result['source']}")
    print(f"Detected colored regions: {result['detection_count']}")
    print(f"Open the annotated image: {args.output_dir / 'annotated.png'}")
    print(f"Structured result: {args.output_dir / 'detections.json'}")


if __name__ == "__main__":
    main()
