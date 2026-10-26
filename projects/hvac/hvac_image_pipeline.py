"""Learn a practical OpenCV pipeline on an HVAC-style drawing.

Learning purpose:
    This file is the full classical-computer-vision reference pipeline for the
    HVAC project. While ``lesson_01_pixels.py`` stops after creating and
    cleaning binary color masks, this script goes all the way from an input
    drawing to bounding boxes, geometric classification, an annotated image,
    and a structured ``detections.json`` file ready for a C# or FastAPI backend.

Run from the Training folder (uses a synthetic demo drawing by default):
    python projects/hvac/hvac_image_pipeline.py

Or process your real shop drawing:
    python projects/hvac/hvac_image_pipeline.py --input "projects/hvac/data/hvac_drawing.jpg"
"""

from __future__ import annotations

# argparse reads command-line flags like --input and --output-dir.
import argparse

# json serializes Python dictionaries and lists into a structured .json file.
import json

# Path builds operating-system-safe file and folder paths.
from pathlib import Path

# cv2 (OpenCV) handles image loading, HSV conversion, masks, connected
# components, drawing boxes, and writing output images.
import cv2

# NumPy provides the multidimensional array structure that holds pixel data.
import numpy as np


# Locate the projects/hvac/ directory relative to this script so default output
# paths always stay inside projects/hvac/output/ regardless of terminal cwd.
HVAC_PROJECT_DIR = Path(__file__).resolve().parent

# Each HVAC subsystem is drawn in a distinct color on engineering shop plans.
# We map each layer name to one or more (lower_hsv, upper_hsv) tuples:
#   - blue_supply: supply air ducts and diffusers (Hue ~90-140)
#   - red_return: return/exhaust ducts and grilles (Hue wraps around 0 and 180)
#   - green_equipment: rooftop units (RTUs), fans, and mechanical equipment
COLOR_RANGES = {
    "blue_supply": [((90, 80, 40), (140, 255, 255))],
    "red_return": [
        ((0, 80, 40), (12, 255, 255)),
        ((168, 80, 40), (180, 255, 255)),
    ],
    "green_equipment": [((35, 60, 40), (90, 255, 255))],
}

# BGR colors used when drawing bounding boxes and numbers onto annotated.png.
# Remember OpenCV uses (Blue, Green, Red) order, not RGB.
ANNOTATION_COLORS = {
    "blue_supply": (255, 120, 0),
    "red_return": (0, 0, 255),
    "green_equipment": (0, 170, 0),
}


# ==============================================================================
# STEP 1A: Synthetic Drawing Generator (Fallback when no --input file is given)
# ==============================================================================
def build_demo_drawing(width: int = 1200, height: int = 700) -> np.ndarray:
    """Create a synthetic HVAC floor plan in memory so the pipeline can be tested
    even without an external image file.

    Returns:
        A BGR NumPy array of shape ``(height, width, 3)`` filled with walls,
        colored ducts, symbol boxes, and text labels.
    """
    # Start with a solid white canvas: every pixel is (255, 255, 255) in BGR.
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # Draw gray building walls and room dividers so we can verify that gray/black
    # architectural lines are ignored by the HSV color masks.
    wall_color = (150, 150, 150)
    cv2.rectangle(image, (45, 45), (1155, 650), wall_color, 3)
    cv2.rectangle(image, (90, 90), (1110, 600), wall_color, 2)
    cv2.line(image, (400, 90), (400, 600), wall_color, 2)
    cv2.line(image, (780, 90), (780, 600), wall_color, 2)
    cv2.line(image, (90, 350), (1110, 350), wall_color, 2)

    # Define BGR ink colors for the three HVAC layers.
    blue = (220, 20, 20)
    red = (20, 20, 220)
    green = (20, 200, 20)

    # Draw blue supply duct lines and square supply diffuser symbols.
    cv2.line(image, (160, 300), (720, 300), blue, 14)
    cv2.line(image, (520, 300), (520, 160), blue, 10)
    cv2.line(image, (720, 300), (900, 180), blue, 10)
    cv2.rectangle(image, (128, 268), (178, 318), blue, 5)
    cv2.line(image, (133, 273), (173, 313), blue, 3)
    cv2.line(image, (173, 273), (133, 313), blue, 3)
    cv2.rectangle(image, (875, 135), (925, 185), blue, 5)
    cv2.circle(image, (900, 160), 13, blue, -1)

    # Draw red return duct lines and return grille symbols.
    cv2.line(image, (150, 525), (900, 525), red, 14)
    cv2.line(image, (900, 525), (900, 390), red, 10)
    cv2.rectangle(image, (118, 470), (172, 524), red, 5)
    cv2.line(image, (123, 475), (167, 519), red, 3)
    cv2.line(image, (167, 475), (123, 519), red, 3)
    cv2.rectangle(image, (875, 365), (925, 415), red, 5)

    # Draw green mechanical equipment (e.g., RTU and exhaust fan).
    cv2.rectangle(image, (975, 430), (1080, 550), green, 5)
    cv2.circle(image, (1010, 490), 24, green, 4)
    cv2.circle(image, (1048, 490), 24, green, 4)
    cv2.rectangle(image, (980, 210), (1040, 300), green, 5)

    # Put engineering text annotations (sizes and equipment tags).
    cv2.putText(image, "16X10 SUPPLY", (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
    cv2.putText(image, "18X8 RETURN", (430, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    cv2.putText(image, "RTU2", (988, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(image, "TF", (992, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    return image


# ==============================================================================
# STEP 1B: Image Ingestion (Load from disk or fall back to demo drawing)
# ==============================================================================
def load_image(input_path: Path | None) -> tuple[np.ndarray, str]:
    """Load an HVAC drawing from disk or generate the synthetic demo drawing.

    Args:
        input_path: Optional path to a PNG/JPG drawing supplied via ``--input``.

    Returns:
        A tuple of ``(bgr_image_array, source_description_string)``.
    """
    # If the user did not pass --input, generate the synthetic drawing in memory.
    if input_path is None:
        return build_demo_drawing(), "generated_demo"

    # Read the image file into a BGR NumPy array of shape (height, width, 3).
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {input_path}")
    return image, str(input_path.resolve())


# ==============================================================================
# STEP 2 & 3: HSV Color Segmentation and Morphological Noise Cleaning
# ==============================================================================
def make_color_mask(
    hsv_image: np.ndarray,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> np.ndarray:
    """Create a cleaned binary mask (0 or 255) for one HVAC color layer.

    Args:
        hsv_image: The full drawing converted to HSV color space.
        ranges: List of ``(lower_hsv, upper_hsv)`` pairs. Most colors need one
            pair; red needs two because Hue wraps around 0/179.

    Returns:
        A 2D uint8 array of shape ``(height, width)`` where 255 = pixel belongs
        to this color layer and 0 = background or other colors.
    """
    # Start with an all-black 2D mask of shape (height, width).
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    # Combine every HSV range for this color using bitwise OR.
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv_image, np.array(lower), np.array(upper)),
        )

    # Clean tiny white noise specks with a 3x3 morphological opening
    # (erosion followed by dilation).
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# ==============================================================================
# STEP 4A: Rule-Based Geometric Classification (Duct vs. Symbol vs. Text)
# ==============================================================================
def classify_region(width: int, height: int, fill_ratio: float) -> str:
    """Use simple geometry to guess what a connected colored blob represents.

    Args:
        width: Bounding box width in pixels.
        height: Bounding box height in pixels.
        fill_ratio: Fraction of the bounding box that is actually white pixels
            (colored_pixel_area / (width * height)).

    Returns:
        A coarse label string: ``"duct_or_line"``,
        ``"symbol_or_equipment_candidate"``, or ``"small_mark_or_text"``.
    """
    longest_side = max(width, height)
    shortest_side = max(1, min(width, height))
    aspect_ratio = longest_side / shortest_side

    # Long, thin rectangles (4x longer than wide) are almost always duct runs.
    if aspect_ratio >= 4:
        return "duct_or_line"

    # Compact, reasonably large regions with enough colored ink are candidate
    # HVAC symbols (diffusers, fans, RTUs) that a CNN/YOLO model can classify.
    if width >= 25 and height >= 25 and fill_ratio >= 0.08:
        return "symbol_or_equipment_candidate"

    # Everything else is likely a letter, number, or small tick mark.
    return "small_mark_or_text"


# ==============================================================================
# STEP 4B: Connected-Components Region Extraction (Finding Bounding Boxes)
# ==============================================================================
def detect_regions(
    mask: np.ndarray,
    layer: str,
    min_area: int,
) -> list[dict[str, object]]:
    """Group neighboring white pixels in a mask into bounding-box detections.

    Args:
        mask: Cleaned 2D binary mask for one HVAC color layer.
        layer: Name of the color layer (e.g., ``"blue_supply"``).
        min_area: Minimum number of white pixels required to keep a region.

    Returns:
        A list of dictionaries, where each dictionary describes one detected
        region's bounding box, center point, area, and coarse geometric type.
    """
    # connectedComponentsWithStats labels every separate island of white pixels.
    # component_id 0 is always the black background, so the loop starts at 1.
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    detections: list[dict[str, object]] = []

    for component_id in range(1, component_count):
        # Extract the bounding box (x, y, width, height) and white pixel count.
        x, y, width, height, pixel_area = stats[component_id]

        # Ignore tiny noise blobs smaller than min_area pixels.
        if int(pixel_area) < min_area:
            continue

        # Calculate how much of the rectangle is filled with colored pixels.
        box_area = int(width) * int(height)
        fill_ratio = float(pixel_area) / box_area
        center_x, center_y = centroids[component_id]

        # Package this detection as a JSON-ready Python dictionary.
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


# ==============================================================================
# STEP 5: Visual Verification (Drawing Boxes and IDs on a Copy of the Plan)
# ==============================================================================
def annotate(image: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    """Draw colored bounding boxes and detection numbers on a copy of the image.

    Args:
        image: Original BGR drawing.
        detections: List of region dictionaries returned by ``detect_regions``.

    Returns:
        A new BGR image array with rectangles and index numbers drawn on top.
    """
    # Copy the image so we do not overwrite the original pixels in memory.
    annotated = image.copy()

    # Number detections starting at 1 so a human can match boxes to JSON items.
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection["bbox"]
        layer = str(detection["layer"])
        color = ANNOTATION_COLORS[layer]

        # Draw the rectangle from top-left (x, y) to bottom-right (x+w, y+h).
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


# ==============================================================================
# STEP 6: End-to-End Pipeline Orchestrator and JSON Export
# ==============================================================================
def process_drawing(
    image: np.ndarray,
    source: str,
    output_dir: Path,
    min_area: int,
) -> dict[str, object]:
    """Run the full HSV -> mask -> region detection -> JSON/image export flow.

    Args:
        image: Loaded BGR drawing.
        source: String describing where the image came from.
        output_dir: Folder where masks, ``annotated.png``, and ``detections.json``
            will be saved.
        min_area: Minimum pixel area for keeping a connected component.

    Returns:
        The structured result dictionary that was written to ``detections.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert BGR to HSV once for the whole image.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections: list[dict[str, object]] = []

    # Process each HVAC subsystem layer (blue_supply, red_return, green_equipment).
    for layer, ranges in COLOR_RANGES.items():
        mask = make_color_mask(hsv_image, ranges)
        cv2.imwrite(str(output_dir / f"mask_{layer}.png"), mask)
        detections.extend(detect_regions(mask, layer, min_area))

    # Build the top-level payload that a .NET API or frontend UI would consume.
    result = {
        "source": source,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "technique": "HSV color segmentation plus connected components",
        "detection_count": len(detections),
        "detections": detections,
    }

    # Save visual artifacts and the structured JSON file.
    cv2.imwrite(str(output_dir / "input_used.png"), image)
    cv2.imwrite(str(output_dir / "annotated.png"), annotate(image, detections))
    (output_dir / "detections.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


# ==============================================================================
# STEP 7: Command-Line Interface and Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments for input image, output path, and filter size."""
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
    """Entry point: parse arguments, load the image, run the pipeline, and print summary."""
    args = parse_args()
    image, source = load_image(args.input)
    result = process_drawing(image, source, args.output_dir, args.min_area)

    print(f"Processed: {result['source']}")
    print(f"Detected colored regions: {result['detection_count']}")
    print(f"Open the annotated image: {args.output_dir / 'annotated.png'}")
    print(f"Structured result: {args.output_dir / 'detections.json'}")


if __name__ == "__main__":
    main()
