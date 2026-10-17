"""Create and clean blue, green, and red masks for an HVAC drawing.

Learning purpose:
    This lesson demonstrates the first classical-computer-vision stages of the
    HVAC project. It turns an ordinary color image into three binary masks.
    A binary mask contains only black pixels (not selected) and white pixels
    (selected). Later lessons can use these white regions to find contours and
    bounding boxes that may represent ducts, symbols, or equipment.

Pipeline:
    1. Load the HVAC drawing as a NumPy array.
    2. Convert OpenCV's BGR colors to HSV colors.
    3. Select blue, green, and red pixels with HSV ranges.
    4. Remove small noise and fill tiny gaps in each mask.
    5. Save the original and cleaned masks for comparison.

Run normally to display the images:
    python lesson_01_pixels.py

Run without opening image windows:
    python lesson_01_pixels.py --no-display
"""

# sys provides access to command-line arguments in sys.argv. We use it only to
# detect whether the user supplied the optional --no-display argument.
import sys

# pathlib.Path represents filesystem paths as objects. It avoids fragile string
# concatenation and works cleanly with parent directories and filenames.
from pathlib import Path

# cv2 is the Python interface for OpenCV. It loads, converts, cleans, displays,
# and saves the images used throughout this lesson.
import cv2

# NumPy stores images as multidimensional arrays of numbers. It also creates
# the lower and upper HSV boundary arrays passed into cv2.inRange.
import numpy as np


# __file__ contains the location of this Python script. resolve() converts it
# to an absolute path, and parent selects the projects/hvac folder.
# This keeps paths working even when the terminal starts in another folder.
PROJECT_DIRECTORY = Path(__file__).resolve().parent

# The / operator joins Path objects safely. IMAGE_PATH now points to the source
# drawing, while OUTPUT_DIRECTORY is where generated masks will be written.
IMAGE_PATH = PROJECT_DIRECTORY / "data" / "hvac_drawing.jpg"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"

# An HSV boundary has the form [Hue, Saturation, Value].
# Hue describes the color family, Saturation describes color strength, and
# Value describes brightness. OpenCV uses H: 0-179 and S/V: 0-255.
#
# np.uint8 is an unsigned 8-bit integer type whose values range from 0 to 255.
# It matches the numeric representation expected by OpenCV image functions.
# These blue boundaries accept hues from 90 through 140 while rejecting pixels
# that are extremely gray (low saturation) or extremely dark (low value).
LOWER_BLUE = np.array([90, 50, 50], dtype=np.uint8)
UPPER_BLUE = np.array([140, 255, 255], dtype=np.uint8)

# Green occupies one continuous area of the OpenCV Hue scale, so one lower and
# one upper boundary are sufficient for the green mask.
LOWER_GREEN = np.array([35, 50, 50], dtype=np.uint8)
UPPER_GREEN = np.array([90, 255, 255], dtype=np.uint8)

# Hue is circular: after its largest value, it returns to its smallest value.
# Red lies on that boundary, near both 0 and 179. One continuous interval would
# miss some red pixels, so the code creates two red masks and combines them.
LOWER_RED_RANGE_1 = np.array([0, 50, 50], dtype=np.uint8)
UPPER_RED_RANGE_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_RANGE_2 = np.array([170, 50, 50], dtype=np.uint8)
UPPER_RED_RANGE_2 = np.array([179, 255, 255], dtype=np.uint8)


def create_red_mask(hsv_image: np.ndarray) -> np.ndarray:
    """Create one binary mask containing pixels from both red Hue ranges.

    Args:
        hsv_image: The complete drawing represented as an HSV NumPy array with
            shape ``(height, width, 3)``.

    Returns:
        A two-dimensional NumPy array with shape ``(height, width)``. A value
        of 255 means that the source pixel is red; 0 means it is not red.
    """
    # cv2.inRange compares every HSV pixel with all three lower and upper
    # boundaries. A pixel must satisfy the Hue, Saturation, and Value tests.
    # Matching pixels become white (255); nonmatching pixels become black (0).
    red_mask_1 = cv2.inRange(
        hsv_image,
        LOWER_RED_RANGE_1,
        UPPER_RED_RANGE_1,
    )
    red_mask_2 = cv2.inRange(
        hsv_image,
        LOWER_RED_RANGE_2,
        UPPER_RED_RANGE_2,
    )

    # bitwise_or compares the masks pixel by pixel. The output pixel is white
    # when red_mask_1 OR red_mask_2 is white, preserving both kinds of red.
    return cv2.bitwise_or(red_mask_1, red_mask_2)


def clean_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Remove small noise and close tiny gaps in a binary mask.

    Args:
        mask: A two-dimensional binary image containing values 0 and 255.
        kernel: The small neighborhood shape OpenCV moves across the mask.

    Returns:
        A cleaned binary mask with the same width and height as the input.
    """
    # Morphological opening is erosion followed by dilation. Erosion first
    # removes tiny white regions, and dilation restores the surviving regions.
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing reverses that order: dilation followed by erosion.
    # It fills tiny black holes and reconnects small breaks in white regions.
    return cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)


def find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find outside boundaries around connected white regions."""

    # Search for connected white regions in the binary mask.
    contours, _ = cv2.findContours(
        mask.copy(),

        # Return only outside boundaries, not holes inside objects.
        cv2.RETR_EXTERNAL,

        # Store only important boundary points instead of every pixel.
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return contours


def save_image(output_path: Path, image: np.ndarray) -> None:
    """Save an image and raise an informative error if writing fails.

    Args:
        output_path: The complete destination path, including the filename.
        image: The OpenCV/NumPy image array that should be written.
    """
    # OpenCV expects a string path, so str converts the Path object. imwrite
    # returns a Boolean instead of automatically raising an error on failure.
    saved_successfully = cv2.imwrite(str(output_path), image)

    # Explicitly raise an exception so a missing output file cannot go unnoticed.
    if not saved_successfully:
        raise OSError(f"Could not save image: {output_path}")


def main() -> None:
    """Load the drawing, create masks, clean them, and report the results."""
    # parents=True creates missing parent folders if necessary. exist_ok=True
    # means running the lesson again is allowed when outputs already exists.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # imread decodes the JPEG and returns a NumPy array. A normal color image
    # has shape (height, width, 3), and OpenCV orders its channels as B, G, R.
    image = cv2.imread(str(IMAGE_PATH))

    # OpenCV returns None when the path is wrong or the file is unreadable.
    if image is None:
        raise ValueError(f"Image was not found or could not be read: {IMAGE_PATH}")

    # NumPy shape returns height first, width second, and channels third. This
    # differs from the common coordinate notation (x, y), where x means width.
    height, width, channels = image.shape

    # HSV separates the basic color (Hue) from color strength and brightness.
    # This makes thresholding more understandable than comparing BGR channels.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Each inRange call processes every pixel at once using optimized OpenCV
    # code. This vectorized operation avoids a slow Python loop over pixels.
    blue_mask = cv2.inRange(hsv_image, LOWER_BLUE, UPPER_BLUE)
    green_mask = cv2.inRange(hsv_image, LOWER_GREEN, UPPER_GREEN)
    red_mask = create_red_mask(hsv_image)

    # MORPH_RECT creates a 3x3 rectangular array of ones. During cleaning,
    # OpenCV centers this kernel on each pixel and examines its eight neighbors.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # A dictionary maps each readable color name to its NumPy mask. The loop
    # below can then apply identical operations without copying code three times.
    original_masks = {
        "blue": blue_mask,
        "green": green_mask,
        "red": red_mask,
    }
    # The type hint says keys are strings and values are NumPy arrays. The empty
    # dictionary will be populated one color at a time inside the loop.
    cleaned_masks: dict[str, np.ndarray] = {}

    # items() supplies both the dictionary key and value during each iteration.
    for color_name, original_mask in original_masks.items():
        # Clean the current mask and retain it for display later.
        cleaned_mask = clean_mask(original_mask, kernel)
        cleaned_masks[color_name] = cleaned_mask

        # Create descriptive output file paths.
        original_path = OUTPUT_DIRECTORY / f"{color_name}_mask.png"
        cleaned_path = OUTPUT_DIRECTORY / f"cleaned_{color_name}_mask.png"

        # Save both versions for visual comparison.
        save_image(original_path, original_mask)
        save_image(cleaned_path, cleaned_mask)

        # absdiff calculates the absolute difference at every pixel. Unchanged
        # pixels become 0; pixels altered by opening or closing become nonzero.
        difference = cv2.absdiff(original_mask, cleaned_mask)

        # Because difference is binary, countNonZero counts exactly how many
        # pixel locations changed. It does not determine whether changes are good.
        changed_pixel_count = cv2.countNonZero(difference)

        print(
            f"{color_name.capitalize()}: saved successfully; "
            f"changed pixels = {changed_pixel_count}"
        )

    # These diagnostic messages verify paths, dimensions, Python types, and mask
    # values. A correct binary mask should normally contain only 0 and 255.
    print(f"Image path: {IMAGE_PATH}")
    print(f"Image type: {type(image)}")
    print(f"Image dimensions: width={width}, height={height}, channels={channels}")
    print(f"Mask dimensions: {blue_mask.shape}")
    print(f"Blue mask values: {np.unique(blue_mask)}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")

    # sys.argv contains the words typed after the script name. The expression is
    # False when --no-display is present, preventing GUI windows during testing.
    display_images = "--no-display" not in sys.argv

    if display_images:
        # Show the source image and before/after masks.
        cv2.imshow("Original HVAC drawing", image)
        cv2.imshow("Original blue mask", original_masks["blue"])
        cv2.imshow("Cleaned blue mask", cleaned_masks["blue"])
        cv2.imshow("Original green mask", original_masks["green"])
        cv2.imshow("Cleaned green mask", cleaned_masks["green"])
        cv2.imshow("Original red mask", original_masks["red"])
        cv2.imshow("Cleaned red mask", cleaned_masks["red"])

        # waitKey(0) pauses indefinitely until a key is pressed while an OpenCV
        # window is active. destroyAllWindows then releases every GUI window.
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Python sets __name__ to "__main__" when this file is executed directly. If a
# future lesson imports these functions, main() will not run automatically.
if __name__ == "__main__":
    main()
