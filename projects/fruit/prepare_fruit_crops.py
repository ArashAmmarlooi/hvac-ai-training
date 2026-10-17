# Path helps us create reliable file locations.
from pathlib import Path

# OpenCV loads and processes the image.
import cv2

# NumPy will be used later for HSV color boundaries.
import numpy as np

# Find the folder containing this Python file.
example_directory = Path(__file__).resolve().parent

# Source photo for manual crops lives under data/.
image_path = example_directory / "data" / "photo1phone.jpg"

# Load the image as a NumPy array.
image = cv2.imread(str(image_path))

# OpenCV returns None if the image cannot be loaded.
if image is None:
    raise ValueError(f"Image could not be loaded: {image_path}")

height, width, channels = image.shape


# NumPy cropping uses: image[y_start:y_end, x_start:x_end]
apple_crop = image[145:325, 65:220]

# Create the dataset folder if it does not exist.
apple_directory = example_directory / "dataset" / "apple"
apple_directory.mkdir(parents=True, exist_ok=True)

# Save our first apple training image.
apple_path = apple_directory / "apple_001.jpg"
cv2.imwrite(str(apple_path), apple_crop)
cv2.imshow("Apple crop", apple_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Crop the orange.
# Format: image[y_start:y_end, x_start:x_end]
orange_crop = image[65:210, 215:375]

# Crop the complete banana bunch.
banana_crop = image[205:435, 260:575]


# Create class folders.
orange_directory = example_directory / "dataset" / "orange"
banana_directory = example_directory / "dataset" / "banana"

orange_directory.mkdir(parents=True, exist_ok=True)
banana_directory.mkdir(parents=True, exist_ok=True)


# Define output filenames.
orange_path = orange_directory / "orange_001.jpg"
banana_path = banana_directory / "banana_001.jpg"


# Save the cropped images.
cv2.imwrite(str(orange_path), orange_crop)
cv2.imwrite(str(banana_path), banana_crop)


# # Display all three crops.
# cv2.imshow("Apple crop", apple_crop)
# cv2.imshow("Orange crop", orange_crop)
# cv2.imshow("Banana crop", banana_crop)

# print("Apple saved:", apple_path)
# print("Orange saved:", orange_path)
# print("Banana saved:", banana_path)

# # Wait until you press a key.
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# print("Apple crop saved:", apple_path)

