# Path provides a reliable way to work with file and folder locations.
from pathlib import Path

# OpenCV reads, resizes, and converts our images.
import cv2

# NumPy stores the complete dataset as numerical arrays.
import numpy as np


# __file__ represents the location of this Python file.
# .resolve() creates its absolute path.
# .parent selects the folder containing this file.
example_directory = Path(__file__).resolve().parent

# Class folders: dataset/apple, dataset/banana, dataset/orange
dataset_directory = example_directory / "dataset"


# Each class receives a numerical label based on its list position:
# apple = 0, banana = 1, orange = 2
class_names = ["apple", "banana", "orange"]


# A CNN requires all input images to have identical dimensions.
# OpenCV expects resize dimensions in (width, height) order.
image_size = (128, 128)


# This list temporarily stores the processed images.
images = []

# This list stores the correct numerical label for every image.
labels = []


# enumerate() gives us both the numerical label and the class name.
for class_index, class_name in enumerate(class_names):
    # Build a path such as fruit_dataset/apple.
    class_directory = dataset_directory / class_name

    # Stop with a clear message if one of the class folders is missing.
    if not class_directory.exists():
        raise ValueError(f"Class folder was not found: {class_directory}")

    # Read every file in a consistent alphabetical order.
    for image_path in sorted(class_directory.iterdir()):
        # Ignore files that are not supported image formats.
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            # Skip the file instead of stopping the complete program.
            print("Unsupported image format:", image_path.name)
            continue

        # Load the image. OpenCV loads color channels in BGR order.
        image = cv2.imread(str(image_path))

        # imread() returns None when an image cannot be opened.
        # Skip that image instead of stopping the complete program.
        if image is None:
            print("Could not load:", image_path)
            continue

        # Resize every image to 128 x 128 pixels.
        # INTER_AREA usually works well when reducing image dimensions.
        resized_image = cv2.resize(
            image,
            image_size,
            interpolation=cv2.INTER_AREA,
        )

        # Convert OpenCV's BGR channel order into RGB.
        # RGB is the common channel order used for neural-network images.
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Store the processed image as one future CNN input.
        images.append(rgb_image)

        # Store the correct answer associated with that image.
        labels.append(class_index)


# Convert the image list into one NumPy array.
# Expected shape: (number_of_images, height, width, color_channels)
# float32 is commonly used for neural-network calculations.
X = np.array(images, dtype=np.float32)

# Original pixels range from 0 to 255.
# Division converts them to values from 0.0 to 1.0, which helps training.
X = X / 255.0

# Convert the labels into a NumPy integer array.
# Example labels: [0, 0, 1, 1, 2, 2]
y = np.array(labels, dtype=np.int32)


# X contains the input images that will be given to the CNN.
print("Image array shape:", X.shape)

# y contains the correct answer for each image.
print("Label array shape:", y.shape)

# Display all numerical labels and their class-name mapping.
print("Labels:", y)
print("Classes:", class_names)




## Split the Dataset into Training and Testing Sets

# Create a repeatable random-number generator.
random_generator = np.random.default_rng(seed=123)


#These lists will contain image positions, not the images themselves.
train_indices = []
validation_indices = []

## Split each fruit class separately.
# This ensures every class appears in both groups.

for class_index in range(len(class_names)):

        # Find the positions belonging to the current class.
        class_indices = np.where(y ==class_index)[0]

        # We need at least two images to place one image in each group.
        if len(class_indices) < 2:
            raise ValueError(f"Class '{class_names[class_index]}' needs at least 2 images."
)


        # Randomize their order.
        random_generator.shuffle(class_indices)

        # Calculate where the 80% training portion ends.
        split_position = int(len(class_indices) * 0.8)

        # Prevent either group from becoming empty.
        split_position = max(1, min(split_position, len(class_indices) - 1))

        # Everything before the split goes into training.
        class_train_indices = class_indices[:split_position]

        # Everything after the split goes into validation.
        class_validation_indices = class_indices[split_position:]

        # Add this class's indices to the complete lists.
        train_indices.extend(class_train_indices)
        validation_indices.extend(class_validation_indices)

# Convert the lists into NumPy arrays.
train_indices = np.array(train_indices, dtype=np.int32)
validation_indices = np.array(validation_indices, dtype=np.int32)

# Display the sizes of the training and validation sets.
print("Training set size:", len(train_indices))
print("Validation set size:", len(validation_indices))