import numpy as np


def create_vertical_image():
    image = np.zeros((5, 5, 3))

    thickness = np.random.randint(1, 3)
    start_col = np.random.randint(0, 5 - thickness)

    intensity = np.random.uniform(0.7, 1.0)

    image[:, start_col:start_col + thickness, 0] = intensity

    return image


def create_horizontal_image():
    image = np.zeros((5, 5, 3))

    thickness = np.random.randint(1, 3)
    start_row = np.random.randint(0, 5 - thickness)

    intensity = np.random.uniform(0.7, 1.0)

    image[start_row:start_row + thickness, :, 1] = intensity

    return image


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

images = []
labels = []


# Class 1 → red vertical
for _ in range(50):

    image = create_vertical_image()

    images.append(image)
    labels.append(1)


# Class 0 → green horizontal
for _ in range(50):

    image = create_horizontal_image()

    images.append(image)
    labels.append(0)


images = np.array(images)
labels = np.array(labels)


print("Dataset shape:", images.shape)
print("Labels shape:", labels.shape)


# --------------------------------------------------
# Shuffle dataset
# --------------------------------------------------

indices = np.random.permutation(len(images))

images = images[indices]
labels = labels[indices]


# --------------------------------------------------
# Train / Validation / Test split
# --------------------------------------------------

train_end = int(0.70 * len(images))
validation_end = int(0.85 * len(images))


train_images = images[:train_end]
train_labels = labels[:train_end]

validation_images = images[train_end:validation_end]
validation_labels = labels[train_end:validation_end]

test_images = images[validation_end:]
test_labels = labels[validation_end:]


print("Training set:", train_images.shape)
print("Validation set:", validation_images.shape)
print("Test set:", test_images.shape)