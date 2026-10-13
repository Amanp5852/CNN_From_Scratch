import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# --------------------------------------------------
# 1. DATASET CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 32
IMAGES_PER_CLASS = 200
NUM_CLASSES = 5
RANDOM_SEED = 42

CLASS_NAMES = [
    "triangle",
    "square",
    "star",
    "cross",
    "pentagon"
]

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# RGB colors: every shape can appear in every color.
COLORS = [
    (230, 45, 45),     # Red
    (35, 180, 65),     # Green
    (40, 90, 230),     # Blue
    (240, 195, 25),    # Yellow
    (165, 65, 195)     # Purple
]


# --------------------------------------------------
# 2. DEFINE THE FIVE SHAPE GEOMETRIES
# --------------------------------------------------

def create_polygon(shape_name):
    """
    Return polygon vertices for one shape.
    Vertices are centered around (16, 16).
    """

    cx = IMAGE_SIZE // 2
    cy = IMAGE_SIZE // 2

    if shape_name == "triangle":
        angles = [-90, 30, 150]
        radius = 10
        points = [
            (
                cx + radius * math.cos(math.radians(a)),
                cy + radius * math.sin(math.radians(a))
            )
            for a in angles
        ]

    elif shape_name == "square":
        points = [
            (cx - 8, cy - 8),
            (cx + 8, cy - 8),
            (cx + 8, cy + 8),
            (cx - 8, cy + 8)
        ]

    elif shape_name == "star":
        points = []

        for i in range(10):
            angle = -90 + i * 36
            radius = 10 if i % 2 == 0 else 4.5

            x = cx + radius * math.cos(
                math.radians(angle)
            )
            y = cy + radius * math.sin(
                math.radians(angle)
            )

            points.append((x, y))

    elif shape_name == "cross":
        points = [
            (-3, -10), (3, -10),
            (3, -3), (10, -3),
            (10, 3), (3, 3),
            (3, 10), (-3, 10),
            (-3, 3), (-10, 3),
            (-10, -3), (-3, -3)
        ]

        points = [(cx + x, cy + y) for x, y in points]

    elif shape_name == "pentagon":
        angles = [-90, -18, 54, 126, 198]
        radius = 10

        points = [
            (
                cx + radius * math.cos(math.radians(a)),
                cy + radius * math.sin(math.radians(a))
            )
            for a in angles
        ]

    else:
        raise ValueError(f"Unknown shape: {shape_name}")

    return np.array(points, dtype=np.float32)


# --------------------------------------------------
# 3. GENERATE ONE IMAGE AND ITS SEGMENTATION MASK
# --------------------------------------------------

def generate_sample(class_id):
    """
    Returns:
        image: (32, 32, 3), RGB uint8
        mask:  (32, 32), binary uint8
    """

    shape_name = CLASS_NAMES[class_id]

    # Slightly varied, near-white background.
    background = rng.integers(238, 256, size=3)

    image = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        background,
        dtype=np.uint8
    )

    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    # Choose a color and vary its brightness.
    base_color = np.array(
        COLORS[rng.integers(len(COLORS))],
        dtype=np.float32
    )

    brightness = rng.uniform(0.75, 1.10)

    color = np.clip(
        base_color * brightness,
        0,
        255
    ).astype(np.uint8)

    # Draw the same shape into both image and mask.
    polygon = create_polygon(shape_name).astype(np.int32)

    cv2.fillPoly(
        image,
        [polygon],
        color=tuple(int(v) for v in color)
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    # Apply geometric transformations to both.
    angle = float(rng.uniform(0, 360))
    scale = float(rng.uniform(0.65, 1.10))
    tx = float(rng.uniform(-4, 4))
    ty = float(rng.uniform(-4, 4))

    matrix = cv2.getRotationMatrix2D(
        (IMAGE_SIZE / 2, IMAGE_SIZE / 2),
        angle,
        scale
    )

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    image = cv2.warpAffine(
        image,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(v) for v in background)
    )

    mask = cv2.warpAffine(
        mask,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Optional blur: image only, not the ground-truth mask.
    if rng.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Optional sensor-like noise: image only.
    if rng.random() < 0.25:
        noise_std = rng.uniform(2, 7)

        noise = rng.normal(
            0,
            noise_std,
            image.shape
        )

        image = np.clip(
            image.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

    # Convert mask values from 0/255 to 0/1.
    mask = (mask > 0).astype(np.uint8)

    return image, mask


# --------------------------------------------------
# 4. GENERATE THE COMPLETE DATASET
# --------------------------------------------------

all_images = []
all_labels = []
all_masks = []

for class_id, class_name in enumerate(CLASS_NAMES):

    print(f"Generating {class_name} images...")

    for _ in range(IMAGES_PER_CLASS):

        image, mask = generate_sample(class_id)

        all_images.append(image)
        all_labels.append(class_id)
        all_masks.append(mask)


X = np.array(all_images, dtype=np.uint8)
y = np.array(all_labels, dtype=np.int64)
masks = np.array(all_masks, dtype=np.uint8)

print("\nComplete dataset generated")
print("Images:", X.shape)
print("Labels:", y.shape)
print("Masks:", masks.shape)


# --------------------------------------------------
# 5. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

splits = {
    "train": [],
    "val": [],
    "test": []
}

for class_id in range(NUM_CLASSES):

    class_indices = np.where(y == class_id)[0]
    class_indices = rng.permutation(class_indices)

    n = len(class_indices)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    splits["train"].extend(class_indices[:n_train])
    splits["val"].extend(
        class_indices[n_train:n_train + n_val]
    )
    splits["test"].extend(
        class_indices[n_train + n_val:]
    )


# Shuffle the samples within each split.
for split_name in splits:
    splits[split_name] = rng.permutation(
        splits[split_name]
    )


# --------------------------------------------------
# 6. SAVE IMAGES, LABELS AND MASKS
# --------------------------------------------------

dataset = {}

for split_name, indices in splits.items():

    dataset[f"X_{split_name}"] = X[indices]
    dataset[f"y_{split_name}"] = y[indices]
    dataset[f"masks_{split_name}"] = masks[indices]

    print(
        f"{split_name}: "
        f"{len(indices)} images"
    )

dataset["class_names"] = np.array(CLASS_NAMES)

np.savez_compressed(
    OUTPUT_DIR / "shapes_dataset.npz",
    **dataset
)

print("\nSaved:", OUTPUT_DIR / "shapes_dataset.npz")


# --------------------------------------------------
# 7. VISUALIZE IMAGES AND THEIR MASKS
# --------------------------------------------------

fig, axes = plt.subplots(
    NUM_CLASSES,
    4,
    figsize=(9, 11)
)

for class_id, class_name in enumerate(CLASS_NAMES):

    class_indices = np.where(y == class_id)[0]
    sample_indices = rng.choice(
        class_indices,
        size=2,
        replace=False
    )

    for sample_number, sample_index in enumerate(sample_indices):

        image = X[sample_index]
        mask = masks[sample_index]

        image_col = sample_number * 2
        mask_col = image_col + 1

        axes[class_id, image_col].imshow(image)
        axes[class_id, image_col].set_title(
            f"{class_name}: image"
        )

        axes[class_id, mask_col].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[class_id, mask_col].set_title(
            f"{class_name}: mask"
        )

        axes[class_id, image_col].axis("off")
        axes[class_id, mask_col].axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dataset_preview.png",
    dpi=150
)

plt.show()