import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Visualize RGB image
# --------------------------------------------------

def show_rgb_image(image, title="RGB Image"):

    # Convert RGB → BGR for OpenCV compatibility
    image_uint8 = (image * 255).astype(np.uint8)

    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR
    )

    # Convert back to RGB for matplotlib display
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


# --------------------------------------------------
# Visualize feature maps
# --------------------------------------------------

def show_feature_maps(
        feature_maps,
        title="Feature Maps"):

    number_of_filters = feature_maps.shape[0]

    fig, axes = plt.subplots(
        1,
        number_of_filters,
        figsize=(8, 4)
    )

    # Handle the case of only one feature map
    if number_of_filters == 1:
        axes = [axes]

    for i in range(number_of_filters):

        axes[i].imshow(
            feature_maps[i],
            cmap="gray"
        )

        axes[i].set_title(
            f"Filter {i}"
        )

        axes[i].axis("off")

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Visualize pooled feature maps
# --------------------------------------------------

def show_pooled_feature_maps(
        pooled_feature_maps,
        title="Pooled Feature Maps"):

    number_of_filters = pooled_feature_maps.shape[0]

    fig, axes = plt.subplots(
        1,
        number_of_filters,
        figsize=(8, 4)
    )

    if number_of_filters == 1:
        axes = [axes]

    for i in range(number_of_filters):

        axes[i].imshow(
            pooled_feature_maps[i],
            cmap="gray"
        )

        axes[i].set_title(
            f"Filter {i}"
        )

        axes[i].axis("off")

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()