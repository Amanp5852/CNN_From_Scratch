import cv2
import numpy as np


def show_image_comparison(
    image,
    feature_maps,
    pooled_feature_maps,
    image_index
):
    """
    Display the original image, ReLU feature maps,
    and pooled feature maps in one window.

    Press any key to move to the next image.
    Press ESC to exit.
    """

    # --------------------------------------------------
    # Convert RGB image for OpenCV
    # --------------------------------------------------

    image_uint8 = (image * 255).astype(np.uint8)

    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------
    # Resize original image
    # --------------------------------------------------

    original = cv2.resize(
        image_bgr,
        (200, 200),
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------------
    # Prepare feature maps
    # --------------------------------------------------

    feature_map_images = []

    for i in range(feature_maps.shape[0]):

        feature_map = feature_maps[i]

        # Normalize feature map to 0-255
        normalized = cv2.normalize(
            feature_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        normalized = cv2.resize(
            normalized,
            (200, 200),
            interpolation=cv2.INTER_NEAREST
        )

        normalized = cv2.cvtColor(
            normalized,
            cv2.COLOR_GRAY2BGR
        )

        feature_map_images.append(normalized)


    # --------------------------------------------------
    # Prepare pooled feature maps
    # --------------------------------------------------

    pooled_images = []

    for i in range(pooled_feature_maps.shape[0]):

        pooled = pooled_feature_maps[i]

        normalized = cv2.normalize(
            pooled,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        normalized = cv2.resize(
            normalized,
            (200, 200),
            interpolation=cv2.INTER_NEAREST
        )

        normalized = cv2.cvtColor(
            normalized,
            cv2.COLOR_GRAY2BGR
        )

        pooled_images.append(normalized)


    # --------------------------------------------------
    # Create labels
    # --------------------------------------------------

    def add_title(image, title):

        title_bar = np.zeros(
            (40, image.shape[1], 3),
            dtype=np.uint8
        )

        cv2.putText(
            title_bar,
            title,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return np.vstack(
            [title_bar, image]
        )


    # --------------------------------------------------
    # Add titles
    # --------------------------------------------------

    panels = []

    panels.append(
        add_title(
            original,
            "Original"
        )
    )

    for i, feature_map in enumerate(feature_map_images):

        panels.append(
            add_title(
                feature_map,
                f"Filter {i}"
            )
        )

    for i, pooled in enumerate(pooled_images):

        panels.append(
            add_title(
                pooled,
                f"Pooled {i}"
            )
        )


    # --------------------------------------------------
    # Combine all panels horizontally
    # --------------------------------------------------

    comparison = np.hstack(panels)


    # --------------------------------------------------
    # Add image number
    # --------------------------------------------------

    cv2.putText(
        comparison,
        f"Image {image_index}",
        (10, comparison.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------
    # Show window
    # --------------------------------------------------

    cv2.imshow(
        "CNN Visualization",
        comparison
    )