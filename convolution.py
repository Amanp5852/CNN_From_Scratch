import numpy as np


# --------------------------------------------------
# 1. Create image
# --------------------------------------------------

image = np.array([
    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ],

    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ],

    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ],

    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ],

    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ]
])

filter_vertical = np.array([
    [
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, 1, -1]
    ],
    [
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, 1, -1]
    ],
    [
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, 1, -1]
    ]
])

filter_horizontal = np.array([
    [
        [-1, -1, -1],
        [ 1,  1,  1],
        [-1, -1, -1]
    ],
    [
        [-1, -1, -1],
        [ 1,  1,  1],
        [-1, -1, -1]
    ],
    [
        [-1, -1, -1],
        [ 1,  1,  1],
        [-1, -1, -1]
    ]
])

filter_diagonal = np.array([
    [
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  1]
    ],
    [
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  1]
    ],
    [
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  1]
    ]
])


filters = np.array([
    filter_vertical,
    filter_horizontal,
    filter_diagonal
])

print("Filters shape:", filters.shape)

def convolution_multiple_filters(
        image,
        filters,
        stride=1,
        padding=0):

    feature_maps = []

    for filter in filters:

        feature_map = convolution(
            image,
            filter,
            stride=stride,
            padding=padding
        )

        feature_maps.append(feature_map)

    return np.array(feature_maps)

# --------------------------------------------------
# 4. Single-kernel convolution
# --------------------------------------------------

def convolution(
        image,
        filter,
        bias=0,
        stride=1,
        padding=0):

    # Add padding
    if padding > 0:

        image = np.pad(
            image,
            (
                (padding, padding),
                (padding, padding),
                (0, 0)
            ),
            mode='constant',
            constant_values=0
        )

    image_height, image_width, input_channels = image.shape

    filter_height, filter_width, filter_channels = filter.shape

    # Check that filter depth matches image channels
    if input_channels != filter_channels:
        raise ValueError(
            "Filter channels must match image channels"
        )

    # Calculate output dimensions
    output_height = (
        (image_height - filter_height) // stride
    ) + 1

    output_width = (
        (image_width - filter_width) // stride
    ) + 1

    # Create feature map
    feature_map = np.zeros(
        (output_height, output_width)
    )

    # Move filter over image
    for i in range(output_height):

        for j in range(output_width):

            row = i * stride
            col = j * stride

            image_region = image[
                row:row + filter_height,
                col:col + filter_width,
                :
            ]

            # Element-wise multiplication
            # followed by summation
            result = np.sum(image_region * filter) + bias

            feature_map[i, j] = result

    return feature_map

def relu(feature_map):

    return np.maximum(0, feature_map)


feature_map = convolution(
    image,
    filter_vertical,
    bias=0
)

feature_map = relu(feature_map)

print(feature_map)