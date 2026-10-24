import numpy as np
"""
Understanding of the flow and backpragation
"""

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

data = np.load("data/shapes_dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

CLASS_NAMES = data["class_names"].tolist()

# Normalize image pixels from [0, 255] to [0, 1].
X_train = X_train.astype(np.float64) / 255.0
X_val = X_val.astype(np.float64) / 255.0
X_test = X_test.astype(np.float64) / 255.0

print("Training images:", X_train.shape)
print("Number of classes:", len(CLASS_NAMES))


# --------------------------------------------------
# 2. INITIALIZE PARAMETERS
# --------------------------------------------------

rng = np.random.default_rng(42)

NUM_FILTERS = 2
FILTER_SIZE = 3
NUM_CLASSES = len(CLASS_NAMES)

# Each filter has dimensions 3 x 3 x 3:
# height x width x RGB channels.
filters = rng.normal(
    0,
    np.sqrt(2.0 / (FILTER_SIZE * FILTER_SIZE * 3)),
    size=(NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, 3)
)

filter_biases = np.zeros(NUM_FILTERS)

# Convolution output: 30 x 30 x 2
# Pooled output: 15 x 15 x 2
# Flattened feature count: 15 * 15 * 2 = 450
NUM_FEATURES = 15 * 15 * NUM_FILTERS

# Dense layer: 450 inputs -> 5 output logits.
classifier_weights = rng.normal(
    0,
    np.sqrt(2.0 / NUM_FEATURES),
    size=(NUM_FEATURES, NUM_CLASSES)
)

classifier_biases = np.zeros(NUM_CLASSES)


# --------------------------------------------------
# 3. MULTI-CHANNEL CONVOLUTION
# --------------------------------------------------

def convolution_multi_channel(image, filter_weights, bias=0.0):
    """
    image:          (H, W, 3)
    filter_weights: (3, 3, 3)
    returns:        (H - 2, W - 2)
    """

    height, width, channels = image.shape
    filter_height, filter_width, filter_channels = (
        filter_weights.shape
    )

    if channels != filter_channels:
        raise ValueError("Image and filter channel counts differ")

    output_height = height - filter_height + 1
    output_width = width - filter_width + 1

    output = np.zeros(
        (output_height, output_width),
        dtype=np.float64
    )

    for i in range(output_height):
        for j in range(output_width):

            region = image[
                i:i + filter_height,
                j:j + filter_width,
                :
            ]

            output[i, j] = (
                np.sum(region * filter_weights) + bias
            )

    return output


def convolution_multiple_filters(image, filters, filter_biases):
    """
    Applies every filter to the same RGB image.

    Returns:
        (number_of_filters, output_height, output_width)
    """

    feature_maps = []

    for i, filter_weights in enumerate(filters):
        feature_map = convolution_multi_channel(
            image,
            filter_weights,
            bias=filter_biases[i]
        )

        feature_maps.append(feature_map)

    return np.array(feature_maps)


# --------------------------------------------------
# 4. RELU ACTIVATION
# --------------------------------------------------

def relu(feature_maps):
    return np.maximum(0, feature_maps)


# --------------------------------------------------
# 5. MAX POOLING
# --------------------------------------------------

def max_pooling(feature_maps, pool_size=2, stride=2):
    """
    Forward max pooling.

    Returns:
        pooled: (filters, pooled_H, pooled_W)
        max_positions: (filters, pooled_H, pooled_W, 2)

    Each max_positions entry stores the (row, column)
    of the maximum within its local pooling window.
    """

    num_filters, height, width = feature_maps.shape

    output_height = ((height - pool_size) // stride) + 1
    output_width = ((width - pool_size) // stride) + 1

    pooled = np.zeros(
        (num_filters, output_height, output_width)
    )

    max_positions = np.zeros(
        (num_filters, output_height, output_width, 2),
        dtype=np.int64
    )

    for f in range(num_filters):
        for i in range(output_height):
            for j in range(output_width):

                h_start = i * stride
                w_start = j * stride

                region = feature_maps[
                    f,
                    h_start:h_start + pool_size,
                    w_start:w_start + pool_size
                ]

                # Find the maximum value in this region.
                flat_index = np.argmax(region)
                local_row, local_col = np.unravel_index(
                    flat_index,
                    region.shape
                )

                pooled[f, i, j] = region[local_row, local_col]

                # Store the maximum's position relative
                # to this pooling window.
                max_positions[f, i, j] = (
                    local_row,
                    local_col
                )

    return pooled, max_positions

def max_pooling_backward(
    gradient_pooled,
    max_positions,
    input_shape,
    pool_size=2,
    stride=2
):
    """
    Distributes pooled gradients to the positions
    that won during the forward max-pooling pass.

    gradient_pooled: shape (filters, pooled_H, pooled_W)
    max_positions:   shape (filters, pooled_H, pooled_W, 2)
    input_shape:     original feature-map shape
    """

    gradient_input = np.zeros(input_shape, dtype=np.float64)

    num_filters, output_height, output_width = (
        gradient_pooled.shape
    )

    for f in range(num_filters):
        for i in range(output_height):
            for j in range(output_width):

                h_start = i * stride
                w_start = j * stride

                local_row, local_col = max_positions[f, i, j]

                gradient_input[
                    f,
                    h_start + local_row,
                    w_start + local_col
                ] += gradient_pooled[f, i, j]

    return gradient_input


def relu_backward(gradient_output, convolution_maps):
    """
    Backpropagates through ReLU.

    Positive convolution outputs pass the gradient.
    Zero or negative convolution outputs receive zero.
    """

    gradient_input = gradient_output.copy()

    gradient_input[convolution_maps <= 0] = 0

    return gradient_input

def convolution_backward(image, gradient_feature_maps, filters):
    """
    Calculates gradients for convolution filters and biases.

    image:                 (H, W, channels)
    gradient_feature_maps: (num_filters, out_H, out_W)
    filters:               (num_filters, filter_H, filter_W, channels)

    Returns:
        gradient_filters:     same shape as filters
        gradient_filter_biases: (num_filters,)
    """

    num_filters, filter_height, filter_width, channels = (
        filters.shape
    )

    _, output_height, output_width = gradient_feature_maps.shape

    gradient_filters = np.zeros_like(filters)
    gradient_filter_biases = np.zeros(num_filters)

    for f in range(num_filters):

        for i in range(output_height):
            for j in range(output_width):

                region = image[
                    i:i + filter_height,
                    j:j + filter_width,
                    :
                ]

                gradient = gradient_feature_maps[f, i, j]

                gradient_filters[f] += region * gradient

                gradient_filter_biases[f] += gradient

    return gradient_filters, gradient_filter_biases

# --------------------------------------------------
# 6. SOFTMAX
# --------------------------------------------------

def softmax(logits):
    """
    Converts class logits into probabilities.
    """

    shifted_logits = logits - np.max(logits)
    exp_values = np.exp(shifted_logits)

    return exp_values / np.sum(exp_values)

def cross_entropy_loss(probabilities, true_class):
    """
    probabilities: predicted probabilities, shape (5,)
    true_class: integer label from 0 to 4
    """

    epsilon = 1e-12

    correct_probability = probabilities[true_class]

    loss = -np.log(
        max(correct_probability, epsilon)
    )

    return loss

def dense_layer_backward(
    flattened_features,
    probabilities,
    true_class,
    classifier_weights
):
    """
    Backpropagation through softmax + cross-entropy
    and the dense layer.

    flattened_features: (450,)
    probabilities:      (5,)
    true_class:         integer from 0 to 4
    classifier_weights: (450, 5)

    Returns:
        gradient_logits:           (5,)
        gradient_classifier_weights: (450, 5)
        gradient_classifier_biases:  (5,)
        gradient_flattened_features: (450,)
    """

    # 1. Gradient of cross-entropy + softmax
    gradient_logits = probabilities.copy()
    gradient_logits[true_class] -= 1.0

    # 2. Gradient of dense-layer weights
    gradient_classifier_weights = np.outer(
        flattened_features,
        gradient_logits
    )

    # 3. Gradient of dense-layer biases
    gradient_classifier_biases = gradient_logits.copy()

    # 4. Pass gradient backward to the flattened features
    gradient_flattened_features = (
        classifier_weights @ gradient_logits
    )

    return (
        gradient_logits,
        gradient_classifier_weights,
        gradient_classifier_biases,
        gradient_flattened_features
    )


# --------------------------------------------------
# 7. FORWARD PASS FOR ONE IMAGE
# --------------------------------------------------

def forward_pass(image, return_intermediates=False):

    # Convolution
    convolution_maps = convolution_multiple_filters(
        image,
        filters,
        filter_biases
    )

    # ReLU
    activated_maps = relu(convolution_maps)

    # Max pooling
    pooled_maps, max_positions = max_pooling(
    activated_maps,
    pool_size=2,
    stride=2
    )

    # Flatten all pooled feature maps into one vector.
    flattened_features = pooled_maps.flatten()

    # Dense layer produces five logits.
    logits = (
        flattened_features @ classifier_weights
        + classifier_biases
    )

    # Convert logits to class probabilities.
    probabilities = softmax(logits)

    if return_intermediates:
        return {
            "convolution_maps": convolution_maps,
            "activated_maps": activated_maps,
            "pooled_maps": pooled_maps,
            "flattened_features": flattened_features,
            "logits": logits,
            "max_positions": max_positions,
            "probabilities": probabilities
        }

    return probabilities


# --------------------------------------------------
# 8. TEST THE FORWARD PASS
# --------------------------------------------------

sample_index = 0
sample_image = X_train[sample_index]

result = forward_pass(
    sample_image,
    return_intermediates=True
)

print("\nFORWARD PASS SHAPES")
print("Input:", sample_image.shape)
print("Convolution:", result["convolution_maps"].shape)
print("After ReLU:", result["activated_maps"].shape)
print("After pooling:", result["pooled_maps"].shape)
print("Flattened:", result["flattened_features"].shape)
print("Logits:", result["logits"].shape)
print("Probabilities:", result["probabilities"].shape)

predicted_class = np.argmax(result["probabilities"])

print("\nActual class:", CLASS_NAMES[y_train[sample_index]])
print("Predicted class:", CLASS_NAMES[predicted_class])

print("\nClass probabilities:")

for class_name, probability in zip(
    CLASS_NAMES,
    result["probabilities"]
):
    print(f"{class_name:10s}: {probability:.4f}")

print(
    "\nProbability sum:",
    np.sum(result["probabilities"])
)

loss = cross_entropy_loss(
    result["probabilities"],
    y_train[sample_index]
)

print("\nCross-entropy loss:", loss)

# --------------------------------------------------
# 9. TEST DENSE-LAYER BACKPROPAGATION
# --------------------------------------------------

(
    gradient_logits,
    gradient_classifier_weights,
    gradient_classifier_biases,
    gradient_flattened_features
) = dense_layer_backward(
    result["flattened_features"],
    result["probabilities"],
    y_train[sample_index],
    classifier_weights
)

print("\nDENSE-LAYER BACKPROPAGATION")
print("Gradient logits:", gradient_logits.shape)
print(
    "Gradient classifier weights:",
    gradient_classifier_weights.shape
)
print(
    "Gradient classifier biases:",
    gradient_classifier_biases.shape
)
print(
    "Gradient flattened features:",
    gradient_flattened_features.shape
)

print("\nGradient logits values:")
print(gradient_logits)

print(
    "\nGradient sum:",
    np.sum(gradient_logits)
)

# --------------------------------------------------
# 10. BACKPROPAGATION THROUGH FLATTEN, POOLING, RELU
# --------------------------------------------------

# A. Reverse flattening.
gradient_pooled_maps = gradient_flattened_features.reshape(
    result["pooled_maps"].shape
)

# B. Reverse max pooling.
gradient_activated_maps = max_pooling_backward(
    gradient_pooled=gradient_pooled_maps,
    max_positions=result["max_positions"],
    input_shape=result["activated_maps"].shape,
    pool_size=2,
    stride=2
)

# C. Reverse ReLU.
gradient_convolution_maps = relu_backward(
    gradient_output=gradient_activated_maps,
    convolution_maps=result["convolution_maps"]
)

print("\nPOOLING AND RELU BACKPROPAGATION")
print("Gradient pooled maps:", gradient_pooled_maps.shape)
print("Gradient activated maps:", gradient_activated_maps.shape)
print("Gradient convolution maps:", gradient_convolution_maps.shape)

print(
    "Nonzero convolution gradients:",
    np.count_nonzero(gradient_convolution_maps)
)

# --------------------------------------------------
# 11. CONVOLUTION BACKPROPAGATION
# --------------------------------------------------

gradient_filters, gradient_filter_biases = convolution_backward(
    image=sample_image,
    gradient_feature_maps=gradient_convolution_maps,
    filters=filters
)

print("\nCONVOLUTION BACKPROPAGATION")
print("Gradient filters:", gradient_filters.shape)
print("Gradient filter biases:", gradient_filter_biases.shape)

print("\nGradient filter biases:")
print(gradient_filter_biases)

print(
    "\nGradient filter 0 norm:",
    np.linalg.norm(gradient_filters[0])
)

print(
    "Gradient filter 1 norm:",
    np.linalg.norm(gradient_filters[1])
)