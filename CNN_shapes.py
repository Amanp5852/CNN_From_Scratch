import numpy as np
import matplotlib.pyplot as plt

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

def update_parameters(
    filters,
    filter_biases,
    classifier_weights,
    classifier_biases,
    gradient_filters,
    gradient_filter_biases,
    gradient_classifier_weights,
    gradient_classifier_biases,
    learning_rate
):
    filters -= learning_rate * gradient_filters
    filter_biases -= learning_rate * gradient_filter_biases

    classifier_weights -= (
        learning_rate * gradient_classifier_weights
    )
    classifier_biases -= (
        learning_rate * gradient_classifier_biases
    )

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

def train_one_sample(image, true_class, learning_rate=0.001):

    # 1. Forward pass
    result = forward_pass(image, return_intermediates=True)

    # 2. Calculate loss before the parameter update
    initial_loss = cross_entropy_loss(
        result["probabilities"],
        true_class
    )

    # 3. Backpropagation through the dense layer
    (
        gradient_logits,
        gradient_classifier_weights,
        gradient_classifier_biases,
        gradient_flattened_features
    ) = dense_layer_backward(
        flattened_features=result["flattened_features"],
        probabilities=result["probabilities"],
        true_class=true_class,
        classifier_weights=classifier_weights
    )

    # 4. Reverse flattening
    gradient_pooled_maps = gradient_flattened_features.reshape(
        result["pooled_maps"].shape
    )

    # 5. Backpropagation through max pooling
    gradient_activated_maps = max_pooling_backward(
        gradient_pooled=gradient_pooled_maps,
        max_positions=result["max_positions"],
        input_shape=result["activated_maps"].shape,
        pool_size=2,
        stride=2
    )

    # 6. Backpropagation through ReLU
    gradient_convolution_maps = relu_backward(
        gradient_output=gradient_activated_maps,
        convolution_maps=result["convolution_maps"]
    )

    # 7. Backpropagation through convolution
    (
        gradient_filters,
        gradient_filter_biases
    ) = convolution_backward(
        image=image,
        gradient_feature_maps=gradient_convolution_maps,
        filters=filters
    )

    # 8. Update all parameters after calculating all gradients
    update_parameters(
        filters=filters,
        filter_biases=filter_biases,
        classifier_weights=classifier_weights,
        classifier_biases=classifier_biases,
        gradient_filters=gradient_filters,
        gradient_filter_biases=gradient_filter_biases,
        gradient_classifier_weights=gradient_classifier_weights,
        gradient_classifier_biases=gradient_classifier_biases,
        learning_rate=learning_rate
    )

    # Return the loss measured before this update.
    # No extra forward pass is needed here.
    return initial_loss 

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
# 8. EVALUATE MODEL ON A DATASET
# --------------------------------------------------

def evaluate_model(images, labels):
    total_loss = 0.0
    correct_predictions = 0

    for image, true_class in zip(images, labels):
        probabilities = forward_pass(image)

        total_loss += cross_entropy_loss(
            probabilities,
            int(true_class)
        )

        predicted_class = int(np.argmax(probabilities))

        if predicted_class == true_class:
            correct_predictions += 1

    average_loss = total_loss / len(images)
    accuracy = correct_predictions / len(images)

    return average_loss, accuracy

# --------------------------------------------------
# 8. TRAINING LOOP
# --------------------------------------------------

EPOCHS = 30
LEARNING_RATE = 0.001
REPORT_EVERY = 10

# Select one image to track throughout training.
sample_index = 0
sample_image = X_train[sample_index]
true_class = int(y_train[sample_index])

# Measure the initial model before any training.
initial_train_loss, initial_train_accuracy = evaluate_model(
    X_train, y_train
)

initial_val_loss, initial_val_accuracy = evaluate_model(
    X_val, y_val
)

initial_probabilities = forward_pass(sample_image).copy()

probability_history = [initial_probabilities.copy()]

# Store the best parameters based on validation loss.
best_val_loss = initial_val_loss
best_epoch = 0

best_parameters = {
    "filters": filters.copy(),
    "filter_biases": filter_biases.copy(),
    "classifier_weights": classifier_weights.copy(),
    "classifier_biases": classifier_biases.copy()
}

# Store metrics for later analysis or plotting.
training_history = {
    "epoch": [0],
    "train_loss": [initial_train_loss],
    "val_loss": [initial_val_loss],
    "train_accuracy": [initial_train_accuracy],
    "val_accuracy": [initial_val_accuracy]
}

print("\nINITIAL MODEL")
print(f"Training loss:      {initial_train_loss:.4f}")
print(f"Training accuracy:  {initial_train_accuracy:.2%}")
print(f"Validation loss:    {initial_val_loss:.4f}")
print(f"Validation accuracy:{initial_val_accuracy:.2%}")
print("Tracked image true class:", CLASS_NAMES[true_class])
print(
    "Tracked image initial prediction:",
    CLASS_NAMES[np.argmax(initial_probabilities)]
)

# Reproducible shuffling.
rng = np.random.default_rng(123)

print("\nTRAINING STARTED")

for epoch in range(1, EPOCHS + 1):

    indices = rng.permutation(len(X_train))

    for index in indices:
        image = X_train[index]
        true_label = int(y_train[index])

        train_one_sample(
            image=image,
            true_class=true_label,
            learning_rate=LEARNING_RATE
        )

    # Report only every REPORT_EVERY epochs.
    if epoch % REPORT_EVERY == 0 or epoch == EPOCHS:

        # Evaluate without updating any parameters.
        train_loss, train_accuracy = evaluate_model(
            X_train, y_train
        )

        val_loss, val_accuracy = evaluate_model(
            X_val, y_val
        )

        current_probabilities = forward_pass(sample_image).copy()

        probability_history.append(current_probabilities)

        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["val_accuracy"].append(val_accuracy)

        # Save parameters whenever validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            best_parameters = {
                "filters": filters.copy(),
                "filter_biases": filter_biases.copy(),
                "classifier_weights": classifier_weights.copy(),
                "classifier_biases": classifier_biases.copy()
            }

        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        print(f"Training loss:      {train_loss:.4f}")
        print(f"Training accuracy:  {train_accuracy:.2%}")
        print(f"Validation loss:    {val_loss:.4f}")
        print(f"Validation accuracy:{val_accuracy:.2%}")

        print("Tracked image probabilities:")

        for name, probability in zip(
            CLASS_NAMES, current_probabilities
        ):
            print(f"  {name:10s}: {probability:.4f}")

# Restore the best parameters observed during training.
filters[:] = best_parameters["filters"]
filter_biases[:] = best_parameters["filter_biases"]
classifier_weights[:] = best_parameters["classifier_weights"]
classifier_biases[:] = best_parameters["classifier_biases"]

print("\nTRAINING FINISHED")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best epoch: {best_epoch}")

# Compare the tracked image before training and with the best model.
final_probabilities = forward_pass(sample_image).copy()

print("\nPROBABILITY COMPARISON")
print(f"True class: {CLASS_NAMES[true_class]}")
print(f"{'Class':10s} {'Before':>10s} {'After':>10s} {'Change':>10s}")

for name, before, after in zip(
    CLASS_NAMES, initial_probabilities, final_probabilities
):
    print(
        f"{name:10s} {before:10.4f} "
        f"{after:10.4f} {after - before:+10.4f}"
    )

print(
    "\nFinal prediction using best model:",
    CLASS_NAMES[np.argmax(final_probabilities)]
)

# --------------------------------------------------
# 10. CONFUSION MATRIX - VALIDATION SET
# --------------------------------------------------

# Number of classes
num_classes = len(CLASS_NAMES)

# Rows = actual classes, columns = predicted classes
confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)

# Predict each validation image
for image, true_label in zip(X_val, y_val):

    probabilities = forward_pass(image)

    predicted_label = int(np.argmax(probabilities))
    true_label = int(true_label)

    confusion_matrix[true_label, predicted_label] += 1


# Print the numerical confusion matrix
print("\nCONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted\n")

print(f"{'Actual / Pred.':15s}", end="")

for name in CLASS_NAMES:
    print(f"{name:>12s}", end="")

print()

for i, name in enumerate(CLASS_NAMES):
    print(f"{name:15s}", end="")

    for j in range(num_classes):
        print(f"{confusion_matrix[i, j]:12d}", end="")

    print()


# Calculate accuracy separately for each actual class
print("\nPER-CLASS ACCURACY")

for i, name in enumerate(CLASS_NAMES):

    total_actual = np.sum(confusion_matrix[i])
    correct = confusion_matrix[i, i]

    accuracy = correct / total_actual if total_actual > 0 else 0.0

    print(
        f"{name:10s}: "
        f"{correct}/{total_actual} correct "
        f"({accuracy:.2%})"
    )


# Overall validation accuracy from the matrix
overall_accuracy = (
    np.trace(confusion_matrix) / np.sum(confusion_matrix)
)

print(f"\nOverall validation accuracy: {overall_accuracy:.2%}")


# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(confusion_matrix, cmap="Blues")

ax.set_title("CNN Confusion Matrix - Validation Set")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))

ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)

# Write each cell's count into the plot
threshold = confusion_matrix.max() / 2

for i in range(num_classes):
    for j in range(num_classes):

        value = confusion_matrix[i, j]

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color="white" if value > threshold else "black"
        )

fig.colorbar(image, ax=ax, label="Number of images")
fig.tight_layout()

plt.show()