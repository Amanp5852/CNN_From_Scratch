import numpy as np
import cv2
from images import (
    train_images,
    train_labels,
    validation_images,
    validation_labels,
    test_images,
    test_labels
)
from visualization import show_image_comparison

np.random.seed(42)

filters = np.random.randn(2, 3, 3, 3)

print("\nInitial filter:")
print(filters)

# --------------------------------------------------
# Convolution function
# --------------------------------------------------

def convolution(image, filter, stride=1, padding=0):

    # Add padding if needed
    if padding > 0:

        image = np.pad(
            image,
            ((padding, padding),
             (padding, padding)),
            mode="constant",
            constant_values=0
        )


    # Image dimensions
    image_height, image_width = image.shape


    # Filter dimensions
    filter_height, filter_width = filter.shape


    # Output feature map dimensions
    output_height = (
        (image_height - filter_height)
        // stride
    ) + 1


    output_width = (
        (image_width - filter_width)
        // stride
    ) + 1


    # Empty feature map
    feature_map = np.zeros(
        (output_height, output_width)
    )


    # Slide filter over image
    for i in range(output_height):

        for j in range(output_width):

            row = i * stride
            col = j * stride


            # Extract image region
            image_region = image[
                row:row + filter_height,
                col:col + filter_width
            ]


            # Multiply and sum
            feature_map[i, j] = np.sum(
                image_region * filter
            )


    return feature_map

def convolution_multi_channel(
        image,
        filter,
        stride=1,
        padding=0):

    # ------------------------------------------
    # Add padding if needed
    # ------------------------------------------

    if padding > 0:

        image = np.pad(
            image,
            (
                (padding, padding),
                (padding, padding),
                (0, 0)
            ),
            mode="constant",
            constant_values=0
        )


    # ------------------------------------------
    # Image dimensions
    # ------------------------------------------

    image_height, image_width, input_channels = image.shape


    # ------------------------------------------
    # Filter dimensions
    # ------------------------------------------

    filter_height, filter_width, filter_channels = filter.shape


    # ------------------------------------------
    # Check channel compatibility
    # ------------------------------------------

    if input_channels != filter_channels:

        raise ValueError(
            "Number of image channels must "
            "match number of filter channels."
        )


    # ------------------------------------------
    # Output dimensions
    # ------------------------------------------

    output_height = (
        (image_height - filter_height)
        // stride
    ) + 1

    output_width = (
        (image_width - filter_width)
        // stride
    ) + 1


    # ------------------------------------------
    # Empty feature map
    # ------------------------------------------

    feature_map = np.zeros(
        (output_height, output_width)
    )


    # ------------------------------------------
    # Slide filter over image
    # ------------------------------------------

    for i in range(output_height):

        for j in range(output_width):

            row = i * stride
            col = j * stride


            # ----------------------------------
            # Extract RGB region
            # ----------------------------------

            image_region = image[
                row:row + filter_height,
                col:col + filter_width,
                :
            ]


            # ----------------------------------
            # Multiply and sum
            # ----------------------------------

            feature_map[i, j] = np.sum(
                image_region * filter
            )


    return feature_map

def convolution_multiple_filters(
        image,
        filters,
        stride=1,
        padding=0):

    feature_maps = []

    for filter in filters:

        feature_map = convolution_multi_channel(
            image,
            filter,
            stride=stride,
            padding=padding
        )

        feature_maps.append(feature_map)

    return np.array(feature_maps)

# --------------------------------------------------
# ReLU activation
# --------------------------------------------------

def relu(x):

    return np.maximum(0, x)

def relu_gradient(x):
    return (x > 0).astype(float)

def max_pooling(feature_map, pool_size=2, stride=2):

    height, width = feature_map.shape

    output_height = (height - pool_size) // stride + 1
    output_width = (width - pool_size) // stride + 1

    pooled = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):

            row = i * stride
            col = j * stride

            region = feature_map[
                row:row + pool_size,
                col:col + pool_size
            ]

            pooled[i, j] = np.max(region)

    return pooled


def max_pooling_gradient(
        feature_map,
        gradient_pooled,
        pool_size=2,
        stride=1):

    height, width = feature_map.shape

    gradient_feature_map = np.zeros_like(feature_map)

    output_height, output_width = gradient_pooled.shape

    for i in range(output_height):

        for j in range(output_width):

            row = i * stride
            col = j * stride

            region = feature_map[
                row:row + pool_size,
                col:col + pool_size
            ]

            max_position = np.unravel_index(
                np.argmax(region),
                region.shape
            )

            max_row = row + max_position[0]
            max_col = col + max_position[1]

            gradient_feature_map[max_row, max_col] += (
                gradient_pooled[i, j]
            )

    return gradient_feature_map

# --------------------------------------------------
# Classifier weights
# --------------------------------------------------

classifier_weights = np.random.randn(8)

print("\nClassifier weights:")
print(classifier_weights)


# --------------------------------------------------
# Gradient of convolution filter
# --------------------------------------------------

def convolution_gradient(
        image,
        filter,
        gradient_feature_map,
        stride=1):

    # ------------------------------------------
    # Image dimensions
    # ------------------------------------------

    image_height, image_width, input_channels = image.shape

    # ------------------------------------------
    # Filter dimensions
    # ------------------------------------------

    filter_height, filter_width, filter_channels = filter.shape

    # ------------------------------------------
    # Check channel compatibility
    # ------------------------------------------

    if input_channels != filter_channels:

        raise ValueError(
            "Number of image channels must "
            "match number of filter channels."
        )

    # ------------------------------------------
    # Feature map dimensions
    # ------------------------------------------

    output_height, output_width = (
        gradient_feature_map.shape
    )

    # ------------------------------------------
    # Gradient accumulator
    # ------------------------------------------

    gradient_filter = np.zeros_like(filter)

    # ------------------------------------------
    # Go through every feature map position
    # ------------------------------------------

    for i in range(output_height):

        for j in range(output_width):

            row = i * stride
            col = j * stride

            # ----------------------------------
            # Extract corresponding RGB region
            # ----------------------------------

            image_region = image[
                row:row + filter_height,
                col:col + filter_width,
                :
            ]

            # ----------------------------------
            # Gradient contribution
            # ----------------------------------

            gradient_filter += (
                gradient_feature_map[i, j]
                * image_region
            )

    return gradient_filter

# --------------------------------------------------
# Training settings
# --------------------------------------------------

learning_rate = 0.01
epochs = 100

# --------------------------------------------------
# Sigmoid activation
# --------------------------------------------------

def sigmoid(x):

    return 1 / (
        1 + np.exp(-x)
    )

# --------------------------------------------------
# Convert image to prediction
# --------------------------------------------------

def predict(image, filters, classifier_weights):

    # Convolution with multiple filters
    feature_maps = convolution_multiple_filters(
        image,
        filters,
        stride=1,
        padding=0
    )

    # Apply ReLU
    activated_feature_maps = relu(
        feature_maps
    )

    # Apply max pooling to each feature map
    pooled_feature_maps = []

    for feature_map in activated_feature_maps:

        pooled_feature_map = max_pooling(
            feature_map,
            pool_size=2,
            stride=1
        )

        pooled_feature_maps.append(
            pooled_feature_map
        )

    pooled_feature_maps = np.array(
        pooled_feature_maps
    )

    # Flatten all pooled feature maps
    flattened_features = pooled_feature_maps.flatten()

    # Classifier weighted sum
    z = np.dot(
        flattened_features,
        classifier_weights
    )

    # Sigmoid
    prediction = sigmoid(z)

    return prediction

# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(epochs):

    total_loss = 0


    # Process every image
    for image, target in zip(train_images, train_labels):

                
        # ------------------------------------------
        # Forward pass
        # ------------------------------------------

        feature_maps = convolution_multiple_filters(
            image,
            filters,
            stride=1,
            padding=0
        )

        activated_feature_maps = relu(
            feature_maps
        )

        pooled_feature_maps = []

        for feature_map in activated_feature_maps:

            pooled_feature_map = max_pooling(
                feature_map,
                pool_size=2,
                stride=1
            )

            pooled_feature_maps.append(
                pooled_feature_map
            )

        pooled_feature_maps = np.array(
            pooled_feature_maps
        )

        flattened_features = pooled_feature_maps.flatten()

        z = np.dot(
            flattened_features,
            classifier_weights
        )

        prediction = sigmoid(z)


        # ------------------------------------------
        # Loss
        # ------------------------------------------

        loss = (
            prediction - target
        ) ** 2

        total_loss += loss


        # ------------------------------------------
        # Backpropagation
        # ------------------------------------------

        # Loss gradient
        dL_dprediction = (
            2 * (prediction - target)
        )

        # Sigmoid gradient
        dprediction_dz = (
            prediction
            * (1 - prediction)
        )


        # ------------------------------------------
        # Gradient of classifier weights
        # ------------------------------------------

        gradient_classifier_weights = (
            dL_dprediction
            * dprediction_dz
            * flattened_features
        )


        # ------------------------------------------
        # Gradient flowing back to pooled feature maps
        # ------------------------------------------

        gradient_pooled_features = (
            dL_dprediction
            * dprediction_dz
            * classifier_weights
        )

        gradient_pooled_feature_maps = (
            gradient_pooled_features.reshape(
                pooled_feature_maps.shape
            )
        )


        # ------------------------------------------
        # Gradient through max pooling
        # ------------------------------------------

        gradient_activated_feature_maps = []

        for i in range(len(filters)):

            gradient_activated_feature_map = max_pooling_gradient(
                activated_feature_maps[i],
                gradient_pooled_feature_maps[i],
                pool_size=2,
                stride=1
            )

            gradient_activated_feature_maps.append(
                gradient_activated_feature_map
            )

        gradient_activated_feature_maps = np.array(
            gradient_activated_feature_maps
        )


        # ------------------------------------------
        # Gradient through ReLU
        # ------------------------------------------

        gradient_feature_maps = (
            gradient_activated_feature_maps
            * relu_gradient(feature_maps)
        )


        # ------------------------------------------
        # Gradient of convolution filters
        # ------------------------------------------

        gradient_filters = []

        for i in range(len(filters)):

            gradient_filter = convolution_gradient(
                image,
                filters[i],
                gradient_feature_maps[i],
                stride=1
            )

            gradient_filters.append(
                gradient_filter
            )

        gradient_filters = np.array(
            gradient_filters
        )


        # ------------------------------------------
        # Update classifier weights
        # ------------------------------------------

        classifier_weights = (
            classifier_weights
            - learning_rate
            * gradient_classifier_weights
        )


        # ------------------------------------------
        # Update convolution filter
        # ------------------------------------------

        filters = (
            filters
            - learning_rate
            * gradient_filters
        )

    # ----------------------------------------------
    # Print progress
    # ----------------------------------------------

    if epoch % 10 == 0:

        average_loss = (
            total_loss / len(train_images)
        )

        print(
            "Epoch:",
            epoch,
            "Average Loss:",
            average_loss
        )

# --------------------------------------------------
# Validation
# --------------------------------------------------

validation_loss = 0
correct_predictions = 0

for image, target in zip(
    validation_images,
    validation_labels
):

    # Get prediction
    prediction = predict(
        image,
        filters,
        classifier_weights
    )

    # Calculate loss
    loss = (
        prediction - target
    ) ** 2

    validation_loss += loss

    # Convert prediction into class
    if prediction >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    # Check prediction
    if predicted_class == target:
        correct_predictions += 1


# Average validation loss
validation_loss /= len(validation_images)


# Validation accuracy
validation_accuracy = (
    correct_predictions
    / len(validation_images)
) * 100


print("\nValidation Loss:", validation_loss)

print(
    "Validation Accuracy:",
    validation_accuracy,
    "%"
)

print("\nLearned filter:")
print(filters)
for i, learned_filter in enumerate(filters):
    print(f"\nFilter {i}:")
    print(learned_filter)

for i, image in enumerate(train_images):

    feature_maps = convolution_multiple_filters(
        image,
        filters,
        stride=1,
        padding=0
    )

    activated_feature_maps = relu(feature_maps)

    print(f"\nImage {i}")
    
    for j in range(len(filters)):
        print(f"\nFilter {j} feature map:")
        print(activated_feature_maps[j])

print("\nFinal classifier weights:")
print(classifier_weights)

# --------------------------------------------------
# Visualization
# --------------------------------------------------

for i, image in enumerate(train_images):

    # Convolution
    feature_maps = convolution_multiple_filters(
        image,
        filters,
        stride=1,
        padding=0
    )

    # ReLU
    activated_feature_maps = relu(
        feature_maps
    )

    # Pooling
    pooled_feature_maps = []

    for feature_map in activated_feature_maps:

        pooled_feature_map = max_pooling(
            feature_map,
            pool_size=2,
            stride=1
        )

        pooled_feature_maps.append(
            pooled_feature_map
        )

    pooled_feature_maps = np.array(
        pooled_feature_maps
    )


    # Show everything in one window
    show_image_comparison(
        image,
        activated_feature_maps,
        pooled_feature_maps,
        i
    )


    # Wait for keyboard input
    key = cv2.waitKey(0) & 0xFF


    # ESC → exit visualization
    if key == 27:
        break


# Close window after visualization
cv2.destroyAllWindows()


