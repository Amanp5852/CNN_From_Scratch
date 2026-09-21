import numpy as np


# --------------------------------------------------
# 1. Vertical line examples
# Label = 1
# --------------------------------------------------

vertical_1 = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
])

vertical_2 = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
])

vertical_3 = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1]
])

vertical_4 = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
])

vertical_5 = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
])


# --------------------------------------------------
# 2. Non-vertical examples
# Label = 0
# --------------------------------------------------

horizontal_1 = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0]
])

horizontal_2 = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
])

diagonal_1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

diagonal_2 = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])

random_pattern = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])


# --------------------------------------------------
# 3. Combine images
# --------------------------------------------------

images = np.array([
    vertical_1,
    vertical_2,
    vertical_3,
    vertical_4,
    vertical_5,
    horizontal_1,
    horizontal_2,
    diagonal_1,
    diagonal_2,
    random_pattern
], dtype=float)


# --------------------------------------------------
# 4. Labels
# --------------------------------------------------

labels = np.array([
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
], dtype=float)


# --------------------------------------------------
# 5. Check shapes
# --------------------------------------------------

print("Images shape:", images.shape)

print("Labels shape:", labels.shape)


print("\nFirst image:")
print(images[0])

print("\nFirst image label:")
print(labels[0])

# --------------------------------------------------
# 6. Initialize learnable filter
# --------------------------------------------------

np.random.seed(42)

filter = np.random.randn(3, 3)

print("\nInitial filter:")
print(filter)

# --------------------------------------------------
# 7. Training settings
# --------------------------------------------------

learning_rate = 0.01
epochs = 100

def sigmoid(x):

    return 1 / (
        1 + np.exp(-x)
    )

# --------------------------------------------------
# 8. Training
# --------------------------------------------------

# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(epochs):

    total_loss = 0

    for image, target in zip(images, labels):

        # ------------------------------
        # Forward pass
        # ------------------------------

        z = np.sum(
            image * filter
        )

        prediction = sigmoid(z)


        # ------------------------------
        # Loss
        # ------------------------------

        loss = (
            prediction - target
        ) ** 2

        total_loss += loss


        # ------------------------------
        # Backpropagation
        # ------------------------------

        gradient = (
            2
            * (prediction - target)
            * prediction
            * (1 - prediction)
            * image
        )


        # ------------------------------
        # Update filter
        # ------------------------------

        filter = (
            filter
            - learning_rate * gradient
        )


    # ------------------------------
    # Print progress
    # ------------------------------

    if epoch % 10 == 0:

        average_loss = (
            total_loss / len(images)
        )

        print(
            "Epoch:",
            epoch,
            "Average Loss:",
            average_loss
        )

print("\nLearned filter:")
print(filter)

# --------------------------------------------------
# 9. Testing
# --------------------------------------------------

print("\nTesting results:\n")

for i, image in enumerate(images):

    z = np.sum(
        image * filter
    )

    prediction = sigmoid(z)

    print(
        "Image:", i,
        "| Target:", labels[i],
        "| Prediction:", prediction
    )