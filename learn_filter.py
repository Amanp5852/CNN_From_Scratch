import numpy as np


# --------------------------------------------------
# 1. Input
# --------------------------------------------------

image = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
], dtype=float)


# --------------------------------------------------
# 2. Random filter
# --------------------------------------------------

np.random.seed(42)

filter = np.random.randn(3, 3)

print("Initial filter:")
print(filter)


# --------------------------------------------------
# 3. Target
# --------------------------------------------------

target = 1.0


# --------------------------------------------------
# 4. Training settings
# --------------------------------------------------

learning_rate = 0.1
epochs = 20


# --------------------------------------------------
# 5. Training loop
# --------------------------------------------------

for epoch in range(epochs):

    # Forward pass
    prediction = np.sum(image * filter)

    # Calculate loss
    loss = (prediction - target) ** 2

    # Calculate gradient
    gradient = (
        2
        * (prediction - target)
        * image
    )

    # Update filter
    filter = (
        filter
        - learning_rate * gradient
    )

    # Print progress
    print(
        "Epoch:", epoch + 1,
        "Prediction:", prediction,
        "Loss:", loss
    )