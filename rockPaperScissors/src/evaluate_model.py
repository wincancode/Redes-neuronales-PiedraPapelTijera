import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
from config import IMG_SIZE

# Load the trained model
model = tf.keras.models.load_model("rps_vit_model.keras")
def generate_synthetic_images(model, class_index, steps=500, learning_rate=0.05):
    """Generate synthetic images that maximize the prediction for a specific class."""
    synthetic_image = tf.Variable(tf.random.uniform((1, *IMG_SIZE, 3)))

    def loss_fn():
        predictions = model(synthetic_image)
        regularization = tf.reduce_mean(tf.square(synthetic_image - 0.5))
        return -predictions[0, class_index] + 0.01 * regularization

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for step in range(steps):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        gradients = tape.gradient(loss, [synthetic_image])
        optimizer.apply_gradients(zip(gradients, [synthetic_image]))
        synthetic_image.assign(tf.clip_by_value(synthetic_image, 0.0, 1.0))

    return synthetic_image[0].numpy()

# Generate and visualize synthetic images for each class
CLASS_NAMES = ['paper', 'rock', 'scissors']
for i, class_name in enumerate(CLASS_NAMES):
    synthetic_image = generate_synthetic_images(model, i)

    # Debugging: Print synthetic image stats
    print(f"Synthetic image stats for class {class_name}: min={np.min(synthetic_image)}, max={np.max(synthetic_image)}")

    plt.figure()
    plt.title(f"Synthetic Image for Class: {class_name}")
    plt.imshow(synthetic_image.clip(0, 1))  # Ensure values are in [0, 1]
    plt.axis('off')
    plt.show()

# Generate and visualize synthetic images for each class
CLASS_NAMES = ['paper', 'rock', 'scissors']
for i, class_name in enumerate(CLASS_NAMES):
    synthetic_image = generate_synthetic_images(model, i)

    plt.figure()
    plt.title(f"Synthetic Image for Class: {class_name}")
    plt.imshow(synthetic_image)
    plt.axis('off')
    plt.show()