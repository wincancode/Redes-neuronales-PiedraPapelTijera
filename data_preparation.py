import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_datasets as tfds

# Load dataset
dataset, info = tfds.load('rock_paper_scissors', with_info=True, as_supervised=True)
train = dataset['train']
test = dataset['test']

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (150, 150))  # Resize to 150x150
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    label = tf.one_hot(label, 3)  # One-hot encode labels
    return image, label

# Apply preprocessing and batch
batch_size = 32
train = train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test = test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return patch + self.position_embedding(positions)

def transformer_encoder(inputs, num_heads, dense_dim, dropout=0.1):
    # Self-attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    
    # Feed-forward network
    y = layers.Dense(dense_dim, activation="gelu")(x)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    return layers.LayerNormalization(epsilon=1e-6)(y + x)

def create_vit_model():
    inputs = layers.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)
    
    # Create patches (15x15)
    patches = layers.Conv2D(64, kernel_size=15, strides=15, padding="valid")(x)
    patches = layers.Reshape((10 * 10, 64))(patches)  # 150/15 = 10
    
    # Add positional embeddings
    encoded_patches = PatchEncoder(num_patches=10*10, projection_dim=64)(patches)
    
    # Transformer encoder blocks
    x = transformer_encoder(encoded_patches, num_heads=4, dense_dim=128)
    x = transformer_encoder(x, num_heads=4, dense_dim=128)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="gelu")(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    
    return Model(inputs, outputs)

model = create_vit_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train,
    validation_data=test,
    epochs=20,
)