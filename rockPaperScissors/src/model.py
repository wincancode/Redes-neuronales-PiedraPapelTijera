import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def create_model():
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=(150,150,3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Resizing(128, 128),

        # Convolutional layers
        tf.keras.layers.Flatten(),


        # Dense layers 
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    return model