import os
import tensorflow as tf
import numpy as np
from PIL import Image
from config import *

def load_processed_dataset(processed_dir):
    """Load pre-processed dataset for training"""
    return tf.keras.utils.image_dataset_from_directory(
        processed_dir,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    ).map(lambda x, y: (x/255.0, y))  # Normalize
