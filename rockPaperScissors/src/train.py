import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import tensorflow as tf
from model import create_model
from data_processing import load_processed_dataset
from config import *

PROCESSED_DATA_DIR = 'processed_dataset/train'  

train_ds = load_processed_dataset(PROCESSED_DATA_DIR)
val_ds = load_processed_dataset('processed_dataset/valid')  

model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
)

model.save("rps_vit_model.keras")