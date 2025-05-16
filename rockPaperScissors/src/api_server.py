from flask import Flask, request, jsonify
from rembg import remove
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('rps_vit_model.keras')
CLASS_NAMES = ['paper', 'rock', 'scissors']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Get the image from the request
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGBA')

    # Remove background
    image_no_bg = remove(image)

    # Preprocess the image
    image_no_bg = Image.open(io.BytesIO(image_no_bg)).convert('RGB')
    image_no_bg = image_no_bg.resize((150, 150))  # Match model input size
    image_array = np.array(image_no_bg) / 255.0  # Normalize
    input_tensor = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(input_tensor, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Return the prediction
    return jsonify({
        'class': CLASS_NAMES[predicted_class],
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
