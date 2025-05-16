import cv2
import numpy as np
import tensorflow as tf
from rembg import remove
from PIL import Image
import io

# Load model without custom layer
model = tf.keras.models.load_model("rps_vit_model.keras")

# Class names (verify order matches your training data)
CLASS_NAMES = ['paper', 'rock', 'scissors']  # Adjust if different

# Video capture setup
cap = cv2.VideoCapture(0)
roi_size = 300  # Detection area size
x_start, y_start = 100, 100  # Top-left corner position


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Draw detection zone
    cv2.rectangle(frame, (x_start, y_start), 
                 (x_start + roi_size, y_start + roi_size), (0, 255, 0), 2)
    
    # Extract and process ROI
    roi = frame[y_start:y_start+roi_size, x_start:x_start+roi_size]

    if roi.size != 0:
        # Convert ROI to PIL Image for background removal
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA))
        roi_no_bg = remove(roi_pil)

        # Convert to NumPy array
        roi_no_bg = np.array(roi_no_bg.convert('RGB'))

        # Preprocess image
        resized = cv2.resize(roi_no_bg, (150, 150))  # Match training size
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # Predict
        predictions = model.predict(input_tensor, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label = f"{CLASS_NAMES[predicted_class]} ({confidence:.2f})"

        # Display prediction
        cv2.putText(frame, label, (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Rock-Paper-Scissors Detection', frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()