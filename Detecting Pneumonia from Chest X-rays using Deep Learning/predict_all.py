import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('pneumonia_detection_model.h5')

# Define test image folder
test_dir = 'data/test'
classes = ['NORMAL', 'PNEUMONIA']  # class folder names

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    return 'Pneumonia' if prediction >= 0.5 else 'Normal', float(prediction)

# Loop through all images in both test class folders
for label in classes:
    folder = os.path.join(test_dir, label)
    for fname in os.listdir(folder):
        img_path = os.path.join(folder, fname)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            result, confidence = predict_image(img_path)
            print(f"{fname} ({label}) ➜ Predicted: {result} (Confidence: {confidence:.2f})")
