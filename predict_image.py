import tensorflow as tf
#We use 'utils' to load images, it is more reliable in newer versions
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import sys

#Configuration
MODEL_PATH = 'gender_detector_cnn.h5'
IMAGE_SIZE = (150, 150)
CLASS_LABELS = {0: 'Female', 1: 'Male'} 

def predict_gender(img_path):
    if not os.path.exists(img_path):
        print(f"Error: The file '{img_path}' does not exist.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print("Error: Could not load model. Make sure you ran 'train_model.py' first.")
        print(e)
        return

    # Load and preprocess the image
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        label = CLASS_LABELS[1]
        confidence = prediction * 100
    else:
        label = CLASS_LABELS[0]
        confidence = (1 - prediction) * 100

    print("--- Prediction Result ---")
    print(f"Image: {img_path}")
    print(f"Gender: {label} ({confidence:.2f}%)")
    print("-------------------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
    else:
        predict_gender(sys.argv[1])
