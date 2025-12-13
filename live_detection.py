import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'gender_detector_cnn.h5'
IMAGE_SIZE = (150, 150)
# Adjust these labels based on your train_model.py output
CLASS_LABELS = {0: 'Female', 1: 'Male'} 

# --- Setup ---
# 1. Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load the pre-trained OpenCV face detector (a simple Haar Cascade)
# You need the 'haarcascade_frontalface_default.xml' file (easily found online)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Initialize the webcam
cap = cv2.VideoCapture(0)

print("Starting camera feed. Press 'q' to exit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the face region (ROI)
        face_roi = frame[y:y + h, x:x + w]
        
        # Pre-process the face for the CNN
        processed_face = cv2.resize(face_roi, IMAGE_SIZE)
        processed_face = cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB)
        processed_face = processed_face.astype('float32') / 255.0
        processed_face = np.expand_dims(processed_face, axis=0)
        
        # Make the prediction
        prediction = model.predict(processed_face)[0][0]
        
        # Determine the label and confidence
        if prediction > 0.5:
            label = CLASS_LABELS[1]
            confidence = prediction * 100
        else:
            label = CLASS_LABELS[0]
            confidence = (1 - prediction) * 100
        
        # Display the result on the frame
        text = f"{label}: {confidence:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gender Detection Live', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
