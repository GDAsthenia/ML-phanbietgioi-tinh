import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# --- Configuration ---
# Your folder is named 'main', so we update this:
DATA_DIR = 'main' 
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32 #number of image it will take for each literation (batch_size bigger -> faster training)
EPOCHS = 10#number of iterations
MODEL_SAVE_PATH = 'gender_detector_cnn.h5'

## --- Step 1: Data Preparation ---
print("Preparing data generators...")

# The generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# The generator for validation/test data
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training data
# NOTE: We look inside 'main/train'
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Load validation data
validation_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Map the class indices
class_labels = {v: k for k, v in train_generator.class_indices.items()}
print(f"Detected Classes: {class_labels}")

## --- Step 2: Build the CNN Model ---
print("Building the CNN model...")
model = Sequential([
    Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## --- Step 3: Train the Model ---
print("Starting model training...")
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

## --- Step 4: Save the Model ---
model.save(MODEL_SAVE_PATH)
print(f"\nTraining Complete. Model saved to: {MODEL_SAVE_PATH}")
