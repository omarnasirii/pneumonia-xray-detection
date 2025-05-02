import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

# Set up image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset and apply transformations
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification: Pneumonia or Normal
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

# Save the model
model.save('pneumonia_detection_model.h5')

# Evaluate the model on the test set (optional)
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")

# Prediction Function
def predict_pneumonia(img_path):
    # Load an image file to be predicted, resizing it to 150x150 pixels
    img = image.load_img(img_path, target_size=(150, 150))

    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)

    # Rescale the image (scale pixel values to [0, 1])
    img_array = img_array / 255.0

    # Add batch dimension (model expects a batch of images, not a single image)
    img_array = np.expand_dims(img_array, axis=0)

    # Load the trained model
    model = tf.keras.models.load_model('pneumonia_detection_model.h5')

    # Make the prediction
    prediction = model.predict(img_array)
    
    # Output the prediction
    if prediction < 0.5:
        print("Prediction: Normal")
    else:
        print("Prediction: Pneumonia")
    
