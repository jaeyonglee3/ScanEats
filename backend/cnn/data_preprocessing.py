import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define constants
IMAGE_SIZE = (224, 224)  # Set your desired image size
BATCH_SIZE = 32

# Construct ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values to [0, 1]
    rotation_range=20,         # Randomly rotate images
    width_shift_range=0.2,    # Randomly shift images horizontally
    height_shift_range=0.2,   # Randomly shift images vertically
    horizontal_flip=True,     # Randomly flip images horizontally
    vertical_flip=True,       # Randomly flip images vertically
    shear_range=0.2,          # Shear transformations
    zoom_range=0.2,           # Zoom transformations
    fill_mode='nearest'       # Fill mode for pixel values outside the image
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for the test set

# Create generators for training and validation data
train_generator = train_datagen.flow_from_directory(
    '../dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' 
)

validation_generator = test_datagen.flow_from_directory(
    '../dataset/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
