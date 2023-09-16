import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from data_preprocessing import train_generator
from data_preprocessing import validation_generator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

model = Sequential()

# Convolutional and pooling layers 1
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional and pooling layers 2
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the layers
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully Connected Layer 2 (Output layer with 2 classes: fresh and stale)
model.add(Dense(4))
model.add(Activation('softmax'))  # Use 'softmax' for multi-class classification

print("Compiling the model...")
model.compile(loss='categorical_crossentropy',  # Use categorical cross-entropy for multi-class
              optimizer='adam',  
              metrics=['accuracy'])
print("Model compiled successfully.")

# Print a summary of the model architecture
model.summary()

print("Training the model...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=5, 
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
print("Training complete")


print("Evaluating the model on the test data...")
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

print("Saving the model...")
model.save('fruit_freshness_model2.h5')
print("Model saved successfully.")