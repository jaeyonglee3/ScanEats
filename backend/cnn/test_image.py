from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# rotten banana
image = Image.open('test images/rotated_by_15_Screen Shot 2018-06-12 at 8.47.51 PM.png')

# rotten banana
# image = Image.open('test images/1605817491-GettyImages-1131356455.jpg')

# fresh banana
# image = Image.open('test images/rotated_by_15_Screen Shot 2018-06-12 at 9.38.10 PM.png')

# rotten apple (incorrectly classified)
# image = Image.open('test images/rotten_apple__by_prussiaart_db1lljj-pre.png')

# rotten apple
# image = Image.open('test images/apple.png')

# fresh apple
# image = Image.open('test images/pngimg.com - apple_PNG12458.png')


# Image preprocessing
image = image.resize((224, 224))
image = image.convert("RGB")

# Convert the image to a numpy array
numpy_image = np.array(image)

# Normalize the pixel values to [0, 1]
normalized_image = numpy_image / 255.0

# Expand dimensions to create a batch of 1
input_image = np.expand_dims(normalized_image, axis=0)

model = load_model("fruit_freshness_model.h5")

predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

if predicted_class <= 1:
    print('Prediction: FRESH, with predicted class ' + str(predicted_class))
else:
    print('Prediction: STALE, with predicted class ' + str(predicted_class))
