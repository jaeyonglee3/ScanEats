from cProfile import label
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from numpy import expand_dims
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from PIL import ImageFile, Image
import numpy as np
import cv2

def get_prediction(file_path, model_path="cnn"):
    model_path = os.path.join(model_path, "fruit_freshness_model.h5")
    model = load_model(model_path)

    image = Image.open(file_path)
    image = image.resize((224, 224))
    image = image.convert("RGB")

    # Convert the image to a numpy array
    numpy_image = np.array(image)

    # Normalize the pixel values to [0, 1]
    normalized_image = numpy_image / 255.0

    # Expand dimensions to create a batch of 1
    input_image = np.expand_dims(normalized_image, axis=0)
    predictions = model.predict(input_image)
    
    print(predictions)
    predicted_class = np.argmax(predictions)
    print(predicted_class)

    if predicted_class <= 1:
        return 'fresh'
    else:
        return 'stale'
    
# def get_prediction(file_path, food_name, model_path="cnn"):
#     model_path = os.path.join(model_path, "fruit_freshness_model.h5")
#     model = load_model(model_path)

#     original_image = Image.open(file_path)
#     original_image = original_image.convert('RGB')
#     original_image = original_image.resize((224, 224))
#     numpy_image = image.img_to_array(original_image)
#     image_batch = expand_dims(numpy_image, axis=0)

#     processed_image = preprocess_input(image_batch, mode='tf')
#     preds = model.predict(processed_image)
    
#     print(preds)
#     predicted_class = np.argmax(preds)
#     print(predicted_class)

#     processed_image = (processed_image[0] * 255).astype(np.uint8)

#     # Create a PIL Image object from the numpy array
#     processed_image = Image.fromarray(processed_image)

#     # Display the image (opens in the default image viewer)
#     processed_image.show()

#     if food_name == "apple" or food_name == "banana":
#         if predicted_class == 1:
#             return "stale"
#         else: 
#             return "fresh"