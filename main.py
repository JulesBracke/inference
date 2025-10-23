from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANIMALS = ['Cat', 'Dog', 'Panda'] # Animal names here, these represent the labels of the images that we trained our model on.

# It would've been better to use an environment variable to fix this line actually...
model_path = os.path.join("model", "model.keras")
model = tf.keras.models.load_model(model_path)

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file) # Read the bytes and process as an image
    resized_image = original_image.resize((64, 64)) # Resize
    images_to_predict = np.expand_dims(np.array(resized_image), axis=0) # Our AI Model wanted a list of images, but we only have one, so we expand it's dimension
    predictions = model.predict(images_to_predict) # The result will be a list with predictions in the one-hot encoded format: [ [0 1 0] ]
    prediction_probabilities = predictions
    classifications = prediction_probabilities.argmax(axis=1) # We try to fetch the index of the highest value in this list [ [1] ]

    return ANIMALS[classifications.tolist()[0]] # Fetch the first item in our classifications array, format it as a list first, result will be e.g.: "Dog"