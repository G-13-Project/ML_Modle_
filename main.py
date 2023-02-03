import io
from fastapi import FastAPI,File,UploadFile   
from fastapi.responses import StreamingResponse
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from fastapi.middleware.cors import CORSMiddleware
import keras_preprocessing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the save model
model = tf.keras.models.load_model('herbs.h5')


@app.post("/")
async def upload_file(file:UploadFile):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB')
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # expand tha array with another demention
    test_image = np.expand_dims(image_array, axis=0)

    result = model.predict(test_image)

    if result[0][0] == 1:
        return "The herb is Papaya"
    elif result[0][1] == 1:
        return "The herb is Alovara"
    elif result[0][2] == 1:
        return "The herb is Bilin"
    elif result[0][3] == 1:
        return "The herb is Jack"
    elif result[0][4] == 1:
        return "The herb is Lemon"
    else:
        return "Cant find !"
