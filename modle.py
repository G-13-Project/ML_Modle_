import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

# herb categories
# 1.pepaya, 2.alovara, 3.bilin, 4.jack, 5.lemon
herbs_category = ['alovara', 'bilin', 'jack', 'lemon', 'Pepaya']

# load the save model
model = tf.keras.models.load_model('herbs.h5')

# print(model.summary())

img_path = "bo.jpg"
# E:/F_PROJECTS/DataSet/test/Pepaya/Pepaya170.jpg

test_image = image.load_img(img_path, target_size=(224, 224))

# convert image in to array
test_image = image.img_to_array(test_image)
# print(test_image.shape)

# expand tha array with another demention
test_image = np.expand_dims(test_image, axis=0)
# print(test_image.shape)

# predict the category of the image
result = model.predict(test_image)

print(result)

if result[0][0] == 1:
    print("The herb is Papaya")
elif result[0][1] == 1:
    print("The herb is Alovara")
elif result[0][2] == 1:
    print("The herb is Bilin")
elif result[0][3] == 1:
    print("The herb is Jack")
elif result[0][4] == 1:
    print("The herb is Lemon")
else:
    print("Cant find !")


