import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf


from keras.applications.vgg16 import VGG16
model = VGG16()


from keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())


# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

from keras.preprocessing.image import load_img
# load an image from file
image = load_img('-27.jpg', target_size=(224, 224))

from keras.preprocessing.image import img_to_array
# convert the image pixels to a numpy array
image = img_to_array(image)


# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

from keras.applications.vgg16 import preprocess_input
# prepare the image for the VGG model
image = preprocess_input(image)


# predict the probability across all output classes
yhat = model.predict(image)


from keras.applications.vgg16 import decode_predictions
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))