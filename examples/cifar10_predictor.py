'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
# epochs = 100
epochs = 25  # 75 % accuracy good enough

data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model_path = os.path.join(save_dir, model_name)
try:
  model = load_model(model_path)
except:
  print("run cifar10_cnn.py first to generate model")

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import random


predict_data = np.array([])
show_shape = False
if show_shape:
  ind = random.randint(0, len(x_test)-20)
  for i in range(ind, ind+5):
    data = x_test[i]
    # print(data.shape)
    plt.imshow(data)
    plt.show()


# Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

results = model.predict(x_test[0:5],
                       batch_size=None,
                       verbose=0,
                       steps=None)

print("Predictions: ")
for idx, pred in enumerate(results):
  print(pred, y_test[idx])
