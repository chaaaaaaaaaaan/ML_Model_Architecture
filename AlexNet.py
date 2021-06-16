# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:03:45 2021

@author: skukm
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

#AlexNet Model
model = Sequential()
model.add(Conv2D(96, kernel_size=(11,11), strides=(4,4), input_shape=(227,227,3), activation='relu', name='Conv2D_1'))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), name='MaxPool2D_1'))
model.add(BatchNormalization(name='BatchNormalization_1'))

model.add(Conv2D(256, kernel_size=(5,5), padding='same', activation='relu', name='Conv2D_2'))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), name='MaxPool2D_2'))
model.add(BatchNormalization(name='BatchNormalization_2'))

model.add(Conv2D(384, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_3'))
model.add(Conv2D(384, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_4'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_5'))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), name='MaxPool2D_3'))
model.add(Flatten(name='Flatten'))

model.add(Dense(4096, activation='relu', name='Dense_1'))
model.add(Dense(4096, activation='relu', name='Dense_2'))
model.add(Dense(1000, activation='softmax', name='Dense_3'))
model.summary()

plot_model(model, to_file='AlexNet.png', show_shapes=(True), show_layer_names=(True))
