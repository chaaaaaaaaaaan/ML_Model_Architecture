# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:46:35 2021

@author: skukm
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

#VGG16 Model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(224,224,3), activation='relu', kernel_regularizer=l2(0.01), name='Conv2D_1'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01), name='Conv2D_2'))
model.add(MaxPool2D(pool_size=(2,2), name='MaxPool2D_1'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01), name='Conv2D_3'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_4'))
model.add(MaxPool2D(pool_size=(2,2), name='MaxPool2D_2'))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01), name='Conv2D_5'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_6'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_7'))
model.add(MaxPool2D(pool_size=(2,2), name='MaxPool2D_3'))

model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01), name='Conv2D_8'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_9'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_10'))
model.add(MaxPool2D(pool_size=(2,2), name='MaxPool2D_4'))

model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01), name='Conv2D_11'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_12'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_13'))
model.add(MaxPool2D(pool_size=(2,2), name='MaxPool2D_5'))
model.add(Flatten(name='Flatten'))

model.add(Dense(4096, activation='relu', name='Dense_1'))
model.add(Dense(4096, activation='relu', name='Dense_2'))
model.add(Dense(1000, activation='softmax', name='Dense_3'))

model.summary()
plot_model(model, to_file='VGG16.png', show_shapes=(True), show_layer_names=(True))
