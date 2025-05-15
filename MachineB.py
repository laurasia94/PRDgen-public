# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:36:49 2020

@author: Shizuoka
"""
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

size = (125, 125, 6)
batch_size = int(input('batch_size : '))
epochs = int(input('epochs : '))
num_classes = 2

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=size))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))              

    model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes*8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes*4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

model = create_model()

total_data = np.load("./Projection_data_211109.npz")
X, Y = total_data['arr_0'], total_data['arr_1']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)

model = create_model()
ES = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
MCP = ModelCheckpoint('./Final_machine_211109_3.h5', monitor='val_loss', mode='min', save_best_only=True)
results = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), callbacks=[ES, MCP])