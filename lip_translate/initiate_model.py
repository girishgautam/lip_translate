import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Activation, MaxPool3D, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def initiate_model():
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), input_shape=(75, 30, 70, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Dropout(0.3))

    model.add(Conv3D(128, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Dropout(0.4))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Dropout(0.5))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(64, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.3))

    model.add(Bidirectional(LSTM(64, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.4))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

    return model
