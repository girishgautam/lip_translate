import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Activation, MaxPool3D, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)



def initiate_model():
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), input_shape=(75, 30, 70, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Dropout(.5))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Dropout(.5))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

    return model

def predict_video(model, video_frames):
    # Assuming `video_frames` is your input data # Ensure this is correctly implemented
    video_frames_batch = np.expand_dims(video_frames, axis=0)

    # Get predictions
    prediction = model.predict(video_frames_batch)

    # The sequence length should match the batch size of 'prediction'
    sequence_length = [len(video_frames)]  # Replace this with the correct sequence length for your data

    # Decode the predictions
    decoded_prediction = tf.keras.backend.ctc_decode(prediction, sequence_length, greedy=False)[0][0].numpy()
    predicted_text = tf.strings.reduce_join(num_to_char(decoded_prediction)).numpy().decode('utf-8')
    return predicted_text
