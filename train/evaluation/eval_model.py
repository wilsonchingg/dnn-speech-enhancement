
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout
from keras import optimizers
import keras.backend as K

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


def get_lstm():
    model = Sequential()
    input = Input(shape=(11, 30,))
    lstm_1 = LSTM(120, kernel_initializer='random_uniform', return_sequences=True)(input)
    dropout_lstm_1 = Dropout(0.2)(lstm_1)
    lstm_2 = LSTM(120, kernel_initializer='random_uniform', return_sequences=True)(dropout_lstm_1)
    dropout_lstm_2 = Dropout(0.2)(lstm_2)
    lstm_3 = LSTM(120, kernel_initializer='random_uniform', return_sequences=False)(dropout_lstm_2)
    dense_1 = Dense(30, kernel_initializer='random_uniform', activation='hard_sigmoid')(lstm_3)
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=input, outputs=dense_1)
    model.compile(loss='mean_squared_error', optimizer=adam,  metrics=[binary_accuracy])
    return model

def get_feedforward():
    model = Sequential()
    input = Input(shape=(90,))
    dense_1 = Dense(360, kernel_initializer='random_uniform', activation='relu')(input)
    dropout_dense_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(360, kernel_initializer='random_uniform', activation='relu')(dropout_dense_1)
    dropout_dense_2 = Dropout(0.2)(dense_2)
    dense_3 = Dense(360, kernel_initializer='random_uniform')(dropout_dense_2)
    dense_1 = Dense(30, kernel_initializer='random_uniform', activation='hard_sigmoid')(dense_3)
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=input, outputs=dense_1)
    model.compile(loss='mean_squared_error', optimizer=adam,  metrics=[binary_accuracy])
    return model
