
import sys
import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Input, LSTM, SimpleRNN
import keras.backend as K

def get_multilayer_model(arr, optimizer = 'adam'):
    model = Sequential()
    for i in range(0, len(arr)):
        if i == 0:
            model.add(GRU(int(arr[i]), return_sequences=True, kernel_initializer='random_uniform',
             input_shape = (1, 22,)))
        elif i + 1 == len(arr):
            model.add(GRU(int(arr[i]), return_sequences=False, kernel_initializer='random_uniform'))
        else:
            model.add(GRU(int(arr[i]), return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dense(22, activation='hard_sigmoid', kernel_initializer='random_uniform'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
