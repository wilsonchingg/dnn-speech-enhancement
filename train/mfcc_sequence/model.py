
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Input, Dropout, Bidirectional
import keras.backend as K
from keras import optimizers
from keras.constraints import min_max_norm
from keras.layers.merge import add

### Model setting
input_size = 30
num_of_frames = 7

def set_frame_size(size):
    global num_of_frames
    num_of_frames = size
def set_input_size(size):
    global input_size
    input_size = size

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)

# dropout on input layer
def get_multilayer_dropout_model(arr, layer_type = 'gru', dropout_rate = 0.0):
    print('Dropout rate: ' + str(dropout_rate))
    model = Sequential()
    for i in range(0, len(arr)):
        if i == 0:
            model.add(GRU(arr[i], return_sequences=True, kernel_initializer='random_uniform',
             input_shape = (num_of_frames, input_size,)))
            model.add(Dropout(dropout_rate))
        elif i + 1 == len(arr):
            model.add(GRU(arr[i], return_sequences=False, kernel_initializer='random_uniform'))
        else:
            model.add(GRU(arr[i], return_sequences=True, kernel_initializer='random_uniform'))
            model.add(Dropout(dropout_rate))
    model.add(Dense(input_size, activation='hard_sigmoid', kernel_initializer='random_uniform'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam,  metrics=[binary_accuracy, 'mean_squared_error'])
    return model

def get_residual_model():
    model = Sequential()
    input = Input(shape=(num_of_frames, input_size))
    gru_1 = GRU(120, kernel_initializer='random_uniform', return_sequences=True, dropout=0)(input)
    dropout_gru_1 = Dropout(0.2)(gru_1)
    gru_2 = GRU(120, kernel_initializer='random_uniform', return_sequences=True, dropout=0)(dropout_gru_1)
    dropout_gru_2 = Dropout(0.2)(gru_2)
    add_1 = add([dropout_gru_1, dropout_gru_2])
    gru_3 = GRU(120, kernel_initializer='random_uniform', return_sequences=False)(add_1)
    dense_1 = Dense(input_size, kernel_initializer='random_uniform', activation='hard_sigmoid')(gru_3)
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=input, outputs=dense_1)
    model.compile(loss='mean_squared_error', optimizer=adam,  metrics=[binary_accuracy, 'mae'])
    return model

def get_bidirectional_model():
    model = Sequential()
    input = Input(shape=(num_of_frames, input_size))
    gru_1 = Bidirectional(GRU(60, kernel_initializer='random_uniform', return_sequences=True))(input)
    dropout_gru_1 = Dropout(0.2)(gru_1)
    gru_2 = Bidirectional(GRU(60, kernel_initializer='random_uniform', return_sequences=True))(dropout_gru_1)
    dropout_gru_2 = Dropout(0.2)(gru_2)
    gru_3 = GRU(120, kernel_initializer='random_uniform', return_sequences=False)(dropout_gru_2)
    dense_1 = Dense(input_size, kernel_initializer='random_uniform', activation='hard_sigmoid')(gru_3)
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=input, outputs=dense_1)
    model.compile(loss='mean_squared_error', optimizer=adam,  metrics=[binary_accuracy, 'mae'])
    return model
