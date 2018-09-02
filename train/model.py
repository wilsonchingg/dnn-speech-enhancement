
import json, os
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Input, Dropout, Bidirectional
import keras.backend as K
from keras import optimizers

with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
	CONF = json.load(f)
	INPUT_SIZE = CONF["train"]["input_size"]
	FRAME_SIZE = CONF["train"]["frame_size"]

def set_frame_size(size):
	global FRAME_SIZE
	FRAME_SIZE = size
def set_input_size(size):
	global INPUT_SIZE
	INPUT_SIZE = size

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)

def get_bidirectional_model():
	model = Sequential()
	_input = Input(shape=(FRAME_SIZE, INPUT_SIZE))
	gru_1 = Bidirectional(GRU(60, kernel_initializer='random_uniform', return_sequences=True))(_input)
	dropout_gru_1 = Dropout(0.2)(gru_1)
	gru_2 = Bidirectional(GRU(60, kernel_initializer='random_uniform',
     return_sequences=True))(dropout_gru_1)
	dropout_gru_2 = Dropout(0.2)(gru_2)
	gru_3 = GRU(120, kernel_initializer='random_uniform', return_sequences=False)(dropout_gru_2)
	dense_1 = Dense(INPUT_SIZE, kernel_initializer='random_uniform', activation='hard_sigmoid')(gru_3)
	adam = optimizers.Adam(lr=0.001)
	model = Model(inputs=_input, outputs=dense_1)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=[binary_accuracy, 'mae'])
	return model
