import sys
import os
abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/..")
from model import get_multilayer_model
import numpy as np
from global_util import get_callbacks

epochs = 250

def readDataSet():
	x = np.load(abs_dir + "/../dataset/" + "input_v1_dev_mfcc_0.npy")
	y = np.load(abs_dir + "/../dataset/" + "output_v1_dev_mfcc_0.npy")
	return x, y
def readEvalDataSet():
	x = np.load(abs_dir + "/../dataset/" + "input_v1_eval_mfcc_0.npy")
	y = np.load(abs_dir + "/../dataset/" + "output_v1_eval_mfcc_0.npy")
	return x, y

x, y = readDataSet()
x = np.expand_dims(x, axis=1)
x_v, y_v = readEvalDataSet()
x_v = np.expand_dims(x_v, axis=1)

models = [[np.repeat(44, 2)],[np.repeat(44, 3)],[np.repeat(44, 4)],
[np.repeat(44, 5)],[np.repeat(44, 6)],[np.repeat(44, 7)],[np.repeat(44, 8)],
[np.repeat(44, 9)],[np.repeat(44, 10)], [[22]], [[44]], [[66]], [[88]], [[110]], [[132]],
[np.repeat(44, 2), 'sgd'], [np.repeat(44, 2), 'rmsprop']]

for i in models:
	if len(i) == 3:
		filename =  i[2]
		model = get_multilayer_model(i[0], optimizer=i[1])
	else:
		filename =  'gru_unit_' + str(i[0][0]) + '_layer_' + str(len(i[0]))
		model = get_multilayer_model(i[0])
	print(model.summary())
	model.fit(x=x[0:200000], y=y[0:200000], epochs=epochs, batch_size=200, verbose=2,
	 validation_data=(x_v[0:40000], y_v[0:40000]), shuffle=True, callbacks=get_callbacks(filename))
