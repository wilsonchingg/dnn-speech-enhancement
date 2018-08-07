import os
import sys
abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/../../train/mfcc_sequence")
sys.path.append(abs_dir + "/../../train/evaluation")
sys.path.append(abs_dir + "/..")
import numpy as np
from model import get_bidirectional_model, set_frame_size
from postprocess_util import smoothing
from eval_model import get_feedforward, get_lstm

set_frame_size(11)
sys_model = get_bidirectional_model()
sys_model.load_weights(abs_dir + "/../model_output/b_bidirectional_network_2_best.hdf5")
lstm_model = get_lstm()
lstm_model.load_weights(abs_dir + "/../model_output/lstm_best.hdf5")
dense_model = get_feedforward()
dense_model.load_weights(abs_dir + "/../model_output/dense_best.hdf5")

models = {}
models['lstm'] = lstm_model
models['dense'] = dense_model
models['sys_model'] = sys_model

def predict(features, model_name = 'sys_model'):
	return smoothing(models[model_name].predict(features))
