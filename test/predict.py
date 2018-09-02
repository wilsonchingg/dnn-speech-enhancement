import os, sys, json
from postprocess_util import smoothing
sys.path.append(os.path.join(os.path.dirname(__file__), "../train"))
from model import get_bidirectional_model, set_frame_size

with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
	CONF = json.load(f)
	FRAME_SIZE = CONF["train"]["frame_size"]

set_frame_size(FRAME_SIZE)
sys_model = get_bidirectional_model()
sys_model.load_weights(os.path.join(os.path.dirname(__file__),
 'model_output/b_bidirectional_network_2_best.hdf5'))
models = {}
models['sys_model'] = sys_model

def predict(features, model_name='sys_model'):
	return smoothing(models[model_name].predict(features))
