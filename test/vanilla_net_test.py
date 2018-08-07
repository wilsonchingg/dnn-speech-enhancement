import os
import sys
import numpy as np
abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/../train/mfcc")
sys.path.append(abs_dir + "/../postprocessing")
sys.path.append(abs_dir + "/../preprocessing/mfcc")

from mfcc_util import generate_sample, get_estimate, write_wav
from postprocess_util import evaluate, smoothing
from audio_utils import set_snr
from model import get_multilayer_model

evalution_size = 500
multipliers = [[0.316, 10], [0.56, 5], [1, 0], [1.79, -5]]

_model = get_multilayer_model([44, 44])
_model.load_weights(abs_dir + "/model_output/gru_unit_44_layer_2_best.hdf5")

for j in multipliers:
    set_snr(j[0])
    for _ in range(0, evalution_size):
        hm, bg, mfcc_feature = generate_sample(_n_filt = 22, _winlen=0.02, _winstep=0.010)
        predictions = smoothing(_model.predict(np.expand_dims(mfcc_feature, axis=1)))
        estimate =  get_estimate(bg, predictions)
        write_wav('estimation.wav', estimate.astype(np.float32))
        evaluate(estimated_sources=['estimation.wav'])
    os.rename('pesq_results.txt', str(j[1]) + 'db.txt')
