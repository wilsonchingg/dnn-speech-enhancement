import os
import sys

abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/../../preprocessing/mfcc")
sys.path.append(abs_dir + "/../../preprocessing")
sys.path.append(abs_dir + "/..")
sys.path.append(abs_dir + "/../../train/mfcc_sequence")

from mfcc_util import generate_sample, get_estimate, write_wav, get_delta
from mfcc_sequence_util import to_sequence_with_stride
from postprocess_util import evaluate
from log_mmse import run_logmmse
from audio_utils import set_snr
from predict import predict
import subprocess
import numpy as np

model_dir = abs_dir + "/../model_output/v2/"
models = []
filenames = ["sys_model.wav", "logmmse.wav", "dense.wav", "lstm.wav"]

evaluation_size = 200

multipliers = [[0.316, 10], [0.56, 5], [1, 0], [1.79, -5]]

for j in multipliers:
    set_snr(j[0])
    for i in range(0, evaluation_size):
        hm, bg, mfcc_feature = generate_sample(_n_filt=30, _winlen=0.02, _winstep=0.01, _winfunc = np.hamming)
        run_logmmse('../../sample_test/mixed.wav', '../../sample_test/logmmse.wav')

        sequence_feature = np.array(to_sequence_with_stride(mfcc_feature, left_pad = 5, right_pad = 5))
        delta_feature = get_delta(mfcc_feature)
        # sys model
        prediction = predict(sequence_feature, filenames[0][0:-4])
        estimate =  get_estimate(bg, prediction, _winlen=0.02, _winstep=0.01, _winfunc = np.hamming)
        write_wav(filenames[0], estimate.astype(np.float32))

        # lstm
        prediction = predict(sequence_feature, filenames[3][0:-4])
        estimate =  get_estimate(bg, prediction, _winlen=0.02, _winstep=0.01, _winfunc = np.hamming)
        write_wav(filenames[3], estimate.astype(np.float32))

        # dense
        prediction = predict(delta_feature, filenames[2][0:-4])
        print(delta_feature.shape)
        estimate =  get_estimate(bg, prediction, _winlen=0.02, _winstep=0.01, _winfunc = np.hamming)
        write_wav(filenames[2], estimate.astype(np.float32))

        evaluate(estimated_sources=filenames)
    os.rename('pesq_results.txt', str(j[1]) + 'db.txt')
