import os
import sys
abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/..")
sys.path.append(abs_dir + "/../../preprocessing/mfcc")
sys.path.append(abs_dir + "/../../preprocessing")
sys.path.append(abs_dir + "/../../train/mfcc_sequence")

from mfcc_util import generate_sample, get_estimate, write_wav, get_mfcc_batch
from mfcc_sequence_util import to_sequence_with_stride
from postprocess_util import evaluate
from audio_utils import set_snr
from predict import predict
import subprocess
import setting
import numpy as np

model_dir = abs_dir + "/../model_output/v2/"
models = []
filenames = ["sys_model.wav", "logmmse.wav"]

evaluation_size = 100

multipliers = [[0.316, 10], [0.56, 5], [1, 0], [1.79, -5]]


def run_logmmse():
    args = ['python2.7', 'logmmse.py', '../../sample_test/mixed.wav', '../../sample_test/logmmse.wav']
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    _, _ = pipe.communicate()

set_snr(0.56) # 0db

for j in setting.sample_sub_dir:
    for i in range(0, evaluation_size):
        def get_noise_type():
            return get_mfcc_batch(cat_dir = j)
        hm, bg, mfcc_feature = generate_sample(_n_filt=30, _winlen=0.02, _winstep=0.01, _winfunc = np.hamming,
         _generator = get_noise_type)
        run_logmmse()
        # Run GRU prediction
        sequence_feature = np.array(to_sequence_with_stride(mfcc_feature, left_pad = 5, right_pad = 5))
        prediction = predict(sequence_feature, filenames[0][0:-4])
        estimate =  get_estimate(bg, prediction, _winlen=0.02, _winstep=0.01, _winfunc = np.hamming)
        write_wav(filenames[0], estimate.astype(np.float32))
        evaluate(estimated_sources=filenames)
    os.rename('pesq_results.txt', j + '.txt')
