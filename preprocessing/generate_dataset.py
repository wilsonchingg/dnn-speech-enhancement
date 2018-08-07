import os
import sys

abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/mfcc")

import numpy as np
from mfcc_util import get_mfcc_batch
from mfcc_source import mfcc
from mfcc_sequence_util import to_sequence

output_dir = abs_dir + "/../train/dataset/"
winlen = 0.02
sampling_rate = 16000
winstep = winlen / 2
nfft = int(sampling_rate * winlen)

epoch_dev = 1500
epoch_eval = 500

min_len = 80000 # 5 seconds
# j = number of audio set data per file
def generate_sample(j, _set, filename, n_filt = 30, winfunc = np.hamming):
    append_clean = False
    def get_audio():
        bg_batch, hm_batch = get_mfcc_batch(_set = _set)
        if len(bg_batch) < min_len:
            return get_audio()
        return bg_batch, hm_batch
    input_param = []
    output_param = []
    for jj in range(0, j):
        print('Epoch ' + str(jj + 1))
        bg_batch, hm_batch = get_audio()
        bg_mfcc, bg_mfcc_energy = mfcc(bg_batch[0:min_len], winlen = winlen, winstep = winstep, numcep = n_filt, nfft = nfft,
        nfilt=n_filt, preemph = 0, ceplifter = 0, appendEnergy=False, winfunc = winfunc)
        hm_mfcc, hm_mfcc_energy = mfcc(hm_batch[0:min_len], winlen = winlen, winstep = winstep, numcep = n_filt, nfft = nfft,
        nfilt=n_filt, preemph = 0,  ceplifter = 0, appendEnergy=False, winfunc = winfunc)
        input_param.extend(bg_mfcc)
        output_param.extend(np.clip((hm_mfcc_energy/bg_mfcc_energy), 0, 1).tolist())
    np.save(output_dir + 'input_' + filename, np.array(input_param))
    np.save(output_dir + 'output_' + filename, np.array(output_param))
if __name__ == "__main__":

    # V1 dataset
    generate_sample(epoch_dev, 'dev', 'v1_dev_mfcc_0', n_filt = 22, winfunc=lambda x:np.ones((x, )))
    generate_sample(epoch_eval, 'eval', 'v1_eval_mfcc_0', n_filt = 22, winfunc=lambda x:np.ones((x, )))

    # V2 dataset
    generate_sample(epoch_dev, 'dev', 'dev_mfcc_0')
    generate_sample(epoch_dev, 'dev', 'dev_mfcc_1')
    generate_sample(epoch_dev, 'dev', 'dev_mfcc_2')
    generate_sample(epoch_dev, 'dev', 'dev_mfcc_3')
    generate_sample(epoch_dev, 'dev', 'dev_mfcc_4')
    generate_sample(epoch_dev, 'dev', 'dev_mfcc_5')
    generate_sample(epoch_eval, 'eval', 'eval_mfcc_0')
