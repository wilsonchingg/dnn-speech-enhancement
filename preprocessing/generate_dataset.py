import os
import sys
import json
import numpy as np

from pprint import pprint
from mfcc_util import get_mfcc_batch
from mfcc_source import mfcc

output_path = os.path.join(os.path.dirname(__file__), '../out')

try:
    with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
        conf = json.load(f)
        winlen = conf["winlen"]
        sampling_rate = conf["sampling_rate"]
        winstep = conf["winstep"]
        epoch_dev = conf["generator"]["dev_dataset_epochs"]
        epoch_eval = conf["generator"]["eval_dataset_epochs"]
        batch_dev = conf["generator"]["dev_dataset_batches"]
        batch_eval = conf["generator"]["eval_dataset_batches"]
        nfft = int(sampling_rate * winlen)
        min_len = conf["generator"]["min_sample_len"] * sampling_rate  # 5 seconds
except:
    sys.exit("preprocessing/generate_dataset.py: Unable to read the config file")

def generate_batch(epoch, _set, filename, n_filt = 30, winfunc = np.hamming):
    def get_audio():
        bg_batch, hm_batch = get_mfcc_batch(_set = _set)
        if len(bg_batch) < min_len:
            return get_audio()
        return bg_batch, hm_batch
    input_param = []
    output_param = []
    for j in range(0, epoch):
        print(filename + ': '+ str(j + 1) + '/' + str(epoch))
        bg_batch, hm_batch = get_audio()
        bg_mfcc, bg_mfcc_energy = mfcc(bg_batch[0:min_len], winlen = winlen, winstep = winstep, numcep = n_filt, nfft = nfft,
        nfilt=n_filt, preemph = 0, ceplifter = 0, appendEnergy=False, winfunc = winfunc)
        hm_mfcc, hm_mfcc_energy = mfcc(hm_batch[0:min_len], winlen = winlen, winstep = winstep, numcep = n_filt, nfft = nfft,
        nfilt=n_filt, preemph = 0,  ceplifter = 0, appendEnergy=False, winfunc = winfunc)
        input_param.extend(bg_mfcc)
        output_param.extend(np.clip((hm_mfcc_energy/bg_mfcc_energy), 0, 1).tolist())
    np.save(os.path.join(output_path,  'in_' + filename), np.array(input_param))
    np.save(os.path.join(output_path,  'out_' + filename), np.array(output_param))

for i in range(0, batch_dev):
    generate_batch(epoch_dev, 'dev', 'dev_' + str(i))
for i in range(0, batch_eval):
    generate_batch(epoch_eval, 'eval', 'eval_' + str(i))
