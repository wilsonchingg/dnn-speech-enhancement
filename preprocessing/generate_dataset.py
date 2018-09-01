import os, json, numpy as np
from randomizer import get_noisy_speech
from python_speech_features import mfcc
from mfcc_utils import compute_gain

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../out')

with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
	CONF = json.load(f)
	WINLEN = CONF["winlen"]
	SAMPLING_RATE = CONF["sampling_rate"]
	NFILT = CONF["n_filt"]
	WINSTEP = CONF["winstep"]
	EPOCH_DEV = CONF["generator"]["dev_dataset_epochs"]
	EPOCH_EVAL = CONF["generator"]["eval_dataset_epochs"]
	BATCH_DEV = CONF["generator"]["dev_dataset_batches"]
	BATCH_EVAL = CONF["generator"]["eval_dataset_batches"]
	NFFT = int(SAMPLING_RATE * WINLEN)
	MIN_LEN = CONF["generator"]["min_sample_len"] * SAMPLING_RATE  # 5 seconds


def generate_batch(epoch, _set, filename, n_filt=30, winfunc=np.hamming):
	def get_audio():
		bg_batch, hm_batch = get_noisy_speech(_set=_set)
		if len(bg_batch) < MIN_LEN:
			return get_audio()
		return bg_batch, hm_batch
	input_param = []
	output_param = []
	for j in range(0, epoch):
		print(filename + ': '+ str(j + 1) + '/' + str(epoch))
		bg_batch, hm_batch = get_audio()
		bg_mfcc, bg_mfcc_energy = mfcc(bg_batch[0:MIN_LEN], winlen=WINLEN,
                                        winstep=WINSTEP, numcep=n_filt,
                                        nfft=NFFT, nfilt=n_filt, preemph=0,
                                        ceplifter=0, appendEnergy=False, cb=compute_gain,
                                        winfunc=winfunc)
		_, hm_mfcc_energy = mfcc(hm_batch[0:MIN_LEN], winlen=WINLEN,
                                 winstep=WINSTEP, numcep=n_filt,
                                 nfft=NFFT, nfilt=n_filt, preemph=0,cb=compute_gain,
                                 ceplifter=0, appendEnergy=False, winfunc=winfunc)
		input_param.extend(bg_mfcc)
		output_param.extend(np.clip((hm_mfcc_energy/bg_mfcc_energy), 0, 1).tolist())
	np.save(os.path.join(OUTPUT_PATH, 'in_' + filename), np.array(input_param))
	np.save(os.path.join(OUTPUT_PATH, 'out_' + filename), np.array(output_param))

for i in range(0, BATCH_DEV):
	generate_batch(EPOCH_DEV, 'dev', 'dev_' + str(i), NFILT)
for i in range(0, BATCH_EVAL):
	generate_batch(EPOCH_EVAL, 'eval', 'eval_' + str(i), NFILT)
