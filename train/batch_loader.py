import sys, os, json, numpy as np
sys.path.append("../../preprocessing/mfcc")
from mfcc_sequence_util import to_sequence_with_stride, to_sequence

CONF_PATH = os.path.join(os.path.dirname(__file__), '../config.json')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../out')

with open(CONF_PATH) as f:
	CONF = json.load(f)
	BATCH_SIZE = CONF["train"]["batch_size"]
	SINGLE_BATCH_STEP = CONF["train"]["single_batch_step"]

def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def add_sequence(x, y, nfft, left_pad=3, right_pad=3, shuffle=True, stride_width=1):
	x_sequence = []
	if len(x) % nfft != 0:
		raise ValueError('Invalid dataset')
	for i in range(0, len(x)//nfft):
		if stride_width == 1:
			tmp = to_sequence(x[i * nfft: i * nfft + nfft], left_pad=left_pad, right_pad=right_pad)
		else:
			tmp = to_sequence_with_stride(x[i * nfft: i * nfft + nfft], left_pad=left_pad,
			right_pad=right_pad, stride_width=stride_width)
		x_sequence.extend(tmp)
	if shuffle:
		return unison_shuffled_copies(np.array(x_sequence), y)
	return np.array(x_sequence), y


def training_generator(number_of_files, nfft=499, left_pad=3, right_pad=3, stride_width=2):
	idx = -1
	tmp = 0
	file_idx = 0
	x = np.load(os.path.join(DATASET_PATH, "input_dev_mfcc_" + str(file_idx) + ".npy"))
	y = np.load(os.path.join(DATASET_PATH, "output_dev_mfcc_" + str(file_idx) + ".npy"))
	x, y = add_sequence(x, y, nfft, left_pad=left_pad, right_pad=right_pad,
		shuffle=True, stride_width=stride_width)
	while True:
		idx += 1
		if idx >= SINGLE_BATCH_STEP: # batch size
			file_idx = (file_idx + 1) % (number_of_files + 1)
			x = np.load(os.path.join(DATASET_PATH, "input_dev_mfcc_" + str(file_idx) + ".npy"))
			y = np.load(os.path.join(DATASET_PATH, "output_dev_mfcc_" + str(file_idx) + ".npy"))
			x, y = add_sequence(x, y, nfft, left_pad=left_pad, right_pad=right_pad,
			shuffle=True, stride_width=stride_width)

			print('File index: ' + str(file_idx) + ', ' + str(tmp))
			idx = 0
		tmp += 1
		yield x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE], y[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]

def get_dataset(nfft, left_pad=3, right_pad=3, stride_width=1):
	x = np.load(os.path.join(DATASET_PATH, "input_dev_mfcc_0.npy"))
	y = np.load(os.path.join(DATASET_PATH, "output_dev_mfcc_0.npy"))
	x, y = add_sequence(x, y, nfft, left_pad=left_pad, right_pad=right_pad,
	 shuffle=False, stride_width=stride_width)
	x_v = np.load(os.path.join(DATASET_PATH, "input_eval_mfcc_0.npy"))
	y_v = np.load(os.path.join(DATASET_PATH, "output_eval_mfcc_0.npy"))
	x_v, y_v = add_sequence(x_v, y_v, nfft, left_pad=left_pad,
	 right_pad=right_pad, shuffle=False, stride_width=stride_width)
	return x[0:600000], y[0:600000], x_v[0:200000], y_v[0:200000]
