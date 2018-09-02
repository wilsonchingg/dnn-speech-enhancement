import sys, os, json, math, random, numpy as np
from scipy.io.wavfile import read, write

with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
	SAMPLING_RATE = json.load(f)["sampling_rate"]

SNR = None

def set_snr(_snr):
	global SNR
	SNR = _snr

# Convert float32 to int16 arr for processing
def write_wav(dest, data):
	to_write = np.empty(len(data), dtype=np.int16)
	for idx, val in enumerate(data):
		to_write[idx] = int(val * 32767)
	# TODO, might not work
	write(os.path.join(os.path.dirname(__file__), '../sample_test', dest), SAMPLING_RATE, to_write)

def int_to_float(signals):
	if len(signals) == 0:
		return []
	if isinstance(signals[0], (int, np.int16)):
		arr = []
		for _, val in enumerate(signals):
			arr.append(val/32768.0)
		return np.array(arr).astype(np.float32)
	return signals

# length here means arbitary signal values
def normalize(signals, length):
	if len(signals) > length:
		return signals[0: length]
	# Repeat the sound if the length of signals is smaller than required length
	return np.tile(signals, int(math.ceil(length/len(signals)) + 1))[0: length]

def scipy_read(filename, length=-1):
	freq, out = read(filename)
	if freq != SAMPLING_RATE:
		sys.exit("preprocessing/audio_utils.py: Non-compatible file: " + filename +
		         ", frequency: " + str(freq))
	return int_to_float(out[0:length])

def overlay(speech_path, noise_paths):
	global SNR
	# 10db, 5db, 0db, -5db
	multipler = random.choice([0.316, 0.56, 1, 1.79])
	if SNR is not None:
		multipler = SNR
	speech_signal = scipy_read(speech_path)
	freq_seg = (len(speech_signal) // 1000) * 1000 # round down to second

	out = speech_signal[0:freq_seg]
	out_copy = np.array(out, copy=True)
	normalizer = np.average(np.fabs((out)))
	for _, val in enumerate(noise_paths):
		noise_signal = scipy_read(val, freq_seg)
		resized = normalize(noise_signal, freq_seg)
		resized = (resized * (multipler * normalizer/((np.average(np.fabs(resized))) *
		                                              len(noise_paths))))
		out += resized
	return out.astype(np.float32), out_copy
