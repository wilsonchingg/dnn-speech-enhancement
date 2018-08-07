import sys, os, json, math, random, numpy as np
from scipy.io.wavfile import read

try:
	with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
		conf = json.load(f)
		sampling_rate = conf["sampling_rate"]
except:
    sys.exit("preprocessing/audio_utils.py: Unable to read the config file")

snr = None

def set_snr(_snr):
	global snr
	snr = _snr

def int_to_float(signals):
	if len(signals) == 0:
		return []
	if isinstance(signals[0], int) or isinstance(signals[0], np.int16):
		arr = []
		for i in range(0, len(signals)):
			arr.append(signals[i]/32768.0)
		return np.array(arr).astype(np.float32)
	else:
		return signals

# length here means arbitary signal values
def normalize(signals, length):
	if len(signals) > length:
		return signals[0: length]
	else:
		# Repeat the sound if the length of signals is smaller than required length
		return np.tile(signals, int(math.ceil(length/len(signals)) + 1))[0: length]

def scipy_read(filename, length = -1):
	freq, out = read(filename)
	if freq != sampling_rate:
		sys.exit("preprocessing/audio_utils.py: Non-compatible file: " + filename + ", frequency: " + str(freq))
	return int_to_float(out[0:length])

def overlay(speech_path, noise_paths):
	global snr
	# 10db, 5db, 0db, -5db
	multipler = random.choice([0.316, 0.56, 1, 1.79])
	if snr != None:
		multipler = snr
	speech_signal = scipy_read(speech_path)
	freq_seg =  (len(speech_signal) // 1000) * 1000 # round down to second

	out = speech_signal[0:freq_seg]
	out_copy = np.array(out, copy=True)
	normalizer = np.average(np.fabs((out)))
	for i in range(0, len(noise_paths)):
		noise_signal = scipy_read(noise_paths[i], freq_seg)
		resized = normalize(noise_signal, freq_seg)
		resized = (resized * (multipler * normalizer/((np.average(np.fabs(resized))) * len(noise_paths))))
		out += resized
	return out.astype(np.float32), out_copy
