from pydub import AudioSegment
from scipy.io.wavfile import read, write
from math import ceil, fabs
from random import choice
import sys
import os
import json
import numpy as np

conf_path = os.path.join(os.path.dirname(__file__), '../config.json')
sampling_rate = None # Number of noise files to be mixed to a speech audio

try:
	with open(conf_path) as f:
		conf = json.load(f)
		sampling_rate = conf["sampling_rate"]
except:
    sys.exit("preprocessing/audio_utils.py: Unable to read the config file")

snr = None

def set_snr(_snr):
	global snr
	snr = _snr

def generate_audio(data, freq, filename):
	to_write = np.empty(len(data), dtype=np.int16)
	for i in range(0, len(data)):
		to_write[i] = int(data[i] * 32767)
	write(filename, freq, to_write)
def int_to_float(signals):
	if len(signals) == 0:
		return
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
		return np.tile(signals, int(ceil(length/len(signals)) + 1))[0: length]
def scipy_read(filename, length = -1):
	freq, out = read(filename)
	if freq != sampling_rate:
		print("preprocessing/audio_utils.py: Non-compatible file: " + filename + ", frequency: " + str(freq))
		return
	return freq, int_to_float(out[0:length])
# Seg here means number of milliseconds
# arr: list of files to mix
# filename: filename of the export
def overlay(filename="", arr = [], hm=""):
	global snr
	# 10db, 5db, 0db, -5db
	multipler = choice([0.316, 0.56, 1, 1.79])
	if snr != None:
		multipler = snr
	_, hm_file = scipy_read(hm)
	# round down to second
	freq_seg =  (len(hm_file) // 1000) * 1000
	out = hm_file[0:freq_seg]
	out_copy = np.array(out, copy=True)
	normalizer = np.average(np.fabs((out)))
	for i in range(0, len(arr)):
		_, r_file = scipy_read(arr[i], freq_seg)
		resized = normalize(r_file, freq_seg)
		resized = (resized * (multipler * normalizer/((np.average(np.fabs(resized))) * len(arr))))
		out += resized
	if filename != "":
		generate_audio(out, sampling_rate, filename)
	return out.astype(np.float32), out_copy
