import sys
import os, random
import wave
from random import randint

import numpy as np
from scipy.io.wavfile import read
import scipy.fftpack as fft

from audio_utils import Audio_Utils
import setting

setting.init()
dir_path = os.path.dirname(os.path.realpath(__file__))
# if the frequency of an audio is more than 16k, print an error
freq_constraint = 16000
sample_dir = dir_path + '/../sample'
output_dir = dir_path + '/../out'
epochs = 1
mixer = 2
audio_addition = Audio_Utils()

def random_file(dir):
	return random.choice(os.listdir(dir))
def get_random_human(_set = 'dev'):
	if _set == 'dev':
		file_dir = random_file(sample_dir + "/" + setting.voice_dev_dir)
		return sample_dir + "/" + setting.voice_dev_dir + "/" + file_dir
	else:
		file_dir = random_file(sample_dir + "/" + setting.voice_dir)
		print(file_dir)
		return sample_dir + "/" + setting.voice_dir + "/" + file_dir
def get_random_bg():
	rand_dir = ""
	while rand_dir not in setting.sample_sub_dir:
		rand_dir = random_file(sample_dir)
	file_dir = random_file(sample_dir + "/" + rand_dir)
	return sample_dir + "/" + rand_dir + "/" + file_dir

def get_bg_by_category(cat_dir):
	if cat_dir == None:
		return get_random_bg()
	else:
		file_dir = random_file(sample_dir + "/" + cat_dir)
		return sample_dir + "/" + cat_dir + "/" + file_dir

def get_mfcc_batch(_set = 'dev', cat_dir = None):
	return get_batch(applyFFT = False, _set = _set, cat_dir = cat_dir)

# If seg -1, the seg will be the entire human file
def get_batch(applyFFT= True, _set = 'dev', cat_dir = None):
	human, mixed = mix( _set = _set, cat_dir = cat_dir)
	if applyFFT:
		return fft.fft(mixed), fft.fft(human)
	else:
		return np.array(mixed), np.array(human)

def batch_to_fft(batch):
	arr = []
	for i in range(0, len(batch)):
		arr.append(fft.fft(batch[i]))
	return np.array(arr)

# Mix human and background noise extracted from random filters
# If seg -1, the seg will be the entire human file
def mix(_set = 'dev', cat_dir=None):
	file_dir_list = []
	r_file = get_random_human(_set = _set)
	print(r_file)
	for _ in range(0, mixer):
		file_dir = get_bg_by_category(cat_dir)
		print(file_dir)
		file_dir_list.append(file_dir)
	# _unmixed is the different from unmix in which the amplitude had been normalized
	mixed, _unmixed = audio_addition.overlay(arr = file_dir_list, hm=r_file)
	return _unmixed, mixed
