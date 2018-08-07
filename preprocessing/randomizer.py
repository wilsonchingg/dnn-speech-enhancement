import sys
import wave
import os, random
import numpy as np
import json
import audio_utils

from random import randint

from scipy.io.wavfile import read
import scipy.fftpack as fft


conf_path = os.path.join(os.path.dirname(__file__), '../config.json')
dataset_path = os.path.join(os.path.dirname(__file__), '../datasets')
output_path = os.path.join(os.path.dirname(__file__), '../out')

try:
	with open(conf_path) as f:
		conf = json.load(f)
		noise_mix = conf["noise_mix"] # Number of noise files to be mixed to a speech audio
except:
    sys.exit("preprocessing/randomizer.py: Unable to read the config file")

def random_file(dir):
	return random.choice(os.listdir(dir))

def listdirs(dir):
	return [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

def random_speeches(_set = 'dev'):
	_path = os.path.join(dataset_path, _set + '/speeches')
	return os.path.join(_path, random_file(_path))

def random_noises(_set):
	# Choose a random noise directory
	_path = os.path.join(dataset_path, _set + '/noises')
	_path = os.path.join(_path, random.choice(listdirs(_path)))
	return os.path.join(_path, random_file(_path))

def get_noisy_speech(_set = 'dev', cat_dir=None):
	noise_files = []
	r_file = random_speeches(_set = _set)
	for _ in range(0, noise_mix):
		noise_files.append(random_noises(_set))
	return audio_utils.overlay(arr = noise_files, hm=r_file)
