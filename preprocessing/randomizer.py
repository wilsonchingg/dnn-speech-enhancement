import sys, os, random, json, audio_utils, scipy.fftpack as fft
from scipy.io.wavfile import read

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
	name = random.choice(os.listdir(dir))
	if name.endswith('.wav'):
		return name
	return random_file(dir)

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

def get_noisy_speech(_set = 'dev'):
	noise_files = []
	speech_file = random_speeches(_set = _set)
	for _ in range(0, noise_mix):
		noise_files.append(random_noises(_set))
	return audio_utils.overlay(speech_file, noise_files)
