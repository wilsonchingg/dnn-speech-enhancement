import os, random, json, audio_utils

CONF_PATH = os.path.join(os.path.dirname(__file__), '../config.json')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../datasets')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../out')

with open(CONF_PATH) as f:
	NOISE_MIX = json.load(f)["noise_mix"] # Number of noise files to be mixed to a speech audio

def random_file(_dir):
	name = random.choice(os.listdir(_dir))
	if name.endswith('.wav'):
		return name
	return random_file(_dir)

def listdirs(_dir):
	return [d for d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, d))]

def random_speeches(_set='dev'):
	_path = os.path.join(DATASET_PATH, _set + '/speeches')
	return os.path.join(_path, random_file(_path))

def random_noises(_set):
	# Choose a random noise directory
	_path = os.path.join(DATASET_PATH, _set + '/noises')
	_path = os.path.join(_path, random.choice(listdirs(_path)))
	return os.path.join(_path, random_file(_path))

def get_noisy_speech(_set='dev'):
	noise_files = []
	speech_file = random_speeches(_set=_set)
	for _ in range(0, NOISE_MIX):
		noise_files.append(random_noises(_set))
	return audio_utils.overlay(speech_file, noise_files)
