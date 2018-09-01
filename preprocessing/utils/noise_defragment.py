import os
from os import listdir
from os.path import isfile, join

import uuid
from scipy.io.wavfile import read, write

def listdirs(_dir):
	return [d for d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, d))]

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/dev/noises')
SAMPLE_SUB_DIRS = listdirs(SAMPLE_DIR)

for noise_type in SAMPLE_SUB_DIRS:
	mypath = os.path.join(SAMPLE_DIR, noise_type)
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for filename in onlyfiles:
		freq, out = read(mypath + '/' + filename)
		num_of_fragment = len(out) // 160000 # 10 seconds
		for i in range(0, num_of_fragment):
			out_file = str(uuid.uuid1())
			write(mypath + '/' + out_file + '.wav', freq, out[i * 160000: (i + 1) * 160000])
