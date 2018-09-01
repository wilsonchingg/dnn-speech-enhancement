from os import listdir
from os.path import isfile, join, dirname
import wave
from scipy.io.wavfile import read

SAMPLE_DIR = join(dirname(__file__), '../../datasets/dev/speeches')
FILES = [f for f in listdir(SAMPLE_DIR) if isfile(join(SAMPLE_DIR, f))]

def generate_audio(data, _freq, filename):
	output = wave.open(join(SAMPLE_DIR, filename), 'w')
	output.setparams((1, 2, _freq, len(data), 'NONE', 'not compressed'))
	output.writeframes(data)
	output.close()

# Cut out empty signals
for f in FILES:
	head_idx = 0
	end_idx = -1
	if f.split(".")[-1] == 'wav':
		freq, out = read(join(SAMPLE_DIR, f))
		while out[head_idx+1] == 0:
			head_idx += 1
		while out[end_idx-1] == 0:
			end_idx -= 1
		generate_audio(out[head_idx:end_idx], freq, f)
