import os
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment

# Change the path to desired location
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), '../../datasets/dev/speeches')
FILELIST = [f for f in listdir(SAMPLE_DIR) if isfile(join(SAMPLE_DIR, f))]

for f in FILELIST:
	if f.split(".")[-1] == 'flac':
		out = AudioSegment.from_file(os.path.join(SAMPLE_DIR, f), f.split(".")[-1])
		out.export(os.path.join(SAMPLE_DIR, '.'.join(f.split(".")[0:-1]) + ".wav"),
         format="wav")
