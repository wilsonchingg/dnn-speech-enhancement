from os import listdir
from pydub import AudioSegment
from os.path import isfile, join
import os
import sys

abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/..")

mypath = "../../sample/LibriSpeech_Dev/"
outpath = "../../sample/LibriSpeech_Dev/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for t_file in onlyfiles:
    # print(outpath + '.'.join(t_file.split(".")[0:-1]))
    if t_file.split(".")[-1] == 'flac':
        out = AudioSegment.from_file(outpath + t_file, t_file.split(".")[-1])
        out.export(outpath + '.'.join(t_file.split(".")[0:-1]) + ".wav",
         format="wav")
