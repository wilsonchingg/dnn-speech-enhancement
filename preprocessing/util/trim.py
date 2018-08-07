from os import listdir
from pydub import AudioSegment
from os.path import isfile, join
from scipy.io.wavfile import read

mypath = "../sample/LibriSpeech"
outpath = "../sample/LibriSpeech/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def generateAudio(data, freq, filename):
	output = wave.open(filename, 'w')
	output.setparams((1, 2, freq, len(data), 'NONE', 'not compressed'))
	output.writeframes(data)
	output.close()


for t_file in onlyfiles:
    head_idx = 0
    end_idx = -1
    if t_file.split(".")[-1] == 'wav':
        freq, out = read(outpath + t_file)
        while(out[head_idx+1] == 0):
            head_idx += 1
        while(out[end_idx-1] == 0):
            end_idx -= 1
        generateAudio()
