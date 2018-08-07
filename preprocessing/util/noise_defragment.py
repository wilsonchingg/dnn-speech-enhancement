import os
import sys
abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/..")
import uuid
from os import listdir
from os.path import isfile, join

from scipy.io.wavfile import read, write




sample_dir = "../../sample/"


for noise_type in setting.sample_sub_dir:
    mypath = sample_dir + noise_type
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for filename in onlyfiles:
        freq, out = read(mypath + '/' + filename)
        print(freq)
        num_of_fragment = len(out) // 160000 # 10 seconds
        for i in range(0, num_of_fragment):
            out_file = str(uuid.uuid1())
            write(mypath + '/' + out_file + '.wav', freq, out[i * 160000: (i + 1) * 160000])
            # sys.exit()
