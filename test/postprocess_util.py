import os
import sys

abs_dir = os.path.dirname(os.path.realpath(__file__))
out_dir = abs_dir + '/../sample_test/'

import numpy as np
import subprocess

# return example P.862 Prediction (Raw MOS, MOS-LQO):  = 3.594	3.682
def get_pesq(s1, s2):
    args = [ 'pesq', '+16000', out_dir + s1, out_dir + s2  ]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    return str(out).split('\\n')[-2]

def get_mso_lqo(s1, s2):
    result = get_pesq(s1, s2)
    num = result.split('\\t')[-1].strip()
    return float(num)

def mso_lqo_mfcc_test(true_source, ideal_source, mixed_source, *sources):
    for i in range(0, len(sources)):
        print('Estimate Quality ' + str(i))
        print(get_pesq(true_source, sources[i]))
    print('Ideal Quality')
    print(get_pesq(true_source, ideal_source))
    print('Mixed Quality')
    print(get_pesq(true_source, mixed_source))

def evaluate(true_source = 'original.wav', ideal_source = 'ideal.wav', mixed_source = 'mixed.wav',
    estimated_sources = ['estimation.wav']):
    mso_lqo_mfcc_test(true_source,
    ideal_source, mixed_source, *estimated_sources)

def smoothing(predictions):
    smoothed_prediction = np.array(predictions, copy=True)
    for i in range(1, len(predictions)):
        for j in range(0, 22):
            smoothed_prediction[i][j] = max(0.6 * smoothed_prediction[i - 1][j], smoothed_prediction[i][j])
    return smoothed_prediction
