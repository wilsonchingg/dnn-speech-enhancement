import os
import sys
import logging

abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/../../postprocessing")
sys.path.append(abs_dir + "/../../preprocessing/mfcc")
sys.path.append(abs_dir + "/../../test")

import numpy as np
from postprocess_util import get_mso_lqo
from mfcc_util import get_mfcc_batch, compute_feature

configs = [lambda x:np.ones((x,)), np.hanning, np.hamming, np.blackman, np.bartlett]
labels = ['rectangular', 'hanning', 'hamming', 'blackman', 'bartlett']

n_filt = 30
eval_sample = 500

if __name__ == "__main__":
    logging.basicConfig(filename=abs_dir + '/../training_output/spec/wintype_analysis.log', level=logging.INFO)
    for j in range(0, eval_sample):
        print('Iteration ' + str(j))
        bg_batch, hm_batch = get_mfcc_batch()
        for i in range(0, len(configs)):
            if i == 0:
                write_option = 0
            else:
                write_option = 1
            compute_feature(bg_batch, hm_batch , _n_filt = n_filt, _winfunc = configs[i], write_option=write_option)
            ideal_mso = get_mso_lqo('original.wav', 'ideal.wav')
            mixed_mso = get_mso_lqo('original.wav', 'mixed.wav')
            mso_improvement = ideal_mso - mixed_mso
            print('Window type (' + labels[i] + ') improvement: ' + str(round(mso_improvement, 3)))
            logging.info(','.join([labels[i], str(j), str(ideal_mso), str(mixed_mso), str(mso_improvement)]))
