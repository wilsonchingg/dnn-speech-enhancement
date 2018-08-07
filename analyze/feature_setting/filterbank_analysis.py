import os
import sys
import logging

abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/../../postprocessing")
sys.path.append(abs_dir + "/../../preprocessing/mfcc")
sys.path.append(abs_dir + "/../../test")

from postprocess_util import get_mso_lqo
from mfcc_util import get_mfcc_batch, compute_feature

num_filterbank = [18, 20, 22, 24, 26, 28, 30]

eval_sample = 1000

if __name__ == "__main__":
    logging.basicConfig(filename=abs_dir + '/../training_output/spec/filterbank_analysis.log', level=logging.INFO)
    for j in range(0, eval_sample):
        print('Iteration ' + str(j))
        bg_batch, hm_batch = get_mfcc_batch()
        for i in num_filterbank:
            if i == 18:
                write_option = 0
            else:
                write_option = 1
            compute_feature(bg_batch, hm_batch , _n_filt = i, write_option = write_option)
            ideal_mso = get_mso_lqo('original.wav', 'ideal.wav')
            mixed_mso = get_mso_lqo('original.wav', 'mixed.wav')
            mso_improvement = ideal_mso - mixed_mso
            print('Filter bank (' + str(i) + ') improvement: ' + str(round(mso_improvement, 3)))
            logging.info(','.join([str(i), str(j), str(ideal_mso), str(mixed_mso), str(mso_improvement)]))
