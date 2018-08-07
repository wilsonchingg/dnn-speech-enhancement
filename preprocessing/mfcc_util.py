import os
import sys
abs_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(abs_dir + "/../postprocessing")

import numpy as np

import randomizer
from python_speech_features import sigproc
from mfcc_source import mfcc, delta
from scipy.io.wavfile import write

winlen = 0.02
sampling_rate = 16000
n_filt = 22
out_dir = abs_dir + '/../../sample_test/'

def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def getmelpoint(_n_filt = n_filt):
    lowmel = hz2mel(0)
    highmel = hz2mel(sampling_rate/2)
    melpoints = np.linspace(lowmel,highmel,_n_filt+1)
    return mel2hz(melpoints)[1: _n_filt+1]

### gains, a 2d array of size (x, n_filt)
def apply_gain(pspec, gains, modify_phase=True):
    if len(gains) == 0: # unexpected error
        return
    melpoints = getmelpoint(len(gains[0]))
    scale =  1. / winlen # 8000hz
    for i in range(0, len(pspec)):
        _melpoints_idx = 0
        for j in range(0, len(pspec[i])):
            if melpoints[_melpoints_idx] < j * scale and _melpoints_idx != len(melpoints) - 1:
                _melpoints_idx += 1
            if modify_phase:
                pspec[i][j] = pspec[i][j] * gains[i][_melpoints_idx]
            else:
                pspec[i][j] = np.complex(pspec[i][j].real * gains[i][_melpoints_idx], pspec[i][j].imag)
    return pspec

# Convert float32 to int16 arr for processing
def write_wav(dest, data):
    to_write = np.empty(len(data), dtype=np.int16)
    for i in range(0, len(data)):
        to_write[i] = int(data[i] * 32767)
    write(out_dir + dest, sampling_rate, to_write)

def get_estimate(_bg, _gains, _winlen = winlen, _winstep = winlen/2, _winfunc = lambda x:np.ones((x,))):
    nfft = int(sampling_rate * _winlen)
    frames = sigproc.framesig(_bg, _winlen*sampling_rate, _winstep*sampling_rate, winfunc = _winfunc)
    mag_spec = np.fft.rfft(frames, nfft)
    mag_spec = apply_gain(mag_spec, _gains, modify_phase=True)
    estimate_signal_frames = np.fft.irfft(mag_spec, nfft)
    return sigproc.deframesig(estimate_signal_frames, len(_bg), _winlen*sampling_rate, _winstep*sampling_rate, winfunc = _winfunc)

def get_mfcc_batch(_set='eval'):
    bg_batch, hm_batch = randomizer.get_noisy_speech(_set = _set)
    if(len(hm_batch) > 0 and len(bg_batch) > 0):
        return bg_batch, hm_batch
    else:
        return get_mfcc_batch(_set=_set)

def get_delta(mfcc_features):
    d_bg_mfcc = delta(mfcc_features, 6)
    dd_bg_mfcc = delta(d_bg_mfcc, 6)
    return np.concatenate((mfcc_features, d_bg_mfcc, dd_bg_mfcc), axis=1)
# Compute the IRM (and optionally estimated IRM) audio output
# write_option: 0 or 1
# 0 will generate mixed and original wav file, 1 will not
# return mfcc of mixed audio
def compute_feature(bg_batch, hm_batch, _n_filt = n_filt, _winlen = winlen, _winstep = winlen/2, write_option = 0,
return_irm = False, _winfunc = lambda x:np.ones((x,))):
    nfft = int(sampling_rate * _winlen)
    bg_mfcc, bg_mfcc_energy = mfcc(bg_batch, winlen = _winlen, winstep = _winstep, numcep = _n_filt, nfft = nfft,
    nfilt=_n_filt, preemph = 0, ceplifter = 0, appendEnergy=False, winfunc=_winfunc)
    hm_mfcc, hm_mfcc_energy = mfcc(hm_batch, winlen = _winlen, winstep = _winstep, numcep = _n_filt, nfft = nfft,
    nfilt=_n_filt, preemph = 0,  ceplifter = 0, appendEnergy=False, winfunc=_winfunc)
    estimate = get_estimate(bg_batch, np.clip(np.sqrt((hm_mfcc_energy/bg_mfcc_energy)), 0, 1),
     _winlen = _winlen, _winfunc = _winfunc, _winstep = _winstep)
    if write_option == 0:
        write_wav('mixed.wav', bg_batch.astype(np.float32))
        write_wav('original.wav', hm_batch.astype(np.float32))
    write_wav('ideal.wav', estimate.astype(np.float32))
    if not return_irm:
        return bg_mfcc
    else:
        return bg_mfcc, np.clip(np.sqrt((hm_mfcc_energy/bg_mfcc_energy)), 0, 1)

def generate_sample(_n_filt = n_filt, _winlen = winlen, _winstep = winlen/2, return_irm = False, _generator = get_mfcc_batch,
 _winfunc = lambda x:np.ones((x,))):
    bg_batch, hm_batch = _generator()
    return hm_batch, bg_batch, compute_feature(bg_batch, hm_batch, _n_filt = _n_filt, _winstep = _winstep,
    return_irm = return_irm,  _winlen = _winlen, _winfunc = _winfunc)
