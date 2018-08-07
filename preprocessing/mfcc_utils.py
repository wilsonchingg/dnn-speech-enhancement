import os, randomizer, json, numpy as np

from python_speech_features import sigproc
from mfcc_source import mfcc, delta, hz2mel, mel2hz
from audio_utils import write_wav

CONF_PATH = os.path.join(os.path.dirname(__file__), '../config.json')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../sample_test')

with open(CONF_PATH) as f:
	CONF = json.load(f)
	N_FILT = CONF["n_filt"]
	SAMPLING_RATE = CONF["sampling_rate"]
	WINLEN = CONF["winlen"]


def getmelpoint(_n_filt=N_FILT):
	lowmel = hz2mel(0)
	highmel = hz2mel(SAMPLING_RATE/2)
	melpoints = np.linspace(lowmel, highmel, _n_filt+1)
	return mel2hz(melpoints)[1: _n_filt+1]

# gains, a 2d array of size (x, n_fil
# TODO: the counting process should operate on bin level instead
def apply_gain(pspec, gains):
	if not gains: # unexpected error
		return []
	melpoints = getmelpoint(len(gains[0]))
	scale = 1. / WINLEN # 8000hz
	for i, val in enumerate(pspec):
		_melpoints_idx = 0
		for j in range(0, len(val)):
			if melpoints[_melpoints_idx] < j * scale and _melpoints_idx != len(melpoints) - 1:
				_melpoints_idx += 1
			pspec[i][j] = pspec[i][j] * gains[i][_melpoints_idx]
	return pspec

def get_estimate(_bg, _gains, _winlen=WINLEN, _winstep=WINLEN/2,
 _winfunc=lambda x: np.ones((x,))):
	nfft = int(SAMPLING_RATE * _winlen)
	frames = sigproc.framesig(_bg, _winlen*SAMPLING_RATE, _winstep*SAMPLING_RATE, winfunc=_winfunc)
	mag_spec = np.fft.rfft(frames, nfft)
	mag_spec = apply_gain(mag_spec, _gains)
	estimate_signal_frames = np.fft.irfft(mag_spec, nfft)
	return sigproc.deframesig(estimate_signal_frames, len(_bg), _winlen*SAMPLING_RATE,
	 _winstep*SAMPLING_RATE, winfunc=_winfunc)

def get_delta(mfcc_features):
	d_bg_mfcc = delta(mfcc_features, 6)
	dd_bg_mfcc = delta(d_bg_mfcc, 6)
	return np.concatenate((mfcc_features, d_bg_mfcc, dd_bg_mfcc), axis=1)

def compute_feature(bg_batch, hm_batch, _n_filt=N_FILT, _winlen=WINLEN, _winstep=WINLEN//2,
 to_write=True, _winfunc=lambda x: np.ones((x,))):
	nfft = int(SAMPLING_RATE * _winlen)
	bg_mfcc, bg_mfcc_energy = mfcc(bg_batch, winlen=_winlen, winstep=_winstep,
	 numcep=_n_filt, nfft=nfft, nfilt=_n_filt, preemph=0,
	 ceplifter=0, appendEnergy=False, winfunc=_winfunc)
	_, hm_mfcc_energy = mfcc(hm_batch, winlen=_winlen, winstep=_winstep,
	 numcep=_n_filt, nfft=nfft, nfilt=_n_filt, preemph=0,
	 ceplifter=0, appendEnergy=False, winfunc=_winfunc)
	# Ideal Ratio Mask applied signal
	estimate = get_estimate(bg_batch, np.clip(np.sqrt((hm_mfcc_energy/bg_mfcc_energy)), 0, 1),
	 _winlen=_winlen, _winfunc=_winfunc, _winstep=_winstep)
	if to_write == 0:
		write_wav('mixed.wav', bg_batch.astype(np.float32))
		write_wav('original.wav', hm_batch.astype(np.float32))
		write_wav('ideal.wav', estimate.astype(np.float32))
	return bg_mfcc

def generate_sample(_n_filt=N_FILT, _winlen=WINLEN, _winstep=WINLEN//2,
 _generator=randomizer.get_noisy_speech, _winfunc=lambda x: np.ones((x,))):
	bg_batch, hm_batch = _generator()
	return hm_batch, bg_batch, compute_feature(bg_batch, hm_batch, _n_filt=_n_filt,
	 _winstep=_winstep, _winlen=_winlen, _winfunc=_winfunc)
