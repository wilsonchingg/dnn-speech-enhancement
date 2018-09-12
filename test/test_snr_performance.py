import os, sys, json, numpy as np
from logmmse import logmmse_from_file
from predict import predict
sys.path.append(os.path.join(os.path.dirname(__file__), "../preprocessing"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../train"))
import randomizer
from mfcc_utils import generate_sample, get_estimate, write_wav
from mfcc_sequence_util import to_sequence_with_stride
from postprocess_util import evaluate
from audio_utils import set_snr

with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
	CONF = json.load(f)
	WINLEN = CONF["winlen"]
	NFILT = CONF["n_filt"]
	EPOCHS = CONF["eval"]["epochs"]
	WINSTEP = CONF["winstep"]
	PAD_L = CONF["train"]["pad_l"]
	PAD_R = CONF["train"]["pad_r"]

multipliers = [[0.316, 10], [0.56, 5], [1, 0], [1.79, -5]]

for j in multipliers:
	def get_audio():
		return randomizer.get_noisy_speech(_set='dev')
	set_snr(j[0])
	for i in range(0, EPOCHS):
		hm, bg, mfcc_feature = generate_sample(_n_filt=NFILT, _winlen=WINLEN, _winstep=WINSTEP,
         _winfunc=np.hamming, _generator=get_audio, to_write=True)
		logmmse_from_file(os.path.join(os.path.dirname(__file__), '../sample_test/mixed.wav'),
        os.path.join(os.path.dirname(__file__), '../sample_test/logmmse.wav'))
		sequence_feature = np.array(to_sequence_with_stride(mfcc_feature,
         left_pad=PAD_L, right_pad=PAD_R))
        # sys model
		prediction = predict(sequence_feature, 'sys_model')
		estimate = get_estimate(bg, prediction, _winlen=WINLEN, _winstep=WINSTEP, _winfunc=np.hamming)
		write_wav('sys_model.wav', estimate.astype(np.float32))
		evaluate(estimated_sources=['sys_model.wav', 'logmmse.wav'])
	os.rename('pesq_results.txt', str(j[1]) + 'db.txt')
