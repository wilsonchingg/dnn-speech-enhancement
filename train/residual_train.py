import os, json
from global_util import get_callbacks
from batch_loader import get_dataset, training_generator
from model import get_bidirectional_model, set_frame_size

with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
	CONF = json.load(f)
	EPOCHS = CONF["train"]["epochs"]
	PAD_L = CONF["train"]["pad_l"]
	PAD_R = CONF["train"]["pad_r"]
	STRIDE_WIDTH = CONF["train"]["stride_width"]
	FRAME_SIZE = CONF["train"]["frame_size"]
	# Each audio clip is 5 second long based on the generator, the framing algorithm in
	# python_speech_features would give 499 frame levels as a result (20ms window size, 10ms overlap).
	# but note the framing algorithm of librosa would give a different number
	NFFT = 499
	NUM_OF_BATCHES = 2
	set_frame_size(PAD_L + PAD_R + 1)

_, _, v_x, v_y = get_dataset(NFFT, left_pad=PAD_L, right_pad=PAD_R, stride_width=STRIDE_WIDTH)

# bidirectional_network
model = get_bidirectional_model()
print(model.summary())
model.fit_generator(generator=training_generator(NUM_OF_BATCHES - 1, left_pad=PAD_L,
 right_pad=PAD_R, nfft=NFFT), epochs=EPOCHS, verbose=2,
callbacks=get_callbacks('b_bidirectional_network_' + str(NUM_OF_BATCHES), early_stop=15),
steps_per_epoch=(NUM_OF_BATCHES * 3000), validation_data=(v_x, v_y), shuffle=True, initial_epoch=0)
