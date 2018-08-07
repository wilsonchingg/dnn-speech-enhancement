import sys
import numpy as np
sys.path.append("..")
from global_util import get_callbacks
from batch_loader import get_dataset, training_generator
from model import get_residual_model, get_bidirectional_model, set_frame_size

epochs = 251
nfft = 499
pad_l = 5
pad_r = 5
stride_width = 2
set_frame_size(pad_l + pad_r + 1)
_, _, v_x, v_y = get_dataset(nfft, left_pad=pad_l, right_pad=pad_r, stride_width=stride_width)

i = 2

# bidirectional_network
setting = 'bidirectional_network'
model = get_bidirectional_model()
print(model.summary())
model.fit_generator(generator=training_generator(i, left_pad=pad_l, right_pad=pad_r), epochs=epochs, verbose=2,
callbacks=get_callbacks('b_' + setting + '_' + str(i), early_stop = 15),
steps_per_epoch=((i+1) * 3000), validation_data = (v_x, v_y), shuffle=True, initial_epoch = 0)

# single_bidirectional_network
setting = 'mse_r'
model = get_residual_model()
print(model.summary())
model.fit_generator(generator=training_generator(i, left_pad=pad_l, right_pad=pad_r), epochs=epochs, verbose=2,
callbacks=get_callbacks('b_' + setting + '_' + str(i), early_stop = 15),
steps_per_epoch=((i+1) * 3000), validation_data = (v_x, v_y), shuffle=True)
