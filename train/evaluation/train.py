import os
import sys
abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/..")
from eval_model import get_feedforward, get_lstm
from global_util import get_callbacks
from batch_loader import get_dataset, training_generator, get_delta_dataset

epochs = 251
nfft = 499
pad_l = 5
pad_r = 5
stride_width = 2

### Dense training ###
model = get_feedforward()
x, y, x_v, y_v = get_delta_dataset(nfft)
model.fit(x=x[0:600000], y=y[0:600000], epochs=epochs, batch_size=200, verbose=2,
 validation_data=(x_v[0:200000], y_v[0:200000]), shuffle=True, callbacks=get_callbacks('dense', early_stop=15))


### LSTM training ###
_, _, v_x, v_y = get_dataset(nfft, left_pad=pad_l, right_pad=pad_r, stride_width=stride_width)
model = get_lstm()
print(model.summary())
model.fit_generator(generator=training_generator(2, left_pad = pad_l, right_pad = pad_r), epochs=epochs, verbose=2,
callbacks=get_callbacks('lstm',  early_stop = 15),
steps_per_epoch=((3) * 12000), validation_data = (v_x, v_y), shuffle=True)
