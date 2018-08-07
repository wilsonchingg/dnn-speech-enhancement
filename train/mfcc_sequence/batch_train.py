import sys
sys.path.append("..")
from global_util import get_callbacks
from batch_loader import get_dataset, training_generator
from model import get_multilayer_dropout_model, set_frame_size

epochs = 251
nfft = 499
pad_l = 5
pad_r = 5
stride_width = 2
set_frame_size(pad_l + pad_r + 1)
_, _, v_x, v_y = get_dataset(nfft, left_pad=pad_l, right_pad=pad_r, stride_width=stride_width)

# 5 corresponds to 6 data shards
for i in [1, 2, 3, 4, 5]:
    model = get_multilayer_dropout_model([120, 120, 120], 'gru', 0.2)
    print(model.summary())
    model.fit_generator(generator=training_generator(i, left_pad = pad_l, right_pad = pad_r), epochs=epochs, verbose=2,
    callbacks=get_callbacks('batch_mse_' + str(i), early_stop = 15),
    steps_per_epoch=((i+1) * 3000), validation_data = (v_x, v_y), shuffle=True)
