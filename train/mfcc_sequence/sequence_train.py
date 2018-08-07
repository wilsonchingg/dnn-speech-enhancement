import sys
sys.path.append("..")
from global_util import get_callbacks
from batch_loader import get_dataset
from model import get_multilayer_dropout_model, set_frame_size

epochs = 251
nfft = 499
paddings = [[5, 5], [7, 7], [9, 9], [11, 11]]
stride_width = 2

for i in paddings:
    set_frame_size(i[0] + i[1] + 1)
    t_x, t_y, v_x, v_y = get_dataset(nfft, left_pad=i[0], right_pad=i[1], stride_width=stride_width)
    model = get_multilayer_dropout_model([120, 120, 120], 'gru', 0.2)
    filename =  'pad_l' + str(i[0]) + '_r'+ str(i[1]) + '_' + str(stride_width)
    print(model.summary())
    model.fit(x=t_x, y=t_y, epochs=epochs, verbose=2, callbacks=get_callbacks(filename, early_stop = 30),
    batch_size=200, validation_data = (v_x, v_y), shuffle=True, initial_epoch = 0)
