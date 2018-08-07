import sys
sys.path.append("..")
from global_util import get_callbacks
from batch_loader import get_dataset
from model import get_multilayer_dropout_model

epochs = 251
nfft = 499

models = [[ [30, 30], 'gru'], [ [30, 30, 30], 'gru'], [ [30, 30, 30, 30], 'gru'],
 [[30, 30, 30, 30, 30], 'gru'], [ [60, 60, 60], 'gru'], [ [90, 90, 90], 'gru'],
 [[120, 120, 120], 'gru'], [ [150, 150, 150], 'gru']]

t_x, t_y, v_x, v_y = get_dataset(nfft, stride_width=2)

for i in models:
    model = get_multilayer_dropout_model(*i) # dropout 0
    filename =  'mse_network_' + '_'.join(str(x) for x in i[0])
    print(model.summary())
    model.fit(x=t_x, y=t_y, epochs=epochs, verbose=2, callbacks=get_callbacks(filename, early_stop=30),
    batch_size=200, validation_data = (v_x, v_y), shuffle=True)
