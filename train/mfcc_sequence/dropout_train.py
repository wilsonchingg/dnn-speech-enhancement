import sys
sys.path.append("..")

from keras.utils import plot_model
from global_util import get_callbacks
from batch_loader import get_dataset
from model import get_multilayer_dropout_model

epochs = 251
nfft = 499

models = [[[120, 120, 120], 'gru', 0.2], [ [120, 120, 120], 'gru', 0.4],
 [ [120, 120, 120], 'gru', 0.6]]

t_x, t_y, v_x, v_y = get_dataset(nfft, stride_width=2)

for i in models:
    model = get_multilayer_dropout_model(*i)
    filename =  'dropout_' + str(i[2])
    print(model.summary())
    model.fit(x=t_x, y=t_y, epochs=epochs, verbose=2, callbacks=get_callbacks(filename, early_stop=30),
    batch_size=200, validation_data = (v_x, v_y), shuffle=True)
