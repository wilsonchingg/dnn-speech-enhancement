import os
import sys
import numpy as np
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_dir + "/../../../preprocessing")

from pesq_reader import get_moslqo
import setting
setting.init()



log_dir = abs_dir + '/../../test_log/evaluation/noise_type/'

def plot_data(y1, y2, y3):
    plt.subplot(111)
    plt.bar(np.arange(6) - 0.15, y1, width=0.3, label = 'GRU network',align='center')
    plt.bar(np.arange(6) + 0.15, y2, width=0.3, label = 'LMMSE',align='center')
    # plt.bar(np.arange(6) + 0.2, y3, width=0.2, label = 'Noisy speech',align='center')
    plt.xticks(np.arange(6), ['Demestic', 'Nature', 'Office', 'Public', 'Street', 'Transportation'])
    plt.xlabel('Noise Category')
    plt.ylabel('Mean Opinion Score')
    plt.ylim(0, 4)
    plt.legend()
    plt.show()

unprocess_noisy_speech = []
log_mmse = []
gru_model = []

for i, filename in enumerate(setting.sample_sub_dir):
    log_mmse_counter = 0
    gru_model_counter = 0
    data = np.array(get_moslqo(log_dir + filename + '.txt', num_of_sources = 4))
    for i in data:
        if i[1] > i[0]: # lmmse has a better score
            log_mmse_counter += 1
        else:
            gru_model_counter += 1

    _sum = np.average(data, axis=0)
    print('----------')
    print(filename)
    print('log_mmse_counter: ' + str(log_mmse_counter) )
    print('gru_model_counter: ' + str(gru_model_counter))
    gru_model.append(_sum[0])
    log_mmse.append(_sum[1])
    unprocess_noisy_speech.append(_sum[3])
    print('----------')

gru_model = np.array(gru_model)
log_mmse = np.array(log_mmse)
unprocess_noisy_speech = np.array(unprocess_noisy_speech)

gru_model = np.average(np.reshape(gru_model, (6, 3)), axis=1)
log_mmse = np.average(np.reshape(log_mmse, (6, 3)), axis=1)
unprocess_noisy_speech = np.average(np.reshape(unprocess_noisy_speech, (6, 3)), axis=1)

print(np.round(unprocess_noisy_speech, 3))
print(np.round(log_mmse, 3))
print(np.round(gru_model, 3))

print('Average')
print(np.round(np.average(unprocess_noisy_speech), 3))
print(np.round(np.average(log_mmse), 3))
print(np.round(np.average(gru_model), 3))

plot_data(gru_model, log_mmse, unprocess_noisy_speech)
