import os
import numpy as np
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(os.path.realpath(__file__))

from pesq_reader import get_moslqo

log_dir = abs_dir + '/../../test_log/evaluation/'
files = ['-5db.txt', '0db.txt', '5db.txt', '10db.txt']
# GRU, LMMSE, FeedForward, LSTM, noisy, ideal
def plot_data(y1, y2, y3, y4, y5, y6):
    print(np.round(np.array(y1), 3))
    print(np.round(np.array(y2), 3))
    print(np.round(np.array(y3), 3))
    print(np.round(np.array(y4), 3))
    print(np.round(np.array(y5), 3))
    plt.subplot(111)
    plt.plot([0, 1, 2, 3], y1, '-o', label = 'GRU model')
    plt.plot([0, 1, 2, 3], y2, '-o', label = 'log-MMSE')
    plt.plot([0, 1, 2, 3], y3, '-o', label = 'FeedForward Neural Net')
    plt.plot([0, 1, 2, 3], y4, '-o', label = 'LSTM model')
    plt.plot([0, 1, 2, 3], y5, '-o', label = 'Noisy Signal')
    # plt.plot([0, 1, 2, 3], y6, '-o', label = 'Target Signal')
    plt.xticks([0,1,2,3], ['-5', '0', '5', '10'])
    plt.xlabel('SNR (db)')
    plt.ylabel('Mean Opinion Score')
    plt.ylim(0, 4)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []

    for filename in files:
        _y1 = 0
        _y2 = 0
        _y3 = 0
        _y4 = 0
        counter = 0
        data = np.array(get_moslqo(log_dir + filename, num_of_sources = 6))
        for i in data:
            _arr = [i[0], i[1], i[2], i[3]]
            if max(_arr) == i[0]:
                _y1 += 1
            elif max(_arr) == i[1]:
                _y2 += 1
            elif max(_arr) == i[2]:
                _y3 += 1
            else:
                _y4 += 1
            if i[0] > i[5]:
                counter += 1

        _sum = np.average(data, axis=0)
        print('----------')
        print(filename)
        print('GRU ' + str(_y1/2))
        print('LMMSE ' + str(_y2/2))
        print('FeedForward ' + str(_y3/2))
        print('LSTM ' + str(_y4/2))
        print('Percentage of improved samples (GRU): ' + str(counter /2))
        print('Improved MOS (GRU): ' + str(_sum[0] - _sum[5]))
        y1.append(_sum[0])
        y2.append(_sum[1])
        y3.append(_sum[2])
        y4.append(_sum[3])
        y5.append(_sum[5])
        y6.append(_sum[4])
        print('----------')
    # GRU, LMMSE, FeedForward, LSTM, noisy, ideal
    plot_data(y1, y2, y3, y4, y5, y6)
