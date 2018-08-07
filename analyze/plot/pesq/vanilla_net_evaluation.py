import os
import numpy as np
import matplotlib.pyplot as plt
abs_dir = os.path.dirname(os.path.realpath(__file__))
from pesq_reader import get_moslqo

log_dir = abs_dir + '/../../test_log/v1/'
files = ['-5db.txt', '0db.txt', '5db.txt', '10db.txt']

def plot_data(y1, y2, y3):
    plt.subplot(111)
    plt.plot([0, 1, 2, 3], y1, '-o', label = 'Estimated Signal')
    plt.plot([0, 1, 2, 3], y2, '-o', label = 'Target signal')
    plt.plot([0, 1, 2, 3], y3, '-o', label = 'Noisy Speech')
    plt.xticks([0,1,2,3], ['-5', '0', '5', '10'])
    plt.xlabel('SNR (db)')
    plt.ylabel('Mean Opinion Score')
    plt.ylim(-0.5, 4.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    y1 = []
    y2 = []
    y3 = []
    for filename in files:
        data = np.array(get_moslqo(log_dir + filename))
        _sum = np.average(data, axis=0)
        y1.append(_sum[0])
        y2.append(_sum[1])
        y3.append(_sum[2])
        counter = 0
        for i in data:
            if i[0] > i[2]:
                counter += 1
        print(filename)
        print('Average Improvement: ' + str(_sum[0] - _sum[2]))
        print('Number of samples with MOS-LQO improvement: ' + str(counter) + '  , percentage: ' + str(counter /5) + '%')

    plot_data(y1, y2, y3)
