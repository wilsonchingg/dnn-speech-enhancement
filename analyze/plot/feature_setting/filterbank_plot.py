import os
import sys
import numpy as np
import matplotlib.pyplot as plt
abs_dir = os.path.dirname(os.path.realpath(__file__))

def plot_data(y1):
    plt.subplot(111)
    plt.plot([0, 1, 2, 3, 4, 5, 6], y1, '-o')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['18', '20', '22', '24', '26', '28', '30'])
    plt.xlabel('Number of Mel filter banks')
    plt.ylabel('Average Improvement on Mean Opinion Score')
    plt.ylim(0.6, 1)

    plt.show()

if __name__ == "__main__":
    f = open(abs_dir + '/../../training_output/spec/filterbank_analysis.log')
    arr = np.zeros(7)
    count = 0
    for line in f:
        data = line.split(',')
        count += 1
        arr[(int(data[0]) - 18)//2] += float(data[4])
    plot_data(arr/(count // 7))
