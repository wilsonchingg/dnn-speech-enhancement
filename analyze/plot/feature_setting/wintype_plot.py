import os
import sys
import numpy as np
import matplotlib.pyplot as plt
abs_dir = os.path.dirname(os.path.realpath(__file__))

types = ['rectangular', 'hanning', 'hamming', 'blackman', 'bartlett']
def plot_data(y1):
    plt.subplot(111)
    plt.bar([0, 1, 2, 3, 4], y1, align='center', alpha=0.5)
    plt.xticks([0, 1, 2, 3, 4], types)
    plt.xlabel('Window Type')
    plt.ylabel('Average Improvement on Mean Opinion Score')
    plt.ylim(0, 1.5)
    plt.show()

if __name__ == "__main__":
    f = open(abs_dir + '/../../training_output/spec/wintype_analysis.log')
    arr = np.zeros(len(types))
    count = 0
    for line in f:
        data = line.split(',')
        count += 1
        arr[types.index(data[0])] += float(data[4])
    plot_data(arr/(count // len(arr)))
