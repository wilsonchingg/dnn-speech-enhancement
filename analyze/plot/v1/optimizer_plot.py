import sys
import matplotlib.pyplot as plt
import numpy as np
import csv

epoch = 250

def plot_data(y1, y2, y3, y4, y5, y6):
    idx = np.linspace(1, epoch, epoch)
    plt.subplot(121)
    plt.title('Training losses')
    plt.plot(idx, y1, label = 'SGD')
    plt.plot(idx, y2, label = 'RMSProp')
    plt.plot(idx, y3, label = 'Adam')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(122)
    plt.title('Validation losses')
    plt.plot(idx, y4, label = 'SGD')
    plt.plot(idx, y5, label = 'RMSProp')
    plt.plot(idx, y6, label = 'Adam')
    plt.xlabel('epoch')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    filename1 = '../../training_output/v1/sgd.log'
    filename2 = '../../training_output/v1/rmsprop.log'
    filename3 = '../../training_output/v1/gru_unit_44_layer_2.log'
    n_rows = -1
    sgd_validation = []
    rmsprop_validation = []
    adam_validation = []

    sgd_training = []
    rmsprop_training = []
    adam_training = []

    spamReader = csv.reader(open(filename1))
    next(spamReader, None)
    for row in spamReader:
        if len(row) == 0:
            continue
        sgd_training.append(float(row[2]))
        sgd_validation.append(float(row[5]))

    spamReader = csv.reader(open(filename2))
    next(spamReader, None)
    for row in spamReader:
        if len(row) == 0:
            continue
        rmsprop_training.append(float(row[2]))
        rmsprop_validation.append(float(row[5]))

    spamReader = csv.reader(open(filename3))
    next(spamReader, None)
    for row in spamReader:
        if len(row) == 0:
            continue
        adam_training.append(float(row[2]))
        adam_validation.append(float(row[5]))

    plot_data(sgd_training[0: epoch], rmsprop_training[0: epoch],
     adam_training[0: epoch], sgd_validation[0: epoch],
    rmsprop_validation[0: epoch], adam_validation[0: epoch])
