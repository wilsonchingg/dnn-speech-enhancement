import sys
import matplotlib.pyplot as plt
import numpy as np
import csv

epoch = 250

def plot_data(training_dataset, validation_dataset):
    idx = np.linspace(1, epoch, epoch)

    plt.subplot(121)
    plt.title('Training losses')
    counter = 1
    for i in training_dataset:
        plt.plot(idx, i, label = str(counter) + ' hidden GRU layers')
        counter += 1
    plt.ylim(0.05, 0.10)
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(122)
    plt.title('Validation losses')
    counter = 1
    for i in validation_dataset:
        plt.plot(idx, i, label = str(counter) + ' hidden GRU layers')
        counter += 1
    plt.ylim(0.05, 0.10)
    plt.xlabel('epoch')
    # plt.legend()
    plt.show()


def plot_data2(training_dataset, validation_dataset):
    best_training = np.amin(training_dataset, axis=1)
    print(best_training)
    best_validation = np.amin(validation_dataset, axis=1)
    idx = np.linspace(1, 10, 10)
    plt.plot(idx, best_training, label='Best training loss by layers')
    plt.plot(idx, best_validation, label='Best validation loss by layers')
    plt.xlabel('Layers')
    plt.legend()

    plt.show()
if __name__ == "__main__":
    training_dataset = []
    validation_dataset = []
    for i in range(1, 11):
        filename = '../../training_output/v1/gru_unit_44_layer_' + str(i) + '.log'
        n_rows = -1
        training = []
        validation = []
        spamReader = csv.reader(open(filename))
        next(spamReader, None)
        for row in spamReader:
            if len(row) == 0:
                continue
            training.append(float(row[2]))
            validation.append(float(row[5]))
        training_dataset.append(training[0:250])
        validation_dataset.append(validation[0:250])
    plot_data2(training_dataset, validation_dataset)
