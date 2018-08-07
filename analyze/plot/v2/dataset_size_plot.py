import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from csv_parser import parse_csv

base_dir = '../../training_output/v2/dataset_size/'
idx = np.array([0, 1, 2, 3, 4, 5])

def sequence_plot(training_dataset, validation_dataset, no_dropout_training_dataset, no_dropout_validation_dataset):
    sequence_idx = (idx + 1) * 6
    plt.plot(sequence_idx, training_dataset, label='Training Set')
    plt.plot(sequence_idx, validation_dataset, label='Validation Set')
    plt.xlabel('Number of training samples (100k)')
    plt.ylabel('Mean Squared Error')
    plt.xticks(sequence_idx)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    training_dataset = []
    validation_dataset = []
    no_dropout_training_dataset = []
    no_dropout_validation_dataset = []
    for i in idx:
        filename = base_dir + 'dropout_batch_mse_' + str(i) + '.log'
        parsed_csv = parse_csv(filename)
        training_dataset.append(min(parsed_csv['loss']))
        validation_dataset.append(min(parsed_csv['val_loss']))
        filename = base_dir + 'b_mse_network_' + str(i) + '.log'
        parsed_csv = parse_csv(filename)
        no_dropout_training_dataset.append(min(parsed_csv['loss']))
        no_dropout_validation_dataset.append(min(parsed_csv['val_loss']))
    sequence_plot(training_dataset, validation_dataset, no_dropout_training_dataset, no_dropout_validation_dataset)
