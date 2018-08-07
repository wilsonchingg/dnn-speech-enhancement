import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from csv_parser import parse_csv

base_dir = '../../training_output/v2/sequence/'

idx = np.array([3, 5, 7, 9, 11])

def sequence_plot(training_dataset, validation_dataset, no_dropout_training_dataset, no_dropout_validation_dataset):
    sequence_idx = idx * 2 + 1
    plt.plot(sequence_idx, training_dataset, label='Training Set (with dropout)')
    plt.plot(sequence_idx, validation_dataset, label='Validation Set (with dropout)')
    plt.plot(sequence_idx, no_dropout_training_dataset, label='Training Set (without dropout)')
    plt.plot(sequence_idx, no_dropout_validation_dataset, label='Validation Set (without dropout)')
    plt.xlabel('Sequence length')
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
        filename = base_dir + 'dropout_pad_l' + str(i) + '_r' + str(i) + '_2_lab.log'
        parsed_csv = parse_csv(filename)
        training_dataset.append(min(parsed_csv['loss']))
        validation_dataset.append(min(parsed_csv['val_loss']))
        filename = base_dir + 'pad_l' + str(i) + '_r' + str(i) + '_2.log'
        parsed_csv = parse_csv(filename)
        no_dropout_training_dataset.append(min(parsed_csv['loss']))
        no_dropout_validation_dataset.append(min(parsed_csv['val_loss']))
    sequence_plot(training_dataset, validation_dataset, no_dropout_training_dataset, no_dropout_validation_dataset)
