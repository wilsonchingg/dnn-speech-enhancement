import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from csv_parser import parse_csv

dropout_rates = [0, 0.2, 0.4, 0.6]
dropout_base_dir = '../../training_output/v2/dropout/'
no_dropout_base_dir = '../../training_output/v2/mse/'

def plot_data_layers(training_dataset, validation_dataset):
    print(training_dataset)
    print(validation_dataset )
    plt.plot([0,1,2,3], training_dataset, label='Training Set')
    plt.plot([0,1,2,3], validation_dataset, label='Validation Set')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Mean-squared error')
    plt.xticks([0,1,2,3], dropout_rates)
    plt.legend()
    plt.show()

training_dataset = []
validation_dataset = []

filename = no_dropout_base_dir + 'mse_network_120_120_120.log'
parsed_csv = parse_csv(filename)
training_dataset.append(min(parsed_csv['loss']))
validation_dataset.append(min(parsed_csv['val_loss']))

for j in dropout_rates[1:]:
    filename = dropout_base_dir + 'dropout_' + str(j) + '.log'
    parsed_csv = parse_csv(filename)
    training_dataset.append(min(parsed_csv['loss']))
    validation_dataset.append(min(parsed_csv['val_loss']))

plot_data_layers(training_dataset, validation_dataset)
