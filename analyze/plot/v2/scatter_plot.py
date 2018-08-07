import sys
from os import listdir
import matplotlib.pyplot as plt
from os.path import isfile, join
sys.path.append('../')
from csv_parser import parse_csv

base_dir = '../../training_output/v2/mse/'

def plot(x, y):
    plt.plot(x, y, "o")
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Binary Accuracy')
    plt.show()


onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
x = []
y = []
for f in onlyfiles:
    if f.endswith('.log'):
        parsed_csv = parse_csv(base_dir + f)
        x.append(min(parsed_csv['loss']))
        x.append(min(parsed_csv['val_loss']))
        y.append(max(parsed_csv['binary_accuracy']))
        y.append(max(parsed_csv['val_binary_accuracy']))
plot(x, y)
