import sys
import matplotlib.pyplot as plt
import numpy as np
import csv

size = [1, 2, 3, 4, 5, 6]
base_dir = '../../training_output/v2/'


def plot_data_units(train_acc,train_loss,val_acc,val_loss,train_acc_2,train_loss_2,val_acc_2,val_loss_2 ):
    print(val_loss_2)
    print('----')
    plt.subplot(121)
    plt.title('MSE')
    plt.plot(train_loss_2, label='Training loss of residual model')
    plt.plot(train_loss, label='Training loss of non-residual model')
    plt.plot(val_loss_2, label='Validation loss of residual model')
    plt.plot(val_loss, label='Validation loss of non-residual model')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(122)
    plt.title('Binary Accuracy')
    plt.plot(np.array(train_acc_2) * 100, label='Training acc of residual model')
    plt.plot(np.array(train_acc) * 100, label='Training acc of unidirectional model')
    plt.plot(np.array(val_acc_2) * 100, label='Validation acc of residual model')
    plt.plot(np.array(val_acc) * 100, label='Validation acc of non-residual model')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def get_data(filename):
    val_acc = []
    train_acc = []
    val_loss = []
    train_loss = []
    spamReader = csv.reader(open(filename))
    next(spamReader, None)
    print(filename)
    for row in spamReader:
        if len(row) == 0:
            continue
        if len(row) < 6:
            train_acc.append(float(row[1]))
            train_loss.append(float(row[2]))
            val_acc.append(float(row[3]))
            val_loss.append(float(row[4]))
            continue
        train_acc.append(float(row[1]))
        train_loss.append(float(row[2]))
        val_acc.append(float(row[4]))
        val_loss.append(float(row[5]))
    return train_acc,train_loss,val_acc,val_loss

filename = base_dir + 'dataset_size/dropout_batch_mse_2.log'
train_acc,train_loss,val_acc,val_loss = get_data(filename)

filename = base_dir + 'other/b_mse_r_2.log'
train_acc_2,train_loss_2,val_acc_2,val_loss_2 = get_data(filename)
print(val_loss_2)
plot_data_units(train_acc,train_loss,val_acc,val_loss,train_acc_2,train_loss_2,val_acc_2,val_loss_2 )
