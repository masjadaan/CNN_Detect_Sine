import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras
import time
from keras.models import load_model


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def divide_data(x_data, y_data, upper_limit, lower_limit):
    """
    It divides the data set into sections
    :param float x_data  : n-dimensional array contains the data.
    :param int y_data    : Labels of the data.
    :param int upper_limit: Data upper limit.
    :param int lower_limit: Data lower limit.
    :return: Divided data and its labels.
    """
    x_train = x_data[lower_limit: upper_limit]
    y_train = y_data[lower_limit: upper_limit]
    return x_train, y_train


def data_limit(x_train, ratio):
    """
    It limits the dataset, usually 20% for test and validation
    :param float x_train: n-dimensional array contains the data.
    :param float ratio  : ratio of the original data.
    """
    x = int(len(x_train) * ratio)
    return x


def plot_signal(x_signal, y_signal, class_number):
    """
    It plots the signal with its class number.
    :param float x_signal : 4 dimensional tensor, its second dimension contains the signal amplitude.
    :param float y_signal : 2 dimensional tensor, its second dimension contains the calss number
    :param int class_number: the desired class number.
    """
    plt.figure(figsize=(5,2))
    sample_length = len(x_signal[0])
    plt.plot(x_signal[class_number].reshape(sample_length,1))
    plt.title("Class {}".format(y_signal.argmax(1)[class_number]))
    plt.show()


def load_original_data():
    """
    It reads the dataset that are stored in hdf5
    :return: training traces and their labels
    """
    original_file = h5py.File('Noisy_Dataset/X_Noisy.hdf5', 'r')
    x = np.array(original_file.get('N_O_T'))
    label_file = h5py.File('Noisy_Dataset/Y_Noisy.hdf5', 'r')
    y = np.array(label_file.get('N_L'))
    return x, y


# *****************************************
# * read the Noisy
# *****************************************
x_Test, y_Test = load_original_data()
x_Test = x_Test.reshape(len(x_Test), len(x_Test[0]), 1)

# 1- Test data range:
testDataUpperLimit = data_limit(x_Test, 0.2)
testDataLowerLimit = 0
x_Test, y_Test = divide_data(x_Test, y_Test, testDataUpperLimit, testDataLowerLimit)

print("The Test data:")
print("  x_Test shape:", x_Test.shape)
print("  x_Test shape:", y_Test.shape)
print('********************************\n')


# *****************************************
# *  Load a trained model and its weights *
# *****************************************
model = load_model("Noisy_Dataset/Model_Original.hdf5")
model.load_weights("Noisy_Dataset/Best_weights_Original.hdf5")


# ***********************************
# *   Evaluate model on test data   *
# ***********************************
batchSize = 32
loss, accuracy = model.evaluate(x_Test, y_Test, batchSize, verbose=0)
print('The Accuracy of Model: {:.2f}%'.format(accuracy * 100))
print('The loss on test data: {:.2f}'.format(loss))
