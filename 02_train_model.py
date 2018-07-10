
import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from keras.callbacks import ModelCheckpoint
import time


def divide_data(x_data, y_data, upper_limit, lower_limit):
    """
    It divides the data set into sections
    :param float x_data  : n-dimensional array contains the data.
    :param int y_data    : Labels of the data.
    :param int upper_limit: Data upper limit.
    :param int lower_limit: Data lower limit.
    :return: Divided data and its labels.
    """
    x = x_data[lower_limit: upper_limit]
    y = y_data[lower_limit: upper_limit]
    return x, y


def data_limit(x_train, ratio):
    """
    It limits the dataset, usually 20% for test and validation
    :param float x_train: n-dimensional array contains the data.
    :param float ratio  : ratio of the original data.
    """
    x = int(len(x_train) * ratio)
    return x


class TimeHistory(keras.callbacks.Callback):
    """
    It returns the a list contains the time for each epoch per second
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def arch_model(input_shape, filter_shape, number_classes):
    """
    It defines the architecture of a sequential model, (C --> P)*2 --> FC.
    Note: the number of classes is defined as 10 classes, but can be modified as required
    :param int input_shape  : Width of the inputed data.
    :param int filter_shape: Height of the filter.
    :param int number_classes   : desired number of classes.
    """
    model = Sequential()
    # activation relu , tanh
    Act = "relu"

    block_1_filters = 4
    # Conv
    model.add(Conv1D(block_1_filters, kernel_size=filter_shape, padding='same', activation=Act,
                     use_bias=True, input_shape=input_shape))
    # Pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))
    # -----------------------------------------------------------------------------------------

    block_2_filters = 8
    # Conv
    model.add(Conv1D(block_2_filters, kernel_size=filter_shape, padding='same', activation=Act,
                     use_bias=True))
    # Pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))
    # # ------------------------------------------------------------------------------------------

    # FC
    final_dense_output = number_classes
    model.add(Flatten())
    model.add(Dense(final_dense_output, activation='softmax'))
    return model


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


def train_model(epoch, batch_size, x, y, x_val, y_val):
    """

    :param int epoch: number of iteration throughout the dataset
    :param int batch_size: a portion of dataset that is fed at a time
    :param x: training data
    :param y: label data
    :param x_val: validation data
    :param y_val: validation labels
    :return: a trained model and the elapsed time
    """
    time_callback = TimeHistory()

    # 1- store the best weights
    file_path = "Noisy_Dataset/Best_weights_Original.hdf5"
    best_accuracy_checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1,
                                               save_best_only=True, mode='max')

    # 2- Early Stop
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                               verbose=0, mode='auto')
    callbacks_list = [best_accuracy_checkpoint, time_callback, early_stop]

    model.fit(x, y, batch_size, epoch, verbose=1, validation_data=(x_val, y_val), shuffle=1,
              callbacks=callbacks_list)

    time_per_epoch = time_callback.times
    # sum up the time of all epochs , divide by 60 to convert to min
    time = np.sum(time_per_epoch) / 60
    return model, time


# **************************************
# *     Load Original Dataset
# **************************************
X_train_original, Y_train_original = load_original_data()
X_train_original = X_train_original.reshape(len(X_train_original), len(X_train_original[0]), 1)

print("The original data:")
print("  X_train_original shape:", X_train_original.shape)
print("  Y_train_original shape:", Y_train_original.shape)


# **************************************
# *     divide the traces              *
# **************************************
# 1- Test data range:
testDataUpperLimit = data_limit(X_train_original, 0.2)
testDataLowerLimit = 0
x_Test, y_Test = divide_data(X_train_original, Y_train_original, testDataUpperLimit, testDataLowerLimit)


# 2- Validation data range:
valDataUpperLimit = 2 * data_limit(X_train_original, 0.2)
valDataLowerLimit = testDataUpperLimit
x_Validation, y_Validation = divide_data(X_train_original, Y_train_original, valDataUpperLimit, valDataLowerLimit)


# 3- training data range:
trainDataUpperLimit = data_limit(X_train_original, 1)
trainDataLowerLimit = valDataUpperLimit
x_Train, y_Train = divide_data(X_train_original, Y_train_original, trainDataUpperLimit, trainDataLowerLimit)

print("The divided data:")
print("  x_Train      :", x_Train.shape)
print("  x_Validation :", x_Validation.shape)
print("  x_Test       :", x_Test.shape)


Input_Shape = (len(x_Train[0]), 1)
Filter_Shape = 16
Nr_Classes = 10
model = arch_model(Input_Shape, Filter_Shape, Nr_Classes)

# summarize the architecture
model.summary()

# ***********************************
# *     Compile model
# ***********************************
Adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# Losses: categorical_crossentropy, mean_squared_error
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])


# *********************************
# *     Train model               *
# *********************************
Batch_Size = 32
Epoch = 25
model, Time = train_model(Epoch, Batch_Size, x_Train, y_Train, x_Validation, y_Validation)
print("Elapsed Time: ", Time)


# ***********************************
# *   Save the model                *
# ***********************************
model.save("Noisy_Dataset/Model_Original.hdf5")
