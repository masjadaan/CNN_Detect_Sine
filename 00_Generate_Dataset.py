import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import h5py


def normalize(ndarr):
    """
    Takes an n-dimensions array and divides its element by the largest number within the array.
    :param ndarr: n-dimensions array.
    :returns n-dimensional array.
    """
    max_amp = np.ndarray.max(ndarr.astype('float32'))
    ndarr /= max_amp
    return ndarr


def generate_sine_wave(length, angle):
    """
    It generates a vector of a specified length that holds a sine wave of specified angle.
    :param int length: vector's length.
    :param float angle: sine's angle.
    :return: a vector holds amplitudes of a sine wave.
    """
    time = np.arange(length)
    sine_wave = np.sin(time * angle)
    return sine_wave


def multiply_sine_by_factors(scaling_factors, sine_vector):
    """
    It multiplies a sine signal by a scaling factors
    :param int scaling_factors: a list of integers
    :param float sine_vector:  vector holds the sine amplitudes
    :return: matrix where each row contains a sine signal multiplied by a different scaling factor.
    """
    matrix = np.zeros(len(scaling_factors) * len(sine_vector)).reshape(len(scaling_factors), len(sine_vector))
    for i in range(len(scaling_factors)):
        matrix[i] = scaling_factors[i] * sine_vector
    return matrix


def combine_two_signals(signal_1, signal_2):
    """
    It concatenates the second signal (sine signal) to the first one (landmark).
    :param float signal_1: matrix of two dimensions.
    :param float signal_2: vector contains amplitudes of signal_2.
    :return: matrix where each row holds the combined signals.
    """
    combined_length = len(signal_1[0]) + len(signal_2)
    matrix = np.zeros(len(signal_1) * combined_length).reshape(len(signal_1), combined_length)
    for i in range(len(signal_1)):
        matrix[i] = np.append(signal_2, signal_1[i])
    return matrix


def pad_zeros(signal, padded_zero):
    """
    It pads zeros into a signal along x-axis in both directions, thus the length of the new signal
        new signal length = padded zero + signal length + padded zero
    :param float signal: vector contains amplitudes of signal
    :param int padded_zero: desired zeros
    :return:
    """
    # each tuple represents axis
    npad = ((0, 0), (padded_zero, padded_zero))
    padded_signal = np.pad(signal, pad_width=npad, mode='constant', constant_values=0)
    return padded_signal


def create_label(nr_classes):
    """
    It creates matrix label
    :param int nr_classes: number of desired classes
    :return: matrix contains labels
    """
    label_vector = np.arange(nr_classes)
    label_vector = np_utils.to_categorical(label_vector, nr_classes)
    return label_vector


def shifts_list(signal, padded_zeros, step):
    """
    It defines the maximum allowed shift: max shift range = added zero/length of signal. Then
    creates a shifts vector according to a define step.
    :param float signal: n-dimensional array.
    :param int padded_zeros: number of padded zeros into signal.
    :param folat step: desired step.
    :return: integer specifies the maximum allowed shift of signal.
    """
    shift = np.around((padded_zeros/len(signal[0])), decimals=2)
    shifts_vector = np.arange(0, shift, step)
    return shifts_vector


def data_augmentation(x_signal, y_signal, shift_list):
    """
    It performs data augmentation by sifting the training data randomly

    :param float x_signal: n-dimensional array holds the training data
    :param int y_signal: n-dimensional array holds the labels
    :param shift_list: list of the desired shifts
    :return:
    """
    datagen = ImageDataGenerator(height_shift_range=shift_list)
    x_signal = x_signal.reshape(len(x_signal), len(x_signal[1]), 1, 1)
    i = 0
    for x_train_aug, y_train_aug in datagen.flow(x_signal, y_signal, batch_size=10, shuffle=True):
        i += 1
        if i > 10:
            break  # otherwise the generator would loop indefinitely
    x_train_aug = x_train_aug.reshape(len(x_signal), len(x_signal[1]))
    return x_train_aug, y_train_aug


def create_dataset(x_signal, y_signal, shift_list):
    """
    It creates a dataset where the x_signal is shifted randomly then concatenated with the
    original x_signal and the labels are reserved. This process is repeated as long as
    there is shifts in the shift_list.

    :param shift_list: contains the desired shifts.
    :param y_signal: n-dimensional array holds the training data.
    :param x_signal: n-dimensional array holds the labels.
    :return: training data with their corresponding labels.
    """
    for i in range(len(shift_list)):
        if i == 0:
            x_train_temp = np.copy(x_signal)
            y_train_temp = np.copy(y_signal)
        else:
            x_train_aug, y_train_aug = data_augmentation(x_signal, y_signal, shift_list[i])
            x_train_temp = np.concatenate((x_train_temp, x_train_aug), axis=0)
            y_train_temp = np.concatenate((y_train_temp, y_train_aug), axis=0)
    return x_train_temp, y_train_temp


def plot_5_classes(signal):
    """
    It plots the first 5 classes on the same figure
    :param signal: n-dimensional array holds the signals
    :return: plots
    """
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 16,
            }
    plt.figure(figsize=(10,5))
    # plt.title('Pattern With Multiple Scaled Sine Signals', fontdict=font)
    plt.xlabel('time', fontdict=font)
    plt.ylabel('Amplitude', fontdict=font)
    plt.plot(signal[0].reshape(124,1), 'r', label='red   -> Class 0')
    plt.plot(signal[1].reshape(124,1), 'b', label='blue  -> Class 1')
    plt.plot(signal[2].reshape(124,1), 'g', label='green -> Class 2')
    plt.plot(signal[3].reshape(124,1), 'y', label='red   -> Class 3')
    plt.plot(signal[4].reshape(124,1), 'k', label='blue  -> Class 4')
    plt.legend()
    plt.show()


# ******************************************************
# * 1- creating a pattern of random shape
# ******************************************************
Pattern_Length = 12
# pattern = np.random.uniform(5, 10, patternLength)

pattern = np.array([9.33318521,  6.31572425,  5.65704239,  5.20796722,  6.19462167,  8.227373,
  8.95299676,  8.00721219,  6.67149686,  5.59714231,  6.54566526,  5.54861893])


# ******************************************************
# * 2- create a sine wave
# ******************************************************
Sine_Length = 12
Angle = np.pi/6
Sine_Wave = generate_sine_wave(Sine_Length, Angle)


# ******************************************************
# * 3- Scale sine wave
# ******************************************************
Scaling_Factors = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
Scaled_Sine = multiply_sine_by_factors(Scaling_Factors, Sine_Wave)


# ******************************************************
# * 4- Combine two signals
# ******************************************************
Combined_Signals = combine_two_signals(Scaled_Sine, pattern)


# ******************************************************
# * 5- Pad zeros
# ******************************************************
Padded_Zeros = 50
Padded_Zeros_Signal = pad_zeros(Combined_Signals, Padded_Zeros)


# ******************************************************
# * 6- Create labels
# ******************************************************
Nr_Classes = 10
Y_signal = create_label(Nr_Classes)


# ******************************************************
# * 7- Create random shifted dataset
# ******************************************************
Step = 0.001
Shifts_Vector = shifts_list(Padded_Zeros_Signal, Padded_Zeros, Step)
X_Original, Y_Original = create_dataset(Padded_Zeros_Signal, Y_signal, Shifts_Vector)
print("  X_original:", X_Original.shape)
print("  Y_original:", Y_Original.shape)


# ******************************************************
# * 8- Plot the first 5 classes
# ******************************************************
plot_5_classes(X_Original)


# ******************************************************
# * 9- Store the original traces
# ******************************************************
Original_Traces_File = h5py.File("Original_Dataset/X_Original.hdf5", "w")
Original_Dataset = Original_Traces_File.create_dataset('X_O', data=X_Original)
Original_Traces_File.close()


# ******************************************************
# * 10- Store the original Labels
# ******************************************************
Original_Labels_File = h5py.File("Original_Dataset/Y_Original.hdf5", "w")
Original_Labels = Original_Labels_File.create_dataset('Y_O', data=Y_Original)
Original_Labels_File.close()
