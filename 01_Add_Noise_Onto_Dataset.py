# import the required python libraries.
import numpy as np
import h5py
from matplotlib import pyplot as plt
np.random.seed(128)


def generate_noise(nr_samples, sample_length, noise_per_sample, sigma):
    """
    It generates as much noise as required from a normal (Gaussian) distribution.
    :param int nr_samples       : Number of samples that different noise vectors must be added to them.
    :param int sample_length    : Number of elements in the sample vector.
    :param int noise_per_sample : Desired number of noise vectors that each sample should get.
    :param float sigma          : standard deviation, it tells how much the data points are
                        close(for small value) / far (for high value) form the mean
    """
    noise_length = nr_samples * sample_length * noise_per_sample
    new_nr_samples = nr_samples * noise_per_sample
    mu = 0  # mean
    noise = np.random.normal(mu, sigma, noise_length).astype('float32')

    noise = noise.reshape(new_nr_samples, sample_length)
    return noise


def add_noise(new_nr_samples, sample_length, x_train, noise, y_train):
    """
    It add a noise onto a signal.
    :param int new_nr_samples: Desired new number of samples.
    :param int sample_length : Length of a sample.
    :param float x_train     : Original signal without noise.
    :param float noise       : Generated noise.
    :param int y_train       : label of the original signal.
    """
    k = 0
    # creating an array to hold both signals
    pattern_x = np.ndarray((new_nr_samples * sample_length)).reshape(new_nr_samples, sample_length)

    for i in range(int(new_nr_samples / (len(x_train)))):
        for j in range(len(x_train)):
            pattern_x[k] = np.add(x_train[j], noise[k])
            k = k + 1
            # create an array to hold the new labels
    pattern_y = np.copy(y_train)
    for i in range(int(new_nr_samples / (len(x_train))) - 1):
        pattern_y = np.concatenate((pattern_y, y_train))
    return pattern_x, pattern_y


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
# * 0- Read Original dataset
# ******************************************************
Original_File = h5py.File("Original_Dataset/X_Original.hdf5", "r")
X_Original = np.array(Original_File.get('X_O'))
print("X_Original", X_Original.shape)

# ******************************************************
# * 1- Read Original Labels
# ******************************************************
Labels_File = h5py.File("Original_Dataset/Y_Original.hdf5", "r")
Y_Original = np.array(Labels_File.get('Y_O'))
print("Y_Original", Y_Original.shape)


# ******************************************************
# *  2- Generating a Noise
# ******************************************************
# each trace gets n different type of noise vectors
Noise_Per_Trace = 2
Nr_Traces = len(X_Original)
Trace_Length = len(X_Original[0])
New_Nr_Traces = Noise_Per_Trace * Nr_Traces
sigma = 1
Noise = generate_noise(Nr_Traces, Trace_Length, Noise_Per_Trace, sigma)


# ******************************************************
# *  3- Adding Noise into Traces Without Shift
# ******************************************************
X_Noisy, Y_Noisy = add_noise(New_Nr_Traces, Trace_Length, X_Original, Noise, Y_Original)
print("X_Noisy", X_Noisy.shape)
print("Y_Noisy", Y_Noisy.shape)


# ******************************************************
# * 4- Store the Noisy Label
# ******************************************************
Noisy_Label_File = h5py.File("Noisy_Dataset/Y_Noisy.hdf5", "w")
set_Noisy = Noisy_Label_File.create_dataset('N_L', data=Y_Noisy)


# ******************************************************
# * 5- Store the Noisy traces
# ******************************************************
Noisy_File = h5py.File("Noisy_Dataset/X_Noisy.hdf5", "w")
set_Noisy_Original = Noisy_File.create_dataset('N_O_T', data=X_Noisy)


# ******************************************************
# * 6- Plot the first 5 classes
# ******************************************************
plot_5_classes(X_Noisy)
