**About**
====

In this project an artificial dataset has been created in order to use it in training and testing a convolutional
neural network (CNN).

**The Dataset:**
====
The dataset consists of n vectors of length m, each vector contains a landmark (generated randomly)
followed by a scaled sine wave.

| 1.  As the supervised learning requires labelled data so that those vectors are labelled
depending on the scaled sine waves as each sine wave has a different amplitude corresponds to its scaling factor.
The scaling factor are:  

                  [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

| Therefore there are 10 defferent classes ( class 0 untill class 9)  

| 2. Each vetor has been padded by 50 zeros.  

| 3. The landmark and the scaled sine wave are shifted to the right or to the left rondomly. 

This how the combined signals look like for the first 5 classes
![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional-Neural-Network/master/Detecting%20Sine%20Wave/Original_Signal.png)

| 4. A Noise from a normal (Gaussian) distribution with standard deviation equals 1 is generated, then each vector
from the dataset gets two noise vectors. 

| 5. Final data shape is: 

      | x_Train      : (4800, 124, 1)  
      | x_Validation : (1600, 124, 1)  
      | x_Test       : (1600, 124, 1)  
      
The following figures show the combined signals when standard deviation  sigma = 1 (Top) where the original signals can be distinguished, and sigma = 5 (Bottom) the original signal is hidden within the noise
![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional-Neural-Network/master/Detecting%20Sine%20Wave/Signal_Noise_sigma1.png)

![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional-Neural-Network/master/Detecting%20Sine%20Wave/Signal_Noise_sigma5.png)


**Training the model**
====
| 1. The model architecture consists of only two blocks, each contains one Convolutional layer and one Pooling layer
| 2. Filter of size 16, in first block there are 4 filters, while in the second 8.
| 3. ReLU is used as an activation function
| 4. Mean Squared Error is used as loss fuction
| 5. Epoch = 25
| 6. Batch size = 32.

**Evaluating the model**
====
The model has been evaluated on test data, the accuracy reaches 70%



**This project doesn't provide the optimal solution, but rather an approach to obtain an understanding on
how the CNN works under different parametrs (number of filters, loss functions, activations ...etc).**

|Author:
Mahmoud Jadaan

