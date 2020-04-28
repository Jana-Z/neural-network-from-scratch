# neural-network-from-scratch
An implimentation of a simple Dense neural network using just numpy.

## Table of coentens
- Introduction
- Technologies
- Features
- Code examples
- To do

## Introduction
This is a simple neural network using backpropergation and dense layers to classify simple projects.
It is based on the keras API.
The major goal was not usability nor speed, but to get a deeper understanding of backpropergation and neural networks in general. 


## Technologies
Project built with:
- Python 3.7
- [numpy 1.18](https://www.numpy.org)
- [prettytable 0.7](http://code.google.com/p/prettytable)
- [matplotlib 3.2](https://matplotlib.org)

## Features

### ScratchNet
#### Initialized with
- ist of layers
- loss_function (default to mean_squared_error)
- learning_rate (default to 0.1)

#### Functions
 - *fit* fits the model to the given training data using gradient descent
 - *inspect* prints a table with parameters of the net and its layers
 - *evaluate* evauluates the net using the test data (prints accuracy)
 - *predicts* predicts one sample

 ### DenseLayer
 #### Initialized with
- n: number of neurons in Layer
- activation_function: either sigmoid or softmax (defaults to sigmoid)
- weights (only initialize for development purposes)
- biases (only initialize for development purposes)

#### Functions
- *initialize_params* initalizes the weights and biases using the neuron count of the previous layer
- *feedforward* passes one input sample through the layer using the activation function
- *compute_cost_gradients* computes the gradients in the output layer
- *feed_backward* backpropergate one sample through the net

## Code Ezample
Code examples for XOR and Mnist can be found in ```main.py```

Note, that if you want to use the this network with mnist, you should also get:
- [python-mnist 0.7](https://github.com/sorki/python-mnist)
- [Mnist data set](http://yann.lecun.com/exdb/mnist/)

```main.py``` expects a folder called "mnist-data" at top-level.