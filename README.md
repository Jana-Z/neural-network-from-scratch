# neural-network-from-scratch
An implimentation of a simple Dense neural network using just numpy

## Features
- DenseLayer
- sigmoid and softmax as activation function
- Gradient Descent (using backprop) as an optimizer

## How to use
The network can be called similiar to keras.
Examples for XOR and Mnist can be found in main.py

## Installation
The net itself just requires numpy and Python3.
If you want to use it with mnist, you will need
 - matplotlib
 - [python-mnist](https://pypi.org/project/python-mnist/)
 - and [Mnist itself](http://yann.lecun.com/exdb/mnist/) in a folder called mnist-data (or change the path in preprocessing.py)
