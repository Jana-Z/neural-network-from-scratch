import numpy as np
import sys
from prettytable import PrettyTable
from abc import ABC

from scratchNet import ScratchNet
from layer import DenseLayer, FlattenLayer
from functions import MeanSquaredError
from functions import Sigmoid, DoNothing, Softmax, MeanSquaredError

#XOR-data
X_train_xor = np.array([[1, 0],
                    [0, 0],
                    [0, 1],
                    [1, 1]])

y_train_xor = np.array([[0, 1],
                    [1, 0],
                    [0, 1],
                    [1, 0]])

repeat = (1000, 1)
X_train_xor = np.tile(X_train_xor, repeat)
y_train_xor = np.tile(y_train_xor, repeat)

a = ScratchNet([
    DenseLayer(2),
    DenseLayer(4),
    DenseLayer(2, activation_function='sigmoid')
    ], learning_rate=0.1)
a.inspect(print_weight_and_biases=False)
a.evaluate(X_train_xor, y_train_xor)
a.fit(X_train_xor, y_train_xor, epochs=300)
