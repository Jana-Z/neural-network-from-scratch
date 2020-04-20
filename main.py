import numpy as np

from scratchNet import ScratchNet
from layer import DenseLayer
from functions import MeanSquaredError
from functions import Sigmoid, DoNothing, Softmax, MeanSquaredError
from preprocessing import load_and_split_mnist, plot_image

# MNIST
print('Importing MNIST...')
X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = load_and_split_mnist()

model = ScratchNet([
    DenseLayer(784),
    DenseLayer(128),
    DenseLayer(16),
    DenseLayer(10, activation_function='sigmoid')
], learning_rate=1.)
model.inspect()
model.evaluate(X_test_mnist, y_test_mnist, print_predictions=True)
model.fit(X_test_mnist, y_test_mnist, epochs=50, plot_every=10)
model.evaluate(X_test_mnist, y_test_mnist, print_predictions=True)

# XOR
# X_train_xor = np.array([[1, 0],
#                     [0, 0],
#                     [0, 1],
#                     [1, 1]])

# y_train_xor = np.array([[0, 1],
#                     [1, 0],
#                     [0, 1],
#                     [1, 0]])

# repeat = (1000, 1)
# X_train_xor = np.tile(X_train_xor, repeat)
# y_train_xor = np.tile(y_train_xor, repeat)

# a = ScratchNet([
#     DenseLayer(2),
#     DenseLayer(4),
#     DenseLayer(2, activation_function='softmax')
#     ], learning_rate=3.0)
# a.inspect(print_weight_and_biases=False)
# a.evaluate(X_train_xor, y_train_xor)
# a.fit(X_train_xor, y_train_xor, epochs=200)
