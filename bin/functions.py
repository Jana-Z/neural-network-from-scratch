from abc import ABC
import numpy as np

class Function(ABC):
    def execute (self, *args):
        '''execute function given the input
        takes as an input either an int, float or scalar numpy array'''
        pass

    def derivative(self, *args):
        '''execute the derivative function on the given input
        takes as an input either an int, float or scalar numpy array'''
        pass

    def test(self, *args):
        '''Tests the function by given the value (both normal and derivative) at x'''
        pass

#  Activation functions
class Sigmoid(Function):
    def execute(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return self.execute(x) * (1 - self.execute(x))

    def test(self, x=3):
        print(f'Sigmoid of {x} is {self.execute(x)}')
        print(f'Derivative of Sigmoid {x} is {self.derivative(x)}')

class Softmax(Function):
    def execute(self, x):
        '''Compute softmax values for each sets of scores in x.'''
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def derivative(self, x):
        return self.execute(x) * (1. - self.execute(x))

    def test(self):
        pass

# for testing purposes
class DoNothing(Function):
    def execute(self, x):
        return x
    def derivative(self, x):
        return x
    def test(self):
        pass

# Loss functions
class MeanSquaredError(Function):
    def execute(self, predictions, targets):
        '''receive both outputs and targets as two dimensional nparrays'''
        return np.mean((1/2*np.square(targets - predictions).sum(axis = 1)))

    def derivative(self, predictions, targets):
        return predictions - targets

    def test(self,
    predictions=np.array([
        [0.1, 0.4, 0.6],
        [0.5, 0.8, 1],
        [1, 0.001, 0.3]
    ]),
    targets=np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])):
        print(f'Calculated MSE with\npredictions: \n{predictions} \nand targets: \n{targets}\nis {self.execute(predictions, targets)}\nshould be 0.285')
