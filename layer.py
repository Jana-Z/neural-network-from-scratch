import numpy as np
import sys
from functions import Sigmoid, DoNothing, Softmax
from abc import ABC
from prettytable import PrettyTable


class Layer(ABC):
    def calculate_output(self):
        pass
    def __init__(self):
        pass
    def reasses_weights(self):
        pass

    # Setter
    def set_prev_layer(self, x):
        self.prev_layer = x
    def set_next_layer(self, x):
        self.next_layer = x
    def set_depth(self, x):
        self.depth = x

    def add_to_weights(self, x):
        if x.shape == self.weights.shape:
            self.weights += x
        else:
            print(f'shapes in add_to_weights did not align\nreceived shape {x.shape}\n but weights are in shape {self.weights.shape}')
            sys.exit()
    def add_to_biases(self, x):
        if x.shape == self.biases.shape:
            self.biases += x
        else:
            print(f'shapes in add_to_biases did not align\nreceived shape {x.shape}\n but biases are in shape {self.biases.shape}')
            sys.exit()

    # Getter
    def get_number_neurons(self):
        return self.n
    def get_depth(self):
        return self.depth
    def get_activation_function(self):
        return self.activation_function
    def get_weights(self):
        return self.weights
    def get_activations(self):
        return self.activations
    def get_biases(self):
        return self.biases
    def get_weight_gradients(self):
        return self.weight_gradients
    def get_bias_gradients(self):
        return self.bias_gradients

    def inspect(self, print_weight_and_biases=False):
        print(f'layer {self.depth}\nof type {self.__class__.__name__}\nwith {self.n} neurons\nand {self.activation_function.__class__.__name__} as activation_function')
        if print_weight_and_biases:
            if print_weight_and_biases:
                print(*self.weights,sep='\n')
                print('biases:')
                print(*self.biases, sep='\n')

class DenseLayer(Layer):
    def __init__(self, n, activation_function='sigmoid', weights=np.empty(0), biases=np.empty(0), prev_layer=None, next_layer=None, depth=None):
        '''input in params:
            n: number of neurons in this layer (necessary)
            biases: corresponding biases of the neurons
            weights: corresponding weights of the neurons with its previous layer
            activation_function: gives an activation_function as a string'''
        if type(n) == int:
            self.n = n
        else:
            print('number of neurons has to be a whole number')
            sys.exit()
        # initialized in initialize_params and in __init__ of ScratchNet.py
        self.weights  = weights
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.biases = biases
        self.depth = depth
        # Initialized during _feed_forward, needed for backprop
        self.activations = None
        self.layer_input = None
        possible_activation_functions = {
            'sigmoid' : Sigmoid(),
            'none': DoNothing(),
            'softmax': Softmax()
        }
        if activation_function in possible_activation_functions:
            self.activation_function = possible_activation_functions[activation_function]
        else:
            print(f'Activation function {activation_function} is not defined.')
            sys.exit()

    def initialize_params(self):
        # No weights for input layer and no biases for output layer
        if np.array_equal(self.weights, np.empty(0)) and self.prev_layer:
            self.weights = np.random.randn(self.n, self.prev_layer.get_number_neurons())
        if np.array_equal(self.biases, np.empty(0)):
            self.biases = np.random.randn(self.n)   #random.randn for normalized weights and biases
        # print(f'initialized a dense layer with the following attributes:\nnumber of neurons:{self.n}\nweights: {self.weights}\nbiases: {self.biases}\nactivation_function: {self.activation_function}')

    def activate(self, layer_input):
        '''layer_input: inputs with the shape 1xn(layer-1), where n(layer - 1) is the number of neurons in the layer before
                (numpy arrays)
        activation_function: function to be used when calculating outputs
                             passed as an instance of Function

        outputs: calculated outputs with the shape 1xn, where n is the number of neurons'''
        # print(f'layer_input: {layer_input}\nweights: {self.weights}')
        if not np.array_equal(self.weights, np.empty(0)):
            layer_output = np.dot(self.weights, layer_input)
        else:
            layer_output = layer_input.astype(np.float32)
        if not np.array_equal(self.biases, np.empty(0)):
            layer_output += self.biases
        self.layer_input = layer_output    #need to safe layer_output and activations for backprop
        self.activations = self.activation_function.execute(layer_output)
        return self.activations

    def compute_cost_gradients(self, targets, cost_function):
        '''computes the gradients for the output layer
            special case since the gradients in the ouput layer on rely on the gradients of the cost function
            starting point for backpropagation'''
        cost_gradients = cost_function.derivative(
            self.activations, targets
        ) * self.activation_function.derivative(self.layer_input)
        self._update_params(cost_gradients)
        return cost_gradients

    def feed_backward(self, prev_input_gradients):
        '''feed backward step through the network'''
        new_input_gradients = np.dot(
            self.next_layer.get_weights().transpose(), prev_input_gradients
        ) * self.activation_function.derivative(self.layer_input)
        self._update_params(new_input_gradients)
        return new_input_gradients

    def _update_params(self, input_gradients):
        '''updates the weights and biases according to gradients'''
        self.bias_gradients = input_gradients
        self.weight_gradients = np.dot(
            input_gradients[np.newaxis].T, self.prev_layer.get_activations()[np.newaxis ]
        )

# Flatten layer is also a DenseLayer
class FlattenLayer(DenseLayer):
    def __init__(self, input_shape, activation_function='sigmoid', weights=np.empty(0), biases=np.empty(0), prev_layer=None, next_layer=None, depth=None):
         total_neurons = 1
         for dim in input_shape:
             total_neurons *= dim
         super().__init__(n=total_neurons, activation_function='sigmoid', weights=np.empty(0), biases=np.empty(0), prev_layer=None, next_layer=None, depth=None)

    def activate(self, x):
        return super().activate(np.reshape(x,(x.shape[0], self.n)))

    def reasses_weights(self):
        pass
