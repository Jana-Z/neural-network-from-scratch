import numpy as np
from layer import DenseLayer
from functions import MeanSquaredError
import sys
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class ScratchNet:
    def __init__(self, layers, loss_function='mean_squared_error', learning_rate=0.1):
        '''input:
            layers: list with objects of Type layer
            loss_function: a loss function given as a string
            learning_rate: the learning_rate with which the net should train (given as a float)'''
        self.layers = layers
        for index, layer in enumerate(self.layers):
            layer.set_prev_layer(self.layers[index-1] if index > 0 else None)
            layer.set_next_layer(self.layers[index+1] if index + 1 < len(self.layers) else None)
            layer.set_depth(index)
            layer.initialize_params()

        possible_losses = {
            'mean_squared_error': MeanSquaredError()
        }
        if loss_function in possible_losses:
            self.loss_function = possible_losses[loss_function]
        else:
            print(f'Loss function {loss_function} is not defined.')
            sys.exit()
        # optimizer: GD, does not support other optimizers
        if type(learning_rate) == float:
            self.learning_rate = learning_rate
        else:
            print(f'Learning rate {learning_rate} is not a float.')
            sys.exit()

    def predict(self, net_input, print_modulo=1):
        '''Go through the net once
            net_input: input that should be predictied given as a 2 dimensional array
                (so predict() always predicts a set of inputs not one single instance)
            print_modulo: states after how many examples a status should be printed

            returns:
                a 2D Matrix with the calculated probabilities for each input to belong to a certain class'''
        net_output = np.zeros((net_input.shape[0], self.layers[-1].get_number_neurons()))
        for index, sample in enumerate(net_input):
            # print(f'Going through {index+1}th sample of {net_input.shape[0]} total') if index%print_modulo == 0 else print('', end='')
            net_output[index] = self. _feed_forward(sample)
        # print(f'\nlast calculated targets:\n{net_output}')
        return net_output

    def evaluate(self, X_test, y_test, print_predictions=False):
        '''evaluates the net given X_test and y_test'''
        predictions = self.predict(X_test)
        if print_predictions:
            print(f'predictions:\n{predictions}')
            print(f'predictions shape: {predictions.shape}')
        loss, accuracy = self._calculate_loss_and_accuracy(predictions, y_test)
        print(f'loss: {loss}\naccuracy: {accuracy}')

    def inspect(self, print_weight_and_biases=False):
        '''prints information about the net'''
        params_count = 0
        t = PrettyTable(['layer', 'type', 'neurons', 'activation', 'params'])
        for layer in self.layers:
            params_layer = 0
            params_layer += layer.get_weights().shape[0] * layer.get_weights().shape[1] if not np.array_equal(layer.get_weights(), np.empty(0)) else 0
            params_layer += layer.get_biases().shape[0] if not np.array_equal(layer.get_biases(), np.empty(0)) else 0
            params_count += params_layer
            t.add_row([layer.get_depth(),
                       layer.__class__.__name__,
                       layer.get_number_neurons(),
                       layer.get_activation_function().__class__.__name__,
                       params_layer])
            if print_weight_and_biases:
                print(f'\nlayer {layer.get_depth()}\nweights:')
                print(*layer.get_weights(),sep='\n')
                print(f'biases:\n{layer.get_biases()}\n')
        print(t)
        print(f'total (trainable) params: {params_count}\n')

    def _backpropagate(self, X_train_sample, y_train_sample):
        '''backpropagates one sample out of X_train and y_train
            called by _update_params'''
        # predictions = self._feed_forward(X_train_sample)
        # print(f'predictions: {predictions}')
        gradients = self.layers[-1].compute_cost_gradients(y_train_sample, self.loss_function)
        for layer in reversed(self.layers[1:-1]):
            gradients = layer.feed_backward(gradients)
        weight_gradients = [layer.get_weight_gradients() for layer in self.layers[1:]]
        bias_gradients = [layer.get_bias_gradients() for layer in self.layers[1:]]
        return weight_gradients, bias_gradients

    def _calculate_loss_and_accuracy(self, predictions, targets):
        print(f'predictions: {predictions}\ntargets: {targets}\nshape predictions: {predictions.shape}\nshape targets: {targets.shape}')
        loss = self.loss_function.execute(predictions, targets)
        accuracy = 100 - loss*100
        return loss, accuracy

    def _feed_forward(self, input_sample):
        output_sample = self.layers[0].feed_forward(input_sample)
        for layer in range(0, len(self.layers)):
            output_sample = self.layers[layer].feed_forward(output_sample)
        return output_sample

    def fit(self, X_train, y_train, epochs=1, plot_every=None):
        ''' train the initilazied model using the training_data (X_train, y_train)
            plot_every: says after what amount of epochs a plot should be saved
        '''
        if not plot_every: plot_every = epochs
        losses_total, accuracies_total = [], []
        for x in range(int(epochs/plot_every)):
            losses, accuracies = self._gradient_descent(X_train, y_train, plot_every)
            losses_total.extend(losses)
            accuracies_total.extend(accuracies)
            plt.title('Accuracies')
            plt.plot(accuracies_total)
            plt.savefig('./plots/' + str(x) + '.png')
            # plt.show()

    def _gradient_descent(self, X_train, y_train, epochs):
        '''Uses backprop to reasses the different weights and biases of the net'''
        losses, accuracies = [], []
        for epoch in range(epochs):
            predictions = self.predict(X_train)
            self._update_params(X_train, y_train)
            loss, accuracy = (
                self._calculate_loss_and_accuracy(predictions, y_train)
            )
            losses.append(loss)
            accuracies.append(accuracy)
            print(f'Epoch {epoch+1}: loss: {loss} accuracy: {accuracy}')
        return losses, accuracies

    def _update_params(self, X_train, y_train):
        weight_gradients = [np.zeros(layer.get_weights().shape)
                          for layer in self.layers[1:]]
        bias_gradients = [np.zeros(layer.get_biases().shape)
                          for layer in self.layers[1:]]
        # print(f'weight_gradients: {weight_gradients}\nbias_gradients: {bias_gradients}')
        for sample_data, sample_target in zip(X_train, y_train):
            sample_weight_gradients, sample_bias_gradients = self._backpropagate(sample_data, sample_target)
            weight_gradients = np.add(weight_gradients, sample_weight_gradients)
            bias_gradients = np.add(bias_gradients, sample_bias_gradients)
        for layer, layer_weight_gradients, layer_bias_gradients in zip(
            self.layers[1:], weight_gradients, bias_gradients
        ):
            layer.add_to_weights(
                -self.learning_rate *
                layer_weight_gradients / len(X_train)
            )
            layer.add_to_biases(
                -self.learning_rate * layer_bias_gradients / len(X_train)
            )
