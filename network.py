import numpy as np
import functions as fn
import random

class Network(object):

    #PARAM sizes(list):a list of the sizes of each layer of neurons
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layer_sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        pass
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        pass
    
    def update_mini_batch(self, mini_batch, eta):
        pass
    
    def backprop(self, x, y):
        pass

    def evaluate(self, test_data):
        pass

    def cost_derivative(self, output_activations, y):
        pass