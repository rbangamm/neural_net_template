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
        for b, w in zip(self.biases, self.weights):
            a = fn.sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)
        #Shuffle the training data for each epoch
        #and create mini batches
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        
        #Update mini batches
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        
        if test_data:
            n_test = len(test_data)
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else:
            print("Epoch {0} complete".format(j))