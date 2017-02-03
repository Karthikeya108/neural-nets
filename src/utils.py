import numpy as np


class Objective(object):

    def cost_function(self, predictions, labels):
        raise NotImplementedError

    def cost_derivative(self, predictions, labels):
        raise NotImplementedError


class MSE(Objective):
    '''
    Mean squared error function and its derivative.
    '''
    def cost_function(self, predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff*diff)[0]

    def cost_derivative(self, predictions, labels):
        return predictions - labels


# Sigmoid function on doubles and its vectorized version
def sigmoid_double(z):
    return 1.0/(1.0+np.exp(-z))

sigmoid = np.vectorize(sigmoid_double)


# Derivative of sigmoid for doubles and numpy arrays
def sigmoid_prime_double(z):
    
    return sigmoid_double(z) * (1.0 - sigmoid_double(z))

sigmoid_prime = np.vectorize(sigmoid_prime_double)
