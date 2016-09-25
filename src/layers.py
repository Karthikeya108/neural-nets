from __future__ import print_function
import numpy as np
from utils import sigmoid, sigmoid_prime


class Layer(object):
    '''
    A layer in a neural network. A layer knows its predecessor ('previous')
    and its successor ('next'). Each layer has a forward function that
    emits output data from input data and a backward function
    that emits an output delta, i.e. a gradient, from an input delta.
    '''
    def __init__(self):
        self.params = []

        self.previous = None
        self.next = None

        self.output_data = None
        self.output_delta = None

        self.input_data = None
        self.input_delta = None

    def connect(self, layer):
        '''
        Connect a layer to its neighbours.
        '''
        self.previous = layer
        layer.next = self

    def forward(self):
        '''
        Feed input data forward. Start with:
            data = self.get_forward_input()
        to receive the output of the previous layer.
        The last line should set the output of the layer, i.e. look like
        this:
            self.output_data = output_data_of_this_layer
        '''
        raise NotImplementedError

    def get_forward_input(self):
        '''
        input_data is reserved for the first layer, all others get their
        input from the previous output.
        '''
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        '''
        Similar to the forward pass compute backpropagation of error terms,
        i.e. feed input errors backward by starting with:
            delta = self.get_backward_input()
        At the end, set the error term of this layer like this:
            self.output_delta = output_error_of_this_layer
        '''
        raise NotImplementedError

    def get_backward_input(self):
        '''
        Input delta is reserved for the very last layer, which will be set
        to the derivative of the cost function. All other layers get their
        error terms from their successor.
        '''
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def describe(self):
        '''
        Describe the properties of a layer by printing them to standard
        output.
        '''
        raise NotImplementedError


class ActivationLayer(Layer):
    '''
    In general, an activation layer computes the activation of neurons in the
    network with some activation function like sigmoid, tanh, etc.
    Here we simply use the sigmoid function for simplicity.
    '''
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)

    def clearDeltas(self):
        pass

    def updateParams(self, rate):
        pass

    def describe(self):
        print("|--- " + self.__class__.__name__)
        print("    |--- dimensions: ({0}, {1})"
              .format(str(self.input_dim), str(self.output_dim)))


class DenseLayer(Layer):
    '''
    Classic feed-forward layer. Output is defined as W * x + b.
    '''

    def __init__(self, input_dim, output_dim):

        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)
        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        #
        self.output_data = np.zeros((self.output_dim, 1))
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight,data) + self.bias

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()

        self.delta_b += delta
	self.delta_w += np.dot(delta,data.T)
        self.output_delta = np.zeros((self.input_dim, 1))
        self.output_delta = np.dot(self.weight.T,delta)

    def clearDeltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def updateParams(self, rate):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def describe(self):
        print("|--- " + self.__class__.__name__)
        print("    |--- dimensions: ({0}, {1})"
              .format(str(self.input_dim), str(self.output_dim)))
