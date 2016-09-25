from __future__ import division, print_function
import random
from utils import MSE


class Network():
    '''
    Neural network class, designed to sequentially stack layers in a simple
    fashion.

    As objective / cost function we use mean square error (MSE), which is found
    in the utils module. All layers should be implemented in the layers module.
    The model is trained by mini-batch gradientdescent.

    Usage:
        - Initialize the network by creating a new instance, e.g.
            net = Network()
        - Add layers one by one using the 'add' method, e.g.
            net.add(layers.ActivationLayer(100))
        - Train the model on data by calling the 'train' method:
            net.train(data, num_of_epochs, mini_batch_size,
                      learning_rate, test_data)
    '''

    def __init__(self, objective=None):
        print("Initialize Network...")
        self.layers = []
        if objective is None:
            self.objective = MSE()

    def add(self, layer):
        '''
        Add a layer, connect it to its predecessor and let it describe itself.
            - layer: A layer from the layers module
        '''
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self, training_data, epochs, mini_batch_size,
              learning_rate, test_data=None):
        '''
            First, shuffle training data, then split it into mini batches.
            Next, for each mini-batch,
            train this batch. In case test data is present, evaluate it.

            - training_data: A numpy array of pairs containing features and
              labels
            - epochs: The number of epochs/iterations to be trained.
            - mini_batch_size: Number of samples to fit in one batch
            - learning_rate: The learning rate for the gradient descent update
              rule
            - test_data: Optional test data for evaluation
        '''
        n_train = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                print("Accuracy for epoch {0}: {1}".
                      format(j, self.evaluate(test_data) / n_test))
            else:
                print("Epoch {0} complete".format(j))

    def train_batch(self, mini_batch, learning_rate):
        '''
        Feed forward and pass backward a mini-batch, then update parameters
        accordingly.
        '''
        self.forward_backward(mini_batch)
        self.update(mini_batch, learning_rate)

    def update(self, mini_batch, learning_rate):
        '''
        Normalize learning rate, then update each layer.
        Afterwards, clear all deltas to start the next batch properly.
        '''
        learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers:
            layer.updateParams(learning_rate)
        for layer in self.layers:
            layer.clearDeltas()

    def forward_backward(self, mini_batch):
        '''
        For each sample in the mini batch, feed the features forward layer by
        layer.
        Then compute the cost derivative and do layer by layer backpropagation.
        '''
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()
            
            ##print("Train MSE: ",self.objective.cost_function(self.layers[-1].output_data, y))

            self.layers[-1].input_delta = self.objective.cost_derivative(
                self.layers[-1].output_data, y)
            for layer in reversed(self.layers):
                layer.backward()

    def single_forward(self, x):
        '''
        Pass a single sample forward and return the result.
        '''
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        '''
        Returns the number of correctly predicted labels.
        '''
        #
        # EXERCISE 6:
        # For each sample in test data compute the outcome produced by the
        # neural net by using 'single_forward'. Then determine the index of the
        # largest element in the outcome vector to compare it to the respective
        # label.
        #
        # For instance, if output_data = (0.0, 0.9, 0.1), then 0.9 is the most
        # likely result of the network and its position in the array should
        # match the label.
        #
        # Return the number of correctly predicted labels.
        #
        count = 0
        for sample in test_data:
	    value = list(self.single_forward(sample[0]))
            prediction = value.index(max(value))
            if prediction == sample[1]:
	        count += 1
        
        return count
