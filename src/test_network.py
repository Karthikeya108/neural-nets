import data_loader as dl
from network import *
from layers import *

(train_data, test_data) = dl.load_csv_data()

net = Network()
net.add(DenseLayer(784,100))
net.add(ActivationLayer(100))
net.add(DenseLayer(100,10))
net.add(ActivationLayer(10))

net.train(train_data, 10, 10, 3.0, test_data)
