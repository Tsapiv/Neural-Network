import numpy as np
import cupy as cu
from time import time
import struct
from array import array
from random import shuffle
from neural_network import NeuralNetwork, sigmoid, sigmoid_prime
import pickle
import gzip

def load_data():
    f = gzip.open('MNIST/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return training_data, validation_data, test_data

def shape(data, size):
    temp = np.zeros((len(data), size))
    temp[np.arange(temp.shape[0]), data] = 1
    return temp

def accuracy(predicted, expected):
    return np.sum(predicted.argmax(1) == np.array(expected)) / len(expected)

if __name__ == '__main__':
    data = load_data()[0:2]
    x_train, y_train = data[0]
    x_test, y_test = data[1]


    print("Loaded")
    config = [784, 100, 10]
    nn = NeuralNetwork(config, "sigmoid")
    y_train = shape(y_train, 10)
    print("Start")
    a = time()
    nn.train(x_train, y_train, rate=0.3, epochs=10, size=20)
    b = time()
    prediction = nn.feedforward(x_test)
    total = accuracy(prediction, y_test)
    print("Training time:", b - a)
    print("Done! Total:", total)