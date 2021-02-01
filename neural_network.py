import numpy as np
# import cupy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    temp = sigmoid(x)
    return temp * (1.0 - temp)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

class Layer:
    def __init__(self, size, activation, prev=None, next=None, weights=True, biases=True):
        self.weights = 2 * np.random.uniform(size=(prev.size, size)) - 1 if weights else None
        self.biases = 2 * np.random.uniform(size=(1, size)) - 1 if biases else None
        self.size = size
        self.next = next
        self.prev = prev
        self.error = None
        self.activations = None
        self.activations_prime = None
        self.activate, self.activate_prime = NeuralNetwork.funcs[activation]

    def feedforward(self):
        z = np.dot(self.prev.activations, self.weights) + self.biases
        self.activations = self.activate(z)
        self.activations_prime = self.activate_prime(z)
        return self.activations

    def calc_error(self):
        self.error = self.next.error.dot(self.next.weights.T) * self.activations_prime

    def corr_error(self, rate):
        self.weights -= self.prev.activations.T.dot(self.error) * rate
        self.biases -= np.sum(self.error, axis=0, keepdims=True) * rate

        

class NeuralNetwork:
    funcs = {"sigmoid": (sigmoid, sigmoid_prime), "tanh": (tanh, tanh_prime), "relu": (relu, relu_prime)}

    def __init__(self, configList, loss_func=lambda x, y: x - y):
        self.loss_func = loss_func
        self.head = self.tail = Layer(configList[0][0], configList[0][1], weights=False, biases=False)
        for size, activate in configList[1:]:
            layer = Layer(size, prev=self.tail, activation=activate)
            self.tail.next = layer
            self.tail = layer

    def train(self, x, y, rate=0.1, epochs=10000, size=10):
        for i in range(epochs):
            for j in range(0, len(x), size):
                self.feedforward(x[j:j + size])
                self.backpropagation(y[j:j + size], rate)
            print("epoch:", i + 1)

    def feedforward(self, input):
        curr_layer = self.head
        curr_layer.activations = input
        curr_layer = curr_layer.next
        while curr_layer:
            curr_layer.feedforward()
            curr_layer = curr_layer.next
        return self.tail.activations

    def backpropagation(self, expected, rate):
        curr_layer = self.tail
        curr_layer.error = self.loss_func(curr_layer.activations, expected) * curr_layer.activations_prime
        curr_layer = curr_layer.prev
        while curr_layer.prev:
            curr_layer.calc_error()
            curr_layer = curr_layer.prev
        curr_layer = self.tail
        while curr_layer.prev:
            curr_layer.corr_error(rate)
            curr_layer = curr_layer.prev
# class NeuralNetwork:
#     funcs = {"sigmoid": (sigmoid, sigmoid_prime), "tanh": (tanh, tanh_prime), "relu": (relu, relu_prime)}
#
#     def __init__(self, configList, activate):
#         self.weights = [None] * (len(configList) - 1)
#         self.biases = [None] * (len(configList) - 1)
#         self.errors = [None] * len(configList)
#         self.activations = [None] * len(configList)
#         self.activations_prime = [None] * len(configList)
#         for i in range(1, len(configList)):
#             self.weights[i - 1] = 2 * np.random.uniform(size=(configList[i - 1], configList[i])) - 1
#             self.biases[i - 1] = 2 * np.random.uniform(size=(1, configList[i])) - 1
#         self.activate, self.activate_prime = NeuralNetwork.funcs[activate]
#
#     def train(self, x, y, rate=0.1, epochs=10000, size=10):
#         for i in range(epochs):
#             for j in range(0, len(x), size):
#                 self.feedforward(x[j:j + size])
#                 self.backpropagation(y[j:j + size], rate)
#             print("epoch:", i + 1)
#
#     def feedforward(self, input):
#         self.activations_prime[0] = input
#         self.activations[0] = input
#         for i in range(1, len(self.activations_prime)):
#             z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
#             self.activations[i] = self.activate(z)
#             self.activations_prime[i] = self.activate_prime(z)
#         return self.activations[-1]
#
#     def backpropagation(self, expected, rate):
#         self.errors[-1] = (self.activations[-1] - expected) * self.activations_prime[-1]
#         for i in range(len(self.errors) - 2, 0, -1):
#             self.errors[i] = self.errors[i + 1].dot(self.weights[i].T) * self.activations_prime[i]
#
#         for i in range(len(self.activations_prime) - 1, 0, -1):
#             self.weights[i - 1] -= self.activations[i - 1].T.dot(self.errors[i]) * rate
#             self.biases[i - 1] -= np.sum(self.errors[i], axis=0, keepdims=True) * rate
