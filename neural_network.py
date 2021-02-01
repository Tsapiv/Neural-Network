import numpy as np

# from numba import cuda
# import cupy as cu


# @cuda.jit(device=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# @njit(nogil=True)
def sigmoid_prime(x):
    temp = sigmoid(x)
    return temp * (1.0 - temp)


# @njit(nogil=True)
def tanh(x):
    return np.tanh(x)


# @njit(nogil=True)
def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


# @njit(nogil=True)
def relu(x):
    return np.maximum(x, 0)


# @njit(nogil=True)
def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class NeuralNetwork:

    def __init__(self, configList, activate):
        self.weights = [None] * (len(configList) - 1)
        self.biases = [None] * (len(configList) - 1)
        self.errors = [None] * len(configList)
        self.activations = [None] * len(configList)
        self.activations_prime = [None] * len(configList)
        for i in range(1, len(configList)):
            self.weights[i - 1] = 2 * np.random.uniform(size=(configList[i - 1], configList[i])) - 1
            self.biases[i - 1] = 2 * np.random.uniform(size=(1, configList[i])) - 1
        self.funcs = {"sigmoid": (sigmoid, sigmoid_prime), "tanh": (tanh, tanh_prime), "relu": (relu, relu_prime)}
        self.activate, self.activate_prime = self.funcs[activate]

    def train(self, x, y, rate=0.1, epochs=10000, size=10):
        for i in range(epochs):
            for j in range(0, len(x), size):
                self.feedforward(x[j:j + size])
                self.backpropagation(y[j:j + size], rate)
            print("epoch:", i + 1)

    def feedforward(self, input):
        self.activations_prime[0] = input
        self.activations[0] = input
        for i in range(1, len(self.activations_prime)):
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            self.activations[i] = self.activate(z)
            self.activations_prime[i] = self.activate_prime(z)
        return self.activations[-1]

    def backpropagation(self, expected, rate):
        self.errors[-1] = (self.activations[-1] - expected) * self.activations_prime[-1]
        for i in range(len(self.errors) - 2, 0, -1):
            self.errors[i] = self.errors[i + 1].dot(self.weights[i].T) * self.activations_prime[i]

        for i in range(len(self.activations_prime) - 1, 0, -1):
            self.weights[i - 1] -= self.activations[i - 1].T.dot(self.errors[i]) * rate
            self.biases[i - 1] -= np.sum(self.errors[i], axis=0, keepdims=True) * rate
