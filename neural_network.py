import numpy as np
from pprint import pprint

# np.random.seed(1)


def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def sigmoidPrime(val):
    temp = sigmoid(val)
    return temp * (1 - temp)


class NeuralNetwork:

    def __init__(self, configList, activate, activatePrime, rate=1):
        self.weights = [2 * np.random.uniform(size=(configList[i - 1], configList[i])) - 1 for i in range(1, len(configList))]
        self.biases = [2 * np.random.uniform(size=(1, configList[i])) - 1 for i in range(1, len(configList))]
        self.layers = np.empty(len(configList), dtype=object)
        self.activations = np.empty(len(configList), dtype=object)
        self.errors = np.empty(len(configList), dtype=object)
        self.rate = rate
        self.activate = activate
        self.activatePrime = activatePrime

    def feedforward(self, input):
        self.layers[0] = input
        self.activations[0] = input
        for i in range(1, len(self.layers)):
            self.layers[i] = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            self.activations[i] = self.activate(self.layers[i])
        return self.activations[-1]

    def backpropagation(self, expected):
        self.errors[-1] = (self.activations[-1] - expected) * self.activatePrime(self.layers[-1])
        for i in range(len(self.errors) - 2, 0, -1):

            # transposed = np.transpose(self.weights[i])
            # matrix_mult = np.matmul(transposed, self.errors[i + 1])
            # term = self.activatePrime(self.layers[i])
            # product = np.multiply(matrix_mult, term)
            self.errors[i] = self.errors[i + 1].dot(self.weights[i].T) * self.activatePrime(self.layers[i])
            # self.errors[i] = product
        for i in range(len(self.layers) - 1, 0, -1):
            # transposed = np.transpose(self.activations[i])
            # err = self.errors[i]
            # product = np.matmul(err, transposed)
            # weight = self.weights[i - 1]
            # a = (np.matmul(self.errors[i], np.transpose(self.activations[i])))
            # am = a.mean(0)
            # b = (self.errors[i])
            # bm = b.mean(1)
            self.weights[i - 1] -= self.activations[i - 1].T.dot(self.errors[i]) * self.rate

            # self.weights[i - 1] += self.activations[i].T.dot(self.errors[i]).mean(0).reshape(1, -1) * self.rate
            err = self.errors[i]
            new = np.sum(self.errors[i], axis=0, keepdims=True)
            self.biases[i - 1] -= np.sum(self.errors[i], axis=0, keepdims=True) * self.rate






if __name__ == '__main__':
    pass


    # inp = np.random.rand(6)
    # out = np.zeros(6)
    # out[np.argmax(inp, axis=0)] = 1
    # print(inp, out)

    # for _ in range(1000):
    #     inp = np.random.rand(4).reshape(2, 2)
    #     # print(inp)
    #     out = np.argmax(inp, axis=0)
    #     # print(out)
    #     nn.feedforward(inp)
        # print("=========================")
        # for layer in nn.layers:
        #     print(layer)
        # print("=========================")
        # for err in nn.errors:
        #     print(err)
        # print("=========================")
        # for weight in nn.weights:
        #     print(weight)
        # print("=========================")
        # for bias in nn.biases:
        #     print(bias)
    #     nn.backpropagation(out)
    #
    # inp = np.random.rand(4).reshape(2, 2)
    #
    # out = np.argmax(inp, axis=0)
    # print(inp, out, nn.feedforward(inp))





