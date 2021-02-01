import numpy as np
import struct
from array import array
from random import shuffle


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(x,0)


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

class Layer:

    def __init__(self, dim, id, act, act_prime, isoutputLayer = False):
        self.weight = 2 * np.random.random(dim) - 1
        self.delta = None
        self.A = None
        self.activation = act
        self.activation_prime = act_prime
        self.isoutputLayer = isoutputLayer
        self.id = id


    def forward(self, x):
        z = np.dot(x, self.weight)
        self.A = self.activation(z)
        self.dZ = np.atleast_2d(self.activation_prime(z));

        return self.A

    def backward(self, y, rightLayer):
        if self.isoutputLayer:
            error = self.A - y
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                np.dot(rightLayer.delta, rightLayer.weight.T)
                * self.dZ)
        return self.delta

    def update(self, learning_rate, left_a):
        a = np.atleast_2d(left_a)
        d = np.atleast_2d(self.delta)
        ad = a.T.dot(d)
        self.weight -= learning_rate * ad


class NeuralNetwork:

    def __init__(self, layersDim, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime

        self.layers = []
        for i in range(1, len(layersDim) - 1):
            dim = (layersDim[i - 1] + 1, layersDim[i] + 1)
            self.layers.append(Layer(dim, i, self.activation, self.activation_prime))

        dim = (layersDim[i] + 1, layersDim[i + 1])
        self.layers.append(Layer(dim, len(layersDim) - 1, self.activation, self.activation_prime, True))

    def fit(self, X, y, learning_rate=0.1, epochs=10000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)


        for k in range(epochs):


            a=X

            for l in range(len(self.layers)):
                a = self.layers[l].forward(a)


            delta = self.layers[-1].backward(y, None)

            for l in range(len(self.layers) - 2, -1, -1):
                delta = self.layers[l].backward(delta, self.layers[l+1])



            a = X
            for layer in self.layers:
                layer.update(learning_rate, a)
                a = layer.A
            print("Epochs:", k)

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.layers)):
            a = self.layers[l].forward(a)
        return a


if __name__ == '__main__':
    # reader = MnistDataloader("train_images", "train_labels", "test_images", "test_labels")
    # (x_train, y_train), (x_test, y_test) = reader.load_data()
    # x_train = np.interp(np.array(x_train), [0, 255], [0, 1])
    # x_test = np.interp(np.array(x_test), [0, 255], [0, 1])
    # temp = np.zeros([len(y_train), 10])
    # temp[np.arange(temp.shape[0]), y_train] = 1
    # y_train = temp
    # print("Loaded")
    # config = [784, 16, 16, 10]
    # r = 10
    # nn = NeuralNetwork([784, 16, 16, 10], activation="sigmoid")
    # nn.fit(x_train, y_train, epochs=500)
    # total = 0
    # print("Trained")
    # for i in range(len(x_test)):
    #     out = nn.predict(x_test[i])
    #     total += nn.predict(x_test[i]).argmax(0) == y_test[i]
    #
    # print("Done!")
    # print(total, "/", len(y_test))

    data = load_data()[0:2]
    x_train, y_train = data[0]
    x_test, y_test = data[1]

    # print("Loaded")
    # config = [784, 100, 10]
    # nn = NeuralNetwork(config, "sigmoid")
    # x_train = np.array(x_train) / 255
    # x_test = np.array(x_test) / 255
    # temp = np.zeros((len(y_train), 10))
    # temp[np.arange(temp.shape[0]), y_train] = 1
    # y_train = temp
    # print("Start")
    # a = time()
    # nn.train(x_train, y_train, rate=3.0, epochs=30)
    # prediction = nn.feedforward(x_test)
    # total = np.sum(prediction.argmax(1) == np.array(y_test))
    #
    # print("Trained", time() - a)
    #
    #
    # print("Done! Total:", total)

    print("Loaded")
    config = [784, 100, 10]
    nn = NeuralNetwork(config, "sigmoid")
    # x_train = np.array(x_train) / 255
    # x_test = np.array(x_test) / 255
    temp = np.zeros((len(y_train), 10))
    temp[np.arange(temp.shape[0]), y_train] = 1
    y_train = temp
    print("Start")
    a = time()
    nn.train(x_train, y_train, rate=0.1, epochs=10, size=1)
    prediction = nn.feedforward(x_test).argmax(1)
    total = np.sum(prediction == np.array(y_test))

    print("Trained", time() - a)

    print("Done! Total:", total)

    nn = NeuralNetwork([2, 2, 1], activation="sigmoid")
    prediction = None
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])
    nn.fit(inputs, expected_output, learning_rate=0.1)
    print(*nn.layers[-1].A)
