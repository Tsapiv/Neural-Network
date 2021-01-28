import numpy as np
import struct
from array import array
from random import shuffle
from neural_network import NeuralNetwork, sigmoid, sigmoidPrime

class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    reader = MnistDataloader("MNIST/train_images", "MNIST/train_labels", "MNIST/test_images", "MNIST/test_labels")
    (x_train, y_train), (x_test, y_test) = reader.load_data()
    train = list(zip(x_train, y_train))
    test = list(zip(x_test, y_test))
    print("Loaded")
    config = [784, 16, 16, 10]
    r = 10
    nn = NeuralNetwork(config, sigmoid, sigmoidPrime, rate=0.1)
    for _ in range(5):
        shuffle(train)
        for i in range(0, len(train), r):
            x_train_r, y_train_r = zip(*train[i: i + r])
            nn.feedforward(np.interp(np.array(x_train_r), [0, 255], [0, 1]))
            expected = np.zeros(10 * r).reshape(r, 10)
            for j in range(r):
                expected[j, y_train_r[j]] = 1
            nn.backpropagation(expected)
        print(_)

    print("Trained")
    total = 0
    for j in range(0, len(test), r):
        x_test_r, y_test_r = zip(*train[j: j + r])
        prediction = nn.feedforward(np.interp(np.array(x_test_r), [0, 255], [0, 1]))
        total += np.sum(prediction.argmax(1) == y_test_r)

    print("Done!")
    print(total, "/", len(y_test))