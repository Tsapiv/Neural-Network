from neural_network import *
import numpy as np
# np.random.seed(22)
if __name__ == '__main__':
    print(np.show_config())
    nn = NeuralNetwork([2, 2, 1], "sigmoid")
    prediction = None
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])
    nn.train(inputs, expected_output, epochs=150, rate=15)
    print(*nn.feedforward(inputs))
