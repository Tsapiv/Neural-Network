import numpy as np
from neural_network import NeuralNetwork, sigmoid, sigmoidPrime

# np.random.seed(1)
if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1], sigmoid, sigmoidPrime, rate=0.1)
    prediction = None
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])
    for _ in range(10000):
        prediction = nn.feedforward(inputs)
        nn.backpropagation(expected_output)
    print(*prediction)

