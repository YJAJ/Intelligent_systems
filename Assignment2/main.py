import numpy as np

class NeuralNetwork:
    def __init__(self, input, output):
        self.x = input
        self.w1 = np.random.rand(self.x.shape[1], 4)
        self.w2 = np.random.rand(4, 1)
        self.y = output
        self.y_hat = np.zeros(output.shape)

    def feedforward(self):
        self.h1 = sigmoid(np.dot(self.x, self.w1))
        self.y_hat = sigmoid(np.dot(self.h1, self.w2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to w2 and w11
        d_w2 = np.dot(self.h1.T, (2 * (self.y - self.y_hat) * sigmoid_derivative(self.y_hat)))
        d_w1 = np.dot(self.x.T, (np.dot(2 * (self.y - self.y_hat) * sigmoid_derivative(self.y_hat), self.w2.T) *
                sigmoid_derivative(self.h1)))

        # update the weights with the derivative (slope) of the loss function
        self.w1 += d_w1
        self.w2 += d_w2