import numpy as np

class NeuralNetwork:
    def __init__(self, input, output, lr):
        self.x = input
        self.bs = input.shape[0]
        self.b = np.array([[1],[1]])
        self.b1 = np.array([[0.1],[0.1]])
        self.b2 = np.array([[0.1],[0.1]])
        self.w1 = np.array([[0.1, 0.1],[0.2, 0.1]])
        self.w2 = np.array([[0.1, 0.1],[0.1, 0.2]])
        self.w1b = np.array([[0.1, 0.1],[0.2, 0.1]])
        self.w2b = np.array([[0.1, 0.1],[0.1, 0.2]])
        self.y = output
        self.y_hat = np.zeros(output.shape)
        self.lr = lr
        self.error = 0

    def forward(self):
        #hidden layer
        self.neth = np.dot(self.x, self.w1.T) + np.dot(self.b, self.b1.T)
        #print(self.neth)
        self.outh = self.sigmoid(self.neth)
        #print(self.outh)
        #output layer
        self.neto = np.dot(self.outh, self.w2.T) + np.dot(self.b, self.b2.T)
        #print(self.neto)
        self.y_hat = self.sigmoid(self.neto)
        #print(self.y_hat)

    def backward(self):
        self.error = self.quad_loss()
        print(((1/self.bs) * (self.y_hat - self.y) * self.d_sigmoid(self.y_hat)))
        d_w2 = np.dot(self.outh.T, ((1/self.bs) * (self.y_hat - self.y) * self.d_sigmoid(self.y_hat)))
        print(d_w2)
        d_w1 = np.dot(self.x.T, (np.dot((1/self.bs)  * (self.y_hat - self.y) * self.d_sigmoid(self.y_hat), self.w2.T) *
                 self.d_sigmoid(self.outh)))
        #update the weights with a learning rate
        self.w1b -= self.lr * d_w1
        self.w2b -= self.lr * d_w2
        print(self.w1b, self.w2b)

    def random_init(self):
        # np.random.rand(self.x.shape[1], 4)
        # np.random.rand(4, 1)
        return

    #kaiming initialisation according to https://arxiv.org/abs/1502.01852
    def kaiming_init(self):
        return

    #sigmoid activation
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    #sigmoid derivative
    def d_sigmoid(self, net):
        return np.multiply(net, (1 - net))

    #relu activation
    def relu(self):
        return

    #mean squared error
    def quad_loss(self):
        return np.square(self.y_hat - self.y).mean()

    #
    def log_softmax(self):
        return (self.y_hat.exp() / (self.y_hat.exp().sum(-1, keepdim=True))).log()

    #cross entropy loss
    def cross_entropy_loss(self, log_softmax):
        return -log_softmax[range(self.y.shape[0]), self.y].mean()

if __name__=="__main__":
    lr = 0.1
    nn = NeuralNetwork(np.array([[0.1, 0.1],[0.1, 0.2]]),np.array([[1, 0],[0, 1]]), lr)
    nn.forward()
    nn.backward()