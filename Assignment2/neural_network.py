import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input, output, n_input, n_hidden, n_output, epochs, bs, lr, activation, bp):
        self.input_len = len(input)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.bp = bp
        self.x = input
        self.pre_y = [int(y) for y in output]
        self.y = self.one_hot_encoding()
        self.act = activation
        #save mean and std of train set so that we can use it for test set

    def forward(self, x):
        #hidden layer
        self.neth = np.dot(x, self.w1.T) + np.dot(self.b, self.b1.T)
        self.outh = self.sigmoid(self.neth)
        #output layer
        self.neto = np.dot(self.outh, self.w2.T) + np.dot(self.b, self.b2.T)
        self.y_hat = self.sigmoid(self.neto)

    def backward(self, x, y):
        #last batch may have lesser samples than the specified batch size
        d_w2 = np.dot(self.outh.T, ((1/x.shape[0]) * (self.y_hat - y) * self.d_sigmoid(self.y_hat)))
        #print(d_w2.shape)
        d_w1 = np.dot(x.T, (np.dot((1/x.shape[0])  * (self.y_hat - y) * self.d_sigmoid(self.y_hat), self.w2) *
                 self.d_sigmoid(self.outh)))
        #print(d_w1.shape)
        #update the weights with a learning rate
        self.w1 -= self.lr * d_w1.T
        self.w2 -= self.lr * d_w2.T
        # print(self.w2)
        # print(self.w1)

    #test initialisation per assignment instruction
    def test_init(self):
        self.w1 = np.array([[0.1, 0.1],[0.2, 0.1]])
        self.w2 = np.array([[0.1, 0.1],[0.1, 0.2]])
        self.b = np.array([[1],[1]])
        self.b1 = np.array([[0.1],[0.1]])
        self.b2 = np.array([[0.1],[0.1]])

    #random initialisation with mean 0. and std 1.
    def random_init(self):
        #make sure self.w.T first dimension aligns with self.x second dimension
        assert self.n_input == self.x.shape[1], "input numbers not equal to matrix dimension"
        mean = 0.0
        std = 1.0
        self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))
        #print(self.w1.mean(), self.w1.std())
        self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))
        #print(self.w2.mean(), self.w2.std())
        self.b = np.ones((self.bs, 1))
        self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
        self.b2 = np.random.normal(mean, std, (self.n_output, 1))

    #kaiming initialisation per https://arxiv.org/abs/1502.01852
    def kaiming_init(self):
        mean = 0.0
        std = 1.0
        self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))*math.sqrt(2./self.n_input)
        #print(self.w1.mean(), self.w1.std())
        self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))*math.sqrt(2./self.n_hidden)
        #print(self.w2.mean(), self.w2.std())
        self.b = np.ones((self.bs, 1))
        self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
        #print(self.w1.mean(), self.w1.std())
        self.b2 = np.random.normal(mean, std, (self.n_output, 1))

    #vectorisation of ground truth label
    def one_hot_encoding(self):
        n_classes = 10
        return np.eye(n_classes)[self.pre_y]

    #sigmoid activation
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    #sigmoid derivative
    def d_sigmoid(self, out):
        return np.multiply(out, (1 - out))

    #relu activation
    def relu(self, x):
        #only x>0 gets passed
        return abs(x)*(x>0)
    #relu derivative
    def d_relu(self, out):
        return 1*(out>0)

    #quadratic loss function i.e. mse
    def quad_loss(self, y):
        return np.square(self.y_hat - y).mean()

    #
    def log_softmax(self):
        return (self.y_hat.exp() / (self.y_hat.exp().sum(-1, keepdim=True))).log()

    #cross entropy loss
    def cross_entropy_loss(self, log_softmax):
        return -log_softmax[range(self.y.shape[0]), self.y].mean()

    def standardise_data(self, data):
        return (data - data.mean())/data.std()

    def train(self):
        #to use random initialisation, uncomment this
        # self.random_init()
        #to use kaiming initialisastion, uncomment this - this generates the best results
        self.kaiming_init()
        #to check test results, uncomment this
        # self.test_init()
        for epoch in range(self.epochs):
            batch_no = math.ceil(self.input_len/self.bs)
            for batch in range(batch_no):
                x = self.x[batch*self.bs:(batch+1)*self.bs]
                x = self.standardise_data(x)
                y = self.y[batch*self.bs:(batch+1)*self.bs]
                self.forward(x)
                self.error = self.quad_loss(y)
                #if batch%100==0:
                #print("Error for batch %d is %f" % (batch, self.error))
                if self.bp:
                    self.backward(x, y)
            print("Error for epoch %d is %f" % (epoch, self.error))

def load_data(file):
    data = np.loadtxt(file, delimiter=',')
    return data

def visualise_individual_data(data):
    img_sz = 28
    img_index = 20
    image = data[img_index,:]
    im_sq = np.reshape(image, (img_sz, img_sz))
    plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
    plt.show()

if __name__=="__main__":
    NInput = 784#sys.argv[1]
    NHidden = 30#sys.argv[2]
    NOutput = 10#sys.argv[3]
    trn_x = os.path.join(os.getcwd(), "trainDigitX.csv.gz")#sys.argv[4]
    trn_y = os.path.join(os.getcwd(), "trainDigitY.csv.gz")#sys.argv[5]
    test_x = os.path.join(os.getcwd(), "testDigitX.csv.gz")#sys.argv[6]
    test_y = os.path.join(os.getcwd(), "testDigitY.csv.gz")#sys.argv[7]
    epochs = 30
    bs = 20
    lr = 3.0
    activation = "sigmoid"
    trn_data = np.array(load_data(trn_x))
    trn_label = np.array(load_data(trn_y))
    # test_data = load_data(test_x)
    # test_label = load_data(test_y)
    nn = NeuralNetwork(trn_data, trn_label,
                       NInput, NHidden, NOutput, epochs, bs, lr, activation, True)
    # nn = NeuralNetwork(np.array([[0.1, 0.1],[0.1, 0.2]]),np.array([[1, 0],[0, 1]]), 2, 2, 2, 1, 2, 0.1, True)
    nn.train()