import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
import sys
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input, output, test_input, test_output, n_input, n_hidden, n_output, epochs, bs, lr, bp):
        self.input_len = len(input)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.bp = bp
        self.x = input
        if not np.array_equal(output,test_output):
            self.pre_y = [int(y) for y in output]
            self.y = self.one_hot_encoding(self.pre_y)
        else:
            self.y = output
        self.test_x = test_input
        if not np.array_equal(output, test_output):
            self.pretest_y = [int(y) for y in test_output]
            self.test_y = self.one_hot_encoding(self.pretest_y)
        else:
            self.test_y = output
        self.trn_accuracy = 0
        self.tst_accuracy = 0
        self.trn_accuracy_list = []
        self.tst_accuracy_list = []

    def forward(self, x):
        #hidden layer
        self.neth = np.dot(x, self.w1.T) + np.dot(self.b, self.b1.T)
        self.outh = self.sigmoid(self.neth)
        #print(self.outh)
        #output layer
        self.neto = np.dot(self.outh, self.w2.T) + np.dot(self.b, self.b2.T)
        self.y_hat = self.sigmoid(self.neto)

    def backward(self, x, y, lr):
        #last batch may have lesser samples than the specified batch size
        d_w2 = np.dot(self.outh.T, ((1/x.shape[0]) * (self.y_hat - y) * self.d_sigmoid(self.y_hat)))
        d_b2 = np.dot(self.b.T, ((1/x.shape[0]) * (self.y_hat - y) * self.d_sigmoid(self.y_hat)))
        d_w1 = np.dot(x.T, (np.dot((1/x.shape[0])  * (self.y_hat - y) * self.d_sigmoid(self.y_hat), self.w2) *
                 self.d_sigmoid(self.outh)))
        d_b1 = np.dot(self.b.T, (np.dot((1/x.shape[0])  * (self.y_hat - y) * self.d_sigmoid(self.y_hat), self.w2) *
                 self.d_sigmoid(self.outh)))
        #update the weights with a learning rate
        self.w1 -= lr * d_w1.T
        self.w2 -= lr * d_w2.T
        self.b1 -= lr * d_b1.T
        self.b2 -= lr * d_b2.T

    #test initialisation per assignment instruction
    def test_init(self):
        self.w1 = np.array([[0.1, 0.1],[0.2, 0.1]])
        self.w2 = np.array([[0.1, 0.1],[0.1, 0.2]])
        self.b = np.array([[1],[1]])
        self.b1 = np.array([[0.1],[0.1]])
        self.b2 = np.array([[0.1],[0.1]])

    #random initialisation with mean 0. and std 1.
    def random_init(self, bs):
        #make sure self.w.T first dimension aligns with self.x second dimension
        assert self.n_input == self.x.shape[1], "input numbers not equal to matrix dimension"
        mean = 0.0
        std = 1.0
        self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))
        self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))
        self.b = np.ones((bs, 1))
        self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
        self.b2 = np.random.normal(mean, std, (self.n_output, 1))

    #kaiming initialisation per https://arxiv.org/abs/1502.01852
    def kaiming_init(self, bs):
        mean = 0.0
        std = 1.0
        self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))*math.sqrt(2./self.n_input)
        #print(self.w1.mean(), self.w1.std())
        self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))*math.sqrt(2./self.n_hidden)
        #print(self.w2.mean(), self.w2.std())
        self.b = np.ones((bs, 1))
        self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
        #print(self.w1.mean(), self.w1.std())
        self.b2 = np.random.normal(mean, std, (self.n_output, 1))

    #vectorisation of ground truth label
    def one_hot_encoding(self, input):
        n_classes = np.max(input)+1
        return np.eye(n_classes)[input]

    #sigmoid activation
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    #sigmoid derivative
    def d_sigmoid(self, out):
        return np.multiply(out, (1 - out))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def d_tanh(self, out):
        return 1 - np.power(out, 2)

    #relu activation
    def relu(self, x):
        #only x>0 gets passed
        return np.maximum(0,x)

    #relu derivative
    def d_relu(self, out):
        return (out > 0) * 1

    #quadratic loss function i.e. mse
    def quad_loss(self, y):
        #print(np.square(self.y_hat - y))
        return np.square(self.y_hat - y).mean()

    #cross entropy loss
    def cross_entropy_loss(self, x, y):
        return -(y*np.log(self.y_hat)+(1.-y)*np.log(1.-self.y_hat)).mean()

    def learning_rate_decay(self, lr, epoch):
        if epoch == 10:
            return lr/3.
        if epoch == 20:
            return lr/10.
        else:
            return lr/10.

    # def cal_accuracy(self, x, y):
    #     targ = np.argmax(y, axis=1)
    #     #print(targ)
    #     pred = np.argmax(self.y_hat, axis=1)
    #     #print(pred)
    #     #check the dimensions
    #     assert targ.shape == pred.shape, "target dimension not equal to prediction dimension"
    #     total = np.count_nonzero(targ==pred)
    #     #print(total)
    #     return (total/x.shape[0])

    def cal_accuracy_total(self, target, prediction):
        targ = np.argmax(target, axis=1)
        #print(targ)
        pred = np.argmax(prediction, axis=1)
        #print(pred)
        #check the dimensions
        assert targ.shape == pred.shape, "target dimension not equal to prediction dimension"
        total = np.count_nonzero(targ==pred)
        #print(total)
        return (total/target.shape[0])

    def standardise_data(self, data):
        return (data - data.mean())/data.std()

    # def normalise_batch(self, output):
    #     sub_mean = output - output.mean()
    #     output_hat = sub_mean/(np.square(sub_mean).mean()+1e-5)
    #     out = gamma * output_hat + beta

    def save_parameters(self):
        np.savetxt('w1.csv.gz', self.w1, delimiter=',')
        np.savetxt('w2.csv.gz', self.w2, delimiter=',')
        np.savetxt('b1.csv.gz', self.b1, delimiter=',')
        np.savetxt('b2.csv.gz', self.b2, delimiter=',')

    def plot_epoch_accuracy(self):
        epoch_list = range(1, self.epochs + 1)
        plt.plot(epoch_list, self.trn_accuracy_list, 'r-.')
        plt.plot(epoch_list, self.tst_accuracy_list, 'b-')
        plt.legend(['Training accuracy', 'Test accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def plot_experiments_accuracy(self):
        epoch_list = range(1, self.epochs + 1)
        graphs = ['r-', 'b-', 'c-', 'y-', 'm-']
        start = 0
        for i in range(5):
            test = self.tst_accuracy_list[start:start+self.epochs]
            plt.plot(epoch_list, test, graphs[i])
            start += self.epochs
        # plt.legend(['lr = 0.001', 'lr = 0.1', 'lr = 1', 'lr = 10','lr = 100'])
        plt.legend(['bs = 1', 'bs = 5', 'bs = 10', 'bs = 20', 'bs = 100'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def train(self, loss, init):
        np.seterr(over='ignore')
        lr = self.lr
        bs = self.bs
        # for bs in self.bs:
        #to use random initialisation
        if init == "random":
            self.random_init(bs)
        #to use kaiming initialisastion - this generates the best results
        if init == "kaiming":
            self.kaiming_init(bs)
        #to check test results
        if init == "test":
            self.test_init()
        for epoch in range(self.epochs):
            self.y_hat_all = np.empty([self.y.shape[0], self.y.shape[1]])
            batch_no = math.ceil(self.input_len/bs)
            for batch in range(batch_no):
                x = self.x[batch*bs:(batch+1)*bs]
                x = self.standardise_data(x)
                y = self.y[batch*bs:(batch+1)*bs]
                self.forward(x)
                self.y_hat_all[batch * bs:(batch + 1) * bs] = self.y_hat
                # score = self.cal_accuracy(x, y)
                # self.trn_accuracy += score
                if loss=="quad":
                    self.error = self.quad_loss(y)
                if loss=="cross_entropy":
                    self.error = self.cross_entropy_loss(x, y)
                if self.bp:
                    self.backward(x, y, lr)
            self.trn_accuracy = self.cal_accuracy_total(self.y, self.y_hat_all)
            self.trn_accuracy_list.append(self.trn_accuracy)
            self.y_hat_test_all = np.empty([self.test_y.shape[0], self.test_y.shape[1]])
            self.test(bs)
            self.tst_accuracy = self.cal_accuracy_total(self.test_y, self.y_hat_test_all)
            self.tst_accuracy_list.append(self.tst_accuracy)
            if epoch%10==0 and epoch!=0:
                lr = self.learning_rate_decay(lr, epoch)
            print("Accuracy for epoch %d is %f" % (epoch+1, self.tst_accuracy))
            # self.save_parameters()
        #self.tst_accuracy_lists.append(self.tst_accuracy_list)
    #uncomment the line below to experiment with different batch size
    # self.plot_experiments_accuracy()
        self.plot_epoch_accuracy()

    def test(self, bs):
        batch_no = math.ceil(len(self.test_x) / bs)
        for batch in range(batch_no):
            x = self.test_x[batch * bs:(batch + 1) * bs]
            x = self.standardise_data(x)
            y = self.test_y[batch * bs:(batch + 1) * bs]
            self.forward(x)
            self.y_hat_test_all[batch * bs:(batch + 1) * bs] = self.y_hat
            #score = self.cal_accuracy(x, y)
        #     score = accuracy_score(np.argmax(y, axis=1), np.argmax(self.y_hat, axis=1))
        #     self.tst_accuracy += score
        # self.tst_accuracy = self.tst_accuracy/batch_no

    def pred(self, bs):
        self.w1 = os.path.join(os.getcwd(), "w1.csv.gz")
        self.w2 = os.path.join(os.getcwd(), "w2.csv.gz")
        self.b1 = os.path.join(os.getcwd(), "b1.csv.gz")
        self.b2 = os.path.join(os.getcwd(), "b2.csv.gz")
        batch_no = math.ceil(len(self.test_x) / bs)
        for batch in range(batch_no):
            x = self.test_x[batch * bs:(batch + 1) * bs]
            x = self.standardise_data(x)
            self.forward(x)
            np.savetxt('PredictTestY2.csv.gz', self.y_hat, delimiter=',')

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
    NHidden = 90#sys.argv[2]
    NOutput = 10#sys.argv[3]
    trn_x = os.path.join(os.getcwd(), "trainDigitX.csv.gz")#sys.argv[4]
    trn_y = os.path.join(os.getcwd(), "trainDigitY.csv.gz")#sys.argv[5]
    test_x = os.path.join(os.getcwd(), "testDigitX.csv.gz")#sys.argv[6]
    test_y = os.path.join(os.getcwd(), "testDigitY.csv.gz")#sys.argv[7]
    test_x2 = os.path.join(os.getcwd(), "testDigitX2.csv.gz")#sys.argv[6]
    epochs = 60
    bs = 20#[1, 5, 10, 20, 100]
    lr = 1.0#[0.001, 0.1, 1.0, 10, 100]
    # parta_data = np.array([[0.1, 0.1],[0.1, 0.2]])
    # parta_label = np.array([[1, 0],[0, 1]])
    trn_data = np.array(load_data(trn_x))
    #to visualise an example of data, uncomment the following
    #visualise_individual_data(trn_data)
    trn_label = np.array(load_data(trn_y))
    trn_data, trn_label = shuffle(trn_data, trn_label)
    test_data = np.array(load_data(test_x))
    test_label = np.array(load_data(test_y))
    nn = NeuralNetwork(trn_data, trn_label, test_data, test_label,
                       NInput, NHidden, NOutput, epochs, bs, lr, True)
    # nn = NeuralNetwork(parta_data, parta_label, parta_data, parta_label,
    #                    NInput, NHidden, NOutput, epochs, bs, lr, True)
    loss_func = "cross_entropy"
    init_method = "kaiming"
    nn.train(loss_func, init_method)