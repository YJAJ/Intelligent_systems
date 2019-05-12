import numpy as np
from sklearn.utils import shuffle
import math
import sys
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input, output, test_input, n_input, n_hidden, n_output, epochs, bs, lr, bp, test_output=None):
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
        if test_output is not None:
            if not np.array_equal(output, test_output):
                self.pretest_y = [int(y) for y in test_output]
                self.test_y = self.one_hot_encoding(self.pretest_y)
            else:
                self.test_y = output
        else:
            self.test_y = None
        self.trn_accuracy = 0
        self.tst_accuracy = 0
        self.trn_accuracy_list = []0.09998587
        self.tst_accuracy_list = []

    #main function - forward implemented with matrices
    def forward(self, x):
        #hidden layer
        self.neth = np.dot(x, self.w1.T) + np.dot(self.b, self.b1.T)
        self.outh = self.sigmoid(self.neth)
        #output layer
        self.neto = np.dot(self.outh, self.w2.T) + np.dot(self.b, self.b2.T)
        self.y_hat = self.sigmoid(self.neto)

    #main function - back propagation implemented with matrices
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
        print(self.w1)
        self.w2 -= lr * d_w2.T
        print(self.w2)
        self.b1 -= lr * d_b1.T
        print(self.b1)
        self.b2 -= lr * d_b2.T
        print(self.b2)

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
        #initialise with mean 0 and std 1 multiplied by square root two divided by input size
        mean = 0.0
        std = 1.0
        self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))*math.sqrt(2./self.n_input)
        self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))*math.sqrt(2./self.n_hidden)
        self.b = np.ones((bs, 1))
        self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
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

    #quadratic loss function
    def quad_loss(self, y):
        #1/2 as part of matrix mean operation
        return np.square(self.y_hat - y).mean()

    #cross entropy loss
    def cross_entropy_loss(self, x, y):
        return -(y*np.log(self.y_hat)+(1.-y)*np.log(1.-self.y_hat)).mean()

    #to take smaller steps towards the optimum, learning rate is reduced every 10 epoch
    def learning_rate_decay(self, lr, epoch):
        if epoch == 10:
            return lr/3.
        else:
            return lr/10.

    #calculate accuracy based on total correct number of predictions
    def cal_accuracy_total(self, target, prediction):
        targ = np.argmax(target, axis=1)
        pred = np.argmax(prediction, axis=1)
        #check the dimensions
        assert targ.shape == pred.shape, "target dimension not equal to prediction dimension"
        total = np.count_nonzero(targ==pred)
        return (total/target.shape[0])

    #standardisation function for input and weights
    def standardise_data(self, data):
        return (data - data.mean())/data.std()

    #plot training and test accuracy comparison
    def plot_epoch_accuracy(self):
        epoch_list = range(1, self.epochs + 1)
        plt.plot(epoch_list, self.trn_accuracy_list, 'r-.')
        plt.plot(epoch_list, self.tst_accuracy_list, 'b-')
        plt.legend(['Training accuracy', 'Test accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
    #plot test accuracy for different learning rates or batch sizes
    def plot_experiments_accuracy(self):
        epoch_list = range(1, self.epochs + 1)
        graphs = ['r-', 'b-', 'c-', 'y-', 'm-']
        start = 0
        for i in range(5):
            test = self.tst_accuracy_list[start:start+self.epochs]
            plt.plot(epoch_list, test, graphs[i])
            start += self.epochs
        #for a learning rate plot, uncomment the following legend
        # plt.legend(['lr = 0.001', 'lr = 0.1', 'lr = 1', 'lr = 10','lr = 100'])
        plt.legend(['bs = 1', 'bs = 5', 'bs = 10', 'bs = 20', 'bs = 100'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def train(self, loss, init):
        np.seterr(over='ignore')
        lr = self.lr
        bs = self.bs
        #to run different learning rates and batch sizes uncomment the following for loop and change accordingly (e.g. for lr in self.lr)
        # for bs in self.bs:
        #-----indent line 167 to 205 to check different learning rates and batch sizes
        #three initialisation methods
        #to use random initialisation
        if init == "random":
            self.random_init(bs)
        #to use kaiming initialisastion - this generates the best results
        if init == "kaiming":
            self.kaiming_init(bs)
        #to check test results
        if init == "test":
            self.test_init()
        #run the number of epochs
        for epoch in range(self.epochs):
            print("epoch: %d" % epoch)
            self.y_hat_all = np.empty([self.y.shape[0], self.y.shape[1]])
            batch_no = math.ceil(self.input_len/bs)
            for batch in range(batch_no):
                x = self.x[batch*bs:(batch+1)*bs]
                # x = self.standardise_data(x)
                y = self.y[batch*bs:(batch+1)*bs]
                #do forward for a training set
                self.forward(x)
                self.y_hat_all[batch * bs:(batch + 1) * bs] = self.y_hat
                #use quadratic cost function
                if loss=="quad":
                    self.error = self.quad_loss(y)
                #use cross entropy cost function
                if loss=="cross_entropy":
                    self.error = self.cross_entropy_loss(x, y)
                #do back propagation for a training set
                if self.bp:
                    self.backward(x, y, lr)
            #calculate training set accuracy to plot them later
            self.trn_accuracy = self.cal_accuracy_total(self.y, self.y_hat_all)
            self.trn_accuracy_list.append(self.trn_accuracy)
            self.y_hat_test_all = np.empty([self.test_x.shape[0], self.n_output])
            #to pass a test set after each epoch training, run test() function
            # self.test(bs)
            # #caluclate test set accuracy to plot them later except when there is no label for a test set
            # if self.test_y is not None:
            #     self.tst_accuracy = self.cal_accuracy_total(self.test_y, self.y_hat_test_all)
            #     self.tst_accuracy_list.append(self.tst_accuracy)
            #     print("Accuracy for epoch %d is %f" % (epoch+1, self.tst_accuracy))
                #learning rate decay every 10 epoch
                # if epoch%10==0 and epoch!=0:
                #     lr = self.learning_rate_decay(lr, epoch)
    #uncomment the line below to experiment with different batch size
        # self.plot_experiments_accuracy()
    #     self.plot_epoch_accuracy()

    #essentially, similar to training function, but there is no backpropagation in this case
    def test(self, bs):
        batch_no = math.ceil(len(self.test_x) / bs)
        for batch in range(batch_no):
            x = self.test_x[batch * bs:(batch + 1) * bs]
            x = self.standardise_data(x)
            if self.test_y is not None:
                y = self.test_y[batch * bs:(batch + 1) * bs]
            self.forward(x)
            self.y_hat_test_all[batch * bs:(batch + 1) * bs] = self.y_hat

    def predict(self, filename):
        #to save the prediction
        pred_label = np.argmax(self.y_hat_test_all, axis=1)
        np.savetxt(filename, pred_label, delimiter=',')

#utility function to load the file
def load_data(file):
    data = np.loadtxt(file, delimiter=',')
    return data

#utility function to visualise a csv row into a digit
def visualise_individual_data(data):
    img_sz = 28
    img_index = 20
    image = data[img_index,:]
    im_sq = np.reshape(image, (img_sz, img_sz))
    plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
    plt.show()

#main function to run the whole training and test processes
if __name__=="__main__":
    NInput = 2#784#sys.argv[1]
    NHidden = 2#30#sys.argv[2]
    NOutput = 2#10#sys.argv[3]
    trn_x = os.path.join(os.getcwd(), "trainDigitX.csv.gz")#sys.argv[4]
    trn_y = os.path.join(os.getcwd(), "trainDigitY.csv.gz")#sys.argv[5]
    test_x = os.path.join(os.getcwd(), "testDigitX.csv.gz")#sys.argv[6]
    test_y = os.path.join(os.getcwd(), "testDigitY.csv.gz")
    predict_y = "PredictTestY.csv.gz" #sys.argv[7]
    test_x2 = os.path.join(os.getcwd(), "testDigitX2.csv.gz")
    epochs = 3
    #uncomment the list below to experiment different batch sizes
    bs = 2#[1, 5, 10, 20, 100]
    #uncomment the list below to experiment different learning rates
    lr = 0.1 #[0.001, 0.1, 1.0, 10, 100]

    #to see the results of Part A assignment, uncomment the following two lines and comment out from line 244 to 250
    parta_data = np.array([[0.1, 0.1],[0.1, 0.2]])
    parta_label = np.array([[1, 0],[0, 1]])
    # trn_data = np.array(load_data(trn_x))
    # #to visualise an example of data, uncomment the following
    # #visualise_individual_data(trn_data)
    # trn_label = np.array(load_data(trn_y))
    # trn_data, trn_label = shuffle(trn_data, trn_label)
    # test_data = np.array(load_data(test_x))
    # test_label = np.array(load_data(test_y))

    # nn = NeuralNetwork(trn_data, trn_label, test_data, NInput, NHidden, NOutput, epochs, bs, lr, True, test_label)
    #to see the results of Part A assignment, uncomment the following line
    nn = NeuralNetwork(parta_data, parta_label, parta_data, NInput, NHidden, NOutput, epochs, bs, lr, True, parta_label)
    #loss functions: quad or cross_entropy
    loss_func = "quad"
    #weights initialisation method: random (mean 0, std 1), kaiming, or test (for the results of Part A)
    init_method = "test"
    nn.train(loss_func, init_method)
    # nn.predict(predict_y)