import numpy as np
from sklearn.utils import shuffle
import math
import sys
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_in, output, test_input, n_input, n_hidden, n_output, epochs, bs, lr, bp, test_output=None):
        self.input_len = len(input_in)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.bp = bp
        self.x = input_in
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
        self.trn_accuracy_list = []
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
        self.w2 -= lr * d_w2.T
        self.b1 -= lr * d_b1.T
        self.b2 -= lr * d_b2.T

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

    def train(self, loss, init):
        np.seterr(over='ignore')
        #three initialisation methods
        #to use random initialisation
        if init == "random":
            self.random_init(self.bs)
        #to use kaiming initialisastion - this generates the best results
        if init == "kaiming":
            self.kaiming_init(self.bs)
        #run the number of epochs
        for epoch in range(self.epochs):
            self.y_hat_all = np.empty([self.y.shape[0], self.y.shape[1]])
            batch_no = math.ceil(self.input_len/self.bs)
            for batch in range(batch_no):
                x = self.x[batch*self.bs:(batch+1)*self.bs]
                x = self.standardise_data(x)
                y = self.y[batch*self.bs:(batch+1)*self.bs]
                #do forward for a training set
                self.forward(x)
                self.y_hat_all[batch * self.bs:(batch + 1) * self.bs] = self.y_hat
                #use quadratic cost function
                if loss=="quad":
                    self.error = self.quad_loss(y)
                #use cross entropy cost function
                if loss=="cross_entropy":
                    self.error = self.cross_entropy_loss(x, y)
                #do back propagation for a training set
                if self.bp:
                    self.backward(x, y, self.lr)
            #calculate training set accuracy to plot them later
            self.trn_accuracy = self.cal_accuracy_total(self.y, self.y_hat_all)
            self.trn_accuracy_list.append(self.trn_accuracy)
            self.y_hat_test_all = np.empty([self.test_x.shape[0], self.n_output])
            #to pass a test set after each epoch training, run test() function
            self.test(self.bs)
            #caluclate test set accuracy to plot them later except when there is no label for a test set
            if self.test_y is not None:
                self.tst_accuracy = self.cal_accuracy_total(self.test_y, self.y_hat_test_all)
                self.tst_accuracy_list.append(self.tst_accuracy)
                print("Accuracy for epoch %d is %f" % (epoch+1, self.tst_accuracy))
            #uncomment two lines below if using learning rate decay every 10 epoch
            # if epoch%10==0 and epoch!=0:
            #     lr = self.learning_rate_decay(lr, epoch)
        return self.trn_accuracy_list, self.tst_accuracy_list

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

#utility function to plot training and test accuracy comparison
def plot_epoch_accuracy(epochs, trn_accuracy_list, tst_accuracy_list):
    epoch_list = range(1, epochs + 1)
    plt.plot(epoch_list, trn_accuracy_list, 'r-.')
    plt.plot(epoch_list, tst_accuracy_list, 'b-')
    plt.legend(['Training accuracy', 'Test accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

#utility function to plot test accuracy for different learning rates or batch sizes
def plot_experiments_accuracy(epochs, tst_accuracy_list, legend):
    epoch_list = range(1, epochs + 1)
    graphs = ['r-', 'b-', 'c-', 'y-', 'm-']
    for i in range(5):
        test = tst_accuracy_list[i]
        plt.plot(epoch_list, test, graphs[i])
    # for a learning rate plot, uncomment the following legend
    if legend == "bs":
        plt.legend(['bs = 1', 'bs = 5', 'bs = 10', 'bs = 20', 'bs = 100'])
    if legend == "lr":
        plt.legend(['lr = 0.001', 'lr = 0.1', 'lr = 1', 'lr = 10', 'lr = 100'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

#main function to run the whole training and test processes
if __name__=="__main__":
    #robust input process
    try:
        NInput = int(sys.argv[1])
        assert NInput > 0
    except:
        print("NInput must be a positive integer greater than zero.")
        sys.exit()
    try:
        NHidden = int(sys.argv[2])
        assert NHidden > 0
    except:
        print("NHidden must be a positive integer greater than zero.")
        sys.exit()
    try:
        NOutput = int(sys.argv[3])
        assert NOutput > 0
    except:
        print("NOutput must be a positive integer greater than zero.")
        sys.exit()
    try:
        trn_x = os.path.join(os.getcwd(), sys.argv[4])
        assert os.path.isfile(trn_x)
    except:
        print("File %s does not exist." % sys.argv[4])
        sys.exit()
    try:
        trn_y = os.path.join(os.getcwd(), sys.argv[5])
        assert os.path.isfile(trn_y)
    except:
        print("File %s does not exist." % sys.argv[5])
        sys.exit()
    try:
        test_x = os.path.join(os.getcwd(), sys.argv[6])
        assert os.path.isfile(test_x)
    except:
        print("File %s does not exist." % sys.argv[6])
        sys.exit()

    test_y = os.path.join(os.getcwd(), "TestDigitY.csv.gz")
    predict_y = sys.argv[7]
    epochs = 30
    #uncomment and change the list below to experiment different batch sizes
    bs = 20#[1, 5, 10, 20, 100]
    #uncomment and change the list below to experiment different learning rates
    lr = 3.0#[0.001, 0.1, 1.0, 10, 100]

    trn_data = np.array(load_data(trn_x))
    #to visualise an example of data, uncomment the following
    #visualise_individual_data(trn_data)
    trn_label = np.array(load_data(trn_y))
    trn_data, trn_label = shuffle(trn_data, trn_label)
    test_data = np.array(load_data(test_x))
    if sys.argv[6]=="TestDigitX.csv.gz":
        test_label = np.array(load_data(test_y))
    else:
        test_label = None

    #loss functions: quad or cross_entropy
    loss_func = "cross_entropy"
    #weights initialisation method: random (mean 0, std 1) or kaiming
    init_method = "random"
    #test accuracy list for multiple bs or lr
    accuracy_list = []
    if type(bs) is list:
        for batch_size in bs:
            #initialise nn
            nn = NeuralNetwork(trn_data, trn_label, test_data, NInput, NHidden, NOutput, epochs, batch_size, lr, True,
                               test_label)
            #train the defined number of epochs
            train_accuracy_list, tst_accuracy_list = nn.train(loss_func, init_method)
            #predict and generate output file
            nn.predict(predict_y)
            accuracy_list.append(tst_accuracy_list)
        plot_experiments_accuracy(epochs, accuracy_list, "bs")
    elif type(lr) is list:
        for learning_rate in lr:
            #initialise nn
            nn = NeuralNetwork(trn_data, trn_label, test_data, NInput, NHidden, NOutput, epochs, bs, learning_rate, True,
                               test_label)
            #train the defined number of epochs
            train_accuracy_list, tst_accuracy_list = nn.train(loss_func, init_method)
            #predict and generate output file
            nn.predict(predict_y)
            accuracy_list.append(tst_accuracy_list)
        plot_experiments_accuracy(epochs, accuracy_list, "lr")
    else:
        #initialise nn
        nn = NeuralNetwork(trn_data, trn_label, test_data, NInput, NHidden, NOutput, epochs, bs, lr, True,
                           test_label)
        #train the defined number of epochs
        train_accuracy_list, tst_accuracy_list = nn.train(loss_func, init_method)
        #predict and generate output file
        nn.predict(predict_y)
        #plot the accuracy over epoch - train vs test
        plot_epoch_accuracy(epochs, train_accuracy_list, tst_accuracy_list)