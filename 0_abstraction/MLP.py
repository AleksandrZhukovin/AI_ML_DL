import numpy as np
import sys
from scipy.special import expit
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, in_l, hid_l, out_l, learning_rate, epoch=None):
        self.in_l = in_l
        self.hid_l = hid_l
        self.out_l = out_l
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.out = None
        self.inp = None
        self.calc_out = None
        self.hidden_out = None
        self.in_h_weights = None
        self.h_out_weights = None

    def forward(self, data_file):
        with open(data_file, 'r') as file:
            inp = [i.replace('\n', '').split(',') for i in file.readlines()]
            inp = np.array(inp, dtype=int)
            self.out = inp[:, 0]
            self.inp = inp[:, 1:] / 255 * 0.99 + 0.01
        self.in_h_weights = np.random.normal(0.0, pow(self.in_l, -0.5), (self.hid_l, self.in_l))
        self.hidden_out = expit(np.dot(self.in_h_weights, self.inp.T))
        self.h_out_weights = np.random.normal(0.0, pow(self.hid_l, -0.5), (self.out_l, self.hid_l))
        self.calc_out = expit(np.dot(self.h_out_weights, self.hidden_out))
        # print(hidden_out)
        # print(calc_out.shape)

    def backward(self):
        out = np.zeros((self.out.shape[0], 10))
        for i in range(self.out.shape[0]):
            out[i, self.out[i]] = 0.99
        self.out = out

        error = (self.out.T - self.calc_out)   # 10 x 60 000

        print(np.dot(error * self.calc_out * (1 - self.calc_out), self.hidden_out.T))
        hid_out_dir = 2 * np.dot((self.out.T - self.calc_out) ** 2 * self.calc_out * (1 - self.calc_out), self.hidden_out.T)
        inp_gid_dir = 2 * np.dot(np.dot(self.h_out_weights.T, error) * self.hidden_out * (1 - self.hidden_out), self.inp)
        # print(hid_out_dir, inp_gid_dir)
        self.in_h_weights -= self.learning_rate * inp_gid_dir
        self.h_out_weights -= self.learning_rate * hid_out_dir

    def predict(self, num):
        with open('mnist_test.csv', 'r') as file:
            inp = [i.replace('\n', '').split(',') for i in file.readlines()]
            inp = np.array(inp, dtype=int)
            inp = inp[:, 1:] / 255 * 0.99 + 0.01
        print(inp)
        print(self.h_out_weights)
        print(self.in_h_weights)
        inp_hid = expit(np.dot(self.in_h_weights, inp[num, :].T))
        hid_out = expit(np.dot(self.h_out_weights, inp_hid))
        print(hid_out)
        plt.imshow(inp[num, :].reshape((28, 28)))
        plt.show()


mlp = MLP(784, 500, 10, 0.1)

mlp.forward('mnist_train.csv')
mlp.backward()
# mlp.predict(100)
