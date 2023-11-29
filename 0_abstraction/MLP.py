import numpy as np
# import sys
from scipy.special import expit
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, in_l, hid_l, out_l, learning_rate, epoch=1):
        self.in_l = in_l
        self.hid_l = hid_l
        self.out_l = out_l
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.in_h_weights = np.random.normal(0.0, pow(self.in_l, -0.5), (self.hid_l, self.in_l))
        self.h_out_weights = np.random.normal(0.0, pow(self.hid_l, -0.5), (self.out_l, self.hid_l))

    def train(self, data_file):
        for _ in range(self.epoch):
            with open(data_file, 'r') as file:
                for i in file.readlines():
                    inp = np.array(i.replace('\n', '').split(','), ndmin=2, dtype=int)
                    out = np.zeros((10, 1))
                    out[inp[0, 0], 0] = 0.99
                    inp = inp[0, 1:] / 255 * 0.99 + 0.01
                    inp.resize((784, 1))
                    hidden_out = expit(np.dot(self.in_h_weights, inp))
                    calc_out = expit(np.dot(self.h_out_weights, hidden_out))

                    error = out - calc_out
                    hid_out_dir = -2 * np.dot(error * calc_out * (1 - calc_out), hidden_out.T)
                    inp_gid_dir = -2 * np.dot(np.dot(self.h_out_weights.T, error) * hidden_out * (1 - hidden_out),
                                              inp.T)

                    self.in_h_weights -= self.learning_rate * inp_gid_dir
                    self.h_out_weights -= self.learning_rate * hid_out_dir

    def predict(self, num):
        with open('mnist_test.csv', 'r') as file:
            inp = [i.replace('\n', '').split(',') for i in file.readlines()]
            inp = np.array(inp, dtype=int)
            inp = inp[:, 1:] / 255 * 0.99 + 0.01

        inp_hid = expit(np.dot(self.in_h_weights, inp[num, :].T))
        hid_out = expit(np.dot(self.h_out_weights, inp_hid))
        print(hid_out)
        plt.imshow(inp[num, :].reshape((28, 28)))
        plt.show()


mlp = MLP(784, 500, 10, 0.1)

mlp.train('mnist_train.csv')
mlp.predict(100)
mlp.predict(123)
mlp.predict(289)
mlp.predict(100)
