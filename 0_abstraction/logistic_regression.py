import numpy as np
from scipy.special import expit


class LogisticRegression:
    def __init__(self, input_n, output, learning_rate):
        self.input = input_n
        self.output = output
        self.weights = np.random.normal(0.0, pow(self.input, -0.5), (self.output, self.input))
        self.learning_rate = learning_rate

    def train(self, data, label):
        input_data = np.array(data, ndmin=2).T

        output = np.array([0])
        if label == 1:
            output[0] = 0.99
        else:
            output[0] = 0

        res = np.dot(self.weights, input_data)
        res = expit(res)

        error = label - res
        self.weights += self.learning_rate * error * expit(res) * (1 - expit(res)) * input_data.T

    def feed(self, data):
        input_data = np.array(data, ndmin=2).T
        res = np.dot(self.weights, input_data)
        res = expit(res)
        return res


nn = LogisticRegression(28*28, 1, 0.1)


with open('../datasets/mnist_train.csv', 'r') as file:
    for i in file.readlines():
        if int(i.split(',')[0]) == 1 or int(i.split(',')[0]) == 0:
            data = np.fromstring(i, dtype=float, sep=',')
            label = int(i.split(',')[0])
            data = np.delete(data, [0]) / 255.0 * 0.99 + 0.01
            nn.train(data, label)

total = 0
correct = 0
with open('../datasets/mnist_test.csv', 'r') as file:
    l = file.readlines()
    for i in l:
        if int(i.split(',')[0]) == 1 or int(i.split(',')[0]) == 0:
            total += 1
            data = np.fromstring(i, dtype=float, sep=',')
            label = int(i.split(',')[0])
            data = np.delete(data, [0])
            data = data / 255.0 * 0.99 + 0.01
            prediction = nn.feed(data)[0, 0]
            print(f'label: {label}')
            print(f'prediction: {prediction}')
            print("---------------")
            if (label == 1 and prediction > 0.7) or (label == 0 and prediction < 0.3):
                correct += 1
print(f'{correct} of {total}\nefficiency: {round(correct/total*100, 2)}%')

