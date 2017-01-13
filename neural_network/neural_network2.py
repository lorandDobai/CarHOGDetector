import joblib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.utils
import math
from neural_network.learning_utils import normalize_data, one_hot_encode, inplace_logistic_sigmoidf64, \
    softmax, cross_entropy_loss, cross_entropy_gradient, inplace_tanhf64, logistic_derivative, tanh_derivative, \
    inplace_relu, relu_derivative

from neural_network.learning_utils import plot_decision_boundary

ACTIVATIONS = {'logistic': inplace_logistic_sigmoidf64, 'tanh':inplace_tanhf64,'relu':inplace_relu}
ACTIVATION_DERIVATIVES = {'logistic': logistic_derivative,'tanh':tanh_derivative,'relu':relu_derivative}


class SimpleMLPClassifier(object):
    def __init__(self, layers=None,  activation='logistic',  learning_rate=0.01,
                 training_epochs=1000, batch_size=1, regularization=0.0001, momentum_term=0.9):

        self.layers = layers
        self.depth = len(self.layers)
        self.learning_rate_init = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.momentum_term = momentum_term
        self.hidden_activation = ACTIVATIONS[activation]
        self.hidden_activation_derivative = ACTIVATION_DERIVATIVES[activation]
        self.output_activation = lambda arr:  np.apply_along_axis(softmax, axis=0, arr=arr).T
        self.weights = []
        self.biases = []
        for i in range(1, self.depth):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]) * math.sqrt(2/(self.layers[i]+self.layers[i-1])))
            self.biases.append(np.zeros((self.layers[i], 1), dtype=np.float64))

        self.error_ = 0

    def fit(self, data, classes):
        self.minibatch_fit(data, classes)

    def partial_fit(self, data, classes):
        self.minibatch_fit(data, classes)

    def minibatch_fit(self, data, classes):
        learning_rate = self.learning_rate_init
        batches = np.arange(0, len(data), self.batch_size)
        if batches[-1] != len(data):
            batches = np.hstack((batches, [len(data)]))

        for epoch in range(self.training_epochs):
            self.error_ = 0
            weight_momentums = [np.zeros_like(arr) for arr in self.weights]
            bias_momentums = [np.zeros_like(arr) for arr in self.biases]
            data,classes = sklearn.utils.shuffle(data,classes)
            for batch_index in range(len(batches) - 1):

                x = data[batches[batch_index]:batches[batch_index + 1]]
                y = classes[batches[batch_index]:batches[batch_index + 1]]

                """Forward Pass"""
                signal = x.T
                signals = [signal]
                for i in range(self.depth-2):
                    signal = self.weights[i].dot(signal) + self.biases[i]
                    self.hidden_activation(signal)
                    signals.append(signal)
                signal = self.weights[self.depth-2].dot(signal) + self.biases[self.depth-2]
                output = self.output_activation(signal)

                """Error Function"""
                self.error_ += self.__calculate_loss(output, y) + self.regularization / 2 * (np.sum((np.sum(np.square(w)) for w in self.weights)))

                """Backpropagation"""
                delta_grad_out = self.loss_gradient(output, y).T
                for i in range(self.depth-2, -1, -1):
                    delta_w = signals[i].dot(delta_grad_out.T).T + self.regularization * self.weights[i]
                    weight_momentums[i] = self.momentum_term * weight_momentums[i] - learning_rate * delta_w

                    delta_b = delta_grad_out.sum(axis=1, keepdims=True)
                    bias_momentums[i] = self.momentum_term * bias_momentums[i] - learning_rate * delta_b

                    delta_grad_out = self.hidden_activation_derivative(signals[i]) * delta_grad_out.T.dot(self.weights[i]).T

                for i in range(self.depth-1):
                    self.weights[i] += weight_momentums[i]
                    self.biases[i] += bias_momentums[i]

            self.error_ /= len(data)
            print("Error at epoch {}: {} ".format(epoch, self.error_))

    def __calculate_loss(self, decision, real):
        return cross_entropy_loss(decision, real)

    def loss_gradient(self, decision, real):
        return cross_entropy_gradient(decision, real)

    def decision_function(self, X):
        """Forward Pass"""
        signal = X.T
        for i in range(self.depth - 2):
            signal = self.weights[i].dot(signal) + self.biases[i]
            self.hidden_activation(signal)
        signal = self.weights[self.depth - 2].dot(signal) + self.biases[self.depth - 2]

        return self.output_activation(signal)

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


def test_2d_classification(model):
    data = joblib.load('points.data')
    X, Y = data[:, :-1], data[:, -1]
    X = normalize_data(X)
    Y = one_hot_encode(Y)
    X_train, Y_train = X[:-10], Y[:-10]
    X_validation, Y_validation = X[-10:], Y[-10:]

    model.fit(X_train, Y_train)
    for x, y in zip(X_validation, Y_validation):
        print(model.decision_function(x), y)
    plot_decision_boundary(lambda x: np.argmax(model.decision_function(x)), X, Y)


def test_2d_moons_classification(model):
    data = joblib.load('moons.data')
    X, Y = data[:, :-1], data[:, -1]
    X = normalize_data(X)
    Y = one_hot_encode(Y)
    X_train, Y_train = X[:-10], Y[:-10]
    X_validation, Y_validation = X[-10:], Y[-10:]

    model.fit(X_train, Y_train)

    print(model.decision_function(X_validation), Y_validation)
    plot_decision_boundary(lambda x: np.argmax(model.decision_function(x), axis=1), X, Y)


if __name__ == '__main__':

    # model = SimpleMLPClassifier(input_size=2, learning_rate=0.02, hidden_layer_size=8, regularization=0.00001,
    #                             batch_size=16, training_epochs=1000)

    model = SimpleMLPClassifier(layers=(2, 3, 2), learning_rate=0.03, activation='tanh', regularization=0.000001,
                                batch_size=16, training_epochs=200)
    test_2d_moons_classification(model)
    print()
