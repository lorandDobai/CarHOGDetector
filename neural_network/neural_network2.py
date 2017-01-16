import joblib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.utils
import math
from neural_network.learning_utils import normalize_data, one_hot_encode, ACTIVATIONS, ACTIVATION_DERIVATIVES,\
cross_entropy_loss,cross_entropy_gradient,softmax

from neural_network.learning_utils import plot_decision_boundary




class MLPClassifier(object):
    """
    Implements a general purpose MLP ANN Neural Network with:
        - customizable number of layers, and their dimensions
        - training done using Mini-batch Gradient Descent through backpropagation.
        - categorical cross entropy loss defined as the error.
        - logistic sigmoid, tanh, or relu activation functions
        - softmax activation on the output layer.
    """
    def __init__(self, layers=None,  activation='logistic',  learning_rate=0.01,
                 training_epochs=1000, batch_size=1, regularization=0.0001, momentum_term=0.9):
        """

        :param layers: tuple, representing the dimensions of the network's layers.
        First element must be the dimensionality of the input. Last element must be an integer >=2
        (because of the softmax activation on the last layer).
        :param activation: 'logistic', 'tanh', or 'relu' . Specifies the activation function of the hidden layers
        :param learning_rate: controls the speed of learning
        :param training_epochs: number of training iterations
        :param batch_size: number of instances processed at once by the mini-batch algorithm
        :param regularization: weight decay term.
        :param momentum_term: momentum term
        """
        self.layers = layers
        self.depth = len(self.layers)
        self.learning_rate_init = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.momentum_term = momentum_term
        self.hidden_activation = ACTIVATIONS[activation]
        self.hidden_activation_deriv = ACTIVATION_DERIVATIVES[activation]

        self.weights = []
        self.biases = []
        for i in range(1, self.depth):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]) * \
                                math.sqrt(2/(self.layers[i]+self.layers[i-1])))
            self.biases.append(np.zeros((self.layers[i], 1), dtype=np.float64))

        self.error_ = 0

    def fit(self, data, classes):
        self.minibatch_fit(data, classes)

    def partial_fit(self, data, classes):
        self.minibatch_fit(data, classes)

    def minibatch_fit(self, data, classes):
        """
        Implementation of Gradient Descent through backpropagation.
        The mini-batch gradient descent variant is used to speed up training times.

        Convergence rate is increased by use of momentum optimization.
        Overfitting is reduced by applying regularization.

        :param data: Training instances
        :param classes: Real classes of the input instances
        :return: None
        """
        learning_rate = self.learning_rate_init
        batches = np.arange(0, len(data), self.batch_size)
        if batches[-1] != len(data):
            batches = np.hstack((batches, [len(data)]))

        for epoch in range(self.training_epochs):
            self.error_ = 0

            # Init momentums
            weight_momentums = [np.zeros_like(arr) for arr in self.weights]
            bias_momentums = [np.zeros_like(arr) for arr in self.biases]

            # Shuffle training data
            data,classes = sklearn.utils.shuffle(data,classes)
            for batch_index in range(len(batches) - 1):

                x = data[batches[batch_index]:batches[batch_index + 1]]
                y = classes[batches[batch_index]:batches[batch_index + 1]]

                # Forward pass
                signal = x.T
                signals = [signal]
                for i in range(self.depth-2):
                    signal = self.weights[i].dot(signal) + self.biases[i]
                    self.hidden_activation(signal)
                    signals.append(signal)
                signal = self.weights[self.depth-2].dot(signal) + self.biases[self.depth-2]
                output = self.__output_activation(signal)

                # Compute Error Function
                self.error_ += self.__calculate_loss(output, y) + \
                               self.regularization / 2 * (np.sum((np.sum(np.square(w)) for w in self.weights)))

                # Backwards pass
                error_gradient = self.__loss_gradient(output, y).T
                for i in range(self.depth-2, -1, -1):
                    # Compute updates for current layer. Apply weight decay.
                    delta_w = signals[i].dot(error_gradient.T).T + self.regularization * self.weights[i]
                    delta_b = error_gradient.sum(axis=1, keepdims=True)

                    # Apply updates to momentum
                    weight_momentums[i] = self.momentum_term * weight_momentums[i] - learning_rate * delta_w
                    bias_momentums[i] = self.momentum_term * bias_momentums[i] - learning_rate * delta_b

                    # Update weights and biases through momentum values
                    self.weights[i] += weight_momentums[i]
                    self.biases[i] += bias_momentums[i]

                    # Backpropagate error
                    error_gradient = self.hidden_activation_deriv(signals[i]) * error_gradient.T.dot(self.weights[i]).T


            #Compute Error
            self.error_ /= len(data)
            print("Error at epoch {}: {} ".format(epoch, self.error_))

    def __output_activation(self, arr):
        return  np.apply_along_axis(softmax, axis=0, arr=arr).T

    def __calculate_loss(self, decision, real):
        return cross_entropy_loss(decision, real)

    def __loss_gradient(self, decision, real):
        return cross_entropy_gradient(decision, real)

    def decision_function(self, X):
        """
        Returns the soft decision (probabilities) of the network for given inputs.
        :param X: input instances
        :return: Soft Decisions for instances X
        """

        #Forward Pass
        signal = X.T
        for i in range(self.depth - 2):
            signal = self.weights[i].dot(signal) + self.biases[i]
            self.hidden_activation(signal)
        signal = self.weights[self.depth - 2].dot(signal) + self.biases[self.depth - 2]

        return self.__output_activation(signal)

    def predict(self, X):
        """
        Returns the hard decision (class with maximum probability) of the network for given inputs.
        :param X: input instances
        :return: Hard decision for instances X

        """
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
    data = joblib.load('test_data/moons.data')
    X, Y = data[:, :-1], data[:, -1]
    X = normalize_data(X)
    Y = one_hot_encode(Y)
    X_train, Y_train = X[:], Y[:]
    model.fit(X_train, Y_train)

    if(input("Plot decision boundary?[Y/N]") in ['Y','y']):
        plot_decision_boundary(lambda x: np.argmax(model.decision_function(x), axis=1), X, Y)


if __name__ == '__main__':


    model = MLPClassifier(layers=(2, 8, 2), learning_rate=0.03, activation='logistic', regularization=0.000001,
                          batch_size=16, training_epochs=200)
    test_2d_moons_classification(model)
    print()
