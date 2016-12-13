import joblib
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from neural_network.learning_utils import normalize_data, process_class_representation, inplace_logistic_sigmoidf64, \
    softmax, \
    cross_entropy_loss, cross_entropy_gradient

from neural_network.learning_utils import plot_decision_boundary


class SimpleMLPClassifier:
    def __init__(self, input_size=None, hidden_layer_size=8, output_size=2, learning_rate=0.001,
                 training_epochs=10, batch_size=1, regularization=0.0001, momentum_term=0.9):
        self.I = input_size
        self.J = hidden_layer_size
        self.K = output_size
        self.learning_rate_init = learning_rate
        self.epochs = training_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.momentum_term = momentum_term

        self.w1 = np.random.uniform(-0.05, 0.05, size=(self.J, self.I)).astype(dtype=np.float64)
        self.b1 = np.random.uniform(-0.05, 0.05, size=(self.J, 1)).astype(dtype=np.float64)

        self.w2 = np.random.uniform(-0.05, 0.05, size=(self.K, self.J)).astype(dtype=np.float64)
        self.b2 = np.random.uniform(-0.05, 0.05, size=(self.K, 1)).astype(dtype=np.float64)
        self.error_ = 0

    def fit(self, data, classes):
        self.minibatch_fit(data, classes)

    def partial_fit(self, data, classes):
        self.minibatch_fit(data, classes)

    def minibatch_fit(self, data, classes):
        learning_rate = self.learning_rate_init

        weights = [self.w2, self.b2, self.w1, self.b1]
        batches = np.arange(0, len(data), self.batch_size)
        if batches[-1] != len(data):
            batches = np.hstack((batches, [len(data)]))

        for epoch in range(self.epochs):
            self.error_ = 0
            momentums = [np.zeros_like(arr) for arr in (self.w2, self.b2, self.w1, self.b1)]
            data,classes = sklearn.utils.shuffle(data,classes)
            for batch_index in range(len(batches) - 1):

                x = data[batches[batch_index]:batches[batch_index + 1]]
                y = classes[batches[batch_index]:batches[batch_index + 1]]
                """Forward Pass"""
                o1 = self.w1.dot(x.T) + self.b1
                inplace_logistic_sigmoidf64(o1)
                o2 = self.w2.dot(o1) + self.b2
                o2 = np.apply_along_axis(softmax, axis=0, arr=o2).T

                """Error Function"""
                self.error_ += self._calculate_loss(o2, y) + \
                         self.regularization / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))

                """Backwards Pass"""
                delta_grad_out = self.loss_gradient(o2, y)
                deltaW2 = o1.dot(delta_grad_out).T
                deltaW2 += self.regularization * self.w2
                deltaB2 = delta_grad_out.T.sum(axis=1, keepdims=True)

                delta_grad_hidden = o1 * (1 - o1) * delta_grad_out.dot(self.w2).T
                deltaW1 = (delta_grad_hidden).dot(x) #/ current_batch_size
                deltaW1 += self.regularization * self.w1
                deltaB1 = delta_grad_hidden.sum(axis=1, keepdims=True)

                for index, delta in enumerate((deltaW2, deltaB2, deltaW1, deltaB1)):
                    momentums[index] = self.momentum_term * momentums[index] - learning_rate * delta
                    weights[index] += momentums[index]

            self.error_ /= len(data)
            print("Error at epoch {}: {} ".format(epoch, self.error_))

    def _calculate_loss(self, decision, real):
        return cross_entropy_loss(decision, real)

    def loss_gradient(self, decision, real):
        return cross_entropy_gradient(decision, real)

    def decision_function(self, X):
        """Forward Pass"""
        o1 = self.w1.dot(X.T) + self.b1
        inplace_logistic_sigmoidf64(o1)
        o2 = self.w2.dot(o1) + self.b2
        """Apply softmax on output signal"""
        results = np.apply_along_axis(softmax, axis=0, arr=o2).T
        return results

    def predict(self, X):
        return np.argmax(self.decision_function(X),axis=1)

def test_2d_classification(model):
    data = joblib.load('points.data')
    X, Y = data[:, :-1], data[:, -1]
    X = normalize_data(X)
    Y = process_class_representation(Y)
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
    Y = process_class_representation(Y)
    X_train, Y_train = X[:-10], Y[:-10]
    X_validation, Y_validation = X[-10:], Y[-10:]

    model.fit(X_train, Y_train)

    print(model.decision_function(X_validation), Y_validation)
    plot_decision_boundary(lambda x: np.argmax(model.decision_function(x), axis=1), X, Y)


if __name__ == '__main__':
    #model = SimpleMLPClassifier(input_size=2, learning_rate=0.02, hidden_layer_size=8, regularization=0.00001,
                                #batch_size=16, training_epochs=1000)

    model = SimpleMLPClassifier(input_size=2, learning_rate=0.03, hidden_layer_size=16, regularization=0.000001,
                                batch_size=8, training_epochs=150)
    test_2d_moons_classification(model)
    print()
