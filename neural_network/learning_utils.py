from numba import *
import math
import numpy as np
import matplotlib.pyplot as plt


@jit(void(float64[:, :]), nopython=True)
def inplace_logistic_sigmoidf64(arr):
    """
    In-place Logistic Sigmoid activation function

    :param arr: Input Numpy array
    :return: None
    """
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = 1/(1+math.exp(-arr[i, j]))

@jit(float64[:, :](float64[:, :]), nopython=True)
def logistic_derivative(arr):
    """
    Logistic sigmoid derivative

    :param arr:
    :return: array of derivative values
    """
    return arr * (1 - arr)


@jit(void(float64[:, :]), nopython=True)
def inplace_tanhf64(arr):
    """

    In-place Tanh Activation function

    :param arr: Input Numpy array
    :return: None
    """
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = np.tanh(arr[i,j])

@jit(float64[:, :](float64[:, :]), nopython=True)
def tanh_derivative(arr):
    """
    Tanh derivative function

    :param arr:
    :return: array of derivative values
    """
    return 1 - np.power(arr,2)


@jit(void(float64[:, :]), nopython=True)
def inplace_relu(arr):
    """
    In-place relu activation function
    :param arr:
    :return:
    """
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = max(0, arr[i, j])

@jit(boolean[:, :](float64[:, :]), nopython=True)
def relu_derivative(arr):
    """
       Relu derivative

       :param arr:
       :return: array of derivative values
       """
    return (arr > 0)

def one_hot_encode(Y):
    """
    Transforms int's into one-hot encoded arrays
    :param Y: input int
    :return: one-hot encoding of input
    """
    classes = len(set(Y))
    ret = []
    for entry in Y:
        class_vector = [0]*classes
        class_vector[int(entry)] = 1
        ret.append(class_vector)
    return np.array(ret)


def normalize_function(arr):
    """
    MinMax normalization of an array
    :param arr: input numpy array
    :return: numpy array of normalized values
    """
    min_value = np.min(arr)
    arr = (arr-min_value)/(np.max(arr)-min_value)
    return arr


def normalize_data(X):
    """
    Normalizes elements of ndarray along the y-axis

    :param X: input 2D ndarray
    :return: ndarray with elements normalized along the y-axis
    """
    return np.apply_along_axis(normalize_function,axis=0,arr=X)


def softmax(arr):
    """
    In-place Softmax Activation function
    :param arr:
    :return:
    """
    arr = np.exp(arr-np.max(arr))
    return arr/np.sum(arr)


def cross_entropy_loss(decision: np.ndarray,real: np.ndarray) -> np.ndarray:
    """
    Implements the categorical cross entropy loss.
    :param decision: ndarray of network prediction
    :param real: ndarray of ground truth values
    :return: value of categorical cross entropy loss
    """

    return -np.sum(real*np.log(decision+1e-10))


def cross_entropy_gradient(decision,real):
    """
    Derivative of categorical cross entropy loss
    :param decision: ndarray of network prediction
    :param real: ndarray of ground truth values
    :return:
    """
    return decision-real



def plot_decision_boundary(pred_func, X, y):
    """
    Function to plot decision boundary of model for 2D classification problems.
    :param pred_func: model prediction method callback
    :param X: input points
    :param y: classes
    :return: None
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=np.apply_along_axis(lambda qq: np.argmax(qq),axis=1,arr=y), cmap=plt.cm.Spectral)
    plt.show()

ACTIVATIONS = {'logistic': inplace_logistic_sigmoidf64, 'tanh': inplace_tanhf64, 'relu': inplace_relu}
ACTIVATION_DERIVATIVES = {'logistic': logistic_derivative, 'tanh': tanh_derivative, 'relu': relu_derivative}