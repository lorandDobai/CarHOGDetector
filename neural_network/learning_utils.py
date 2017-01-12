from numba import *
import math
import numpy as np
import matplotlib.pyplot as plt


@jit(void(float64[:, :]), nopython=True)
def inplace_logistic_sigmoidf64(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = 1/(1+math.exp(-arr[i, j]))


@jit(void(float64[:, :]), nopython=True)
def inplace_tanhf64(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = np.tanh(arr[i,j])

@jit(void(float64[:, :]), nopython=True)
def inplace_relu(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = max(0,arr[i,j])

def process_class_representation(Y):
    classes = len(set(Y))
    ret = []
    for entry in Y:
        class_vector = [0]*classes
        class_vector[int(entry)] = 1
        ret.append(class_vector)
    return np.array(ret)


def normalize_function(arr):
    min_value = np.min(arr)
    arr = (arr-min_value)/(np.max(arr)-min_value)
    return arr


def normalize_data(X):
    return np.apply_along_axis(normalize_function,axis=0,arr=X)


def softmax(arr):
    arr = np.exp(arr-np.max(arr))
    return arr/np.sum(arr)


def cross_entropy_loss(decision: np.ndarray,real: np.ndarray) -> np.ndarray:

    return -np.sum(real*np.log(decision+1e-9) + (1-real)*np.log(1-decision+1e-9))


def cross_entropy_gradient(decision,real):
    return decision-real


def logistic_derivative(arr):
    return arr * (1 - arr)


def tanh_derivative(arr):
    return 1 - np.power(arr,2)

def relu_derivative(arr):
    return arr > 0
def plot_decision_boundary(pred_func, X, y):
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