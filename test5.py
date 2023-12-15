from operator import truth
import cvxopt
from cvxopt import matrix
from cvxopt.solvers import qp
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def kernel_polynomial(a, b, degree=3):
    return pow((np.dot(a, b) + 1), degree)


def make_polynomial(d=3):
    return lambda a, b, d=d: pow((np.dot(a, b) + 1), d)


def kernel_linear(a, b):
    return np.dot(a, b)

def kernel_radial(a, b, gamma=1):
    tmp = np.exp(-pow(linalg.norm(a-b), 2) / (pow(gamma, 2) * 2))
    return tmp


def rbf(gamma=1):  # gauss
    return lambda a, b, gamma=gamma: np.exp(-pow(linalg.norm(a-b), 2) / (pow(gamma, 2) * 2))


class support_vector_machine():
    def __init__(self, kernel=kernel_polynomial, C=None):
        if C is not None:
            self.C = float(C)
        else:
            self.C = C
        self.kernel = kernel

    def get_kernel(self):
        return self.kernel

    def get_C(self):
        return self.C

    def get_b(self):
        return self.b

    def set_b(self, b):
        self.b = b

    def get_w(self):
        return self.w

    def set_w(self, w):
        self.w = w

    def prediction(self, x):
        res = np.sign(self.projection(x))
        return res

    def projection(self, x):
        if self.get_w() is None:
            length = len(x)
            prediction_for_y = np.zeros(length)
            for i in range(length):
                res = 0
                for support_vector_y, arr, support_vector in zip(self.support_vector_y, self.array, self.support_vector):
                    res = res + self.get_kernel()(x[i], support_vector) * \
                        arr * support_vector_y
                prediction_for_y[i] = res
            result = self.get_b() + prediction_for_y
            return result

        else:
            res = self.get_b() + np.dot(x, self.get_w())
            return res

    def algorithm(self, x, y):
        number_of_observations, number_of_attributes = x.shape
        # create matrix for qp solving
        K = np.zeros((number_of_observations, number_of_observations))
        for k in range(0, number_of_observations):
            for l in range(0, number_of_observations):
                K[k, l] = self.get_kernel()(x[k], x[l])

        A = matrix(y, (1, number_of_observations))
        print(len(A))
        P = matrix(K * np.outer(y, y))
        b = matrix(float(0))
        q = matrix(-np.ones(number_of_observations))

        if self.get_C() is not None:
            diagonal = np.diag(-np.ones(number_of_observations))
            identity_matrix = np.identity(number_of_observations)
            G = matrix(np.vstack((diagonal, identity_matrix)))
            empty = np.zeros(number_of_observations)
            onesxc = self.get_C() * np.ones(number_of_observations)
            h = matrix(np.hstack((empty, onesxc)))
        else:
            G = matrix(np.diag(-np.ones(number_of_observations)))
            h = matrix(np.zeros(number_of_observations))

        # solve qp
        result = qp(P, q, G, h, A, b)
        array = np.ravel(result['x'])
        support_vector = 0.000000000000000001 < array

        self.support_vector = x[support_vector]
        self.support_vector_y = y[support_vector]
        self.array = array[support_vector]
        ind = np.arange(len(array))[support_vector]

        # Weight vector for linear kernel (algorithm works faster)
        if kernel_linear != self.get_kernel():
            self.set_w(None)
        else:
            self.set_w(np.zeros(number_of_attributes))
            for i in range(len(self.array)):
                self.set_w(
                    self.get_w() + self.support_vector[i] * self.array[i] * self.support_vector_y[i])

        self.set_b(0)
        for i in range(len(self.array)):
            self.set_b(self.get_b(
            ) - np.sum(K[ind[i], support_vector] * self.array * self.support_vector_y))
            self.set_b(self.get_b() + self.support_vector_y[i])
        length_arr = len(self.array)
        self.set_b(self.get_b() / length_arr)


def read_csv():
    data = pd.read_csv('winequality-red.csv', sep=';')
    data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6 else -1)
    return data
    

def calculate(svm_algorithm, x_training, y_training, x, y):
    svm_algorithm.algorithm(x_training, y_training)
    prediction_for_y = svm_algorithm.prediction(x)
    correct = np.sum(prediction_for_y == y)
    return correct, prediction_for_y


def output_result(correct, prediction):
    print(f"Correctly guessed {correct} in {len(prediction)} tries")
    ratio = correct / len(prediction)
    print(ratio)
    return ratio


def get_data():
    data = read_csv()
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def hard_margin(kernel=kernel_linear, number=1000):
    svm_algorithm = support_vector_machine(kernel=kernel)
    return output_result(*calculate(svm_algorithm, *get_data()))


def soft_margin(kernel=kernel_linear, C=None, number=1000):
    svm_algorithm = support_vector_machine(kernel=kernel, C=C)
    return output_result(*calculate(svm_algorithm, *get_data()))


def main():
    # hard_margin(kernel=kernel_linear, number=1000)
    # hard_margin(kernel=make_polynomial(d=3), number=100)
    # hard_margin(kernel=rbf(0.01), number=100)
    # soft_margin(kernel=kernel_linear, C=0.01)
    # soft_margin(kernel=make_polynomial(d=2), C=0.01)
    # soft_margin(kernel=rbf(gamma=100), C=10000)

    # soft_margin(kernel=rbf(gamma=100), C=10000)  # 75%
    # soft_margin(kernel=ma ke_polynomial(d=3), C=10**i, number=200)

    for i in range(-4, 6):
        soft_margin(kernel=kernel_linear, C=pow(10, i))
        print(pow(10, i))


if __name__ == "__main__":
    main()
