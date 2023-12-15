from cvxopt import matrix
from cvxopt.solvers import qp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from dataclasses import dataclass
import pandas as pd
import numpy as np


class Kernel:
    def __init__(self):
        self.func = None

    def __call__(self, xi, xj):
        if self.func is not None:
            return self.func(xi, xj)
        else:
            raise NotImplementedError("Kernel function is not defined.")


class LinearKernel(Kernel):
    def __init__(self):
        self.func = lambda xi, xj: np.dot(xi.T, xj)


class RBFKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma
        self.func = lambda xi, xj: np.exp(-self.gamma * np.linalg.norm(xi - xj) ** 2)


class PolyKernel(Kernel):
    def __init__(self, c, d):
        self.c = c
        self.d = d
        self.func = lambda xi, xj: pow((np.dot(xi, xj) + self.c), self.d)


@dataclass
class params_t:
    file_name: str
    column_name: str
    minimum_quality: int


@dataclass
class hiperparams_t:
    kernel: Kernel = LinearKernel
    C: float = 1


class SVM:
    def __init__(self, hiperparams: hiperparams_t):
        self.kernel = hiperparams.kernel
        self.C = hiperparams.C
        self.bias = 0
        self.K = np.array([])

    def calculate_K_matrix(self, X):
        number_of_samples, _ = X.shape
        self.K = np.array(
            [
                [self.kernel(X[i], X[j]) for j in range(number_of_samples)]
                for i in range(number_of_samples)
            ]
        )

    def calculate_bias_svm(self, support_vectors, ids_support_vectors):
        for i in range(len(self.alphas)):
            self.bias += self.support_vect_y[i]
            self.bias -= np.sum(
                self.alphas
                * self.support_vect_y
                * self.K[ids_support_vectors[i], support_vectors]
            )
        self.bias /= len(self.alphas)

    def fit(self, X, y):
        number_of_samples, _ = X.shape
        self.calculate_K_matrix(X)

        P = matrix(np.outer(y, y) * self.K)
        A = matrix(y.reshape(1, -1).astype("double"), (1, number_of_samples))
        q = matrix(-1 * np.ones(number_of_samples))
        b = matrix(0.0)
        G = matrix(
            np.vstack((np.eye(number_of_samples) * -1, np.eye(number_of_samples)))
        )
        h = matrix(
            np.hstack(
                (np.zeros(number_of_samples), np.ones(number_of_samples) * self.C)
            )
        )

        result = qp(P, q, G, h, A, b)
        alphas = np.ravel(result["x"])

        support_vectors = alphas > 1e-6
        ids_support_vectors = np.arange(len(alphas))[support_vectors]
        self.alphas = alphas[support_vectors]
        self.support_vect_x = X[support_vectors]
        self.support_vect_y = y[support_vectors]

        self.calculate_bias_svm(support_vectors, ids_support_vectors)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        y_predict = np.zeros(len(X))
        for j in range(len(X)):
            s = 0
            for alpha, yi, xi in zip(
                self.alphas, self.support_vect_y, self.support_vect_x
            ):
                s += alpha * yi * self.kernel(X[j], xi)
            y_predict[j] = s + self.bias
        return y_predict


def transform_data(X):
    enc = OrdinalEncoder().fit(X)
    X = enc.transform(X)

    scaler = MinMaxScaler().fit(X)
    return scaler.transform(X)


if __name__ == "__main__":
    kernel = RBFKernel(gamma=16)
    # kernel = PolyKernel(c=6, d=4)
    hiperparams = hiperparams_t(kernel, C=1.0)
    params = params_t("winequality-red.csv", "quality", 6)

    dataframe = pd.read_csv(params.file_name, sep=";")
    dataframe[params.column_name] = dataframe[params.column_name].apply(
        lambda x: 1 if x >= params.minimum_quality else -1
    )

    y = np.array(dataframe.loc[:, params.column_name])
    X = dataframe.iloc[:, :-1]
    X = transform_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    svm_algorithm = SVM(hiperparams)
    svm_algorithm.fit(X_train, y_train)
    predictions = svm_algorithm.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)
