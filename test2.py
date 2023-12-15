import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from gradient_descent import Solver_Results
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

df.loc[df["quality"] >= 7, "quality"] = 1
df.loc[df["quality"] == 6, "quality"] = 1
df.loc[df["quality"] == 5, "quality"] = -1
df.loc[df["quality"] == 4, "quality"] = -1
df.loc[df["quality"] == 3, "quality"] = -1

X_values = df.iloc[:, :-1]
Y_values = df.iloc[:, -1]


def euclidean(xin: list) -> float:
    p = np.array(xin)
    return np.sqrt(np.dot(p.T, p))


def Kernal(x1, x2, dval, pval):
    result = np.dot(x1, x2) + dval
    return pow(result, pval)


class Solver_Results:
    def init(self, fx: list, i: int, x: float):
        self._fx = fx
        self._imax = i
        self._x_fin = x

    def f_array(self) -> list:
        return self._fx

    def i_max(self) -> int:
        return self._imax

    def final_x(self) -> float:
        return self._x_fin


class SVM_algorithm:
    def __init__(self, learning_rate, lambda_par, imax, error, kernal, pval, dval):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_par
        self.imax = imax
        self.error = error
        self.kernal = kernal
        self.pval = pval
        self.dval = dval


    def fit(self, X, Y):
        self.alpha = [0] * len(X[0])
        self.x_new = X
        self.Y_new = Y

        # tutaj używam algorytmu gradient
        # updateuje wagi i obciążenie
        self.alpha = self.gradient_descent(self.alpha)
        self.w = 0
        for i in range(len(self.alpha)):
            self.w += self.alpha[i] * self.Y_new[i] * self.x_new[i]

        # Calculate the bias
        bias_sum = 0
        for i in range(len(self.Y_new)):
            bias_new = self.Y_new[i] - np.dot(self.w, self.x_new[i])
            bias_sum += bias_new
        self.bias = bias_sum / len(self.Y_new)

        sum = 0
        for i in range(len(self.alpha)):
            sum += self.alpha[i] * self.Y_new[i]


    def predict(self, Xin):
        decision_function_result = np.dot(self.w, Xin) + self.bias
        prediction = np.sign(decision_function_result)
        true_prediction = np.where(prediction <= -1, -1, 1)
        return true_prediction


    def grad_of_minimal(self, alpha):
        x = self.x_new
        y = self.Y_new
        kernal = self.kernal
        N = len(alpha)
        grad = [1] * N
        for Ind in range(N):
            grad[Ind] = 1
            for n in range(N):
                grad[Ind] -= (
                    y[Ind]
                    * y[n]
                    * alpha[n]
                    * kernal(x[n], x[Ind], self.dval, self.pval)
                )
        return grad
        

    def gradient_descent(self, alphaIn):
        alpha = alphaIn
        beta = self.learning_rate
        t = self.imax
        error = self.error
        f_out = []
        last = 2000
        for i in range(t):
            grad1 = self.grad_of_minimal(alpha)
            sum = 0
            for one in range(len(self.alpha)):
                sum += alpha[one] * self.Y_new[one]
            if abs(grad1[0]) < error and i > 100:
                break
            else:
                for ind, one in enumerate(alpha):
                    if (
                        alpha[ind] + grad1[ind] * beta >= 0
                        and alpha[ind] + grad1[ind] * beta < 100000
                    ):
                        alpha[ind] += grad1[ind] * beta
            last = grad1[0]
        return alpha

X_train, X_test, y_train, y_test = train_test_split(
    X_values.to_numpy(), Y_values.to_numpy(), test_size=0.6, random_state=43
)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel="linear", random_state=1, C=0.1)
svm.fit(X_train_std, y_train)

# Mode performance

y_pred = svm.predict(X_test_std)
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
y = y_train
x = X_train_std

# SVM parameters
learning_rate = 0.02
lambda_param = 0.3
imax = 1000
error = 0.0003
kernal = euclidean
pval = 4
dval = 0.1

# Create SVM instance
svmtest = SVM_algorithm(learning_rate, lambda_param, imax, error, kernal, pval, dval)

# Train the SVM model
svmtest.fit(x, y)

prediction = []
x_data = X_test_std
for one in x_data:
    prediction.append(svmtest.predict(one))

print("Accuracy_score:", accuracy_score(y_test, prediction))

