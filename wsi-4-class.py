import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix


class WineQualityClassifier:
    def __init__(self, filename ="winequality-red.csv"):
        self.filename = filename
        self.dataframe = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.enc = OrdinalEncoder()
        self.scaler = MinMaxScaler()

    def preprocess_data(self):
        self.dataframe = pd.read_csv(self.filename, sep=';')
        self.dataframe['quality'] = self.dataframe['quality'].apply(lambda x: 1 if x >= 6 else -1)
        
        y = np.array(self.dataframe.loc[:, 'quality'])
        X = self.dataframe.iloc[:, :-1]

        self.enc.fit(X)
        X = self.enc.transform(X)

        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


class Kernel:
    def __init__(self):
        self.func = None

class LinearKernel(Kernel):
    def __init__(self):
        self.func = lambda x, y: np.dot(x.T, y)

class RBFKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma
        self.func = lambda x, y: np.exp(-self.gamma * np.linalg.norm(x-y)**2)
        
class PolyKernel(Kernel):
    def __init__(self,c,d):
        self.c = c
        self.d = d
        self.func = lambda x, y: pow((np.dot(x, y) + self.c), self.d)


class SVM:
    def __init__(self, kernel, C=1):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel.func(X[i], X[j])

        P = matrix(np.outer(y, y) * K)
        q = matrix(-1 * np.ones(n_samples))

        G = matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        A = matrix(y.reshape(1, -1).astype('double'), (1, n_samples))
        b = matrix(0.0)

        sol = solvers.qp(P, q, G, h, A, b)

        a = np.ravel(sol["x"])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel.func(X[i], sv)
            y_predict[i] = s + self.b
        return y_predict
    


if __name__ == "__main__":
    dataframe = pd.read_csv("winequality-red.csv", sep=';')
    dataframe['quality'] = dataframe['quality'].apply(lambda x: 1 if x >= 6 else -1)
        
    y = np.array(dataframe.loc[:, 'quality'])
    X = dataframe.iloc[:, :-1]

    enc = OrdinalEncoder()
    enc.fit(X)
    X = enc.transform(X)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    kernel = "poly"
    if kernel == "rbf":
        #kernel rbf
        rbf_kernel = RBFKernel(gamma=16) #16 duzo, 80procent
        rbf_kernel_clf = SVM(kernel=rbf_kernel, C=1.0) #C=1
        rbf_kernel_clf.fit(X_train, y_train)
        rbf_predictions = rbf_kernel_clf.predict(X_test)
        rbf_accuracy = accuracy_score(y_true=y_test, y_pred=rbf_predictions)
        rbf_precision = precision_score(y_true=y_test, y_pred=rbf_predictions)
        print(rbf_accuracy)
        
    elif kernel == "linear":
        #linear kernel
        lin_kernel = LinearKernel() #75procent
        lin_kernel_clf = SVM(kernel=lin_kernel, C=10) #C=10
        lin_kernel_clf.fit(X_train, y_train)
        lin_predictions = lin_kernel_clf.predict(X_test)
        lin_accuracy = accuracy_score(y_true=y_test, y_pred=lin_predictions)
        lin_precision = precision_score(y_true=y_test, y_pred=lin_predictions)
        print(lin_accuracy)
        
    else:
        #poly kernel
        poly_kernel = PolyKernel(c=6, d=4) #80,9procent, c=6, d=4
        poly_kernel_clf = SVM(kernel=poly_kernel) 
        poly_kernel_clf.fit(X_train, y_train)
        poly_predictions = poly_kernel_clf.predict(X_test)
        poly_accuracy = accuracy_score(y_true=y_test, y_pred=poly_predictions)
        poly_precision = precision_score(y_true=y_test, y_pred=poly_predictions)
        print(poly_accuracy)
    
# Użycie klasy
wine_classifier = WineQualityClassifier()
wine_classifier.preprocess_data()

