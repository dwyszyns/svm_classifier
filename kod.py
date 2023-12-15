import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import linalg

from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

minimum_quality = 6

#obiekt do parametrów 
#class params_t


#obiekt do danych
data = pd.read_csv('winequality-red.csv', sep=';')
data['quality'] = data['quality'].apply(lambda x: 1 if x >= minimum_quality else -1)
X_values = data.iloc[:, :-1]
Y_values = data.iloc[:, -1]

#0.01 - naj - 0.7125
def rbf_kernel(xi, xj, gamma=0.01):
        return np.exp(-gamma*linalg.norm(xi - xj) ** 2)

#obiekt do kerneli
def euclidean(xin: list) -> float:
    p = np.array(xin)
    return np.sqrt(np.dot(p.T, p))

def poly_kernel(xi, xj, c=1, d=2):
    result = np.dot(xi, xj) + c
    return pow(result, d)


class SVM_algorithm:
    def __init__(self, learning_rate, imax, error, kernal, pval, dval):
        self.learning_rate = learning_rate
        self.imax = imax
        self.error = error
        self.kernal = kernal
        self.pval = pval
        self.dval = dval


    def fit(self, X, Y):
        self.alpha = [0] * len(X[0])
        self.x_new = X
        self.Y_new = Y

        # count alphas
        self.alpha = self.gradient_descent(self.alpha)
        
        #count w
        self.w = 0
        for i in range(len(self.alpha)):
            self.w += self.alpha[i] * self.Y_new[i] * self.x_new[i]

        #count bias
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
                    # * kernal(x[n], x[Ind]) #gamma
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

svm = SVC(kernel="rbf", random_state=1, C=1.0)
svm.fit(X_train_std, y_train)

# Mode performance

y_pred = svm.predict(X_test_std)
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
y = y_train
x = X_train_std

# SVM parameters
#gamma - 0.01
learning_rate = 0.01
imax = 1000
error = 0.001
kernal = poly_kernel
pval = 4
dval = 0.1

# # Create SVM instance
# svmtest = SVM_algorithm(learning_rate, imax, error, kernal, pval, dval)

# # Train the SVM model
# svmtest.fit(x, y)

# prediction = []
# x_data = X_test_std
# for one in x_data:
#     prediction.append(svmtest.predict(one))


# print("Accuracy_score:", accuracy_score(y_test, prediction))

# Test different values of pval and dval
best_accuracy = 0
best_pval = 0
best_dval = 0

with open('wyniki.txt', 'w') as file:
    file.write("pval\t\tdval\t\tAccuracy\n")
    
    for pval in np.arange(0, 6, 0.2):
        for dval in np.arange(0, 15, 0.2):
            svm_test = SVM_algorithm(learning_rate, imax, error, poly_kernel, pval, dval)
            svm_test.fit(X_train_std, y_train)
            prediction = []
            x_data = X_test_std
            for one in x_data:
                prediction.append(svm_test.predict(one))

            accuracy = accuracy_score(y_test, prediction)
            print(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pval = pval
                best_dval = dval
            file.write(f"{pval:.2f}\t\t{dval:.2f}\t\t{accuracy:.4f}\n")

print("Best Accuracy:", best_accuracy)
print("Best pval:", best_pval)
print("Best dval:", best_dval)