import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SVM:

    def __init__(self, C=1.0, kernel='linear', degree=2, gamma=0.1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.w = 0
        self.b = 0

    def kernel_function(self, xi, xj):
        if self.kernel == 'linear':
            return np.dot(xi, xj)
        elif self.kernel == 'poly':
            return (np.dot(xi, xj) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(xi - xj) ** 2)
        else:
            raise ValueError("Unsupported kernel type")

    def hingeloss(self, w, b, x, y):
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            opt_term = y.iloc[i] * ((self.kernel_function(w, x[i])) + b)
            loss = reg + self.C * max(0, 1 - opt_term)
        return loss[0][0]

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=100):
        number_of_features = X.shape[1]
        number_of_samples = X.shape[0]
        ids = np.arange(number_of_samples)
        np.random.shuffle(ids)
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        for i in range(epochs):
            l = self.hingeloss(w, b, X, Y)
            losses.append(l)

            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y.iloc[x] * (self.kernel_function(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += self.C * Y.iloc[x] * X[x]
                            gradb += self.C * Y.iloc[x]

                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb

        self.w = w
        self.b = b
        return self.w, self.b, losses

    def predict(self, X):
        prediction = np.dot(X, self.w[0]) + self.b
        return np.sign(prediction)



# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Discretize the target variable
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6 else -1)

# Split data into training and testing sets
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implement SVM with RBF kernel
svm_model = SVM(kernel='rbf', gamma=1000)
#0.1 - gamma, 0.769
svm_model.fit(X_train, y_train)

# Evaluate model accuracy
predictions = svm_model.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print(f'Accuracy for SVM: {accuracy}')