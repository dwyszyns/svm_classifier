import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SVM:
    def __init__(self, C = 1.0, kernel=None):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0
        self.kernel = kernel

    # Hinge Loss Function / Calculation
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y.iloc[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        if(self.kernel == "poly"):
            print("chuj")
            X = self.transform_poly(X, Y)
        elif(self.kernel == "rbf"):
            X = self.RBF(X, gamma=None)
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, Y)

            # Appending all losses 
            losses.append(l)
            
            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial+ batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y.iloc[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculating the gradients

                            #w.r.t w 
                            gradw += c * Y.iloc[x] * X[x]
                            # w.r.t b
                            gradb += c * Y.iloc[x]

                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses

    def predict(self, X):
        if(self.kernel == "poly"):
            X = self.transform_poly(X, np.array([]))
        elif(self.kernel == "rbf"):
            X = self.RBF(X, gamma=None)
        prediction = np.dot(X, self.w[0]) + self.b
        return np.sign(prediction)
    
    def transform_poly(self, X, Y=None):
    # Finding the Square of X1, X2
        X['x1^2'] = X['x1'] ** 2
        X['x2^2'] = X['x2'] ** 2
        # Finding the product of X1 and X2
        X['x1 * x2'] = X['x1'] * X['x2']
        # Converting dataset to numpy array
        return X
        
    def RBF(self, X, gamma):
        # Free parameter gamma
        if gamma == None:
            gamma = 1.0/X.shape[1]
            
        # RBF kernel Equation
        K = np.exp(-gamma * np.sum((X - X[:,np.newaxis])**2, axis = -1))
        
        return K
    





# X, y = make_circles(n_samples=500, noise=0.06, random_state=42)
# df = pd.DataFrame(dict(x1=X[:, 0], x2=X[:, 1], y=y))
# X = df[['x1', 'x2']]
# y = df['y']
# # Replacing 0 with -1 for the SVM model to recognize labels
# y = y.replace(0, -1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

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

svm = SVM(kernel="linear")

w, b, losses = svm.fit(X_train, y_train)

# svm = SVC(kernel="poly")

# svm.fit(X_train, y_train)

pred = svm.predict(X_test)

print("Accuracy:",accuracy_score(pred, y_test))